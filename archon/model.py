from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from archon.config import ModelConfig


def compute_rope_params(
    head_dim: int,
    theta_base: float = 10_000.0,
    context_length: int = 4096,
    dtype: torch.dtype = torch.float32,
) -> tuple[torch.Tensor, torch.Tensor]:
    if head_dim % 2 != 0:
        raise ValueError("Embedding dimension must be even")

    inv_freq = 1.0 / (
        theta_base
        ** (
            torch.arange(0, head_dim, 2, dtype=dtype)[: (head_dim // 2)].float()
            / head_dim
        )
    )
    positions = torch.arange(context_length, dtype=dtype)
    angles = positions[:, None] * inv_freq[None, :]
    angles = torch.cat([angles, angles], dim=1)
    return torch.cos(angles), torch.sin(angles)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    # x: (batch_size, num_heads, seq_len, head_dim)
    _, _, seq_len, head_dim = x.shape
    if head_dim % 2 != 0:
        raise ValueError("Head dimension must be even")

    x1 = x[..., : head_dim // 2]
    x2 = x[..., head_dim // 2 :]
    cos = cos[:seq_len, :].unsqueeze(0).unsqueeze(0)
    sin = sin[:seq_len, :].unsqueeze(0).unsqueeze(0)
    rotated = torch.cat((-x2, x1), dim=-1)
    return ((x * cos) + (rotated * sin)).to(dtype=x.dtype)



class RMSNorm(nn.Module):
    def __init__(self, emb_dim: int, eps: float = 1e-6, bias: bool = False) -> None:
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.zeros(emb_dim))
        self.shift = nn.Parameter(torch.zeros(emb_dim)) if bias else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_dtype = x.dtype
        x_f = x.float()
        var = x_f.pow(2).mean(dim=-1, keepdim=True)
        x_norm = x_f * torch.rsqrt(var + self.eps)
        out = x_norm * (1.0 + self.scale.float())
        if self.shift is not None:
            out = out + self.shift.float()
        return out.to(input_dtype)


class GroupedQueryAttention(nn.Module):
    def __init__(
        self,
        d_in: int,
        num_heads: int,
        num_kv_groups: int,
        head_dim: int | None = None,
        qk_norm: bool = False,
        query_pre_attn_scalar: float | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        if num_heads % num_kv_groups != 0:
            raise ValueError("num_heads must be divisible by num_kv_groups")

        self.num_heads = num_heads
        self.num_kv_groups = num_kv_groups
        self.group_size = num_heads // num_kv_groups

        if head_dim is None:
            if d_in % num_heads != 0:
                raise ValueError("`d_in` must be divisible by `num_heads` if `head_dim` is not set")
            head_dim = d_in // num_heads

        self.head_dim = head_dim
        self.d_out = num_heads * head_dim

        self.W_query = nn.Linear(d_in, self.d_out, bias=False, dtype=dtype)
        self.W_key = nn.Linear(d_in, num_kv_groups * head_dim, bias=False, dtype=dtype)
        self.W_value = nn.Linear(d_in, num_kv_groups * head_dim, bias=False, dtype=dtype)
        self.out_proj = nn.Linear(self.d_out, d_in, bias=False, dtype=dtype)

        if qk_norm:
            self.q_norm: RMSNorm | None = RMSNorm(head_dim, eps=1e-6)
            self.k_norm: RMSNorm | None = RMSNorm(head_dim, eps=1e-6)
        else:
            self.q_norm = self.k_norm = None

        if query_pre_attn_scalar is not None:
            self.scaling = query_pre_attn_scalar**-0.5
        else:
            self.scaling = head_dim**-0.5

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> torch.Tensor:
        b, num_tokens, _ = x.shape

        queries = self.W_query(x)
        keys = self.W_key(x)
        values = self.W_value(x)

        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)
        keys = keys.view(b, num_tokens, self.num_kv_groups, self.head_dim).transpose(1, 2)
        values = values.view(b, num_tokens, self.num_kv_groups, self.head_dim).transpose(1, 2)

        if self.q_norm is not None:
            queries = self.q_norm(queries)
        if self.k_norm is not None:
            keys = self.k_norm(keys)

        queries = apply_rope(queries, cos, sin)
        keys = apply_rope(keys, cos, sin)

        keys = keys.repeat_interleave(self.group_size, dim=1)
        values = values.repeat_interleave(self.group_size, dim=1)

        queries = queries * self.scaling
        attn_scores = queries @ keys.transpose(2, 3)
        attn_scores = attn_scores.masked_fill(mask, -torch.inf)
        attn_weights = torch.softmax(attn_scores, dim=-1)

        context = (attn_weights @ values).transpose(1, 2).reshape(b, num_tokens, self.d_out)
        return self.out_proj(context)


class FeedForward(nn.Module):
    def __init__(self, cfg: dict[str, Any]) -> None:
        super().__init__()
        self.fc1 = nn.Linear(cfg["emb_dim"], cfg["hidden_dim"], dtype=cfg["dtype"], bias=False)
        self.fc2 = nn.Linear(cfg["emb_dim"], cfg["hidden_dim"], dtype=cfg["dtype"], bias=False)
        self.fc3 = nn.Linear(cfg["hidden_dim"], cfg["emb_dim"], dtype=cfg["dtype"], bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_fc1 = self.fc1(x)
        x_fc2 = self.fc2(x)
        x = nn.functional.gelu(x_fc1, approximate="tanh") * x_fc2
        return self.fc3(x)


class TransformerBlock(nn.Module):
    def __init__(self, cfg: dict[str, Any], attn_type: str) -> None:
        super().__init__()
        self.attn_type = attn_type
        self.att = GroupedQueryAttention(
            d_in=cfg["emb_dim"],
            num_heads=cfg["n_heads"],
            num_kv_groups=cfg["n_kv_groups"],
            head_dim=cfg["head_dim"],
            qk_norm=cfg["qk_norm"],
            query_pre_attn_scalar=cfg["query_pre_attn_scalar"],
            dtype=cfg["dtype"],
        )
        self.ff = FeedForward(cfg)
        self.input_layernorm = RMSNorm(cfg["emb_dim"], eps=1e-6)
        self.post_attention_layernorm = RMSNorm(cfg["emb_dim"], eps=1e-6)
        self.pre_feedforward_layernorm = RMSNorm(cfg["emb_dim"], eps=1e-6)
        self.post_feedforward_layernorm = RMSNorm(cfg["emb_dim"], eps=1e-6)

    def forward(
        self,
        x: torch.Tensor,
        mask_global: torch.Tensor,
        mask_local: torch.Tensor,
        cos_global: torch.Tensor,
        sin_global: torch.Tensor,
        cos_local: torch.Tensor,
        sin_local: torch.Tensor,
    ) -> torch.Tensor:
        shortcut = x
        x = self.input_layernorm(x)

        if self.attn_type == "sliding_attention":
            attn_mask, cos, sin = mask_local, cos_local, sin_local
        else:
            attn_mask, cos, sin = mask_global, cos_global, sin_global

        x_attn = self.att(x, attn_mask, cos, sin)
        x_attn = self.post_attention_layernorm(x_attn)
        x = shortcut + x_attn

        shortcut = x
        x_ffn = self.pre_feedforward_layernorm(x)
        x_ffn = self.ff(x_ffn)
        x_ffn = self.post_feedforward_layernorm(x_ffn)
        return shortcut + x_ffn



class ArchonModel(nn.Module):
    def __init__(self, cfg: ModelConfig | dict[str, Any]) -> None:
        super().__init__()
        if isinstance(cfg, ModelConfig):
            cfg = cfg.to_dict()

        if cfg["layer_types"] is None or len(cfg["layer_types"]) != cfg["n_layers"]:
            raise ValueError("layer_types must be provided and match n_layers")

        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"], dtype=cfg["dtype"])
        self.blocks = nn.ModuleList(
            [TransformerBlock(cfg, attn_type) for attn_type in cfg["layer_types"]]
        )
        self.final_norm = RMSNorm(cfg["emb_dim"], eps=1e-6)
        self.out_head = nn.Linear(
            cfg["emb_dim"], cfg["vocab_size"], bias=False, dtype=cfg["dtype"]
        )
        self.tok_emb.weight = self.out_head.weight

        cos_local, sin_local = compute_rope_params(
            head_dim=cfg["head_dim"],
            theta_base=cfg["rope_local_base"],
            context_length=cfg["context_length"],
            dtype=torch.float32,
        )
        cos_global, sin_global = compute_rope_params(
            head_dim=cfg["head_dim"],
            theta_base=cfg["rope_base"],
            context_length=cfg["context_length"],
            dtype=torch.float32,
        )
        self.register_buffer("cos_local", cos_local, persistent=False)
        self.register_buffer("sin_local", sin_local, persistent=False)
        self.register_buffer("cos_global", cos_global, persistent=False)
        self.register_buffer("sin_global", sin_global, persistent=False)

    def _create_masks(
        self, seq_len: int, device: torch.device
    ) -> tuple[torch.Tensor, torch.Tensor]:
        ones = torch.ones((seq_len, seq_len), dtype=torch.bool, device=device)
        mask_global = torch.triu(ones, diagonal=1)
        far_past = torch.triu(ones, diagonal=self.cfg["sliding_window"]).T
        mask_local = mask_global | far_past
        return mask_global, mask_local

    def forward(
        self,
        input_ids: torch.Tensor,
        targets: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        _, seq_len = input_ids.shape
        x = self.tok_emb(input_ids) * (self.cfg["emb_dim"] ** 0.5)
        mask_global, mask_local = self._create_masks(seq_len, x.device)

        for block in self.blocks:
            x = block(
                x,
                mask_global=mask_global,
                mask_local=mask_local,
                cos_global=self.cos_global,
                sin_global=self.sin_global,
                cos_local=self.cos_local,
                sin_local=self.sin_local,
            )

        x = self.final_norm(x)
        logits = self.out_head(x.to(self.cfg["dtype"]) if self.cfg["dtype"] is not None else x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                targets.reshape(-1),
            )
        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int | None = None,
    ) -> torch.Tensor:
        for _ in range(max_new_tokens):
            ctx_len = self.cfg["context_length"]
            idx_cond = idx if idx.size(1) <= ctx_len else idx[:, -ctx_len:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx
