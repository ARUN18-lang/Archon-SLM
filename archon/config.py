from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent.parent / "configs" / "Archon_270m.json"

_DTYPE_MAP: dict[str | None, torch.dtype | None] = {
    None: None,
    "null": None,
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}


@dataclass(frozen=True)
class ModelConfig:
    vocab_size: int
    context_length: int
    emb_dim: int
    n_heads: int
    n_layers: int
    hidden_dim: int
    head_dim: int
    qk_norm: bool
    n_kv_groups: int
    rope_local_base: float
    rope_base: float
    sliding_window: int
    layer_types: list[str]
    query_pre_attn_scalar: int
    dtype: torch.dtype | None = None

    def __post_init__(self) -> None:
        if len(self.layer_types) != self.n_layers:
            raise ValueError(
                f"layer_types length ({len(self.layer_types)}) must equal n_layers ({self.n_layers})"
            )
        if self.n_heads % self.n_kv_groups != 0:
            raise ValueError("n_heads must be divisible by n_kv_groups")

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        if self.dtype is not None:
            data["dtype"] = str(self.dtype).replace("torch.", "")
        else:
            data["dtype"] = None
        return data


def _resolve_dtype(raw: Any) -> torch.dtype | None:
    if isinstance(raw, torch.dtype):
        return raw
    if raw is None:
        return None
    key = str(raw).lower().replace("torch.", "")
    if key not in _DTYPE_MAP:
        raise ValueError(f"Unsupported dtype: {raw!r}")
    return _DTYPE_MAP[key]


def load_config(path: str | Path | None = None) -> ModelConfig:
    config_path = Path(path) if path is not None else DEFAULT_CONFIG_PATH
    with config_path.open(encoding="utf-8") as f:
        raw: dict[str, Any] = json.load(f)

    raw["dtype"] = _resolve_dtype(raw.get("dtype"))
    return ModelConfig(**raw)


def config_from_dict(data: dict[str, Any]) -> ModelConfig:
    payload = dict(data)
    payload["dtype"] = _resolve_dtype(payload.get("dtype"))
    return ModelConfig(**payload)
