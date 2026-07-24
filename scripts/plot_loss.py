
from __future__ import annotations
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


def load_loss_history(path: Path) -> dict[str, list[float]]:
    with path.open(encoding="utf-8") as f:
        data = json.load(f)
    if "train" not in data or "val" not in data:
        raise KeyError(f"Expected 'train' and 'val' keys in {path}")
    return data


def plot_loss(
    data: dict[str, list[float]],
    output_path: Path,
    show: bool = False,
    dpi: int = 150,
) -> Path:
    train_loss = data["train"]
    val_loss = data["val"]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(train_loss, color="tab:blue", label="train_loss")
    ax.plot(val_loss, color="tab:red", label="validation_loss")
    ax.set_xlabel("Steps (every 500 epochs)")
    ax.set_ylabel("Loss")
    ax.set_title("Training / Validation Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)

    if show:
        # Re-open for interactive display if requested
        saved = plt.imread(output_path)
        plt.imshow(saved)
        plt.axis("off")
        plt.show()

    return output_path


def build_arg_parser() -> argparse.ArgumentParser:
    root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description="Save loss curves as PNG")
    parser.add_argument(
        "--input",
        type=Path,
        default=root / "loss_history.json",
        help="Path to loss_history.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=root / "artifacts" / "loss_curve.png",
        help="Output PNG path",
    )
    parser.add_argument("--show", action="store_true", help="Also display the plot")
    parser.add_argument("--dpi", type=int, default=150)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    data = load_loss_history(args.input)
    out = plot_loss(data, args.output, show=args.show, dpi=args.dpi)
    print(f"Saved plot to {out}")


if __name__ == "__main__":
    main()
