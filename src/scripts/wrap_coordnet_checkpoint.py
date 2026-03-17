import argparse
from pathlib import Path

import torch


def wrap_checkpoint(input_path: Path, output_path: Path, y_mean: float, y_std: float) -> None:
    payload = torch.load(str(input_path), map_location="cpu")
    if isinstance(payload, dict) and "model_state" in payload:
        state_dict = payload["model_state"]
        meta = dict(payload.get("meta", {}))
    else:
        state_dict = payload
        meta = {}

    wrapped = {
        "model_state": state_dict,
        "y_mean": float(y_mean),
        "y_std": float(y_std),
        "meta": {
            **meta,
            "wrapped_from": str(input_path),
            "coord_order": meta.get("coord_order", "x_y_z_t"),
            "output": meta.get("output", "scalar_value"),
        },
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(wrapped, str(output_path))
    print(f"Saved wrapped checkpoint to: {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Wrap CoordNet checkpoint for validate_PSNR.py compatibility.")
    parser.add_argument("--input", type=str, required=True, help="Path to CoordNet checkpoint (.pth)")
    parser.add_argument("--output", type=str, required=True, help="Path to wrapped checkpoint (.pth)")
    parser.add_argument("--y-mean", type=float, default=0.0, help="Denormalization mean saved in checkpoint")
    parser.add_argument("--y-std", type=float, default=1.0, help="Denormalization std saved in checkpoint")
    args = parser.parse_args()

    wrap_checkpoint(Path(args.input), Path(args.output), args.y_mean, args.y_std)


if __name__ == "__main__":
    main()
