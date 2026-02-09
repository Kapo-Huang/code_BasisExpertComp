import argparse
from pathlib import Path

import numpy as np


def save_first_n(src_path: Path, dst_path: Path, n_rows: int, chunk_size: int) -> tuple:
    src = np.load(src_path, mmap_mode="r")
    if src.shape[0] < n_rows:
        raise ValueError(f"{src_path} has {src.shape[0]} rows, smaller than requested {n_rows}.")

    dst_path.parent.mkdir(parents=True, exist_ok=True)
    dst = np.lib.format.open_memmap(
        str(dst_path),
        mode="w+",
        dtype=src.dtype,
        shape=(n_rows,) + src.shape[1:],
    )

    for start in range(0, n_rows, chunk_size):
        end = min(start + chunk_size, n_rows)
        dst[start:end] = src[start:end]

    dst.flush()
    return src.shape, dst.shape


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("data/raw/sukong/train"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/raw/sukong/train_subset_first_2108510"),
    )
    parser.add_argument(
        "--n-rows",
        type=int,
        default=2_108_510,
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=200_000,
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()

    files = {
        "source_cell_XYZT.npy": "source_cell_XYZT.npy",
        "target_cell_E.npy": "target_cell_E.npy",
        "target_cell_E_IntegrationPoints.npy": "target_cell_E_IntegrationPoints.npy",
    }

    print(f"Creating subset with first {args.n_rows} rows")
    print(f"Input dir:  {args.input_dir}")
    print(f"Output dir: {args.output_dir}")

    for src_name, dst_name in files.items():
        src_path = args.input_dir / src_name
        dst_path = args.output_dir / dst_name
        src_shape, dst_shape = save_first_n(src_path, dst_path, args.n_rows, args.chunk_size)
        print(f"{src_name}: {src_shape} -> {dst_shape}")

    print("Done.")


if __name__ == "__main__":
    main()
