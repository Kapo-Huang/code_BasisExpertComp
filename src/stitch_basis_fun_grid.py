import argparse
import re
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


DEFAULT_INPUT_DIR = Path("validate_out/basis_fun/BasisExpert_LossBalancing/20")
DEFAULT_REFERENCE_SIZE = (2800, 800)
DEFAULT_CROP = (10, 45, 2780, 790)
DEFAULT_CELL_GAP = 24
DEFAULT_HEADER_SIZE = 48
DEFAULT_EXPERT_COUNT = 6

EXPERT_RE = re.compile(r"^(?P<attr>.+)_expert_(?P<idx>\d+)_orth\.png$", re.IGNORECASE)
GT_RE = re.compile(r"^(?P<attr>.+)_gt_orth\.png$", re.IGNORECASE)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Stitch basis-function expert/GT images into a single overview grid."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=DEFAULT_INPUT_DIR,
        help="Directory containing *_expert_XX_orth.png and *_gt_orth.png images.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output image path. Defaults to basis_fun_expert_grid_t<dirname>.png inside input-dir.",
    )
    parser.add_argument(
        "--attr-order",
        type=str,
        default=None,
        help="Comma-separated attribute order. Unspecified discovered attrs are appended in sorted order.",
    )
    parser.add_argument(
        "--crop",
        type=str,
        default=None,
        help="Crop box as left,top,right,bottom. Defaults to a box tuned for 2800x800 inputs.",
    )
    parser.add_argument(
        "--cell-gap",
        type=int,
        default=DEFAULT_CELL_GAP,
        help="Gap in pixels between cells.",
    )
    parser.add_argument(
        "--header-size",
        type=int,
        default=DEFAULT_HEADER_SIZE,
        help="Font size for row/column headers.",
    )
    return parser.parse_args()


def parse_crop(text: str):
    parts = [part.strip() for part in str(text).split(",")]
    if len(parts) != 4:
        raise ValueError("--crop must have exactly 4 comma-separated integers: left,top,right,bottom")
    try:
        crop = tuple(int(part) for part in parts)
    except ValueError as exc:
        raise ValueError("--crop values must be integers") from exc
    if crop[0] >= crop[2] or crop[1] >= crop[3]:
        raise ValueError("--crop must satisfy left < right and top < bottom")
    return crop


def parse_attr_order(text: str | None):
    if text is None:
        return None
    items = [item.strip() for item in str(text).split(",") if item.strip()]
    return items or None


def scaled_default_crop(size):
    src_w, src_h = size
    ref_w, ref_h = DEFAULT_REFERENCE_SIZE
    left, top, right, bottom = DEFAULT_CROP
    scaled = (
        round(left * src_w / ref_w),
        round(top * src_h / ref_h),
        round(right * src_w / ref_w),
        round(bottom * src_h / ref_h),
    )
    return scaled


def infer_output_path(input_dir: Path):
    name = input_dir.name
    if name.isdigit():
        return input_dir / f"basis_fun_expert_grid_t{name}.png"
    return input_dir / "basis_fun_expert_grid.png"


def load_font(size: int, bold: bool = False):
    candidates = [
        "C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf",
        "arialbd.ttf" if bold else "arial.ttf",
        "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf",
        "C:/Windows/Fonts/segoeuib.ttf" if bold else "C:/Windows/Fonts/segoeui.ttf",
    ]
    for candidate in candidates:
        try:
            return ImageFont.truetype(candidate, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


def measure_text(draw: ImageDraw.ImageDraw, text: str, font):
    left, top, right, bottom = draw.textbbox((0, 0), text, font=font)
    return right - left, bottom - top


def scan_groups(input_dir: Path):
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")
    if not input_dir.is_dir():
        raise NotADirectoryError(f"Input path is not a directory: {input_dir}")

    groups = {}
    ignored = []
    for path in sorted(input_dir.iterdir()):
        if not path.is_file() or path.suffix.lower() != ".png":
            continue

        gt_match = GT_RE.match(path.name)
        if gt_match:
            attr = gt_match.group("attr")
            slot = groups.setdefault(attr, {"gt": None, "experts": {}})
            if slot["gt"] is not None:
                raise ValueError(f"Duplicate GT image for attr '{attr}': {path.name}")
            slot["gt"] = path
            continue

        expert_match = EXPERT_RE.match(path.name)
        if expert_match:
            attr = expert_match.group("attr")
            idx = int(expert_match.group("idx"))
            slot = groups.setdefault(attr, {"gt": None, "experts": {}})
            if idx in slot["experts"]:
                raise ValueError(f"Duplicate expert {idx} image for attr '{attr}': {path.name}")
            slot["experts"][idx] = path
            continue

        ignored.append(path.name)

    for name in ignored:
        print(f"Ignoring unmatched PNG: {name}", file=sys.stderr)

    if not groups:
        raise ValueError(f"No matching PNG images found in: {input_dir}")
    return groups


def resolve_attr_order(groups, requested_order):
    discovered = sorted(groups)
    if requested_order is None:
        return discovered

    missing = [attr for attr in requested_order if attr not in groups]
    if missing:
        raise ValueError(f"Unknown attrs in --attr-order: {missing}. Discovered attrs: {discovered}")

    remaining = [attr for attr in discovered if attr not in requested_order]
    return requested_order + remaining


def validate_groups(groups, attr_order):
    for attr in attr_order:
        item = groups[attr]
        if item["gt"] is None:
            raise ValueError(f"Missing GT image for attr '{attr}'")

        experts = item["experts"]
        if len(experts) != DEFAULT_EXPERT_COUNT:
            present = sorted(experts)
            missing = [idx for idx in range(DEFAULT_EXPERT_COUNT) if idx not in experts]
            raise ValueError(
                f"Attr '{attr}' must have {DEFAULT_EXPERT_COUNT} experts; "
                f"present={present}, missing={missing}"
            )

        expected = list(range(DEFAULT_EXPERT_COUNT))
        present = sorted(experts)
        if present != expected:
            raise ValueError(f"Attr '{attr}' expert indices must be {expected}, got {present}")


def resolve_crop_box(size, crop_arg):
    crop = parse_crop(crop_arg) if crop_arg else scaled_default_crop(size)
    left, top, right, bottom = crop
    width, height = size
    if left < 0 or top < 0 or right > width or bottom > height:
        raise ValueError(
            f"Crop {crop} is outside image size {size}. "
            "Provide --crop explicitly if the defaults do not fit."
        )
    return crop


def collect_layout_info(groups, attr_order, crop_box):
    sample_size = None
    cell_size = None

    for attr in attr_order:
        image_paths = [groups[attr]["experts"][idx] for idx in range(DEFAULT_EXPERT_COUNT)] + [groups[attr]["gt"]]
        for path in image_paths:
            with Image.open(path) as image:
                if sample_size is None:
                    sample_size = image.size
                elif image.size != sample_size:
                    raise ValueError(f"All images must have the same size; got {image.size} and {sample_size}")

    left, top, right, bottom = crop_box
    cell_size = (right - left, bottom - top)
    return sample_size, cell_size


def render_grid(groups, attr_order, output_path: Path, crop_box, cell_gap: int, header_size: int):
    column_labels = [f"Expert {idx}" for idx in range(DEFAULT_EXPERT_COUNT)] + ["GT"]
    _, (cell_w, cell_h) = collect_layout_info(groups, attr_order, crop_box)

    header_font = load_font(header_size, bold=True)
    row_font = load_font(header_size, bold=True)

    probe = Image.new("RGB", (32, 32), "white")
    probe_draw = ImageDraw.Draw(probe)

    row_label_width = max(measure_text(probe_draw, attr, row_font)[0] for attr in attr_order)
    col_header_height = max(measure_text(probe_draw, label, header_font)[1] for label in column_labels)

    outer_pad = max(cell_gap, header_size // 2)
    row_label_width += outer_pad
    col_header_height += outer_pad

    grid_origin_x = outer_pad + row_label_width + cell_gap
    grid_origin_y = outer_pad + col_header_height + cell_gap
    canvas_w = grid_origin_x + len(column_labels) * cell_w + (len(column_labels) - 1) * cell_gap + outer_pad
    canvas_h = grid_origin_y + len(attr_order) * cell_h + (len(attr_order) - 1) * cell_gap + outer_pad

    canvas = Image.new("RGB", (canvas_w, canvas_h), "white")
    draw = ImageDraw.Draw(canvas)

    for col_idx, label in enumerate(column_labels):
        x = grid_origin_x + col_idx * (cell_w + cell_gap)
        text_w, text_h = measure_text(draw, label, header_font)
        text_x = x + (cell_w - text_w) // 2
        text_y = outer_pad + (col_header_height - text_h) // 2
        draw.text((text_x, text_y), label, fill="black", font=header_font)

    for row_idx, attr in enumerate(attr_order):
        y = grid_origin_y + row_idx * (cell_h + cell_gap)
        text_w, text_h = measure_text(draw, attr, row_font)
        text_x = outer_pad + row_label_width - text_w - cell_gap // 2
        text_y = y + (cell_h - text_h) // 2
        draw.text((text_x, text_y), attr, fill="black", font=row_font)

        image_paths = [groups[attr]["experts"][idx] for idx in range(DEFAULT_EXPERT_COUNT)] + [groups[attr]["gt"]]
        for col_idx, path in enumerate(image_paths):
            x = grid_origin_x + col_idx * (cell_w + cell_gap)
            with Image.open(path) as image:
                cropped = image.convert("RGB").crop(crop_box)
            canvas.paste(cropped, (x, y))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)
    return canvas.size


def main():
    args = parse_args()
    if args.cell_gap < 0:
        raise ValueError("--cell-gap must be >= 0")
    if args.header_size <= 0:
        raise ValueError("--header-size must be > 0")

    groups = scan_groups(args.input_dir)
    attr_order = resolve_attr_order(groups, parse_attr_order(args.attr_order))
    validate_groups(groups, attr_order)

    first_attr = attr_order[0]
    sample_path = groups[first_attr]["experts"][0]
    with Image.open(sample_path) as image:
        crop_box = resolve_crop_box(image.size, args.crop)

    output_path = args.output or infer_output_path(args.input_dir)
    canvas_size = render_grid(
        groups=groups,
        attr_order=attr_order,
        output_path=output_path,
        crop_box=crop_box,
        cell_gap=args.cell_gap,
        header_size=args.header_size,
    )

    print(f"Saved grid to: {output_path}")
    print(f"Attributes: {attr_order}")
    print(f"Crop box: {crop_box}")
    print(f"Canvas size: {canvas_size[0]}x{canvas_size[1]}")


if __name__ == "__main__":
    main()
