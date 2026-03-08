def unpack_batch(batch):
    if len(batch) == 2:
        xb, yb = batch
        return xb, yb
    raise ValueError(f"Unexpected batch structure: {len(batch)}")


def is_multiview_target(targets) -> bool:
    return isinstance(targets, dict)