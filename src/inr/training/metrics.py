import math

import torch

from inr.training.batches import unpack_batch


def compute_psnr_streaming_single(model, loader, dataset, device, hard_topk: bool = True, progress_factory=None) -> float:
    model.eval()
    total_se = 0.0
    total_count = 0
    gt_min = float("inf")
    gt_max = float("-inf")
    with torch.no_grad():
        iterator = loader
        if progress_factory is not None:
            iterator = progress_factory(loader, desc="psnr", leave=False)
        for batch in iterator:
            xb, yb = unpack_batch(batch)
            xb = xb.to(device)
            yb = yb.to(device)
            try:
                pred = model(xb, hard_topk=hard_topk)
            except TypeError:
                pred = model(xb)
            if hasattr(dataset, "denormalize_targets"):
                pred = dataset.denormalize_targets(pred)
                yb = dataset.denormalize_targets(yb)
            se = torch.sum((pred - yb) ** 2)
            total_se += float(se.item())
            total_count += int(pred.numel())
            gt_min = min(gt_min, float(yb.min().item()))
            gt_max = max(gt_max, float(yb.max().item()))
    data_range = gt_max - gt_min
    if data_range <= 0:
        data_range = 1.0
    if total_count == 0:
        return float("nan")
    mse = total_se / total_count
    if mse <= 0:
        return float("inf")
    return 10.0 * math.log10((data_range ** 2) / mse)


def compute_psnr_streaming_multiview(model, loader, dataset, device, hard_topk: bool = True, progress_factory=None) -> dict:
    model.eval()
    total_se = {}
    total_count = {}
    gt_min = {}
    gt_max = {}
    with torch.no_grad():
        iterator = loader
        if progress_factory is not None:
            iterator = progress_factory(loader, desc="psnr", leave=False)
        for batch in iterator:
            xb, yb = unpack_batch(batch)
            xb = xb.to(device)
            try:
                preds = model(xb, hard_topk=hard_topk)
            except TypeError:
                preds = model(xb)
            for name, pred in preds.items():
                target = yb[name].to(device)
                if hasattr(dataset, "denormalize_attr"):
                    pred = dataset.denormalize_attr(name, pred)
                    target = dataset.denormalize_attr(name, target)
                se = torch.sum((pred - target) ** 2)
                total_se[name] = total_se.get(name, 0.0) + float(se.item())
                total_count[name] = total_count.get(name, 0) + int(pred.numel())
                cur_min = float(target.min().item())
                cur_max = float(target.max().item())
                gt_min[name] = min(gt_min.get(name, cur_min), cur_min)
                gt_max[name] = max(gt_max.get(name, cur_max), cur_max)

    psnr_vals = {}
    for name in total_se.keys():
        data_range = gt_max[name] - gt_min[name]
        if data_range <= 0:
            data_range = 1.0
        mse = total_se[name] / max(1, total_count[name])
        psnr_vals[name] = float("inf") if mse <= 0 else 10.0 * math.log10((data_range ** 2) / mse)
    return psnr_vals