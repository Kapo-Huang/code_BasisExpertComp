import torch
import torch.nn as nn
import torch.nn.functional as F

import models.utils as model_utils
import utils.utils as utils


class DummyLoss(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()

    def compute_loss(self, *args, **kwargs):
        return None

    def forward(self, *args, **kwargs):
        return None


class ValReconLossMoE(nn.Module):
    def compute_loss(self, pred_vals, gt_vals, q, *args, **kwargs):
        return (((pred_vals - gt_vals.unsqueeze(-1)) ** 2) * q.unsqueeze(-2)).mean()


class ValReconLossSingle(nn.Module):
    def compute_loss(self, pred_vals, gt_vals, *args, **kwargs):
        return ((pred_vals - gt_vals) ** 2).mean()


class ValReconLoss(nn.Module):
    def __init__(self, loss_type):
        super().__init__()
        loss_dict = {
            "none": DummyLoss(),
            "single": ValReconLossSingle(),
            "moe": ValReconLossMoE(),
            "sparsemoe": ValReconLossMoE(),
        }
        self.loss = loss_dict[loss_type]

    def forward(self, pred_vals, gt_vals, q, *args, **kwargs):
        result = self.loss.compute_loss(pred_vals, gt_vals, q, *args, **kwargs)
        if result is None:
            return gt_vals.new_zeros(())
        return result


class ValReconAllLossMoE(nn.Module):
    def compute_loss(self, pred_vals, gt_vals, *args, **kwargs):
        return ((pred_vals - gt_vals.unsqueeze(-1)) ** 2).mean()


class ValReconAllLoss(nn.Module):
    def __init__(self, loss_type):
        super().__init__()
        loss_dict = {
            "none": DummyLoss(),
            "single": ValReconLossSingle(),
            "moe": ValReconAllLossMoE(),
            "sparsemoe": ValReconAllLossMoE(),
        }
        self.loss = loss_dict[loss_type]

    def forward(self, pred_vals, gt_vals, *args, **kwargs):
        result = self.loss.compute_loss(pred_vals, gt_vals, *args, **kwargs)
        if result is None:
            return gt_vals.new_zeros(())
        return result


class BalancingLossMoE(nn.Module):
    def __init__(self, n_experts, sample_bias_correction):
        super().__init__()
        self.n_experts = n_experts
        self.sample_bias_correction = sample_bias_correction

    def top1(self, tensor):
        values, index = tensor.topk(k=1, dim=-1)
        values, index = map(lambda x: x.squeeze(dim=-1), (values, index))
        return values, index

    def compute_loss(self, q, *args, **kwargs):
        _, index_1 = self.top1(q)
        mask_1 = F.one_hot(index_1, self.n_experts).float()
        density_1_proxy = q.mean(dim=-2)
        density_1 = mask_1.mean(dim=-2)
        return (density_1_proxy * density_1).mean() * float(self.n_experts ** 2)


class BalancingLoss(nn.Module):
    def __init__(self, loss_type, n_experts, sample_bias_correction):
        super().__init__()
        loss_dict = {
            "none": DummyLoss(),
            "single": DummyLoss(),
            "moe": BalancingLossMoE(n_experts, sample_bias_correction),
            "sparsemoe": DummyLoss(),
        }
        self.loss = loss_dict[loss_type]

    def forward(self, q, *args, **kwargs):
        result = self.loss.compute_loss(q, *args, **kwargs)
        if result is None:
            return torch.zeros((), device=q.device if torch.is_tensor(q) else "cpu")
        return result


class SegmentationLossMoE(nn.Module):
    def __init__(self, n_experts, seg_type="ce"):
        super().__init__()
        self.n_experts = n_experts
        self.seg_type = seg_type

    def compute_loss(self, q, gt_segment, *args, **kwargs):
        if self.seg_type == "ce":
            return F.cross_entropy(q, gt_segment.long(), label_smoothing=0.01)
        if self.seg_type == "binary_ce":
            return F.binary_cross_entropy_with_logits(q, torch.ones_like(q.squeeze()) / self.n_experts)
        if self.seg_type == "both":
            return F.cross_entropy(q, gt_segment.long(), label_smoothing=0.01) + 0.1 * F.binary_cross_entropy_with_logits(
                q, torch.ones_like(q.squeeze()) / self.n_experts
            )
        raise ValueError(f"Unsupported segmentation loss: {self.seg_type}")


class SegmentationLoss(nn.Module):
    def __init__(self, loss_type, n_experts, seg_type="ce"):
        super().__init__()
        loss_dict = {
            "none": DummyLoss(),
            "single": DummyLoss(),
            "moe": SegmentationLossMoE(n_experts, seg_type),
            "sparsemoe": SegmentationLossMoE(n_experts, seg_type),
        }
        self.loss = loss_dict[loss_type]

    def forward(self, q, gt_segment, *args, **kwargs):
        result = self.loss.compute_loss(q, gt_segment, *args, **kwargs)
        if result is None:
            return torch.zeros((), device=q.device if torch.is_tensor(q) else "cpu")
        return result


class LoadLossMoE(nn.Module):
    def cv_squared(self, tensor, eps=1.0e-5):
        return tensor.float().var() / (tensor.float().abs().mean() + eps)

    def compute_loss(self, importance, load, q, *args, **kwargs):
        importance = q.sum(1)
        load = (q > 0).sum(1)
        return self.cv_squared(importance) + self.cv_squared(load)


class LoadLoss(nn.Module):
    def __init__(self, loss_type):
        super().__init__()
        loss_dict = {
            "none": DummyLoss(),
            "single": DummyLoss(),
            "moe": LoadLossMoE(),
            "sparsemoe": LoadLossMoE(),
        }
        self.loss = loss_dict[loss_type]

    def forward(self, importance, load, q, *args, **kwargs):
        result = self.loss.compute_loss(importance, load, q, *args, **kwargs)
        if result is None:
            if torch.is_tensor(q):
                return torch.zeros((), device=q.device, dtype=q.dtype)
            return torch.tensor(0.0)
        return result


class MSEEachExpert(nn.Module):
    def forward(self, pred_vals, gt_vals, *args, **kwargs):
        return ((pred_vals - gt_vals.unsqueeze(-1)) ** 2).mean(1).mean(1).squeeze()


LOSS_LIST = ["valrecon", "valreconall", "balance", "segmentation", "load"]


class MeshReconstructionLoss(nn.Module):
    def __init__(self, cfg, model_name, in_dim=4, model=None, n_experts=1):
        super().__init__()
        self.sample_bias_correction = bool(cfg.get("sample_bias_correction", False))
        if "_moe" in model_name:
            self.model_type = "moe"
            moe_indicator = True
        else:
            self.model_type = "single"
            moe_indicator = False

        self.model = model
        self.gradient_comp = utils.experts_gradient if moe_indicator else utils.gradient
        self.weight_dict = {}

        required_loss_list, weights = model_utils.parse_loss_string(cfg["loss_type"])
        self.required_loss_dict, self.weight_dict = model_utils.build_loss_dictionary(
            required_loss_list,
            weights,
            self.model_type,
            full_loss_list=LOSS_LIST,
        )

        self.valrecon_loss = ValReconLoss(self.required_loss_dict["valrecon"])
        self.valreconall_loss = ValReconAllLoss(self.required_loss_dict["valreconall"])
        self.balance_loss = BalancingLoss(self.required_loss_dict["balance"], n_experts, self.sample_bias_correction)
        self.segmentation_loss = SegmentationLoss(
            self.required_loss_dict["segmentation"], n_experts, cfg.get("segmentation_type", "ce")
        )
        self.load_loss = LoadLoss(self.required_loss_dict["load"])
        self.recon_mse = ValReconLossSingle()
        self.mse_each_expert = MSEEachExpert()

    def forward(self, output_pred, data=None, dataset=None):
        if data is None:
            raise ValueError("MeshReconstructionLoss requires a data batch")

        if self.model_type == "moe":
            q = output_pred["nonmnfld_q"].permute(0, 2, 1)
            raw_q = output_pred["nonmnfld_raw_q"].permute(0, 2, 1).squeeze()
            final_vals = output_pred["selected_nonmanifold_pnts_pred"]
        else:
            q = None
            raw_q = None
            final_vals = output_pred["nonmanifold_pnts_pred"].permute(0, 2, 1)

        pred_vals = (
            output_pred["nonmanifold_pnts_pred"].permute(0, 2, 3, 1)
            if output_pred["nonmanifold_pnts_pred"].dim() == 4
            else output_pred["nonmanifold_pnts_pred"].permute(0, 2, 1)
        )

        gt_segments = data.get("nonmnfld_segments_gt")
        if gt_segments is not None:
            gt_segments = gt_segments.squeeze()
        gt_vals = data.get("nonmnfld_val")

        if gt_vals is None and gt_segments is not None:
            segmentation_term = self.segmentation_loss(raw_q, gt_segments)
            return {
                "loss": segmentation_term * self.weight_dict["segmentation"],
                "segmentation_term": segmentation_term,
            }

        if gt_vals is None:
            raise ValueError("MeshReconstructionLoss expected 'nonmnfld_val' when segmentation_mode=False")

        if self.sample_bias_correction and q is not None:
            q = q * q.shape[1] / torch.clamp(q.sum(-2, keepdim=True), 1.0e-5)

        valrecon_term = self.valrecon_loss(pred_vals, gt_vals, q)
        valreconall_term = self.valreconall_loss(pred_vals, gt_vals)
        balance_term = self.balance_loss(q) if q is not None else gt_vals.new_zeros(())
        segmentation_term = (
            self.segmentation_loss(raw_q, gt_segments) if (raw_q is not None and gt_segments is not None) else gt_vals.new_zeros(())
        )
        load_term = self.load_loss(output_pred.get("importance", None), output_pred.get("load", None), q) if q is not None else gt_vals.new_zeros(())

        loss = (
            valrecon_term * self.weight_dict["valrecon"]
            + balance_term * self.weight_dict["balance"]
            + segmentation_term * self.weight_dict["segmentation"]
            + load_term * self.weight_dict["load"]
            + valreconall_term * self.weight_dict["valreconall"]
        )

        with torch.no_grad():
            recon_error = self.recon_mse.compute_loss(final_vals.squeeze(), gt_vals.squeeze())

        out = {
            "loss": loss,
            "valrecon_term": valrecon_term,
            "valreconall_term": valreconall_term,
            "balance_term": balance_term,
            "segmentation_term": segmentation_term,
            "load_term": load_term,
            "reconerror_term": recon_error,
        }
        if self.model_type != "single":
            with torch.no_grad():
                mse_each_expert = self.mse_each_expert(pred_vals, gt_vals)
                for idx, mse_value in enumerate(torch.atleast_1d(mse_each_expert)):
                    out[f"mse-expert-{idx}_term"] = mse_value
        return out
