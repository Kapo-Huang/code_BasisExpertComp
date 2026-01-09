from typing import Dict, List, Optional, Sequence, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from .moe_inr import ExpertDecoder, PolicyNetwork, SharedSirenEncoder


class TimeMLP(nn.Module):
    def __init__(self, in_dim: int = 1, hidden_dim: int = 32, out_dim: int = 16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def _as_tuple3(value) -> Tuple[int, int, int]:
    if isinstance(value, (list, tuple)) and len(value) == 3:
        return int(value[0]), int(value[1]), int(value[2])
    v = int(value)
    return v, v, v


def _compute_bbox(coords: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    bbox_min = coords.min(dim=0).values
    bbox_max = coords.max(dim=0).values
    return bbox_min, bbox_max


def _compute_bin_stats(
    coords: torch.Tensor,
    values: torch.Tensor,
    grid_size: Tuple[int, int, int],
    bbox_min: torch.Tensor,
    bbox_max: torch.Tensor,
    eps: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    coords = coords[:, :3]
    values = values.view(values.shape[0], -1).mean(dim=-1)
    grid_size_t = torch.tensor(grid_size, device=coords.device, dtype=torch.float32)
    scale = (bbox_max - bbox_min).clamp_min(eps)
    normalized = (coords - bbox_min) / scale
    normalized = normalized.clamp(0.0, 1.0)
    idx = torch.floor(normalized * grid_size_t).to(torch.int64)
    idx = torch.minimum(idx, torch.tensor(grid_size, device=coords.device, dtype=torch.int64) - 1)

    flat_idx = idx[:, 0] * (grid_size[1] * grid_size[2]) + idx[:, 1] * grid_size[2] + idx[:, 2]
    total_bins = grid_size[0] * grid_size[1] * grid_size[2]

    grid_sum = torch.zeros(total_bins, device=coords.device, dtype=values.dtype)
    grid_cnt = torch.zeros(total_bins, device=coords.device, dtype=values.dtype)
    grid_sum_sq = torch.zeros(total_bins, device=coords.device, dtype=values.dtype)
    grid_sum.scatter_add_(0, flat_idx, values)
    grid_cnt.scatter_add_(0, flat_idx, torch.ones_like(values))
    grid_sum_sq.scatter_add_(0, flat_idx, values * values)

    grid_sum = grid_sum.view(*grid_size)
    grid_cnt = grid_cnt.view(*grid_size)
    grid_sum_sq = grid_sum_sq.view(*grid_size)
    return grid_sum, grid_cnt, grid_sum_sq


def _make_gaussian_kernel(size: int, sigma: Optional[float] = None, device=None, dtype=None) -> torch.Tensor:
    if sigma is None:
        sigma = float(size) / 3.0
    coords = torch.arange(size, device=device, dtype=dtype) - (size - 1) / 2.0
    kernel_1d = torch.exp(-0.5 * (coords / sigma) ** 2)
    kernel_1d = kernel_1d / kernel_1d.sum()
    kernel_3d = kernel_1d[:, None, None] * kernel_1d[None, :, None] * kernel_1d[None, None, :]
    return kernel_3d


def _smooth_grid(
    grid_sum: torch.Tensor, grid_cnt: torch.Tensor, kernel: torch.Tensor, eps: float
) -> torch.Tensor:
    grid_sum = grid_sum.unsqueeze(0).unsqueeze(0)
    grid_cnt = grid_cnt.unsqueeze(0).unsqueeze(0)
    kernel = kernel.unsqueeze(0).unsqueeze(0)
    pad = kernel.shape[-1] // 2
    smooth_sum = F.conv3d(grid_sum, kernel, padding=pad)
    smooth_cnt = F.conv3d(grid_cnt, kernel, padding=pad)
    smooth = smooth_sum / smooth_cnt.clamp_min(eps)
    return smooth.squeeze(0).squeeze(0)


def compute_indicator_grid(
    coords: torch.Tensor,
    values: torch.Tensor,
    grid_size: Tuple[int, int, int],
    kernel_sizes: Sequence[int],
    eps: float = 1e-6,
) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
    bbox_min, bbox_max = _compute_bbox(coords[:, :3])
    grid_sum, grid_cnt, grid_sum_sq = _compute_bin_stats(coords, values, grid_size, bbox_min, bbox_max, eps)
    grid_cnt_safe = grid_cnt.clamp_min(1.0)
    grid_mean = grid_sum / grid_cnt_safe
    grid_var = grid_sum_sq / grid_cnt_safe - grid_mean * grid_mean

    smoothed = []
    for size in kernel_sizes:
        size = int(size)
        if size % 2 == 0:
            size += 1
        kernel = _make_gaussian_kernel(size, device=coords.device, dtype=coords.dtype)
        smoothed.append(_smooth_grid(grid_sum, grid_cnt, kernel, eps))

    if len(smoothed) == 0:
        diffs = [grid_mean.abs()]
    elif len(smoothed) == 1:
        diffs = [(grid_mean - smoothed[0]).abs()]
    else:
        diffs = [(smoothed[i - 1] - smoothed[i]).abs() for i in range(1, len(smoothed))]

    hf_energy = torch.zeros_like(grid_mean)
    for d in diffs:
        hf_energy = hf_energy + d

    def _normalize(t: torch.Tensor) -> torch.Tensor:
        min_val = t.min()
        max_val = t.max()
        return (t - min_val) / (max_val - min_val + eps)

    density = _normalize(grid_cnt)
    variance = _normalize(grid_var.clamp_min(0.0))
    hf_energy = _normalize(hf_energy)
    diffs = [_normalize(d) for d in diffs]

    channels: Dict[str, torch.Tensor] = {
        "hf": hf_energy,
        "density": density,
        "variance": variance,
        "scale_diffs": torch.stack(diffs, dim=-1) if len(diffs) > 1 else diffs[0].unsqueeze(-1),
    }
    return channels, bbox_min, bbox_max


class PatchIndicatorGrid(nn.Module):
    def __init__(
        self,
        grid_values: torch.Tensor,
        bbox_min: torch.Tensor,
        bbox_max: torch.Tensor,
        prior: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        self.grid = nn.Parameter(grid_values)
        self.register_buffer("bbox_min", bbox_min)
        self.register_buffer("bbox_max", bbox_max)
        self.register_buffer("grid_size", torch.tensor(grid_values.shape[:3], dtype=torch.int64))
        if prior is None:
            prior = grid_values.detach().clone()
        self.register_buffer("prior", prior)

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        coords = coords[:, :3]
        eps = 1e-6
        scale = (self.bbox_max - self.bbox_min).clamp_min(eps)
        normalized = (coords - self.bbox_min) / scale
        normalized = normalized.clamp(0.0, 1.0)

        grid_size = self.grid_size.to(coords.device).to(coords.dtype)
        pos = normalized * (grid_size - 1.0)
        idx0 = torch.floor(pos).to(torch.int64)
        idx1 = torch.minimum(idx0 + 1, (self.grid_size - 1).to(coords.device))
        weight = (pos - idx0.to(coords.dtype)).clamp(0.0, 1.0)

        x0, y0, z0 = idx0[:, 0], idx0[:, 1], idx0[:, 2]
        x1, y1, z1 = idx1[:, 0], idx1[:, 1], idx1[:, 2]
        wx, wy, wz = weight[:, 0:1], weight[:, 1:2], weight[:, 2:3]

        c000 = self.grid[x0, y0, z0]
        c100 = self.grid[x1, y0, z0]
        c010 = self.grid[x0, y1, z0]
        c110 = self.grid[x1, y1, z0]
        c001 = self.grid[x0, y0, z1]
        c101 = self.grid[x1, y0, z1]
        c011 = self.grid[x0, y1, z1]
        c111 = self.grid[x1, y1, z1]

        c00 = c000 * (1.0 - wx) + c100 * wx
        c01 = c001 * (1.0 - wx) + c101 * wx
        c10 = c010 * (1.0 - wx) + c110 * wx
        c11 = c011 * (1.0 - wx) + c111 * wx
        c0 = c00 * (1.0 - wy) + c10 * wy
        c1 = c01 * (1.0 - wy) + c11 * wy
        return c0 * (1.0 - wz) + c1 * wz

    def regularization(
        self,
        smooth_weight: float = 0.0,
        prior_weight: float = 0.0,
        amplitude_weight: float = 0.0,
        amplitude_min: float = 0.0,
        amplitude_max: float = 1.0,
        smooth_kernel: int = 3,
    ) -> torch.Tensor:
        reg = torch.zeros(1, device=self.grid.device, dtype=self.grid.dtype)
        if smooth_weight > 0.0:
            k = max(1, int(smooth_kernel))
            if k % 2 == 0:
                k += 1
            grid = self.grid.permute(3, 0, 1, 2).unsqueeze(0)
            smooth = F.avg_pool3d(grid, kernel_size=k, stride=1, padding=k // 2)
            reg = reg + smooth_weight * F.mse_loss(grid, smooth)
        if prior_weight > 0.0:
            reg = reg + prior_weight * F.mse_loss(self.grid, self.prior)
        if amplitude_weight > 0.0:
            hi = F.relu(self.grid - float(amplitude_max))
            lo = F.relu(float(amplitude_min) - self.grid)
            reg = reg + amplitude_weight * (hi * hi + lo * lo).mean()
        return reg


class ConditioningMoEINR(nn.Module):
    def __init__(
        self,
        in_features: int = 4,
        out_features: int = 1,
        num_experts: int = 7,
        indicator_dim: int = 1,
        indicator_grid: Optional[PatchIndicatorGrid] = None,
        indicator_features: Optional[List[str]] = None,
        time_feature_dim: int = 0,
        time_mlp_hidden_dim: int = 32,
        indicator_reg_smooth: float = 0.0,
        indicator_reg_prior: float = 0.0,
        indicator_reg_amplitude: float = 0.0,
        indicator_reg_amplitude_min: float = 0.0,
        indicator_reg_amplitude_max: float = 1.0,
        indicator_reg_smooth_kernel: int = 3,
        hard_routing_at_eval: bool = True,
        straight_through_routing: bool = False,
        routing_top_k: int = 1,
        routing_load_balance_weight: float = 0.0,
        routing_entropy_weight: float = 0.0,
        encoder_feature_dim: int = 256,
        encoder_first_omega_0: float = 30.0,
        encoder_hidden_omega_0: float = 30.0,
        policy_hidden_dim: int = 128,
        policy_num_layers: int = 3,
        policy_first_omega_0: float = 30.0,
        policy_hidden_omega_0: float = 30.0,
        expert_hidden_dim: int = 256,
        expert_num_layers: int = 3,
        expert_first_omega_0: float = 30.0,
        expert_hidden_omega_0: float = 30.0,
        input_mean: Optional[torch.Tensor] = None,
        input_std: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        self.encoder = SharedSirenEncoder(
            in_features=in_features,
            feature_dim=encoder_feature_dim,
            num_frequencies=6,
            include_input=True,
            first_omega_0=encoder_first_omega_0,
            hidden_omega_0=encoder_hidden_omega_0,
        )

        self.grid_indicator_dim = int(indicator_dim)
        self.time_feature_dim = max(0, int(time_feature_dim))
        self.time_mlp = None
        if self.time_feature_dim > 0:
            self.time_mlp = TimeMLP(in_dim=1, hidden_dim=int(time_mlp_hidden_dim), out_dim=self.time_feature_dim)

        total_indicator_dim = self.grid_indicator_dim + self.time_feature_dim
        policy_in_features = in_features + total_indicator_dim
        self.policy = PolicyNetwork(
            in_features=policy_in_features,
            hidden_dim=policy_hidden_dim,
            num_layers=policy_num_layers,
            num_experts=num_experts,
            first_omega_0=policy_first_omega_0,
            hidden_omega_0=policy_hidden_omega_0,
        )

        fused_dim = encoder_feature_dim + policy_hidden_dim + total_indicator_dim
        self.num_experts = num_experts
        self.experts = nn.ModuleList(
            [
                ExpertDecoder(
                    in_dim=fused_dim,
                    out_features=out_features,
                    hidden_dim=expert_hidden_dim,
                    num_layers=expert_num_layers,
                    first_omega_0=expert_first_omega_0,
                    hidden_omega_0=expert_hidden_omega_0,
                )
                for _ in range(num_experts)
            ]
        )
        self.out_features = out_features
        self.indicator_grid = indicator_grid
        self.indicator_dim = total_indicator_dim
        self.indicator_features = indicator_features or ["hf"]
        self.indicator_reg_smooth = float(indicator_reg_smooth)
        self.indicator_reg_prior = float(indicator_reg_prior)
        self.indicator_reg_amplitude = float(indicator_reg_amplitude)
        self.indicator_reg_amplitude_min = float(indicator_reg_amplitude_min)
        self.indicator_reg_amplitude_max = float(indicator_reg_amplitude_max)
        self.indicator_reg_smooth_kernel = int(indicator_reg_smooth_kernel)
        self.hard_routing_at_eval = bool(hard_routing_at_eval)
        self.straight_through_routing = bool(straight_through_routing)
        self.routing_top_k = max(1, int(routing_top_k))
        self.routing_load_balance_weight = float(routing_load_balance_weight)
        self.routing_entropy_weight = float(routing_entropy_weight)
        self._last_probs = None

        self.register_buffer("input_mean", input_mean if input_mean is not None else torch.zeros(1, in_features))
        self.register_buffer("input_std", input_std if input_std is not None else torch.ones(1, in_features))
        self.use_input_stats = input_mean is not None and input_std is not None

    def _to_raw_coords(self, x: torch.Tensor) -> torch.Tensor:
        if not self.use_input_stats:
            return x
        return x * self.input_std.to(x.device) + self.input_mean.to(x.device)

    def indicator_regularization(self) -> torch.Tensor:
        if self.indicator_grid is None:
            return torch.zeros(1, device=self.input_mean.device, dtype=self.input_mean.dtype)
        return self.indicator_grid.regularization(
            smooth_weight=self.indicator_reg_smooth,
            prior_weight=self.indicator_reg_prior,
            amplitude_weight=self.indicator_reg_amplitude,
            amplitude_min=self.indicator_reg_amplitude_min,
            amplitude_max=self.indicator_reg_amplitude_max,
            smooth_kernel=self.indicator_reg_smooth_kernel,
        )

    def routing_regularization(self) -> torch.Tensor:
        if self._last_probs is None:
            return torch.zeros(1, device=self.input_mean.device, dtype=self.input_mean.dtype)
        probs = self._last_probs
        reg = torch.zeros(1, device=probs.device, dtype=probs.dtype)
        if self.routing_load_balance_weight > 0.0:
            mean_probs = probs.mean(dim=0)
            target = torch.full_like(mean_probs, 1.0 / float(self.num_experts))
            reg = reg + self.routing_load_balance_weight * torch.sum((mean_probs - target) ** 2)
        if self.routing_entropy_weight > 0.0:
            entropy = -(probs * (probs + 1e-9).log()).sum(dim=-1).mean()
            reg = reg - self.routing_entropy_weight * entropy
        return reg

    def regularization_loss(self) -> torch.Tensor:
        return self.indicator_regularization() + self.routing_regularization()

    def forward(
        self,
        x: torch.Tensor,
        *,
        hard_routing: bool = False,
        return_all: bool = False,
    ):
        enc_feat = self.encoder(x)
        indicator = torch.zeros(x.shape[0], self.indicator_dim, device=x.device, dtype=x.dtype)
        if self.indicator_grid is not None:
            raw_coords = self._to_raw_coords(x)
            grid_feat = self.indicator_grid(raw_coords)
        else:
            grid_feat = torch.zeros(x.shape[0], self.grid_indicator_dim, device=x.device, dtype=x.dtype)

        if self.time_mlp is not None:
            raw_coords = self._to_raw_coords(x)
            t_feat = self.time_mlp(raw_coords[:, -1:].contiguous())
        else:
            t_feat = torch.zeros(x.shape[0], self.time_feature_dim, device=x.device, dtype=x.dtype)

        indicator = torch.cat([grid_feat, t_feat], dim=-1)
        router_in = torch.cat([x, indicator], dim=-1)
        probs, logits, pol_feat = self.policy(router_in)
        self._last_probs = probs
        fused = torch.cat([enc_feat, pol_feat, indicator], dim=-1)

        preds = [expert(fused) for expert in self.experts]
        preds_all = torch.stack(preds, dim=1)

        use_hard = hard_routing or (not self.training and self.hard_routing_at_eval)
        if use_hard:
            if self.routing_top_k > 1:
                topk = torch.topk(probs, k=self.routing_top_k, dim=-1)
                mask = torch.zeros_like(probs).scatter_(1, topk.indices, 1.0)
                hard_probs = probs * mask
                hard_probs = hard_probs / hard_probs.sum(dim=-1, keepdim=True).clamp_min(1e-6)
            else:
                indices = torch.argmax(probs, dim=-1)
                hard_probs = F.one_hot(indices, num_classes=probs.shape[-1]).to(probs.dtype)

            if self.training and self.straight_through_routing:
                probs = hard_probs.detach() - probs.detach() + probs
            else:
                probs = hard_probs
        y = torch.sum(preds_all * probs.unsqueeze(-1), dim=1)

        if return_all:
            return y, preds_all, probs, logits, indicator
        return y


def build_conditioning_moe_inr_from_config(cfg, dataset=None) -> ConditioningMoEINR:
    indicator_dim = int(cfg.get("indicator_dim", 1))
    grid_size = _as_tuple3(cfg.get("indicator_grid_size", 16))
    kernel_sizes = cfg.get("indicator_kernel_sizes", [3, 7, 11])
    kernel_sizes = [int(k) for k in kernel_sizes]
    indicator_features = cfg.get("indicator_features", ["hf", "density", "variance", "scale_diffs"])

    indicator_grid = None
    input_mean = None
    input_std = None

    if dataset is not None:
        x_raw = dataset.x
        y_raw = dataset.y
        if dataset.normalize:
            input_mean = dataset.x_mean
            input_std = dataset.x_std
            x_raw = x_raw * dataset.x_std + dataset.x_mean
            y_raw = y_raw * dataset.y_std + dataset.y_mean
        indicator_channels, bbox_min, bbox_max = compute_indicator_grid(
            x_raw.to(torch.float32),
            y_raw.to(torch.float32),
            grid_size=grid_size,
            kernel_sizes=kernel_sizes,
        )
        channel_list: List[torch.Tensor] = []
        for name in indicator_features:
            if name == "scale_diffs":
                scale_tensor = indicator_channels[name]
                if scale_tensor.dim() == 3:
                    scale_tensor = scale_tensor.unsqueeze(-1)
                for i in range(scale_tensor.shape[-1]):
                    channel_list.append(scale_tensor[..., i])
            else:
                channel_list.append(indicator_channels[name])
        if len(channel_list) == 0:
            channel_list = [indicator_channels["hf"]]
        indicator_init = torch.stack(channel_list, dim=-1)
        if indicator_dim != indicator_init.shape[-1]:
            if indicator_dim < indicator_init.shape[-1]:
                indicator_init = indicator_init[..., :indicator_dim]
            else:
                pad = indicator_dim - indicator_init.shape[-1]
                indicator_init = torch.cat(
                    [indicator_init, torch.zeros(*indicator_init.shape[:-1], pad, device=indicator_init.device, dtype=indicator_init.dtype)],
                    dim=-1,
                )
        indicator_grid = PatchIndicatorGrid(indicator_init, bbox_min, bbox_max, prior=indicator_init.clone())
    else:
        bbox_min = torch.tensor(cfg.get("indicator_bbox_min", [0.0, 0.0, 0.0]), dtype=torch.float32)
        bbox_max = torch.tensor(cfg.get("indicator_bbox_max", [1.0, 1.0, 1.0]), dtype=torch.float32)
        indicator_init = torch.zeros((*grid_size, indicator_dim), dtype=torch.float32)
        indicator_grid = PatchIndicatorGrid(indicator_init, bbox_min, bbox_max, prior=indicator_init.clone())

    return ConditioningMoEINR(
        in_features=int(cfg.get("in_features", 4)),
        out_features=int(cfg.get("out_features", 1)),
        num_experts=int(cfg.get("num_experts", 7)),
        indicator_dim=indicator_dim,
        indicator_grid=indicator_grid,
        indicator_features=indicator_features,
        time_feature_dim=int(cfg.get("time_feature_dim", 0)),
        time_mlp_hidden_dim=int(cfg.get("time_mlp_hidden_dim", 32)),
        indicator_reg_smooth=float(cfg.get("indicator_reg_smooth", 0.0)),
        indicator_reg_prior=float(cfg.get("indicator_reg_prior", 0.0)),
        indicator_reg_amplitude=float(cfg.get("indicator_reg_amplitude", 0.0)),
        indicator_reg_amplitude_min=float(cfg.get("indicator_reg_amplitude_min", 0.0)),
        indicator_reg_amplitude_max=float(cfg.get("indicator_reg_amplitude_max", 1.0)),
        indicator_reg_smooth_kernel=int(cfg.get("indicator_reg_smooth_kernel", 3)),
        hard_routing_at_eval=bool(cfg.get("hard_routing_at_eval", True)),
        straight_through_routing=bool(cfg.get("straight_through_routing", False)),
        routing_top_k=int(cfg.get("routing_top_k", 1)),
        routing_load_balance_weight=float(cfg.get("routing_load_balance_weight", 0.0)),
        routing_entropy_weight=float(cfg.get("routing_entropy_weight", 0.0)),
        encoder_feature_dim=int(cfg.get("encoder_feature_dim", 256)),
        encoder_first_omega_0=float(cfg.get("encoder_first_omega_0", 30.0)),
        encoder_hidden_omega_0=float(cfg.get("encoder_hidden_omega_0", 30.0)),
        policy_hidden_dim=int(cfg.get("policy_hidden_dim", 128)),
        policy_num_layers=int(cfg.get("policy_num_layers", 3)),
        policy_first_omega_0=float(cfg.get("policy_first_omega_0", 30.0)),
        policy_hidden_omega_0=float(cfg.get("policy_hidden_omega_0", 30.0)),
        expert_hidden_dim=int(cfg.get("expert_hidden_dim", 256)),
        expert_num_layers=int(cfg.get("expert_num_layers", 3)),
        expert_first_omega_0=float(cfg.get("expert_first_omega_0", 30.0)),
        expert_hidden_omega_0=float(cfg.get("expert_hidden_omega_0", 30.0)),
        input_mean=input_mean,
        input_std=input_std,
    )
