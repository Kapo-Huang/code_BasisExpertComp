import math
from typing import Dict, Optional

import torch
import torch.nn as nn


class Sine(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return torch.sin(30.0 * input)


class SineLayer(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        is_first: bool = False,
        omega_0: float = 30.0,
        use_bn: bool = False,
    ):
        super().__init__()
        self.omega_0 = omega_0
        self.is_first = is_first
        self.use_bn = use_bn
        self.in_features = in_features
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        if self.use_bn:
            self.bn = nn.BatchNorm1d(out_features)
        self._init_weights()

    def _init_weights(self) -> None:
        with torch.no_grad():
            if self.is_first:
                bound = 1.0 / self.in_features
            else:
                bound = math.sqrt(6.0 / self.in_features) / self.omega_0
            self.linear.weight.uniform_(-bound, bound)
            # if self.linear.bias is not None:
            #     self.linear.bias.zero_()

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        h = self.omega_0 * self.linear(input)
        if self.use_bn:
            h = self.bn(h)
        return torch.sin(h)


class ResBlock(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        nonlinearity: str = "sine",
        use_bn: bool = False,
        omega_0: float = 30.0,
    ):
        super().__init__()
        nls_and_inits = {
            "sine": Sine(),
            "relu": nn.ReLU(inplace=True),
            "sigmoid": nn.Sigmoid(),
            "tanh": nn.Tanh(),
            "selu": nn.SELU(inplace=True),
            "softplus": nn.Softplus(),
            "elu": nn.ELU(inplace=True),
        }
        self.nl = nls_and_inits[nonlinearity]

        self.net = nn.Sequential(
            SineLayer(in_features, out_features, omega_0=omega_0),
            SineLayer(out_features, out_features, use_bn=use_bn, omega_0=omega_0),
        )
        self.flag = in_features != out_features
        if self.flag:
            self.transform = SineLayer(in_features, out_features, use_bn=use_bn, omega_0=omega_0)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        outputs = self.net(features)
        if self.flag:
            features = self.transform(features)
        return 0.5 * (outputs + features)


class Body(nn.Module):
    def __init__(self, in_features: int, init_features: int = 64, num_res: int = 5, omega_0: float = 30.0):
        super().__init__()
        self.num_res = num_res
        layers = [
            SineLayer(in_features, init_features, omega_0=omega_0),
            SineLayer(init_features, 2 * init_features, omega_0=omega_0),
            SineLayer(2 * init_features, 4 * init_features, omega_0=omega_0),
        ]
        for _ in range(self.num_res):
            layers.append(ResBlock(4 * init_features, 4 * init_features, omega_0=omega_0))
        self.net = nn.Sequential(*layers)

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        return self.net(coords)


class Head(nn.Module):
    def __init__(
        self,
        feature_dims: int,
        outermost_linear: bool,
        output_features: int = 1,
        omega_0: float = 30.0,
    ):
        super().__init__()
        self.num_layers = 3
        self.synthesis_net = nn.ModuleList(
            [
                ResBlock(feature_dims, feature_dims, omega_0=omega_0),
                SineLayer(feature_dims, feature_dims // 2, omega_0=omega_0),
                SineLayer(feature_dims // 2, feature_dims // 4, omega_0=omega_0),
            ]
        )
        self.modulator_net = nn.ModuleList(
            [
                ResBlock(feature_dims, feature_dims, omega_0=omega_0),
                SineLayer(feature_dims, feature_dims // 2, omega_0=omega_0),
                SineLayer(feature_dims // 2, feature_dims // 4, omega_0=omega_0),
            ]
        )
        if outermost_linear:
            self.final_layer = nn.Sequential(
                nn.Linear(feature_dims // 4, output_features),
                nn.Tanh(),
            )
        else:
            self.final_layer = SineLayer(feature_dims // 4, output_features, omega_0=omega_0)

    def forward(self, feature: torch.Tensor, latent: torch.Tensor) -> torch.Tensor:
        for i in range(self.num_layers):
            feature = self.synthesis_net[i](feature * latent)
            latent = self.modulator_net[i](latent)
        return self.final_layer(feature * latent)


class STSR_INR(nn.Module):
    def __init__(
        self,
        in_coords_dims: int = 4,
        out_features: int = 1,
        init_features: int = 64,
        num_res: int = 5,
        outermost_linear: bool = True,
        embedding_dims: int = 256,
        omega_0: float = 30.0,
        use_global_latent: bool = True,
    ):
        super().__init__()
        self.in_coords_dims = in_coords_dims
        self.out_features = out_features
        self.Modulated_Net = Body(
            in_features=embedding_dims,
            init_features=init_features,
            num_res=num_res,
            omega_0=omega_0,
        )
        self.Synthesis_Net = Body(
            in_features=in_coords_dims,
            init_features=init_features,
            num_res=num_res,
            omega_0=omega_0,
        )
        self.layer_num = len(self.Synthesis_Net.net)
        self.final_layers = nn.ModuleList(
            [Head(4 * init_features, outermost_linear=outermost_linear) for _ in range(out_features)]
        )
        self.embedding_dims = embedding_dims
        self.use_global_latent = use_global_latent
        if use_global_latent:
            self.global_latent = nn.Parameter(torch.zeros(1, embedding_dims))

    def _resolve_latent(self, coords: torch.Tensor, latent: Optional[torch.Tensor]) -> torch.Tensor:
        if latent is not None:
            return latent
        if not self.use_global_latent:
            raise ValueError("latent must be provided when use_global_latent is False.")
        return self.global_latent.expand(coords.shape[0], -1)

    def forward(self, coords: torch.Tensor, latent: Optional[torch.Tensor] = None) -> torch.Tensor:
        latent = self._resolve_latent(coords, latent)
        latent_code = self.Modulated_Net.net[0](latent)
        coords_feat = self.Synthesis_Net.net[0](coords)
        for i in range(1, self.layer_num):
            coords_feat = self.Synthesis_Net.net[i](coords_feat * latent_code)
            latent_code = self.Modulated_Net.net[i](latent_code)
        if self.out_features == 1:
            output = self.final_layers[0](coords_feat, latent_code)
        else:
            output = torch.cat(
                [self.final_layers[j](coords_feat, latent_code).reshape(-1, 1) for j in range(self.out_features)],
                dim=-1,
            )
        return output


class STSRINRMultiView(nn.Module):
    def __init__(
        self,
        in_coords_dims: int,
        view_specs: Dict[str, int],
        init_features: int = 64,
        num_res: int = 5,
        outermost_linear: bool = True,
        embedding_dims: int = 256,
        omega_0: float = 30.0,
        use_global_latent: bool = True,
    ):
        super().__init__()
        if not view_specs:
            raise ValueError("view_specs must be a non-empty dict")
        self.view_names = list(view_specs.keys())
        self.view_dims = dict(view_specs)
        self.num_views = len(self.view_names)
        self.num_experts = 1

        self.Modulated_Net = Body(
            in_features=embedding_dims,
            init_features=init_features,
            num_res=num_res,
            omega_0=omega_0,
        )
        self.Synthesis_Net = Body(
            in_features=in_coords_dims,
            init_features=init_features,
            num_res=num_res,
            omega_0=omega_0,
        )
        self.layer_num = len(self.Synthesis_Net.net)
        self.heads = nn.ModuleDict(
            {
                name: Head(4 * init_features, outermost_linear=outermost_linear, output_features=out_dim)
                for name, out_dim in self.view_dims.items()
            }
        )
        self.embedding_dims = embedding_dims
        self.use_global_latent = use_global_latent
        if use_global_latent:
            self.global_latent = nn.Parameter(torch.zeros(1, embedding_dims))

    def _resolve_latent(self, coords: torch.Tensor, latent: Optional[torch.Tensor]) -> torch.Tensor:
        if latent is not None:
            return latent
        if not self.use_global_latent:
            raise ValueError("latent must be provided when use_global_latent is False.")
        return self.global_latent.expand(coords.shape[0], -1)

    def forward(
        self,
        coords: torch.Tensor,
        request: Optional[str] = None,
        *,
        return_aux: bool = False,
        latent=None,
    ):
        if request is not None and request not in self.view_dims:
            raise KeyError(f"Unknown view '{request}'. Available: {list(self.view_dims.keys())}")

        if isinstance(latent, dict):
            target_names = self.view_names if request is None else [request]
            preds = {}
            feat_list = []
            for name in target_names:
                latent_one = self._resolve_latent(coords, latent.get(name))
                latent_code = self.Modulated_Net.net[0](latent_one)
                coords_feat = self.Synthesis_Net.net[0](coords)
                for i in range(1, self.layer_num):
                    coords_feat = self.Synthesis_Net.net[i](coords_feat * latent_code)
                    latent_code = self.Modulated_Net.net[i](latent_code)
                preds[name] = self.heads[name](coords_feat, latent_code)
                feat_list.append(coords_feat)

            output = preds if request is None else preds[request]
            if return_aux:
                bsz = coords.shape[0]
                probs = torch.zeros(bsz, self.num_views, self.num_experts, device=coords.device)
                masks = torch.zeros_like(probs)
                expert_feats = torch.stack(feat_list, dim=1)
                aux = {
                    "probs": probs,
                    "masks": masks,
                    "expert_feats": expert_feats,
                }
                return output, aux
            return output

        latent = self._resolve_latent(coords, latent)
        latent_code = self.Modulated_Net.net[0](latent)
        coords_feat = self.Synthesis_Net.net[0](coords)
        for i in range(1, self.layer_num):
            coords_feat = self.Synthesis_Net.net[i](coords_feat * latent_code)
            latent_code = self.Modulated_Net.net[i](latent_code)

        preds = {name: self.heads[name](coords_feat, latent_code) for name in self.view_names}
        output = preds if request is None else preds[request]

        if return_aux:
            bsz = coords.shape[0]
            probs = torch.zeros(bsz, self.num_views, self.num_experts, device=coords.device)
            masks = torch.zeros_like(probs)
            expert_feats = coords_feat.unsqueeze(1)
            aux = {
                "probs": probs,
                "masks": masks,
                "expert_feats": expert_feats,
            }
            return output, aux
        return output


class VarVADEmbedding(nn.Module):
    def __init__(self, embedding_dims: int = 256, embedding_nums: int = 90):
        super().__init__()
        self.weight_mu = nn.Parameter(torch.Tensor(embedding_nums, embedding_dims))
        self.weight_logvar = nn.Parameter(
            torch.ones_like(self.weight_mu) * 0.001, requires_grad=False
        )
        self.dim = embedding_dims
        self.embedding_nums = embedding_nums
        self.reset_parameters()

    def reset_parameters(self) -> None:
        mu_init_std = 1.0 / math.sqrt(self.dim)
        torch.nn.init.normal_(self.weight_mu.data, 0.0, mu_init_std)

    def kl_loss(self) -> torch.Tensor:
        kl_loss = (
            0.5
            * torch.sum(
                torch.exp(self.weight_logvar) + self.weight_mu**2 - 1.0 - self.weight_logvar
            )
            / self.embedding_nums
        )
        return kl_loss

    def forward(self, query_index: torch.Tensor, train: bool = True) -> torch.Tensor:
        noise = torch.randn_like(self.weight_logvar[query_index]) * torch.exp(
            0.5 * self.weight_logvar[query_index]
        )
        if train:
            return self.weight_mu[query_index] + noise
        return self.weight_mu[query_index]


def build_stsr_inr_from_config(cfg) -> STSR_INR:
    init_features = int(cfg.get("init_features", cfg.get("init", 64)))
    return STSR_INR(
        in_coords_dims=int(cfg.get("in_features", 4)),
        out_features=int(cfg.get("out_features", 1)),
        init_features=init_features,
        num_res=int(cfg.get("num_res", 5)),
        outermost_linear=bool(cfg.get("outermost_linear", True)),
        embedding_dims=int(cfg.get("embedding_dims", 256)),
        omega_0=float(cfg.get("omega_0", 30.0)),
        use_global_latent=bool(cfg.get("use_global_latent", True)),
    )


def build_stsr_inr_multiview_from_config(cfg, view_specs: Dict[str, int]) -> STSRINRMultiView:
    init_features = int(cfg.get("init_features", cfg.get("init", 64)))
    return STSRINRMultiView(
        in_coords_dims=int(cfg.get("in_features", 4)),
        view_specs=view_specs,
        init_features=init_features,
        num_res=int(cfg.get("num_res", 5)),
        outermost_linear=bool(cfg.get("outermost_linear", True)),
        embedding_dims=int(cfg.get("embedding_dims", 256)),
        omega_0=float(cfg.get("omega_0", 30.0)),
        use_global_latent=bool(cfg.get("use_global_latent", True)),
    )
