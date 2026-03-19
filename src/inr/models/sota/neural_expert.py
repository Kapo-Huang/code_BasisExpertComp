import math
import logging
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class Sine(nn.Module):
    def __init__(self, freq: float = 30.0, trainable: bool = False):
        super().__init__()
        if trainable:
            self.freq = nn.Parameter(torch.tensor(float(freq), dtype=torch.float32))
        else:
            self.register_buffer("freq", torch.tensor(float(freq), dtype=torch.float32))

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return torch.sin(self.freq * input)


class FINER(nn.Module):
    def __init__(self, freq: float = 30.0, trainable: bool = False):
        super().__init__()
        if trainable:
            self.freq = nn.Parameter(torch.tensor(float(freq), dtype=torch.float32))
        else:
            self.register_buffer("freq", torch.tensor(float(freq), dtype=torch.float32))

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        scale = input.detach().abs() + 1.0
        return torch.sin(self.freq * scale * input)


class DummyModule(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


class tSoftMax(nn.Module):
    def __init__(self, temperature: float, dim: int = -1, trainable: bool = False):
        super().__init__()
        if trainable:
            self.temperature = nn.Parameter(torch.tensor(float(temperature), dtype=torch.float32))
        else:
            self.register_buffer("temperature", torch.tensor(float(temperature), dtype=torch.float32))
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.softmax(x / self.temperature.clamp_min(1e-6), dim=self.dim)


def _init_linear_weight(linear: nn.Linear, init_type: str, nonlinearity: str, *, is_first: bool, freq: float) -> None:
    init_type = str(init_type).strip().lower()
    nonlinearity = str(nonlinearity).strip().lower()
    with torch.no_grad():
        if init_type in {"siren", "finer"}:
            in_features = linear.in_features
            if is_first:
                bound = 1.0 / max(1, in_features)
            else:
                bound = math.sqrt(6.0 / max(1, in_features)) / float(freq)
            linear.weight.uniform_(-bound, bound)
            if linear.bias is not None:
                linear.bias.zero_()
            return

        if init_type == "normal":
            nn.init.kaiming_normal_(linear.weight, a=0.0, nonlinearity="relu", mode="fan_in")
        elif init_type == "kaiminguniform":
            nn.init.kaiming_uniform_(linear.weight, a=0.0, nonlinearity="relu")
        else:
            raise ValueError(f"Unsupported init_type: {init_type}")

        if linear.bias is not None:
            linear.bias.zero_()


def _make_activation(name: str, freq: float, trainable_freqs: bool = False) -> nn.Module:
    name = str(name).strip().lower()
    if name == "sine":
        return Sine(freq=freq, trainable=trainable_freqs)
    if name == "finer":
        return FINER(freq=freq, trainable=trainable_freqs)
    if name == "relu":
        return nn.ReLU(inplace=True)
    if name == "softplus":
        return nn.Softplus(beta=100)
    if name == "tanh":
        return nn.Tanh()
    if name == "sigmoid":
        return nn.Sigmoid()
    raise ValueError(f"Unsupported nonlinearity: {name}")


class FullyConnectedNN(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        num_hidden_layers: int,
        hidden_features: int,
        outermost_linear: bool = False,
        nonlinearity: str = "sine",
        init_type: str = "siren",
        freq: float = 30.0,
        trainable_freqs: bool = False,
    ):
        super().__init__()
        self.outermost_linear = bool(outermost_linear)
        self.nonlinearity = str(nonlinearity).strip().lower()
        self.init_type = str(init_type).strip().lower()
        self.freq = float(freq)

        hidden_count = int(num_hidden_layers) + 1
        if hidden_count <= 0:
            raise ValueError("num_hidden_layers must be >= 0")

        layers = []
        first = nn.Linear(int(in_features), int(hidden_features))
        _init_linear_weight(first, self.init_type, self.nonlinearity, is_first=True, freq=self.freq)
        layers.append(first)

        for _ in range(int(num_hidden_layers)):
            linear = nn.Linear(int(hidden_features), int(hidden_features))
            _init_linear_weight(linear, self.init_type, self.nonlinearity, is_first=False, freq=self.freq)
            layers.append(linear)

        self.hidden_layers = nn.ModuleList(layers)
        self.activations = nn.ModuleList(
            [_make_activation(self.nonlinearity, self.freq, trainable_freqs) for _ in range(hidden_count)]
        )
        self.final = nn.Linear(int(hidden_features), int(out_features))
        _init_linear_weight(self.final, self.init_type, self.nonlinearity, is_first=False, freq=self.freq)
        if not self.outermost_linear:
            self.final_activation = _make_activation(self.nonlinearity, self.freq, trainable_freqs)
        else:
            self.final_activation = None

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        x = input
        for linear, activation in zip(self.hidden_layers, self.activations):
            x = activation(linear(x))
        x = self.final(x)
        if self.final_activation is not None:
            x = self.final_activation(x)
        return x


class ParallelFullyConnectedNN(nn.Module):
    def __init__(
        self,
        k: int,
        in_features: int,
        out_features: int,
        num_hidden_layers: int,
        hidden_features: int,
        outermost_linear: bool = False,
        nonlinearity: str = "sine",
        init_type: str = "siren",
        freq: float = 30.0,
        trainable_freqs: bool = False,
    ):
        super().__init__()
        self.k = int(k)
        self.experts = nn.ModuleList(
            [
                FullyConnectedNN(
                    in_features=in_features,
                    out_features=out_features,
                    num_hidden_layers=num_hidden_layers,
                    hidden_features=hidden_features,
                    outermost_linear=outermost_linear,
                    nonlinearity=nonlinearity,
                    init_type=init_type,
                    freq=freq,
                    trainable_freqs=trainable_freqs,
                )
                for _ in range(self.k)
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        outputs = [expert(x) for expert in self.experts]
        return torch.stack(outputs, dim=1)


class InputEncoder(nn.Module):
    def __init__(
        self,
        in_dim: int,
        input_encoding: str,
        hidden_dim: int,
        *,
        trainable_freqs: bool = False,
    ):
        super().__init__()
        self.input_encoding = str(input_encoding)
        self.in_dim = int(in_dim)
        hidden_features = int(hidden_dim)
        self.first_layer_dim = self.in_dim
        self.encoder = None

        if "FF" in self.input_encoding:
            if hidden_features % 2 != 0:
                raise ValueError("hidden_dim must be even when using FF encoding")
            self.bvals_size = hidden_features // 2
            bvals = torch.randn(size=[self.bvals_size, self.in_dim], dtype=torch.float32)
            self.register_buffer("bvals", bvals)
            self.first_layer_dim = hidden_features + self.in_dim
        elif "PE" in self.input_encoding:
            bvals = 2 ** torch.linspace(0.0, 5.0, 6)
            self.register_buffer("bvals", bvals)
            self.first_layer_dim = self.in_dim * 6 * 2 + self.in_dim

        if "learned" in self.input_encoding:
            parsed = self.input_encoding.split("_")
            if len(parsed) < 5:
                raise ValueError(
                    f"learned input encoding must be like learned_128_2_sine_siren, got: {self.input_encoding}"
                )
            enc_hidden_features = int(parsed[1])
            enc_n_layers = int(parsed[2])
            enc_nl = parsed[3]
            enc_init = parsed[4]
            self.encoder = FullyConnectedNN(
                in_features=self.first_layer_dim,
                out_features=enc_hidden_features,
                num_hidden_layers=enc_n_layers,
                hidden_features=enc_hidden_features,
                outermost_linear=False,
                nonlinearity=enc_nl,
                init_type=enc_init,
                freq=30.0,
                trainable_freqs=trainable_freqs,
            )
            self.first_layer_dim = (
                enc_hidden_features + self.first_layer_dim if "cat" in self.input_encoding else enc_hidden_features
            )

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        if "FF" in self.input_encoding:
            x = (2.0 * np.pi * coords) @ self.bvals.t()
            x = torch.cat([torch.sin(x), torch.cos(x)], dim=-1) / math.sqrt(self.bvals_size)
            x = torch.cat([coords, x], dim=-1)
        elif "PE" in self.input_encoding:
            x = coords[..., None] * self.bvals
            x = x.reshape(*x.shape[:-2], -1)
            x = torch.sin(torch.cat([x, x + np.pi / 2.0], dim=-1))
            x = torch.cat([coords, x], dim=-1)
        else:
            x = coords

        if self.encoder is not None:
            encoded = self.encoder(x)
            if "cat" in self.input_encoding:
                x = torch.cat([coords, encoded], dim=-1)
            else:
                x = encoded
        return x


class ManagerConditioner(nn.Module):
    def __init__(self, manager_conditioning: str):
        super().__init__()
        self.manager_conditioning = str(manager_conditioning).strip().lower()

    def forward(self, expert_encoded: torch.Tensor, manager_input: torch.Tensor) -> torch.Tensor:
        mode = self.manager_conditioning
        if mode == "none":
            return manager_input
        if mode == "max":
            point_rep = expert_encoded.max(dim=1, keepdim=True)[0].expand(-1, manager_input.shape[1], -1)
        elif mode == "mean":
            point_rep = expert_encoded.mean(dim=1, keepdim=True).expand(-1, manager_input.shape[1], -1)
        elif mode == "cat":
            point_rep = expert_encoded
        else:
            raise ValueError(f"Unsupported manager_conditioning: {self.manager_conditioning}")
        return torch.cat([manager_input, point_rep], dim=-1)


class Manager(nn.Module):
    def __init__(
        self,
        *,
        in_dim: int,
        n_experts: int,
        hidden_dim: int,
        num_hidden_layers: int,
        nonlinearity: str,
        init_type: str,
        input_encoding: str,
        q_activation: str,
        clamp_q: float,
        temperature: float,
        temp_trainable: bool,
        manager_type: str = "standard",
    ):
        super().__init__()
        self.n_experts = int(n_experts)
        self.manager_type = str(manager_type).strip().lower()
        self.clamp_q = float(clamp_q)

        if self.manager_type == "none":
            self.manager_net = None
        elif self.manager_type == "standard":
            self.manager_net = FullyConnectedNN(
                in_features=int(in_dim),
                out_features=self.n_experts,
                num_hidden_layers=int(num_hidden_layers),
                hidden_features=int(hidden_dim),
                outermost_linear=True,
                nonlinearity=nonlinearity,
                init_type=init_type,
                freq=30.0,
            )
        else:
            raise ValueError(f"Unsupported manager_type: {manager_type}")

        q_activation = str(q_activation).strip().lower()
        if q_activation == "softmax":
            self.q_activation = tSoftMax(float(temperature), dim=-1, trainable=bool(temp_trainable))
        elif q_activation == "sigmoid":
            self.q_activation = nn.Sigmoid()
        elif q_activation == "none":
            self.q_activation = DummyModule()
        else:
            raise ValueError(f"Unsupported manager_q_activation: {q_activation}")
        self.input_encoding = str(input_encoding)

    def forward(self, points: torch.Tensor):
        if self.manager_net is None:
            raw_q = torch.zeros(points.shape[0], self.n_experts, device=points.device, dtype=points.dtype)
        else:
            raw_q = self.manager_net(points)
        q = self.q_activation(raw_q)
        if self.clamp_q > 0.0:
            q = torch.clamp(q, min=self.clamp_q)
        selected_expert_idx = torch.argmax(q, dim=-1)
        return q, selected_expert_idx, raw_q


class RoutingStack(nn.Module):
    def __init__(
        self,
        *,
        in_dim: int,
        n_experts: int,
        manager_hidden_dim: int,
        manager_n_hidden_layers: int,
        manager_input_encoding: str,
        manager_nl: str,
        manager_init: str,
        manager_softmax_temperature: float,
        manager_softmax_temp_trainable: bool,
        manager_q_activation: str,
        manager_clamp_q: float,
        manager_conditioning: str,
        expert_feature_dim: int,
        shared_input_encoder: Optional[InputEncoder] = None,
        decoder_hidden_dim: int = 128,
        manager_type: str = "standard",
    ):
        super().__init__()
        if shared_input_encoder is None:
            self.input_encoder = InputEncoder(
                in_dim=in_dim,
                input_encoding=manager_input_encoding,
                hidden_dim=decoder_hidden_dim,
            )
        else:
            self.input_encoder = shared_input_encoder

        self.conditioner = ManagerConditioner(manager_conditioning=manager_conditioning)
        conditioned_dim = int(self.input_encoder.first_layer_dim)
        if str(manager_conditioning).strip().lower() != "none":
            conditioned_dim += int(expert_feature_dim)
        self.manager = Manager(
            in_dim=conditioned_dim,
            n_experts=n_experts,
            hidden_dim=manager_hidden_dim,
            num_hidden_layers=manager_n_hidden_layers,
            nonlinearity=manager_nl,
            init_type=manager_init,
            input_encoding=manager_input_encoding,
            q_activation=manager_q_activation,
            clamp_q=manager_clamp_q,
            temperature=manager_softmax_temperature,
            temp_trainable=manager_softmax_temp_trainable,
            manager_type=manager_type,
        )

    def forward(self, coords: torch.Tensor, expert_encoded: torch.Tensor):
        manager_input = self.input_encoder(coords)
        manager_input = self.conditioner(expert_encoded, manager_input)
        batch_size, num_points, feat_dim = manager_input.shape
        flat = manager_input.reshape(batch_size * num_points, feat_dim)
        probs, selected_expert_idx, logits = self.manager(flat)
        probs = probs.view(batch_size, num_points, -1)
        selected_expert_idx = selected_expert_idx.view(batch_size, num_points)
        logits = logits.view(batch_size, num_points, -1)
        return probs, selected_expert_idx, logits


class NeuralExpertINR(nn.Module):
    def __init__(
        self,
        *,
        in_features: int = 4,
        out_features: int = 1,
        num_experts: int = 7,
        top_k: int = 1,
        decoder_hidden_dim: int = 128,
        decoder_n_hidden_layers: int = 2,
        decoder_input_encoding: str = "learned_128_2_sine_siren_none",
        decoder_nl: str = "sine",
        decoder_init_type: str = "siren",
        decoder_freqs: float = 30.0,
        decoder_trainable_freqs: bool = False,
        manager_hidden_dim: int = 128,
        manager_n_hidden_layers: int = 2,
        manager_input_encoding: str = "learned_128_2_sine_siren_none",
        manager_nl: str = "sine",
        manager_init: str = "siren",
        manager_softmax_temperature: float = 1.0,
        manager_softmax_temp_trainable: bool = False,
        manager_q_activation: str = "softmax",
        manager_clamp_q: float = 0.0,
        manager_conditioning: str = "cat",
        shared_encoder: bool = False,
        manager_type: str = "standard",
    ):
        super().__init__()
        if int(top_k) < 1:
            raise ValueError("top_k must be >= 1")

        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.num_experts = int(num_experts)
        self.top_k = min(int(top_k), self.num_experts)
        self.shared_encoder = bool(shared_encoder)

        self.decoder_input_encoding_module = InputEncoder(
            in_dim=self.in_features,
            input_encoding=decoder_input_encoding,
            hidden_dim=decoder_hidden_dim,
            trainable_freqs=bool(decoder_trainable_freqs),
        )
        decoder_first_layer_dim = int(self.decoder_input_encoding_module.first_layer_dim)
        self.decoder = ParallelFullyConnectedNN(
            k=self.num_experts,
            in_features=decoder_first_layer_dim,
            out_features=self.out_features,
            num_hidden_layers=int(decoder_n_hidden_layers),
            hidden_features=int(decoder_hidden_dim),
            outermost_linear=True,
            nonlinearity=decoder_nl,
            init_type=decoder_init_type,
            freq=float(decoder_freqs),
            trainable_freqs=bool(decoder_trainable_freqs),
        )

        shared_input_encoder = self.decoder_input_encoding_module if self.shared_encoder else None
        self.gating = RoutingStack(
            in_dim=self.in_features,
            n_experts=self.num_experts,
            manager_hidden_dim=int(manager_hidden_dim),
            manager_n_hidden_layers=int(manager_n_hidden_layers),
            manager_input_encoding=manager_input_encoding,
            manager_nl=manager_nl,
            manager_init=manager_init,
            manager_softmax_temperature=float(manager_softmax_temperature),
            manager_softmax_temp_trainable=bool(manager_softmax_temp_trainable),
            manager_q_activation=manager_q_activation,
            manager_clamp_q=float(manager_clamp_q),
            manager_conditioning=manager_conditioning,
            expert_feature_dim=decoder_first_layer_dim,
            shared_input_encoder=shared_input_encoder,
            decoder_hidden_dim=int(manager_hidden_dim),
            manager_type=manager_type,
        )

        n_params = sum(p.numel() for p in self.parameters())
        logger.info(
            "NeuralExpertINR init: experts=%s top_k=%s params=%s",
            self.num_experts,
            self.top_k,
            f"{n_params:,}",
        )

    def _prepare_coords(self, coords: torch.Tensor):
        if coords.dim() == 2:
            return coords.unsqueeze(0), True
        if coords.dim() == 3:
            return coords, False
        raise ValueError(f"coords must have shape [N, D] or [B, N, D], got {tuple(coords.shape)}")

    def _normalize_probs(self, probs: torch.Tensor) -> torch.Tensor:
        probs = probs.clamp_min(0.0)
        denom = probs.sum(dim=-1, keepdim=True).clamp_min(1e-9)
        return probs / denom

    def _apply_topk(self, probs: torch.Tensor, hard_topk: bool) -> torch.Tensor:
        normalized = self._normalize_probs(probs)
        if not hard_topk or self.top_k >= self.num_experts:
            return normalized
        _, indices = torch.topk(normalized, k=self.top_k, dim=-1)
        mask = torch.zeros_like(normalized)
        mask.scatter_(-1, indices, 1.0)
        return self._normalize_probs(normalized * mask)

    def _forward_impl(self, coords: torch.Tensor, *, hard_topk: bool):
        coords_batched, squeeze_output = self._prepare_coords(coords)
        encoded_experts = self.decoder_input_encoding_module(coords_batched)
        preds_all = self.decoder(encoded_experts)
        probs_raw, selected_expert_idx, logits = self.gating(coords_batched, encoded_experts)
        weights = self._apply_topk(probs_raw, hard_topk=hard_topk)

        preds_all_perm = preds_all.permute(0, 2, 1, 3)
        mixed_pred = torch.sum(preds_all_perm * weights.unsqueeze(-1), dim=-2)
        gather_index = selected_expert_idx.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 1, self.out_features)
        selected_pred = torch.gather(preds_all_perm, dim=2, index=gather_index).squeeze(2)

        aux = {
            "probs": weights.squeeze(0) if squeeze_output else weights,
            "raw_probs": self._normalize_probs(probs_raw).squeeze(0) if squeeze_output else self._normalize_probs(probs_raw),
            "logits": logits.squeeze(0) if squeeze_output else logits,
            "preds_all": preds_all_perm.squeeze(0) if squeeze_output else preds_all_perm,
            "selected_expert_idx": selected_expert_idx.squeeze(0) if squeeze_output else selected_expert_idx,
            "selected_pred": selected_pred.squeeze(0) if squeeze_output else selected_pred,
        }
        output = mixed_pred.squeeze(0) if squeeze_output else mixed_pred
        return output, aux

    def forward(
        self,
        coords: torch.Tensor,
        *,
        hard_topk: bool = True,
        return_aux: bool = False,
    ):
        output, aux = self._forward_impl(coords, hard_topk=hard_topk)
        if return_aux:
            return output, aux
        return output

    def pretrain_forward(self, coords: torch.Tensor) -> torch.Tensor:
        coords_batched, squeeze_output = self._prepare_coords(coords)
        encoded_experts = self.decoder_input_encoding_module(coords_batched)
        _, _, logits = self.gating(coords_batched, encoded_experts)
        return logits.squeeze(0) if squeeze_output else logits

    def pretrain_parameters(self):
        return list(self.gating.parameters())


def build_neural_expert_from_config(cfg) -> NeuralExpertINR:
    return NeuralExpertINR(
        in_features=int(cfg.get("in_features", 4)),
        out_features=int(cfg.get("out_features", 1)),
        num_experts=int(cfg.get("num_experts", 7)),
        top_k=int(cfg.get("top_k", 1)),
        decoder_hidden_dim=int(cfg.get("decoder_hidden_dim", 128)),
        decoder_n_hidden_layers=int(cfg.get("decoder_n_hidden_layers", 2)),
        decoder_input_encoding=str(cfg.get("decoder_input_encoding", "learned_128_2_sine_siren_none")),
        decoder_nl=str(cfg.get("decoder_nl", "sine")),
        decoder_init_type=str(cfg.get("decoder_init_type", "siren")),
        decoder_freqs=float(cfg.get("decoder_freqs", 30.0)),
        decoder_trainable_freqs=bool(cfg.get("decoder_trainable_freqs", False)),
        manager_hidden_dim=int(cfg.get("manager_hidden_dim", 128)),
        manager_n_hidden_layers=int(cfg.get("manager_n_hidden_layers", 2)),
        manager_input_encoding=str(cfg.get("manager_input_encoding", "learned_128_2_sine_siren_none")),
        manager_nl=str(cfg.get("manager_nl", "sine")),
        manager_init=str(cfg.get("manager_init", "siren")),
        manager_softmax_temperature=float(cfg.get("manager_softmax_temperature", 1.0)),
        manager_softmax_temp_trainable=bool(cfg.get("manager_softmax_temp_trainable", False)),
        manager_q_activation=str(cfg.get("manager_q_activation", "softmax")),
        manager_clamp_q=float(cfg.get("manager_clamp_q", 0.0)),
        manager_conditioning=str(cfg.get("manager_conditioning", "cat")),
        shared_encoder=bool(cfg.get("shared_encoder", False)),
        manager_type=str(cfg.get("manager_type", "standard")),
    )
