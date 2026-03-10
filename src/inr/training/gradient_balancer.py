from typing import Dict, List, Optional, Sequence, Tuple

import torch


def _get_trainable_params(model: torch.nn.Module) -> List[torch.nn.Parameter]:
    return [param for param in model.parameters() if param.requires_grad]


def _flatten_grads(
    grads: Sequence[Optional[torch.Tensor]],
    params: Sequence[torch.nn.Parameter],
) -> torch.Tensor:
    flat_parts = []
    for grad, param in zip(grads, params):
        if grad is None:
            grad = torch.zeros_like(param)
        flat_parts.append(grad.reshape(-1))
    if not flat_parts:
        return torch.zeros(0)
    return torch.cat(flat_parts, dim=0)


def get_shared_named_params(model: torch.nn.Module) -> List[Tuple[str, torch.nn.Parameter]]:
    return [
        (name, param)
        for name, param in model.named_parameters()
        if param.requires_grad and not name.startswith("heads.")
    ]


def flatten_named_grads(
    grads: Sequence[Optional[torch.Tensor]],
    named_params: Sequence[Tuple[str, torch.nn.Parameter]],
) -> torch.Tensor:
    flat_parts = []
    for grad, (_, param) in zip(grads, named_params):
        grad_tensor = grad if grad is not None else torch.zeros_like(param)
        flat_parts.append(grad_tensor.reshape(-1))
    if not flat_parts:
        if named_params:
            return torch.zeros(0, device=named_params[0][1].device)
        return torch.zeros(0)
    return torch.cat(flat_parts, dim=0)


class GradNormBalancer:
    def __init__(self, task_names, cfg, device):
        self.task_names = list(task_names)
        self.task_weights = torch.nn.Parameter(torch.ones(len(self.task_names), device=device))
        self.initial_losses = None
        self.alpha = float(cfg.gradnorm_alpha)
        self.optimizer = torch.optim.Adam([self.task_weights], lr=float(cfg.gradnorm_lr))

    def get_weight_dict(self) -> Dict[str, float]:
        weights = self.task_weights.detach().cpu().tolist()
        return {task_name: float(weight) for task_name, weight in zip(self.task_names, weights)}

    def build_weighted_loss(
        self,
        task_losses: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if list(task_losses.keys()) != self.task_names:
            raise ValueError(
                f"GradNorm task order mismatch. Expected {self.task_names}, got {list(task_losses.keys())}"
            )
        missing_tasks = [task_name for task_name in self.task_names if task_name not in task_losses]
        if missing_tasks:
            raise ValueError(f"Missing task losses for GradNorm: {missing_tasks}")

        ordered_losses = torch.stack([task_losses[task_name] for task_name in self.task_names], dim=0)
        weighted_loss = torch.sum(self.task_weights * ordered_losses)
        return weighted_loss, ordered_losses

    def build_weighted_loss_only(self, task_losses: Dict[str, torch.Tensor]) -> torch.Tensor:
        weighted_loss, _ = self.build_weighted_loss(task_losses)
        return weighted_loss

    def update(
        self,
        task_losses: Dict[str, torch.Tensor],
        model: torch.nn.Module,
    ) -> Dict[str, object]:
        _, ordered_losses = self.build_weighted_loss(task_losses)
        if self.initial_losses is None:
            self.initial_losses = ordered_losses.detach().clamp_min(1e-12)

        shared_named_params = get_shared_named_params(model)
        if not shared_named_params:
            raise ValueError("No shared trainable parameters found for GradNorm.")

        shared_params = [param for _, param in shared_named_params]
        grad_norm_list = []
        for index, loss_tensor in enumerate(ordered_losses):
            weighted_task_loss_i = self.task_weights[index] * loss_tensor
            shared_grads = torch.autograd.grad(
                weighted_task_loss_i,
                shared_params,
                allow_unused=True,
                create_graph=True,
                retain_graph=True,
            )
            flat_shared_grad = flatten_named_grads(shared_grads, shared_named_params)
            grad_norm_list.append(torch.norm(flat_shared_grad, p=2))
        grad_norms = torch.stack(grad_norm_list, dim=0)

        loss_ratios = ordered_losses.detach() / self.initial_losses
        relative_rates = loss_ratios / loss_ratios.mean().clamp_min(1e-12)
        grad_norm_avg = grad_norms.detach().mean()
        target_grad_norms = grad_norm_avg * (relative_rates.detach() ** self.alpha)

        grad_loss = torch.sum(torch.abs(grad_norms - target_grad_norms))

        self.optimizer.zero_grad()
        torch.autograd.backward(grad_loss, inputs=[self.task_weights], retain_graph=True)
        self.optimizer.step()

        with torch.no_grad():
            self.task_weights.data.clamp_(min=1e-6)
            self.task_weights.data = self.task_weights.data * (
                len(self.task_names) / self.task_weights.data.sum().clamp_min(1e-12)
            )

        return {
            "grad_loss": float(grad_loss.detach().item()),
            "weights": self.get_weight_dict(),
            "grad_norms": {
                task_name: float(value)
                for task_name, value in zip(self.task_names, grad_norms.detach().cpu().tolist())
            },
            "relative_rates": {
                task_name: float(value)
                for task_name, value in zip(self.task_names, relative_rates.detach().cpu().tolist())
            },
        }


def _set_flat_grad(params: Sequence[torch.nn.Parameter], flat_grad: torch.Tensor) -> None:
    offset = 0
    for param in params:
        numel = param.numel()
        grad_slice = flat_grad[offset : offset + numel].view_as(param).clone()
        param.grad = grad_slice
        offset += numel


def _compute_task_grad_matrix(
    task_losses: Dict[str, torch.Tensor],
    params: Sequence[torch.nn.Parameter],
) -> Tuple[List[str], torch.Tensor]:
    task_names = list(task_losses.keys())
    if not task_names:
        raise ValueError("task_losses must contain at least one task")

    rows = []
    for task_name in task_names:
        loss = task_losses[task_name]
        grads = torch.autograd.grad(
            loss,
            params,
            allow_unused=True,
            create_graph=False,
            retain_graph=True,
        )
        rows.append(_flatten_grads(grads, params))
    G = torch.stack(rows, dim=0)
    return task_names, G


def _project_to_simplex(v: torch.Tensor) -> torch.Tensor:
    if v.ndim != 1:
        raise ValueError("_project_to_simplex expects a 1D tensor")
    n = int(v.numel())
    if n == 0:
        return v
    if n == 1:
        return torch.ones_like(v)

    u, _ = torch.sort(v, descending=True)
    cssv = torch.cumsum(u, dim=0) - 1.0
    ind = torch.arange(1, n + 1, device=v.device, dtype=v.dtype)
    cond = u - cssv / ind > 0
    rho = int(torch.nonzero(cond, as_tuple=False)[-1].item())
    theta = cssv[rho] / ind[rho]
    w = torch.clamp(v - theta, min=0.0)

    w_sum = torch.sum(w)
    if float(w_sum.item()) <= 0.0:
        return torch.full_like(v, 1.0 / float(n))
    return w / w_sum


def _merge_grads_pcgrad(G: torch.Tensor) -> torch.Tensor:
    if G.ndim != 2:
        raise ValueError("G must have shape (T, P)")
    T = int(G.shape[0])
    device = G.device

    projected = G.clone()

    for i in range(T):
        gi = G[i].clone()
        order = torch.randperm(T, device=device)
        for j in order.tolist():
            if i == j:
                continue
            gj = G[j]
            dot_ij = torch.dot(gi, gj)
            if dot_ij < 0:
                gi = gi - (dot_ij / (torch.dot(gj, gj) + 1e-12)) * gj
        projected[i] = gi

    return projected.mean(dim=0)


# def _solve_mgda_weights(G: torch.Tensor, max_iter: int, lr: float) -> torch.Tensor:
#     if G.ndim != 2:
#         raise ValueError("G must have shape (T, P)")
#     T = int(G.shape[0])
#     alpha = torch.full((T,), 1.0 / float(T), device=G.device, dtype=G.dtype)
#     A = G @ G.transpose(0, 1)
#     for _ in range(int(max_iter)):
#         grad = 2.0 * (A @ alpha)
#         alpha = _project_to_simplex(alpha - float(lr) * grad)
#     return alpha

def _solve_mgda_weights(G: torch.Tensor, max_iter: int, lr: float) -> torch.Tensor:
    if G.ndim != 2:
        raise ValueError("G must have shape (T, P)")
    T = int(G.shape[0])
    if T == 1:
        return torch.ones(1, device=G.device, dtype=G.dtype)

    alpha = torch.full((T,), 1.0 / float(T), device=G.device, dtype=G.dtype)
    A = G @ G.transpose(0, 1)

    # Use a safe step size based on spectral norm
    eigvals = torch.linalg.eigvalsh(A)
    L = 2.0 * torch.clamp(eigvals.max(), min=1e-12)
    step = min(float(lr), float(1.0 / L))

    for _ in range(int(max_iter)):
        grad = 2.0 * (A @ alpha)
        alpha = _project_to_simplex(alpha - step * grad)

    return alpha


def _merge_grads_mgda(G: torch.Tensor, max_iter: int, lr: float) -> Tuple[torch.Tensor, torch.Tensor]:
    alpha = _solve_mgda_weights(G, max_iter=max_iter, lr=lr)
    merged_grad = alpha @ G
    return merged_grad, alpha


# def _solve_cagrad_weights(G: torch.Tensor, c: float, max_iter: int, lr: float) -> torch.Tensor:
#     if G.ndim != 2:
#         raise ValueError("G must have shape (T, P)")
#     T = int(G.shape[0])
#     w = torch.full((T,), 1.0 / float(T), device=G.device, dtype=G.dtype)
#     g0 = torch.mean(G, dim=0)
#     for _ in range(int(max_iter)):
#         gw = w @ G
#         gw_norm = torch.norm(gw) + 1e-12
#         grad = (G @ g0) + float(c) * ((G @ gw) / gw_norm)
#         w = _project_to_simplex(w - float(lr) * grad)
#     return w


# def _merge_grads_cagrad(
#     G: torch.Tensor,
#     c: float,
#     max_iter: int,
#     lr: float,
# ) -> Tuple[torch.Tensor, torch.Tensor]:
#     w = _solve_cagrad_weights(G, c=c, max_iter=max_iter, lr=lr)
#     g0 = torch.mean(G, dim=0)
#     gw = w @ G
#     merged_grad = g0 + float(c) * gw / (torch.norm(gw) + 1e-12)
#     return merged_grad, w

def _solve_cagrad_weights(
    G: torch.Tensor,
    c: float,
    max_iter: int,
    lr: float,
) -> torch.Tensor:
    if G.ndim != 2:
        raise ValueError("G must have shape (T, P)")

    T = int(G.shape[0])
    if T == 0:
        raise ValueError("G must contain at least one task gradient")
    if T == 1:
        return torch.ones(1, device=G.device, dtype=G.dtype)

    # Average gradient
    g0 = torch.mean(G, dim=0)                  # (P,)

    # Gram matrix and linear term
    A = G @ G.transpose(0, 1)                  # (T, T)
    b = G @ g0                                 # (T,)

    # Initialize on simplex
    w = torch.full((T,), 1.0 / float(T), device=G.device, dtype=G.dtype)

    # Use a safer step size based on spectral norm of A
    eigvals = torch.linalg.eigvalsh(A)
    L = torch.clamp(eigvals.max(), min=1e-12)
    step = min(float(lr), float(1.0 / L))

    eps = 1e-12

    for _ in range(int(max_iter)):
        quad = torch.clamp(w @ A @ w, min=eps)           # scalar
        sqrt_quad = torch.sqrt(quad)                     # ||gw||

        # Objective:
        #   f(w) = b^T w + c * sqrt(w^T A w + eps)
        #
        # Gradient:
        #   grad = b + c * (A w) / sqrt(w^T A w + eps)
        grad = b + float(c) * (A @ w) / sqrt_quad

        w = _project_to_simplex(w - step * grad)

    return w


def _merge_grads_cagrad(
    G: torch.Tensor,
    c: float,
    max_iter: int,
    lr: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if G.ndim != 2:
        raise ValueError("G must have shape (T, P)")

    g0 = torch.mean(G, dim=0)  # (P,)
    w = _solve_cagrad_weights(G, c=c, max_iter=max_iter, lr=lr)
    gw = w @ G                 # (P,)

    gw_norm = torch.norm(gw).clamp_min(1e-12)

    # Practical CAGrad-style merged gradient
    merged_grad = g0 + float(c) * gw / gw_norm
    return merged_grad, w


def _apply_multitask_gradient(model: torch.nn.Module, task_losses: Dict[str, torch.Tensor], cfg) -> Dict[str, object]:
    params = _get_trainable_params(model)
    if not params:
        raise ValueError("No trainable parameters found")

    task_names, G = _compute_task_grad_matrix(task_losses, params)
    method = str(cfg.gradient_balancer.method).strip().lower()

    if method == "pcgrad":
        merged_grad = _merge_grads_pcgrad(G)
        task_weights = None
    elif method == "mgda":
        merged_grad, alpha = _merge_grads_mgda(
            G,
            max_iter=int(cfg.gradient_balancer.solver_max_iter),
            lr=float(cfg.gradient_balancer.solver_lr),
        )
        task_weights = alpha.detach().cpu().tolist()
    elif method == "cagrad":
        merged_grad, w = _merge_grads_cagrad(
            G,
            c=float(cfg.gradient_balancer.cagrad_c),
            max_iter=int(cfg.gradient_balancer.solver_max_iter),
            lr=float(cfg.gradient_balancer.solver_lr),
        )
        task_weights = w.detach().cpu().tolist()
    else:
        raise ValueError(f"Unsupported gradient balancer method: {method}")

    _set_flat_grad(params, merged_grad)
    return {
        "method": method,
        "task_names": task_names,
        "task_weights": task_weights,
    }
