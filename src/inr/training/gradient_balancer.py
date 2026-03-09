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
    projected = G.clone()
    for i in range(T):
        gi = projected[i]
        for j in range(T):
            if i == j:
                continue
            gj = projected[j]
            dot_ij = torch.dot(gi, gj)
            if float(dot_ij.item()) < 0.0:
                denom = torch.dot(gj, gj) + 1e-12
                gi = gi - (dot_ij / denom) * gj
        projected[i] = gi
    return torch.mean(projected, dim=0)


def _solve_mgda_weights(G: torch.Tensor, max_iter: int, lr: float) -> torch.Tensor:
    if G.ndim != 2:
        raise ValueError("G must have shape (T, P)")
    T = int(G.shape[0])
    alpha = torch.full((T,), 1.0 / float(T), device=G.device, dtype=G.dtype)
    A = G @ G.transpose(0, 1)
    for _ in range(int(max_iter)):
        grad = 2.0 * (A @ alpha)
        alpha = _project_to_simplex(alpha - float(lr) * grad)
    return alpha


def _merge_grads_mgda(G: torch.Tensor, max_iter: int, lr: float) -> Tuple[torch.Tensor, torch.Tensor]:
    alpha = _solve_mgda_weights(G, max_iter=max_iter, lr=lr)
    merged_grad = alpha @ G
    return merged_grad, alpha


def _solve_cagrad_weights(G: torch.Tensor, c: float, max_iter: int, lr: float) -> torch.Tensor:
    if G.ndim != 2:
        raise ValueError("G must have shape (T, P)")
    T = int(G.shape[0])
    w = torch.full((T,), 1.0 / float(T), device=G.device, dtype=G.dtype)
    g0 = torch.mean(G, dim=0)
    for _ in range(int(max_iter)):
        gw = w @ G
        gw_norm = torch.norm(gw) + 1e-12
        grad = (G @ g0) + float(c) * ((G @ gw) / gw_norm)
        w = _project_to_simplex(w - float(lr) * grad)
    return w


def _merge_grads_cagrad(
    G: torch.Tensor,
    c: float,
    max_iter: int,
    lr: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    w = _solve_cagrad_weights(G, c=c, max_iter=max_iter, lr=lr)
    g0 = torch.mean(G, dim=0)
    gw = w @ G
    merged_grad = g0 + float(c) * gw / (torch.norm(gw) + 1e-12)
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
