from typing import Dict, List, Optional, Sequence, Tuple

import torch


def get_named_trainable_params(model: torch.nn.Module) -> List[Tuple[str, torch.nn.Parameter]]:
    return [(name, param) for name, param in model.named_parameters() if param.requires_grad]


def flatten_named_grads(
    grads: Sequence[Optional[torch.Tensor]],
    named_params: Sequence[Tuple[str, torch.nn.Parameter]],
) -> torch.Tensor:
    flat_parts = []
    for grad, (_, param) in zip(grads, named_params):
        grad_tensor = grad if grad is not None else torch.zeros_like(param)
        flat_parts.append(grad_tensor.reshape(-1))
    if not flat_parts:
        return torch.zeros(0)
    return torch.cat(flat_parts, dim=0)


def compute_task_grad_diagnostics(
    task_losses: Dict[str, torch.Tensor],
    model: torch.nn.Module,
) -> Dict[str, object]:
    if not task_losses:
        raise ValueError("task_losses must contain at least one task")

    named_params = get_named_trainable_params(model)
    if not named_params:
        raise ValueError("No trainable parameters found")

    params = [param for _, param in named_params]
    task_names = list(task_losses.keys())
    flat_grads: Dict[str, torch.Tensor] = {}
    layer_grads: Dict[str, Dict[str, torch.Tensor]] = {}

    for task_name in task_names:
        grads = torch.autograd.grad(
            task_losses[task_name],
            params,
            allow_unused=True,
            create_graph=False,
            retain_graph=True,
        )
        flat_grads[task_name] = flatten_named_grads(grads, named_params)
        layer_grads[task_name] = {}
        for (param_name, param), grad in zip(named_params, grads):
            grad_tensor = grad if grad is not None else torch.zeros_like(param)
            layer_grads[task_name][param_name] = grad_tensor.reshape(-1)

    return {
        "task_names": task_names,
        "flat_grads": flat_grads,
        "layer_grads": layer_grads,
    }


def summarize_gradient_norms(grad_info: Dict[str, object]) -> Dict[str, float]:
    task_names = list(grad_info["task_names"])
    flat_grads = grad_info["flat_grads"]
    return {
        task_name: float(torch.norm(flat_grads[task_name], p=2).detach().item())
        for task_name in task_names
    }


def summarize_pairwise_cosine(grad_info: Dict[str, object]) -> Dict[Tuple[str, str], float]:
    task_names = list(grad_info["task_names"])
    flat_grads = grad_info["flat_grads"]
    pairwise_cos: Dict[Tuple[str, str], float] = {}

    for i, task_i in enumerate(task_names):
        grad_i = flat_grads[task_i]
        norm_i = torch.norm(grad_i, p=2)
        for j in range(i + 1, len(task_names)):
            task_j = task_names[j]
            grad_j = flat_grads[task_j]
            norm_j = torch.norm(grad_j, p=2)
            if float(norm_i.detach().item()) == 0.0 or float(norm_j.detach().item()) == 0.0:
                cosine = 0.0
            else:
                cosine = float(torch.dot(grad_i, grad_j).div(norm_i * norm_j).detach().item())
                cosine = max(-1.0, min(1.0, cosine))
            pairwise_cos[(task_i, task_j)] = cosine

    return pairwise_cos


def summarize_layer_conflicts(grad_info: Dict[str, object]) -> List[Dict[str, object]]:
    task_names = list(grad_info["task_names"])
    layer_grads = grad_info["layer_grads"]
    if not task_names:
        return []

    layers = list(layer_grads[task_names[0]].keys())
    layer_conflicts = []

    for layer_name in layers:
        pair_cosines = []
        nonzero_pairs = 0
        for i, task_i in enumerate(task_names):
            grad_i = layer_grads[task_i][layer_name]
            norm_i = torch.norm(grad_i, p=2)
            for j in range(i + 1, len(task_names)):
                task_j = task_names[j]
                grad_j = layer_grads[task_j][layer_name]
                norm_j = torch.norm(grad_j, p=2)
                if float(norm_i.detach().item()) == 0.0 or float(norm_j.detach().item()) == 0.0:
                    cosine = 0.0
                else:
                    cosine = float(torch.dot(grad_i, grad_j).div(norm_i * norm_j).detach().item())
                    cosine = max(-1.0, min(1.0, cosine))
                    nonzero_pairs += 1
                pair_cosines.append(cosine)

        if not pair_cosines or nonzero_pairs == 0:
            avg_cos = 0.0
            min_cos = 0.0
            neg_ratio = 0.0
        else:
            avg_cos = float(sum(pair_cosines) / float(len(pair_cosines)))
            min_cos = float(min(pair_cosines))
            neg_ratio = float(sum(1 for cosine in pair_cosines if cosine < 0.0) / float(len(pair_cosines)))

        layer_conflicts.append(
            {
                "layer": layer_name,
                "avg_cos": avg_cos,
                "min_cos": min_cos,
                "neg_ratio": neg_ratio,
            }
        )

    layer_conflicts.sort(key=lambda item: item["avg_cos"])
    return layer_conflicts


def diagnose_multitask_gradients(
    task_losses: Dict[str, torch.Tensor],
    model: torch.nn.Module,
) -> Dict[str, object]:
    grad_info = compute_task_grad_diagnostics(task_losses, model)
    return {
        "grad_norms": summarize_gradient_norms(grad_info),
        "pairwise_cos": summarize_pairwise_cosine(grad_info),
        "layer_conflicts": summarize_layer_conflicts(grad_info),
    }