import torch
from torch.autograd import grad

try:
    import wandb
except Exception:
    class _DummyWandb(object):
        def log(self, *args, **kwargs):
            return None

    wandb = _DummyWandb()


def log_losses_wandb(epoch, bach_idx, num_batches, log_dict, batch_size, weights):
    fraction_done = (bach_idx + 1) / num_batches
    step = (epoch + fraction_done) * num_batches * batch_size
    log_dict["train/step"] = step
    for loss_term_name in weights.keys():
        term_name = loss_term_name + "_term"
        if term_name in log_dict:
            log_dict["weighted/" + term_name] = weights[loss_term_name] * log_dict[term_name].detach().cpu().numpy()
    wandb.log(log_dict)
    return step


def log_string(out_str, log_file):
    log_file.write(out_str + "\n")
    log_file.flush()
    print(out_str)


def gradient(inputs, outputs, create_graph=True, retain_graph=True):
    d_points = torch.ones_like(outputs, requires_grad=False, device=outputs.device)
    return grad(
        outputs=outputs,
        inputs=inputs,
        grad_outputs=d_points,
        create_graph=create_graph,
        retain_graph=retain_graph,
        only_inputs=True,
    )[0]


def experts_gradient(inputs, outputs, create_graph=True, retain_graph=True):
    d_points = torch.ones_like(outputs, requires_grad=False, device=outputs.device)
    points_grad = []
    for i, expert_pred in enumerate(outputs[0]):
        points_grad.append(
            grad(
                outputs=expert_pred.unsqueeze(0),
                inputs=inputs,
                grad_outputs=d_points[0][i].unsqueeze(0),
                create_graph=create_graph,
                retain_graph=retain_graph,
                only_inputs=True,
            )[0]
        )
    return torch.cat(points_grad).unsqueeze(0)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
