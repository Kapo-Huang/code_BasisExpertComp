import importlib
import os
import sys

from .INR import INR
from .INR_MoE import INR_MoE
from .mesh_losses import MeshReconstructionLoss

__all__ = {
    "inr_mesh": INR,
    "inr_moe_mesh": INR_MoE,
}

__losses__ = {
    "inr_mesh": MeshReconstructionLoss,
    "inr_moe_mesh": MeshReconstructionLoss,
}

file_name_dict = {
    "inr_mesh": "INR.py",
    "inr_moe_mesh": "INR_MoE.py",
}


def build_model(cfg, loss_cfg):
    model_cfg = cfg["MODEL"]
    model_name = model_cfg["model_name"]
    if model_name not in __all__:
        raise NotImplementedError("Neural-Experts now only supports mesh models.")
    model = __all__[model_name](cfg_all=cfg)
    loss = __losses__[model_name](
        cfg=loss_cfg,
        model_name=model_name,
        model=model,
        n_experts=cfg["MODEL"]["n_experts"],
    )
    return model, loss


class build_model_from_logdir(object):
    def __init__(self, logdir, cfg, loss_cfg):
        model_cfg = cfg["MODEL"]
        model_name = model_cfg.get("model_name")
        if model_name not in __all__:
            raise NotImplementedError("Neural-Experts now only supports mesh models.")

        model_instance = __all__[model_name]
        model_file = file_name_dict.get(model_name)
        spec = importlib.util.spec_from_file_location(model_instance.__name__, os.path.join(logdir, "models", model_file))
        import_model = importlib.util.module_from_spec(spec)
        sys.modules[model_instance.__name__] = import_model
        spec.loader.exec_module(import_model)
        self.model = model_instance(cfg_all=cfg)

        loss_instance = __losses__[model_name]
        loss_spec = importlib.util.spec_from_file_location(
            loss_instance.__name__,
            os.path.join(logdir, "models", "mesh_losses.py"),
        )
        import_loss = importlib.util.module_from_spec(loss_spec)
        sys.modules[loss_instance.__name__] = import_loss
        loss_spec.loader.exec_module(import_loss)
        self.loss = loss_instance(
            cfg=loss_cfg,
            model_name=model_name,
            model=self.model,
            n_experts=cfg["MODEL"]["n_experts"],
        )

    def get(self):
        return self.model, self.loss
