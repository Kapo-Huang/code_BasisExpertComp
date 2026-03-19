from .INR import INR
from .INR_MoE import INR_MoE
from .ionization_losses import IonizationLoss
import importlib
import sys
import os

__all__ = {
    'inr_ionization': INR,
    'inr_moe_ionization': INR_MoE,
}

__losses__ = {
    'inr_ionization': IonizationLoss,
    'inr_moe_ionization': IonizationLoss,
}

file_name_dict = {
    'inr_ionization': "INR.py",
    'inr_moe_ionization': "INR_MoE.py",
}

def build_model(cfg, loss_cfg):
    model_cfg = cfg['MODEL']
    if model_cfg['model_name'] not in __all__:
        raise NotImplementedError("Neural-Experts now only supports ionization models.")
    model = __all__[model_cfg['model_name']](cfg_all=cfg)
    loss = __losses__[model_cfg['model_name']](cfg=loss_cfg,
                                               model_name=model_cfg['model_name'],
                                               model=model, n_experts=cfg['MODEL']['n_experts'])

    return model, loss

class build_model_from_logdir(object):
    def __init__(self, logdir, cfg, loss_cfg):
        model_cfg = cfg['MODEL']
        pc_model = model_cfg.get('model_name')
        if pc_model not in __all__:
            raise NotImplementedError("Neural-Experts now only supports ionization models.")
        model_instance = __all__[pc_model]
        model_name = model_instance.__name__
        file_name = file_name_dict.get(pc_model)
        spec = importlib.util.spec_from_file_location(model_name, os.path.join(logdir, 'models', file_name))
        import_model = importlib.util.module_from_spec(spec)
        sys.modules[model_name] = import_model
        spec.loader.exec_module(import_model)
        self.model = model_instance(cfg_all=cfg)

        loss_model = model_cfg.get('model_name')
        loss_instance = __losses__[loss_model]
        loss_name = loss_instance.__name__
        file_name = file_name_dict.get(loss_model)
        spec = importlib.util.spec_from_file_location(loss_name, os.path.join(logdir, 'models', file_name))
        import_model = importlib.util.module_from_spec(spec)
        sys.modules[loss_name] = import_model
        spec.loader.exec_module(import_model)
        self.loss = loss_instance(cfg=loss_cfg, model_name=model_cfg['model_name'],
                                  model=self.model, n_experts=cfg['MODEL']['n_experts'])

    def get(self):
        return self.model, self.loss
