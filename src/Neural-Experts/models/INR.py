import torch.nn as nn

from models.modules import FullyConnectedNN, InputEncoder


class INR(nn.Module):
    def __init__(self, cfg_all):
        super().__init__()
        cfg = cfg_all["MODEL"]
        self.init_type = cfg["decoder_init_type"]

        self.decoder_input_encoding_module = InputEncoder(
            cfg,
            cfg["decoder_input_encoding"],
            cfg["decoder_hidden_dim"],
            module_name="decoder_input_encoding_module",
        )
        first_layer_dim = self.decoder_input_encoding_module.first_layer_dim
        self.decoder = FullyConnectedNN(
            first_layer_dim,
            cfg["out_dim"],
            num_hidden_layers=cfg["decoder_n_hidden_layers"],
            hidden_features=cfg["decoder_hidden_dim"],
            outermost_linear=bool(cfg.get("outermost_linear", True)),
            nonlinearity=cfg["decoder_nl"],
            init_type=self.init_type,
            input_encoding=cfg["decoder_input_encoding"],
            freq=cfg.get("decoder_freqs", 30.0),
            trainable_freqs=bool(cfg.get("decoder_trainable_freqs", False)),
            module_name="decoder",
        )

    def forward(self, non_mnfld_pnts, mnfld_pnts=None, **kwargs):
        non_mnfld_pnts = self.decoder_input_encoding_module(non_mnfld_pnts)

        if mnfld_pnts is not None:
            mnfld_pnts = self.decoder_input_encoding_module(mnfld_pnts)
            manifold_pnts_pred = self.decoder(mnfld_pnts)
        else:
            manifold_pnts_pred = None

        nonmanifold_pnts_pred = self.decoder(non_mnfld_pnts)

        return {
            "manifold_pnts_pred": manifold_pnts_pred,
            "nonmanifold_pnts_pred": nonmanifold_pnts_pred.permute(0, 2, 1),
        }
