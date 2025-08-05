from typing import Tuple

import torch
from torch import Tensor
import torch.nn as nn

from neuralop.models import FNO

from util.fick_params import FickParams, Normalizer
from util.G_method import compute_G_in, compute_G_out, G_error


class FickModel(nn.Module):
    def __init__(self, n_modes: Tuple[int], n_layers: int, hidden_channels: int, device: torch.device):
        super().__init__()  # type: ignore
        self.fno = FNO(n_modes=n_modes, in_channels=5, out_channels=2, n_layers=n_layers, hidden_channels=hidden_channels)
        self.n_modes = n_modes
        self.n_layers = n_layers
        self.hidden_channels = hidden_channels
        self.device = device
        self.P_normalizer: Normalizer = Normalizer()
        self.C_normalizer: Normalizer = Normalizer()
        self.D_normalizer: Normalizer = Normalizer()
        self.R_normalizer: Normalizer = Normalizer()
        self.T1_normalizer: Normalizer = Normalizer()
        self.P0_normalizer: Normalizer = Normalizer()

    def preprocess(self, list_params: list[FickParams], Nt: int, Nr: int) -> Tensor:
        for i, params in enumerate(list_params):
            list_params[i] = params.get_root_parent().rescaling().nondimensionalize().compression().normalize(
                C_normalizer=self.C_normalizer.normalize,
                D_normalizer=self.D_normalizer.normalize,
                R_normalizer=self.R_normalizer.normalize,
                T1_normalizer=self.T1_normalizer.normalize
            )
        PARAMS = torch.empty(len(list_params), 5, 2 * Nt, 2 * Nr, device=self.device)
        for i, params in enumerate(list_params):
            s = (2 * Nt, 2 * Nr)
            PARAMS[i, 0] = params.C_in.to(self.device).expand(s)
            PARAMS[i, 1] = params.D_in.to(self.device).expand(s)
            PARAMS[i, 2] = params.D_out.to(self.device).expand(s)
            PARAMS[i, 3] = params.T1_in.to(self.device).expand(s)
            PARAMS[i, 4] = params.R.to(self.device).expand(s)
        return PARAMS

    def forward(self, params: list[FickParams] | FickParams) -> Tensor:
        if isinstance(params, FickParams):
            params = [params]
        Nt, Nr = int(params[0].Nt.item()), int(params[0].Nr.item())
        PARAMS = self.preprocess(params, Nt, Nr)
        fick_fno = self.fno(PARAMS)
        return fick_fno

    def search_R_in(self, list_params: list[FickParams], G_true: Tensor, Nt: int, Nr: int) -> list[Tensor]:
        optim_params_processed: list[FickParams] = []
        trainable_R_list: list[nn.Parameter] = []

        # valeurs "vraies" et r_max en ABSOLU (avant rescaling) pour l'affichage
        R_true_abs = torch.tensor(
            [float(p.get_root_parent().R) for p in list_params], device=self.device, dtype=torch.float32
        )
        rmax_abs = torch.tensor(
            [float(p.get_root_parent().r_max) for p in list_params], device=self.device, dtype=torch.float32
        )

        for p in list_params:
            p_proc = p.get_root_parent().rescaling().nondimensionalize().compression().normalize(
                C_normalizer=self.C_normalizer.normalize,
                D_normalizer=self.D_normalizer.normalize,
                R_normalizer=self.R_normalizer.normalize,
                T1_normalizer=self.T1_normalizer.normalize
            )
            p_proc.R = nn.Parameter((p_proc.r_max/2).detach().clone().to(self.device))
            optim_params_processed.append(p_proc)
            trainable_R_list.append(p_proc.R)

        optimizer = torch.optim.Adam(trainable_R_list, lr=5e-2)
        n_steps = 100

        prev_abs = [None for _ in optim_params_processed]

        for step in range(n_steps):
            optimizer.zero_grad()
            s = (2 * Nt, 2 * Nr)
            PARAMS = torch.empty(len(optim_params_processed), 5, *s, device=self.device)
            for i, params in enumerate(optim_params_processed):
                PARAMS[i, 0] = params.C_in.expand(s)
                PARAMS[i, 1] = params.D_in.expand(s)
                PARAMS[i, 2] = params.D_out.expand(s)
                PARAMS[i, 3] = params.T1_in.expand(s)
                PARAMS[i, 4] = params.R.expand(s)

            fick = postprocess(self.fno(PARAMS))
            G_pred = compute_G_in(fick)
            loss = G_error(G_pred, G_true)
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                for b, p_proc in enumerate(optim_params_processed):
                    R_pred_abs = (p_proc.R * rmax_abs[b]).item()
                    rel_err = abs(R_pred_abs - R_true_abs[b].item()) / (R_true_abs[b].item() + 1e-12)
                    if prev_abs[b] is None:
                        delta = float('nan')
                    else:
                        delta = abs(R_pred_abs - prev_abs[b])
                    prev_abs[b] = R_pred_abs
                    print(f"[search_R_in] step={step:03d} batch={b:02d} "
                          f"R_pred={R_pred_abs:.6g}  |Δ|/R={rel_err:.3e}  Δstep={delta:.3e}")

        return [p.R.detach().clone() for p in optim_params_processed]

    def search_R_out(self, list_params: list[FickParams], G_true: Tensor, Nt: int, Nr: int) -> list[Tensor]:
        optim_params_processed: list[FickParams] = []
        trainable_R_list: list[nn.Parameter] = []

        # valeurs "vraies" et r_max en ABSOLU (avant rescaling) pour l'affichage
        R_true_abs = torch.tensor(
            [float(p.get_root_parent().R) for p in list_params], device=self.device, dtype=torch.float32
        )
        rmax_abs = torch.tensor(
            [float(p.get_root_parent().r_max) for p in list_params], device=self.device, dtype=torch.float32
        )

        for p in list_params:
            p_proc = p.get_root_parent().rescaling().nondimensionalize().compression().normalize(
                C_normalizer=self.C_normalizer.normalize,
                D_normalizer=self.D_normalizer.normalize,
                R_normalizer=self.R_normalizer.normalize,
                T1_normalizer=self.T1_normalizer.normalize
            )
            p_proc.R = nn.Parameter((p_proc.r_max/2).detach().clone().to(self.device))
            optim_params_processed.append(p_proc)
            trainable_R_list.append(p_proc.R)

        optimizer = torch.optim.Adam(trainable_R_list, lr=5e-2)
        n_steps, eps = 100, 1e-4
        prev_abs = [None for _ in optim_params_processed]

        for step in range(n_steps):
            optimizer.zero_grad()
            s = (2 * Nt, 2 * Nr)
            PARAMS = torch.empty(len(optim_params_processed), 5, *s, device=self.device)
            for i, params in enumerate(optim_params_processed):
                PARAMS[i, 0] = params.C_in.expand(s)
                PARAMS[i, 1] = params.D_in.expand(s)
                PARAMS[i, 2] = params.D_out.expand(s)
                PARAMS[i, 3] = params.T1_in.expand(s)
                PARAMS[i, 4] = params.R.expand(s)

            fick = postprocess(self.fno(PARAMS))
            G_pred = compute_G_out(fick, optim_params_processed)
            loss = G_error(G_pred, G_true)
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                for b, p_proc in enumerate(optim_params_processed):
                    # clamp en espace rescalé (r_max=1)
                    p_proc.R.data.clamp_(min=eps, max=float(p_proc.r_max) - eps)
                    R_pred_abs = (p_proc.R * rmax_abs[b]).item()
                    rel_err = abs(R_pred_abs - R_true_abs[b].item()) / (R_true_abs[b].item() + 1e-12)
                    if prev_abs[b] is None:
                        delta = float('nan')
                    else:
                        delta = abs(R_pred_abs - prev_abs[b])
                    prev_abs[b] = R_pred_abs
                    print(f"[search_R_out] step={step:03d} batch={b:02d} "
                          f"R_pred={R_pred_abs:.6g}  |Δ|/R={rel_err:.3e}  Δstep={delta:.3e}")

        return [p.R.detach().clone() for p in optim_params_processed]


def postprocess(fick_fno: Tensor) -> Tensor:
    Nt, Nr = fick_fno.shape[2] // 2, fick_fno.shape[3] // 2
    q1 = fick_fno[:, :, Nt:, Nr:]                      # (B=1, 2, Nt, Nr)
    q2 = torch.flip(fick_fno[:, :, Nt:, :Nr], dims=[-1])
    q3 = torch.flip(fick_fno[:, :, :Nt, Nr:], dims=[-2])
    q4 = torch.flip(fick_fno[:, :, :Nt, :Nr], dims=[-1, -2])
    fick = (q1 + q2 + q3 + q4) / 4.0
    return fick
