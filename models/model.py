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
            list_params[i] = (
                params.get_root_parent()
                .rescaling()
                .nondimensionalize()
                .compression()
                .normalize(
                    C_normalizer=self.C_normalizer.normalize,
                    D_normalizer=self.D_normalizer.normalize,
                    R_normalizer=self.R_normalizer.normalize,
                    T1_normalizer=self.T1_normalizer.normalize,
                )
            )

        PARAMS = torch.empty(len(list_params), 5, 2 * Nt, 2 * Nr, device=self.device)

        for i, params in enumerate(list_params):
            s = (2 * Nt, 2 * Nr)
            PARAMS[i, 0, :, :] = params.C_in.to(self.device).expand(s)
            PARAMS[i, 1, :, :] = params.D_in.to(self.device).expand(s)
            PARAMS[i, 2, :, :] = params.D_out.to(self.device).expand(s)
            PARAMS[i, 3, :, :] = params.T1_in.to(self.device).expand(s)
            PARAMS[i, 4, :, :] = params.R.to(self.device).expand(s)

        return PARAMS

    def _forward_from_processed(self, processed_params: list[FickParams], Nt: int, Nr: int) -> Tensor:
        s = (2 * Nt, 2 * Nr)
        B = len(processed_params)
        PARAMS = torch.empty(B, 5, *s, device=self.device)
        for i, params in enumerate(processed_params):
            PARAMS[i, 0] = params.C_in.to(self.device).expand(s)
            PARAMS[i, 1] = params.D_in.to(self.device).expand(s)
            PARAMS[i, 2] = params.D_out.to(self.device).expand(s)
            PARAMS[i, 3] = params.T1_in.to(self.device).expand(s)
            PARAMS[i, 4] = params.R.to(self.device).expand(s)
        return self.fno(PARAMS)

    def forward(self, params: list[FickParams] | FickParams) -> Tensor:
        if isinstance(params, FickParams):
            params = [params]
        Nt, Nr = int(params[0].Nt.item()), int(params[0].Nr.item())

        processed = []
        for p in params:
            processed.append(
                p.get_root_parent()
                .rescaling()
                .nondimensionalize()
                .compression()
                .normalize(
                    C_normalizer=self.C_normalizer.normalize,
                    D_normalizer=self.D_normalizer.normalize,
                    R_normalizer=self.R_normalizer.normalize,
                    T1_normalizer=self.T1_normalizer.normalize,
                )
            )

        fick_fno = self._forward_from_processed(processed, Nt, Nr)
        return fick_fno

    # ---------- Recherche de R (Golden-section, sans autodiff) ----------

    def search_R_in(self, list_params: list[FickParams], G_true: Tensor, Nt: int, Nr: int) -> list[Tensor]:
        eps = 1e-4
        phi = (5 ** 0.5 - 1) / 2
        n_steps = 60
        out_list: list[Tensor] = []

        with torch.no_grad():
            for bidx, p in enumerate(list_params):
                p_proc = (
                    p.get_root_parent()
                    .rescaling()
                    .nondimensionalize()
                    .compression()
                    .normalize(
                        C_normalizer=self.C_normalizer.normalize,
                        D_normalizer=self.D_normalizer.normalize,
                        R_normalizer=self.R_normalizer.normalize,
                        T1_normalizer=self.T1_normalizer.normalize,
                    )
                )

                # Intervalle large par défaut
                lo = 0.1*float(p_proc.r_max)
                hi = 0.9*float(p_proc.r_max)  # = 1 - eps après rescaling

                p_root = p.get_root_parent()
                R_true_dimless = float(p_root.R) / (float(p_root.r_max) + 1e-12)

                def loss_at(Rval: float) -> float:
                    p_tmp = FickParams(
                        Nt=p_proc.Nt,
                        Nr=p_proc.Nr,
                        R=torch.tensor(Rval, device=self.device, dtype=torch.float32),
                        r_max=p_proc.r_max,
                        t_max=p_proc.t_max,
                        C_in=p_proc.C_in,
                        C_out=p_proc.C_out,
                        D_in=p_proc.D_in,
                        D_out=p_proc.D_out,
                        T1_in=p_proc.T1_in,
                        T1_out=p_proc.T1_out,
                        P0_in=p_proc.P0_in,
                        P0_out=p_proc.P0_out,
                        parent=p_proc,
                        parenthood="R_override",
                    )
                    fick_fno = self._forward_from_processed([p_tmp], Nt, Nr)
                    fick = postprocess(fick_fno)
                    G_pred = compute_G_in(fick)
                    l = G_error(G_pred, G_true[bidx : bidx + 1])
                    return float(l)

                # --- 1) Balayage grossier pour bracketer le minimum
                grid_N = 21
                grid = torch.linspace(lo, hi, grid_N, device=self.device).tolist()
                vals = [loss_at(float(x)) for x in grid]
                i_min = int(torch.tensor(vals).argmin().item())

                # choisir un intervalle [a, c] qui encadre le minimum (prendre les voisins)
                if i_min == 0:
                    a, c = grid[0], grid[1]
                elif i_min == grid_N - 1:
                    a, c = grid[-2], grid[-1]
                else:
                    a, c = grid[i_min - 1], grid[i_min + 1]

                # --- 2) Golden-section sur [a, c]
                b = c - phi * (c - a)
                d = a + phi * (c - a)

                fb = loss_at(float(b))
                fd = loss_at(float(d))

                prev_R = float("nan")
                for k in range(n_steps):
                    if fb > fd:
                        a = b
                        b = d
                        fb = fd
                        d = a + phi * (c - a)
                        fd = loss_at(float(d))
                        R_pred = float(d)
                    else:
                        c = d
                        d = b
                        fd = fb
                        b = c - phi * (c - a)
                        fb = loss_at(float(b))
                        R_pred = float(b)

                    rel_err = abs(R_pred - R_true_dimless) / (R_true_dimless + 1e-12)
                    dstep = abs(R_pred - prev_R) if k > 0 else float("nan")
                    print(f"[search_R_in] step={k:03d} batch={bidx:02d} R_pred={R_pred:.6f}  |Δ|/R={rel_err:.3e}  Δstep={dstep:.3e}")
                    prev_R = R_pred

                # Vérifie toutes les bornes pour être sûr (utile si le min est très proche d'un bord)
                cand = [a, b, d, c]
                cand_vals = [loss_at(float(x)) for x in cand]
                R_best = cand[int(torch.tensor(cand_vals).argmin().item())]

                out_list.append(torch.as_tensor(float(R_best), device=self.device))
        return out_list



    def search_R_out(self, list_params: list[FickParams], G_true: Tensor, Nt: int, Nr: int) -> list[Tensor]:
        eps = 1e-4
        phi = (5 ** 0.5 - 1) / 2
        n_steps = 60
        out_list: list[Tensor] = []

        with torch.no_grad():
            for bidx, p in enumerate(list_params):
                p_proc = (
                    p.get_root_parent()
                    .rescaling()
                    .nondimensionalize()
                    .compression()
                    .normalize(
                        C_normalizer=self.C_normalizer.normalize,
                        D_normalizer=self.D_normalizer.normalize,
                        R_normalizer=self.R_normalizer.normalize,
                        T1_normalizer=self.T1_normalizer.normalize,
                    )
                )

                rmax = float(p_proc.r_max)  # = 1 après rescaling()
                a = 0.1*rmax
                c = 0.9*rmax
                b = c - phi * (c - a)
                d = a + phi * (c - a)

                p_root = p.get_root_parent()
                R_true_dimless = float(p_root.R) / (float(p_root.r_max) + 1e-12)

                def loss_at(Rval: float) -> float:
                    p_tmp = FickParams(
                        Nt=p_proc.Nt,
                        Nr=p_proc.Nr,
                        R=torch.tensor(Rval, device=self.device, dtype=torch.float32),
                        r_max=p_proc.r_max,
                        t_max=p_proc.t_max,
                        C_in=p_proc.C_in,
                        C_out=p_proc.C_out,
                        D_in=p_proc.D_in,
                        D_out=p_proc.D_out,
                        T1_in=p_proc.T1_in,
                        T1_out=p_proc.T1_out,
                        P0_in=p_proc.P0_in,
                        P0_out=p_proc.P0_out,
                        parent=p_proc,
                        parenthood="R_override",
                    )
                    fick_fno = self._forward_from_processed([p_tmp], Nt, Nr)
                    fick = postprocess(fick_fno)
                    # compute_G_out a besoin d'un paramètre cohérent pour R (dimensionless) → on passe p_tmp
                    G_pred = compute_G_out(fick, [p_tmp])
                    l = G_error(G_pred, G_true[bidx : bidx + 1])
                    return float(l)

                fb = loss_at(float(b))
                fd = loss_at(float(d))

                prev_R = float("nan")
                for k in range(n_steps):
                    if fb > fd:
                        a = b
                        b = d
                        fb = fd
                        d = a + phi * (c - a)
                        fd = loss_at(float(d))
                        R_pred = float(d)
                    else:
                        c = d
                        d = b
                        fd = fb
                        b = c - phi * (c - a)
                        fb = loss_at(float(b))
                        R_pred = float(b)

                    rel_err = abs(R_pred - R_true_dimless) / (R_true_dimless + 1e-12)
                    dstep = abs(R_pred - prev_R) if k > 0 else float("nan")
                    print(f"[search_R_out] step={k:03d} batch={bidx:02d} R_pred={R_pred:.6f}  |Δ|/R={rel_err:.3e}  Δstep={dstep:.3e}")
                    prev_R = R_pred

                R_best = b if fb <= fd else d
                out_list.append(torch.as_tensor(float(R_best), device=self.device))
        return out_list


def postprocess(fick_fno: Tensor) -> Tensor:
    Nt, Nr = fick_fno.shape[2] // 2, fick_fno.shape[3] // 2

    q1 = fick_fno[:, :, Nt:, Nr:]  # BR
    q2 = torch.flip(fick_fno[:, :, Nt:, :Nr], dims=[-1])  # BL (flip r)
    q3 = torch.flip(fick_fno[:, :, :Nt, Nr:], dims=[-2])  # TR (flip t)
    q4 = torch.flip(fick_fno[:, :, :Nt, :Nr], dims=[-1, -2])  # TL (flip r,t)

    fick = (q1 + q2 + q3 + q4) / 4.0
    return fick
