from __future__ import annotations

from copy import deepcopy
import matplotlib.pyplot as plt
import torch
from torch import Tensor
from neuralop.models import FNO

from experiments.fno2d_2out.dataset import Dataset, collate_fn
from experiments.fno2d.edp_parameters import EDPParameters


def _quadrant_avg(P_full_2d: Tensor, Nt: int, Ns: int) -> Tensor:
    q1 = P_full_2d[Nt:, Ns:]
    q2 = torch.flip(P_full_2d[Nt:, :Ns], dims=[-1])
    q3 = torch.flip(P_full_2d[:Nt, :Ns], dims=[-2, -1])
    q4 = torch.flip(P_full_2d[:Nt, Ns:], dims=[-2])
    return (q1 + q2 + q3 + q4) / 4.0  # (Nt, Ns)


@torch.no_grad()
def compute_G_in_from_full(P2c_full: Tensor, params: EDPParameters) -> Tensor:
    # Accepte (2, 2Nt, 2Ns) OU (B, 2, 2Nt, 2Ns)
    if P2c_full.ndim == 3:
        P2c_full = P2c_full.unsqueeze(0)  # (1, 2, 2Nt, 2Ns)

    B, _, pNt, pNs = P2c_full.shape
    Nt, Ns = pNt // 2, pNs // 2

    # quadrant-average sur le canal "in", en préservant la dimension batch
    A = P2c_full[:, 0]                 # (B, 2Nt, 2Ns)
    q1 = A[:, Nt:, Ns:]
    q2 = torch.flip(A[:, Nt:, :Ns], dims=[-1])
    q3 = torch.flip(A[:, :Nt, :Ns], dims=[-2, -1])
    q4 = torch.flip(A[:, :Nt, Ns:], dims=[-2])
    P_in_avg = (q1 + q2 + q3 + q4) / 4.0   # (B, Nt, Ns)

    # intégration G_in(s) avec s in [0,1]
    s = torch.linspace(0.0, 1.0, Ns, device=P2c_full.device, dtype=P2c_full.dtype)
    integrand = P_in_avg * s.pow(2)        # (B, Nt, Ns)
    G_in = 3.0 * torch.trapz(integrand, s, dim=-1)  # (B, Nt)

    return G_in.squeeze(0)  # -> (Nt,) si B=1, sinon (B, Nt)



@torch.no_grad()
def plot_search_R(model: FNO,
                  dataset: Dataset,
                  i: int,
                  device: torch.device,
                  out_path: str,
                  r_max_fixed: bool = True,
                  R_scan_points: int = 25,
                  R_scan_factor: float = 0.5):
    model.eval().to(device)

    params_transfo_true, P2c_true = dataset.elements[i]
    params_root_true = params_transfo_true.get_root_parent()

    X_true, Y_true_full = collate_fn([(params_transfo_true, P2c_true)])
    X_true = X_true.to(device)
    Y_true_full = Y_true_full.to(device).squeeze(0)  # (2, pNt, pNs)

    G_true_in = compute_G_in_from_full(Y_true_full, params_transfo_true)  # (Nt,)

    R_true = float(params_root_true.R)
    if r_max_fixed:
        R_min = max(1e-9, R_true * (1.0 - R_scan_factor))
        R_max = R_true * (1.0 + R_scan_factor)
    else:
        R_min = max(1e-9, R_true * (1.0 - R_scan_factor))
        R_max = R_true * (1.0 + R_scan_factor)

    R_grid = torch.linspace(R_min, R_max, R_scan_points, device=device, dtype=X_true.dtype)
    errors = []

    for R_guess in R_grid:
        params_guess_root = deepcopy(params_root_true)
        params_guess_root.R = float(R_guess.item())

        params_guess = (
            params_guess_root
            .rescaling()
            .nondimensionalize()
            .compression()
            .normalize(
                dataset.C_normalizer.normalize,
                dataset.D_normalizer.normalize,
                None,
                dataset.T1_normalizer.normalize
            )
        )

        X_guess, _ = collate_fn([(params_guess, P2c_true)])
        X_guess = X_guess.to(device)

        Y_pred_full = model(X_guess).squeeze(0)  # (2, pNt, pNs)
        G_pred_in = compute_G_in_from_full(Y_pred_full, params_guess)  # (Nt,)

        err = torch.linalg.norm(G_pred_in - G_true_in).item()
        errors.append(err)

    errors_t = torch.tensor(errors, device=device)
    best_idx = int(torch.argmin(errors_t).item())
    R_best = float(R_grid[best_idx].item())

    plt.figure(figsize=(6, 4))
    plt.plot(R_grid.detach().cpu().numpy(), errors_t.detach().cpu().numpy(), marker="o", lw=1.2)
    plt.axvline(R_true, color="r", linestyle="--", label=f"R_true={R_true:.4g}")
    plt.axvline(R_best, color="g", linestyle="--", label=f"R_best={R_best:.4g}")
    plt.xlabel("R (scan)")
    plt.ylabel("|| G_in_pred(R) - G_in_true ||")
    plt.title(f"Scan R (sample #{i})")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()

    return errors_t.detach().cpu().numpy(), R_grid.detach().cpu().numpy()
