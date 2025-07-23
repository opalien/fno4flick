from copy import deepcopy
from matplotlib import pyplot as plt
import torch
from torch import Tensor
from neuralop.models import FNO

from experiments.fno2d.dataset import Dataset, collate_fn
from experiments.fno2d.edp_parameters import EDPParameters


def compute_G(P_fno: Tensor, params: EDPParameters) -> Tensor:
    # P_fno: (B, pNt, pNr) or (pNt, pNr)
    was_unbatched = P_fno.ndim == 2
    if was_unbatched:
        P_fno = P_fno.unsqueeze(0)

    B, pNt, pNr = P_fno.shape
    Nt, Nr = pNt // 2, pNr // 2
    R, r_max = params.R, params.r_max

    q1 = P_fno[:, Nt:, Nr:]
    q2 = torch.flip(P_fno[:, Nt:, :Nr], dims=[-1])
    q3 = torch.flip(P_fno[:, :Nt, :Nr], dims=[-2, -1])
    q4 = torch.flip(P_fno[:, :Nt, Nr:], dims=[-2])

    P_quad_avg = (q1 + q2 + q3 + q4) / 4.0                   # (B, Nt, Nr)

    r_grid = torch.linspace(0, r_max, Nr, device=P_fno.device, dtype=P_fno.dtype)

    idx_R = torch.searchsorted(r_grid, R)
    r_sub = r_grid[:idx_R]                                  # (Nr_sub,)
    P_sub = P_quad_avg[..., :idx_R]                         # (B, Nt, Nr_sub)

    # Intégrale de G(R,t) = (3/R^3) * ∫[0,R] P(r,t) * r^2 dr
    integrand = P_sub * r_sub.pow(2)
    integral = torch.trapz(integrand, r_sub, dim=-1)        # (B, Nt)
    G = (3.0 / R**3) * integral if R > 1e-9 else torch.zeros_like(integral)

    return G.squeeze(0) if was_unbatched else G             # (Nt,) or (B, Nt)




def plot_search_R(model: FNO,
                  dataset: Dataset,
                  i: int,
                  device: torch.device,
                  name: str,
                  r_max_fixed: bool = False):
    
    with torch.no_grad():    
        model.eval()
        model.to(device)

        params_true, P_true = dataset.elements[i]
        params_original = params_true.get_root_parent()

        params_tensored, P_tensored = collate_fn([dataset.elements[i]])
        P_tensored = P_tensored.squeeze(1).to(device)

        G_true = compute_G(P_tensored, params_true)

        params_guess = deepcopy(params_original)

        linespace = torch.linspace(0.5 * params_original.R, 1.5 * params_original.R, 10, device=device)
        errors: list[float] = []

        for R in linespace:
            params_guess.R = R.item()

            params = params_guess.rescaling().nondimensionalize().compression().normalize(
                dataset.C_normalizer.normalize, 
                dataset.D_normalizer.normalize, 
                dataset.R_normalizer.normalize,
                dataset.T1_normalizer.normalize
            )

            params_tensored, _ = collate_fn([(params, P_true)])

            P_pred = model(params_tensored.to(device)).squeeze(1)
            G_pred = compute_G(P_pred, params)

            errors.append(torch.norm(G_pred - G_true).cpu().item())


        plt.plot(linespace.cpu().numpy(), errors, label=f'{name} - {i}')        
        plt.axvline(x=params_original.R, color='r', linestyle='--', label=f'True R = {params_original.R:.4f}')
        plt.legend()
        plt.savefig(f"out/plots/{name}.png")
        plt.close()

