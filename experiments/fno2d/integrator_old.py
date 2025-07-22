import torch
from torch import Tensor
# experiments/fno2d/integrator.py
from experiments.fno2d.edp_parameters import EDPParameters


import torch

def compute_G_R_fno(P_pred: torch.Tensor,
                    R: float,
                    r_max: float) -> torch.Tensor:
    
    B, pNt, pNs = P_pred.shape
    Nt, Ns = pNt // 2, pNs // 2
    P_quad = P_pred[:, Nt:, Ns:]               # (B, Nt, Ns)

    # Gérer le cas où Ns <= 1 pour éviter la division par zéro
    dr = r_max / (Ns - 1) if Ns > 1 else 0.0
    r_grid = torch.arange(Ns, device=P_pred.device,
                          dtype=P_pred.dtype) * dr            # (Ns,)

    idx_R = torch.searchsorted(r_grid, torch.tensor(R, device=r_grid.device))
    
    # On clone les tenseurs pour éviter de modifier les originaux par inadvertance
    r_sub = r_grid[:idx_R + 1].clone()
    P_sub = P_quad[:, :, :idx_R + 1].clone()

    # Si R est dans la grille, on ajuste le dernier point de r_sub pour qu'il soit exactement R
    if idx_R < Ns and r_sub[-1] > R:
        r_sub[-1] = R

    # On interpole P à la position R si R ne tombe pas exactement sur un point de la grille.
    # On ajoute une condition pour s'assurer que idx_R est un indice valide avant d'accéder à r_grid[idx_R].
    if idx_R < Ns and r_sub[-1] != r_grid[idx_R]:
        # On s'assure également que idx_R > 0 avant d'accéder à r_grid[idx_R-1].
        if idx_R > 0:
            w = (R - r_grid[idx_R-1]) / dr
            P_R = (1 - w) * P_quad[:, :, idx_R-1] + w * P_quad[:, :, idx_R]
            P_sub[:, :, -1] = P_R
        # Si idx_R est 0, on ne peut pas interpoler, donc on utilise simplement la valeur à l'index 0.

    integrand = P_sub * (r_sub**2)
    integral = torch.trapz(integrand, r_sub, dim=2)          # (B, Nt)

    # Éviter la division par zéro si R est très petit
    if R > 1e-9:
        G_R_fno = 3.0 / (R ** 3) * integral
    else:
        G_R_fno = torch.zeros_like(integral)
        
    return G_R_fno






def compute_G_R_fno_(P_pred: torch.Tensor,
                    R: float,
                    r_max: float) -> torch.Tensor:
    
    B, pNt, pNs = P_pred.shape
    Nt, Ns = pNt // 2, pNs // 2
    P_quad = P_pred[:, Nt:, Ns:]               # (B, Nt, Ns)

    dr = r_max / (Ns - 1)
    r_grid = torch.arange(Ns, device=P_pred.device,
                          dtype=P_pred.dtype) * dr            # (Ns,)

    idx_R = torch.searchsorted(r_grid, torch.tensor(R, device=r_grid.device))
    r_sub = r_grid[:idx_R + 1]                               # <= R
    if r_sub[-1] > R:                                        # cas exact rarement vrai
        r_sub[-1] = R

    P_sub = P_quad[:, :, :idx_R + 1]                         # (B, Nt, len(r_sub))
    if r_sub[-1] != r_grid[idx_R]:
        w = (R - r_grid[idx_R-1]) / dr
        P_R = (1 - w) * P_quad[:, :, idx_R-1] + w * P_quad[:, :, idx_R]
        P_sub[:, :, -1] = P_R

    integrand = P_sub * (r_sub**2)
    integral = torch.trapz(integrand, r_sub, dim=2)          # (B, Nt)

    G_R_fno = 3.0 / (R ** 3) * integral
    return G_R_fno






def compute_G(P: Tensor, params: EDPParameters) -> Tensor:
    prop = params.R/params.r_max

    if P.ndim == 3:
        B, Nt, Nr = P.shape        
    else:
        Nt, Nr = P.shape

    P_in = P[..., Nr//4:3*Nr//4]

    return torch.mean(P_in, dim=(-1, -2))