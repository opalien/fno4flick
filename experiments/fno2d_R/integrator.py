import torch

def compute_G_R_fno(P_pred: torch.Tensor,
                    R: float,
                    r_max: float) -> torch.Tensor:
    """
    P_pred : shape (B, 2*Nt, 2*Ns)  — sortie complète du FNO
    R      : rayon interne (même unité que r_max)
    r_max  : rayon maxi du maillage
    -------
    Retour : shape (B, Nt) — G_R_fno pour chaque temps et chaque élément du batch
    """

    # dimensions et quadrant physique (r∈[0,r_max], t∈[0,T_final])
    B, pNt, pNs = P_pred.shape
    Nt, Ns = pNt // 2, pNs // 2
    P_quad = P_pred[:, Nt:, Ns:]               # (B, Nt, Ns)

    # grille radiale uniforme
    dr = r_max / (Ns - 1)
    r_grid = torch.arange(Ns, device=P_pred.device,
                          dtype=P_pred.dtype) * dr            # (Ns,)

    # indice de coupure à R et recopie 1D pour vecteurisation
    idx_R = torch.searchsorted(r_grid, torch.tensor(R, device=r_grid.device))
    r_sub = r_grid[:idx_R + 1]                               # <= R
    if r_sub[-1] > R:                                        # cas exact rarement vrai
        r_sub[-1] = R

    # extraction & éventuelle interpolation de P(r=R)
    P_sub = P_quad[:, :, :idx_R + 1]                         # (B, Nt, len(r_sub))
    if r_sub[-1] != r_grid[idx_R]:
        # interpolation linéaire sur la dernière maille
        w = (R - r_grid[idx_R-1]) / dr
        P_R = (1 - w) * P_quad[:, :, idx_R-1] + w * P_quad[:, :, idx_R]
        P_sub[:, :, -1] = P_R

    # intégrale ∫₀ᴿ P μ² dμ   (trapèzes vectorisés)
    integrand = P_sub * (r_sub**2)
    integral = torch.trapz(integrand, r_sub, dim=2)          # (B, Nt)

    # G(R,t) = 3/R³ * intégrale
    G_R_fno = 3.0 / (R ** 3) * integral
    return G_R_fno                                           # (B, Nt)
