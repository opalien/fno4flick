import torch
from torch import Tensor

from util.fick_params import FickParams


def compute_G_in(fick: Tensor) -> Tensor:
    if fick.ndim != 4 or fick.shape[1] != 2:
        raise ValueError(f"compute_G attend un tenseur (B, 2, Nt, Nr), reçu {tuple(fick.shape)}")
    P_in = fick[:, 0, :, :]
    B, Nt, Nr = P_in.shape
    s = torch.linspace(0.0, 1.0, Nr, device=P_in.device, dtype=P_in.dtype)
    integrand = P_in * s.pow(2)
    G_in = 3.0 * torch.trapz(integrand, s, dim=-1)
    return G_in


def compute_G_out(fick: Tensor, list_params: list[FickParams]) -> Tensor:
    """
    fick: (B, 2, Nt, Nr)  avec canal 1 = P_out(η,t), η∈[0,1]
    list_params: liste de FickParams de taille B
    retourne: (B, Nt)
    """
    if fick.ndim != 4 or fick.shape[1] != 2:
        raise ValueError(f"compute_G_out attend (B, 2, Nt, Nr), reçu {tuple(fick.shape)}")
    B, _, Nt, Nr = fick.shape
    if len(list_params) != B:
        raise ValueError(f"len(list_params)={len(list_params)} ≠ B={B}")

    P_out = fick[:, 1]  # (B, Nt, Nr)

    # R normalisé (après rescaling ⇒ r_max=1)
    R = torch.stack([p.get_root_parent().rescaling().R for p in list_params]) \
            .to(device=P_out.device, dtype=P_out.dtype)                    # (B,)

    eta = torch.linspace(0.0, 1.0, Nr, device=P_out.device, dtype=P_out.dtype)  # (Nr,)
    r   = R[:, None, None] + (1.0 - R)[:, None, None] * eta[None, None, :]      # (B, Nt, Nr)

    I = torch.trapz(P_out * (r ** 2), eta, dim=-1)                               # (B, Nt)
    factor = (3.0 * (1.0 - R) / (1.0 - R**3).clamp_min(1e-12))[:, None]          # (B, 1)

    return factor * I   


def G_error(G_pred: Tensor, G_true: Tensor) -> Tensor:
    G_diff = (k:=torch.max(torch.tensor(0.), G_true))*(G_pred - G_true)

    return torch.norm(G_diff, p=2)/torch.norm(k, p=2)