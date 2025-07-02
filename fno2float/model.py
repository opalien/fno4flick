import torch
from torch import nn, Tensor
from neuralop.models import FNO


class FNO2float(nn.Module):
    """
    FNO → scalaire : prédit un rayon R à partir d’un champ P(r,t).

    Entrée  : (batch, 1, Nt, Nr)   – le champ P périodisé
    Sortie  : (batch,)             – rayon moyen (ou (batch,1) si non squeezé)
    """
    def __init__(
        self,
        n_modes: int,
        hidden_channels: int,
        n_layers: int,
        in_channels: int = 1,
        out_channels: int = 1,
    ):
        super().__init__()
        self.fno = FNO(
            n_modes=(n_modes, n_modes),
            hidden_channels=hidden_channels,
            in_channels=in_channels,
            out_channels=out_channels,
            n_layers=n_layers,
        )

    def forward(self, x: Tensor) -> Tensor:
        """
        Parameters
        ----------
        x : Tensor  – shape (B, 1, Nt, Nr)

        Returns
        -------
        Tensor     – shape (B,) si out_channels == 1, sinon (B, out_channels)
        """
        grid = self.fno(x)           # (B, C, Nt, Nr)
        y = grid.mean(dim=(-2, -1))  # intégrale / (Nt·Nr) sur t & r
        if y.shape[1] == 1:
            y = y.squeeze(1)         # -> (B,)
        return y
