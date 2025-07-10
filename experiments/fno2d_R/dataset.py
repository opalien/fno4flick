import torch
from torch import Tensor
import abc
import os
import math
import matplotlib.pyplot as plt


class Params(abc.ABC):
    Nr: int
    Nt: int
    R: float
    r_max: float
    C_in: float
    D_in: float
    T1_in: float
    C_out: float
    D_out: float
    T1_out: float
    num_spatial_points: int
    n_in_points: int
    log_R: float          # valeur normalisée de log(R)


class Dataset(torch.utils.data.Dataset[tuple[Params, Tensor]]):
    """Ensemble d’exemples (Paramètres, carte P) pour l’entraînement FNO."""
    def __init__(self):
        super().__init__()
        self.elements: list[tuple[Params, Tensor]] = []

        # statistiques de normalisation
        self.P_mean: float | None = None
        self.P_std:  float | None = None
        self.param_mean: dict[str, float] | None = None
        self.param_std:  dict[str, float] | None = None

    # --------------------------------------------------------------------- #
    # Méthodes PyTorch usuelles
    # --------------------------------------------------------------------- #
    def __len__(self):            return len(self.elements)
    def __getitem__(self, idx:int): return self.elements[idx]

    # --------------------------------------------------------------------- #
    # Chargement / ajout d’éléments
    # --------------------------------------------------------------------- #
    def add_element(self, d: dict[str, Tensor | float | int]):
        """Convertit un dict sérialisé (torch.save) en (Params, Tensor)."""
        P      = d["P"].clone() if isinstance(d["P"], Tensor) else torch.tensor(d["P"])
        Nr     = int(d["Nr"]);      Nt   = int(d["Nt"])
        r_max  = float(d["r_max"]); R    = float(d["R"])
        C_in   = float(d["C_in"]);  D_in = float(d["D_in"]);  T1_in = float(d["T1_in"])
        C_out  = float(d["C_out"]); D_out= float(d["D_out"]); T1_out= float(d["T1_out"])

        nsp = Nr + 1
        nin = int(nsp * R / r_max)

        p = Params()
        p.Nr, p.Nt, p.R, p.r_max  = Nr, Nt, R, r_max
        p.C_in,  p.D_in,  p.T1_in = C_in, D_in, T1_in
        p.C_out, p.D_out, p.T1_out= C_out, D_out, T1_out
        p.num_spatial_points, p.n_in_points = nsp, nin
        p.log_R = 0.0   # sera renseigné après normalisation

        self.elements.append((p, P))

    def load(self, path:str):
        """Charge tous les *.pt d’un dossier."""
        for f in os.listdir(path):
            if f.endswith(".pt"):
                self.add_element(torch.load(os.path.join(path, f), weights_only=False))

    # --------------------------------------------------------------------- #
    # Normalisation
    # --------------------------------------------------------------------- #
    def normalize(self, eps:float=1e-8, dataset:"Dataset|None"=None):
        """Centre-réduit (C, D, T1, log R) et P."""
        if not self.elements:
            return

        # ---------- a) reprendre les stats d’un autre jeu ---------------- #
        if dataset is not None:
            if (dataset.param_mean is None or dataset.param_std is None or
                dataset.P_mean    is None or dataset.P_std  is None):
                raise ValueError("Le jeu source n’est pas normalisé.")
            self.param_mean, self.param_std = dataset.param_mean, dataset.param_std
            self.P_mean,    self.P_std      = dataset.P_mean,    dataset.P_std

        # ---------- b) calculer nos propres stats ------------------------ #
        else:
            acc = {"C":0.,"D":0.,"T1":0., "R":0.}
            acc_sq = {"C":0.,"D":0.,"T1":0., "R":0.}
            length_tot = 0.0
            P_sum = P_sq = 0.0
            P_count = 0

            for p, P in self.elements:
                li, lo = p.R, p.r_max - p.R
                L      = p.r_max

                # intégrales pondérées (C,D,T1) le long du rayon
                acc["C"]  += p.C_in * li + p.C_out * lo
                acc["D"]  += p.D_in * li + p.D_out * lo
                acc["T1"] += p.T1_in * li + p.T1_out * lo
                acc_sq["C"]  += p.C_in**2  * li + p.C_out**2  * lo
                acc_sq["D"]  += p.D_in**2  * li + p.D_out**2  * lo
                acc_sq["T1"] += p.T1_in**2 * li + p.T1_out**2 * lo
                length_tot += L

                # log(R) (scalaire) – moyenne sur l’ensemble
                lr = math.log(p.R)
                acc["R"]  += lr
                acc_sq["R"]+= lr**2

                # P
                P_sum += P.sum().item()
                P_sq  += (P*P).sum().item()
                P_count += P.numel()

            mean = {
                "C":  acc["C"]  / length_tot,
                "D":  acc["D"]  / length_tot,
                "T1": acc["T1"] / length_tot,
                "R":  acc["R"]  / len(self.elements)
            }
            std  = {
                k: math.sqrt(max(acc_sq[k]/(length_tot if k!="R" else len(self.elements))
                                - mean[k]**2, eps))
                for k in acc
            }

            self.param_mean, self.param_std = mean, std
            self.P_mean = P_sum / P_count
            self.P_std  = math.sqrt(max(P_sq / P_count - self.P_mean**2, eps))

        # ---------- c) appliquer la normalisation à chaque élément -------- #
        for p, P in self.elements:
            # P
            P.sub_(self.P_mean).div_(self.P_std)

            # C / D / T1
            for k in ("C","D","T1"):
                for suffix in ("in","out"):
                    val = getattr(p, f"{k}_{suffix}")
                    val = (val - self.param_mean[k]) / self.param_std[k]
                    setattr(p, f"{k}_{suffix}", val)

            # log R
            lr = math.log(p.R)
            p.log_R = (lr - self.param_mean["R"]) / self.param_std["R"]

    # --------------------------------------------------------------------- #
    # Visualisation rapide
    # --------------------------------------------------------------------- #
    def plot_element(self, idx:int, normalised:bool=True):
        if idx >= len(self.elements):
            return
        A, P = collate_fn([self.elements[idx]])
        A, P = A.squeeze(0), P.squeeze(0)

        if not normalised:
            if (self.param_mean is None or self.param_std is None or
                self.P_mean is None or self.P_std is None):
                raise RuntimeError("Dataset non normalisé")
            A[0].mul_(self.param_std["C"]).add_(self.param_mean["C"])
            A[1].mul_(self.param_std["D"]).add_(self.param_mean["D"])
            A[2].mul_(self.param_std["T1"]).add_(self.param_mean["T1"])
            A[3].mul_(self.param_std["R"]).add_(self.param_mean["R"])
            P.mul_(self.P_std).add_(self.P_mean)

        titles = ["C", "D", "T1", "log R", "P"]
        data   = [A[0].cpu(), A[1].cpu(), A[2].cpu(), A[3].cpu(), P.cpu()]

        fig, axs = plt.subplots(2, 3, figsize=(16, 10))
        for ax, t, d in zip(axs.flat, titles, data):
            im = ax.imshow(d.numpy(), origin="lower", aspect="auto", cmap="viridis")
            ax.set_title(t)
            fig.colorbar(im, ax=ax)
        plt.tight_layout()
        plt.savefig(f"element_{idx}_{'norm' if normalised else 'denorm'}.png", dpi=180)
        plt.close(fig)

    # --------------------------------------------------------------------- #
    # DataLoader
    # --------------------------------------------------------------------- #
    def get_dataloader(self, bs:int, shuffle:bool=True):
        return torch.utils.data.DataLoader(
            self, batch_size=bs, shuffle=shuffle, collate_fn=collate_fn
        )

# ------------------------------------------------------------------------- #
# Collate : construit les tenseurs entrée/target pour le FNO
# ------------------------------------------------------------------------- #
def collate_fn(batch:list[tuple[Params, Tensor]]):
    if not batch:
        return torch.empty(0), torch.empty(0)

    p0, P0 = batch[0]
    B      = len(batch)
    Nt     = p0.Nt
    Ns     = p0.num_spatial_points
    pNt, pNs = 2*Nt, 2*Ns

    device, dtype = P0.device, P0.dtype
    P_out      = torch.empty((B, pNt,         pNs),    device=device, dtype=dtype)
    params_out = torch.empty((B, 4, pNt, pNs), device=device, dtype=dtype)  # 4 canaux !

    for b, (p, P) in enumerate(batch):
        # ---------------- target P (symétrie quadrants) ------------------ #
        P_out[b, Nt:, Ns:]  = P
        P_out[b, Nt:, :Ns]  = torch.flip(P, (-1,))
        P_out[b, :Nt,  :]   = torch.flip(P_out[b, Nt:, :], (0,))

        # ---------------- paramètres C, D, T1 ---------------------------- #
        n  = p.n_in_points
        br = params_out[b, :3, Nt:, Ns:]        # vue (C,D,T1) quadrant+

        br[0, :, :n] = p.C_in;  br[0, :, n:] = p.C_out
        br[1, :, :n] = p.D_in;  br[1, :, n:] = p.D_out
        br[2, :, :n] = p.T1_in; br[2, :, n:] = p.T1_out

        params_out[b, :3, Nt:, :Ns] = torch.flip(br, (-1,))
        params_out[b, :3, :Nt,  :]  = torch.flip(params_out[b, :3, Nt:, :], (1,))

        # ---------------- canal log(R) constant -------------------------- #
        params_out[b, 3].fill_(p.log_R)

    return params_out, P_out
