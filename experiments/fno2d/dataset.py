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


class Dataset(torch.utils.data.Dataset[tuple[Params, Tensor]]):
    def __init__(self):
        super().__init__()
        self.elements: list[tuple[Params, Tensor]] = []
        self.P_mean: float | None = None
        self.P_std: float | None = None
        self.param_mean: dict[str, float] | None = None
        self.param_std: dict[str, float] | None = None

    def __len__(self):
        return len(self.elements)

    def __getitem__(self, idx: int):
        return self.elements[idx]

    def add_element(self, d: dict[str, Tensor | float | int]):
        P = d["P"].clone() if isinstance(d["P"], Tensor) else torch.tensor(d["P"])
        Nr = int(d["Nr"])
        Nt = int(d["Nt"])
        r_max = float(d["r_max"])
        R = float(d["R"])
        C_in = float(d["C_in"])
        D_in = float(d["D_in"])
        T1_in = float(d["T1_in"])
        C_out = float(d["C_out"])
        D_out = float(d["D_out"])
        T1_out = float(d["T1_out"])
        nsp = Nr + 1
        nin = int(nsp * R / r_max)

        p = Params()
        p.Nr = Nr
        p.Nt = Nt
        p.R = R
        p.r_max = r_max
        p.C_in = C_in
        p.D_in = D_in
        p.T1_in = T1_in
        p.C_out = C_out
        p.D_out = D_out
        p.T1_out = T1_out
        p.num_spatial_points = nsp
        p.n_in_points = nin

        self.elements.append((p, P))

    def load(self, path: str):
        for f in os.listdir(path):
            if f.endswith(".pt"):
                self.add_element(torch.load(os.path.join(path, f), weights_only=False))

    def normalize(self, eps: float = 1e-8):
        if not self.elements:
            return

        acc = {"C": 0.0, "D": 0.0, "T1": 0.0}
        acc_sq = {"C": 0.0, "D": 0.0, "T1": 0.0}
        length_tot = 0.0
        P_sum = 0.0
        P_sq = 0.0
        P_count = 0

        for p, P in self.elements:
            li = p.R
            lo = p.r_max - p.R
            L = p.r_max

            acc["C"] += p.C_in * li + p.C_out * lo
            acc["D"] += p.D_in * li + p.D_out * lo
            acc["T1"] += p.T1_in * li + p.T1_out * lo

            acc_sq["C"] += p.C_in ** 2 * li + p.C_out ** 2 * lo
            acc_sq["D"] += p.D_in ** 2 * li + p.D_out ** 2 * lo
            acc_sq["T1"] += p.T1_in ** 2 * li + p.T1_out ** 2 * lo

            length_tot += L

            P_sum += P.sum().item()
            P_sq += (P * P).sum().item()
            P_count += P.numel()

        mean = {k: acc[k] / length_tot for k in acc}
        std = {k: math.sqrt(max(acc_sq[k] / length_tot - mean[k] ** 2, eps)) for k in acc}

        self.param_mean = mean
        self.param_std = std

        self.P_mean = P_sum / P_count
        self.P_std = math.sqrt(max(P_sq / P_count - self.P_mean ** 2, eps))

        for p, P in self.elements:
            P.sub_(self.P_mean).div_(self.P_std)

            p.C_in = (p.C_in - mean["C"]) / std["C"]
            p.C_out = (p.C_out - mean["C"]) / std["C"]

            p.D_in = (p.D_in - mean["D"]) / std["D"]
            p.D_out = (p.D_out - mean["D"]) / std["D"]

            p.T1_in = (p.T1_in - mean["T1"]) / std["T1"]
            p.T1_out = (p.T1_out - mean["T1"]) / std["T1"]

    def plot_element(self, idx: int, normalised: bool = True):
        if idx >= len(self.elements):
            return

        params_batch, P_batch = collate_fn([self.elements[idx]])
        A = params_batch.squeeze(0)
        B = P_batch.squeeze(0)

        if not normalised:
            if self.param_mean is None or self.param_std is None or self.P_mean is None or self.P_std is None:
                raise RuntimeError("Dataset not normalised")
            A[0].mul_(self.param_std["C"]).add_(self.param_mean["C"])
            A[1].mul_(self.param_std["D"]).add_(self.param_mean["D"])
            A[2].mul_(self.param_std["T1"]).add_(self.param_mean["T1"])
            B.mul_(self.P_std).add_(self.P_mean)

        titles = ["C", "D", "T1", "P"]
        data = [A[0].cpu().numpy(), A[1].cpu().numpy(), A[2].cpu().numpy(), B.cpu().numpy()]

        fig, axs = plt.subplots(2, 2, figsize=(15, 10))
        for ax, t, d in zip(axs.flat, titles, data):
            im = ax.imshow(d, origin="lower", aspect="auto", cmap="viridis")
            ax.set_title(t)
            fig.colorbar(im, ax=ax)
        plt.tight_layout()
        fname = f"element_{idx}_{'norm' if normalised else 'denorm'}.png"
        plt.savefig(fname, dpi=180)
        plt.close(fig)

    def get_dataloader(self, bs: int, shuffle: bool = True):
        return torch.utils.data.DataLoader(
            self,
            batch_size=bs,
            shuffle=shuffle,
            collate_fn=collate_fn
        )


def collate_fn(batch: list[tuple[Params, Tensor]]):
    if not batch:
        return torch.empty(0), torch.empty(0)

    p0, P0 = batch[0]
    B = len(batch)
    Nt = p0.Nt
    Ns = p0.num_spatial_points
    pNt = 2 * Nt
    pNs = 2 * Ns

    device = P0.device
    dtype = P0.dtype

    P_out = torch.empty((B, pNt, pNs), device=device, dtype=dtype)
    params_out = torch.empty((B, 3, pNt, pNs), device=device, dtype=dtype)

    for b, (p, P) in enumerate(batch):
        P_out[b, Nt:, Ns:] = P
        P_out[b, Nt:, :Ns] = torch.flip(P, (-1,))
        P_out[b, :Nt, :] = torch.flip(P_out[b, Nt:, :], (0,))

        n = p.n_in_points
        br = params_out[b, :, Nt:, Ns:]

        br[0, :, :n] = p.C_in
        br[0, :, n:] = p.C_out
        br[1, :, :n] = p.D_in
        br[1, :, n:] = p.D_out
        br[2, :, :n] = p.T1_in
        br[2, :, n:] = p.T1_out

        params_out[b, :, Nt:, :Ns] = torch.flip(br, (-1,))
        params_out[b, :, :Nt, :] = torch.flip(params_out[b, :, Nt:, :], (1,))

    return params_out, P_out
