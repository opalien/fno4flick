from __future__ import annotations

import os
import math
import numpy as np

import torch
from torch import Tensor

from experiments.fno2d.edp_parameters import EDPParameters, Normalizer


class Dataset(torch.utils.data.Dataset[tuple[EDPParameters, Tensor]]):
    def __init__(self, target_nt: int = 100, target_ns: int = 100):
        super().__init__()
        self.elements: list[tuple[EDPParameters, Tensor]] = []
        self.target_nt = target_nt
        self.target_ns = target_ns
        self.P_normalizer: Normalizer = Normalizer(mean=0.0, std=1.0)
        self.C_normalizer: Normalizer = Normalizer(mean=0.0, std=1.0)
        self.D_normalizer: Normalizer = Normalizer(mean=0.0, std=1.0)
        self.R_normalizer: Normalizer = Normalizer(mean=0.0, std=1.0)
        self.T1_normalizer: Normalizer = Normalizer(mean=0.0, std=1.0)
        self.P0_normalizer: Normalizer = Normalizer(mean=0.0, std=1.0)

    def add_element(self, d: dict[str, Tensor | float | int]):
        R, r_max = float(d["R"]), float(d["r_max"])
        Tfinal = float(d["Tfinal"])
        C_in, C_out = float(d["C_in"]), float(d["C_out"])
        D_in, D_out = float(d["D_in"]), float(d["D_out"])
        T1_in, T1_out = float(d["T1_in"]), float(d["T1_out"])
        P0_in, P0_out = float(d["P0_in"]), float(d["P0_out"])
        P_src = d["P"] if isinstance(d["P"], Tensor) else torch.tensor(d["P"])
        P_src = P_src.float()
        Nt_src, Ns_src = P_src.shape
        t_src = np.linspace(0.0, Tfinal, Nt_src, dtype=np.float64)
        t_tgt = np.linspace(0.0, Tfinal, self.target_nt, dtype=np.float64)
        P_src_np = P_src.numpy().astype(np.float64)
        P_tgt_time = np.empty((self.target_nt, Ns_src), dtype=np.float64)
        for j in range(Ns_src):
            P_tgt_time[:, j] = np.interp(t_tgt, t_src, P_src_np[:, j])
        r_src = np.linspace(0.0, r_max, Ns_src, dtype=np.float64)
        r_in = np.linspace(0.0, R, self.target_ns, dtype=np.float64)
        r_out = np.linspace(R, r_max, self.target_ns, dtype=np.float64)
        P_in_np = np.stack([np.interp(r_in, r_src, P_tgt_time[t, :]) for t in range(self.target_nt)], axis=0)
        P_out_np = np.stack([np.interp(r_out, r_src, P_tgt_time[t, :]) for t in range(self.target_nt)], axis=0)
        P_in = torch.from_numpy(P_in_np.astype(np.float32))
        P_out = torch.from_numpy(P_out_np.astype(np.float32))
        P_2c = torch.stack([P_in, P_out], dim=0)
        params = EDPParameters(
            Nr=self.target_ns - 1,
            Nt=self.target_nt,
            R=R,
            r_max=r_max,
            t_max=Tfinal,
            C_in=C_in,
            C_out=C_out,
            D_in=D_in,
            D_out=D_out,
            T1_in=T1_in,
            T1_out=T1_out,
            P0_in=P0_in,
            P0_out=P0_out,
        )
        self.elements.append((params, P_2c))

    def load(self, path: str):
        for f in os.listdir(path):
            if f.endswith(".pt"):
                self.add_element(torch.load(os.path.join(path, f), weights_only=False))

    def set_normalizers(self, dataset: Dataset):
        self.P_normalizer = Normalizer(mean=dataset.P_normalizer.mean, std=dataset.P_normalizer.std)
        self.C_normalizer = Normalizer(mean=dataset.C_normalizer.mean, std=dataset.C_normalizer.std)
        self.D_normalizer = Normalizer(mean=dataset.D_normalizer.mean, std=dataset.D_normalizer.std)
        self.R_normalizer = Normalizer(mean=dataset.R_normalizer.mean, std=dataset.R_normalizer.std)
        self.T1_normalizer = Normalizer(mean=dataset.T1_normalizer.mean, std=dataset.T1_normalizer.std)

    def normalize(self, dataset: Dataset | None = None):
        if dataset is not None:
            self.P_normalizer = Normalizer(mean=dataset.P_normalizer.mean, std=dataset.P_normalizer.std)
            self.C_normalizer = Normalizer(mean=dataset.C_normalizer.mean, std=dataset.C_normalizer.std)
            self.D_normalizer = Normalizer(mean=dataset.D_normalizer.mean, std=dataset.D_normalizer.std)
            self.R_normalizer = Normalizer(mean=dataset.R_normalizer.mean, std=dataset.R_normalizer.std)
            self.T1_normalizer = Normalizer(mean=dataset.T1_normalizer.mean, std=dataset.T1_normalizer.std)
        else:
            acc = {"C": 0.0, "D": 0.0, "T1": 0.0, "P0": 0.0, "R": 0.0}
            acc_sq = {"C": 0.0, "D": 0.0, "T1": 0.0, "P0": 0.0, "R": 0.0}
            P_sum = P_sq = 0.0
            P_count = 0
            for p, P in self.elements:
                li, lo = p.R, p.r_max - p.R
                acc_C_avg = p.C_in * li + p.C_out * lo
                acc_D_avg = p.D_in * li + p.D_out * lo
                acc_T1_avg = p.T1_in * li + p.T1_out * lo
                acc_P0_avg = p.P0_in * li + p.P0_out * lo
                acc["R"] += p.R
                acc["C"] += acc_C_avg
                acc["D"] += acc_D_avg
                acc["T1"] += acc_T1_avg
                acc["P0"] += acc_P0_avg
                acc_sq["R"] += p.R**2
                acc_sq["C"] += acc_C_avg**2
                acc_sq["D"] += acc_D_avg**2
                acc_sq["T1"] += acc_T1_avg**2
                acc_sq["P0"] += acc_P0_avg**2
                P_sum += P.sum().item()
                P_sq += (P * P).sum().item()
                P_count += P.numel()
            self.C_normalizer.mean = acc["C"] / len(self.elements)
            self.D_normalizer.mean = acc["D"] / len(self.elements)
            self.T1_normalizer.mean = acc["T1"] / len(self.elements)
            self.P0_normalizer.mean = acc["P0"] / len(self.elements)
            self.R_normalizer.mean = acc["R"] / len(self.elements)
            self.C_normalizer.std = math.sqrt(max(acc_sq["C"] / (len(self.elements)) - self.C_normalizer.mean**2, 1e-8))
            self.D_normalizer.std = math.sqrt(max(acc_sq["D"] / (len(self.elements)) - self.D_normalizer.mean**2, 1e-8))
            self.T1_normalizer.std = math.sqrt(max(acc_sq["T1"] / (len(self.elements)) - self.T1_normalizer.mean**2, 1e-8))
            self.P0_normalizer.std = math.sqrt(max(acc_sq["P0"] / (len(self.elements)) - self.P0_normalizer.mean**2, 1e-8))
            self.R_normalizer.std = math.sqrt(max(acc_sq["R"] / (len(self.elements)) - self.R_normalizer.mean**2, 1e-8))
            self.P_normalizer.mean = P_sum / P_count
            self.P_normalizer.std = math.sqrt(max(P_sq / P_count - self.P_normalizer.mean**2, 1e-8))
        for i, (p, P) in enumerate(self.elements):
            P = self.P_normalizer.normalize(P)
            p = p.normalize(self.C_normalizer.normalize, self.D_normalizer.normalize, None, self.T1_normalizer.normalize)
            self.elements[i] = (p, P)

    def rescale(self):
        for i, (p, P) in enumerate(self.elements):
            p = p.rescaling()
            self.elements[i] = (p, P)

    def nondimensionalize(self):
        for i, (p, P) in enumerate(self.elements):
            p = p.nondimensionalize()
            self.elements[i] = (p, P)

    def compress(self):
        for i, (p, P) in enumerate(self.elements):
            p = p.compression()
            self.elements[i] = (p, P)

    def plot_element(self, idx: int, normalised: bool = True):
        import copy
        import matplotlib.pyplot as plt
        if idx >= len(self.elements):
            return
        params, P2c = self.elements[idx]
        P2c = P2c.clone()
        plot_params = params
        if not normalised:
            plot_params = copy.deepcopy(params)
            plot_params.C_in = self.C_normalizer.unnormalize(params.C_in)
            plot_params.C_out = self.C_normalizer.unnormalize(params.C_out)
            plot_params.D_in = self.D_normalizer.unnormalize(params.D_in)
            plot_params.D_out = self.D_normalizer.unnormalize(params.D_out)
            plot_params.T1_in = self.T1_normalizer.unnormalize(params.T1_in)
            plot_params.T1_out = self.T1_normalizer.unnormalize(params.T1_out)
            plot_params.P0_in = self.P0_normalizer.unnormalize(params.P0_in)
            plot_params.P0_out = self.P0_normalizer.unnormalize(params.P0_out)
            plot_params.R = self.R_normalizer.unnormalize(params.R)
            P2c = self.P_normalizer.unnormalize(P2c)
        P_in = P2c[0]
        P_out = P2c[1]
        fig, axs = plt.subplots(1, 2, figsize=(12, 5))
        im0 = axs[0].imshow(P_in.cpu().numpy(), origin="lower", aspect="auto", cmap="viridis")
        axs[0].set_title("P_in  (r∈[0,R] → [0,1])")
        fig.colorbar(im0, ax=axs[0], fraction=0.046, pad=0.04)
        im1 = axs[1].imshow(P_out.cpu().numpy(), origin="lower", aspect="auto", cmap="viridis")
        axs[1].set_title("P_out (r∈[R,r_max] → [0,1])")
        fig.colorbar(im1, ax=axs[1], fraction=0.046, pad=0.04)
        plt.tight_layout()
        plt.savefig(f"element_{idx}_{'norm' if normalised else 'denorm'}_2c.png", dpi=180)
        plt.close(fig)

    def __len__(self):
        return len(self.elements)

    def __getitem__(self, idx: int):
        return self.elements[idx]

    def plot_distribution(self, path: str, name: str):
        import matplotlib.pyplot as plt
        C_in = [p.C_in for p, _ in self.elements]
        C_out = [p.C_out for p, _ in self.elements]
        D_in = [p.D_in for p, _ in self.elements]
        D_out = [p.D_out for p, _ in self.elements]
        T1_in = [p.T1_in for p, _ in self.elements]
        T1_out = [p.T1_out for p, _ in self.elements]
        P0_in = [p.P0_in for p, _ in self.elements]
        fig, axs = plt.subplots(2, 3, figsize=(16, 10))
        axs[0, 0].hist(C_in, bins=100, color="blue", alpha=0.7)
        axs[0, 0].set_title("C_in")
        axs[0, 0].set_xlabel("Value")
        axs[0, 0].set_ylabel("Frequency")
        axs[0, 1].hist(C_out, bins=100, color="red", alpha=0.7)
        axs[0, 1].set_title("C_out")
        axs[0, 1].set_xlabel("Value")
        axs[0, 1].set_ylabel("Frequency")
        axs[0, 2].hist(D_in, bins=100, color="green", alpha=0.7)
        axs[0, 2].set_title("D_in")
        axs[0, 2].set_xlabel("Value")
        axs[0, 2].set_ylabel("Frequency")
        axs[1, 0].hist(D_out, bins=100, color="orange", alpha=0.7)
        axs[1, 0].set_title("D_out")
        axs[1, 0].set_xlabel("Value")
        axs[1, 0].set_ylabel("Frequency")
        axs[1, 1].hist(T1_in, bins=100, color="purple", alpha=0.7)
        axs[1, 1].set_title("T1_in")
        axs[1, 1].set_xlabel("Value")
        axs[1, 1].set_ylabel("Frequency")
        axs[1, 2].hist(T1_out, bins=100, color="brown", alpha=0.7)
        axs[1, 2].set_title("T1_out")
        axs[1, 2].set_xlabel("Value")
        axs[1, 2].set_ylabel("Frequency")
        plt.tight_layout()
        plt.savefig(os.path.join(path, f"{name}_parameter_distributions.png"), dpi=180)
        plt.close(fig)

    def get_dataloader(self, bs: int, shuffle: bool = True):
        return torch.utils.data.DataLoader(self, batch_size=bs, shuffle=shuffle, collate_fn=collate_fn)


def collate_fn(batch: list[tuple[EDPParameters, Tensor]]):
    if not batch:
        return torch.empty(0), torch.empty(0)
    p0, P0 = batch[0]
    B = len(batch)
    Nt = p0.Nt
    Ns = p0.Nr + 1
    pNt, pNs = 2 * Nt, 2 * Ns
    device, dtype = P0.device, P0.dtype
    X = torch.empty((B, 6, pNt, pNs), device=device, dtype=dtype)
    Y = torch.empty((B, 2, pNt, pNs), device=device, dtype=dtype)
    for b, (p, P2c) in enumerate(batch):
        P2c = P2c.to(device=device, dtype=dtype)
        Cin = torch.full((Nt, Ns), p.C_in, device=device, dtype=dtype)
        Din = torch.full((Nt, Ns), p.D_in, device=device, dtype=dtype)
        Dout = torch.full((Nt, Ns), p.D_out, device=device, dtype=dtype)
        T1in = torch.full((Nt, Ns), p.T1_in, device=device, dtype=dtype)
        T1out = torch.full((Nt, Ns), p.T1_out, device=device, dtype=dtype)
        rho = torch.full((Nt, Ns), p.R / (p.r_max + 1e-12), device=device, dtype=dtype)
        X_q = torch.stack([Cin, Din, Dout, T1in, T1out, rho], dim=0)
        X[b, :, Nt:, Ns:] = X_q
        X[b, :, Nt:, :Ns] = torch.flip(X_q, dims=(-1,))
        X[b, :, :Nt, :] = torch.flip(X[b, :, Nt:, :], dims=(1,))
        Y[b, :, Nt:, Ns:] = P2c
        Y[b, :, Nt:, :Ns] = torch.flip(P2c, dims=(-1,))
        Y[b, :, :Nt, :] = torch.flip(Y[b, :, Nt:, :], dims=(1,))
    return X, Y
