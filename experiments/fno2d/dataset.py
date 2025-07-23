from __future__ import annotations

import os
import math 

import torch
from torch import Tensor

from experiments.fno2d.edp_parameters import EDPParameters, Normalizer


class Dataset(torch.utils.data.Dataset[tuple[EDPParameters, Tensor]]):

    def __init__(self):
        super().__init__()
        
        self.elements: list[tuple[EDPParameters, Tensor]] = []

        self.P_normalizer: Normalizer = Normalizer(mean=0.0, std=1.0)
        self.C_normalizer: Normalizer = Normalizer(mean=0.0, std=1.0)
        self.D_normalizer: Normalizer = Normalizer(mean=0.0, std=1.0)
        self.R_normalizer: Normalizer = Normalizer(mean=0.0, std=1.0)
        self.T1_normalizer: Normalizer = Normalizer(mean=0.0, std=1.0)
        self.P0_normalizer: Normalizer = Normalizer(mean=0.0, std=1.0)


    def add_element(self, d: dict[str, Tensor | float | int]):
        P= d["P"].clone() if isinstance(d["P"], Tensor) else torch.tensor(d["P"])

        Nr, Nt = int(d["Nr"]), int(d["Nt"])
        R, r_max = float(d["R"]), float(d["r_max"])
        t_max = float(d["Tfinal"])
        C_in, C_out = float(d["C_in"]), float(d["C_out"])
        D_in, D_out = float(d["D_in"]), float(d["D_out"])
        T1_in, T1_out = float(d["T1_in"]), float(d["T1_out"])
        P0_in, P0_out = float(d["P0_in"]), float(d["P0_out"])


        params = EDPParameters(
            Nr=Nr, Nt=Nt,
            R=R, r_max=r_max, t_max=t_max,
            C_in=C_in, C_out=C_out,
            D_in=D_in, D_out=D_out,
            T1_in=T1_in, T1_out=T1_out,
            P0_in=P0_in, P0_out=P0_out,
        )

        self.elements.append((params, P))


    def load(self, path:str):
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
            acc = {"C":0.,"D":0.,"T1":0., "P0":0., "R":0.}
            acc_sq = {"C":0.,"D":0.,"T1":0., "P0":0., "R":0.}
            P_sum = P_sq = 0.0
            P_count = 0

            for p, P in self.elements:
                li, lo = p.R, p.r_max - p.R

                acc["R"] += p.R
                acc["C"]  += p.C_in * li + p.C_out * lo
                acc["D"]  += p.D_in * li + p.D_out * lo
                acc["T1"] += p.T1_in * li + p.T1_out * lo
                acc["P0"] += p.P0_in * li + p.P0_out * lo

                acc_sq["R"] += p.R**2
                acc_sq["C"]  += p.C_in**2  * li + p.C_out**2  * lo
                acc_sq["D"]  += p.D_in**2  * li + p.D_out**2  * lo
                acc_sq["T1"] += p.T1_in**2 * li + p.T1_out**2 * lo
                acc_sq["P0"] += p.P0_in**2 * li + p.P0_out**2 * lo

                P_sum += P.sum().item()
                P_sq  += (P*P).sum().item()
                P_count += P.numel()

            

            self.C_normalizer.mean = acc["C"]  / len(self.elements)
            self.D_normalizer.mean = acc["D"]  / len(self.elements)
            self.T1_normalizer.mean = acc["T1"] / len(self.elements)
            self.P0_normalizer.mean = acc["P0"] / len(self.elements)
            self.R_normalizer.mean = acc["R"]  / len(self.elements)

            self.C_normalizer.std = math.sqrt(max(acc_sq["C"]/(len(self.elements)) - self.C_normalizer.mean**2, 1e-8))
            self.D_normalizer.std = math.sqrt(max(acc_sq["D"]/(len(self.elements)) - self.D_normalizer.mean**2, 1e-8))
            self.T1_normalizer.std = math.sqrt(max(acc_sq["T1"]/(len(self.elements)) - self.T1_normalizer.mean**2, 1e-8))
            self.P0_normalizer.std = math.sqrt(max(acc_sq["P0"]/(len(self.elements)) - self.P0_normalizer.mean**2, 1e-8))
            self.R_normalizer.std = math.sqrt(max(acc_sq["R"]/(len(self.elements)) - self.R_normalizer.mean**2, 1e-8))

            self.P_normalizer.mean = P_sum / P_count
            self.P_normalizer.std = math.sqrt(max(P_sq / P_count - self.P_normalizer.mean**2, 1e-8))


        for i, (p, P) in enumerate(self.elements):
            
            P: Tensor = self.P_normalizer.normalize(P) # type: ignore

            p = p.normalize(self.C_normalizer.normalize, self.D_normalizer.normalize, self.R_normalizer.normalize, self.T1_normalizer.normalize)

            #p.C_in, p.C_out = self.C_normalizer.normalize(p.C_in), self.C_normalizer.normalize(p.C_out) # type: ignore
            #p.D_in, p.D_out = self.D_normalizer.normalize(p.D_in), self.D_normalizer.normalize(p.D_out) # type: ignore
            #p.T1_in, p.T1_out = self.T1_normalizer.normalize(p.T1_in), self.T1_normalizer.normalize(p.T1_out) # type: ignore
            #p.P0_in, p.P0_out = self.P0_normalizer.normalize(p.P0_in), self.P0_normalizer.normalize(p.P0_out) # type: ignore
#
            #p.R = self.R_normalizer.normalize(p.R) # type: ignore

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
      

    def plot_element(self, idx:int, normalised:bool=True):
        import copy
        import matplotlib.pyplot as plt
        
        if idx >= len(self.elements):
            return

        params, p_quadrant = self.elements[idx]
        p_quadrant = p_quadrant.clone()

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
            p_quadrant = self.P_normalizer.unnormalize(p_quadrant)

        Nr, Nt = plot_params.Nr, plot_params.Nt
        r_max, R = plot_params.r_max, plot_params.R
        Ns = Nr + 1
        n_in_points = int(Ns * R / r_max) if r_max > 0 else 0

        c_map = torch.full((Nt, Ns), plot_params.C_out)
        c_map[:, :n_in_points] = plot_params.C_in
        
        d_map = torch.full((Nt, Ns), plot_params.D_out)
        d_map[:, :n_in_points] = plot_params.D_in
        
        t1_map = torch.full((Nt, Ns), plot_params.T1_out)
        t1_map[:, :n_in_points] = plot_params.T1_in
        
        p0_map = torch.full((Nt, Ns), plot_params.P0_out)
        p0_map[:, :n_in_points] = plot_params.P0_in

        r_map = torch.full((Nt, Ns), plot_params.R)

        def apply_symmetry(quadrant):
            q_h_flip = torch.flip(quadrant, (0,))
            row1 = torch.cat((torch.flip(q_h_flip, (1,)), q_h_flip), dim=1)
            q_w_flip = torch.flip(quadrant, (1,))
            row2 = torch.cat((q_w_flip, quadrant), dim=1)
            return torch.cat((row1, row2), dim=0)

        data_maps = [apply_symmetry(m) for m in [c_map, d_map, t1_map, p0_map, r_map, p_quadrant]]
        titles = ["C", "D", "T1", "P0", "R", "P"]

        fig, axs = plt.subplots(2, 3, figsize=(16, 10))
        for ax, t, d in zip(axs.flat, titles, data_maps):
            im = ax.imshow(d.cpu().numpy(), origin="lower", aspect="auto", cmap="viridis")
            ax.set_title(t)
            fig.colorbar(im, ax=ax)
        plt.tight_layout()
        plt.savefig(f"element_{idx}_{'norm' if normalised else 'denorm'}.png", dpi=180)
        plt.close(fig)




    def __len__(self):
        return len(self.elements)
    
    def __getitem__(self, idx: int):
        return self.elements[idx]
    

    def get_dataloader(self, bs:int, shuffle:bool=True):
        return torch.utils.data.DataLoader(
            self, batch_size=bs, shuffle=shuffle, collate_fn=collate_fn
        )
    

def collate_fn(batch:list[tuple[EDPParameters, Tensor]]):
    if not batch:
        return torch.empty(0), torch.empty(0)
    
    p0, P0 = batch[0]
    B = len(batch)
    Nt = p0.Nt
    Ns = p0.Nr + 1
    pNt, pNs = 2 * Nt, 2 * Ns

    device, dtype = P0.device, P0.dtype
    P_out = torch.empty((B, pNt, pNs), device=device, dtype=dtype)
    params_out = torch.empty((B, 3, pNt, pNs), device=device, dtype=dtype)  # 3 canaux: C, D, T1

    for b, (p, P) in enumerate(batch):
        # Target P (symétrie des quadrants)
        P_out[b, Nt:, Ns:] = P
        P_out[b, Nt:, :Ns] = torch.flip(P, dims=(-1,))
        P_out[b, :Nt, :] = torch.flip(P_out[b, Nt:, :], dims=(0,))

        # Paramètres C, D, T1
        n_in_points = int(Ns * p.R / p.r_max) if p.r_max > 0 else 0
        
        br = params_out[b, :, Nt:, Ns:]  # Vue sur le quadrant positif

        br[0, :, :n_in_points] = p.C_in
        br[0, :, n_in_points:] = p.C_out
        
        br[1, :, :n_in_points] = p.D_in
        br[1, :, n_in_points:] = p.D_out
        
        br[2, :, :n_in_points] = p.T1_in
        br[2, :, n_in_points:] = p.T1_out

        params_out[b, :, Nt:, :Ns] = torch.flip(br, dims=(-1,))
        params_out[b, :, :Nt, :] = torch.flip(params_out[b, :, Nt:, :], dims=(1,))

    return params_out, P_out
    
    