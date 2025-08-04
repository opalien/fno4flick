from __future__ import annotations

import os
import math
import numpy as np

import torch
from torch import Tensor
import torch.nn.functional as F
from matplotlib import pyplot as plt

from util.fick_params import FickParams, Normalizer


class FickDataset(torch.utils.data.Dataset):
    
    def __init__(self, target_nt: int = 100, target_nr: int = 100):
        """
        Initialise le dataset.
        Args:
            target_nt (int): Résolution temporelle cible pour l'interpolation.
            target_nr (int): Résolution radiale cible pour l'interpolation (par canal).
        """
        super().__init__()
        self.elements: list[tuple[FickParams, Tensor]] = []
        self.target_nt = target_nt
        self.target_nr = target_nr

        # Initialisation des normalizers
        self.P_normalizer = Normalizer()
        self.C_normalizer = Normalizer()
        self.D_normalizer = Normalizer()
        self.R_normalizer = Normalizer()
        self.T1_normalizer = Normalizer()
        self.P0_normalizer = Normalizer()

    
    def add_element(self, d: dict[str, float | int | Tensor]):
        """
        Ajoute un élément au dataset en effectuant l'interpolation
        et la séparation en deux canaux (P_in, P_out).
        """
        # Extraction des paramètres
        R, r_max = float(d["R"]), float(d["r_max"])
        t_max = float(d["Tfinal"])
        C_in, C_out = float(d["C_in"]), float(d["C_out"])
        D_in, D_out = float(d["D_in"]), float(d["D_out"])
        T1_in, T1_out = float(d["T1_in"]), float(d["T1_out"])
        P0_in, P0_out = float(d["P0_in"]), float(d["P0_out"])

        # Tenseur source
        P_src = d["P"].clone() if isinstance(d["P"], Tensor) else torch.tensor(d["P"])
        P_src = P_src.float()
        Nt_src, Nr_src = P_src.shape

        # --- Interpolation Temporelle ---
        t_src = np.linspace(0.0, t_max, Nt_src, dtype=np.float64)
        t_tgt = np.linspace(0.0, t_max, self.target_nt, dtype=np.float64)
        P_src_np = P_src.numpy().astype(np.float64)
        P_tgt_time = np.empty((self.target_nt, Nr_src), dtype=np.float64)
        for j in range(Nr_src):
            P_tgt_time[:, j] = np.interp(t_tgt, t_src, P_src_np[:, j])

        # --- Interpolation Radiale (In & Out) ---
        r_src = np.linspace(0.0, r_max, Nr_src, dtype=np.float64)
        r_in_tgt = np.linspace(0.0, R, self.target_nr, dtype=np.float64)
        r_out_tgt = np.linspace(R, r_max, self.target_nr, dtype=np.float64)
        
        P_in_np = np.stack([np.interp(r_in_tgt, r_src, P_tgt_time[t, :]) for t in range(self.target_nt)], axis=0)
        P_out_np = np.stack([np.interp(r_out_tgt, r_src, P_tgt_time[t, :]) for t in range(self.target_nt)], axis=0)

        # Conversion en tenseurs PyTorch
        P_in = torch.from_numpy(P_in_np.astype(np.float32))
        P_out = torch.from_numpy(P_out_np.astype(np.float32))
        
        # Stack pour avoir la forme (2, Nt, Nr)
        P_2c = torch.stack([P_in, P_out], dim=0)

        # Création de l'objet FickParams avec les nouvelles dimensions
        params = FickParams.init_from(
            Nt=self.target_nt,
            Nr=self.target_nr,
            R=R, r_max=r_max, t_max=t_max,
            C_in=C_in, C_out=C_out,
            D_in=D_in, D_out=D_out,
            T1_in=T1_in, T1_out=T1_out,
            P0_in=P0_in, P0_out=P0_out
        )
        self.elements.append((params, P_2c))


    def __len__(self):
        return len(self.elements)
    
    def __getitem__(self, idx: int):
        return self.elements[idx]
    
    
    def load(self, path: str):
        for file in os.listdir(path):
            if file.endswith(".pt"):
                self.add_element(torch.load(os.path.join(path, file), weights_only=False))

        
    def set_normalizers(self, dataset: 'FickDataset'):
        self.P_normalizer = Normalizer(mean=dataset.P_normalizer.mean, std=dataset.P_normalizer.std)
        self.C_normalizer = Normalizer(mean=dataset.C_normalizer.mean, std=dataset.C_normalizer.std)
        self.D_normalizer = Normalizer(mean=dataset.D_normalizer.mean, std=dataset.D_normalizer.std)
        self.R_normalizer = Normalizer(mean=dataset.R_normalizer.mean, std=dataset.R_normalizer.std)
        self.T1_normalizer = Normalizer(mean=dataset.T1_normalizer.mean, std=dataset.T1_normalizer.std)
        self.P0_normalizer = Normalizer(mean=dataset.P0_normalizer.mean, std=dataset.P0_normalizer.std)


    def get_normalizers(self):
        acc = {"C":0.,"D":0.,"T1":0., "P0":0., "R":0.}
        acc_sq = {"C":0.,"D":0.,"T1":0., "P0":0., "R":0.}
        P_sum = P_sq = 0.0
        P_count = 0

        for p, P in self.elements:
            p_transformed = p.get_root_parent().rescaling().nondimensionalize().compression()
            li, lo = p_transformed.R, p_transformed.r_max - p_transformed.R

            acc_C_avg = p_transformed.C_in * li + p_transformed.C_out * lo
            acc_D_avg = p_transformed.D_in * li + p_transformed.D_out * lo
            acc_T1_avg = p_transformed.T1_in * li + p_transformed.T1_out * lo
            acc_P0_avg = p_transformed.P0_in * li + p_transformed.P0_out * lo

            acc["R"] += p_transformed.R
            acc["C"]  += acc_C_avg
            acc["D"]  += acc_D_avg
            acc["T1"] += acc_T1_avg
            acc["P0"] += acc_P0_avg

            acc_sq["R"] += p_transformed.R**2
            acc_sq["C"]  += acc_C_avg**2
            acc_sq["D"]  += acc_D_avg**2
            acc_sq["T1"] += acc_T1_avg**2
            acc_sq["P0"] += acc_P0_avg**2

            P_sum += P.sum().item()
            P_sq  += (P*P).sum().item()
            P_count += P.numel()

        n_elements = len(self.elements)
        self.C_normalizer.mean = acc["C"]  / n_elements
        self.D_normalizer.mean = acc["D"]  / n_elements
        self.T1_normalizer.mean = acc["T1"] / n_elements
        self.P0_normalizer.mean = acc["P0"] / n_elements

        self.C_normalizer.std = math.sqrt(max(acc_sq["C"]/n_elements - self.C_normalizer.mean**2, 1e-8))
        self.D_normalizer.std = math.sqrt(max(acc_sq["D"]/n_elements - self.D_normalizer.mean**2, 1e-8))
        self.T1_normalizer.std = math.sqrt(max(acc_sq["T1"]/n_elements - self.T1_normalizer.mean**2, 1e-8))
        self.P0_normalizer.std = math.sqrt(max(acc_sq["P0"]/n_elements - self.P0_normalizer.mean**2, 1e-8))

        self.P_normalizer.mean = P_sum / P_count
        self.P_normalizer.std = math.sqrt(max(P_sq / P_count - self.P_normalizer.mean**2, 1e-8))

        return self.P_normalizer, self.C_normalizer, self.D_normalizer, self.R_normalizer, self.T1_normalizer, self.P0_normalizer

        
    def apply_P_normalizer(self):
        for i, (p, P) in enumerate(self.elements):
            P_norm = self.P_normalizer.normalize(P)
            self.elements[i] = (p, P_norm)


    def deapply_P_normalizer(self):
        for i, (p, P) in enumerate(self.elements):
            P_unnorm = self.P_normalizer.unnormalize(P)
            self.elements[i] = (p, P_unnorm)


    def plot_distribution(self, path: str, actions: set[str] = {"rescale", "nondimensionalize", "compress", "normalize"}):
        list_params = []
        name = "brut"
        for p, P in self.elements:
            if "rescale" in actions:
                p = p.rescaling()
                name = "rescaled"
            if "nondimensionalize" in actions:
                p = p.nondimensionalize()
                name = "nondimensionalized"
            if "compress" in actions:
                p = p.compression()
                name = "compressed"
            if "normalize" in actions:
                p = p.normalize(self.C_normalizer.normalize, 
                                self.D_normalizer.normalize, 
                                self.R_normalizer.normalize, 
                                self.T1_normalizer.normalize)
                name = "normalized"
            list_params.append(p)

        fig, axs = plt.subplots(2, 3, figsize=(16, 10))
        axs[0, 0].hist([p.C_in for p in list_params], bins=100, color="blue", label="C_in")
        axs[0, 0].hist([p.C_out for p in list_params], bins=100, color="red", label="C_out", alpha=0.7)
        axs[0, 0].set_title("Distribution de C")
        axs[0, 0].legend()
        
        axs[0, 1].hist([p.D_in for p in list_params], bins=100, color="green", label="D_in")
        axs[0, 1].hist([p.D_out for p in list_params], bins=100, color="orange", label="D_out", alpha=0.7)
        axs[0, 1].set_title("Distribution de D")
        axs[0, 1].legend()

        axs[0, 2].hist([p.T1_in for p in list_params], bins=100, color="purple", label="T1_in")
        axs[0, 2].hist([p.T1_out for p in list_params], bins=100, color="brown", label="T1_out", alpha=0.7)
        axs[0, 2].set_title("Distribution de T1")
        axs[0, 2].legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(path, f"{name}.png"))
        plt.close(fig)


    def get_dataloader(self, bs:int, shuffle:bool=True):
        return torch.utils.data.DataLoader(
            self, batch_size=bs, shuffle=shuffle, collate_fn=collate_fn
        )


def collate_fn(batch: list[tuple[FickParams, Tensor]]) -> tuple[list[FickParams], Tensor]:
    """
    Assemble un batch de données. Les tenseurs P sont déjà interpolés.
    Cette fonction se charge de les assembler et d'appliquer la périodisation.
    """
    if not batch:
        return [], torch.empty(0, 2, 0, 0)
    
    # Sépare les paramètres et les tenseurs de P
    params_list = [item[0] for item in batch]
    p_tensors = [item[1] for item in batch]

    # Les tenseurs P sont déjà de la forme (2, Nt, Nr).
    # On les stacke pour former un batch (B, 2, Nt, Nr).
    P_batch = torch.stack(p_tensors, dim=0)
    
    # Périodisation par flip & cat sur temps (dim=-2) et espace radial (dim=-1)
    # (B, 2, Nt, Nr) -> (B, 2, Nt, 2*Nr)
    P_wide = torch.cat([torch.flip(P_batch, dims=[-1]), P_batch], dim=-1)
    
    # (B, 2, 2*Nt, 2*Nr)
    P_tosend = torch.cat([torch.flip(P_wide, dims=[-2]), P_wide], dim=-2)

    return params_list, P_tosend