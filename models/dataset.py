from __future__ import annotations

import os
import math 

from matplotlib import pyplot as plt
import torch
from torch import Tensor

from util.fick_params import FickParams, Normalizer

import torch.nn.functional as F


class FickDataset(torch.utils.data.Dataset):
    
    def __init__(self):
        super().__init__()

        self.elements: list[tuple[FickParams, Tensor]] = []

        self.P_normalizer = Normalizer()
        self.C_normalizer = Normalizer()
        self.D_normalizer = Normalizer()
        self.R_normalizer = Normalizer()
        self.T1_normalizer = Normalizer()
        self.P0_normalizer = Normalizer()

    
    def add_element(self, d: dict[str, float | int | Tensor]):
        P= d["P"].clone() if isinstance(d["P"], Tensor) else torch.tensor(d["P"])

        Nr, Nt = int(d["Nr"]), int(d["Nt"])
        R, r_max = float(d["R"]), float(d["r_max"])
        t_max = float(d["Tfinal"])
        C_in, C_out = float(d["C_in"]), float(d["C_out"])
        D_in, D_out = float(d["D_in"]), float(d["D_out"])
        T1_in, T1_out = float(d["T1_in"]), float(d["T1_out"])
        P0_in, P0_out = float(d["P0_in"]), float(d["P0_out"])

        params = FickParams.init_from(Nt, Nr, R, r_max, t_max, C_in, C_out, D_in, D_out, T1_in, T1_out, P0_in, P0_out)

        self.elements.append((params, P))


    def __len__(self):
        return len(self.elements)
    
    def __getitem__(self, idx: int):
        return self.elements[idx]
    
    
    def load(self, path: str):
        for file in os.listdir(path):
            if file.endswith(".pt"):
                self.add_element(torch.load(os.path.join(path, file), weights_only=False))

        
    def set_normalizers(self, dataset: FickDataset):
        self.P_normalizer = Normalizer(mean=dataset.P_normalizer.mean, std=dataset.P_normalizer.std)
        self.C_normalizer = Normalizer(mean=dataset.C_normalizer.mean, std=dataset.C_normalizer.std)
        self.D_normalizer = Normalizer(mean=dataset.D_normalizer.mean, std=dataset.D_normalizer.std)
        self.R_normalizer = Normalizer(mean=dataset.R_normalizer.mean, std=dataset.R_normalizer.std)
        self.T1_normalizer = Normalizer(mean=dataset.T1_normalizer.mean, std=dataset.T1_normalizer.std)


    def get_normalizers(self):

        
        acc = {"C":0.,"D":0.,"T1":0., "P0":0., "R":0.}
        acc_sq = {"C":0.,"D":0.,"T1":0., "P0":0., "R":0.}
        P_sum = P_sq = 0.0
        P_count = 0

        for p, P in self.elements:

            p = p.get_root_parent().rescaling().nondimensionalize().compression()
            li, lo = p.R, p.r_max - p.R

            acc_C_avg = p.C_in * li + p.C_out * lo
            acc_D_avg = p.D_in * li + p.D_out * lo
            acc_T1_avg = p.T1_in * li + p.T1_out * lo
            acc_P0_avg = p.P0_in * li + p.P0_out * lo

            acc["R"] += p.R
            acc["C"]  += acc_C_avg
            acc["D"]  += acc_D_avg
            acc["T1"] += acc_T1_avg
            acc["P0"] += acc_P0_avg

            acc_sq["R"] += p.R**2
            acc_sq["C"]  += acc_C_avg**2
            acc_sq["D"]  += acc_D_avg**2
            acc_sq["T1"] += acc_T1_avg**2
            acc_sq["P0"] += acc_P0_avg**2

            P_sum += P.sum().item()
            P_sq  += (P*P).sum().item()
            P_count += P.numel()

            

        self.C_normalizer.mean = acc["C"]  / len(self.elements)
        self.D_normalizer.mean = acc["D"]  / len(self.elements)
        self.T1_normalizer.mean = acc["T1"] / len(self.elements)
        self.P0_normalizer.mean = acc["P0"] / len(self.elements)
        #self.R_normalizer.mean = acc["R"]  / len(self.elements)

        self.C_normalizer.std = math.sqrt(max(acc_sq["C"]/(len(self.elements)) - self.C_normalizer.mean**2, 1e-8))
        self.D_normalizer.std = math.sqrt(max(acc_sq["D"]/(len(self.elements)) - self.D_normalizer.mean**2, 1e-8))
        self.T1_normalizer.std = math.sqrt(max(acc_sq["T1"]/(len(self.elements)) - self.T1_normalizer.mean**2, 1e-8))
        self.P0_normalizer.std = math.sqrt(max(acc_sq["P0"]/(len(self.elements)) - self.P0_normalizer.mean**2, 1e-8))
        #self.R_normalizer.std = math.sqrt(max(acc_sq["R"]/(len(self.elements)) - self.R_normalizer.mean**2, 1e-8))

        self.P_normalizer.mean = P_sum / P_count
        self.P_normalizer.std = math.sqrt(max(P_sq / P_count - self.P_normalizer.mean**2, 1e-8))


        return self.P_normalizer, self.C_normalizer, self.D_normalizer, self.R_normalizer, self.T1_normalizer, self.P0_normalizer

        
    def apply_P_normalizer(self):
        for i, (p, P) in enumerate(self.elements):
            P = self.P_normalizer.normalize(P)
            self.elements[i] = (p, P)


    def deapply_P_normalizer(self):
        for i, (p, P) in enumerate(self.elements):
            P = self.P_normalizer.unnormalize(P)
            self.elements[i] = (p, P)


    def plot_element(self, idx: int):
        params, P = self.elements[idx]
        params = params.get_root_parent().rescaling().nondimensionalize().compression()

        params,  P = collate_fn([(params, P)])

        title = f"Nt={params.Nt}, Nr={params.Nr}, R={params.R}, r_max={params.r_max}, t_max={params.t_max}, C_in={params.C_in}, C_out={params.C_out}, D_in={params.D_in}, D_out={params.D_out}, T1_in={params.T1_in}, T1_out={params.T1_out}, P0_in={params.P0_in}, P0_out={params.P0_out}"

        plt.imshow(P.squeeze(0), cmap="viridis")
        plt.colorbar()
        plt.title(title)
        plt.xlabel("r")
        plt.ylabel("t")
        plt.savefig(f"plots/P_{idx}.png")
        plt.close()


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
                p.normalize(self.C_normalizer.normalize, 
                            self.D_normalizer.normalize, 
                            self.R_normalizer.normalize, 
                            self.T1_normalizer.normalize 
                            )
                name = "normalized"

            list_params.append(p)

        
        fig, axs = plt.subplots(2, 3, figsize=(16, 10))

        axs[0, 0].hist([p.C_in for p in list_params], bins=100, color="blue")
        axs[0, 0].set_title("C_in")
        axs[0, 0].set_xlabel("C_in Value")
        axs[0, 0].set_ylabel("Frequency")

        axs[1, 0].hist([p.C_out for p in list_params], bins=100, color="red")
        axs[1, 0].set_title("C_out")
        axs[1, 0].set_xlabel("C_out Value")
        axs[1, 0].set_ylabel("Frequency")
        
        axs[0, 1].hist([p.D_in for p in list_params], bins=100, color="green")
        axs[0, 1].set_title("D_in")
        axs[0, 1].set_xlabel("D_in Value")
        axs[0, 1].set_ylabel("Frequency")
        
        axs[1, 1].hist([p.D_out for p in list_params], bins=100, color="orange")
        axs[1, 1].set_title("D_out")
        axs[1, 1].set_xlabel("D_out Value")
        axs[1, 1].set_ylabel("Frequency")

        axs[0, 2].hist([p.T1_in for p in list_params], bins=100, color="purple")
        axs[0, 2].set_title("T1_in")
        axs[0, 2].set_xlabel("T1_in Value")
        axs[0, 2].set_ylabel("Frequency")
        
        axs[1, 2].hist([p.T1_out for p in list_params], bins=100, color="brown")
        axs[1, 2].set_title("T1_out")
        axs[1, 2].set_xlabel("T1_out Value")
        axs[1, 2].set_ylabel("Frequency")

        plt.tight_layout()
        plt.savefig(os.path.join(path, f"{name}.png"))
        plt.close(fig)











    def get_dataloader(self, bs:int, shuffle:bool=True):
        return torch.utils.data.DataLoader(
            self, batch_size=bs, shuffle=shuffle, collate_fn=collate_fn
        )
    



#def collate_fn(batch: list[tuple[FickParams, Tensor]]) -> tuple[list[FickParams], list[Tensor]]:
#
#    B = len(batch)
#
#    params = [p for p, _ in batch]
#    P = [P for _, P in batch]
#
#    Nt = params[0].Nt
#    Nr = params[0].Nr
#
#    P_out = torch.zeros((B, 2*Nt, 2*Nr))
#
#    for i, (p, P) in enumerate(batch):
#        P_out[i, Nt:, Nr:] = P
#    
#    P_out[:, Nt:, :Nr] = torch.flip(P_out[:, Nt:, Nr:], dims=(-1,))
#    P_out[:, :Nt, Nr:] = torch.flip(P_out[:, Nt:, :], dims=(0,))
#
#
#    return params, P_out



def collate_fn(batch: list[tuple[FickParams, Tensor]]) -> tuple[list[FickParams], Tensor]:
    """
    Entrée :
      - batch = list[(FickParams, P)], P de taille (Nt, Nr_tot) sur r∈[0, r_max]
    Sortie :
      - params: liste des FickParams (inchangés)
      - P_tosend: (B, 2, 2*Nt, 2*Nr) = [P_in, P_out] périodisés par flip&cat
    Hypothèses :
      - Grille radiale source uniforme sur [0, r_max]
      - Séparation in/out au rayon R ; chaque canal est réinterpolé sur Nr points.
    """
    B = len(batch)
    assert B > 0, "batch vide"

    params: list[FickParams] = [p for p, _ in batch]

    # On suppose que tous les éléments du batch partagent Nt et Nr cibles
    Nt = int(params[0].Nt.item() if isinstance(params[0].Nt, torch.Tensor) else params[0].Nt)
    Nr = int(params[0].Nr.item() if isinstance(params[0].Nr, torch.Tensor) else params[0].Nr)

    # Conteneur de sortie (B, 2, Nt, Nr) avant périodisation
    P_2c = torch.empty((B, 2, Nt, Nr), dtype=torch.float32)

    for i, (p, P) in enumerate(batch):
        # P: (Nt_src, Nr_src) — on tolère Nt_src==Nt, sinon on recopie tel quel (pas de resampling temporel)
        P = P.detach().to(dtype=torch.float32)
        Nt_src, Nr_src = P.shape

        # R, r_max (scalaires flottants)
        R = float(p.R.item() if isinstance(p.R, torch.Tensor) else p.R)
        r_max = float(p.r_max.item() if isinstance(p.r_max, torch.Tensor) else p.r_max)

        # index de coupure "in/out" dans la grille source
        # (on borne entre 1 et Nr_src-1 pour éviter des segments vides)
        cut = int(round(R / max(r_max, 1e-12) * Nr_src))
        cut = max(1, min(cut, Nr_src - 1))

        # Segments radiaux source
        Pin_src  = P[:, :cut]        # (Nt_src, Nr_in_src)
        Pout_src = P[:, cut:]        # (Nt_src, Nr_out_src)

        # On veut (Nt, Nr) pour chaque canal. On n'interpole qu'en r.
        # Astuce : utiliser F.interpolate en 4D avec H=Nt invariant.
        # Si Nt_src != Nt, on tronque/pad temporellement sans interpolation (au besoin).
        def _match_time(Q: Tensor, Nt_target: int) -> Tensor:
            Nt_q = Q.shape[0]
            if Nt_q == Nt_target:
                return Q
            if Nt_q > Nt_target:
                return Q[:Nt_target, :]
            # padding par répétition de la dernière ligne
            pad_rows = Nt_target - Nt_q
            last = Q[-1:, :].expand(pad_rows, Q.shape[1])
            return torch.cat([Q, last], dim=0)

        Pin_src  = _match_time(Pin_src,  Nt)
        Pout_src = _match_time(Pout_src, Nt)

        # (Nt, Nr_seg) -> (1,1,Nt,Nr_seg) pour interpolation bilinéaire (seulement en W)
        def _resize_radial(Q: Tensor, Nr_target: int) -> Tensor:
            Q4 = Q.unsqueeze(0).unsqueeze(0)          # (1,1,Nt,Nr_seg)
            Q4r = F.interpolate(Q4, size=(Nt, Nr_target), mode="bilinear", align_corners=True)
            return Q4r.squeeze(0).squeeze(0)          # (Nt, Nr_target)

        Pin  = _resize_radial(Pin_src,  Nr)
        Pout = _resize_radial(Pout_src, Nr)

        P_2c[i, 0] = Pin
        P_2c[i, 1] = Pout

    # Périodisation par flip & cat sur temps (dim=-2) et espace radial (dim=-1)
    # (B,2,Nt,Nr) -> (B,2,Nt,2Nr)
    P_wide = torch.cat([torch.flip(P_2c, dims=[-1]), P_2c], dim=-1)
    # (B,2,2Nt,2Nr)
    P_tosend = torch.cat([torch.flip(P_wide, dims=[-2]), P_wide], dim=-2)

    return params, P_tosend

