# TODOimport os
import json
import torch
from torch import Tensor
from torch.utils.data import Dataset as TorchDataset
import os

class Dataset(TorchDataset):
    """
    Élément = (P_periodic, R) où
        P_periodic : Tensor (1, 2·Nt, 2·(Nr+1))
        R          : Tensor (1,)  – rayon de la bille
    """
    def __init__(self):
        super().__init__()
        self.elements: list[tuple[Tensor, Tensor]] = []

    # -------- construction d’un élément --------------------------------------------------
    def add_element(self, data_dict: dict):
        P: Tensor = data_dict["P"]              # (Nt, Nr+1)
        Nt, Nr_plus1 = P.shape

        # -- périodisation spatiale puis temporelle –
        P_spatial = torch.cat((torch.flip(P, (-1,)), P), dim=-1)     # (Nt, 2·(Nr+1))
        P_periodic = torch.cat((torch.flip(P_spatial, (0,)), P_spatial), dim=0)  # (2·Nt, ...)

        P_periodic = P_periodic.unsqueeze(0)    # canal = 1   => (1, 2·Nt, 2·(Nr+1))
        R_tensor = torch.tensor([float(data_dict["R"])], dtype=torch.float32)

        self.elements.append((P_periodic, R_tensor))

    # ------------------------------------------------------------------------------------
    def __len__(self):  return len(self.elements)
    def __getitem__(self, idx):  return self.elements[idx]

    # -------- lecture depuis un dossier JSON --------------------------------------------
    def load(self, folder: str):
        for fname in os.listdir(folder):
            if not fname.endswith(".json"):
                continue
            with open(os.path.join(folder, fname), "r") as f:
                data = json.load(f)
            data["P"] = torch.tensor(data["P"], dtype=torch.float32)
            self.add_element(data)
