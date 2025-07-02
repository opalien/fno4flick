import torch
from torch import Tensor
import matplotlib.pyplot as plt
import os
import json

class Dataset(torch.utils.data.Dataset):
    def __init__(self):
        super().__init__()
        self.elements: list[tuple[Tensor, Tensor]] = []

    def add_element(self, data_dict: dict[str, Tensor | float | int]):
        P = data_dict["P"]
        # Make P spatially periodic
        P_spatial_periodic = torch.cat((torch.flip(P, dims=(-1,)), P), dim=-1)
        # Make P also temporally periodic
        P_periodic = torch.cat((torch.flip(P_spatial_periodic, dims=(0,)), P_spatial_periodic), dim=0)


        Nr: int = data_dict["Nr"]
        Nt: int = data_dict["Nt"]
        R: float = data_dict["R"]
        cuve_width: float = data_dict["cuve_width"]
        C_ball: float = data_dict["C_ball"]
        D_ball: float = data_dict["D_ball"]
        T_re_ball: float = data_dict["Tre_ball"]
        C_cuve: float = data_dict["C_cuve"]
        D_cuve: float = data_dict["D_cuve"]
        T_re_cuve: float = data_dict["Tre_cuve"]

        num_spatial_points = Nr + 1
        
        params = torch.zeros(Nt, num_spatial_points, 3)

        n_ball_points = int(num_spatial_points * R / cuve_width)

        params[:, :n_ball_points, 0] = C_ball
        params[:, :n_ball_points, 1] = D_ball
        params[:, :n_ball_points, 2] = T_re_ball

        params[:, n_ball_points:, 0] = C_cuve
        params[:, n_ball_points:, 1] = D_cuve
        params[:, n_ball_points:, 2] = T_re_cuve
        
        # Make params spatially periodic
        params_spatial_periodic = torch.cat((torch.flip(params, dims=(1,)), params), dim=1)
        # Make params also temporally periodic
        params_periodic = torch.cat((torch.flip(params_spatial_periodic, dims=(0,)), params_spatial_periodic), dim=0)

        self.elements.append((params_periodic.permute(2, 0, 1), P_periodic))   

    def __len__(self):
        return len(self.elements)

    def __getitem__(self, idx: int):
        return self.elements[idx]
    
    def plot_element(self, idx: int):
        if idx >= len(self.elements):
            print(f"Error: Index {idx} is out of bounds for dataset of size {len(self.elements)}.")
            return

        params, P = self.__getitem__(idx)
        
        params_np = params.cpu().numpy()
        P_np = P.cpu().numpy()

        fig, axs = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f"Visualisation de l'élément {idx} du Dataset", fontsize=16)

        titles = ["Paramètre C (Input)", "Paramètre D (Input)", "Paramètre T_re (Input)", "Solution P (Target)"]
        data_to_plot = [params_np[0, :, :], params_np[1, :, :], params_np[2, :, :], P_np]
        
        for i, ax in enumerate(axs.flat):
            im = ax.imshow(data_to_plot[i], origin='lower', aspect='auto', cmap='viridis')
            ax.set_title(titles[i])
            ax.set_xlabel("Coordonnée Spatiale Périodique")
            ax.set_ylabel("Pas de Temps")
            fig.colorbar(im, ax=ax)

        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.savefig(f"element_{idx}_visualization.png")
        plt.show()



    def load(self, file_path: str):
        for json_file in os.listdir(file_path):
            if json_file.endswith(".json"):
                with open(os.path.join(file_path, json_file), "r") as f:
                    data = json.load(f)
                
                    data["P"] = torch.tensor(data["P"])

                    self.add_element(data)