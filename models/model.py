from typing import Tuple

import torch
from torch import Tensor
import torch.nn as nn

from neuralop.models import FNO

from util.fick_params import FickParams, Normalizer
from util.G_method import compute_G_in, compute_G_out, G_error





class FickModel(nn.Module):
    def __init__(self, n_modes: Tuple[int], n_layers: int, hidden_channels: int, device: torch.device):
        super().__init__() # type: ignore


        # MODEL
        self.fno = FNO(n_modes=n_modes, in_channels=5, out_channels=2, n_layers=n_layers, hidden_channels=hidden_channels)

        self.n_modes = n_modes
        self.n_layers = n_layers
        self.hidden_channels = hidden_channels
        self.device = device
        # DATA
        self.P_normalizer: Normalizer = Normalizer()
        self.C_normalizer: Normalizer = Normalizer()
        self.D_normalizer: Normalizer = Normalizer()
        self.R_normalizer: Normalizer = Normalizer()
        self.T1_normalizer: Normalizer = Normalizer()
        self.P0_normalizer: Normalizer = Normalizer()



    def preprocess(self, list_params: list[FickParams], Nt: int, Nr: int) -> Tensor:
        for i, params in enumerate(list_params):
            list_params[i] = params.get_root_parent().rescaling().nondimensionalize().compression().normalize(C_normalizer=self.C_normalizer.normalize, D_normalizer=self.D_normalizer.normalize, R_normalizer=self.R_normalizer.normalize, T1_normalizer=self.T1_normalizer.normalize)

        PARAMS = torch.empty(len(list_params), 5, 2*Nt, 2*Nr, device=self.device)

        for i, params in enumerate(list_params):
            s = (2*Nt, 2*Nr)
            PARAMS[i, 0, :, :] = params.C_in.to(self.device).expand(s)
            PARAMS[i, 1, :, :] = params.D_in.to(self.device).expand(s)
            PARAMS[i, 2, :, :] = params.D_out.to(self.device).expand(s)
            PARAMS[i, 3, :, :] = params.T1_in.to(self.device).expand(s)
            PARAMS[i, 4, :, :] = params.R.to(self.device).expand(s)

        return PARAMS


    def forward(self, params: list[FickParams] | FickParams) -> Tensor:


        

        if isinstance(params, FickParams):
            params = [params]

        Nt, Nr = int(params[0].Nt.item()), int(params[0].Nr.item())
        

        PARAMS = self.preprocess(params, Nt, Nr)

        fick_fno = self.fno(PARAMS)

        return fick_fno




    def search_R_in(self, list_params: list[FickParams], G_true: Tensor, Nt: int, Nr: int) -> list[Tensor]:
        
        optim_params_processed = []
        trainable_R_list = []

        # Prépare une liste de paramètres spécifiquement pour cette optimisation
        for p in list_params:
            # 1. Applique le même prétraitement que la méthode `preprocess`
            p_proc = p.get_root_parent().rescaling().nondimensionalize().compression().normalize(
                C_normalizer=self.C_normalizer.normalize, 
                D_normalizer=self.D_normalizer.normalize, 
                R_normalizer=self.R_normalizer.normalize, 
                T1_normalizer=self.T1_normalizer.normalize
            )
            # 2. Remplace R par un paramètre optimisable
            initial_R = p_proc.R.detach().clone()
            p_proc.R = nn.Parameter(p_proc.R.detach().clone().to(self.device))
            optim_params_processed.append(p_proc)
            trainable_R_list.append(p_proc.R)

        optimizer = torch.optim.Adam(trainable_R_list, lr=1e-3)

        for _ in range(100):
            optimizer.zero_grad()

            # 3. Construit le tenseur d'entrée directement, sans appeler self.forward
            s = (2 * Nt, 2 * Nr)
            PARAMS = torch.empty(len(optim_params_processed), 5, *s, device=self.device)
            for i, params in enumerate(optim_params_processed):
                PARAMS[i, 0] = params.C_in.to(self.device).expand(s)
                PARAMS[i, 1] = params.D_in.to(self.device).expand(s)
                PARAMS[i, 2] = params.D_out.to(self.device).expand(s)
                PARAMS[i, 3] = params.T1_in.to(self.device).expand(s)
                PARAMS[i, 4] = params.R.to(self.device).expand(s) # Utilise le nn.Parameter

            # 4. Appelle fno directement
            fick_fno = self.fno(PARAMS)
            fick = postprocess(fick_fno)

            G_pred = compute_G_in(fick)
            loss = G_error(G_pred, G_true)

            loss.backward()
            optimizer.step()
            # print(loss) # Le print est commenté pour éviter la sortie massive

        return [p.R.detach().clone() for p in optim_params_processed]


    def search_R_out(self, list_params: list[FickParams], G_true: Tensor, Nt: int, Nr: int) -> list[Tensor]:
        
        optim_params_processed = []
        trainable_R_list = []

        # Prépare une liste de paramètres spécifiquement pour cette optimisation
        for p in list_params:
            # 1. Applique le même prétraitement
            p_proc = p.get_root_parent().rescaling().nondimensionalize().compression().normalize(
                C_normalizer=self.C_normalizer.normalize, 
                D_normalizer=self.D_normalizer.normalize, 
                R_normalizer=self.R_normalizer.normalize, 
                T1_normalizer=self.T1_normalizer.normalize
            )
            # 2. Remplace R par un paramètre optimisable
            initial_R = p_proc.R.detach().clone()
            p_proc.R = nn.Parameter(p_proc.R.detach().clone().to(self.device))
            optim_params_processed.append(p_proc)
            trainable_R_list.append(p_proc.R)

        optimizer = torch.optim.Adam(trainable_R_list, lr=1e-3)
        n_steps, eps = 100, 1e-4

        for _ in range(n_steps):
            optimizer.zero_grad()

            # 3. Construit le tenseur d'entrée directement
            s = (2 * Nt, 2 * Nr)
            PARAMS = torch.empty(len(optim_params_processed), 5, *s, device=self.device)
            for i, params in enumerate(optim_params_processed):
                PARAMS[i, 0] = params.C_in.to(self.device).expand(s)
                PARAMS[i, 1] = params.D_in.to(self.device).expand(s)
                PARAMS[i, 2] = params.D_out.to(self.device).expand(s)
                PARAMS[i, 3] = params.T1_in.to(self.device).expand(s)
                PARAMS[i, 4] = params.R.to(self.device).expand(s)

            # 4. Appelle fno directement
            fick_fno = self.fno(PARAMS)
            fick = postprocess(fick_fno)

            G_pred = compute_G_out(fick, optim_params_processed) # Important: passer les params traités
            loss = G_error(G_pred, G_true)
            
            loss.backward()
            optimizer.step()

            # Assure que R reste dans des limites valides
            with torch.no_grad():
                for p_proc in optim_params_processed:
                    # r_max vient des paramètres traités
                    rmax = p_proc.r_max 
                    p_proc.R.data.clamp_(min=eps, max=float(rmax) - eps)

        return [p.R.detach().clone() for p in optim_params_processed]



    
#
#
#    def search_R_in(self, list_params: list[FickParams], G_true: Tensor, Nt: int, Nr: int) -> list[Tensor]:
#
#        for _, params in enumerate(list_params):
#            params.R = torch.nn.Parameter(params.r_max/2.).to(self.device)
#
#        optimizer = torch.optim.Adam([params.R for params in list_params], lr=1e-3)
#
#        for _ in range(100):
#            optimizer.zero_grad()
#
#            fick_fno = self.forward(list_params)
#            fick = postprocess(fick_fno)
#
#            G_pred = compute_G_in(fick)
#            loss = G_error(G_pred, G_true)
#
#            loss.backward()
#            print(loss)
#            optimizer.step()
#
#
#        return [params.R.detach().clone() for params in list_params]
#
#
#
#    def search_R_out(self, list_params: list[FickParams], G_true: Tensor, Nt: int, Nr: int) -> list[Tensor]:
#        trainable_R = []
#        for p in list_params:
#            R0 = (p.r_max / 2.0).to(self.device) if isinstance(p.r_max, torch.Tensor) else torch.tensor(float(p.r_max)/2.0, device=self.device)
#            p.R = torch.nn.Parameter(R0)
#            trainable_R.append(p.R)
#
#        optimizer = torch.optim.Adam(trainable_R, lr=1e-3)
#        n_steps, eps = 100, 1e-4
#
#        for _ in range(n_steps):
#            optimizer.zero_grad()
#            fick_fno = self.forward(list_params)
#            fick = postprocess(fick_fno)
#            G_pred = compute_G_out(fick, list_params)
#            loss = G_error(G_pred, G_true)
#            loss.backward()
#            optimizer.step()
#            with torch.no_grad():
#                for p in list_params:
#                    rmax = p.r_max if isinstance(p.r_max, torch.Tensor) else torch.tensor(float(p.r_max), device=p.R.device, dtype=p.R.dtype)
#                    p.R.data.clamp_(min=eps, max=float(rmax.item()) - eps)
#
#        return [p.R.detach().clone() for p in list_params]
#



def postprocess(fick_fno: Tensor) -> Tensor:
        Nt, Nr = fick_fno.shape[2] // 2, fick_fno.shape[3] // 2
        
        # Périodisation par moyenne des 4 quadrants
        q1 = fick_fno[:, :, Nt:, Nr:]          # Bottom-Right
        q2 = torch.flip(fick_fno[:, :, Nt:, :Nr], dims=[-1]) # Flipped Bottom-Left
        q3 = torch.flip(fick_fno[:, :, :Nt, Nr:], dims=[-2]) # Flipped Top-Right
        q4 = torch.flip(fick_fno[:, :, :Nt, :Nr], dims=[-1, -2]) # Flipped Top-Left
        
        fick = (q1 + q2 + q3 + q4) / 4.0
        
        # Appliquez la dé-normalisation si nécessaire (ici commentée comme dans votre code)
        # return self.P_normalizer.unnormalize(fick)
        return fick



#def postprocess(fick_fno: Tensor) -> Tensor:
#    Nt, Nr = fick_fno.shape[2] // 2, fick_fno.shape[3] // 2
#    
#    q1 = fick_fno[:, :, Nt:, Nr:]
#    q2 = torch.flip(fick_fno[:, :, Nt, :Nr], dims=[-1])
#    q3 = torch.flip(fick_fno[:, :, :Nt, Nr:], dims=[-2])
#    q4 = torch.flip(fick_fno[:, :, :Nt, :Nr], dims=[-1, -2])
#    
#    fick = (q1 + q2 + q3 + q4) / 4
#    
#    return fick #self.P_normalizer.unnormalize(fick)
        

#def postprocess(fick_fno: Tensor) -> Tensor:
#    Nt, Nr = fick_fno.shape[2] // 2, fick_fno.shape[3] // 2
#
#    q1 = fick_fno[:, :, Nt:,  Nr:]          # BR
#    q2 = fick_fno[:, :, Nt:,  :Nr]          # BL
#    q3 = fick_fno[:, :, :Nt,  Nr:]          # TR
#    q4 = fick_fno[:, :, :Nt,  :Nr]          # TL
#
#    return (q1 + torch.flip(q2, [-1]) + torch.flip(q3, [-2]) + torch.flip(q4, [-1, -2])) / 4

    



        




