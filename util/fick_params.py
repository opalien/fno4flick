from __future__ import annotations
from collections.abc import Callable
import math

import torch
from torch import Tensor
import torch.nn as nn

class Normalizer:
    def __init__(self, mean: Tensor = torch.tensor(0.0), std: Tensor = torch.tensor(1.0)):
        self.mean, self.std = mean, std

    def normalize(self, value: Tensor) -> Tensor:
        return (value - self.mean) / self.std
    
    def unnormalize(self, value: Tensor) -> Tensor:
        return value * self.std + self.mean

    def __str__(self):
        return f"(mean={self.mean}, std={self.std})"

class FickParams:
    def __init__(self, 
                 Nt: Tensor,Nr: Tensor,
                 R: Tensor | nn.Parameter, r_max: Tensor,
                 t_max: Tensor,
                 C_in: Tensor, C_out: Tensor,
                 D_in: Tensor, D_out: Tensor,
                 T1_in: Tensor, T1_out: Tensor,
                 P0_in: Tensor, P0_out: Tensor,
                 parent: FickParams | None = None, parenthood: str | None = None,
                 ):
        self.Nt, self.Nr = Nt, Nr
        self.R, self.r_max = R, r_max
        self.t_max = t_max
        self.C_in, self.C_out = C_in, C_out
        self.D_in, self.D_out = D_in, D_out
        self.T1_in, self.T1_out = T1_in, T1_out
        self.P0_in, self.P0_out = P0_in, P0_out
        self.parent, self.parenthood = parent, parenthood


    @staticmethod
    def init_from(Nt: int,Nr: int,
                 R: float, r_max: float,
                 t_max: float,
                 C_in: float, C_out: float,
                 D_in: float, D_out: float,
                 T1_in: float, T1_out: float,
                 P0_in: float, P0_out: float,
                 parent: FickParams | None = None, parenthood: str | None = None,
                 ):
        return FickParams(
            Nt=torch.tensor(Nt),
            Nr=torch.tensor(Nr),
            R=torch.tensor(R),
            r_max=torch.tensor(r_max),
            t_max=torch.tensor(t_max),
            C_in=torch.tensor(C_in),
            C_out=torch.tensor(C_out),
            D_in=torch.tensor(D_in),
            D_out=torch.tensor(D_out),
            T1_in=torch.tensor(T1_in),
            T1_out=torch.tensor(T1_out),
            P0_in=torch.tensor(P0_in),
            P0_out=torch.tensor(P0_out),
            parent=parent,
            parenthood=parenthood)
    
    def to(self, device: torch.device) -> FickParams:
        return FickParams(
            Nr=self.Nr.to(device), Nt=self.Nt.to(device),
            R=self.R.to(device), r_max=self.r_max.to(device), t_max=self.t_max.to(device),
            C_in=self.C_in.to(device), C_out=self.C_out.to(device),
            D_in=self.D_in.to(device), D_out=self.D_out.to(device),
            T1_in=self.T1_in.to(device), T1_out=self.T1_out.to(device),
            P0_in=self.P0_in.to(device), P0_out=self.P0_out.to(device),
            parent=self,
            parenthood="device"
        )

    def rescaling(self) -> FickParams:
        t_max = torch.tensor(1.0)  

        r_max = torch.tensor(1.0)
        R = self.R * (1/self.r_max)

        C_in = self.C_in * (self.r_max**3)
        C_out = self.C_out * (self.r_max**3)
        
        D_in = self.D_in * (self.t_max/self.r_max**2)
        D_out = self.D_out * (self.t_max/self.r_max**2)

        T1_in = self.T1_in * (1/self.t_max)
        T1_out = self.T1_out * (1/self.t_max)

        P0_in = self.P0_in
        P0_out = self.P0_out        

        return FickParams(
            Nr=self.Nr, Nt=self.Nt,
            R=R, r_max=r_max, t_max=t_max,
            C_in=C_in, C_out=C_out,
            D_in=D_in, D_out=D_out,
            T1_in=T1_in, T1_out=T1_out,
            P0_in=P0_in, P0_out=P0_out,
            parent=self,
            parenthood="rescaling"
        )
    

    def nondimensionalize(self) -> FickParams:
        
        C_out = torch.tensor(1.0)
        C_in = self.C_in * (1/self.C_out)

        return FickParams(
            Nr=self.Nr, Nt=self.Nt,
            R=self.R, r_max=self.r_max, t_max=self.t_max,
            C_in=C_in, C_out=C_out,
            D_in=self.D_in, D_out=self.D_out,
            T1_in=self.T1_in, T1_out=self.T1_out,
            P0_in=self.P0_in, P0_out=self.P0_out,
            parent=self,
            parenthood="nondimensionalize"
        )


    def compression(self) -> FickParams:
        def C_compress(C: Tensor) -> Tensor:
            return C #math.log(C + 1)
        
        def D_compress(D: Tensor) -> Tensor:
            return torch.log(D)
        
        def R_compress(R: Tensor) -> Tensor:
            return R #math.log(R + 1)
        

        return FickParams(
            Nr=self.Nr, Nt=self.Nt,
            R=R_compress(self.R), r_max=self.r_max, 
            t_max=self.t_max,
            C_in=C_compress(self.C_in), C_out=C_compress(self.C_out),
            D_in=D_compress(self.D_in), D_out=D_compress(self.D_out),
            T1_in=self.T1_in, T1_out=self.T1_out,
            P0_in=self.P0_in, P0_out=self.P0_out,
            parent=self,
            parenthood="compression"
        )
    

    def normalize(self, C_normalizer: Callable[[Tensor], Tensor] | None = None, D_normalizer: Callable[[Tensor], Tensor] | None = None, R_normalizer: Callable[[Tensor], Tensor] | None = None, T1_normalizer: Callable[[Tensor], Tensor] | None = None) -> FickParams:
        
        if C_normalizer is None:
            C_normalizer = lambda x: x
        if D_normalizer is None:
            D_normalizer = lambda x: x
        if R_normalizer is None:
            R_normalizer = lambda x: x
        if T1_normalizer is None:
            T1_normalizer = lambda x: x
        
        return FickParams(
            Nr=self.Nr, Nt=self.Nt,
            R=R_normalizer(self.R), r_max=self.r_max, 
            t_max=self.t_max,
            C_in=C_normalizer(self.C_in), C_out=C_normalizer(self.C_out),
            D_in=D_normalizer(self.D_in), D_out=D_normalizer(self.D_out),
            T1_in=self.T1_in, T1_out=self.T1_out,
            P0_in=self.P0_in, P0_out=self.P0_out,
            parent=self,
            parenthood="normalize"
        )


    def get_root_parent(self) -> FickParams:
        if self.parent is None:
            return self
        return self.parent.get_root_parent()


    def __str__(self):
        return f"FickParams(R={self.R}, r_max={self.r_max}, t_max={self.t_max}, C_in={self.C_in}, C_out={self.C_out}, D_in={self.D_in}, D_out={self.D_out}, T1_in={self.T1_in}, T1_out={self.T1_out}, P0_in={self.P0_in}, P0_out={self.P0_out}, parenthood={self.parenthood}, parent={self.parent})"
