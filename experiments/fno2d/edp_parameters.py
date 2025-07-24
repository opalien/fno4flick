from __future__ import annotations

from collections.abc import Callable
import math

from torch import Tensor

class Normalizer:
    def __init__(self, mean: float, std: float):
        self.mean = mean
        self.std = std

    def normalize(self, value: float | Tensor) -> float | Tensor:
        return (value - self.mean) / self.std
    
    def unnormalize(self, value: float | Tensor) -> float | Tensor:
        return value * self.std + self.mean

    def __str__(self):
        return f"(mean={self.mean}, std={self.std})"

class EDPParameters:
    def __init__(self, 
                 Nr: int, Nt: int, 
                 R: float, r_max: float, 
                 t_max: float,
                 C_in: float, C_out: float,
                 D_in: float, D_out: float,
                 T1_in: float, T1_out: float,
                 P0_in: float, P0_out: float,
                 parent: EDPParameters | None = None,
                 parenthood_label: str | None = None,
    ):
        self.Nr = Nr
        self.Nt = Nt
        self.R = R
        self.r_max = r_max
        self.t_max = t_max
        self.C_in = C_in
        self.C_out = C_out
        self.D_in = D_in
        self.D_out = D_out
        self.T1_in = T1_in
        self.T1_out = T1_out
        self.P0_in = P0_in
        self.P0_out = P0_out

        self.parent = parent
        self.parenthood_label = parenthood_label
    
    
    def rescaling(self) -> "EDPParameters":
        t_max = 1.0  

        r_max = 1.0
        R = self.R * (1/self.r_max)

        C_in = self.C_in * (self.r_max**3)
        C_out = self.C_out * (self.r_max**3)
        
        D_in = self.D_in * (self.t_max/self.r_max**2)
        D_out = self.D_out * (self.t_max/self.r_max**2)

        T1_in = self.T1_in * (1/self.t_max)
        T1_out = self.T1_out * (1/self.t_max)

        P0_in = self.P0_in
        P0_out = self.P0_out        

        return EDPParameters(
            Nr=self.Nr, Nt=self.Nt,
            R=R, r_max=r_max, t_max=t_max,
            C_in=C_in, C_out=C_out,
            D_in=D_in, D_out=D_out,
            T1_in=T1_in, T1_out=T1_out,
            P0_in=P0_in, P0_out=P0_out,
            parent=self,
            parenthood_label="rescaling"
        )
    

    def nondimensionalize(self) -> "EDPParameters":
        
        C_out = 1.0
        C_in = self.C_in * (1/self.C_out)

        return EDPParameters(
            Nr=self.Nr, Nt=self.Nt,
            R=self.R, r_max=self.r_max, t_max=self.t_max,
            C_in=C_in, C_out=C_out,
            D_in=self.D_in, D_out=self.D_out,
            T1_in=self.T1_in, T1_out=self.T1_out,
            P0_in=self.P0_in, P0_out=self.P0_out,
            parent=self,
            parenthood_label="nondimensionalize"
        )


    def compression(self) -> "EDPParameters":
        def C_compress(C: float) -> float:
            return C #math.log(C + 1)
        
        def D_compress(D: float) -> float:
            return math.log(D)
        
        def R_compress(R: float) -> float:
            return R #math.log(R + 1)
        

        return EDPParameters(
            Nr=self.Nr, Nt=self.Nt,
            R=R_compress(self.R), r_max=self.r_max, 
            t_max=self.t_max,
            C_in=C_compress(self.C_in), C_out=C_compress(self.C_out),
            D_in=D_compress(self.D_in), D_out=D_compress(self.D_out),
            T1_in=self.T1_in, T1_out=self.T1_out,
            P0_in=self.P0_in, P0_out=self.P0_out,
            parent=self,
            parenthood_label="compression"
        )
    

    def normalize(self, C_normalizer: Callable[[float], float] | None = None, D_normalizer: Callable[[float], float] | None = None, R_normalizer: Callable[[float], float] | None = None, T1_normalizer: Callable[[float], float] | None = None) -> "EDPParameters":
        
        if C_normalizer is None:
            C_normalizer = lambda x: x
        if D_normalizer is None:
            D_normalizer = lambda x: x
        if R_normalizer is None:
            R_normalizer = lambda x: x
        if T1_normalizer is None:
            T1_normalizer = lambda x: x
        
        return EDPParameters(
            Nr=self.Nr, Nt=self.Nt,
            R=R_normalizer(self.R), r_max=self.r_max, 
            t_max=self.t_max,
            C_in=C_normalizer(self.C_in), C_out=C_normalizer(self.C_out),
            D_in=D_normalizer(self.D_in), D_out=D_normalizer(self.D_out),
            T1_in=self.T1_in, T1_out=self.T1_out,
            P0_in=self.P0_in, P0_out=self.P0_out,
            parent=self,
            parenthood_label="normalize"
        )

    def get_root_parent(self) -> EDPParameters:
        if self.parent is None:
            return self
        return self.parent.get_root_parent()

    
    def __str__(self):
        return f"EDPParameters(R={self.R}, r_max={self.r_max}, t_max={self.t_max}, C_in={self.C_in}, C_out={self.C_out}, D_in={self.D_in}, D_out={self.D_out}, T1_in={self.T1_in}, T1_out={self.T1_out}, P0_in={self.P0_in}, P0_out={self.P0_out}, parenthood_label={self.parenthood_label}, parent={self.parent})"

if __name__ == "__main__":
    params = EDPParameters(
        Nr=100, Nt=100,
        R=1.0, r_max=1.0, t_max=1.0,
        C_in=1.0, C_out=1.0,
        D_in=1.0, D_out=1.0,
        T1_in=1.0, T1_out=1.0,
        P0_in=1.0, P0_out=1.0
    )

    # basic usage
    params_to_learn = params.rescaling().nondimensionalize().compression().normalize()

    print(params_to_learn)