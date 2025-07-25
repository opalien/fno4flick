import os
from utils_.data_generator import DataGenerator
from utils_.convert import C_converter, R_converter, D_converter

from typing import Any
import random as rd
import numpy as np
import math
import torch

def empty_database(folder: str):
    if os.path.exists(folder):
        for sub_folder in ["train", "test", "dev", "plot"]:
            sub_folder_path = os.path.join(folder, sub_folder)
            if os.path.exists(sub_folder_path):
                for file in os.listdir(sub_folder_path):
                    os.remove(os.path.join(sub_folder_path, file))
            else:
                os.makedirs(sub_folder_path)
    else:
        os.makedirs(folder)
        os.makedirs(os.path.join(folder, "train"))
        os.makedirs(os.path.join(folder, "test"))
        os.makedirs(os.path.join(folder, "dev"))
        os.makedirs(os.path.join(folder, "plot"))


def extract_params_from_brut(folder: str):
    list_params: list[dict[Any, Any]] = []
    for sub_folder in os.listdir(folder):

        if not os.path.isdir(os.path.join(folder, sub_folder)):
            continue

        try:
            C_out, C_in, T1_in, R_nm = map(float, sub_folder.split('_'))
        except ValueError:
            print(f"Skipping folder {sub_folder} due to ValueError")
            continue


        for file in os.listdir(os.path.join(folder, sub_folder)):

            P0, TB, Time, P = np.loadtxt(
                os.path.join(folder, sub_folder, file),
                comments='%',          # saute toutes les lignes qui commencent par %
                unpack=True            # renvoie 4 tableaux séparés
            )

            P0 = float(P0[0])
            TB = float(TB[0])


            # Convertir les valeurs
            C_out = C_converter(C_out)  # C_out from mol.L^-1 to Fmol.µm^3
            C_in = C_converter(C_in)  # C_in from mol.L^-1 to Fmol.µm^3

            R = R_converter(R_nm)  # R in µm

            D_ref = D_converter(500.)  # D_ref = 500 nm^2.s^-1
            C_ref = C_converter(60.)   # C_ref = 60 mol.L^-1      
            D_in = D_ref * (C_in / C_ref) ** (1/3)
            D_out = D_ref * (C_out/ C_ref) ** (1/3)


            T1_out = TB

            list_params.append({
                "C_out": C_out,
                "C_in": C_in,
                "D_out": D_out,
                "D_in": D_in,
                "T1_out": T1_out,
                "T1_in": T1_in,
                "P0_out": P0,
                "P0_in": 1., # car Boltzmann est 1
                "R": R, 
            })

    return list_params


def create_database(list_dict: list[dict[Any, Any]], folder: str, n=20, micro_ondes: bool= True, test: float = 0.1, dev: float = 0.1):

    R_max = max([params["R"] for params in list_dict]) # typ: ignore
    R_min = min([params["R"] for params in list_dict]) # typ: ignore

    C_out_max = max([params["C_out"] for params in list_dict]) # typ: ignore
    C_out_min = min([params["C_out"] for params in list_dict]) # typ: ignore
    C_in_max = max([params["C_in"] for params in list_dict]) # typ: ignore
    C_in_min = min([params["C_in"] for params in list_dict]) # typ: ignore

    D_out_max = max([params["D_out"] for params in list_dict]) # typ: ignore
    D_out_min = min([params["D_out"] for params in list_dict]) # typ: ignore
    D_in_max = max([params["D_in"] for params in list_dict]) # typ: ignore
    D_in_min = min([params["D_in"] for params in list_dict]) # typ: ignore

    T1_in_max = max([params["T1_in"] for params in list_dict]) # typ: ignore
    T1_in_min = min([params["T1_in"] for params in list_dict]) # typ: ignore

    #P0_out_max = max([params["P0_out"] for params in list_dict])
    #P0_out_min = min([params["P0_out"] for params in list_dict])
    #T1_out_max = max([params["T1_out"] for params in list_dict])
    #T1_out_min = min([params["T1_out"] for params in list_dict])
    #P0_in_max = max([params["P0_in"] for params in list_dict])
    #P0_in_min = min([params["P0_in"] for params in list_dict])


    if micro_ondes:
        P0_out = 100
        T1_out = 3.
    
    else:
        P0_out = 0.5
        T1_out = 3.5


    print(f"""{R_min} < R < {R_max} 
            {C_in_min} < C_in < {C_in_max:2f}, {C_out_min} < C_out < {C_out_max:2f} 
            {D_in_min:2f} < D_in < {D_in_max:2f}, {D_out_min:2f} < D_out < {D_out_max:2f} 
            {T1_in_min:2f} < T1_in < {T1_in_max:2f}, T1_out = {T1_out:2f} 
            P0_in = 1, P0_out = {P0_out:2f} 
    """)

    


    for _ in range(n):
        #R =  rd.uniform(0.001, 0.5) 
        #r_max = math.exp(rd.uniform(math.log(R_min), math.log(R_max)*5))
        #R = rd.uniform(r_max*0.05, r_max*0.9)
        #rd.uniform(R_min, R_max)
        #math.exp(rd.uniform(math.log(R_min), math.log(R_max)))
        
        R = math.exp(rd.uniform(math.log(R_min), math.log(R_max)))
        r_max = 4*R #rd.uniform(R*1.5, R*10.)

        #r_max = 1.
        #R = rd.uniform(0.1, 0.8)

        dg = DataGenerator(
            R=R,
            r_max=r_max,
            C_in=rd.uniform(C_in_min, C_in_max),
            C_out=rd.uniform(C_out_min, C_out_max),
            D_in=rd.uniform(D_in_min, D_in_max),
            D_out=rd.uniform(D_out_min, D_out_max),
            T1_in=rd.uniform(T1_in_min, T1_in_max),
            T1_out=T1_out,  #rd.uniform(T1_out_min, T1_out_max),
            P0_in=1., #rd.uniform(P0_in_min, P0_in_max),
            P0_out=P0_out, #rd.uniform(P0_out_min, P0_out_max),
            Tfinal=40.,
            Nr=200,
            Nt=100,
            tanh_slope=0
        )
        dg.solve()
        dg.plot(os.path.join(folder, "plot"))
        data = dg.get()

        if (a:=rd.random()) < test:
            sub_folder = "test"
        elif a < test + dev:
            sub_folder = "dev"
        else:
            sub_folder = "train"

        file_path = os.path.join(folder, sub_folder, f"data_{data['C_out']}_{data['C_in']}_{data['D_in']}_{data['D_out']}_{data['R']}.pt")
        torch.save(data, file_path)

        data["P"] = []
        data["G_R"] = []
        print(f"{data=}")
        

        del dg
        del data

        
if __name__ == "__main__":
    folder = "examples/r_max=4R/low_res"
    #empty_database(folder)
    list_dict = extract_params_from_brut("data/brut")
    create_database(list_dict, folder, test=0.0, dev=0.0, n=1_000, micro_ondes=True)