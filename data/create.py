
import itertools
from data.data_generator import DataGenerator

import json
import os

import random
import numpy as np

C_jus = [11, 110]
C_cristal = [58, 20, 90 ]
T1_cristal = [40, 300]
R_sphere = [25, 50, 75, 250, 500, 750, 2500, 5000]





def extract_params_from_brut(folder: str):
    list_params: list[dict] = []
    for sub_folder in os.listdir(folder):

        if not os.path.isdir(os.path.join(folder, sub_folder)):
            continue

        try:
            C_cuve, C_ball, T1_ball, R_nm = map(float, sub_folder.split('_'))
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


            D_ref = 500          # nm^2/s
            C_ref = 60           # mol/L
            D_ball = D_ref * (C_ball / C_ref) ** (1/3)
            D_cuve = D_ref * (C_cuve / C_ref) ** (1/3)


            D_ball = D_ball * 1000
            D_cuve = D_cuve* 1000

            T1_ball = T1_ball * 10
            TB = TB * 10

            list_params.append({
                "C_cuve": C_cuve,
                "C_ball": C_ball,
                "D_cuve": D_cuve,
                "D_ball": D_ball,
                "T1_cuve": TB,
                "T1_ball": T1_ball,
                "P0_cuve": P0,
                "P0_ball": 1.,
                "R": R_nm, # NM !!!!
                #"TB": TB
            })

    return list_params




def empty_database(folder: str):
    if os.path.exists(folder):
        for sub_folder in ["train", "test", "plot"]:
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


def create_database(list_dict: list[dict], folder: str, prop: float = 0.1):
    if not os.path.exists(folder):
        os.makedirs(folder)

    for params in list_dict:
        dg = DataGenerator(R=params["R"],
                        C_cuve=params["C_cuve"],
                        C_ball=params["C_ball"],
                        D_cuve=params["D_cuve"],
                        D_ball=params["D_ball"],
                        Tre_cuve=params["T1_cuve"],
                        Tre_ball=params["T1_ball"],
                        P0_cuve=params["P0_cuve"],
                        P0_ball=params["P0_ball"],
                        cuve_width=6_000.,
                        Tfinal=10.,
                        Nr=100,
                        Nt=100)

        print(f"{params=}")

        dg.solve()
        dg.plot("data/plot")

        data = dg.get()

        data_json = json.dumps(data, default=lambda x: x.tolist(), indent=4)


        if random.random() < prop:
            sub_folder = "test"
        else:
            sub_folder = "train"

        with open(os.path.join(folder, sub_folder, f"data_{params['C_cuve']}_{params['C_ball']}_{params['T1_cuve']}_{params['R']}.json"), "w") as f:
            f.write(data_json)







if __name__ == "__main__":
    empty_database("data")
    list_dict = extract_params_from_brut("data/brut")
    create_database(list_dict, "data", prop=0.1)












def create_database_():
    for c_j, c_r, t1_c, r_s in itertools.product(C_jus, C_cristal, T1_cristal, R_sphere):
        dg = DataGenerator(R=r_s,
                        C_cuve=c_j,
                        C_ball=c_r,
                        D_cuve=1.,
                        D_ball=0.1,
                        Tre_cuve=t1_c,
                        Tre_ball=t1_c,
                        cuve_width=1.,
                        Tfinal=1.,
                        Nr=100,
                        Nt=100)
        

        print(f"Dataset size for C_j={c_j}, C_r={c_r}, T1_c={t1_c}, R_sphere={r_s}")

        dg.solve()

        data = dg.get()


        data_json = json.dumps(data, default=lambda x: x.tolist(), indent=4)

        if random.random() < 0.1:
            with open(f"data/test/data_{c_j}_{c_r}_{t1_c}_{r_s}.json", "w") as f:
                f.write(data_json)
        else:
            with open(f"data/train/data_{c_j}_{c_r}_{t1_c}_{r_s}.json", "w") as f:
                f.write(data_json)
        
        #with open(f"data/all/data_{c_j}_{c_r}_{t1_c}_{r_s}.json", "w") as f:
        #    f.write(data_json)
        

        # Save the dataset to a file

        

