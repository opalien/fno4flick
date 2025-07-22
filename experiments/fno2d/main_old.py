import os
import argparse

import torch
from torch import Tensor

from sklearn.model_selection import train_test_split

from neuralop.models import FNO

from experiments.fno2d.dataset import Dataset
from experiments.fno2d.integrator import compute_G
from experiments.fno2d.train import accuracy, G_accuracy, train



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
match os.cpu_count():
    case None:  torch.set_num_threads(1)
    case n:     torch.set_num_threads(n)


parser = argparse.ArgumentParser(description="PDE solving.")
parser.add_argument("-l", "--n_layers", type=int, default=2, help="number of layers")
parser.add_argument("-m", "--n_modes", type=int, default=16, help="number of modes")
parser.add_argument("-c", "--hidden_channels", type=int, default=16, help="number of hidden channels")
parser.add_argument("-e", "--epochs", type=int, default=100, help="number of training epochs")
parser.add_argument("-p", "--model_path", type=str, default="", help="path to model")
parser.add_argument("-d", "--dataset_path", type=str, default="data", help="path to dataset")
parser.add_argument("-n", "--name", type=str, default="fno2d", help="name of the experiment")
args = parser.parse_args()
n_layers, n_modes, hidden_channels, epochs, model_path, dataset_path = args.n_layers, args.n_modes, args.hidden_channels, args.epochs, args.model_path, args.dataset_path
name = args.name


def plot_search_R_(model: FNO,
                  dataset: Dataset,
                  i: int,
                  device: torch.device):
    import torch
    from neuralop.models import FNO
    from experiments.fno2d.dataset import Dataset, collate_fn
    from experiments.fno2d.integrator import compute_G_R_fno
    from experiments.fno2d.edp_parameters import EDPParameters
    import matplotlib.pyplot as plt
    from copy import deepcopy

    model.eval()
    model.to(device)

    params_true, P_quadrant_true = dataset.elements[i]
    
    R_true_norm = params_true.R 
    r_max = params_true.r_max


    _, P_true_full = collate_fn([(params_true, P_quadrant_true)])
    P_true_full = P_true_full.to(device)
    
    with torch.no_grad():
        G_true = compute_G_R_fno(P_true_full, R_true_norm, r_max)

    R_search_space = torch.linspace(0.5 * R_true_norm, 1.5 * R_true_norm, 10, device=device)
    
    input_batch = []
    for r_guess_norm in R_search_space:
        params_guess = deepcopy(params_true)
        params_guess.R = r_guess_norm.item()
        input_batch.append((params_guess, P_quadrant_true))

    a_batch, _ = collate_fn(input_batch)
    a_batch = a_batch.to(device)

    mse_values = []
    with torch.no_grad():
        P_pred_batch = model(a_batch).squeeze(1)

        for j in range(len(R_search_space)):
            r_guess_norm = R_search_space[j]
            
            P_pred_j = P_pred_batch[j:j+1]
            
            G_pred_j = compute_G_R_fno(P_pred_j, r_guess_norm.item(), r_max)
            
            loss = torch.nn.functional.mse_loss(G_true, G_pred_j)
            mse_values.append(loss.item())

    plt.figure(figsize=(10, 6))
    plt.plot(R_search_space.cpu().numpy(), mse_values, label='MSE(G_true, G_pred(r))')
    plt.axvline(x=R_true_norm, color='r', linestyle='--', label=f'Vrai R normalisé = {R_true_norm:.3f}')
    plt.xlabel("Rayon normalisé deviné (r)")
    plt.ylabel("Erreur Quadratique Moyenne (MSE)")
    plt.title(f"Recherche de R par minimisation de l'erreur sur G (Échantillon {i})")
    plt.legend()
    plt.grid(True)
    plt.yscale('log')
    plt.savefig(f"search_R_{name}_{i}.png")
    plt.show() 


def plot_search_R(model: FNO,
                  dataset: Dataset,
                  i: int,
                  device: torch.device,
                  name: str):
    # ... (les imports locaux restent les mêmes)
    from copy import deepcopy
    import torch
    from neuralop.models import FNO
    from experiments.fno2d.dataset import Dataset, collate_fn
    from experiments.fno2d.integrator import compute_G_R_fno
    from experiments.fno2d.edp_parameters import EDPParameters
    import matplotlib.pyplot as plt
    import math
    

    model.eval()
    model.to(device)

    params_true, P_quadrant_true = dataset.elements[i]
    
    # Ceci est la valeur de log(R) normalisée
    R_true_norm_log = params_true.R 
    # r_max est 1.0 après la mise à l'échelle
    r_max = params_true.r_max
    
    # Récupérer le normaliseur depuis le dataset
    R_normalizer = dataset.R_normalizer

    # --- Étape 1: Reconvertir le R vrai en valeur physique (mise à l'échelle) ---
    log_R_true_unnorm = R_normalizer.unnormalize(R_true_norm_log)
    R_true_physical = math.exp(log_R_true_unnorm)

    _, P_true_full = collate_fn([(params_true, P_quadrant_true)])
    P_true_full = P_true_full.to(device)
    
    with torch.no_grad():
        # Utiliser la valeur physique de R pour le calcul
        G_true = compute_G_R_fno(P_true_full, R_true_physical, r_max)

    # L'espace de recherche reste sur les valeurs de log(R) normalisées
    R_norm_log_search_space = torch.linspace(0.5 * R_true_norm_log, 1.5 * R_true_norm_log, 100, device=device)
    
    input_batch = []
    for r_guess_norm_log in R_norm_log_search_space:
        params_guess = deepcopy(params_true)
        params_guess.R = r_guess_norm_log.item()
        input_batch.append((params_guess, P_quadrant_true))

    a_batch, _ = collate_fn(input_batch)
    a_batch = a_batch.to(device)

    mse_values = []
    R_physical_search_space_cpu = [] # Pour l'axe des x du graphique
    with torch.no_grad():
        # Correction précédente : .squeeze(1) pour passer de 4D à 3D
        P_pred_batch = model(a_batch).squeeze(1)

        for j in range(len(R_norm_log_search_space)):
            r_guess_norm_log = R_norm_log_search_space[j]
            
            # --- Étape 2: Reconvertir chaque R deviné en valeur physique ---
            log_r_guess_unnorm = R_normalizer.unnormalize(r_guess_norm_log.item())
            r_guess_physical = math.exp(log_r_guess_unnorm)
            R_physical_search_space_cpu.append(r_guess_physical)
            
            P_pred_j = P_pred_batch[j:j+1]
            
            # Utiliser la valeur physique de R pour le calcul
            G_pred_j = compute_G_R_fno(P_pred_j, r_guess_physical, r_max)
            
            loss = torch.nn.functional.mse_loss(G_true, G_pred_j)
            mse_values.append(loss.item())

    plt.figure(figsize=(10, 6))
    # --- Étape 3: Mettre à jour le graphique pour utiliser les valeurs physiques ---
    plt.plot(R_physical_search_space_cpu, mse_values, label='MSE(G_true, G_pred(r))')
    plt.axvline(x=R_true_physical, color='r', linestyle='--', label=f'Vrai R (physical) = {R_true_physical:.3f}')
    plt.xlabel("Rayon physique deviné (r)")
    plt.ylabel("Erreur Quadratique Moyenne (MSE)")
    plt.title(f"Recherche de R par minimisation de l'erreur sur G (Échantillon {i})")
    plt.legend()
    plt.grid(True)
    plt.yscale('log')
    plt.savefig(f"search_R_{name}_{i}.png")
    plt.show()


if __name__ == "__main__":
    dataset = Dataset()
    dataset.load(dataset_path)

    dataset.rescale()
    dataset.nondimensionalize()
    dataset.compress()
    dataset.normalize()

    train_dataset, test_dataset = Dataset(), Dataset()
    
    train_dataset.set_normalizers(dataset)
    test_dataset.set_normalizers(dataset)

    train_dataset.elements, test_dataset.elements = train_test_split(
        dataset.elements, 
        test_size=0.1, 
        random_state=42
    )

    train_dataloader = train_dataset.get_dataloader(64, shuffle=True)
    test_dataloader = test_dataset.get_dataloader(64, shuffle=False)


    if model_path:
        checkpoint = torch.load(args.model_path)
        model = FNO(**checkpoint["parameters"])
        model.load_state_dict(checkpoint["model_state_dict"])

    else:
        lift_dropout = 0.0
        projection_dropout = 0.0

        checkpoint = {
            "parameters": {
                "n_modes": (n_modes,n_modes),
                "hidden_channels": hidden_channels,
                "n_layers": n_layers,
                "lift_dropout": lift_dropout,
                "projection_dropout": projection_dropout,
            },

            "model_state_dict": None,
            "iterations": []
        }

        model = FNO(n_modes=(n_modes,n_modes),
                    hidden_channels=hidden_channels,
                    in_channels=3,
                    out_channels=1,
                    n_layers=n_layers,
                    lift_dropout=lift_dropout, 
                    projection_dropout=projection_dropout
        )

    model.to(device)

    print(f"Normalisation parameters: {train_dataset.C_normalizer=}, {train_dataset.D_normalizer=}, {train_dataset.T1_normalizer=}")

    print("Accuracy without training : ", accuracy(model, test_dataloader, device))
    print("G accuracy without training : ", G_accuracy(model, test_dataloader, device))


    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-2,
        #betas=(0.9, 0.999),
        #eps=1e-8,
        weight_decay=1e-4
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=10, factor=0.5)

    train_losses, test_losses, times = train(model=model,
        dataloader=train_dataloader,
        optimizer=optimizer,
        scheduler=scheduler,
        epochs=epochs,
        device=device,
        test_loader=test_dataloader
    )

    print("Accuracy with training : ", accuracy(model, test_dataloader, device))
    print("G accuracy with training : ", G_accuracy:=G_accuracy(model, test_dataloader, device))

    for i in range(len(test_dataset.elements))[:10]:
        plot_search_R(model, test_dataset, i, device, "test")

    for i in range(len(train_dataset.elements))[:10]:
        plot_search_R(model, train_dataset, i, device, "train")















