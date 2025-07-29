import os
import argparse
import random
import torch
from torch import Tensor

from sklearn.model_selection import train_test_split

from neuralop.models import FNO

from experiments.fno2d_2out.dataset import Dataset
from experiments.fno2d_2out.integrator import plot_search_R
from experiments.fno2d_2out.some_tests import run_diagnostics, plot_correlations
from experiments.fno2d_2out.train import accuracy, train



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
parser.add_argument("-s", "--dataset_size", type=int, default=-1, help="size of the dataset")
parser.add_argument("-n", "--name", type=str, default="fno2d_2out", help="name of the experiment")
parser.add_argument("-r", "--r_max_fixed", type=bool, default=False, help="if True, r_max is fixed to the true value")
parser.add_argument("-b", "--batch_size", type=int, default=64, help="batch size")
args = parser.parse_args()
n_layers, n_modes, hidden_channels, epochs, model_path, dataset_path, name, r_max_fixed = args.n_layers, args.n_modes, args.hidden_channels, args.epochs, args.model_path, args.dataset_path, args.name, args.r_max_fixed
batch_size = args.batch_size
dataset_size = args.dataset_size



if __name__ == "__main__":

    directory = f"out/{name}/"
    os.makedirs(directory, exist_ok=True)

    
    dataset = Dataset()

    print(f"Loading dataset from {dataset_path}")
    dataset.load(dataset_path)
    
    if dataset_size > 0:
        dataset.elements = random.sample(dataset.elements, dataset_size)

    directory_dataset = os.path.join(directory, "dataset")
    os.makedirs(directory_dataset, exist_ok=True)
    

    dataset.plot_distribution(directory_dataset, "dataset")
    dataset.rescale()
    dataset.plot_distribution(directory_dataset, "rescaled")
    dataset.nondimensionalize()
    dataset.plot_distribution(directory_dataset, "nondimensionalized")
    dataset.compress()
    dataset.plot_distribution(directory_dataset, "compressed")
    dataset.normalize()
    dataset.plot_distribution(directory_dataset, "normalized")

    train_dataset, test_dataset = Dataset(), Dataset()
    
    train_dataset.set_normalizers(dataset)
    test_dataset.set_normalizers(dataset)

    train_dataset.elements, test_dataset.elements = train_test_split(
        dataset.elements, 
        test_size=0.1, 
        random_state=42
    )

    train_dataloader = train_dataset.get_dataloader(batch_size, shuffle=True)
    test_dataloader = test_dataset.get_dataloader(batch_size, shuffle=False)


    if model_path:
        checkpoint = torch.load(args.model_path, weights_only=False)
        model = FNO(**checkpoint["parameters"])
        model = torch.compile(model)
        model.load_state_dict(checkpoint["model_state_dict"], strict=False)

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
                "in_channels": 6,
                "out_channels": 2,
            },

            "model_state_dict": None,
            "iterations": []
        }

        model = FNO(n_modes=(n_modes,n_modes),
                    hidden_channels=hidden_channels,
                    in_channels=6,
                    out_channels=2,
                    n_layers=n_layers,
                    lift_dropout=lift_dropout, 
                    projection_dropout=projection_dropout
        )

        model = torch.compile(model)

    model = model.to(device)


    print(f"Normalisation parameters: {train_dataset.C_normalizer=}, {train_dataset.D_normalizer=}, {train_dataset.T1_normalizer=}")

    print("Accuracy without training : ", accuracy(model, test_dataloader, device))
    

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

    checkpoint["iterations"].append({
        "train_losses": train_losses,
        "test_losses": test_losses,
        "times": times
    })

    checkpoint["model_state_dict"] = model.state_dict()

    
    os.makedirs(os.path.join(directory, "models"), exist_ok=True)
    os.makedirs(os.path.join(directory, "plots"), exist_ok=True)
    os.makedirs(os.path.join(directory, "plots", "test"), exist_ok=True)
    os.makedirs(os.path.join(directory, "plots", "train"), exist_ok=True)

    torch.save(checkpoint, os.path.join(directory, "models", f"{n_layers}_{n_modes}_{hidden_channels}_{len(checkpoint['iterations'])}.pth"))

    print("Accuracy with training : ", accuracy(model, test_dataloader, device))
    # PAR :
    

    diagn_dir = os.path.join(directory, "diagnostics")
    os.makedirs(diagn_dir, exist_ok=True)

    # Test set diagnostics
    test_dir = os.path.join(diagn_dir, "test")
    test_results = run_diagnostics(
        model=model,
        dataset=test_dataset,
        device=device,
        out_dir=test_dir,
        tag=f"{name}_test",
        r_max_fixed=r_max_fixed,
        R_scan_points=200,
        R_scan_factor=0.2
    )
    plot_correlations(test_results, out_dir=test_dir, tag=f"{name}_test")

    # Train set diagnostics (optionnel)
    train_dir = os.path.join(diagn_dir, "train")
    train_results = run_diagnostics(
        model=model,
        dataset=train_dataset,
        device=device,
        out_dir=train_dir,
        tag=f"{name}_train",
        r_max_fixed=r_max_fixed
    )
    plot_correlations(train_results, out_dir=train_dir, tag=f"{name}_train")
    
    #for i in range(len(test_dataset.elements)):
    #    plot_search_R(model, test_dataset, i, device, os.path.join(directory, "plots", "test", f"{i}.png"), r_max_fixed=r_max_fixed)
#
    #for i in range(len(train_dataset.elements)):
    #    plot_search_R(model, train_dataset, i, device, os.path.join(directory, "plots", "train", f"{i}.png"), r_max_fixed=r_max_fixed)















