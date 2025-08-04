import os
import argparse
import random
import shutil

from typing import Any

import torch

from sklearn.model_selection import train_test_split

from models.dataset import FickDataset
from models.model import FickModel
from util.test_model import plot_correlations, run_diagnostics
from util.train import accuracy, train





device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
match os.cpu_count():
    case None:  torch.set_num_threads(1)
    case n:     torch.set_num_threads(n)




parser = argparse.ArgumentParser(description="Fick PDE model solver training.")
parser.add_argument("-l", "--n_layers", type=int, default=2, help="number of layers")
parser.add_argument("-m", "--n_modes", type=int, default=16, help="number of modes")
parser.add_argument("-c", "--hidden_channels", type=int, default=16, help="number of hidden channels")
parser.add_argument("-e", "--epochs", type=int, default=100, help="number of training epochs")
parser.add_argument("-p", "--model_path", type=str, default="", help="path to model")
parser.add_argument("-d", "--dataset_path", type=str, default="data", help="path to dataset")
parser.add_argument("-s", "--dataset_size", type=int, default=-1, help="size of the dataset")
parser.add_argument("-n", "--name", type=str, default="fno4fick", help="name of the experiment")
parser.add_argument("-b", "--batch_size", type=int, default=64, help="batch size")

args = parser.parse_args()


path = f"out/{args.name}/"
path_dataset = os.path.join(path, "dataset")
path_models = os.path.join(path, "models")

path_diagnostics = os.path.join(path, "diagnostics")
path_diagnostics_test = os.path.join(path_diagnostics, "test")
path_diagnostics_train = os.path.join(path_diagnostics, "train")


if os.path.isdir(path):
    shutil.rmtree(path)
os.makedirs(path, exist_ok=True)
os.makedirs(path_dataset, exist_ok=True)
os.makedirs(path_models, exist_ok=True)

os.makedirs(path_diagnostics, exist_ok=True)
os.makedirs(path_diagnostics_test, exist_ok=True)
os.makedirs(path_diagnostics_train, exist_ok=True)



def setup_datasets():
    dataset = FickDataset() 
    print(f"Loading dataset from {args.dataset_path}")
    dataset.load(args.dataset_path)

    if args.dataset_size > 0:
        dataset.elements = random.sample(dataset.elements, args.dataset_size)

    dataset.get_normalizers()

    print(f"Plotting dataset distribution ", end="")
    dataset.plot_distribution(path_dataset, actions={""})
    print("rescale", end=", ")
    dataset.plot_distribution(path_dataset, actions={"rescale"})
    print("nondimensionalize", end=", ")
    dataset.plot_distribution(path_dataset, actions={"rescale", "nondimensionalize"})
    print("compress", end=", ")
    dataset.plot_distribution(path_dataset, actions={"rescale", "nondimensionalize", "compress"})
    print("normalize", end=", ")
    dataset.plot_distribution(path_dataset, actions={"rescale", "nondimensionalize", "compress", "normalize"})
    print("done")

    print(f"Normalisations parameters: \nC : {dataset.C_normalizer}, \nD : {dataset.D_normalizer}, \nR : {dataset.R_normalizer}, \nT1 : {dataset.T1_normalizer}, \nP0 : {dataset.P0_normalizer}")


    train_dataset, test_dataset = FickDataset(), FickDataset()

    train_dataset.elements, test_dataset.elements = train_test_split(
        dataset.elements, 
        test_size=0.1, 
        random_state=42
    )

    train_dataset.set_normalizers(dataset)
    test_dataset.set_normalizers(dataset)

    return train_dataset, test_dataset


def get_model() -> tuple[dict[Any, Any], FickModel]:
    if args.model_path:
        checkpoint = torch.load(args.model_path, weights_only=False)
        model = FickModel(**checkpoint["parameters"])
        model = torch.compile(model)
        model.load_state_dict(checkpoint["model_state_dict"], strict=False)

        return checkpoint, model


    else:
        checkpoint = {
            "parameters": {
                "n_modes": (args.n_modes, args.n_modes),
                "n_layers": args.n_layers,
                "hidden_channels": args.hidden_channels,
                "device": device
            },

            "model_state_dict": None,
            "iterations": []
        }

        model = FickModel(
            n_modes=(args.n_modes, args.n_modes),
            n_layers=args.n_layers,
            hidden_channels=args.hidden_channels,
            device=device
        )
        model = torch.compile(model)

        return checkpoint, model




if __name__ == "__main__":

    train_dataset, test_dataset = setup_datasets()
    train_dataloader = train_dataset.get_dataloader(args.batch_size, shuffle=True)
    test_dataloader = test_dataset.get_dataloader(args.batch_size, shuffle=False)

    checkpoint, model = get_model()

    print(f"Accuracy without training: {accuracy(model, test_dataloader, device)}")


    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-2,
        weight_decay=1e-4
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)


    train_losses, test_losses, times = train(model=model,
        dataloader=train_dataloader,
        optimizer=optimizer,
        scheduler=scheduler,
        epochs=args.epochs,
        device=device,
        test_loader=test_dataloader
    )

    print(f"Training done")

    print(f"Saving checkpoint")

    checkpoint["iterations"].append({
        "train_losses": train_losses,
        "test_losses": test_losses,
        "times": times
    })
    checkpoint["model_state_dict"] = model.state_dict()
    torch.save(checkpoint, os.path.join(path_models, "checkpoint.pth"))


    print(f"Accuracy after training: {accuracy(model, test_dataloader, device)}")


    print(f"Running diagnostics")
    test_results = run_diagnostics(
        model=model,
        dataset=test_dataset,
        device=device,
        out_dir=path_diagnostics_test,
    )

    print(f"Plotting correlations")
    plot_correlations(test_results, out_dir=path_diagnostics_test)


    train_results = run_diagnostics(
        model=model,
        dataset=train_dataset,
        device=device,
        out_dir=path_diagnostics_train,
    )

    print(f"Plotting correlations")
    plot_correlations(train_results, out_dir=path_diagnostics_train)







