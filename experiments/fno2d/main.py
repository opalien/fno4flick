from ..fno2d.dataset import Dataset
import torch
from torch import Tensor
import argparse
import os
import time
from neuralop.models import FNO
from experiments.fno2d.train import train

from util.save import save_result


# n_modes = 32
# hidden_channels = 64
# n_layers = 8


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
args = parser.parse_args()

n_modes = args.n_modes
hidden_channels = args.hidden_channels
n_layers = args.n_layers
epochs = args.epochs



if __name__ == "__main__":
    train_dataset = Dataset()
    train_dataset.load("data/train")

    test_dataset = Dataset()
    test_dataset.load("data/test")

    train_dataset.normalize()
    test_dataset.normalize()

    train_dataloader = train_dataset.get_dataloader(64, shuffle=True)
    test_dataloader = test_dataset.get_dataloader(64, shuffle=False)

    model = FNO(n_modes=(n_modes,n_modes),
                hidden_channels=hidden_channels,
                in_channels=3,
                out_channels=1,
                n_layers=n_layers
    )



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

    to_save = {
        "train_losses": train_losses,
        "test_losses": test_losses,
        "times": times,
    }

    save_result(f"out/fno2d/results_{n_modes}_{hidden_channels}_{n_layers}_{epochs}.json", to_save)

