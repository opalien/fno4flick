from fno2d.dataset import Dataset
import torch
from torch import Tensor
import argparse
import os
import time
from neuralop.models import FNO
from fno2d.train import train

from utils.save import save_result

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

def plot_model(model: FNO, params: Tensor, u_mean: Tensor, u_std: Tensor):
    import matplotlib.pyplot as plt

    u_pred_normalized = model(params.unsqueeze(0))
    u_pred_normalized = u_pred_normalized.squeeze(0).squeeze(0).detach().cpu()

    u_pred_denormalized = u_pred_normalized * u_std + u_mean
    u_pred_denormalized = u_pred_denormalized.numpy()

    plt.figure(figsize=(10, 5))
    plt.clf()
    plt.imshow(u_pred_denormalized, aspect='auto', cmap='hot')
    plt.colorbar()
    plt.title("Model Output (Denormalized)")
    plt.xlabel("Spatial Points")
    plt.ylabel("Time Steps")
    plt.savefig("model_output.png", dpi=180)
    plt.show()

if __name__ == "__main__":
    train_dataset = Dataset()
    train_dataset.load("data/train")

    test_dataset = Dataset()
    test_dataset.load("data/test")

    
    all_a_tensors = torch.stack([params for params, u in train_dataset.elements])
    all_u_tensors = torch.stack([u for params, u in train_dataset.elements])

    a_mean = all_a_tensors.mean(dim=(0, 2, 3), keepdim=True).squeeze(0)
    a_std = all_a_tensors.std(dim=(0, 2, 3), keepdim=True).squeeze(0)
    a_std[a_std == 0] = 1e-8

    u_mean = all_u_tensors.mean()
    u_std = all_u_tensors.std()

    train_dataset.elements = [
        ((params - a_mean) / a_std, (u - u_mean) / u_std)
        for params, u in train_dataset.elements
    ]

    test_dataset.elements = [
        ((params - a_mean) / a_std, (u - u_mean) / u_std)
        for params, u in test_dataset.elements
    ]

    model = FNO(n_modes=(n_modes,n_modes),
                hidden_channels=hidden_channels,
                in_channels=3,
                out_channels=1,
                n_layers=n_layers
    )

    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=10, shuffle=True)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=10, shuffle=False)


    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-2,
        #betas=(0.9, 0.999),
        #eps=1e-8,
        weight_decay=1e-4         # décorrélé grâce à AdamW
    )

    train_losses, test_losses, times = train(model=model,
        dataloader=train_dataloader,
        optimizer=optimizer,
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

