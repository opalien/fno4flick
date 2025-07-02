"""
Entraîne FNO2float à prédire le rayon R depuis la solution P(r,t).

> python -m fno2float.main -l 4 -m 32 -c 64 -e 300
"""
import argparse
import os
import torch
from torch.utils.data import DataLoader
from torch import Tensor

from fno2float.dataset import Dataset
from fno2float.model   import FNO2float
from fno2float.train   import train


from utils_project.save import save_result


# -------- arguments CLI ---------------------------------------------------------------
parser = argparse.ArgumentParser(description="Learn R from P with FNO.")
parser.add_argument("-l", "--n_layers", type=int, default=4)
parser.add_argument("-m", "--n_modes", type=int, default=32)
parser.add_argument("-c", "--hidden_channels", type=int, default=64)
parser.add_argument("-e", "--epochs", type=int, default=300)
parser.add_argument("--train_dir", type=str, default="data/train")
parser.add_argument("--test_dir",  type=str, default="data/test")
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using", device)

# -------- charge datasets -------------------------------------------------------------
train_ds = Dataset()
train_ds.load(args.train_dir)

test_ds = Dataset()
test_ds.load(args.test_dir)

# -------- normalisation ----------------------------------------------------------------
P_stack = torch.stack([P for P, _ in train_ds.elements])   # (N, 1, H, W)
R_stack = torch.stack([R for _, R in train_ds.elements])   # (N, 1)

P_mean = P_stack.mean()
P_std  = P_stack.std().clamp_min(1e-8)

R_mean = R_stack.mean()
R_std  = R_stack.std().clamp_min(1e-8)

def normalize(ds):
    ds.elements = [
        ((P - P_mean) / P_std,
         (R - R_mean) / R_std)
        for P, R in ds.elements
    ]

normalize(train_ds)
normalize(test_ds)

# -------- data-loaders -----------------------------------------------------------------
train_dl = DataLoader(train_ds, batch_size=32, shuffle=True,  num_workers=os.cpu_count())
test_dl  = DataLoader(test_ds,  batch_size=32, shuffle=False, num_workers=os.cpu_count())

# -------- modèle + optim ---------------------------------------------------------------
model = FNO2float(
    n_modes=args.n_modes,
    hidden_channels=args.hidden_channels,
    n_layers=args.n_layers,
)

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=1e-3,
    weight_decay=1e-4,
)

# -------- entraînement -----------------------------------------------------------------
train_losses, test_losses, times = train(
    model=model,
    dl=train_dl,
    optim=optimizer,
    epochs=args.epochs,
    device=device,
    test_dl=test_dl,
)

# -------- démonstration ----------------------------------------------------------------
model.eval()
with torch.no_grad():
    P_ex, R_true_norm = test_ds[0]
    R_pred_norm: Tensor = model(P_ex.unsqueeze(0).to(device))
    R_pred = R_pred_norm * R_std + R_mean
    R_true = R_true_norm * R_std + R_mean
print(f"\nExemple :  R vrai = {R_true.item():.2f}  |  R prédit = {R_pred.item():.2f}")


to_save = {
    "train_losses": train_losses,
    "test_losses": test_losses,
    "times": times,
}

save_result(f"out/fno2float/results_{args.n_modes}_{args.hidden_channels}_{args.n_layers}_{args.epochs}.json", to_save)