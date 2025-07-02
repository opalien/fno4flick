import time
import torch
from torch.utils.data import DataLoader
from torch import Tensor, nn


@torch.no_grad()
def accuracy(model: nn.Module, dl: DataLoader, device: torch.device) -> float:
    model.eval()
    total = 0.0
    for P, R in dl:
        P, R = P.to(device), R.to(device)
        R_pred: Tensor = model(P)
        total += nn.functional.mse_loss(R_pred, R).item()
    return total / len(dl)


def train_one_epoch(
    model: nn.Module,
    dl: DataLoader,
    optim: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    model.train()
    running = 0.0
    for P, R in dl:
        P, R = P.to(device), R.to(device)

        optim.zero_grad()
        R_pred = model(P)
        loss: Tensor = nn.functional.mse_loss(R_pred, R)
        loss.backward()
        optim.step()

        running += loss.item()
    return running / len(dl)


def train(
    model: nn.Module,
    dl: DataLoader,
    optim: torch.optim.Optimizer,
    epochs: int,
    device: torch.device,
    test_dl: DataLoader | None = None,
):
    model.to(device)

    test_losses = []
    train_losses = []
    times = []

    for ep in range(1, epochs + 1):
        t0 = time.time()
        l_train = train_one_epoch(model, dl, optim, device)
        t1 = time.time()

        if test_dl is not None:
            l_test = accuracy(model, test_dl, device)

            test_losses.append(l_test)
            train_losses.append(l_train)
            times.append(t1 - t0)

            print(
                f"Epoch {ep:3d}/{epochs} | train {l_train:.4e} | "
                f"test {l_test:.4e} | {t1 - t0:.2f}s"
            )
        else:
            print(f"Epoch {ep:3d}/{epochs} | train {l_train:.4e} | {t1 - t0:.2f}s")

    return train_losses, test_losses, times