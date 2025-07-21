import torch
import time

from neuralop import LpLoss
from experiments.fno2d_R.dataset import Dataset
from experiments.fno2d_R.integrator import compute_G_R_fno

lp_loss = LpLoss(d=2, p=2, reduction="mean") 

def accuracy(model: torch.nn.Module,
            dataloader: torch.utils.data.DataLoader,
            device: torch.device) -> float:

    model.eval()

    total_error: float = 0.0

    with torch.no_grad():
        for a, u in dataloader:
            a, u = a.to(device), u.to(device)

            u_pred = model(a).squeeze(1)  # Assuming model outputs shape (batch_size, 1, Nt, Nr)
            #error: torch.Tensor = torch.nn.functional.mse_loss(u, u_pred)

            error = lp_loss(u_pred, u)
            total_error += error.item()

    return total_error / len(dataloader), total_error_G / len(dataloader)


def G_accuracy(model: torch.nn.Module,
               dataset: Dataset,
               device: torch.device) -> float:
    
    model.eval()
    
    if not dataset.elements:
        return 0.0

    if dataset.param_mean is None or dataset.param_std is None:
        raise ValueError("Dataset is not normalized, cannot compute G accuracy.")
        
    total_error_G: float = 0.0
    g_loss = LpLoss(d=1, p=2, reduction="mean")

    r_max = dataset.elements[0][0].r_max
    R_mean = dataset.param_mean['R']
    R_std = dataset.param_std['R']

    dataloader = dataset.get_dataloader(bs=16, shuffle=False)
    
    with torch.no_grad():
        for a, u_true in dataloader:
            a, u_true = a.to(device), u_true.to(device)
            
            u_pred = model(a).squeeze(1)
            
            for i in range(a.shape[0]):
                log_R_norm = a[i, 3, 0, 0]
                log_R = log_R_norm * R_std + R_mean
                R = math.exp(log_R)

                G_pred = compute_G_R_fno(u_pred[i:i+1], R, r_max)
                G_true = compute_G_R_fno(u_true[i:i+1], R, r_max)

                total_error_G += g_loss(G_pred, G_true).item()
    
    return total_error_G / len(dataset)
 



def train_one_epoch(model: torch.nn.Module, 
                    dataloader: torch.utils.data.DataLoader, 
                    optimizer: torch.optim.Optimizer, 
                    device: torch.device) -> float:
    
    model.train()

    total_loss: float = 0.0

    for a, u in dataloader:
        a, u = a.to(device), u.to(device)

        optimizer.zero_grad()

        u_pred = model(a).squeeze(1)  # Assuming model outputs shape (batch_size, 1, Nt, Nr)

        #loss: torch.Tensor = torch.nn.functional.mse_loss(u, u_pred)
        loss = lp_loss(u_pred, u)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    
    total_loss = total_loss / len(dataloader)

    return total_loss


def train(model: torch.nn.Module, 
          dataloader: torch.utils.data.DataLoader, 
          optimizer: torch.optim.Optimizer, 
          scheduler: torch.optim.lr_scheduler._LRScheduler,
          epochs: int, 
          device: torch.device,
          test_loader: torch.utils.data.DataLoader | None) -> tuple[list[float], list[float], list[float]]:

    model.to(device)

    
    

    train_losses: list[float] = []
    times: list[float] = []
    test_losses: list[float] = []

    for epoch in range(epochs):
        t0: float = time.time()
        loss: float = train_one_epoch(model, dataloader, optimizer, device)
        t1: float = time.time()
        train_losses.append(loss)
        times.append(t1 - t0)

        if test_loader is not None:
            test_loss: float = accuracy(model, test_loader, device)
            test_losses.append(test_loss)
            
            # Mettre à jour le scheduler avec la perte de test
            scheduler.step(loss)
            
            current_lr = optimizer.param_groups[0]['lr']
            print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss:.4f}, Test Loss: {test_loss:.4f}, LR: {current_lr:.6f}, Time: {t1 - t0:.2f}s")
        
        else:
            # Sans test_loader, ReduceLROnPlateau ne peut pas fonctionner.
            # On peut appeler step() sur la perte d'entraînement, mais c'est moins courant.
            scheduler.step(loss)
            current_lr = optimizer.param_groups[0]['lr']
            print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss:.4f}, LR: {current_lr:.6f}, Time: {t1 - t0:.2f}s")

    return train_losses, test_losses, times
