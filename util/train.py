import torch
import time

from neuralop import LpLoss
from models.dataset import FickDataset
from models.model import FickModel, postprocess
from util.G_method import compute_G_in, compute_G_out

lp_loss = LpLoss(d=2, p=2, reduction="mean") 


def R_error(model: FickModel,
            dataloader: torch.utils.data.DataLoader,
            device: torch.device) -> tuple[float, float]:
    
    model.eval()
    model.to(device)
    
    loss_in = 0.0
    loss_out = 0.0
    n = 0
    with torch.no_grad():
        for params, P in dataloader:
            batch_size = len(params)
            n+=batch_size
            params = [p.to(device) for p in params]
            
            P = postprocess(P)
            
            G_in_true = compute_G_in(P)
            list_R_in_pred = torch.tensor(model.search_R_in(params,  G_in_true, device))

            R_true_tensor = torch.stack([p.R for p in params])
            loss_in += torch.mean(torch.abs(R_true_tensor - list_R_in_pred)/R_true_tensor) * batch_size

            G_out_true = compute_G_out(P, params)
            list_R_out_pred = torch.tensor(model.search_R_out(params,  G_out_true, device))

            R_true_tensor = torch.stack([p.R for p in params])
            loss_out += torch.mean(torch.abs(R_true_tensor - list_R_out_pred)/R_true_tensor) * batch_size
            
            params = [p.to("cpu") for p in params]

    return loss_in / n, loss_out / n





def accuracy(model: torch.nn.Module, dataloader: torch.utils.data.DataLoader, device: torch.device) -> float:
    model.eval()
    model.to(device)
    loss = 0.0
    with torch.no_grad():
        for params, P in dataloader:
            params = [p.to(device) for p in params]
            P = P.to(device)

            P_pred = model(params)
            loss += lp_loss(P_pred, P)
            params = [p.to("cpu") for p in params]
    return loss.item() / len(dataloader)


def train_one_epoch(model: torch.nn.Module, 
                    dataloader: torch.utils.data.DataLoader, 
                    optimizer: torch.optim.Optimizer, 
                    device: torch.device) -> float:
    
    model.train()
    model.to(device)
    total_loss = 0.0

    for i, (params, P) in enumerate(dataloader):
        params = [p.to(device) for p in params]
        P = P.to(device)

        optimizer.zero_grad()
        P_pred = model(params)
        loss = lp_loss(P_pred, P)
        total_loss+=loss.item()

        loss.backward()
        optimizer.step()
        params = [p.to("cpu") for p in params]
    return total_loss / len(dataloader)




def train(model: torch.nn.Module, 
          dataloader: torch.utils.data.DataLoader, 
          optimizer: torch.optim.Optimizer, 
          scheduler: torch.optim.lr_scheduler._LRScheduler,
          epochs: int, 
          device: torch.device,
          test_loader: torch.utils.data.DataLoader | None) -> tuple[list[float], list[float], list[float]]:

    model.to(device)

    train_losses: list[float] = []
    test_losses: list[float] = []
    times: list[float] = []


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
            scheduler.step(test_loss)
            
            current_lr = optimizer.param_groups[0]['lr']
            print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss:.4f}, Test Loss: {test_loss:.4f}, LR: {current_lr:.6f}, Time: {t1 - t0:.2f}s")
        
        else:
            # Sans test_loader, ReduceLROnPlateau ne peut pas fonctionner.
            # On peut appeler step() sur la perte d'entraînement, mais c'est moins courant.
            scheduler.step(loss)
            current_lr = optimizer.param_groups[0]['lr']
            print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss:.4f}, LR: {current_lr:.6f}, Time: {t1 - t0:.2f}s")

    return train_losses, test_losses, times




