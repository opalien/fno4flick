import torch
import time


def accuracy(model: torch.nn.Module,
            dataloader: torch.utils.data.DataLoader,
            device: torch.device) -> float:

    model.eval()

    total_error: float = 0.0

    with torch.no_grad():
        for a, u in dataloader:
            a, u = a.to(device), u.to(device)

            u_pred = model(a).squeeze(1)  # Assuming model outputs shape (batch_size, 1, Nt, Nr)
            error: torch.Tensor = torch.nn.functional.mse_loss(u, u_pred)
            total_error += error.item()

    return total_error / len(dataloader)



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
        
        loss: torch.Tensor = torch.nn.functional.mse_loss(u, u_pred)        
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    
    total_loss = total_loss / len(dataloader)

    return total_loss


def train(model: torch.nn.Module, 
          dataloader: torch.utils.data.DataLoader, 
          optimizer: torch.optim.Optimizer, 
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
            print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss:.4f}, Test Loss: {test_loss:.4f}, Time: {t1 - t0:.2f}s")
        else:

            print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss:.4f}, Time: {t1 - t0:.2f}s")

    return train_losses, test_losses, times
