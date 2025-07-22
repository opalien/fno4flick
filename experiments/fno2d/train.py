import torch
import time

from neuralop import LpLoss
from experiments.fno2d.dataset import Dataset
from experiments.fno2d.integrator import compute_G_R_fno

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

    return total_error / len(dataloader)


def G_accuracy(model: torch.nn.Module,
               dataloader: torch.utils.data.DataLoader,
               device: torch.device) -> float:
    model.eval()
    total_error_G: float = 0.0
    g_loss = LpLoss(d=1, p=1, reduction="mean") # L1 Loss est souvent plus stable pour ce type de valeur

    # r_max devient 1.0 après le prétraitement .rescale() dans le dataset.
    r_max = 1.0

    with torch.no_grad():
        for a, u_true in dataloader:
            a, u_true = a.to(device), u_true.to(device)
            u_pred = model(a).squeeze(1)
            
            B, pNt, pNs = u_pred.shape
            Ns = pNs // 2
            dr = r_max / (Ns - 1) if Ns > 1 else 0.0

            # On utilise le canal 0 (carte de C) pour trouver la frontière R.
            c_map_quadrant_batch = a[:, 0, pNt // 2, Ns:]

            batch_g_pred = []
            batch_g_true = []

            for i in range(B):
                c_map = c_map_quadrant_batch[i]
                val_in = c_map[0]

                # Trouve l'index du premier point où la valeur du paramètre change.
                # torch.where est plus robuste que argmax.
                indices_changement = torch.where(c_map != val_in)[0]
                
                if len(indices_changement) == 0:
                    num_points_in_R = Ns
                else:
                    num_points_in_R = indices_changement[0].item()

                # Convertit le nombre de points en une distance physique (dans l'espace mis à l'échelle)
                R_scaled = num_points_in_R * dr

                # Appel à la bonne fonction d'intégration
                G_pred = compute_G_R_fno(u_pred[i:i+1], R_scaled, r_max)
                G_true = compute_G_R_fno(u_true[i:i+1], R_scaled, r_max)

                batch_g_pred.append(G_pred)
                batch_g_true.append(G_true)

            batch_g_pred_tensor = torch.cat(batch_g_pred)
            batch_g_true_tensor = torch.cat(batch_g_true)
            total_error_G += g_loss(batch_g_pred_tensor, batch_g_true_tensor).item()
            
    return total_error_G / len(dataloader)


def G_accuracy_old2(model: torch.nn.Module,
               dataloader: torch.utils.data.DataLoader,
               device: torch.device) -> float:
    
    model.eval()
    
    total_error_G: float = 0.0

    with torch.no_grad():
        for a, u_true in dataloader:
            a, u_true = a.to(device), u_true.to(device)
            
            u_pred = model(a).squeeze(1) # La prédiction du modèle, shape: (B, pNt, pNs)
            
            # Récupération des dimensions
            B, pNt, pNs = u_pred.shape 
            # Ns est la taille du domaine spatial original [0, r_max] (correspond à Nr+1 points)
            # L'axe spatial complet va de -r_max à +r_max, donc pNs = 2 * Ns
            # Le centre (r=0) se trouve à l'indice Ns
            Ns = pNs // 2

            # Le tenseur d'entrée 'a' (shape: B, 3, pNt, pNs) contient les cartes des paramètres C, D, T1.
            # On utilise le canal 0 (C) pour trouver la frontière R.
            # On observe une tranche du quadrant (t>0, r>0), par exemple à a[:, 0, pNt//2, Ns:].
            c_map_positive_r_batch = a[:, 0, pNt // 2, Ns:] # Shape: (B, Ns)
            
            batch_abs_error = 0.0
            for i in range(B):
                c_map = c_map_positive_r_batch[i]
                val_in = c_map[0] # Valeur du paramètre à l'intérieur du rayon R

                # On trouve l'indice du premier point où la valeur change.
                # (c_map != val_in) crée un tenseur de booléens (False/True).
                # .int().argmax() trouve l'index du premier True.
                first_change_idx = (c_map != val_in).int().argmax()

                # Si argmax renvoie 0, cela peut signifier soit que le changement est au premier indice,
                # soit qu'il n'y a aucun changement. On vérifie le dernier élément pour les distinguer.
                if first_change_idx == 0 and c_map[-1] == val_in:
                    # Aucun changement détecté, R couvre tout le domaine.
                    num_points_in_R = Ns
                else:
                    # Le nombre de points dans le rayon R sur l'axe positif.
                    num_points_in_R = first_change_idx.item()

                # Définition des bornes pour le slicing sur la bande [-R, R]
                # L'axe spatial est centré sur Ns.
                NR_min = Ns - num_points_in_R
                NR_max = Ns + num_points_in_R

                # On extrait la tranche correspondant à [-R, R]
                pred_slice = u_pred[i, :, NR_min:NR_max]
                true_slice = u_true[i, :, NR_min:NR_max]

                if pred_slice.numel() > 0:
                    G_pred = torch.mean(pred_slice)
                    G_true = torch.mean(true_slice)
                    batch_abs_error += torch.abs(G_pred - G_true)

            total_error_G += (batch_abs_error / B)
            
    return total_error_G.item() / len(dataloader)


def G_accuracy_(model: torch.nn.Module,
               dataloader: torch.utils.data.DataLoader,
               device: torch.device) -> float:
    
    model.eval()
    
    total_error_G: float = .0

    for a, u_true in dataloader:
        a, u_true = a.to(device), u_true.to(device)
        
        u_pred = model(a).squeeze(1)
        Nt, Nr = u_pred.shape[2], u_pred.shape[3]

        NR_min = 0
        NR_max = 0
        
        G_pred = torch.mean(u_pred[..., NR_min:NR_max, :])
        G_true = torch.mean(u_true[..., NR_min:NR_max, :])
        
        total_error_G += torch.mean(torch.abs(G_pred - G_true))
        
    return total_error_G / len(dataloader)



def G_accuracy_old(model: torch.nn.Module,
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
