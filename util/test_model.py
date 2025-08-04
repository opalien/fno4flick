from __future__ import annotations

import os
import json
import math
from typing import List, Dict, Any

import numpy as np
import torch
from torch import Tensor
import matplotlib.pyplot as plt

from models.model import FickModel, postprocess
from util.fick_params import FickParams
from util.G_method import compute_G_in, compute_G_out


def _to_numpy(t: Tensor) -> np.ndarray:
    """Détache un tenseur, le déplace vers le CPU et le convertit en tableau numpy."""
    return t.detach().cpu().numpy()


def _interp_time_space_P_to_2channels(
    P: Tensor,
    R: float, r_max: float, t_max: float,
    Nt_tgt: int = 100, Nr_tgt: int = 100,
) -> Tensor:
    """
    Construit un tenseur (2, Nt_tgt, Nr_tgt) = [P_in(s,t), P_out(eta,t)],
    avec s, eta ∈ [0,1], en interpolant P(r,t) défini sur [0,r_max]×[0,t_max].
    """
    P = P.detach().cpu().float()
    Nt_src, Ns_src = P.shape

    t_src = np.linspace(0.0, float(t_max), Nt_src, dtype=np.float64)
    r_src = np.linspace(0.0, float(r_max), Ns_src, dtype=np.float64)

    t_tgt = np.linspace(0.0, float(t_max), Nt_tgt, dtype=np.float64)
    r_in  = np.linspace(0.0, float(R),   Nr_tgt, dtype=np.float64)
    r_out = np.linspace(float(R), float(r_max), Nr_tgt, dtype=np.float64)

    P_np = P.numpy().astype(np.float64)
    P_t = np.empty((Nt_tgt, Ns_src), dtype=np.float64)
    for j in range(Ns_src):
        P_t[:, j] = np.interp(t_tgt, t_src, P_np[:, j])

    P_in  = np.stack([np.interp(r_in,  r_src, P_t[t, :]) for t in range(Nt_tgt)], axis=0)
    P_out = np.stack([np.interp(r_out, r_src, P_t[t, :]) for t in range(Nt_tgt)], axis=0)

    P2c = np.stack([P_in, P_out], axis=0).astype(np.float32)
    return torch.from_numpy(P2c)


def run_diagnostics(model: FickModel, dataset: torch.utils.data.Dataset, device: torch.device, out_dir: str,
                    Nt_tgt: int = 100, Nr_tgt: int = 100) -> List[dict]:
    """
    Exécute des diagnostics sur chaque échantillon du dataset, génère des graphiques
    et sauvegarde les métriques de performance.
    """
    os.makedirs(out_dir, exist_ok=True)
    per_sample_dir = os.path.join(out_dir, "per_sample")
    os.makedirs(per_sample_dir, exist_ok=True)

    results = []
    model.eval()
    model.to(device)

    for i, (p_raw, P_full) in enumerate(dataset.elements):
        try:
            p_root: FickParams = p_raw.get_root_parent()
            R_true = float(p_root.R.item())
            rmax_true = float(p_root.r_max.item())
            tmax_true = float(p_root.t_max.item())

            P2c_true = _interp_time_space_P_to_2channels(
                P=P_full, R=R_true, r_max=rmax_true, t_max=tmax_true,
                Nt_tgt=Nt_tgt, Nr_tgt=Nr_tgt
            )
            P2c_true_b = P2c_true.unsqueeze(0).to(device)

            p_for_pred = p_root.to(device)
            p_for_pred.Nt = torch.tensor(Nt_tgt, device=device)
            p_for_pred.Nr = torch.tensor(Nr_tgt, device=device)
            
            fick_fno = model.forward([p_for_pred])
            P2c_pred_b = postprocess(fick_fno)

            G_in_true  = compute_G_in(P2c_true_b)
            G_in_pred  = compute_G_in(P2c_pred_b)
            G_out_true = compute_G_out(P2c_true_b, [p_root])
            G_out_pred = compute_G_out(P2c_pred_b, [p_root])

            def rel_err(a: Tensor, b: Tensor) -> float:
                num = torch.linalg.norm((a - b).reshape(-1)).item()
                den = torch.linalg.norm(b.reshape(-1)).item() + 1e-12
                return num / den

            Pin_rel  = rel_err(P2c_pred_b[:, 0], P2c_true_b[:, 0])
            Pout_rel = rel_err(P2c_pred_b[:, 1], P2c_true_b[:, 1])
            Gin_rel  = rel_err(G_in_pred, G_in_true)
            Gout_rel = rel_err(G_out_pred, G_out_true)

            Rin_pred_val, Rout_pred_val = None, None
            if hasattr(model, "search_R_in"):
                try:
                    Rin_list = model.search_R_in([p_root], G_in_true, Nt_tgt, Nr_tgt)
                    Rin_pred_val = float(Rin_list[0].item())
                except Exception as e:
                    print(f"[WARN] search_R_in a échoué pour l'échantillon {i}: {e}")
            
            if hasattr(model, "search_R_out"):
                try:
                    Rout_list = model.search_R_out([p_root], G_out_true, Nt_tgt, Nr_tgt)
                    Rout_pred_val = float(Rout_list[0].item())
                except Exception as e:
                    print(f"[WARN] search_R_out a échoué pour l'échantillon {i}: {e}")

            # Calcul des ratios et autres métriques
            Cin, Cout = float(p_root.C_in.item()), float(p_root.C_out.item())
            Din, Dout = float(p_root.D_in.item()), float(p_root.D_out.item())
            T1in, T1out = float(p_root.T1_in.item()), float(p_root.T1_out.item())
            P0in, P0out = float(p_root.P0_in.item()), float(p_root.P0_out.item())

            R_over_rmax = R_true / (rmax_true + 1e-12)
            Din_over_Dout = Din / (Dout + 1e-12)
            Cin_over_Cout = Cin / (Cout + 1e-12)
            T1in_over_T1out = T1in / (T1out + 1e-12)
            P0in_over_P0out = P0in / (P0out + 1e-12)
            Lin_eff = np.sqrt(Din * min(T1in, tmax_true))
            Lout_eff = np.sqrt(Dout * min(T1out, tmax_true))

            # Création de la figure de diagnostic
            fig, axes = plt.subplots(3, 3, figsize=(20, 16))
            
            im = axes[0, 0].imshow(_to_numpy(P2c_true[0]), origin="lower", aspect="auto", cmap="viridis")
            axes[0, 0].set_title("P_in (Vrai)")
            fig.colorbar(im, ax=axes[0, 0])

            im = axes[0, 1].imshow(_to_numpy(P2c_pred_b.squeeze(0)[0]), origin="lower", aspect="auto", cmap="viridis")
            axes[0, 1].set_title(f"P_in (Prédit) | err={Pin_rel:.3e}")
            fig.colorbar(im, ax=axes[0, 1])
            
            im = axes[0, 2].imshow(_to_numpy(P2c_pred_b.squeeze(0)[0] - P2c_true[0]), origin="lower", aspect="auto", cmap="coolwarm")
            axes[0, 2].set_title("Erreur P_in")
            fig.colorbar(im, ax=axes[0, 2])

            im = axes[1, 0].imshow(_to_numpy(P2c_true[1]), origin="lower", aspect="auto", cmap="viridis")
            axes[1, 0].set_title("P_out (Vrai)")
            fig.colorbar(im, ax=axes[1, 0])

            im = axes[1, 1].imshow(_to_numpy(P2c_pred_b.squeeze(0)[1]), origin="lower", aspect="auto", cmap="viridis")
            axes[1, 1].set_title(f"P_out (Prédit) | err={Pout_rel:.3e}")
            fig.colorbar(im, ax=axes[1, 1])

            im = axes[1, 2].imshow(_to_numpy(P2c_pred_b.squeeze(0)[1] - P2c_true[1]), origin="lower", aspect="auto", cmap="coolwarm")
            axes[1, 2].set_title("Erreur P_out")
            fig.colorbar(im, ax=axes[1, 2])

            axes[2, 0].plot(_to_numpy(G_in_true.squeeze(0)), label="G_in vrai")
            axes[2, 0].plot(_to_numpy(G_in_pred.squeeze(0)), label="G_in prédit", linestyle="--")
            axes[2, 0].set_title(f"G_in(t) | err={Gin_rel:.3e}")
            axes[2, 0].legend()

            axes[2, 1].plot(_to_numpy(G_out_true.squeeze(0)), label="G_out vrai")
            axes[2, 1].plot(_to_numpy(G_out_pred.squeeze(0)), label="G_out prédit", linestyle="--")
            axes[2, 1].set_title(f"G_out(t) | err={Gout_rel:.3e}")
            axes[2, 1].legend()

            axes[2, 2].axis("off")
            txt = (
                f"Échantillon #{i}\n"
                f"R={R_true:.4g}, r_max={rmax_true:.4g}, R/r_max={R_over_rmax:.3f}\n"
                f"C_in/C_out={Cin_over_Cout:.3f}, D_in/D_out={Din_over_Dout:.3f}\n"
                f"T1_in/T1_out={T1in_over_T1out:.3f}, P0_in/P0_out={P0in_over_P0out:.3f}\n"
                f"L_eff_in={Lin_eff:.4g}, L_eff_out={Lout_eff:.4g}\n"
            )
            if Rin_pred_val is not None:
                txt += f"R_in_pred={Rin_pred_val:.4g}, |Δ|/R={abs(Rin_pred_val-R_true)/(R_true+1e-12):.3e}\n"
            if Rout_pred_val is not None:
                txt += f"R_out_pred={Rout_pred_val:.4g}, |Δ|/R={abs(Rout_pred_val-R_true)/(R_true+1e-12):.3e}\n"
            axes[2, 2].text(0, 0.5, txt, va="center", family="monospace", fontsize=10)

            fig.suptitle(f"Diagnostic Échantillon #{i}", fontsize=16)
            fig.tight_layout(rect=[0, 0.03, 1, 0.95])
            fig.savefig(os.path.join(per_sample_dir, f"sample_{i:04d}.png"))
            plt.close(fig)

            results.append({
                "idx": i, "Pin_rel": Pin_rel, "Pout_rel": Pout_rel, "Gin_rel": Gin_rel, "Gout_rel": Gout_rel,
                "R_true": R_true, "R_in_pred": Rin_pred_val, "R_out_pred": Rout_pred_val,
                "R_over_rmax": R_over_rmax, "Cin_over_Cout": Cin_over_Cout, "Din_over_Dout": Din_over_Dout,
                "T1in_over_T1out": T1in_over_T1out, "P0in_over_P0out": P0in_over_P0out,
                "Lin_eff": Lin_eff, "Lout_eff": Lout_eff, "t_max": tmax_true
            })

        except Exception as e:
            print(f"[ERREUR] Le diagnostic a échoué pour l'échantillon {i}: {e}")
            import traceback
            traceback.print_exc()

    with open(os.path.join(out_dir, "diagnostics_metrics.jsonl"), "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")

    return results


def _vec(results: List[Dict[str, Any]], key: str) -> np.ndarray:
    """Extrait une colonne de données des résultats sous forme de tableau numpy."""
    return np.array([r.get(key, np.nan) for r in results], dtype=float)


def _finite_mask(*arrays: np.ndarray) -> np.ndarray:
    """Crée un masque booléen pour les valeurs finies communes à plusieurs tableaux."""
    mask = np.ones_like(arrays[0], dtype=bool)
    for a in arrays:
        mask &= np.isfinite(a)
    return mask


def plot_correlations(results: List[Dict[str, Any]], out_dir: str) -> None:
    """
    Génère des nuages de points pour visualiser les corrélations entre les erreurs
    de prédiction et les paramètres physiques du problème.
    """
    os.makedirs(out_dir, exist_ok=True)
    if not results:
        print("Aucun résultat à afficher pour les corrélations.")
        return

    R_true = _vec(results, "R_true")
    Rin_pred = _vec(results, "R_in_pred")
    Rout_pred = _vec(results, "R_out_pred")

    Y_dict = {
        "Erreur P_in": _vec(results, "Pin_rel"),
        "Erreur P_out": _vec(results, "Pout_rel"),
        "Erreur G_in": _vec(results, "Gin_rel"),
        "Erreur G_out": _vec(results, "Gout_rel"),
        "Erreur R_in": np.abs(Rin_pred - R_true) / (np.abs(R_true) + 1e-12),
        "Erreur R_out": np.abs(Rout_pred - R_true) / (np.abs(R_true) + 1e-12),
    }

    X_dict = {
        "R/r_max": _vec(results, "R_over_rmax"),
        "D_in/D_out": _vec(results, "Din_over_Dout"),
        "C_in/C_out": _vec(results, "Cin_over_Cout"),
        "T1_in/T1_out": _vec(results, "T1in_over_T1out"),
        "L_eff_in": _vec(results, "Lin_eff"),
        "L_eff_out": _vec(results, "Lout_eff"),
    }

    for x_label, x_vals in X_dict.items():
        for y_label, y_vals in Y_dict.items():
            mask = _finite_mask(x_vals, y_vals)
            if np.sum(mask) < 2: continue

            xv, yv = x_vals[mask], y_vals[mask]
            r = np.corrcoef(xv, yv)[0, 1]

            plt.figure(figsize=(8, 6))
            plt.scatter(xv, yv, alpha=0.6)
            plt.xlabel(x_label)
            plt.ylabel(y_label)
            plt.title(f"{y_label} vs {x_label} (r={r:.3f})")
            plt.yscale("log")
            plt.grid(True, which="both", ls="--")
            plt.tight_layout()
            
            clean_xlabel = x_label.replace('/', 'sur')
            clean_ylabel = y_label.replace(' ', '_')
            fname = f"{clean_ylabel}_vs_{clean_xlabel}.png"
            plt.savefig(os.path.join(out_dir, fname))
            plt.close()

    print(f"Graphiques de corrélation sauvegardés dans : {out_dir}")




