from __future__ import annotations

import os
import json
from typing import List, Dict, Any, Tuple

import numpy as np
import torch
from torch import Tensor
import matplotlib.pyplot as plt

from models.model import FickModel, postprocess
from util.fick_params import FickParams
from util.G_method import compute_G_in, compute_G_out


def _to_numpy(t: Tensor) -> np.ndarray:
    return t.detach().cpu().numpy()


def _ensure_pair(elem) -> Tuple[FickParams, Tensor]:
    if isinstance(elem, tuple) and len(elem) >= 2:
        params, P = elem[0], elem[1]
        if not isinstance(params, FickParams):
            raise TypeError(f"Premier élément n'est pas un FickParams: {type(params)}")
        if not isinstance(P, Tensor):
            raise TypeError(f"Deuxième élément n'est pas un Tensor: {type(P)}")
        return params, P
    raise ValueError(f"Élément du dataset invalide (attendu tuple(FickParams, Tensor)), reçu: {type(elem)}")


def _rel_err(a: Tensor, b: Tensor) -> float:
    num = torch.linalg.norm((a - b).reshape(-1)).item()
    den = torch.linalg.norm(b.reshape(-1)).item() + 1e-12
    return num / den

def run_diagnostics(
    model: FickModel,
    dataset: torch.utils.data.Dataset,
    device: torch.device,
    out_dir: str,
) -> List[dict]:
    """
    Génère, pour chaque échantillon du dataset, un diagnostic visuel et des
    métriques d'erreur. Les tenseurs P_true et P_pred sont maintenant
    DÉ-NORMALISÉS avant le calcul de G_in/out et des erreurs.
    """
    os.makedirs(out_dir, exist_ok=True)
    per_sample_dir = os.path.join(out_dir, "per_sample")
    os.makedirs(per_sample_dir, exist_ok=True)

    results: List[Dict[str, Any]] = []

    model.eval()
    model.to(device)

    for i, elem in enumerate(getattr(dataset, "elements", [])):
        try:
            # ---------------- lecture / préparation ----------------
            p_raw, P_2c_true = _ensure_pair(elem)  # (2, Nt, Nr)
            if P_2c_true.ndim != 3 or P_2c_true.shape[0] != 2:
                raise ValueError(f"P_2c_true doit être (2, Nt, Nr), reçu {tuple(P_2c_true.shape)}")

            Nt = int(p_raw.Nt.item()) if isinstance(p_raw.Nt, Tensor) else int(p_raw.Nt)
            Nr = int(p_raw.Nr.item()) if isinstance(p_raw.Nr, Tensor) else int(p_raw.Nr)

            # --- paramètres racine (pour valeurs réelles / affichage) ---
            p_root: FickParams = p_raw.get_root_parent()
            R_true   = float(p_root.R.item()) if isinstance(p_root.R, Tensor) else float(p_root.R)
            rmax     = float(p_root.r_max.item()) if isinstance(p_root.r_max, Tensor) else float(p_root.r_max)
            tmax     = float(p_root.t_max.item()) if isinstance(p_root.t_max, Tensor) else float(p_root.t_max)
            Cin, Cout   = float(p_root.C_in.item()), float(p_root.C_out.item())
            Din, Dout   = float(p_root.D_in.item()), float(p_root.D_out.item())
            T1in, T1out = float(p_root.T1_in.item()), float(p_root.T1_out.item())
            P0in, P0out = float(p_root.P0_in.item()), float(p_root.P0_out.item())

            # --------------------- prédiction modèle ---------------------
            p_for_pred = p_raw.to(device)
            p_for_pred.Nt = torch.tensor(Nt, device=device)
            p_for_pred.Nr = torch.tensor(Nr, device=device)

            with torch.no_grad():
                fick_fno = model.forward([p_for_pred])     # (B=1, 2, 2Nt, 2Nr)
                fick = postprocess(fick_fno)               # (B=1, 2, Nt, Nr)

            # ------------------- DÉ-NORMALISATION -------------------
            P_true_b = P_2c_true.unsqueeze(0).to(device)          # (B=1, 2, Nt, Nr)  normalisé
            P_pred_b = fick                                        # (B=1, 2, Nt, Nr)  normalisé
            P_true_b = model.P_normalizer.unnormalize(P_true_b)    # ← nouveau
            P_pred_b = model.P_normalizer.unnormalize(P_pred_b)    # ← nouveau
            # -------------------------------------------------------

            # ------------------- métriques G / erreurs -------------------
            with torch.no_grad():
                G_in_true  = compute_G_in(P_true_b)                    # (B=1, Nt)
                G_in_pred  = compute_G_in(P_pred_b)                    # (B=1, Nt)
                G_out_true = compute_G_out(P_true_b, [p_raw])          # (B=1, Nt)
                G_out_pred = compute_G_out(P_pred_b, [p_raw])          # (B=1, Nt)

            Pin_rel  = _rel_err(P_pred_b[:, 0], P_true_b[:, 0])
            Pout_rel = _rel_err(P_pred_b[:, 1], P_true_b[:, 1])
            Gin_rel  = _rel_err(G_in_pred, G_in_true)
            Gout_rel = _rel_err(G_out_pred, G_out_true)

            # ------------------- recherche de R -------------------
            Rin_pred_val, Rout_pred_val = None, None
            if hasattr(model, "search_R_in"):
                try:
                    Rin_list = model.search_R_in([p_raw], G_in_true, Nt, Nr)
                    Rin_pred_dimless = float(Rin_list[0].detach().cpu().item())
                    Rin_pred_val = Rin_pred_dimless * rmax
                except Exception as e:
                    print(f"[WARN] search_R_in a échoué pour échantillon {i}: {e}")
            if hasattr(model, "search_R_out"):
                try:
                    Rout_list = model.search_R_out([p_raw], G_out_true, Nt, Nr)
                    Rout_pred_dimless = float(Rout_list[0].detach().cpu().item())
                    Rout_pred_val = Rout_pred_dimless * rmax
                except Exception as e:
                    print(f"[WARN] search_R_out a échoué pour échantillon {i}: {e}")

            # ------------------- ratios / longueurs utiles -------------------
            R_over_rmax     = R_true / (rmax + 1e-12)
            Din_over_Dout   = Din / (Dout + 1e-12)
            Cin_over_Cout   = Cin / (Cout + 1e-12)
            T1in_over_T1out = T1in / (T1out + 1e-12)
            P0in_over_P0out = P0in / (P0out + 1e-12)
            Lin_eff  = np.sqrt(max(Din, 0.0)  * max(min(T1in,  tmax), 0.0))
            Lout_eff = np.sqrt(max(Dout, 0.0) * max(min(T1out, tmax), 0.0))

            # ------------------- graphiques -------------------
            P_true_np = _to_numpy(P_true_b.squeeze(0))
            P_pred_np = _to_numpy(P_pred_b.squeeze(0))
            Gin_t = _to_numpy(G_in_true.squeeze(0))
            Gin_p = _to_numpy(G_in_pred.squeeze(0))
            Gout_t = _to_numpy(G_out_true.squeeze(0))
            Gout_p = _to_numpy(G_out_pred.squeeze(0))

            fig, axes = plt.subplots(3, 3, figsize=(20, 16))

            im = axes[0, 0].imshow(P_true_np[0], origin="lower", aspect="auto")
            axes[0, 0].set_title("P_in (Vrai)")
            fig.colorbar(im, ax=axes[0, 0])

            im = axes[0, 1].imshow(P_pred_np[0], origin="lower", aspect="auto")
            axes[0, 1].set_title(f"P_in (Prédit) | err={Pin_rel:.3e}")
            fig.colorbar(im, ax=axes[0, 1])

            im = axes[0, 2].imshow(P_pred_np[0] - P_true_np[0], origin="lower", aspect="auto")
            axes[0, 2].set_title("Erreur P_in")
            fig.colorbar(im, ax=axes[0, 2])

            im = axes[1, 0].imshow(P_true_np[1], origin="lower", aspect="auto")
            axes[1, 0].set_title("P_out (Vrai)")
            fig.colorbar(im, ax=axes[1, 0])

            im = axes[1, 1].imshow(P_pred_np[1], origin="lower", aspect="auto")
            axes[1, 1].set_title(f"P_out (Prédit) | err={Pout_rel:.3e}")
            fig.colorbar(im, ax=axes[1, 1])

            im = axes[1, 2].imshow(P_pred_np[1] - P_true_np[1], origin="lower", aspect="auto")
            axes[1, 2].set_title("Erreur P_out")
            fig.colorbar(im, ax=axes[1, 2])

            axes[2, 0].plot(Gin_t, label="G_in vrai")
            axes[2, 0].plot(Gin_p, label="G_in prédit", linestyle="--")
            axes[2, 0].set_title(f"G_in(t) | err={Gin_rel:.3e}")
            axes[2, 0].legend()

            axes[2, 1].plot(Gout_t, label="G_out vrai")
            axes[2, 1].plot(Gout_p, label="G_out prédit", linestyle="--")
            axes[2, 1].set_title(f"G_out(t) | err={Gout_rel:.3e}")
            axes[2, 1].legend()

            axes[2, 2].axis("off")
            txt = (
                f"Échantillon #{i}\n"
                f"R={R_true:.4g}, r_max={rmax:.4g}, R/r_max={R_over_rmax:.3f}\n"
                f"C_in/C_out={Cin_over_Cout:.3f}, D_in/D_out={Din_over_Dout:.3f}\n"
                f"T1_in/T1_out={T1in_over_T1out:.3f}, P0_in/P0_out={P0in_over_P0out:.3f}\n"
                f"L_eff_in={Lin_eff:.4g}, L_eff_out={Lout_eff:.4g}\n"
            )
            if Rin_pred_val is not None:
                txt += f"R_in_pred={Rin_pred_val:.4g}, |Δ|/R={abs(Rin_pred_val - R_true)/(R_true+1e-12):.3e}\n"
            if Rout_pred_val is not None:
                txt += f"R_out_pred={Rout_pred_val:.4g}, |Δ|/R={abs(Rout_pred_val - R_true)/(R_true+1e-12):.3e}\n"
            axes[2, 2].text(0, 0.5, txt, va="center", family="monospace", fontsize=10)

            fig.suptitle(f"Diagnostic Échantillon #{i}", fontsize=16)
            fig.tight_layout(rect=[0, 0.03, 1, 0.95])
            fig.savefig(os.path.join(per_sample_dir, f"sample_{i:04d}.png"))
            plt.close(fig)

            # ------------------- stockage résultats -------------------
            result = {
                "idx": i,
                "Pin_rel": Pin_rel,
                "Pout_rel": Pout_rel,
                "Gin_rel": Gin_rel,
                "Gout_rel": Gout_rel,
                "R_true": R_true,
                "R_in_pred": Rin_pred_val,
                "R_out_pred": Rout_pred_val,
                "R_over_rmax": R_over_rmax,
                "Cin_over_Cout": Cin_over_Cout,
                "Din_over_Dout": Din_over_Dout,
                "T1in_over_T1out": T1in_over_T1out,
                "P0in_over_P0out": P0in_over_P0out,
                "Lin_eff": Lin_eff,
                "Lout_eff": Lout_eff,
                "t_max": tmax,
                "Nt": Nt,
                "Nr": Nr,
            }
            results.append(result)

        except Exception as e:
            print(f"[ERREUR] Le diagnostic a échoué pour l'échantillon {i}: {e}")
            import traceback
            traceback.print_exc()

    # ------------------- écriture JSONL -------------------
    jsonl_path = os.path.join(out_dir, "diagnostics_metrics.jsonl")
    with open(jsonl_path, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")

    return results


def _vec(results: List[Dict[str, Any]], key: str) -> np.ndarray:
    return np.array([r.get(key, np.nan) for r in results], dtype=float)


def _finite_mask(*arrays: np.ndarray) -> np.ndarray:
    mask = np.ones_like(arrays[0], dtype=bool)
    for a in arrays:
        mask &= np.isfinite(a)
    return mask


def plot_correlations(results: List[Dict[str, Any]], out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    if not results:
        print("Aucun résultat pour les corrélations.")
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
            if np.sum(mask) < 2:
                continue
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

            clean_xlabel = x_label.replace('/', 'sur').replace(' ', '_')
            clean_ylabel = y_label.replace(' ', '_')
            fname = f"{clean_ylabel}_vs_{clean_xlabel}.png"
            plt.savefig(os.path.join(out_dir, fname))
            plt.close()

    print(f"Graphiques de corrélation sauvegardés dans : {out_dir}")

