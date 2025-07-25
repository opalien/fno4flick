# experiments/fno2d/some_tests.py
from __future__ import annotations

import os
from copy import deepcopy
from typing import Any, Dict, List, Tuple

import json
import math
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import Tensor
from neuralop.models import FNO

from experiments.fno2d.dataset import Dataset, collate_fn
from experiments.fno2d.edp_parameters import EDPParameters
from experiments.fno2d.integrator import compute_G


def _to_numpy(t: Tensor) -> np.ndarray:
    return t.detach().cpu().numpy()


@torch.no_grad()
def run_diagnostics(
    model: FNO,
    dataset: Dataset,
    device: torch.device,
    out_dir: str,
    tag: str,
    r_max_fixed: bool = False,
    R_scan_points: int = 25,
    R_scan_factor: float = 0.5,
) -> List[Dict[str, Any]]:
    """
    Pour chaque élément du dataset :
      - Calcule erreurs P et G (absolue et relative),
      - Scanne R autour de la vraie valeur, trace l'erreur G vs R, et récupère R_best,
      - Produit une figure multi-lignes répondant exactement aux 4 points demandés,
      - Sauvegarde un jsonl des métriques + paramètres.
    """
    os.makedirs(out_dir, exist_ok=True)
    per_sample_dir = os.path.join(out_dir, "per_sample")
    os.makedirs(per_sample_dir, exist_ok=True)

    model.eval().to(device)

    results: List[Dict[str, Any]] = []

    for i, (params_transfo, P_quadrant) in enumerate(dataset.elements):
        try:
            # --------------- Préparation ---------------
            params_root: EDPParameters = params_transfo.get_root_parent()

            params_tensored, P_full_target = collate_fn([(params_transfo, P_quadrant)])
            params_tensored = params_tensored.to(device)
            P_full_target = P_full_target.to(device).squeeze(0)  # (pNt, pNr)

            # --------------- Prédiction ---------------
            P_full_pred = model(params_tensored).squeeze(0).squeeze(0)

            # --------------- Erreurs P ---------------
            P_err = P_full_pred - P_full_target
            P_norm_abs = torch.linalg.norm(P_err.reshape(-1)).item()
            denom_P = torch.linalg.norm(P_full_target.reshape(-1)).item()
            P_norm_rel = P_norm_abs / (denom_P + 1e-12)

            # --------------- Erreurs G ---------------
            G_true = compute_G(P_full_target, params_transfo)  # (Nt,)
            G_pred = compute_G(P_full_pred, params_transfo)    # (Nt,)
            G_err = G_pred - G_true
            G_norm_abs = torch.linalg.norm(G_err.reshape(-1)).item()
            denom_G = torch.linalg.norm(G_true.reshape(-1)).item()
            G_norm_rel = G_norm_abs / (denom_G + 1e-12)

            # --------------- Scan R ---------------
            R_true = float(params_root.R)
            r_max_true = float(params_root.r_max)
            R_ratio_true = R_true / (r_max_true + 1e-12)

            if r_max_fixed:
                R_min = max(1e-9, R_true * (1.0 - R_scan_factor))
                R_max = R_true * (1.0 + R_scan_factor)
            else:
                R_min = max(1e-9, R_true * (1.0 - R_scan_factor))
                R_max = R_true * (1.0 + R_scan_factor)

            R_grid = torch.linspace(
                R_min, R_max, R_scan_points, device=device, dtype=P_full_target.dtype
            )
            scan_errors: List[float] = []

            for R_guess in R_grid:
                # Params "guess" au niveau root
                params_guess_root = deepcopy(params_root)
                params_guess_root.R = float(R_guess.item())

                # Pipeline de normalisation identique à celle appliquée au dataset
                params_guess = (
                    params_guess_root
                    .rescaling()
                    .nondimensionalize()
                    .compression()
                    .normalize(
                        dataset.C_normalizer.normalize,
                        dataset.D_normalizer.normalize,
                        None,
                        dataset.T1_normalizer.normalize
                    )
                )

                params_guess_tensored, _ = collate_fn([(params_guess, P_quadrant)])
                params_guess_tensored = params_guess_tensored.to(device)

                P_pred_guess = model(params_guess_tensored).squeeze(0).squeeze(0)
                G_pred_guess = compute_G(P_pred_guess, params_guess)

                scan_err = torch.linalg.norm((G_pred_guess - G_true).reshape(-1)).item()
                scan_errors.append(scan_err)

            scan_errors_np = np.array(scan_errors)
            R_grid_np = _to_numpy(R_grid)
            best_idx = int(scan_errors_np.argmin())
            R_best = float(R_grid_np[best_idx])
            R_abs_err = abs(R_best - R_true)
            R_rel_err = R_abs_err / (abs(R_true) + 1e-12)

            # --------------- Ratios demandés ---------------
            Cin_over_Cout = float(params_root.C_in) / (float(params_root.C_out) + 1e-12)
            Din_over_Dout = float(params_root.D_in) / (float(params_root.D_out) + 1e-12)
            T1in_over_T1out = float(params_root.T1_in) / (float(params_root.T1_out) + 1e-12)

            # --------------- Figure multi-lignes ---------------
            fig = plt.figure(figsize=(18, 16))
            gs = fig.add_gridspec(
                4, 3,
                height_ratios=[1.2, 2.4, 1.2, 0.9]
            )

            # --- Ligne 1) G_true vs G_pred + G_err vs t ---
            ax_g = fig.add_subplot(gs[0, 0:2])
            ax_g.plot(_to_numpy(G_true), label="G_true")
            ax_g.plot(_to_numpy(G_pred), label="G_pred", linestyle="--")
            ax_g.set_title(f"G(t)  |  ||G_err||_rel = {G_norm_rel:.3e}")
            ax_g.set_xlabel("t (index)")
            ax_g.legend()

            ax_g_err = fig.add_subplot(gs[0, 2])
            ax_g_err.plot(_to_numpy(G_err))
            ax_g_err.set_title("G_err(t) = G_pred - G_true")
            ax_g_err.set_xlabel("t (index)")

            # --- Ligne 2) P_true, P_pred, P_err heatmaps ---
            ax_p_true = fig.add_subplot(gs[1, 0])
            im_true = ax_p_true.imshow(_to_numpy(P_full_target), origin="lower", aspect="auto", cmap="viridis")
            ax_p_true.set_title("P_true(r, t)")
            fig.colorbar(im_true, ax=ax_p_true, fraction=0.046, pad=0.04)

            ax_p_pred = fig.add_subplot(gs[1, 1])
            im_pred = ax_p_pred.imshow(_to_numpy(P_full_pred), origin="lower", aspect="auto", cmap="viridis")
            ax_p_pred.set_title("P_pred(r, t)")
            fig.colorbar(im_pred, ax=ax_p_pred, fraction=0.046, pad=0.04)

            ax_p_err = fig.add_subplot(gs[1, 2])
            im_err = ax_p_err.imshow(_to_numpy(P_err), origin="lower", aspect="auto", cmap="coolwarm")
            ax_p_err.set_title(f"P_err(r, t) | ||P_err||_rel = {P_norm_rel:.3e}")
            fig.colorbar(im_err, ax=ax_p_err, fraction=0.046, pad=0.04)

            # --- Ligne 3) erreur vs R scanné ---
            ax_rscan = fig.add_subplot(gs[2, :])
            ax_rscan.plot(R_grid_np, scan_errors_np, marker="o")
            ax_rscan.axvline(R_true, color="r", linestyle="--", label=f"R_true={R_true:.4g}")
            ax_rscan.axvline(R_best, color="g", linestyle="--", label=f"R_best={R_best:.4g}")
            ax_rscan.set_title(f"Scan R - best @ {R_best:.4g} | abs_err={R_abs_err:.3e}")
            ax_rscan.set_xlabel("R guess")
            ax_rscan.set_ylabel("||G_pred(R) - G_true||")
            ax_rscan.legend()

            # --- Ligne 4) Texte paramétrique ---
            ax_txt = fig.add_subplot(gs[3, :])
            ax_txt.axis("off")

            txt = (
                f"[{tag}] sample #{i}\n"
                f"--- Errors ---\n"
                f"P_norm_abs={P_norm_abs:.3e}, P_norm_rel={P_norm_rel:.3e}\n"
                f"G_norm_abs={G_norm_abs:.3e}, G_norm_rel={G_norm_rel:.3e}\n"
                f"R_true={R_true:.6g}, r_max_true={r_max_true:.6g}, R/r_max={R_ratio_true:.6g}\n"
                f"R_best={R_best:.6g}, |R_best-R_true|={R_abs_err:.3e} (rel={R_rel_err:.3e})\n"
                f"--- Root params ---\n"
                f"C_in={params_root.C_in:.6g}, C_out={params_root.C_out:.6g}, C_in/C_out={Cin_over_Cout:.6g}\n"
                f"D_in={params_root.D_in:.6g}, D_out={params_root.D_out:.6g}, D_in/D_out={Din_over_Dout:.6g}\n"
                f"T1_in={params_root.T1_in:.6g}, T1_out={params_root.T1_out:.6g}, T1_in/T1_out={T1in_over_T1out:.6g}\n"
                f"P0_in={params_root.P0_in:.6g}, P0_out={params_root.P0_out:.6g}\n"
            )
            ax_txt.text(0.01, 0.95, txt, va="top", ha="left", family="monospace", fontsize=9)

            fig.suptitle(f"[{tag}] Diagnostic sample #{i}", fontsize=12)
            fig.tight_layout()
            fig.savefig(os.path.join(per_sample_dir, f"sample_{i:05d}.png"), dpi=180)
            plt.close(fig)

            # --------------- Résultats (pour corrélations) ---------------
            res: Dict[str, Any] = {
                "idx": i,
                # erreurs
                "P_norm_abs": P_norm_abs,
                "P_norm_rel": P_norm_rel,
                "G_norm_abs": G_norm_abs,
                "G_norm_rel": G_norm_rel,
                # paramètres root
                "R_true": R_true,
                "r_max_true": r_max_true,
                "R_over_rmax": R_ratio_true,
                "C_in": float(params_root.C_in),
                "C_out": float(params_root.C_out),
                "D_in": float(params_root.D_in),
                "D_out": float(params_root.D_out),
                "T1_in": float(params_root.T1_in),
                "T1_out": float(params_root.T1_out),
                "P0_in": float(params_root.P0_in),
                "P0_out": float(params_root.P0_out),
                # ratios
                "Cin_over_Cout": Cin_over_Cout,
                "Din_over_Dout": Din_over_Dout,
                "T1in_over_T1out": T1in_over_T1out,
                # scan R
                "R_best": R_best,
                "R_abs_err": R_abs_err,
                "R_rel_err": R_rel_err,
            }
            results.append(res)

        except Exception as e:
            print(f"[WARN] sample {i} failed in diagnostics ({e}). Skipping.")
            continue

    # Sauvegarde jsonl
    with open(os.path.join(out_dir, f"{tag}_metrics.jsonl"), "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")

    return results


def plot_correlations(
    results: List[Dict[str, Any]],
    out_dir: str,
    tag: str,
    y_keys: Tuple[str, str] = ("P_norm_rel", "G_norm_rel"),
):
    """
    Trace des corrélations: pour chaque paramètre scalaire dans results,
    produit deux nuages de points (param vs y_keys[0], param vs y_keys[1]).
    """
    os.makedirs(out_dir, exist_ok=True)

    if not results:
        print(f"[plot_correlations] No results to plot for tag={tag}")
        return

    # Clés scalaires
    example = results[0]
    scalar_keys = []
    for k, v in example.items():
        if isinstance(v, (int, float)) and k not in ("idx",):
            scalar_keys.append(k)

    x_keys = [k for k in scalar_keys if k not in y_keys]
    data = {k: np.array([float(r.get(k, np.nan)) for r in results]) for k in scalar_keys}

    for xk in x_keys:
        for yk in y_keys:
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.scatter(data[xk], data[yk], s=10, alpha=0.7)
            ax.set_xlabel(xk)
            ax.set_ylabel(yk)
            ax.set_title(f"{tag}: {yk} vs {xk}")
            fig.tight_layout()
            fig.savefig(os.path.join(out_dir, f"{tag}_{yk}_vs_{xk}.png"), dpi=180)
            plt.close(fig)

    # Bonus: P_norm_rel vs G_norm_rel
    if all(k in data for k in y_keys):
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.scatter(data[y_keys[0]], data[y_keys[1]], s=10, alpha=0.7)
        ax.set_xlabel(y_keys[0])
        ax.set_ylabel(y_keys[1])
        ax.set_title(f"{tag}: {y_keys[1]} vs {y_keys[0]}")
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, f"{tag}_{y_keys[1]}_vs_{y_keys[0]}.png"), dpi=180)
        plt.close(fig)


# alias demandé
some_tests = run_diagnostics
