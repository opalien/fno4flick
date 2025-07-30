from __future__ import annotations

import os
from copy import deepcopy
from typing import Any, Dict, List, Tuple

import json
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import Tensor
from neuralop.models import FNO

from experiments.fno2d_2out.dataset import Dataset, collate_fn
from experiments.fno2d.edp_parameters import EDPParameters
from experiments.fno2d_2out.integrator import compute_G_in_from_full


def _to_numpy(t: Tensor) -> np.ndarray:
    return t.detach().cpu().numpy()


def _safe_div(a: float, b: float, eps: float = 1e-12) -> float:
    return a / (b + eps)


def _safe_log_ratio(num: float, den: float, eps: float = 1e-12) -> float:
    return float(np.log(_safe_div(num, den, eps)))


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
      - Calcule erreurs P_in / P_out (absolue & relative),
      - Calcule G_in et erreur associée,
      - Scanne R autour de la vraie valeur et retient R_best,
      - Produit une figure synthèse,
      - Sauvegarde un jsonl des métriques + paramètres + contrasts sans dimension.
    """
    os.makedirs(out_dir, exist_ok=True)
    per_sample_dir = os.path.join(out_dir, "per_sample")
    os.makedirs(per_sample_dir, exist_ok=True)

    model.eval().to(device)

    results: List[Dict[str, Any]] = []

    for i, (params_transfo, P2c_quadrant) in enumerate(dataset.elements):
        try:
            params_root: EDPParameters = params_transfo.get_root_parent()

            # Prépare tenseurs
            X_true, Y_true_full = collate_fn([(params_transfo, P2c_quadrant)])
            X_true = X_true.to(device)
            Y_true_full = Y_true_full.to(device).squeeze(0)  # (2, pNt, pNs)

            # Prédiction
            Y_pred_full = model(X_true).squeeze(0)  # (2, pNt, pNs)

            # Quadrant-avg pour comparer dans le quadrant positif (Nt,Ns)
            pNt, pNs = Y_true_full.shape[-2:]
            Nt, Ns = pNt // 2, pNs // 2

            def quadrant_avg_channels(Y_full: Tensor) -> Tuple[Tensor, Tensor]:
                def qa(A2d: Tensor) -> Tensor:
                    q1 = A2d[Nt:, Ns:]
                    q2 = torch.flip(A2d[Nt:, :Ns], dims=[-1])
                    q3 = torch.flip(A2d[:Nt, :Ns], dims=[-2, -1])
                    q4 = torch.flip(A2d[:Nt, Ns:], dims=[-2])
                    return (q1 + q2 + q3 + q4) / 4.0
                return qa(Y_full[0]), qa(Y_full[1])  # (Nt,Ns) chacun

            P_in_true, P_out_true = quadrant_avg_channels(Y_true_full)
            P_in_pred, P_out_pred = quadrant_avg_channels(Y_pred_full)

            # Erreurs P_in / P_out
            Pin_err = P_in_pred - P_in_true
            Pout_err = P_out_pred - P_out_true

            def norms(A: Tensor, ref: Tensor) -> Tuple[float, float]:
                na = torch.linalg.norm(A.reshape(-1)).item()
                nb = torch.linalg.norm(ref.reshape(-1)).item()
                return na, (na / (nb + 1e-12))

            Pin_abs, Pin_rel = norms(Pin_err, P_in_true)
            Pout_abs, Pout_rel = norms(Pout_err, P_out_true)

            # Erreurs G_in
            G_true_in = compute_G_in_from_full(Y_true_full, params_transfo)  # (Nt,)
            G_pred_in = compute_G_in_from_full(Y_pred_full, params_transfo)  # (Nt,)
            G_err = G_pred_in - G_true_in
            G_abs = torch.linalg.norm(G_err.reshape(-1)).item()
            G_rel = G_abs / (torch.linalg.norm(G_true_in.reshape(-1)).item() + 1e-12)

            # Scan R
            R_true = float(params_root.R)
            if r_max_fixed:
                R_min = max(1e-9, R_true * (1.0 - R_scan_factor))
                R_max = R_true * (1.0 + R_scan_factor)
            else:
                R_min = max(1e-9, R_true * (1.0 - R_scan_factor))
                R_max = R_true * (1.0 + R_scan_factor)

            R_grid = torch.linspace(R_min, R_max, R_scan_points, device=device, dtype=X_true.dtype)
            scan_errors: List[float] = []

            for R_guess in R_grid:
                params_guess_root = deepcopy(params_root)
                params_guess_root.R = float(R_guess.item())
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
                X_guess, _ = collate_fn([(params_guess, P2c_quadrant)])
                X_guess = X_guess.to(device)

                Y_pred_guess = model(X_guess).squeeze(0)  # (2, pNt, pNs)
                G_pred_in_guess = compute_G_in_from_full(Y_pred_guess, params_guess)
                scan_err = torch.linalg.norm((G_pred_in_guess - G_true_in).reshape(-1)).item()
                scan_errors.append(scan_err)

            scan_errors_np = np.array(scan_errors)
            R_grid_np = _to_numpy(R_grid)
            best_idx = int(scan_errors_np.argmin())
            R_best = float(R_grid_np[best_idx])
            R_abs_err = abs(R_best - R_true)
            R_rel_err = R_abs_err / (abs(R_true) + 1e-12)

            # Contrastes & grandeurs sans dimension
            Cin, Cout = float(params_root.C_in), float(params_root.C_out)
            Din, Dout = float(params_root.D_in), float(params_root.D_out)
            T1in, T1out = float(params_root.T1_in), float(params_root.T1_out)
            P0in, P0out = float(params_root.P0_in), float(params_root.P0_out)
            rmax = float(params_root.r_max)

            R_over_rmax = _safe_div(R_true, rmax)
            Cin_over_Cout = _safe_div(Cin, Cout)
            Din_over_Dout = _safe_div(Din, Dout)
            T1in_over_T1out = _safe_div(T1in, T1out)

            logC_ratio = _safe_log_ratio(Cin, Cout)
            logD_ratio = _safe_log_ratio(Din, Dout)
            logT1_ratio = _safe_log_ratio(T1in, T1out)

            dC, dD, dT1 = Cin - Cout, Din - Dout, T1in - T1out

            l_in = float(np.sqrt(max(Din * T1in, 1e-12)))
            l_out = float(np.sqrt(max(Dout * T1out, 1e-12)))
            R_over_l_in = _safe_div(R_true, l_in)
            Rout_over_lout = _safe_div(rmax - R_true, l_out)

            Da_in = _safe_div(R_true ** 2, Din * T1in)  # (L^2)/(DT1)
            Da_out = _safe_div((rmax - R_true) ** 2, Dout * T1out)

            P0_ratio = _safe_div(P0in, P0out)

            # Figure multi-panneaux
            fig = plt.figure(figsize=(18, 14))
            gs = fig.add_gridspec(3, 3, height_ratios=[2.0, 2.0, 1.0])

            ax_pin_t = fig.add_subplot(gs[0, 0])
            im = ax_pin_t.imshow(_to_numpy(P_in_true), origin="lower", aspect="auto", cmap="viridis")
            ax_pin_t.set_title("P_in_true(s,t)")
            fig.colorbar(im, ax=ax_pin_t, fraction=0.046, pad=0.04)

            ax_pin_p = fig.add_subplot(gs[0, 1])
            im = ax_pin_p.imshow(_to_numpy(P_in_pred), origin="lower", aspect="auto", cmap="viridis")
            ax_pin_p.set_title(f"P_in_pred(s,t) | rel={Pin_rel:.3e}")
            fig.colorbar(im, ax=ax_pin_p, fraction=0.046, pad=0.04)

            ax_pin_e = fig.add_subplot(gs[0, 2])
            im = ax_pin_e.imshow(_to_numpy(Pin_err), origin="lower", aspect="auto", cmap="coolwarm")
            ax_pin_e.set_title("P_in_err")
            fig.colorbar(im, ax=ax_pin_e, fraction=0.046, pad=0.04)

            ax_pout_t = fig.add_subplot(gs[1, 0])
            im = ax_pout_t.imshow(_to_numpy(P_out_true), origin="lower", aspect="auto", cmap="viridis")
            ax_pout_t.set_title("P_out_true(η,t)")
            fig.colorbar(im, ax=ax_pout_t, fraction=0.046, pad=0.04)

            ax_pout_p = fig.add_subplot(gs[1, 1])
            im = ax_pout_p.imshow(_to_numpy(P_out_pred), origin="lower", aspect="auto", cmap="viridis")
            ax_pout_p.set_title(f"P_out_pred(η,t) | rel={Pout_rel:.3e}")
            fig.colorbar(im, ax=ax_pout_p, fraction=0.046, pad=0.04)

            ax_pout_e = fig.add_subplot(gs[1, 2])
            im = ax_pout_e.imshow(_to_numpy(Pout_err), origin="lower", aspect="auto", cmap="coolwarm")
            ax_pout_e.set_title("P_out_err")
            fig.colorbar(im, ax=ax_pout_e, fraction=0.046, pad=0.04)

            ax_g = fig.add_subplot(gs[2, 0])
            ax_g.plot(_to_numpy(G_true_in), label="G_in_true")
            ax_g.plot(_to_numpy(G_pred_in), label="G_in_pred", linestyle="--")
            ax_g.set_title(f"G_in(t) | rel={G_rel:.3e}")
            ax_g.legend()

            ax_scan = fig.add_subplot(gs[2, 1:])
            ax_scan.plot(R_grid_np, scan_errors_np, marker="o")
            ax_scan.axvline(R_true, color="r", linestyle="--", label=f"R_true={R_true:.4g}")
            ax_scan.axvline(R_best, color="g", linestyle="--", label=f"R_best={R_best:.4g}")
            ax_scan.set_title(f"Scan R | best={R_best:.4g} | |Δ|={R_abs_err:.3e}")
            ax_scan.set_xlabel("R guess")
            ax_scan.set_ylabel("|| G_in_pred(R) - G_in_true ||")
            ax_scan.legend()

            fig.suptitle(f"[{tag}] sample #{i}", fontsize=12)
            fig.tight_layout()
            fig.savefig(os.path.join(per_sample_dir, f"sample_{i:05d}.png"), dpi=180)
            plt.close(fig)

            # Résultats (pour corrélations)
            res: Dict[str, Any] = {
                "idx": i,
                # erreurs
                "Pin_norm_abs": Pin_abs,
                "Pin_norm_rel": Pin_rel,
                "Pout_norm_abs": Pout_abs,
                "Pout_norm_rel": Pout_rel,
                "G_in_abs": G_abs,
                "G_in_rel": G_rel,
                # paramètres root & géométrie
                "R_true": R_true,
                "r_max_true": rmax,
                "R_over_rmax": R_over_rmax,
                # paramètres bruts
                "C_in": Cin,
                "C_out": Cout,
                "D_in": Din,
                "D_out": Dout,
                "T1_in": T1in,
                "T1_out": T1out,
                "P0_in": P0in,
                "P0_out": P0out,
                # contrastes / features sans dimension
                "Cin_over_Cout": Cin_over_Cout,
                "Din_over_Dout": Din_over_Dout,
                "T1in_over_T1out": T1in_over_T1out,
                "logC_ratio": logC_ratio,
                "logD_ratio": logD_ratio,
                "logT1_ratio": logT1_ratio,
                "dC": dC,
                "dD": dD,
                "dT1": dT1,
                "l_diff_in": l_in,
                "l_diff_out": l_out,
                "R_over_l_in": R_over_l_in,
                "Rout_over_lout": Rout_over_lout,
                "Da_in": Da_in,
                "Da_out": Da_out,
                "P0_ratio": P0_ratio,
                # scan R
                "R_best": R_best,
                "R_abs_err": R_abs_err,
                "R_rel_err": R_rel_err,
                # tailles discrètes
                "Nt": int(params_root.Nt),
                "Nr": int(params_root.Nr),
                "t_max": float(params_root.t_max),
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
    y_keys: Tuple[str, str] = ("Pin_norm_rel", "Pout_norm_rel"),
):
    """
    Trace des corrélations: pour chaque paramètre scalaire dans results,
    produit deux nuages de points (param vs y_keys[0], param vs y_keys[1]).
    Les nouvelles features (contrastes, R/ℓ, Da, etc.) seront automatiquement prises en compte.
    """
    os.makedirs(out_dir, exist_ok=True)
    if not results:
        print(f"[plot_correlations] No results to plot for tag={tag}")
        return

    ex = results[0]
    scalar_keys = [k for k, v in ex.items() if isinstance(v, (int, float)) and k not in ("idx",)]
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

    # Bonus: scatter entre les deux Y
    if all(k in data for k in y_keys):
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.scatter(data[y_keys[0]], data[y_keys[1]], s=10, alpha=0.7)
        ax.set_xlabel(y_keys[0])
        ax.set_ylabel(y_keys[1])
        ax.set_title(f"{tag}: {y_keys[1]} vs {y_keys[0]}")
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, f"{tag}_{y_keys[1]}_vs_{y_keys[0]}.png"), dpi=180)
        plt.close(fig)


# alias historique (si jamais utilisé ailleurs)
some_tests = run_diagnostics
