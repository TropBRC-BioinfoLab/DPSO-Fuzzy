import itertools
import random
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import networkx as nx

from dpso import PSOCommunityDetection  # pastikan dpso.py ada di path yang sama / PYTHONPATH


def tune_dpso_hyperparams_disease(
    G: nx.Graph,
    runs_per_cfg: int = 3,
    param_grid: Dict | None = None,
    base_seed: int = 42,
    verbose: bool = True,
) -> Tuple[pd.DataFrame, Dict]:
    """
    Tuning hyperparameter DPSO untuk graph penyakit (tanpa ground truth).
    Objective: rata-rata modularity (semakin tinggi semakin baik).

    Parameters
    ----------
    G : nx.Graph
        Graph (misal: komponen terbesar dari graph penyakit).
    runs_per_cfg : int
        Berapa kali DPSO dijalankan per kombinasi hyperparameter.
    param_grid : dict or None
        Grid hyperparameter:
          {
            "w": [0.5, 0.7, 0.9],
            "c1": [1.0, 1.5, 2.0],
            "c2": [1.0, 1.5, 2.0],
            "rho": [0.5, 0.7],
            "p": [0.4, 0.6],
            "num_particles": [50, 100],
            "max_iter": [100, 300],
          }
        Kalau None → pakai default di atas.
    base_seed : int
        Seed dasar agar hasil bisa diulang.
    verbose : bool
        Print progress kalau True.

    Returns
    -------
    df_results : pd.DataFrame
        Tabel semua konfigurasi dan statistik modularity.
    best_cfg : dict
        Konfigurasi terbaik (mean_modularity tertinggi).
    """
    # --------------------------
    # 1. Definisikan param grid
    # --------------------------
    if param_grid is None:
        param_grid = {
            "w": [0.5, 0.7, 0.9],
            "c1": [1.0, 1.5, 2.0],
            "c2": [1.0, 1.5, 2.0],
            "rho": [0.5, 0.7],
            "p": [0.4, 0.6],
            "num_particles": [50, 100],
            "max_iter": [100, 300],
        }

    keys = list(param_grid.keys())
    grid_tuples = list(itertools.product(*[param_grid[k] for k in keys]))

    if verbose:
        print("🔎 Tuning DPSO (unsupervised, objective = modularity)")
        print(f"   Total kombinasi hyperparameter: {len(grid_tuples)}")
        print(f"   runs_per_cfg = {runs_per_cfg}")
        print("   Grid:")
        for k in keys:
            print(f"     - {k}: {param_grid[k]}")

    all_rows = []

    # --------------------------
    # 2. Loop semua kombinasi
    # --------------------------
    for cfg_idx, values in enumerate(grid_tuples, start=1):
        cfg = dict(zip(keys, values))

        if verbose:
            print(f"\n=== Konfigurasi {cfg_idx}/{len(grid_tuples)} ===")
            print("   ", cfg)

        modularities = []
        num_comms_list = []

        # --- Jalankan beberapa run per konfigurasi ---
        for r in range(runs_per_cfg):
            seed = base_seed + cfg_idx * 1000 + r
            random.seed(seed)
            np.random.seed(seed)

            # Inisialisasi DPSO
            pso = PSOCommunityDetection(
                graph=G,
                num_particles=cfg["num_particles"],
                max_iter=cfg["max_iter"],
                rho=cfg["rho"],
                sig_variant="paper",
            )

            # Run dengan hyperparameter (w, c1, c2, p)
            best_position = pso.run(
                w=cfg["w"],
                c1=cfg["c1"],
                c2=cfg["c2"],
                p=cfg["p"],
                verbose=False,
            )

            best_modularity = pso.global_best_modularity
            modularities.append(best_modularity)

            if best_position is not None:
                num_communities = len(set(best_position))
            else:
                num_communities = 0
            num_comms_list.append(num_communities)

            if verbose:
                print(
                    f"   Run {r+1}/{runs_per_cfg}: "
                    f"Q = {best_modularity:.4f}, #comm = {num_communities}"
                )

        # Statistik per konfigurasi
        mean_mod = float(np.mean(modularities))
        std_mod = float(np.std(modularities))
        mean_comms = float(np.mean(num_comms_list))

        row = {
            **cfg,
            "runs_per_cfg": runs_per_cfg,
            "mean_modularity": mean_mod,
            "std_modularity": std_mod,
            "mean_num_communities": mean_comms,
        }
        all_rows.append(row)

        if verbose:
            print(
                f"   >> Rata-rata modularity = {mean_mod:.4f} ± {std_mod:.4f} "
                f"(avg #comm = {mean_comms:.1f})"
            )

    # --------------------------
    # 3. Kumpulkan hasil + pilih terbaik
    # --------------------------
    df_results = pd.DataFrame(all_rows)

    best_idx = df_results["mean_modularity"].idxmax()
    best_cfg = df_results.loc[best_idx].to_dict()

    if verbose:
        print("\n🏆 KONFIGURASI DPSO TERBAIK (berdasarkan modularity rata-rata):")
        for k in ["w", "c1", "c2", "rho", "p", "num_particles", "max_iter",
                  "mean_modularity", "std_modularity", "mean_num_communities"]:
            print(f"   {k}: {best_cfg[k]}")

    return df_results, best_cfg
