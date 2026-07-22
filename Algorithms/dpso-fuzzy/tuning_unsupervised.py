# tuning_unsupervised.py
import itertools
import json
from collections import defaultdict
from datetime import datetime

import networkx as nx
import numpy as np
import pandas as pd
from cdlib.classes import NodeClustering
from cdlib import evaluation

from msefcd_hybrid import compute_fuzzy_membership_from_DPSO_MSEFCD_full, fuzzy_to_communities


# ============================================================
# ENTROPY
# ============================================================
def _compute_entropy_for_fuzzy_membership(fuzzy_membership, eps=1e-10):
    entropies = []
    for _, mem in fuzzy_membership.items():
        if not mem:
            continue
        probs = np.array(list(mem.values()), dtype=float)
        s = probs.sum()
        if s <= 0:
            continue
        probs = probs / s
        probs = np.clip(probs, eps, 1.0)
        entropies.append(float(-np.sum(probs * np.log(probs))))
    return float(np.mean(entropies)) if entropies else 0.0


# ============================================================
# FUZZY / OVERLAP-COMPATIBLE (SOFT) MODULARITY
# ============================================================
def _normalize_fuzzy_membership(fm, eps=1e-12):
    out = {}
    for n, mem in fm.items():
        if not mem:
            out[n] = {}
            continue
        keys = list(mem.keys())
        vals = np.array([float(mem[k]) for k in keys], dtype=float)
        vals[vals < 0] = 0.0
        s = vals.sum()
        if s <= eps:
            out[n] = {k: 1.0 / len(keys) for k in keys}
        else:
            out[n] = {k: float(v / s) for k, v in zip(keys, vals)}
    return out


def fuzzy_overlapping_modularity(G, fm, eps=1e-12):
    """
    Soft fuzzy modularity compatible with overlap:
      Qf = sum_c ( e_c/m - (a_c/(2m))^2 )
    """
    if G.number_of_edges() == 0:
        return 0.0

    fm = _normalize_fuzzy_membership(fm, eps=eps)
    deg = dict(G.degree())
    m = float(G.number_of_edges())
    two_m = 2.0 * m

    comms = set()
    for mem in fm.values():
        comms.update(mem.keys())
    if not comms:
        return 0.0

    a = defaultdict(float)
    for i, mem in fm.items():
        ki = float(deg.get(i, 0))
        for c, u in mem.items():
            a[c] += ki * float(u)

    e = defaultdict(float)
    for i, j in G.edges():
        mi = fm.get(i, {})
        mj = fm.get(j, {})
        if len(mi) > len(mj):
            mi, mj = mj, mi
        for c, uic in mi.items():
            ujc = mj.get(c, 0.0)
            if ujc:
                e[c] += float(uic) * float(ujc)

    Qf = 0.0
    for c in comms:
        ec = e.get(c, 0.0)
        ac = a.get(c, 0.0)
        Qf += (ec / m) - (ac / two_m) ** 2

    return float(Qf)


# ============================================================
# OPTIONAL: VALID PARTITION FOR nx.modularity (ARGMAX)
# ============================================================
def fuzzy_to_partition_argmax(fm, all_nodes, eps=1e-12):
    assign = {}
    for n in all_nodes:
        mem = fm.get(n, {})
        if not mem:
            assign[n] = "__unassigned__"
            continue
        best_c, best_w = None, -1.0
        for c, w in mem.items():
            w = float(w)
            if w > best_w + eps:
                best_c, best_w = c, w
        assign[n] = best_c if best_c is not None else "__unassigned__"

    comm_map = defaultdict(set)
    for n, c in assign.items():
        comm_map[c].add(n)

    return [set(nodes) for nodes in comm_map.values() if nodes]


# ============================================================
# METRICS FOR ONE CONFIG
# ============================================================
def _compute_unsupervised_metrics_for_config(
    G,
    fuzzy_results,
    alpha_threshold=0.4,
    eps=1e-10,
    compute_partition_modularity=True,
):
    all_nodes = set(G.nodes())

    fuzzy_mods = []
    part_mods = []
    entropies = []
    nodeclusterings = []

    for run_data in fuzzy_results:
        fm = run_data["fuzzy_membership"]

        # 1) soft fuzzy modularity (overlap-compatible)
        try:
            fuzzy_mods.append(fuzzy_overlapping_modularity(G, fm, eps=eps))
        except Exception:
            pass

        # 2) optional: partition modularity (argmax -> valid partition)
        if compute_partition_modularity:
            try:
                part = fuzzy_to_partition_argmax(fm, all_nodes=all_nodes, eps=eps)
                if len(part) >= 2:
                    part_mods.append(float(nx.community.modularity(G, part)))
            except Exception:
                pass

        # 3) entropy
        entropies.append(_compute_entropy_for_fuzzy_membership(fm, eps=eps))

        # 4) stability ONMI (need overlap communities via threshold)
        try:
            comms = fuzzy_to_communities(fm, alpha=alpha_threshold)
            if len(comms) >= 2:
                nodeclusterings.append(NodeClustering(comms, G, f"run_{run_data.get('run', 0)}"))
        except Exception:
            pass

    fuzzy_mod_mean = float(np.mean(fuzzy_mods)) if fuzzy_mods else 0.0
    fuzzy_mod_std  = float(np.std(fuzzy_mods)) if fuzzy_mods else 0.0

    part_mod_mean = float(np.mean(part_mods)) if (compute_partition_modularity and part_mods) else 0.0
    part_mod_std  = float(np.std(part_mods)) if (compute_partition_modularity and part_mods) else 0.0

    entropy_mean = float(np.mean(entropies)) if entropies else 0.0
    entropy_std  = float(np.std(entropies)) if entropies else 0.0

    stability_scores = []
    if len(nodeclusterings) >= 2:
        for nc1, nc2 in itertools.combinations(nodeclusterings, 2):
            try:
                stability_scores.append(
                    evaluation.overlapping_normalized_mutual_information_LFK(nc1, nc2).score
                )
            except Exception:
                continue

    stability_mean = float(np.mean(stability_scores)) if stability_scores else 0.0
    stability_std  = float(np.std(stability_scores)) if stability_scores else 0.0

    return {
        "fuzzy_modularity_mean": fuzzy_mod_mean,
        "fuzzy_modularity_std": fuzzy_mod_std,
        "partition_modularity_mean": part_mod_mean,
        "partition_modularity_std": part_mod_std,
        "entropy_mean": entropy_mean,
        "entropy_std": entropy_std,
        "stability_mean": stability_mean,
        "stability_std": stability_std,
    }


# ============================================================
# OPTION C SELECTION
# ============================================================
def _select_best_option_c(df_tuning, top_frac=0.10, modularity_tiebreak="fuzzy"):
    if df_tuning is None or len(df_tuning) == 0:
        return None

    df_sorted = df_tuning.sort_values("composite_score", ascending=False).reset_index(drop=True)
    N = max(1, int(len(df_sorted) * float(top_frac)))
    df_top = df_sorted.head(N).copy()

    mod_col = "fuzzy_modularity_mean" if modularity_tiebreak.lower() == "fuzzy" else "partition_modularity_mean"

    df_top2 = df_top.sort_values(
        by=[mod_col, "entropy_mean"],
        ascending=[False, True]
    ).reset_index(drop=True)

    return df_top2.iloc[0]


# ============================================================
# MAIN TUNER (SIGNATURE COMPATIBLE WITH YOUR MAIN)
# ============================================================
def tune_hybrid_msefcd_params_unsupervised(
    G,
    dpso_results,
    alpha_beta_grid=None,
    prior_boost_list=None,
    k_comm_candidates_list=None,
    alpha_threshold_list=None,
    iter_smooth=12,
    iter_enhance=7,
    min_comm_size=1,
    save_csv_path="tuning_unsupervised_hybrid_dpso_msefcd.csv",
    best_config_json_path="best_config.json",
    # OPTION C (can be ignored by your main; defaults are fine)
    top_frac=0.10,
    modularity_tiebreak="fuzzy",   # "fuzzy" (recommended) or "partition"
    compute_partition_modularity=True,
    verbose=True
):
    if alpha_beta_grid is None:
        alpha_beta_grid = [(0.8, 0.2), (0.7, 0.3), (0.6, 0.4), (0.5, 0.5)]
    if prior_boost_list is None:
        prior_boost_list = [0.05, 0.08, 0.10, 0.15]
    if k_comm_candidates_list is None:
        k_comm_candidates_list = [2, 3, 4]
    if alpha_threshold_list is None:
        alpha_threshold_list = [0.20, 0.30, 0.40, 0.45, 0.50]

    total_configs = (
        len(alpha_beta_grid)
        * len(prior_boost_list)
        * len(k_comm_candidates_list)
        * len(alpha_threshold_list)
    )

    if verbose:
        print(f"🔧 Mulai UNSUPERVISED tuning Hybrid DPSO–MSEFCD, total konfigurasi = {total_configs}")

    all_rows = []
    config_id = 0

    for (alpha_self, beta_neighbor) in alpha_beta_grid:
        for prior_boost in prior_boost_list:
            for k_comm in k_comm_candidates_list:
                if verbose:
                    print("\n----------------------------------------------")
                    print(f"Compute fuzzy for: alpha_self={alpha_self}, beta_neighbor={beta_neighbor}, "
                          f"prior_boost={prior_boost}, k_comm_candidates={k_comm}")
                    print("----------------------------------------------")

                fuzzy_results = compute_fuzzy_membership_from_DPSO_MSEFCD_full(
                    G,
                    dpso_results,
                    r=2.0,
                    iter_smooth=iter_smooth,
                    iter_enhance=iter_enhance,
                    k_comm_candidates=k_comm,
                    prior_boost=prior_boost,
                    min_comm_size=min_comm_size,
                    alpha_self=alpha_self,
                    beta_neighbor=beta_neighbor,
                    verbose=False
                )

                for alpha_threshold in alpha_threshold_list:
                    config_id += 1

                    metrics = _compute_unsupervised_metrics_for_config(
                        G,
                        fuzzy_results,
                        alpha_threshold=alpha_threshold,
                        compute_partition_modularity=compute_partition_modularity
                    )

                    composite_score = (
                        metrics["fuzzy_modularity_mean"]
                        + metrics["stability_mean"]
                        - metrics["entropy_mean"]
                    )

                    row = {
                        "config_id": config_id,
                        "alpha_self": alpha_self,
                        "beta_neighbor": beta_neighbor,
                        "prior_boost": prior_boost,
                        "k_comm_candidates": k_comm,
                        "iter_smooth": iter_smooth,
                        "iter_enhance": iter_enhance,
                        "alpha_threshold": alpha_threshold,

                        "fuzzy_modularity_mean": metrics["fuzzy_modularity_mean"],
                        "fuzzy_modularity_std": metrics["fuzzy_modularity_std"],

                        "partition_modularity_mean": metrics["partition_modularity_mean"],
                        "partition_modularity_std": metrics["partition_modularity_std"],

                        "entropy_mean": metrics["entropy_mean"],
                        "entropy_std": metrics["entropy_std"],

                        "stability_mean": metrics["stability_mean"],
                        "stability_std": metrics["stability_std"],

                        "composite_score": composite_score,
                    }
                    all_rows.append(row)

                    if verbose:
                        print(f"[Config {config_id}/{total_configs}] "
                              f"a={alpha_self}, b={beta_neighbor}, pb={prior_boost}, k={k_comm}, thr={alpha_threshold} | "
                              f"Qf={row['fuzzy_modularity_mean']:.6f} | "
                              f"ONMI={row['stability_mean']:.6f} | "
                              f"Ent={row['entropy_mean']:.6f} | "
                              f"Comp={row['composite_score']:.6f}")

    df_tuning = pd.DataFrame(all_rows)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    final_csv_path = None
    final_cfg_path = None

    # Save CSV (same behavior as your code)
    if save_csv_path is not None:
        base, ext = (save_csv_path.rsplit(".", 1) + ["csv"])[:2]
        final_csv_path = f"{base}_{ts}.{ext}"
        df_tuning.to_csv(final_csv_path, index=False)
        if verbose:
            print(f"\n📁 Hasil unsupervised tuning disimpan ke: {final_csv_path}")

    # Select best (Option C)
    best_row = _select_best_option_c(df_tuning, top_frac=top_frac, modularity_tiebreak=modularity_tiebreak)

    if best_row is not None and verbose:
        print("\n⭐ Konfigurasi Terbaik (UNSUPERVISED, Option C) ⭐")
        print(f"Top frac={top_frac} | modularity_tiebreak={modularity_tiebreak}")
        print(best_row)

    # Save JSON best config (same behavior as your code)
    if best_row is not None and best_config_json_path is not None:
        base_cfg, ext_cfg = (best_config_json_path.rsplit(".", 1) + ["json"])[:2]
        final_cfg_path = f"{base_cfg}_{ts}.{ext_cfg}"

        best_cfg_dict = {
            "alpha_self": float(best_row["alpha_self"]),
            "beta_neighbor": float(best_row["beta_neighbor"]),
            "prior_boost": float(best_row["prior_boost"]),
            "k_comm_candidates": int(best_row["k_comm_candidates"]),
            "iter_smooth": int(best_row["iter_smooth"]),
            "iter_enhance": int(best_row["iter_enhance"]),
            "alpha_threshold": float(best_row["alpha_threshold"]),
            "selection": {
                "type": "option_c",
                "top_frac": float(top_frac),
                "modularity_tiebreak": str(modularity_tiebreak),
            }
        }

        with open(final_cfg_path, "w") as f:
            json.dump(best_cfg_dict, f, indent=4)

        if verbose:
            print(f"💾 Best config disimpan ke: {final_cfg_path}")

    return df_tuning, best_row, final_csv_path, final_cfg_path
