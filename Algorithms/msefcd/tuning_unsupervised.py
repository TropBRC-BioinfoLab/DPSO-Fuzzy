# tuning_unsupervised.py

import os
import json
import itertools
from datetime import datetime
from collections import Counter

import numpy as np
import pandas as pd
from cdlib.classes import NodeClustering
from cdlib import evaluation

from msefcd import MSEFCD, membership_to_communities


# ============================================================
# COMMUNITY EVALUATION HELPERS
# ============================================================

def _clean_communities(communities, valid_nodes=None):
    """
    Membersihkan komunitas:
    - convert semua node ke string
    - hapus komunitas kosong
    - jika valid_nodes diberikan, hanya ambil node yang ada di graph
    - hapus komunitas duplikat
    """
    valid_nodes = None if valid_nodes is None else set(str(n) for n in valid_nodes)

    clean = []
    seen = set()

    for comm in communities:
        c = set(str(x) for x in comm)

        if valid_nodes is not None:
            c = c & valid_nodes

        if len(c) == 0:
            continue

        key = tuple(sorted(c))

        if key not in seen:
            clean.append(c)
            seen.add(key)

    return clean


def community_match_score(true_comm, pred_comm, eps=1e-12):
    """
    Skor kemiripan antara satu komunitas ground truth dan satu komunitas prediksi.

    Rumus:
        precision_pair = |GT ∩ Pred| / |Pred|
        recall_pair    = |GT ∩ Pred| / |GT|
        match_score    = F1 dari precision_pair dan recall_pair

    Skor ini dipakai untuk mencari pasangan komunitas terbaik.
    """
    gt = set(str(x) for x in true_comm)
    pred = set(str(x) for x in pred_comm)

    if len(gt) == 0 or len(pred) == 0:
        return 0.0

    inter = len(gt & pred)

    p = inter / (len(pred) + eps)
    r = inter / (len(gt) + eps)

    if p + r <= eps:
        return 0.0

    return float(2.0 * p * r / (p + r))


def community_recall_precision_f1(true_communities, pred_communities, valid_nodes=None):
    """
    Menghitung comm_recall, comm_precision, dan comm_F1 level komunitas.

    comm_recall:
        Untuk setiap komunitas ground truth, cari komunitas prediksi
        dengan match_score tertinggi, lalu rata-ratakan.

    comm_precision:
        Untuk setiap komunitas prediksi, cari komunitas ground truth
        dengan match_score tertinggi, lalu rata-ratakan.

    comm_F1:
        Harmonic mean dari comm_recall dan comm_precision.
    """

    true_communities = _clean_communities(true_communities, valid_nodes=valid_nodes)
    pred_communities = _clean_communities(pred_communities, valid_nodes=valid_nodes)

    if len(true_communities) == 0 or len(pred_communities) == 0:
        return 0.0, 0.0, 0.0

    # Recall: setiap komunitas GT dicari prediksi terbaiknya
    recall_scores = []

    for gt_comm in true_communities:
        best_score = 0.0

        for pred_comm in pred_communities:
            score = community_match_score(gt_comm, pred_comm)
            if score > best_score:
                best_score = score

        recall_scores.append(best_score)

    comm_recall = float(np.mean(recall_scores)) if recall_scores else 0.0

    # Precision: setiap komunitas prediksi dicari GT terbaiknya
    precision_scores = []

    for pred_comm in pred_communities:
        best_score = 0.0

        for gt_comm in true_communities:
            score = community_match_score(gt_comm, pred_comm)
            if score > best_score:
                best_score = score

        precision_scores.append(best_score)

    comm_precision = float(np.mean(precision_scores)) if precision_scores else 0.0

    if comm_recall + comm_precision == 0:
        comm_F1 = 0.0
    else:
        comm_F1 = float(
            2.0 * comm_recall * comm_precision
            / (comm_recall + comm_precision)
        )

    return comm_precision, comm_recall, comm_F1


# ============================================================
# OMEGA FALLBACK
# ============================================================

def _omega_fallback(true_communities, pred_communities, nodes):
    """
    Fallback Omega Index jika CDlib gagal.
    Ini tetap berbasis jumlah shared communities pada setiap pasangan node.
    """
    nodes = sorted([str(x) for x in nodes])

    if len(nodes) < 2:
        return np.nan

    true_communities = _clean_communities(true_communities, valid_nodes=nodes)
    pred_communities = _clean_communities(pred_communities, valid_nodes=nodes)

    def pair_count_dict(communities):
        pair_count = Counter()

        for comm in communities:
            comm = sorted([str(x) for x in comm])

            for i in range(len(comm)):
                for j in range(i + 1, len(comm)):
                    pair_count[(comm[i], comm[j])] += 1

        return pair_count

    true_count = pair_count_dict(true_communities)
    pred_count = pair_count_dict(pred_communities)

    total_pairs = len(nodes) * (len(nodes) - 1) / 2

    if total_pairs <= 0:
        return np.nan

    observed_same = 0
    true_hist = Counter()
    pred_hist = Counter()

    for i in range(len(nodes)):
        for j in range(i + 1, len(nodes)):
            pair = (nodes[i], nodes[j])

            t = true_count.get(pair, 0)
            p = pred_count.get(pair, 0)

            if t == p:
                observed_same += 1

            true_hist[t] += 1
            pred_hist[p] += 1

    observed = observed_same / total_pairs

    expected = 0.0

    for k in set(true_hist.keys()) | set(pred_hist.keys()):
        expected += true_hist[k] * pred_hist[k]

    expected = expected / (total_pairs ** 2)

    if abs(1.0 - expected) < 1e-12:
        return np.nan

    return float((observed - expected) / (1.0 - expected))


# ============================================================
# EVALUATION WITH GROUND TRUTH
# ============================================================

def evaluate_with_ground_truth(G, nodes, pred_communities, true_communities):
    """
    Evaluasi akhir dengan ground truth.

    Output:
        onmi
        omega
        comm_recall
        comm_precision
        comm_F1

    Catatan:
        comm_recall, comm_precision, dan comm_F1 di sini sudah level komunitas,
        bukan pairwise node.
    """
    valid_nodes = set(str(n) for n in nodes)

    true_communities = _clean_communities(
        true_communities,
        valid_nodes=valid_nodes
    )

    pred_communities = _clean_communities(
        pred_communities,
        valid_nodes=valid_nodes
    )

    precision, recall, f1 = community_recall_precision_f1(
        true_communities=true_communities,
        pred_communities=pred_communities,
        valid_nodes=valid_nodes
    )

    out = {
        "onmi": np.nan,
        "omega": np.nan,
        "comm_recall": recall,
        "comm_precision": precision,
        "comm_F1": f1,
    }

    try:
        pred_nc = NodeClustering(
            communities=[list(c) for c in pred_communities],
            graph=G,
            method_name="MSEFCD",
            overlap=True
        )

        true_nc = NodeClustering(
            communities=[list(c) for c in true_communities],
            graph=G,
            method_name="ground_truth",
            overlap=True
        )

        out["onmi"] = evaluation.overlapping_normalized_mutual_information_LFK(
            pred_nc,
            true_nc
        ).score

        out["omega"] = evaluation.omega(
            pred_nc,
            true_nc
        ).score

    except Exception as e:
        print(f"[WARNING] CDlib gagal menghitung ONMI/Omega: {e}")

        out["omega"] = _omega_fallback(
            true_communities=true_communities,
            pred_communities=pred_communities,
            nodes=nodes
        )

    return out


# ============================================================
# OVERLAP COUNT
# ============================================================

def count_overlapping_nodes(U, alpha_threshold):
    count = 0

    for j in range(U.shape[1]):
        selected = np.where(U[:, j] >= alpha_threshold)[0]

        if len(selected) >= 2:
            count += 1

    return count


# ============================================================
# SAVE HELPERS
# ============================================================

def save_membership(result, path):
    U = result["membership"]
    nodes = result["nodes"]

    df = pd.DataFrame(
        U.T,
        columns=[f"community_{i + 1}" for i in range(U.shape[0])]
    )

    df.insert(0, "node", nodes)
    df.to_csv(path, index=False)


def save_communities(pred_communities, path):
    rows = []

    for idx, comm in enumerate(pred_communities, start=1):
        for node in sorted(comm):
            rows.append({
                "community": idx,
                "node": node
            })

    df = pd.DataFrame(rows, columns=["community", "node"])
    df.to_csv(path, index=False)


# ============================================================
# GRID SEARCH MSEFCD BY FUZZY MODULARITY / Qg
# ============================================================

def tune_msefcd_params_unsupervised(
    G,
    k_list=None,
    beta_list=None,
    r_list=None,
    init_iter_list=None,
    max_iter_list=None,
    max_communities_list=None,
    save_csv_path="tuning_unsupervised_msefcd.csv",
    best_config_json_path="best_config_msefcd.json",
    verbose=True
):
    """
    Grid search hyperparameter MSEFCD.

    Objective:
        maximize fuzzy_modularity / generalized modularity Qg.

    alpha_threshold tidak dimasukkan ke tuning Qg karena threshold
    hanya dipakai setelah membership matrix terbentuk.
    """

    if k_list is None:
        k_list = [3, 4, 5, 6, 7, 8, 9, 10]

    if beta_list is None:
        beta_list = [0.5, 1.0, 1.5, 2.0]

    if r_list is None:
        r_list = [1.5, 2.0, 2.5, 3.0]

    if init_iter_list is None:
        init_iter_list = [5]

    if max_iter_list is None:
        max_iter_list = [100]

    if max_communities_list is None:
        max_communities_list = [None]

    total_configs = (
        len(k_list)
        * len(beta_list)
        * len(r_list)
        * len(init_iter_list)
        * len(max_iter_list)
        * len(max_communities_list)
    )

    if verbose:
        print("🔧 Mulai UNSUPERVISED tuning MSEFCD")
        print("Objective: maximize fuzzy modularity / Qg")
        print(f"Total konfigurasi = {total_configs}")

    all_rows = []
    config_id = 0

    best_qg = -np.inf
    best_row = None
    best_result = None

    for k, beta, r, init_iter, max_iter, max_communities in itertools.product(
        k_list,
        beta_list,
        r_list,
        init_iter_list,
        max_iter_list,
        max_communities_list
    ):
        config_id += 1

        if verbose:
            print("\n----------------------------------------------")
            print(f"[Config {config_id}/{total_configs}]")
            print(
                f"k={k}, beta={beta}, r={r}, "
                f"init_iter={init_iter}, max_iter={max_iter}, "
                f"max_communities={max_communities}"
            )
            print("----------------------------------------------")

        model = MSEFCD(
            init_iter=init_iter,
            max_iter=max_iter,
            k=k,
            beta=beta,
            r=r,
            max_communities=max_communities,
            verbose=False
        )

        result = model.fit(G)

        qg = float(result["qg"])

        row = {
            "config_id": config_id,
            "k": k,
            "beta": beta,
            "r": r,
            "init_iter": init_iter,
            "max_iter": max_iter,
            "max_communities": max_communities,
            "fuzzy_modularity_qg": qg,
            "n_communities": result["n_communities"],
            "n_iterations": len(result["qg_history"]),
        }

        all_rows.append(row)

        if verbose:
            print(
                f"Qg={qg:.6f} | "
                f"n_communities={result['n_communities']} | "
                f"n_iterations={len(result['qg_history'])}"
            )

        if qg > best_qg:
            best_qg = qg
            best_row = row.copy()
            best_result = result

    df_tuning = pd.DataFrame(all_rows)

    if len(df_tuning) > 0:
        df_tuning = df_tuning.sort_values(
            "fuzzy_modularity_qg",
            ascending=False
        ).reset_index(drop=True)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    final_csv_path = None
    final_cfg_path = None

    if save_csv_path is not None:
        base, ext = (save_csv_path.rsplit(".", 1) + ["csv"])[:2]
        final_csv_path = f"{base}_{ts}.{ext}"
        df_tuning.to_csv(final_csv_path, index=False)

        if verbose:
            print(f"\n📁 Hasil tuning MSEFCD disimpan ke: {final_csv_path}")

    if best_row is not None and best_config_json_path is not None:
        base_cfg, ext_cfg = (best_config_json_path.rsplit(".", 1) + ["json"])[:2]
        final_cfg_path = f"{base_cfg}_{ts}.{ext_cfg}"

        best_cfg = {
            "k": int(best_row["k"]),
            "beta": float(best_row["beta"]),
            "r": float(best_row["r"]),
            "init_iter": int(best_row["init_iter"]),
            "max_iter": int(best_row["max_iter"]),
            "max_communities": (
                None
                if pd.isna(best_row["max_communities"])
                else int(best_row["max_communities"])
            ),
            "selection": {
                "type": "grid_search",
                "objective": "maximize_fuzzy_modularity_qg",
                "best_qg": float(best_qg)
            }
        }

        with open(final_cfg_path, "w") as f:
            json.dump(best_cfg, f, indent=4)

        if verbose:
            print(f"💾 Best config disimpan ke: {final_cfg_path}")

    if verbose:
        print("\n⭐ Konfigurasi terbaik MSEFCD ⭐")
        print(best_row)

    return df_tuning, best_row, best_result, final_csv_path, final_cfg_path


# ============================================================
# THRESHOLD TUNING AFTER FINAL MSEFCD
# ============================================================

def tune_alpha_threshold_by_ground_truth(
    G,
    result,
    true_communities,
    alpha_threshold_list=None,
    objective="comm_F1",
    out_dir=None,
    verbose=True
):
    """
    Tuning alpha_threshold setelah membership matrix diperoleh.

    Threshold tidak memengaruhi Qg.
    Threshold hanya memengaruhi pembentukan komunitas overlapping.

    objective:
        onmi
        omega
        comm_recall
        comm_precision
        comm_F1
        average_score
    """

    if alpha_threshold_list is None:
        alpha_threshold_list = [
            0.30, 0.35, 0.40, 0.45, 0.50
        ]

    rows = []

    best_score = -np.inf
    best_row = None
    best_pred_communities = None

    nodes = result["nodes"]
    U = result["membership"]

    if verbose:
        print("\n============================================")
        print("THRESHOLD TUNING")
        print(f"Objective: {objective}")
        print("============================================")

    for alpha in alpha_threshold_list:
        pred_communities = membership_to_communities(
            nodes=nodes,
            U=U,
            alpha=alpha,
            include_argmax=True
        )

        eval_result = evaluate_with_ground_truth(
            G=G,
            nodes=nodes,
            pred_communities=pred_communities,
            true_communities=true_communities
        )

        row = {
            "alpha_threshold": float(alpha),
            "fuzzy_modularity_qg": float(result["qg"]),
            "n_communities": result["n_communities"],
            "n_predicted_communities": len(pred_communities),
            "n_overlapping_nodes": count_overlapping_nodes(U, alpha),
            "onmi": eval_result.get("onmi", np.nan),
            "omega": eval_result.get("omega", np.nan),
            "comm_recall": eval_result.get("comm_recall", np.nan),
            "comm_precision": eval_result.get("comm_precision", np.nan),
            "comm_F1": eval_result.get("comm_F1", np.nan),
        }

        values = [
            row["onmi"],
            row["omega"],
            row["comm_recall"],
            row["comm_precision"],
            row["comm_F1"],
        ]

        valid_values = [float(v) for v in values if not pd.isna(v)]

        row["average_score"] = (
            float(np.mean(valid_values))
            if len(valid_values) > 0
            else np.nan
        )

        rows.append(row)

        score = row.get(objective, np.nan)

        if pd.isna(score):
            score = -np.inf

        if verbose:
            print(
                f"alpha={alpha:.2f} | "
                f"ONMI={row['onmi']:.6f} | "
                f"Omega={row['omega']:.6f} | "
                f"CommRecall={row['comm_recall']:.6f} | "
                f"CommPrecision={row['comm_precision']:.6f} | "
                f"CommF1={row['comm_F1']:.6f} | "
                f"Overlap={row['n_overlapping_nodes']} | "
                f"Score={score:.6f}"
            )

        if score > best_score:
            best_score = score
            best_row = row.copy()
            best_pred_communities = pred_communities

    df_threshold = pd.DataFrame(rows)

    if len(df_threshold) > 0 and objective in df_threshold.columns:
        df_threshold = df_threshold.sort_values(
            by=objective,
            ascending=False
        ).reset_index(drop=True)

    if out_dir is not None:
        os.makedirs(out_dir, exist_ok=True)

        threshold_path = os.path.join(out_dir, "msefcd_threshold_tuning.csv")
        best_threshold_path = os.path.join(out_dir, "msefcd_best_threshold.csv")

        df_threshold.to_csv(threshold_path, index=False)

        if best_row is not None:
            pd.DataFrame([best_row]).to_csv(best_threshold_path, index=False)

        if verbose:
            print("\nThreshold tuning saved:")
            print(f"- {threshold_path}")
            print(f"- {best_threshold_path}")

    if verbose:
        print("\n⭐ Best threshold ⭐")
        print(best_row)

    return best_row, best_pred_communities, df_threshold


# ============================================================
# FINAL RUN AFTER TUNING
# ============================================================

def run_final_msefcd_with_best_params(
    G,
    best_row,
    true_communities=None,
    alpha_threshold=0.3,
    alpha_threshold_list=None,
    threshold_objective="comm_F1",
    out_dir="msefcd_final_result",
    verbose=True
):
    """
    Menjalankan MSEFCD sekali menggunakan parameter terbaik dari tuning Qg.

    Jika true_communities tersedia:
        threshold terbaik dicari berdasarkan threshold_objective.

    Jika true_communities tidak tersedia:
        alpha_threshold manual digunakan.
    """

    os.makedirs(out_dir, exist_ok=True)

    model = MSEFCD(
        init_iter=int(best_row["init_iter"]),
        max_iter=int(best_row["max_iter"]),
        k=int(best_row["k"]),
        beta=float(best_row["beta"]),
        r=float(best_row["r"]),
        max_communities=(
            None
            if pd.isna(best_row["max_communities"])
            else int(best_row["max_communities"])
        ),
        verbose=False
    )

    result = model.fit(G)

    best_threshold_row = None

    if true_communities is not None:
        best_threshold_row, pred_communities, df_threshold = tune_alpha_threshold_by_ground_truth(
            G=G,
            result=result,
            true_communities=true_communities,
            alpha_threshold_list=alpha_threshold_list,
            objective=threshold_objective,
            out_dir=out_dir,
            verbose=verbose
        )

        selected_alpha = float(best_threshold_row["alpha_threshold"])

        eval_gt = {
            "onmi": best_threshold_row["onmi"],
            "omega": best_threshold_row["omega"],
            "comm_recall": best_threshold_row["comm_recall"],
            "comm_precision": best_threshold_row["comm_precision"],
            "comm_F1": best_threshold_row["comm_F1"],
            "average_score": best_threshold_row["average_score"],
        }

    else:
        selected_alpha = float(alpha_threshold)

        pred_communities = membership_to_communities(
            nodes=result["nodes"],
            U=result["membership"],
            alpha=selected_alpha,
            include_argmax=True
        )

        eval_gt = {}

    final_eval = {
        "k": int(best_row["k"]),
        "beta": float(best_row["beta"]),
        "r": float(best_row["r"]),
        "init_iter": int(best_row["init_iter"]),
        "max_iter": int(best_row["max_iter"]),
        "max_communities": best_row["max_communities"],
        "selected_alpha_threshold": selected_alpha,
        "threshold_objective": threshold_objective,
        "fuzzy_modularity_qg": float(result["qg"]),
        "n_communities": result["n_communities"],
        "n_predicted_communities": len(pred_communities),
        "n_overlapping_nodes": count_overlapping_nodes(
            result["membership"],
            selected_alpha
        )
    }

    final_eval.update(eval_gt)

    final_eval_path = os.path.join(out_dir, "msefcd_final_evaluation.csv")
    pd.DataFrame([final_eval]).to_csv(final_eval_path, index=False)

    save_membership(
        result,
        os.path.join(out_dir, "msefcd_final_membership.csv")
    )

    save_communities(
        pred_communities,
        os.path.join(out_dir, "msefcd_final_communities.csv")
    )

    pd.DataFrame({
        "iteration": list(range(1, len(result["qg_history"]) + 1)),
        "qg": result["qg_history"]
    }).to_csv(
        os.path.join(out_dir, "msefcd_final_qg_history.csv"),
        index=False
    )

    if verbose:
        print("\n========== FINAL MSEFCD EVALUATION ==========")

        for k, v in final_eval.items():
            print(f"{k}: {v}")

        print("\nOutput:")
        print(f"- {final_eval_path}")
        print(f"- {os.path.join(out_dir, 'msefcd_final_membership.csv')}")
        print(f"- {os.path.join(out_dir, 'msefcd_final_communities.csv')}")
        print(f"- {os.path.join(out_dir, 'msefcd_final_qg_history.csv')}")

        if true_communities is not None:
            print(f"- {os.path.join(out_dir, 'msefcd_threshold_tuning.csv')}")
            print(f"- {os.path.join(out_dir, 'msefcd_best_threshold.csv')}")

    return result, pred_communities, final_eval
