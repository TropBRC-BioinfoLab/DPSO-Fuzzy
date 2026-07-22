# -*- coding: utf-8 -*-
"""
significant_comorbidity_vote20_only.py

Mencari significant comorbidity dari 20 run DPSO-Fuzzy.

Input:
1. Graph penyakit asli dalam bentuk edge list CSV.
2. Folder detail hasil 20 run DPSO-Fuzzy:
   <prefix>_dpso_fuzzy_run_details/
   Contoh:
   nsclc_dpso_fuzzy_run_details/
   sclc_dpso_fuzzy_run_details/

Setiap file JSON detail run minimal berisi:
- run
- seed
- communities

Metode:
1. Untuk setiap run dan setiap komunitas:
   - bentuk subgraph komunitas
   - hitung 4 centrality:
     a. degree centrality
     b. betweenness centrality
     c. closeness centrality
     d. eigenvector centrality
   - node dengan nilai tertinggi pada masing-masing centrality diberi vote
   - node dengan vote >= min_centrality_vote menjadi significant candidate pada run tersebut

2. Voting 20 run:
   - significant_run_count = jumlah run ketika node muncul sebagai significant candidate
   - significant_frequency = significant_run_count / total_runs
   - node dengan significant_frequency >= min_significant_freq menjadi stable significant comorbidity

Output:
1. <prefix>_centrality_per_community_20runs.csv
2. <prefix>_significant_candidates_20runs.csv
3. <prefix>_significant_frequency_vote20.csv
4. <prefix>_stable_significant_comorbidities_vote20.csv
5. <prefix>_significant_vote20_summary.json
"""

import argparse
import glob
import json
import math
import os
import re
from collections import defaultdict
from typing import Any, Dict, List, Tuple

import networkx as nx
import numpy as np
import pandas as pd


# ============================================================
# 1. Utility
# ============================================================

def load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(obj: Any, path: str):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def clean_node(x: Any) -> str:
    return str(x).strip()


def infer_prefix_from_details_dir(details_dir: str) -> str:
    base = os.path.basename(os.path.normpath(details_dir))
    suffix = "_dpso_fuzzy_run_details"
    if base.endswith(suffix):
        return base.replace(suffix, "")
    return "network"


def run_sort_key(path: str) -> Tuple[int, str]:
    name = os.path.basename(path)
    m = re.search(r"run_(\d+)", name)
    if m:
        return (int(m.group(1)), name)
    return (10**9, name)


def threshold_count(freq: float, total_runs: int) -> int:
    return int(math.ceil(float(freq) * int(total_runs)))


# ============================================================
# 2. Load Graph
# ============================================================

def infer_edge_columns(df: pd.DataFrame):
    cols = list(df.columns)
    lower_cols = {c.lower(): c for c in cols}

    source_candidates = ["source", "src", "node1", "disease1", "from"]
    target_candidates = ["target", "dst", "node2", "disease2", "to"]
    weight_candidates = ["weight", "similarity", "score", "wang", "sim", "wang_similarity"]

    source_col = None
    target_col = None
    weight_col = None

    for c in source_candidates:
        if c in lower_cols:
            source_col = lower_cols[c]
            break

    for c in target_candidates:
        if c in lower_cols:
            target_col = lower_cols[c]
            break

    for c in weight_candidates:
        if c in lower_cols:
            weight_col = lower_cols[c]
            break

    if source_col is None or target_col is None:
        source_col = cols[0]
        target_col = cols[1]

    if weight_col is None and len(cols) >= 3:
        third_col = cols[2]
        if pd.api.types.is_numeric_dtype(df[third_col]):
            weight_col = third_col

    return source_col, target_col, weight_col


def load_graph_from_csv(
    graph_path: str,
    source_col: str = None,
    target_col: str = None,
    weight_col: str = None,
) -> nx.Graph:
    df = pd.read_csv(graph_path)

    if df.shape[1] < 2:
        raise ValueError("CSV graph harus memiliki minimal dua kolom untuk source dan target.")

    if source_col is None or target_col is None:
        inferred_source, inferred_target, inferred_weight = infer_edge_columns(df)
        source_col = source_col or inferred_source
        target_col = target_col or inferred_target
        weight_col = weight_col or inferred_weight

    print("Source column :", source_col)
    print("Target column :", target_col)
    print("Weight column :", weight_col)

    G = nx.Graph()

    for _, row in df.iterrows():
        u = row[source_col]
        v = row[target_col]

        if pd.isna(u) or pd.isna(v):
            continue

        u = clean_node(u)
        v = clean_node(v)

        if weight_col is not None and weight_col in df.columns and not pd.isna(row[weight_col]):
            try:
                w = float(row[weight_col])
            except Exception:
                w = 1.0
        else:
            w = 1.0

        if w <= 0:
            w = 1e-12

        # weight = similarity
        # distance = inverse similarity untuk betweenness dan closeness
        G.add_edge(u, v, weight=w, distance=1.0 / w)

    G.remove_edges_from(nx.selfloop_edges(G))

    if G.number_of_nodes() == 0 or G.number_of_edges() == 0:
        raise ValueError("Graph kosong. Periksa path dan kolom CSV.")

    return G


# ============================================================
# 3. Load Run Details
# ============================================================

def normalize_communities(obj: Any) -> Dict[str, List[str]]:
    """
    Mengubah format communities menjadi:
    {community_id: [node1, node2, ...]}

    Format yang didukung:
    1. {"1": ["A", "B"], "2": ["C", "D"]}
    2. [["A", "B"], ["C", "D"]]
    3. {"1": {"A": 0.7, "B": 0.8}}
    """
    communities = {}

    if obj is None:
        return communities

    if isinstance(obj, dict):
        for comm_id, nodes in obj.items():
            if isinstance(nodes, dict):
                nodes = list(nodes.keys())

            if isinstance(nodes, (list, tuple, set)):
                communities[str(comm_id)] = [clean_node(n) for n in nodes]

    elif isinstance(obj, list):
        for i, nodes in enumerate(obj, start=1):
            if isinstance(nodes, dict):
                nodes = list(nodes.keys())

            if isinstance(nodes, (list, tuple, set)):
                communities[str(i)] = [clean_node(n) for n in nodes]

    return communities


def load_detail_files(details_dir: str = None, details_index: str = None) -> List[str]:
    files = []

    if details_dir:
        files = sorted(
            glob.glob(os.path.join(details_dir, "*.json")),
            key=run_sort_key
        )

    elif details_index:
        idx_df = pd.read_csv(details_index)

        if "detail_file" not in idx_df.columns:
            raise ValueError("details_index harus memiliki kolom 'detail_file'.")

        base_dir = os.path.dirname(os.path.abspath(details_index))

        for p in idx_df["detail_file"].tolist():
            p = str(p)

            if os.path.exists(p):
                files.append(p)
                continue

            candidate = os.path.join(base_dir, os.path.basename(p))
            if os.path.exists(candidate):
                files.append(candidate)
                continue

            candidates = glob.glob(os.path.join(base_dir, "*run_details", os.path.basename(p)))
            if candidates:
                files.append(candidates[0])

        files = sorted(files, key=run_sort_key)

    if not files:
        raise ValueError("Tidak ada file detail run ditemukan. Gunakan --details-dir atau --details-index.")

    return files


# ============================================================
# 4. Centrality per Community
# ============================================================

def safe_eigenvector_centrality(H: nx.Graph) -> Dict[str, float]:
    try:
        return nx.eigenvector_centrality(
            H,
            weight="weight",
            max_iter=1000,
            tol=1e-06
        )
    except Exception:
        try:
            return nx.pagerank(H, weight="weight")
        except Exception:
            n = H.number_of_nodes()
            if n == 0:
                return {}
            return {node: 1.0 / n for node in H.nodes()}


def top_nodes_for_metric(
    df: pd.DataFrame,
    metric: str,
    top_k: int = 1,
    skip_zero_metric: bool = True,
) -> List[str]:
    """
    Mengambil top-k node pada satu metrik centrality.
    Jika terjadi tie pada nilai batas, semua node dengan nilai sama tetap ikut.
    """
    if metric not in df.columns or df.empty:
        return []

    values = df[metric].astype(float)

    if skip_zero_metric and np.isclose(values.max(), 0.0):
        return []

    unique_values = sorted(values.unique(), reverse=True)
    if not unique_values:
        return []

    k = max(1, int(top_k))
    cutoff_idx = min(k - 1, len(unique_values) - 1)
    cutoff_value = unique_values[cutoff_idx]

    return df[df[metric] >= cutoff_value]["node"].astype(str).tolist()


def compute_centrality_for_one_community(
    G: nx.Graph,
    nodes: List[str],
    run_id: int,
    seed: Any,
    community_id: str,
    min_community_size: int,
    top_k: int,
    min_centrality_vote: int,
    skip_zero_metric: bool,
) -> Tuple[pd.DataFrame, pd.DataFrame]:

    nodes = [clean_node(n) for n in nodes]
    nodes_in_graph = [n for n in nodes if n in G.nodes()]

    if len(nodes_in_graph) < min_community_size:
        return pd.DataFrame(), pd.DataFrame()

    H = G.subgraph(nodes_in_graph).copy()

    if H.number_of_nodes() < min_community_size:
        return pd.DataFrame(), pd.DataFrame()

    if H.number_of_edges() == 0:
        return pd.DataFrame(), pd.DataFrame()

    degree_cent = nx.degree_centrality(H)
    weighted_degree = dict(H.degree(weight="weight"))

    max_wdeg = max(weighted_degree.values()) if weighted_degree else 0.0
    weighted_degree_norm = {
        node: (val / max_wdeg if max_wdeg > 0 else 0.0)
        for node, val in weighted_degree.items()
    }

    betweenness_cent = nx.betweenness_centrality(
        H,
        weight="distance",
        normalized=True
    )

    closeness_cent = nx.closeness_centrality(
        H,
        distance="distance"
    )

    eigenvector_cent = safe_eigenvector_centrality(H)

    rows = []
    for node in H.nodes():
        rows.append({
            "run": run_id,
            "seed": seed,
            "community": str(community_id),
            "node": clean_node(node),
            "community_size": H.number_of_nodes(),
            "community_edges": H.number_of_edges(),
            "degree_centrality": float(degree_cent.get(node, 0.0)),
            "weighted_degree": float(weighted_degree.get(node, 0.0)),
            "weighted_degree_norm": float(weighted_degree_norm.get(node, 0.0)),
            "betweenness_centrality": float(betweenness_cent.get(node, 0.0)),
            "closeness_centrality": float(closeness_cent.get(node, 0.0)),
            "eigenvector_centrality": float(eigenvector_cent.get(node, 0.0)),
        })

    df = pd.DataFrame(rows)

    vote_map = defaultdict(list)
    centrality_metrics = [
        "degree_centrality",
        "betweenness_centrality",
        "closeness_centrality",
        "eigenvector_centrality",
    ]

    for metric in centrality_metrics:
        top_nodes = top_nodes_for_metric(
            df,
            metric=metric,
            top_k=top_k,
            skip_zero_metric=skip_zero_metric
        )

        for node in top_nodes:
            vote_map[node].append(metric)

    df["centrality_vote"] = df["node"].map(lambda n: len(vote_map.get(str(n), [])))
    df["voted_by"] = df["node"].map(lambda n: ", ".join(vote_map.get(str(n), [])))
    df["is_significant_candidate"] = df["centrality_vote"] >= int(min_centrality_vote)

    df_sig = df[df["is_significant_candidate"]].copy()

    return df, df_sig


def analyze_one_run(
    G: nx.Graph,
    detail_path: str,
    min_community_size: int,
    top_k: int,
    min_centrality_vote: int,
    skip_zero_metric: bool,
) -> Tuple[pd.DataFrame, pd.DataFrame]:

    detail = load_json(detail_path)

    run_id = int(detail.get("run", run_sort_key(detail_path)[0]))
    seed = detail.get("seed", None)
    communities = normalize_communities(detail.get("communities"))

    all_rows = []
    sig_rows = []

    for community_id, nodes in communities.items():
        df_comm, df_sig_comm = compute_centrality_for_one_community(
            G=G,
            nodes=nodes,
            run_id=run_id,
            seed=seed,
            community_id=community_id,
            min_community_size=min_community_size,
            top_k=top_k,
            min_centrality_vote=min_centrality_vote,
            skip_zero_metric=skip_zero_metric,
        )

        if not df_comm.empty:
            all_rows.append(df_comm)

        if not df_sig_comm.empty:
            sig_rows.append(df_sig_comm)

    df_all = pd.concat(all_rows, ignore_index=True) if all_rows else pd.DataFrame()
    df_sig = pd.concat(sig_rows, ignore_index=True) if sig_rows else pd.DataFrame()

    return df_all, df_sig


# ============================================================
# 5. Vote Majority 20 Runs
# ============================================================

def summarize_significant_frequency(
    df_sig_candidates: pd.DataFrame,
    total_runs: int,
    min_significant_freq: float,
) -> pd.DataFrame:

    if df_sig_candidates.empty:
        return pd.DataFrame(columns=[
            "node",
            "significant_run_count",
            "significant_frequency",
            "candidate_occurrences",
            "mean_centrality_vote",
            "max_centrality_vote",
            "communities_found",
            "voted_by_summary",
            "is_stable_significant"
        ])

    # Satu node hanya dihitung satu kali per run untuk vote antar-run.
    # Jika node signifikan pada beberapa komunitas dalam run yang sama,
    # tetap significant_run_count-nya hanya 1.
    df_run_node = (
        df_sig_candidates
        .groupby(["run", "node"], as_index=False)
        .agg(
            seed=("seed", "first"),
            candidate_occurrences_in_run=("community", "nunique"),
            max_centrality_vote_in_run=("centrality_vote", "max"),
            mean_centrality_vote_in_run=("centrality_vote", "mean"),
            communities_in_run=("community", lambda x: ", ".join(sorted(set(map(str, x))))),
            voted_by_in_run=("voted_by", lambda x: " | ".join(sorted(set([v for v in map(str, x) if v])))),
        )
    )

    min_count = threshold_count(min_significant_freq, total_runs)

    df_freq = (
        df_run_node
        .groupby("node", as_index=False)
        .agg(
            significant_run_count=("run", "nunique"),
            candidate_occurrences=("candidate_occurrences_in_run", "sum"),
            mean_centrality_vote=("mean_centrality_vote_in_run", "mean"),
            max_centrality_vote=("max_centrality_vote_in_run", "max"),
            communities_found=("communities_in_run", lambda x: " | ".join(sorted(set(map(str, x))))),
            voted_by_summary=("voted_by_in_run", lambda x: " | ".join(sorted(set([v for v in map(str, x) if v])))),
        )
    )

    df_freq["significant_frequency"] = df_freq["significant_run_count"] / float(total_runs)
    df_freq["min_significant_run_required"] = min_count
    df_freq["is_stable_significant"] = df_freq["significant_run_count"] >= min_count

    df_freq = df_freq.sort_values(
        [
            "is_stable_significant",
            "significant_run_count",
            "mean_centrality_vote",
            "candidate_occurrences"
        ],
        ascending=[False, False, False, False]
    ).reset_index(drop=True)

    return df_freq


# ============================================================
# 6. Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Mencari significant comorbidity dari 20 run DPSO-Fuzzy berdasarkan centrality per community."
    )

    parser.add_argument("--graph", required=True, help="Path file graph disease CSV.")
    parser.add_argument("--details-dir", default=None, help="Folder *_dpso_fuzzy_run_details.")
    parser.add_argument("--details-index", default=None, help="CSV index detail run, optional.")
    parser.add_argument("--outdir", default=None, help="Folder output. Default: parent details-dir.")
    parser.add_argument("--prefix", default=None, help="Prefix output, contoh: nsclc atau sclc.")

    parser.add_argument("--source-col", default=None, help="Nama kolom source, optional.")
    parser.add_argument("--target-col", default=None, help="Nama kolom target, optional.")
    parser.add_argument("--weight-col", default=None, help="Nama kolom weight/similarity, optional.")

    parser.add_argument(
        "--min-community-size",
        type=int,
        default=2,
        help="Komunitas dengan ukuran lebih kecil dari ini diabaikan. Default=2."
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=1,
        help="Top-k node per centrality dalam komunitas. Default=1."
    )
    parser.add_argument(
        "--min-centrality-vote",
        type=int,
        default=2,
        help="Minimal vote dari 4 centrality agar node menjadi significant candidate pada satu run. Default=2."
    )
    parser.add_argument(
        "--min-significant-freq",
        type=float,
        default=0.50,
        help="Minimal frekuensi run untuk stable significant comorbidity. Default=0.50."
    )
    parser.add_argument(
        "--include-zero-centrality-votes",
        action="store_true",
        help="Jika diaktifkan, centrality yang semua nilainya 0 tetap diberi vote. Default: tidak."
    )

    args = parser.parse_args()

    if args.details_dir is None and args.details_index is None:
        raise ValueError("Gunakan salah satu: --details-dir atau --details-index.")

    detail_files = load_detail_files(
        details_dir=args.details_dir,
        details_index=args.details_index
    )

    if args.prefix is not None:
        prefix = args.prefix.lower()
    elif args.details_dir is not None:
        prefix = infer_prefix_from_details_dir(args.details_dir).lower()
    else:
        prefix = "network"

    if args.outdir is None:
        if args.details_dir:
            outdir = os.path.dirname(os.path.abspath(os.path.normpath(args.details_dir)))
        else:
            outdir = os.path.dirname(os.path.abspath(args.details_index))
    else:
        outdir = args.outdir

    os.makedirs(outdir, exist_ok=True)

    print("=" * 80)
    print("Significant Comorbidity Identification from DPSO-Fuzzy 20 Runs")
    print("=" * 80)
    print("Graph            :", args.graph)
    print("Details dir      :", args.details_dir)
    print("Details index    :", args.details_index)
    print("Total detail JSON:", len(detail_files))
    print("Output dir       :", outdir)
    print("Prefix           :", prefix)
    print("min community    :", args.min_community_size)
    print("top-k            :", args.top_k)
    print("min central vote :", args.min_centrality_vote)
    print("min signif freq  :", args.min_significant_freq)
    print("=" * 80)

    G = load_graph_from_csv(
        args.graph,
        source_col=args.source_col,
        target_col=args.target_col,
        weight_col=args.weight_col,
    )

    print(f"Loaded graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    all_centrality = []
    sig_candidates = []

    skip_zero_metric = not args.include_zero_centrality_votes

    for i, detail_path in enumerate(detail_files, start=1):
        print(f"[{i}/{len(detail_files)}] Processing {os.path.basename(detail_path)}")

        df_all_run, df_sig_run = analyze_one_run(
            G=G,
            detail_path=detail_path,
            min_community_size=args.min_community_size,
            top_k=args.top_k,
            min_centrality_vote=args.min_centrality_vote,
            skip_zero_metric=skip_zero_metric,
        )

        if not df_all_run.empty:
            all_centrality.append(df_all_run)

        if not df_sig_run.empty:
            sig_candidates.append(df_sig_run)

    df_all_centrality = pd.concat(all_centrality, ignore_index=True) if all_centrality else pd.DataFrame()
    df_sig_candidates = pd.concat(sig_candidates, ignore_index=True) if sig_candidates else pd.DataFrame()

    total_runs = len(detail_files)

    df_sig_freq = summarize_significant_frequency(
        df_sig_candidates=df_sig_candidates,
        total_runs=total_runs,
        min_significant_freq=args.min_significant_freq,
    )

    df_stable_sig = df_sig_freq[df_sig_freq["is_stable_significant"] == True].copy()

    # ========================================================
    # Save outputs
    # ========================================================

    out_all = os.path.join(outdir, f"{prefix}_centrality_per_community_20runs.csv")
    out_candidates = os.path.join(outdir, f"{prefix}_significant_candidates_20runs.csv")
    out_sig_freq = os.path.join(outdir, f"{prefix}_significant_frequency_vote20.csv")
    out_stable_sig = os.path.join(outdir, f"{prefix}_stable_significant_comorbidities_vote20.csv")
    out_summary = os.path.join(outdir, f"{prefix}_significant_vote20_summary.json")

    df_all_centrality.to_csv(out_all, index=False)
    df_sig_candidates.to_csv(out_candidates, index=False)
    df_sig_freq.to_csv(out_sig_freq, index=False)
    df_stable_sig.to_csv(out_stable_sig, index=False)

    summary = {
        "prefix": prefix,
        "total_runs": total_runs,
        "graph_nodes": G.number_of_nodes(),
        "graph_edges": G.number_of_edges(),
        "min_community_size": args.min_community_size,
        "top_k": args.top_k,
        "min_centrality_vote": args.min_centrality_vote,
        "min_significant_freq": args.min_significant_freq,
        "min_significant_run_required": threshold_count(args.min_significant_freq, total_runs),
        "total_centrality_rows": int(len(df_all_centrality)),
        "total_significant_candidate_rows": int(len(df_sig_candidates)),
        "total_unique_significant_candidates": int(len(df_sig_freq)),
        "total_stable_significant_comorbidities": int(len(df_stable_sig)),
        "outputs": {
            "centrality_per_community": out_all,
            "significant_candidates": out_candidates,
            "significant_frequency": out_sig_freq,
            "stable_significant": out_stable_sig,
        }
    }

    save_json(summary, out_summary)

    # ========================================================
    # Print summary
    # ========================================================

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Total runs                         : {total_runs}")
    print(f"Unique significant candidates       : {len(df_sig_freq)}")
    print(f"Stable significant comorbidities    : {len(df_stable_sig)}")
    print(f"Minimum run required                : {threshold_count(args.min_significant_freq, total_runs)}")

    print("\nTop stable significant comorbidities:")
    cols = [
        "node",
        "significant_run_count",
        "significant_frequency",
        "mean_centrality_vote",
        "max_centrality_vote",
        "candidate_occurrences",
        "communities_found",
        "voted_by_summary",
    ]
    cols = [c for c in cols if c in df_stable_sig.columns]

    if not df_stable_sig.empty:
        print(df_stable_sig[cols].head(30).to_string(index=False))
    else:
        print("(Tidak ada node yang memenuhi threshold stable significant.)")

    print("\nSaved files:")
    for k, v in summary["outputs"].items():
        print(f"- {k}: {v}")
    print(f"- summary: {out_summary}")


if __name__ == "__main__":
    main()
