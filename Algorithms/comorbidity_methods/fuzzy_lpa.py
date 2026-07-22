# -*- coding: utf-8 -*-
"""
Fuzzy-LPA tuning and 20-run evaluation for comorbidity networks.

Example:
python main.py --graph /home/toto/Eska/Data/nsclc_disease.csv --name NSCLC --outdir results_fuzzy_lpa/nsclc
python main.py --graph /home/toto/Eska/Data/sclc_disease.csv  --name SCLC  --outdir results_fuzzy_lpa/sclc

Default tuning grid:
- threshold: 0.01, 0.05, 0.10
- lambda   : 1.0, 2.0, 5.0
- max_iter : 50, 100, 150

Selection logic:
1. Tune parameters using repeated runs and composite score.
2. Run best parameters 20 times.
3. Compute fuzzy modularity, stability, entropy, and composite score.
4. Select the best run based on highest composite score.
5. Save mean and standard deviation summary from the final runs.
"""

import argparse
import json
import math
import os
import random
from collections import defaultdict
from itertools import combinations, product

import networkx as nx
import numpy as np
import pandas as pd
from sklearn.metrics import normalized_mutual_info_score


# =====================================================
# 1. Graph loader
# =====================================================

def load_graph_from_csv(path, source_col=None, target_col=None, weight_col=None):
    """
    Load graph from CSV edge list.

    Supported formats:
    - source,target
    - source,target,weight
    - disease1,disease2,similarity
    - any CSV with at least two columns: first two columns are treated as edges.

    Weight is optional. Fuzzy-LPA below uses unweighted neighborhood voting,
    but weights are stored in the graph for possible later analysis.
    """
    df = pd.read_csv(path)

    if df.shape[1] < 2:
        raise ValueError("CSV must have at least two columns for source and target nodes.")

    cols = list(df.columns)

    if source_col is None or target_col is None:
        lower_cols = {c.lower(): c for c in cols}
        possible_sources = ["source", "src", "node1", "disease1", "from"]
        possible_targets = ["target", "dst", "node2", "disease2", "to"]

        source_col = next((lower_cols[c] for c in possible_sources if c in lower_cols), cols[0])
        target_col = next((lower_cols[c] for c in possible_targets if c in lower_cols), cols[1])

    if weight_col is None:
        lower_cols = {c.lower(): c for c in cols}
        possible_weights = ["weight", "similarity", "score", "wang", "sim"]
        weight_col = next((lower_cols[c] for c in possible_weights if c in lower_cols), None)

        # fallback: use third column if numeric
        if weight_col is None and df.shape[1] >= 3:
            third_col = cols[2]
            if pd.api.types.is_numeric_dtype(df[third_col]):
                weight_col = third_col

    G = nx.Graph()

    for _, row in df.iterrows():
        u = row[source_col]
        v = row[target_col]

        if pd.isna(u) or pd.isna(v):
            continue

        if weight_col is not None and weight_col in df.columns and not pd.isna(row[weight_col]):
            try:
                w = float(row[weight_col])
            except Exception:
                w = 1.0
        else:
            w = 1.0

        G.add_edge(str(u), str(v), weight=w)

    # remove self-loops if any
    G.remove_edges_from(nx.selfloop_edges(G))

    if G.number_of_nodes() == 0 or G.number_of_edges() == 0:
        raise ValueError("Graph is empty. Check the input CSV columns.")

    return G


# =====================================================
# 2. Fuzzy-LPA
# =====================================================

def run_fuzzy_lpa(G, max_iter=50, threshold=0.05, lambd=1.0, seed=None):
    """
    Fuzzy Label Propagation Algorithm based on user-provided function.

    Returns:
    - community_dict: {community_label: [nodes]}
    - memberships: {node: {community_label: membership_value}}
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    labels = {u: u for u in G.nodes()}

    for _ in range(max_iter):
        changes = 0

        # shuffled order helps reduce deterministic bias
        nodes = list(G.nodes())
        random.shuffle(nodes)

        for u in nodes:
            label_count = {}
            for v in G.neighbors(u):
                lbl = labels[v]
                label_count[lbl] = label_count.get(lbl, 0) + 1

            if not label_count:
                continue

            max_count = max(label_count.values())
            max_labels = [lbl for lbl, count in label_count.items() if count == max_count]
            new_label = random.choice(max_labels)

            if labels[u] != new_label:
                labels[u] = new_label
                changes += 1

        if changes == 0:
            break

    # Compute fuzzy memberships
    memberships = {}
    for u in G.nodes():
        label_count = {}
        for v in G.neighbors(u):
            lbl = labels[v]
            label_count[lbl] = label_count.get(lbl, 0) + 1

        deg_u = G.degree[u]
        memberships[u] = {}

        for lbl, count in label_count.items():
            gamma = count / (deg_u + 1)
            mu = 1 - math.exp(-lambd * gamma)
            if mu >= threshold:
                memberships[u][lbl] = mu

        # fallback for isolated/empty membership nodes
        if len(memberships[u]) == 0:
            memberships[u][labels[u]] = 1.0

    community_dict = defaultdict(list)
    for node, labels_mus in memberships.items():
        for label in labels_mus:
            community_dict[label].append(node)

    return dict(community_dict), dict(memberships)


# =====================================================
# 3. Metric helpers
# =====================================================

def normalize_memberships(memberships):
    """Normalize membership values per node so they sum to 1."""
    norm = {}
    for node, cmu in memberships.items():
        total = sum(float(v) for v in cmu.values())
        if total <= 0:
            norm[node] = {}
        else:
            norm[node] = {str(c): float(v) / total for c, v in cmu.items()}
    return norm


def membership_entropy(memberships):
    """
    Mean normalized entropy of fuzzy memberships.
    Lower value indicates clearer membership.
    Range is approximately 0-1.
    """
    norm = normalize_memberships(memberships)
    entropies = []

    for _, cmu in norm.items():
        probs = np.array(list(cmu.values()), dtype=float)
        probs = probs[probs > 0]

        if len(probs) <= 1:
            entropies.append(0.0)
        else:
            H = -np.sum(probs * np.log(probs))
            H_norm = H / np.log(len(probs))
            entropies.append(float(H_norm))

    return float(np.mean(entropies)) if entropies else 0.0


def dominant_labels(G, memberships):
    """Convert fuzzy memberships into dominant hard labels for stability calculation."""
    labels = []
    for node in G.nodes():
        cmu = memberships.get(node, {})
        if not cmu:
            labels.append(str(node))
        else:
            best_label = max(cmu.items(), key=lambda x: x[1])[0]
            labels.append(str(best_label))
    return labels


def pairwise_stability(G, run_outputs):
    """
    Stability per run = mean NMI between dominant labels of one run
    and all other runs.
    """
    n = len(run_outputs)
    if n <= 1:
        return [1.0]

    hard_labels = [dominant_labels(G, r["memberships"]) for r in run_outputs]
    stability_vals = [[] for _ in range(n)]

    for i, j in combinations(range(n), 2):
        score = normalized_mutual_info_score(hard_labels[i], hard_labels[j])
        stability_vals[i].append(score)
        stability_vals[j].append(score)

    return [float(np.mean(vals)) if vals else 0.0 for vals in stability_vals]


def fuzzy_modularity(G, memberships):
    """
    Fuzzy modularity using membership similarity:
    Q = 1/(2m) * sum_ij [A_ij - (k_i k_j)/(2m)] * sum_c(mu_ic * mu_jc)

    Memberships are normalized per node before calculation.
    For weighted graph, edge weights are used in A_ij and weighted degree.
    """
    if G.number_of_edges() == 0:
        return 0.0

    norm = normalize_memberships(memberships)
    nodes = list(G.nodes())

    # weighted degree and total edge weight
    degree = dict(G.degree(weight="weight"))
    m = G.size(weight="weight")

    if m <= 0:
        return 0.0

    # adjacency lookup
    adj = {u: {} for u in nodes}
    for u, v, data in G.edges(data=True):
        w = float(data.get("weight", 1.0))
        adj[u][v] = w
        adj[v][u] = w

    Q = 0.0
    two_m = 2.0 * m

    for u in nodes:
        mu_u = norm.get(u, {})
        ku = degree.get(u, 0.0)
        for v in nodes:
            mu_v = norm.get(v, {})
            kv = degree.get(v, 0.0)

            if not mu_u or not mu_v:
                sim_uv = 0.0
            else:
                common_labels = set(mu_u.keys()) & set(mu_v.keys())
                sim_uv = sum(mu_u[c] * mu_v[c] for c in common_labels)

            Auv = adj[u].get(v, 0.0)
            expected = (ku * kv) / two_m
            Q += (Auv - expected) * sim_uv

    return float(Q / two_m)


def count_overlapping_nodes(memberships):
    """Count nodes with membership in more than one community."""
    return int(sum(1 for cmu in memberships.values() if len(cmu) > 1))


def minmax(values):
    arr = np.array(values, dtype=float)
    mn, mx = np.min(arr), np.max(arr)
    if np.isclose(mx - mn, 0):
        return np.ones_like(arr)
    return (arr - mn) / (mx - mn)


def add_composite_scores(df_metrics):
    """
    Composite = mean(M_norm, S_norm, 1 - E_norm)
    where:
    - fuzzy_modularity higher is better
    - stability higher is better
    - entropy lower is better
    """
    out = df_metrics.copy()
    out["M_norm"] = minmax(out["fuzzy_modularity"].values)
    out["S_norm"] = minmax(out["stability"].values)
    out["E_norm"] = minmax(out["entropy"].values)
    out["entropy_score"] = 1.0 - out["E_norm"]

    out["composite"] = (
        out["M_norm"] +
        out["S_norm"] +
        out["entropy_score"]
    ) / 3.0

    return out


# =====================================================
# 4. Tuning and final runs
# =====================================================

def evaluate_param_combo(G, threshold, lambd, max_iter, n_runs=5, base_seed=1000):
    run_outputs = []

    for r in range(n_runs):
        seed = base_seed + r
        communities, memberships = run_fuzzy_lpa(
            G,
            max_iter=max_iter,
            threshold=threshold,
            lambd=lambd,
            seed=seed
        )

        run_outputs.append({
            "run": r + 1,
            "seed": seed,
            "communities": communities,
            "memberships": memberships,
            "num_communities": len(communities),
            "num_overlapping_nodes": count_overlapping_nodes(memberships),
            "fuzzy_modularity": fuzzy_modularity(G, memberships),
            "entropy": membership_entropy(memberships)
        })

    stabilities = pairwise_stability(G, run_outputs)

    for i, s in enumerate(stabilities):
        run_outputs[i]["stability"] = s

    per_run_df = pd.DataFrame([
        {
            "run": r["run"],
            "seed": r["seed"],
            "threshold": threshold,
            "lambda": lambd,
            "max_iter": max_iter,
            "num_communities": r["num_communities"],
            "num_overlapping_nodes": r["num_overlapping_nodes"],
            "fuzzy_modularity": r["fuzzy_modularity"],
            "stability": r["stability"],
            "entropy": r["entropy"]
        }
        for r in run_outputs
    ])

    per_run_df = add_composite_scores(per_run_df)

    summary = {
        "threshold": threshold,
        "lambda": lambd,
        "max_iter": max_iter,
        "runs": n_runs,
        "mean_num_communities": per_run_df["num_communities"].mean(),
        "std_num_communities": per_run_df["num_communities"].std(ddof=0),
        "mean_num_overlapping_nodes": per_run_df["num_overlapping_nodes"].mean(),
        "std_num_overlapping_nodes": per_run_df["num_overlapping_nodes"].std(ddof=0),
        "mean_fuzzy_modularity": per_run_df["fuzzy_modularity"].mean(),
        "std_fuzzy_modularity": per_run_df["fuzzy_modularity"].std(ddof=0),
        "mean_stability": per_run_df["stability"].mean(),
        "std_stability": per_run_df["stability"].std(ddof=0),
        "mean_entropy": per_run_df["entropy"].mean(),
        "std_entropy": per_run_df["entropy"].std(ddof=0),
        "mean_composite": per_run_df["composite"].mean(),
        "std_composite": per_run_df["composite"].std(ddof=0)
    }

    return summary


def tune_fuzzy_lpa(G, thresholds, lambdas, max_iters, tune_runs=5, base_seed=1000):
    rows = []

    grid = list(product(thresholds, lambdas, max_iters))
    print(f"Total parameter combinations: {len(grid)}")

    for idx, (threshold, lambd, max_iter) in enumerate(grid, start=1):
        print(
            f"[{idx}/{len(grid)}] Tuning threshold={threshold}, "
            f"lambda={lambd}, max_iter={max_iter}"
        )

        summary = evaluate_param_combo(
            G,
            threshold=threshold,
            lambd=lambd,
            max_iter=max_iter,
            n_runs=tune_runs,
            base_seed=base_seed + idx * 100
        )
        rows.append(summary)

    tuning_df = pd.DataFrame(rows)

    # Normalize summary metrics across parameter combinations
    tuning_df["M_norm"] = minmax(tuning_df["mean_fuzzy_modularity"].values)
    tuning_df["S_norm"] = minmax(tuning_df["mean_stability"].values)
    tuning_df["E_norm"] = minmax(tuning_df["mean_entropy"].values)
    tuning_df["entropy_score"] = 1.0 - tuning_df["E_norm"]

    tuning_df["selection_composite"] = (
        tuning_df["M_norm"] +
        tuning_df["S_norm"] +
        tuning_df["entropy_score"]
    ) / 3.0

    tuning_df = tuning_df.sort_values(
        ["selection_composite", "mean_fuzzy_modularity", "mean_stability"],
        ascending=[False, False, False]
    ).reset_index(drop=True)

    tuning_df["rank"] = tuning_df.index + 1

    best = tuning_df.iloc[0].to_dict()
    return tuning_df, best


def final_20_runs(G, threshold, lambd, max_iter, n_runs=20, base_seed=5000):
    run_outputs = []

    for r in range(n_runs):
        seed = base_seed + r
        communities, memberships = run_fuzzy_lpa(
            G,
            max_iter=max_iter,
            threshold=threshold,
            lambd=lambd,
            seed=seed
        )

        run_outputs.append({
            "run": r + 1,
            "seed": seed,
            "communities": communities,
            "memberships": memberships,
            "num_communities": len(communities),
            "num_overlapping_nodes": count_overlapping_nodes(memberships),
            "fuzzy_modularity": fuzzy_modularity(G, memberships),
            "entropy": membership_entropy(memberships)
        })

    stabilities = pairwise_stability(G, run_outputs)

    for i, s in enumerate(stabilities):
        run_outputs[i]["stability"] = s

    per_run_df = pd.DataFrame([
        {
            "run": r["run"],
            "seed": r["seed"],
            "threshold": threshold,
            "lambda": lambd,
            "max_iter": max_iter,
            "num_communities": r["num_communities"],
            "num_overlapping_nodes": r["num_overlapping_nodes"],
            "fuzzy_modularity": r["fuzzy_modularity"],
            "stability": r["stability"],
            "entropy": r["entropy"]
        }
        for r in run_outputs
    ])

    per_run_df = add_composite_scores(per_run_df)
    per_run_df = per_run_df.sort_values("composite", ascending=False).reset_index(drop=True)
    per_run_df["rank"] = per_run_df.index + 1

    best_run_number = int(per_run_df.iloc[0]["run"])
    best_run_output = next(r for r in run_outputs if r["run"] == best_run_number)

    return per_run_df, best_run_output


# =====================================================
# 5. Save helpers
# =====================================================

def save_json(obj, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def save_best_outputs(best_run_output, outdir, prefix):
    communities = {
        str(k): [str(x) for x in v]
        for k, v in best_run_output["communities"].items()
    }

    memberships = {
        str(node): {str(c): float(mu) for c, mu in cmu.items()}
        for node, cmu in best_run_output["memberships"].items()
    }

    save_json(communities, os.path.join(outdir, f"{prefix}_best_communities.json"))
    save_json(memberships, os.path.join(outdir, f"{prefix}_best_memberships.json"))

    # Overlapping node table
    overlapping_rows = []
    for node, cmu in memberships.items():
        if len(cmu) > 1:
            overlapping_rows.append({
                "node": node,
                "num_memberships": len(cmu),
                "memberships": json.dumps(cmu, ensure_ascii=False)
            })

    overlapping_df = pd.DataFrame(overlapping_rows)
    overlapping_df.to_csv(os.path.join(outdir, f"{prefix}_overlapping_nodes.csv"), index=False)


def save_20runs_summary(per_run_df, outdir, prefix):
    """
    Save mean, standard deviation, min, and max from final runs.
    This summary is used for Chapter 4.3 reporting.
    """
    summary_cols = [
        "num_communities",
        "num_overlapping_nodes",
        "fuzzy_modularity",
        "stability",
        "entropy",
        "composite"
    ]

    summary_rows = []
    for col in summary_cols:
        summary_rows.append({
            "metric": col,
            "mean": float(per_run_df[col].mean()),
            "std": float(per_run_df[col].std(ddof=0)),
            "min": float(per_run_df[col].min()),
            "max": float(per_run_df[col].max())
        })

    summary_df = pd.DataFrame(summary_rows)

    summary_path = os.path.join(outdir, f"{prefix}_fuzzy_lpa_20runs_summary.csv")
    summary_df.to_csv(summary_path, index=False)

    compact_summary = {
        "mean_num_communities": float(per_run_df["num_communities"].mean()),
        "std_num_communities": float(per_run_df["num_communities"].std(ddof=0)),
        "mean_num_overlapping_nodes": float(per_run_df["num_overlapping_nodes"].mean()),
        "std_num_overlapping_nodes": float(per_run_df["num_overlapping_nodes"].std(ddof=0)),
        "mean_fuzzy_modularity": float(per_run_df["fuzzy_modularity"].mean()),
        "std_fuzzy_modularity": float(per_run_df["fuzzy_modularity"].std(ddof=0)),
        "mean_stability": float(per_run_df["stability"].mean()),
        "std_stability": float(per_run_df["stability"].std(ddof=0)),
        "mean_entropy": float(per_run_df["entropy"].mean()),
        "std_entropy": float(per_run_df["entropy"].std(ddof=0)),
        "mean_composite": float(per_run_df["composite"].mean()),
        "std_composite": float(per_run_df["composite"].std(ddof=0))
    }

    compact_path = os.path.join(outdir, f"{prefix}_fuzzy_lpa_20runs_compact_summary.json")
    save_json(compact_summary, compact_path)

    return summary_df, compact_summary


# =====================================================
# 6. Main
# =====================================================

def parse_float_list(text):
    return [float(x.strip()) for x in text.split(",") if x.strip()]


def parse_int_list(text):
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def main():
    parser = argparse.ArgumentParser(description="Fuzzy-LPA tuning and final 20 runs")

    parser.add_argument("--graph", required=True, help="Path to disease network CSV")
    parser.add_argument("--name", default="network", help="Network name, e.g., NSCLC or SCLC")
    parser.add_argument("--outdir", default="fuzzy_lpa_results", help="Output directory")

    parser.add_argument("--thresholds", default="0.01,0.05,0.1", help="Comma-separated thresholds")
    parser.add_argument("--lambdas", default="1.0,2.0,5.0", help="Comma-separated lambda values")
    parser.add_argument("--max-iters", default="50,100,150", help="Comma-separated max_iter values")

    parser.add_argument("--tune-runs", type=int, default=5, help="Runs per parameter combo during tuning")
    parser.add_argument("--final-runs", type=int, default=20, help="Final runs using best parameters")
    parser.add_argument("--seed", type=int, default=1234, help="Base random seed")

    parser.add_argument("--source-col", default=None, help="Optional source column")
    parser.add_argument("--target-col", default=None, help="Optional target column")
    parser.add_argument("--weight-col", default=None, help="Optional weight column")

    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    prefix = args.name.lower()

    thresholds = parse_float_list(args.thresholds)
    lambdas = parse_float_list(args.lambdas)
    max_iters = parse_int_list(args.max_iters)

    print("=" * 70)
    print(f"Network       : {args.name}")
    print(f"Graph path    : {args.graph}")
    print(f"Output dir    : {args.outdir}")
    print(f"Thresholds    : {thresholds}")
    print(f"Lambdas       : {lambdas}")
    print(f"Max iterations: {max_iters}")
    print("=" * 70)

    G = load_graph_from_csv(
        args.graph,
        source_col=args.source_col,
        target_col=args.target_col,
        weight_col=args.weight_col
    )

    print(f"Loaded graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    # Ambil giant connected component / komponen terbesar
    if not nx.is_connected(G):
        largest_cc = max(nx.connected_components(G), key=len)
        G = G.subgraph(largest_cc).copy()
        print(f"Using largest connected component: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    else:
        print("Graph is connected. Using full graph.")
    # -------------------------------
    # Tuning
    # -------------------------------
    tuning_df, best_params = tune_fuzzy_lpa(
        G,
        thresholds=thresholds,
        lambdas=lambdas,
        max_iters=max_iters,
        tune_runs=args.tune_runs,
        base_seed=args.seed
    )

    tuning_path = os.path.join(args.outdir, f"{prefix}_fuzzy_lpa_tuning.csv")
    tuning_df.to_csv(tuning_path, index=False)

    print("\nBest parameters from tuning:")
    print(best_params)

    best_threshold = float(best_params["threshold"])
    best_lambda = float(best_params["lambda"])
    best_max_iter = int(best_params["max_iter"])

    # -------------------------------
    # Final 20 runs
    # -------------------------------
    per_run_df, best_run_output = final_20_runs(
        G,
        threshold=best_threshold,
        lambd=best_lambda,
        max_iter=best_max_iter,
        n_runs=args.final_runs,
        base_seed=args.seed + 10000
    )

    final_path = os.path.join(args.outdir, f"{prefix}_fuzzy_lpa_20runs.csv")
    per_run_df.to_csv(final_path, index=False)

    # Save summary from all final runs
    summary_df, compact_summary = save_20runs_summary(per_run_df, args.outdir, prefix)

    # Save best run outputs
    save_best_outputs(best_run_output, args.outdir, prefix)

    best_summary = per_run_df.iloc[0].to_dict()
    save_json(best_summary, os.path.join(args.outdir, f"{prefix}_best_run_summary.json"))

    print("\nFinal 20-run results saved to:")
    print(final_path)

    print("\n20-run summary saved to:")
    print(os.path.join(args.outdir, f"{prefix}_fuzzy_lpa_20runs_summary.csv"))
    print(os.path.join(args.outdir, f"{prefix}_fuzzy_lpa_20runs_compact_summary.json"))

    print("\n20-run summary:")
    print(summary_df.to_string(index=False))

    print("\nCompact summary:")
    print(compact_summary)

    print("\nBest run summary:")
    print(best_summary)

    print("\nTop 5 final runs:")
    print(per_run_df.head(5).to_string(index=False))


if __name__ == "__main__":
    main()
