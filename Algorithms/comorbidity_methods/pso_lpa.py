# -*- coding: utf-8 -*-
"""
PSO-LPA / DPSO-BMLPA tuning and 20-run evaluation for comorbidity networks.

This script uses the DPSO-BMLPA logic as PSO-LPA candidate algorithm:
1. Tune parameters using topology-independent internal metrics.
2. Select the best parameter combination using composite score.
3. Run the best parameter combination 20 times.
4. Save best run outputs and mean ± std summary from the 20 runs.

Example commands:
python main.py --graph /home/toto/Eska/Data/nsclc_disease.csv --name NSCLC --outdir results_pso_lpa/nsclc
python main.py --graph /home/toto/Eska/Data/sclc_disease.csv  --name SCLC  --outdir results_pso_lpa/sclc

Default tuning grid:
- weight / inertia w : 0.5, 0.7, 0.9
- c1                : 1.0, 1.5, 2.0
- c2                : 1.0, 1.5, 2.0
- particles          : 50, 100
- max_iter           : 100, 300

Fixed parameters:
- rho = 0.5
- p   = 0.6

Composite score:
Composite = mean(M_norm, S_norm, 1 - E_norm)
where:
- fuzzy_modularity higher is better
- stability higher is better
- entropy lower is better
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

    G.remove_edges_from(nx.selfloop_edges(G))

    if G.number_of_nodes() == 0 or G.number_of_edges() == 0:
        raise ValueError("Graph is empty. Check the input CSV columns.")

    return G


# =====================================================
# 2. DPSO-BMLPA / PSO-LPA class
# =====================================================

class DPSOBMLPA:
    """
    DPSO + BMLPA implementation for PSO-LPA candidate algorithm.

    - Positions are encoded as integer community labels per node index.
    - DPSO optimizes hard modularity.
    - BMLPA-like rule identifies overlapping nodes from neighbor community context.
    """

    def __init__(self, graph, num_particles=30, max_iter=100, rho=0.5, sig_variant="paper", seed=None):
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        self.graph = graph
        self.node_list = list(graph.nodes())
        self.num_nodes = len(self.node_list)
        self.node_index = {node: idx for idx, node in enumerate(self.node_list)}
        self.index_node = {idx: node for idx, node in enumerate(self.node_list)}

        self.num_particles = num_particles
        self.max_iter = max_iter
        self.rho = rho
        self.sig_variant = sig_variant

        self.global_best_position = None
        self.global_best_modularity = -float("inf")
        self.particles = []

    def initialize_particle(self, p=0.6):
        position = [0] * self.num_nodes
        assigned = [False] * self.num_nodes
        k = 1

        while not all(assigned):
            start_idx = assigned.index(False)
            queue = [start_idx]
            assigned[start_idx] = True
            position[start_idx] = k

            while queue:
                cur_idx = queue.pop(0)
                cur_node = self.index_node[cur_idx]
                for nbr in self.graph.neighbors(cur_node):
                    nbr_idx = self.node_index[nbr]
                    if not assigned[nbr_idx]:
                        assigned[nbr_idx] = True
                        if random.random() > p:
                            position[nbr_idx] = k
                            queue.append(nbr_idx)
            k += 1

        return {
            "position": position,
            "velocity": [random.uniform(-1, 1) for _ in range(self.num_nodes)],
            "best_position": position.copy(),
            "best_modularity": -float("inf")
        }

    def evaluate_particle(self, particle):
        pos = particle["position"]
        communities = defaultdict(list)

        for idx, label in enumerate(pos):
            node = self.index_node[idx]
            communities[label].append(node)

        communities_list = [c for c in communities.values() if len(c) > 0]

        if len(communities_list) <= 1:
            return -1.0

        try:
            return nx.community.modularity(self.graph, communities_list, weight="weight")
        except Exception:
            return -1.0

    def update_velocity(self, particle, w, c1, c2):
        for i in range(self.num_nodes):
            v_old = particle["velocity"][i]
            inertia = w * v_old

            p_i = particle["best_position"][i]
            g_i = self.global_best_position[i] if self.global_best_position is not None else particle["position"][i]
            x_i = particle["position"][i]

            cognitive = c1 * random.random() * (1 if p_i != x_i else 0)
            social = c2 * random.random() * (1 if g_i != x_i else 0)
            random_explore = 0.1 * (random.random() - 0.5)

            v_new = inertia + cognitive + social + random_explore
            v_new = max(-4.0, min(4.0, v_new))
            particle["velocity"][i] = v_new

    def update_position(self, particle):
        pos = particle["position"]
        vel = particle["velocity"]
        current_max_label = max(pos) if pos else 0

        for i in range(self.num_nodes):
            s = 1.0 / (1.0 + math.exp(-vel[i]))

            if s > self.rho:
                choices = []
                weights = []

                choices.append(pos[i]); weights.append(0.35)

                p_label = particle["best_position"][i]
                if p_label != pos[i]:
                    choices.append(p_label); weights.append(0.25)

                if self.global_best_position is not None:
                    g_label = self.global_best_position[i]
                    if g_label != pos[i]:
                        choices.append(g_label); weights.append(0.25)

                node = self.index_node[i]
                nbrs = list(self.graph.neighbors(node))
                if nbrs:
                    nbr_idxs = [self.node_index[n] for n in nbrs]
                    nbr_labels = [pos[j] for j in nbr_idxs]
                    maj_label = max(set(nbr_labels), key=nbr_labels.count)
                    if maj_label != pos[i]:
                        choices.append(maj_label); weights.append(0.10)

                new_label = current_max_label + 1
                choices.append(new_label); weights.append(0.05)

                total = sum(weights)
                probs = [weight / total for weight in weights]
                chosen_label = random.choices(choices, probs, k=1)[0]

                pos[i] = chosen_label
                current_max_label = max(current_max_label, pos[i])

        particle["position"] = self._fix_invalid_partition(pos)

    def _fix_invalid_partition(self, labels):
        label_nodes = defaultdict(list)
        for idx, lab in enumerate(labels):
            label_nodes[lab].append(idx)

        new_labels = labels.copy()
        max_label = max(labels) if labels else 0

        for lab, nodes in list(label_nodes.items()):
            if lab == 0 or len(nodes) <= 1:
                continue

            sub_nodes = [self.index_node[idx] for idx in nodes]
            subG = self.graph.subgraph(sub_nodes)

            if subG.number_of_nodes() <= 1:
                continue

            if not nx.is_connected(subG):
                comps = list(nx.connected_components(subG))
                for comp_idx, comp in enumerate(comps):
                    if comp_idx == 0:
                        continue
                    max_label += 1
                    for node in comp:
                        idx = self.node_index[node]
                        new_labels[idx] = max_label

        return new_labels

    def detect_overlapping_bmlpa(self, communities_dict):
        node_to_comm = {}
        for cid, nodes in communities_dict.items():
            for node in nodes:
                node_to_comm.setdefault(node, set()).add(cid)

        overlapping = set()
        for node in self.graph.nodes():
            neighbor_comms = set()
            for nbr in self.graph.neighbors(node):
                if nbr in node_to_comm:
                    neighbor_comms.update(node_to_comm[nbr])
            if len(neighbor_comms) >= 2:
                overlapping.add(node)

        return sorted(overlapping)

    def run(self, w=0.9, c1=1.5, c2=2.0, p=0.6, verbose=True):
        self.particles = [self.initialize_particle(p=p) for _ in range(self.num_particles)]

        for particle in self.particles:
            score = self.evaluate_particle(particle)
            particle["best_modularity"] = score
            particle["best_position"] = particle["position"].copy()

            if score > self.global_best_modularity:
                self.global_best_modularity = score
                self.global_best_position = particle["position"].copy()

        if verbose:
            print(f"Initial global modularity: {self.global_best_modularity:.4f}")

        for iteration in range(self.max_iter):
            for particle in self.particles:
                self.update_velocity(particle, w, c1, c2)
                self.update_position(particle)

                score = self.evaluate_particle(particle)
                if score > particle["best_modularity"]:
                    particle["best_modularity"] = score
                    particle["best_position"] = particle["position"].copy()

                if score > self.global_best_modularity:
                    self.global_best_modularity = score
                    self.global_best_position = particle["position"].copy()

            if verbose and (iteration % max(1, self.max_iter // 10) == 0 or iteration == self.max_iter - 1):
                print(f"Iter {iteration + 1}/{self.max_iter} - Global best modularity: {self.global_best_modularity:.4f}")

        communities = defaultdict(list)
        for idx, label in enumerate(self.global_best_position):
            communities[label].append(self.index_node[idx])

        communities = dict(communities)
        overlapping_nodes = self.detect_overlapping_bmlpa(communities)

        return communities, overlapping_nodes


# =====================================================
# 3. Metric helpers
# =====================================================

def normalize_memberships(memberships):
    norm = {}
    for node, cmu in memberships.items():
        total = sum(float(v) for v in cmu.values())
        if total <= 0:
            norm[node] = {}
        else:
            norm[node] = {str(c): float(v) / total for c, v in cmu.items()}
    return norm


def build_memberships_from_pso_lpa(G, communities, overlapping_nodes):
    """
    Convert PSO-LPA hard communities and BMLPA overlapping nodes into fuzzy-like memberships.

    Non-overlapping node:
        membership = 1.0 for its primary DPSO community.

    Overlapping node:
        membership labels are derived from neighboring communities.
        Weight is based on neighbor community frequency plus primary community support.
    """
    node_primary = {}
    for cid, nodes in communities.items():
        for node in nodes:
            node_primary[node] = cid

    overlapping_set = set(overlapping_nodes)
    memberships = {}

    for node in G.nodes():
        primary = node_primary.get(node, str(node))

        if node not in overlapping_set:
            memberships[node] = {str(primary): 1.0}
            continue

        counts = defaultdict(float)
        counts[str(primary)] += 1.0

        for nbr in G.neighbors(node):
            nbr_comm = node_primary.get(nbr)
            if nbr_comm is not None:
                counts[str(nbr_comm)] += 1.0

        if len(counts) == 0:
            counts[str(primary)] = 1.0

        memberships[node] = dict(counts)

    return memberships


def membership_entropy(memberships):
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
    Fuzzy modularity using membership similarity.
    Q = 1/(2m) * sum_ij [A_ij - (k_i k_j)/(2m)] * sum_c(mu_ic * mu_jc)
    """
    if G.number_of_edges() == 0:
        return 0.0

    norm = normalize_memberships(memberships)
    nodes = list(G.nodes())

    degree = dict(G.degree(weight="weight"))
    m = G.size(weight="weight")

    if m <= 0:
        return 0.0

    adj = {u: {} for u in nodes}
    for u, v, data in G.edges(data=True):
        weight = float(data.get("weight", 1.0))
        adj[u][v] = weight
        adj[v][u] = weight

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


def minmax(values):
    arr = np.array(values, dtype=float)
    mn, mx = np.min(arr), np.max(arr)
    if np.isclose(mx - mn, 0):
        return np.ones_like(arr)
    return (arr - mn) / (mx - mn)


def add_composite_scores(df_metrics):
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

def evaluate_param_combo(G, w, c1, c2, num_particles, max_iter, rho=0.5, p=0.6, n_runs=3, base_seed=1000):
    run_outputs = []

    for r in range(n_runs):
        seed = base_seed + r
        model = DPSOBMLPA(
            G,
            num_particles=num_particles,
            max_iter=max_iter,
            rho=rho,
            seed=seed
        )
        communities, overlapping_nodes = model.run(
            w=w,
            c1=c1,
            c2=c2,
            p=p,
            verbose=False
        )

        memberships = build_memberships_from_pso_lpa(G, communities, overlapping_nodes)

        run_outputs.append({
            "run": r + 1,
            "seed": seed,
            "communities": communities,
            "overlapping_nodes": overlapping_nodes,
            "memberships": memberships,
            "num_communities": len(communities),
            "num_overlapping_nodes": len(overlapping_nodes),
            "hard_modularity": float(model.global_best_modularity),
            "fuzzy_modularity": fuzzy_modularity(G, memberships),
            "entropy": membership_entropy(memberships)
        })

    stabilities = pairwise_stability(G, run_outputs)
    for i, stability in enumerate(stabilities):
        run_outputs[i]["stability"] = stability

    per_run_df = pd.DataFrame([
        {
            "run": r["run"],
            "seed": r["seed"],
            "w": w,
            "c1": c1,
            "c2": c2,
            "num_particles": num_particles,
            "max_iter": max_iter,
            "rho": rho,
            "p": p,
            "num_communities": r["num_communities"],
            "num_overlapping_nodes": r["num_overlapping_nodes"],
            "hard_modularity": r["hard_modularity"],
            "fuzzy_modularity": r["fuzzy_modularity"],
            "stability": r["stability"],
            "entropy": r["entropy"]
        }
        for r in run_outputs
    ])

    per_run_df = add_composite_scores(per_run_df)

    summary = {
        "w": w,
        "c1": c1,
        "c2": c2,
        "num_particles": num_particles,
        "max_iter": max_iter,
        "rho": rho,
        "p": p,
        "runs": n_runs,
        "mean_num_communities": per_run_df["num_communities"].mean(),
        "mean_num_overlapping_nodes": per_run_df["num_overlapping_nodes"].mean(),
        "mean_hard_modularity": per_run_df["hard_modularity"].mean(),
        "mean_fuzzy_modularity": per_run_df["fuzzy_modularity"].mean(),
        "mean_stability": per_run_df["stability"].mean(),
        "mean_entropy": per_run_df["entropy"].mean(),
        "mean_composite": per_run_df["composite"].mean(),
        "std_composite": per_run_df["composite"].std(ddof=0)
    }

    return summary


def tune_pso_lpa(G, weights, c1_values, c2_values, particles, max_iters, rho=0.5, p=0.6, tune_runs=3, base_seed=1000):
    rows = []
    grid = list(product(weights, c1_values, c2_values, particles, max_iters))

    print(f"Total parameter combinations: {len(grid)}")

    for idx, (w, c1, c2, num_particles, max_iter) in enumerate(grid, start=1):
        print(
            f"[{idx}/{len(grid)}] Tuning w={w}, c1={c1}, c2={c2}, "
            f"particles={num_particles}, max_iter={max_iter}"
        )

        summary = evaluate_param_combo(
            G,
            w=w,
            c1=c1,
            c2=c2,
            num_particles=num_particles,
            max_iter=max_iter,
            rho=rho,
            p=p,
            n_runs=tune_runs,
            base_seed=base_seed + idx * 100
        )
        rows.append(summary)

    tuning_df = pd.DataFrame(rows)

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


def final_20_runs(G, w, c1, c2, num_particles, max_iter, rho=0.5, p=0.6, n_runs=20, base_seed=5000):
    run_outputs = []

    for r in range(n_runs):
        seed = base_seed + r
        print(f"Final run {r + 1}/{n_runs} | seed={seed}")

        model = DPSOBMLPA(
            G,
            num_particles=num_particles,
            max_iter=max_iter,
            rho=rho,
            seed=seed
        )
        communities, overlapping_nodes = model.run(
            w=w,
            c1=c1,
            c2=c2,
            p=p,
            verbose=False
        )

        memberships = build_memberships_from_pso_lpa(G, communities, overlapping_nodes)

        run_outputs.append({
            "run": r + 1,
            "seed": seed,
            "communities": communities,
            "overlapping_nodes": overlapping_nodes,
            "memberships": memberships,
            "num_communities": len(communities),
            "num_overlapping_nodes": len(overlapping_nodes),
            "hard_modularity": float(model.global_best_modularity),
            "fuzzy_modularity": fuzzy_modularity(G, memberships),
            "entropy": membership_entropy(memberships)
        })

    stabilities = pairwise_stability(G, run_outputs)
    for i, stability in enumerate(stabilities):
        run_outputs[i]["stability"] = stability

    per_run_df = pd.DataFrame([
        {
            "run": r["run"],
            "seed": r["seed"],
            "w": w,
            "c1": c1,
            "c2": c2,
            "num_particles": num_particles,
            "max_iter": max_iter,
            "rho": rho,
            "p": p,
            "num_communities": r["num_communities"],
            "num_overlapping_nodes": r["num_overlapping_nodes"],
            "hard_modularity": r["hard_modularity"],
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

def to_jsonable(obj):
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, tuple):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.ndarray,)):
        return obj.tolist()
    return obj


def save_json(obj, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(to_jsonable(obj), f, ensure_ascii=False, indent=2)


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

    overlapping_rows = []
    for node in best_run_output["overlapping_nodes"]:
        cmu = memberships.get(str(node), {})
        overlapping_rows.append({
            "node": str(node),
            "num_memberships": len(cmu),
            "memberships": json.dumps(cmu, ensure_ascii=False)
        })

    overlapping_df = pd.DataFrame(overlapping_rows)
    overlapping_df.to_csv(os.path.join(outdir, f"{prefix}_overlapping_nodes.csv"), index=False)


def save_20runs_summary(per_run_df, outdir, prefix):
    summary_cols = [
        "num_communities",
        "num_overlapping_nodes",
        "hard_modularity",
        "fuzzy_modularity",
        "stability",
        "entropy",
        "composite"
    ]

    rows = []
    for col in summary_cols:
        rows.append({
            "metric": col,
            "mean": per_run_df[col].mean(),
            "std": per_run_df[col].std(ddof=0),
            "min": per_run_df[col].min(),
            "max": per_run_df[col].max()
        })

    summary_df = pd.DataFrame(rows)
    summary_path = os.path.join(outdir, f"{prefix}_pso_lpa_20runs_summary.csv")
    summary_df.to_csv(summary_path, index=False)

    compact_summary = {
        "mean_num_communities": float(per_run_df["num_communities"].mean()),
        "std_num_communities": float(per_run_df["num_communities"].std(ddof=0)),
        "mean_num_overlapping_nodes": float(per_run_df["num_overlapping_nodes"].mean()),
        "std_num_overlapping_nodes": float(per_run_df["num_overlapping_nodes"].std(ddof=0)),
        "mean_hard_modularity": float(per_run_df["hard_modularity"].mean()),
        "std_hard_modularity": float(per_run_df["hard_modularity"].std(ddof=0)),
        "mean_fuzzy_modularity": float(per_run_df["fuzzy_modularity"].mean()),
        "std_fuzzy_modularity": float(per_run_df["fuzzy_modularity"].std(ddof=0)),
        "mean_stability": float(per_run_df["stability"].mean()),
        "std_stability": float(per_run_df["stability"].std(ddof=0)),
        "mean_entropy": float(per_run_df["entropy"].mean()),
        "std_entropy": float(per_run_df["entropy"].std(ddof=0)),
        "mean_composite": float(per_run_df["composite"].mean()),
        "std_composite": float(per_run_df["composite"].std(ddof=0))
    }

    compact_path = os.path.join(outdir, f"{prefix}_pso_lpa_20runs_compact_summary.json")
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
    parser = argparse.ArgumentParser(description="PSO-LPA / DPSO-BMLPA tuning and final 20 runs")

    parser.add_argument("--graph", required=True, help="Path to disease network CSV")
    parser.add_argument("--name", default="network", help="Network name, e.g., NSCLC or SCLC")
    parser.add_argument("--outdir", default="pso_lpa_results", help="Output directory")

    parser.add_argument("--weights", default="0.5,0.7,0.9", help="Comma-separated inertia weights")
    parser.add_argument("--c1-values", default="1.0,1.5,2.0", help="Comma-separated c1 values")
    parser.add_argument("--c2-values", default="1.0,1.5,2.0", help="Comma-separated c2 values")
    parser.add_argument("--particles", default="50,100", help="Comma-separated particle counts")
    parser.add_argument("--max-iters", default="100,300", help="Comma-separated max_iter values")

    parser.add_argument("--rho", type=float, default=0.5, help="Sigmoid threshold rho")
    parser.add_argument("--p", type=float, default=0.6, help="Initialization BFS probability parameter")

    parser.add_argument("--tune-runs", type=int, default=3, help="Runs per parameter combo during tuning")
    parser.add_argument("--final-runs", type=int, default=20, help="Final runs using best parameters")
    parser.add_argument("--seed", type=int, default=1234, help="Base random seed")

    parser.add_argument("--source-col", default=None, help="Optional source column")
    parser.add_argument("--target-col", default=None, help="Optional target column")
    parser.add_argument("--weight-col", default=None, help="Optional weight column")

    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    prefix = args.name.lower()

    weights = parse_float_list(args.weights)
    c1_values = parse_float_list(args.c1_values)
    c2_values = parse_float_list(args.c2_values)
    particles = parse_int_list(args.particles)
    max_iters = parse_int_list(args.max_iters)

    print("=" * 70)
    print(f"Network       : {args.name}")
    print(f"Graph path    : {args.graph}")
    print(f"Output dir    : {args.outdir}")
    print(f"Weights       : {weights}")
    print(f"C1 values     : {c1_values}")
    print(f"C2 values     : {c2_values}")
    print(f"Particles     : {particles}")
    print(f"Max iterations: {max_iters}")
    print(f"rho           : {args.rho}")
    print(f"p             : {args.p}")
    print(f"Tune runs     : {args.tune_runs}")
    print(f"Final runs    : {args.final_runs}")
    print("=" * 70)

    G = load_graph_from_csv(
        args.graph,
        source_col=args.source_col,
        target_col=args.target_col,
        weight_col=args.weight_col
    )

    #print(f"Loaded graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    # Mengambil largest connected component
    largest_component_nodes = max(nx.connected_components(G), key=len)
    G = G.subgraph(largest_component_nodes).copy()

    print(
        f"Largest connected component: "
        f"{G.number_of_nodes()} nodes, {G.number_of_edges()} edges"
    )
    
    tuning_df, best_params = tune_pso_lpa(
        G,
        weights=weights,
        c1_values=c1_values,
        c2_values=c2_values,
        particles=particles,
        max_iters=max_iters,
        rho=args.rho,
        p=args.p,
        tune_runs=args.tune_runs,
        base_seed=args.seed
    )

    tuning_path = os.path.join(args.outdir, f"{prefix}_pso_lpa_tuning.csv")
    tuning_df.to_csv(tuning_path, index=False)

    print("\nBest parameters from tuning:")
    print(best_params)

    best_w = float(best_params["w"])
    best_c1 = float(best_params["c1"])
    best_c2 = float(best_params["c2"])
    best_num_particles = int(best_params["num_particles"])
    best_max_iter = int(best_params["max_iter"])

    per_run_df, best_run_output = final_20_runs(
        G,
        w=best_w,
        c1=best_c1,
        c2=best_c2,
        num_particles=best_num_particles,
        max_iter=best_max_iter,
        rho=args.rho,
        p=args.p,
        n_runs=args.final_runs,
        base_seed=args.seed + 10000
    )

    final_path = os.path.join(args.outdir, f"{prefix}_pso_lpa_20runs.csv")
    per_run_df.to_csv(final_path, index=False)

    summary_df, compact_summary = save_20runs_summary(per_run_df, args.outdir, prefix)

    save_best_outputs(best_run_output, args.outdir, prefix)

    best_summary = per_run_df.iloc[0].to_dict()
    save_json(best_summary, os.path.join(args.outdir, f"{prefix}_best_run_summary.json"))

    print("\nFinal 20-run results saved to:")
    print(final_path)

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
