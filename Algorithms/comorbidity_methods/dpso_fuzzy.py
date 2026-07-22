# -*- coding: utf-8 -*-
"""
dpso_fuzzy.py

DPSO-Fuzzy for comorbidity networks.

Design:
1. DPSO parameters are loaded from PSO-LPA best_run_summary.json.
2. DPSO is used to generate initial hard communities.
3. MSEFCD-inspired fuzzy refinement is applied to DPSO communities using
   one smoothing step followed by one enhancement step per fuzzy iteration.
4. alpha_threshold is fixed from benchmark calibration.
5. alpha_self, beta_neighbor, prior_boost, and k_comm_candidates are tuned
   using unsupervised internal metrics:
   - fuzzy modularity  : higher is better
   - stability         : higher is better
   - entropy           : lower is better
6. Final 20 runs are executed using fixed DPSO parameters and best fuzzy parameters.

Example NSCLC:
python dpso_fuzzy.py \
  --graph /home/toto/Eska/Data/nsclc_disease.csv \
  --name NSCLC \
  --dpso-params /home/toto/Eska/top3/results_pso_lpa/nsclc/nsclc_best_run_summary.json \
  --outdir /home/toto/Eska/top3/results_dpso_fuzzy/nsclc \
  --alpha-threshold 0.4

Example SCLC:
python dpso_fuzzy.py \
  --graph /home/toto/Eska/Data/sclc_disease.csv \
  --name SCLC \
  --dpso-params /home/toto/Eska/top3/results_pso_lpa/sclc/sclc_best_run_summary.json \
  --outdir /home/toto/Eska/top3/results_dpso_fuzzy/sclc \
  --alpha-threshold 0.4
"""

import argparse
import json
import math
import os
import random
from collections import Counter, defaultdict
from itertools import combinations, product
from typing import Dict, List, Tuple, Any

import networkx as nx
import numpy as np
import pandas as pd
from sklearn.metrics import normalized_mutual_info_score

try:
    from cdlib.classes import NodeClustering
    from cdlib import evaluation
    HAS_CDLIB = True
except Exception:
    HAS_CDLIB = False


# =====================================================
# 1. Utilities
# =====================================================

def parse_float_list(text: str) -> List[float]:
    return [float(x.strip()) for x in text.split(",") if x.strip()]


def parse_int_list(text: str) -> List[int]:
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def parse_alpha_beta_grid(text: str) -> List[Tuple[float, float]]:
    """
    Format: "0.8:0.2,0.7:0.3,0.6:0.4,0.5:0.5"
    """
    pairs = []
    for item in text.split(","):
        item = item.strip()
        if not item:
            continue
        if ":" not in item:
            raise ValueError("alpha_beta_grid must use format alpha:beta, e.g. 0.7:0.3")
        a, b = item.split(":", 1)
        pairs.append((float(a.strip()), float(b.strip())))
    return pairs


def save_json(obj: Any, path: str):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def minmax(values):
    arr = np.array(values, dtype=float)
    if len(arr) == 0:
        return arr
    mn, mx = np.min(arr), np.max(arr)
    if np.isclose(mx - mn, 0):
        return np.ones_like(arr)
    return (arr - mn) / (mx - mn)


# =====================================================
# 2. Graph loader
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
        raise ValueError("Graph is empty. Check input CSV columns.")

    return G


# =====================================================
# 3. DPSO from PSO-LPA / DPSO-BMLPA
# =====================================================

class DPSOCommunity:
    """
    DPSO community detection using the same discrete PSO structure as PSO-LPA/DPSO-BMLPA.

    Position is encoded as integer community label per node index.
    """

    def __init__(self, graph, num_particles=100, max_iter=300, rho=0.5, seed=None):
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        self.graph = graph
        self.node_list = list(graph.nodes())
        self.num_nodes = len(self.node_list)
        self.node_index = {node: idx for idx, node in enumerate(self.node_list)}
        self.index_node = {idx: node for idx, node in enumerate(self.node_list)}

        self.num_particles = int(num_particles)
        self.max_iter = int(max_iter)
        self.rho = float(rho)

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
            "best_modularity": -float("inf"),
        }

    def position_to_communities(self, position):
        communities = defaultdict(list)
        for idx, label in enumerate(position):
            communities[label].append(self.index_node[idx])
        return {k: v for k, v in communities.items() if len(v) > 0}

    def position_to_assignment(self, position):
        return {self.index_node[idx]: int(label) for idx, label in enumerate(position)}

    def evaluate_particle(self, particle):
        communities_dict = self.position_to_communities(particle["position"])
        communities_list = [nodes for nodes in communities_dict.values() if len(nodes) > 0]

        if len(communities_list) <= 1:
            return -1.0
        try:
            return float(nx.community.modularity(self.graph, communities_list, weight="weight"))
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
                probs = [x / total for x in weights]
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

    def run(self, w=0.5, c1=1.0, c2=1.0, p=0.6, verbose=False):
        self.particles = [self.initialize_particle(p=p) for _ in range(self.num_particles)]

        for particle in self.particles:
            score = self.evaluate_particle(particle)
            particle["best_modularity"] = score
            particle["best_position"] = particle["position"].copy()
            if score > self.global_best_modularity:
                self.global_best_modularity = score
                self.global_best_position = particle["position"].copy()

        if verbose:
            print(f"Initial global modularity: {self.global_best_modularity:.6f}")

        for iteration in range(self.max_iter):
            for particle in self.particles:
                self.update_velocity(particle, w=w, c1=c1, c2=c2)
                self.update_position(particle)

                score = self.evaluate_particle(particle)
                if score > particle["best_modularity"]:
                    particle["best_modularity"] = score
                    particle["best_position"] = particle["position"].copy()

                if score > self.global_best_modularity:
                    self.global_best_modularity = score
                    self.global_best_position = particle["position"].copy()

            if verbose and (iteration % max(1, self.max_iter // 10) == 0 or iteration == self.max_iter - 1):
                print(f"Iter {iteration + 1}/{self.max_iter} - Global best modularity: {self.global_best_modularity:.6f}")

        communities = self.position_to_communities(self.global_best_position)
        assignment = self.position_to_assignment(self.global_best_position)

        return communities, assignment, float(self.global_best_modularity)


# =====================================================
# 4. Run fixed DPSO parameters
# =====================================================

def load_dpso_params_from_summary(path):
    data = load_json(path)
    required = ["w", "c1", "c2", "num_particles", "max_iter"]
    for key in required:
        if key not in data:
            raise ValueError(f"Missing '{key}' in DPSO parameter JSON: {path}")

    params = {
        "w": float(data["w"]),
        "c1": float(data["c1"]),
        "c2": float(data["c2"]),
        "num_particles": int(round(float(data["num_particles"]))),
        "max_iter": int(round(float(data["max_iter"]))),
        "rho": float(data.get("rho", 0.5)),
        "p": float(data.get("p", 0.6)),
    }
    return params


def run_dpso_fixed_params(G, dpso_params, n_runs=20, base_seed=10000, verbose=False):
    results = []

    for r in range(n_runs):
        seed = base_seed + r
        if verbose:
            print(f"DPSO run {r + 1}/{n_runs} | seed={seed}")

        dpso = DPSOCommunity(
            G,
            num_particles=dpso_params["num_particles"],
            max_iter=dpso_params["max_iter"],
            rho=dpso_params["rho"],
            seed=seed,
        )

        communities, assignment, hard_modularity = dpso.run(
            w=dpso_params["w"],
            c1=dpso_params["c1"],
            c2=dpso_params["c2"],
            p=dpso_params["p"],
            verbose=False,
        )

        results.append({
            "run": r + 1,
            "seed": seed,
            "community_assignment": assignment,
            "hard_communities": communities,
            "num_hard_communities": len(communities),
            "hard_modularity": hard_modularity,
        })

    return results


# =====================================================
# 5. MSEFCD-inspired one-to-one fuzzy refinement from DPSO
# =====================================================

def compute_fuzzy_membership_from_DPSO_MSEFCD_full(
    G,
    dPSO_results,
    fuzzy_iterations=12,
    k_comm_candidates=3,
    prior_boost=0.10,
    min_comm_size=1,
    alpha_self=0.7,
    beta_neighbor=0.3,
    eps=1e-8,
    verbose=True,
):
    """
    Hybrid DPSO + MSEFCD-inspired fuzzy refinement.

    Each fuzzy iteration performs exactly one membership-smoothing step
    followed by exactly one DPSO-guided membership-enhancement step.

    Input dPSO_results must contain:
    - run
    - community_assignment: dict(node -> hard community label)

    Output list of dict:
    - run
    - fuzzy_membership: dict(node -> dict(label -> membership value))
    - fuzzy_labels: dict(node -> dominant label)
    - num_fuzzy_coms
    - hard_modularity
    """
    nodes = list(G.nodes())
    n = len(nodes)
    idx = {nodes[i]: i for i in range(n)}

    A = np.zeros((n, n), dtype=float)
    for u, v, data in G.edges(data=True):
        w = float(data.get("weight", 1.0))
        i, j = idx[u], idx[v]
        A[i, j] = A[j, i] = w

    deg = A.sum(axis=1)
    m = deg.sum() / 2.0
    if m <= 0:
        if verbose:
            print("Graph empty or no edges.")
        return []

    expected = np.outer(deg, deg) / (2.0 * m)
    B = A - expected

    nbr_sets = [set(G.neighbors(u)) for u in nodes]
    K = np.zeros((n, n), dtype=float)
    B_pos = np.maximum(B, 0.0)
    Bpos_norm = B_pos / (B_pos.max() + eps) if B_pos.max() > 0 else B_pos

    for i in range(n):
        for j in range(i + 1, n):
            inter = len(nbr_sets[i].intersection(nbr_sets[j]))
            union = len(nbr_sets[i].union(nbr_sets[j]))
            jacc = (inter / union) if union > 0 else 0.0
            adj_flag = 1.0 if A[i, j] > 0 else 0.0
            bscore = Bpos_norm[i, j]
            Kval = 0.45 * jacc + 0.35 * adj_flag + 0.20 * bscore
            K[i, j] = K[j, i] = Kval

    kmax = K.max() if K.max() > 0 else 1.0
    K = K / (kmax + eps)

    results_out = []

    for run_data in dPSO_results:
        run_idx = int(run_data.get("run", -1))
        best_pos_raw = run_data.get("community_assignment", None)

        if best_pos_raw is None:
            if verbose:
                print(f"Run {run_idx} missing 'community_assignment'. Skipping.")
            continue

        best_pos_list = [best_pos_raw.get(node) for node in nodes]
        if any(x is None for x in best_pos_list):
            if verbose:
                print(f"Run {run_idx} node label missing. Skipping.")
            continue

        unique_labels = sorted(set(best_pos_list), key=lambda x: str(x))
        unique_labels = [l for l in unique_labels if l is not None and l != -1]

        if len(unique_labels) < 1:
            if verbose:
                print(f"Run {run_idx} has no communities. Skipping.")
            continue

        mapping = {old: i for i, old in enumerate(unique_labels)}
        cidx_to_label = {i: old for old, i in mapping.items()}
        best_pos_compact = [mapping[x] if x in mapping else -1 for x in best_pos_list]

        comm_members = defaultdict(list)
        for i_node, lab in enumerate(best_pos_compact):
            if lab != -1:
                comm_members[lab].append(i_node)

        # Merge tiny communities into closest large communities
        small = [k for k, mem in comm_members.items() if len(mem) < min_comm_size]
        large = [k for k, mem in comm_members.items() if len(mem) >= min_comm_size]

        if len(large) == 0 and len(comm_members) > 0:
            max_size = 0
            large = []
            for k, mem in comm_members.items():
                if len(mem) > max_size:
                    max_size = len(mem)
                    large = [k]
                elif len(mem) == max_size:
                    large.append(k)
            small = [k for k in comm_members if k not in large]

        if len(large) > 0 and len(small) > 0:
            for k_small in small:
                for node_i in list(comm_members[k_small]):
                    sims = {}
                    for k_big in large:
                        members_idx = comm_members[k_big]
                        sims[k_big] = np.mean(K[node_i, members_idx]) if len(members_idx) > 0 else -1.0
                    if not sims:
                        continue
                    best_big = max(sims.items(), key=lambda x: x[1])[0]
                    comm_members[best_big].append(node_i)
                    comm_members[k_small].remove(node_i)

            comm_members = {k: v for k, v in comm_members.items() if len(v) > 0}
            new_keys = sorted(comm_members.keys())
            remap = {old_k: new_k for new_k, old_k in enumerate(new_keys)}
            new_cidx_to_label = {new_k: cidx_to_label[old_k] for old_k, new_k in remap.items()}
            cidx_to_label = new_cidx_to_label

            final_label_of_node = {}
            for old_k in new_keys:
                new_k = remap[old_k]
                for node_i in comm_members[old_k]:
                    final_label_of_node[node_i] = new_k
            best_pos_compact = [final_label_of_node.get(i, -1) for i in range(n)]

        communities_compact = sorted(list(set(l for l in best_pos_compact if l != -1)))
        c = len(communities_compact)
        if c < 1:
            if verbose:
                print(f"Run {run_idx} has <1 community after merge. Skipping.")
            continue

        label_to_cidx = {lab: k for k, lab in enumerate(communities_compact)}
        U = np.zeros((n, c), dtype=float)

        # Initial membership from neighbor community composition + prior boost
        for i_node in range(n):
            original_compact_label = best_pos_compact[i_node]
            if original_compact_label == -1:
                continue

            current_cidx = label_to_cidx.get(original_compact_label)
            if current_cidx is None:
                continue

            node_name = nodes[i_node]
            nbrs = list(G.neighbors(node_name))

            if nbrs:
                neigh_labels = [best_pos_compact[idx[n]] for n in nbrs if n in idx]
                neigh_labels = [l for l in neigh_labels if l != -1]
                if neigh_labels:
                    cnt = Counter(neigh_labels)
                    total = sum(cnt.values())
                    for lab_, cntv in cnt.items():
                        c_idx = label_to_cidx.get(lab_)
                        if c_idx is not None:
                            U[i_node, c_idx] = cntv / total

            U[i_node, current_cidx] += prior_boost
            s = U[i_node].sum()
            if s <= 0:
                U[i_node, current_cidx] = 1.0
            else:
                U[i_node] = U[i_node] / (s + eps)

        # One-to-one alternating fuzzy refinement:
        # one smoothing step followed by one enhancement step per iteration.
        community_members_idx = {
            k: [j for j, lab in enumerate(best_pos_compact) if label_to_cidx.get(lab) == k]
            for k in range(c)
        }

        for _ in range(fuzzy_iterations):
            # Step 1: membership smoothing
            U_smooth = np.zeros_like(U)
            for i_node in range(n):
                N_co = np.where(B[i_node, :] > 0)[0]
                accum = alpha_self * U[i_node].copy()
                if len(N_co) > 0:
                    accum += beta_neighbor * U[N_co].mean(axis=0)
                s_acc = accum.sum()
                if s_acc <= eps:
                    U_smooth[i_node] = U[i_node]
                else:
                    U_smooth[i_node] = accum / (s_acc + eps)

            # Step 2: DPSO-guided membership enhancement
            U_enh = np.zeros_like(U_smooth)
            for i_node in range(n):
                d_ik = np.zeros(c, dtype=float)
                for k in range(c):
                    members_idx = community_members_idx.get(k, [])
                    if len(members_idx) == 0:
                        d_ik[k] = np.inf
                    else:
                        sim = np.mean(K[i_node, members_idx])
                        d_ik[k] = 1.0 - sim

                finite_idxs = np.where(np.isfinite(d_ik))[0]
                if finite_idxs.size == 0:
                    U_enh[i_node] = U_smooth[i_node].copy()
                    continue

                ksel = finite_idxs[
                    np.argsort(d_ik[finite_idxs])[:min(k_comm_candidates, finite_idxs.size)]
                ]
                selected = ksel.tolist()
                selected_sum = U_smooth[i_node, selected].sum() if selected else 0.0

                if selected_sum <= eps:
                    U_enh[i_node] = U_smooth[i_node].copy()
                else:
                    U_enh[i_node, selected] = (
                        U_smooth[i_node, selected] / (selected_sum + eps)
                    )
                    row_sum = U_enh[i_node].sum()
                    if row_sum <= eps:
                        U_enh[i_node] = U_smooth[i_node].copy()
                    else:
                        U_enh[i_node] /= row_sum + eps

            U = U_enh

        all_original_labels = [cidx_to_label.get(k, k) for k in range(c)]
        fuzzy_membership = {}
        fuzzy_labels = {}

        for i_node, node in enumerate(nodes):
            row = U[i_node]
            s_row = row.sum()
            mem_dict = {label: 0.0 for label in all_original_labels}

            if s_row <= eps:
                orig_lab = cidx_to_label.get(best_pos_compact[i_node], None)
                if orig_lab is None:
                    largest_idx = max(range(c), key=lambda k: len(comm_members.get(k, [])))
                    orig_lab = cidx_to_label.get(largest_idx, all_original_labels[0])
                if orig_lab in mem_dict:
                    mem_dict[orig_lab] = 1.0
                    fuzzy_labels[node] = orig_lab
                else:
                    mem_dict[all_original_labels[0]] = 1.0
                    fuzzy_labels[node] = all_original_labels[0]
            else:
                normalized = row / (s_row + eps)
                for k in range(c):
                    lab = all_original_labels[k]
                    mem_dict[lab] = float(normalized[k])
                dom_k = int(np.argmax(normalized))
                fuzzy_labels[node] = all_original_labels[dom_k]

            scheck = sum(mem_dict.values())
            if scheck > 0:
                mem_dict = {lab: float(val / scheck) for lab, val in mem_dict.items()}
            fuzzy_membership[node] = mem_dict

        num_fuzzy_coms = len(set(fuzzy_labels.values()))
        results_out.append({
            "run": run_idx,
            "seed": run_data.get("seed"),
            "fuzzy_membership": fuzzy_membership,
            "fuzzy_labels": fuzzy_labels,
            "num_fuzzy_coms": num_fuzzy_coms,
            "hard_modularity": float(run_data.get("hard_modularity", 0.0)),
            "num_hard_communities": int(run_data.get("num_hard_communities", 0)),
        })

        if verbose:
            print(f"Run {run_idx:02d} selesai — komunitas fuzzy = {num_fuzzy_coms}")

    return results_out


# =====================================================
# 6. Metrics
# =====================================================

def normalize_fuzzy_membership(fm, eps=1e-12):
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


def membership_entropy(fm, eps=1e-10):
    entropies = []
    norm = normalize_fuzzy_membership(fm, eps=eps)

    for _, mem in norm.items():
        probs = np.array(list(mem.values()), dtype=float)
        probs = probs[probs > 0]
        if len(probs) <= 1:
            entropies.append(0.0)
        else:
            H = -np.sum(probs * np.log(probs))
            H_norm = H / np.log(len(probs))
            entropies.append(float(H_norm))

    return float(np.mean(entropies)) if entropies else 0.0


def fuzzy_overlapping_modularity(G, fm, eps=1e-12):
    """
    Soft fuzzy modularity compatible with overlap:
    Qf = sum_c ( e_c/m - (a_c/(2m))^2 )
    """
    if G.number_of_edges() == 0:
        return 0.0

    fm = normalize_fuzzy_membership(fm, eps=eps)
    deg = dict(G.degree(weight="weight"))
    m = float(G.size(weight="weight"))
    two_m = 2.0 * m

    if m <= 0:
        return 0.0

    comms = set()
    for mem in fm.values():
        comms.update(mem.keys())
    if not comms:
        return 0.0

    a = defaultdict(float)
    for i, mem in fm.items():
        ki = float(deg.get(i, 0.0))
        for c, u in mem.items():
            a[c] += ki * float(u)

    e = defaultdict(float)
    for i, j, data in G.edges(data=True):
        wij = float(data.get("weight", 1.0))
        mi = fm.get(i, {})
        mj = fm.get(j, {})
        if len(mi) > len(mj):
            mi, mj = mj, mi
        for c, uic in mi.items():
            ujc = mj.get(c, 0.0)
            if ujc:
                e[c] += wij * float(uic) * float(ujc)

    Qf = 0.0
    for c in comms:
        ec = e.get(c, 0.0)
        ac = a.get(c, 0.0)
        Qf += (ec / m) - (ac / two_m) ** 2

    return float(Qf)


def fuzzy_to_communities(fm, alpha=0.4, min_size=2):
    if not fm:
        return []

    all_labels = sorted({lab for mem in fm.values() for lab in mem.keys()}, key=lambda x: str(x))
    communities = {lab: [] for lab in all_labels}

    for node, mem in fm.items():
        assigned = [lab for lab, val in mem.items() if float(val) >= alpha]
        if assigned:
            for lab in assigned:
                communities[lab].append(node)
        else:
            best_lab = max(mem.items(), key=lambda x: x[1])[0]
            communities[best_lab].append(node)

    return [nodes for nodes in communities.values() if len(nodes) >= min_size]


def overlapping_nodes_from_membership(fm, alpha=0.4):
    rows = []
    for node, mem in fm.items():
        labels_above = {str(lab): float(val) for lab, val in mem.items() if float(val) >= alpha}
        if len(labels_above) > 1:
            rows.append({
                "node": str(node),
                "num_memberships": len(labels_above),
                "memberships": json.dumps(labels_above, ensure_ascii=False),
            })
    return rows


def dominant_labels_from_membership(G, fm):
    labels = []
    for node in G.nodes():
        mem = fm.get(node, {})
        if not mem:
            labels.append(str(node))
        else:
            best_lab = max(mem.items(), key=lambda x: x[1])[0]
            labels.append(str(best_lab))
    return labels


def compute_stability_per_run(G, fuzzy_results, alpha_threshold=0.4):
    n = len(fuzzy_results)
    if n <= 1:
        return [1.0]

    # Prefer ONMI between overlapping communities if CDlib is available.
    if HAS_CDLIB:
        clusterings = []
        valid_idx = []
        for idx, run_data in enumerate(fuzzy_results):
            try:
                comms = fuzzy_to_communities(run_data["fuzzy_membership"], alpha=alpha_threshold)
                if len(comms) >= 2:
                    clusterings.append(NodeClustering(comms, G, f"run_{run_data.get('run', idx)}"))
                    valid_idx.append(idx)
            except Exception:
                continue

        stability_vals = [[] for _ in range(n)]
        if len(clusterings) >= 2:
            for local_i, local_j in combinations(range(len(clusterings)), 2):
                idx_i = valid_idx[local_i]
                idx_j = valid_idx[local_j]
                try:
                    score = evaluation.overlapping_normalized_mutual_information_LFK(
                        clusterings[local_i], clusterings[local_j]
                    ).score
                    if not np.isnan(score):
                        stability_vals[idx_i].append(float(score))
                        stability_vals[idx_j].append(float(score))
                except Exception:
                    continue

            if any(stability_vals):
                return [float(np.mean(vals)) if vals else 0.0 for vals in stability_vals]

    # Fallback: NMI of dominant labels.
    hard_labels = [dominant_labels_from_membership(G, r["fuzzy_membership"]) for r in fuzzy_results]
    stability_vals = [[] for _ in range(n)]

    for i, j in combinations(range(n), 2):
        score = normalized_mutual_info_score(hard_labels[i], hard_labels[j])
        stability_vals[i].append(float(score))
        stability_vals[j].append(float(score))

    return [float(np.mean(vals)) if vals else 0.0 for vals in stability_vals]


def add_composite_scores(df):
    out = df.copy()
    out["M_norm"] = minmax(out["fuzzy_modularity"].values)
    out["S_norm"] = minmax(out["stability"].values)
    out["E_norm"] = minmax(out["entropy"].values)
    out["entropy_score"] = 1.0 - out["E_norm"]
    out["composite"] = (out["M_norm"] + out["S_norm"] + out["entropy_score"]) / 3.0
    return out


def evaluate_fuzzy_results(G, fuzzy_results, alpha_threshold=0.4, dpso_params=None, fuzzy_params=None):
    stabilities = compute_stability_per_run(G, fuzzy_results, alpha_threshold=alpha_threshold)

    rows = []
    for idx, run_data in enumerate(fuzzy_results):
        fm = run_data["fuzzy_membership"]
        overlap_rows = overlapping_nodes_from_membership(fm, alpha=alpha_threshold)
        comms = fuzzy_to_communities(fm, alpha=alpha_threshold)

        row = {
            "run": int(run_data.get("run", idx + 1)),
            "seed": int(run_data.get("seed")) if run_data.get("seed") is not None else None,
            "num_hard_communities": int(run_data.get("num_hard_communities", 0)),
            "num_fuzzy_communities": int(run_data.get("num_fuzzy_coms", len(comms))),
            "num_pred_communities_alpha": int(len(comms)),
            "num_overlapping_nodes": int(len(overlap_rows)),
            "hard_modularity": float(run_data.get("hard_modularity", 0.0)),
            "fuzzy_modularity": fuzzy_overlapping_modularity(G, fm),
            "stability": float(stabilities[idx]),
            "entropy": membership_entropy(fm),
        }

        if dpso_params:
            row.update({
                "w": dpso_params["w"],
                "c1": dpso_params["c1"],
                "c2": dpso_params["c2"],
                "num_particles": dpso_params["num_particles"],
                "max_iter": dpso_params["max_iter"],
                "rho": dpso_params["rho"],
                "p": dpso_params["p"],
            })

        if fuzzy_params:
            row.update({
                "alpha_self": fuzzy_params["alpha_self"],
                "beta_neighbor": fuzzy_params["beta_neighbor"],
                "prior_boost": fuzzy_params["prior_boost"],
                "k_comm_candidates": fuzzy_params["k_comm_candidates"],
                "fuzzy_iterations": fuzzy_params["fuzzy_iterations"],
                "min_comm_size": fuzzy_params["min_comm_size"],
                "alpha_threshold": alpha_threshold,
            })

        rows.append(row)

    df = pd.DataFrame(rows)
    df = add_composite_scores(df)
    df = df.sort_values("composite", ascending=False).reset_index(drop=True)
    df["rank"] = df.index + 1
    return df


# =====================================================
# 7. Fuzzy refinement tuning
# =====================================================

def summarize_config_metrics(per_run_df):
    return {
        "mean_num_hard_communities": float(per_run_df["num_hard_communities"].mean()),
        "std_num_hard_communities": float(per_run_df["num_hard_communities"].std(ddof=0)),
        "mean_num_fuzzy_communities": float(per_run_df["num_fuzzy_communities"].mean()),
        "std_num_fuzzy_communities": float(per_run_df["num_fuzzy_communities"].std(ddof=0)),
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
        "mean_composite_per_run": float(per_run_df["composite"].mean()),
        "std_composite_per_run": float(per_run_df["composite"].std(ddof=0)),
    }


def add_selection_composite_to_tuning(df):
    out = df.copy()
    out["M_norm"] = minmax(out["mean_fuzzy_modularity"].values)
    out["S_norm"] = minmax(out["mean_stability"].values)
    out["E_norm"] = minmax(out["mean_entropy"].values)
    out["entropy_score"] = 1.0 - out["E_norm"]
    out["selection_composite"] = (out["M_norm"] + out["S_norm"] + out["entropy_score"]) / 3.0
    out = out.sort_values(
        ["selection_composite", "mean_fuzzy_modularity", "mean_stability", "mean_entropy"],
        ascending=[False, False, False, True],
    ).reset_index(drop=True)
    out["rank"] = out.index + 1
    return out


def tune_fuzzy_params_unsupervised(
    G,
    dpso_results,
    dpso_params,
    alpha_beta_grid,
    prior_boost_list,
    k_comm_candidates_list,
    alpha_threshold=0.4,
    fuzzy_iterations=12,
    min_comm_size=1,
    verbose=True,
):
    grid = list(product(alpha_beta_grid, prior_boost_list, k_comm_candidates_list))
    rows = []

    print(f"Total fuzzy refinement configurations: {len(grid)}")
    print(f"Fixed alpha_threshold from benchmark: {alpha_threshold}")

    for config_id, ((alpha_self, beta_neighbor), prior_boost, k_comm) in enumerate(grid, start=1):
        if verbose:
            print(
                f"[{config_id}/{len(grid)}] "
                f"alpha_self={alpha_self}, beta_neighbor={beta_neighbor}, "
                f"prior_boost={prior_boost}, k_comm_candidates={k_comm}"
            )

        fuzzy_params = {
            "alpha_self": float(alpha_self),
            "beta_neighbor": float(beta_neighbor),
            "prior_boost": float(prior_boost),
            "k_comm_candidates": int(k_comm),
            "fuzzy_iterations": int(fuzzy_iterations),
            "min_comm_size": int(min_comm_size),
        }

        fuzzy_results = compute_fuzzy_membership_from_DPSO_MSEFCD_full(
            G,
            dpso_results,
            fuzzy_iterations=fuzzy_iterations,
            k_comm_candidates=int(k_comm),
            prior_boost=float(prior_boost),
            min_comm_size=int(min_comm_size),
            alpha_self=float(alpha_self),
            beta_neighbor=float(beta_neighbor),
            verbose=False,
        )

        per_run_df = evaluate_fuzzy_results(
            G,
            fuzzy_results,
            alpha_threshold=alpha_threshold,
            dpso_params=dpso_params,
            fuzzy_params=fuzzy_params,
        )

        summary = summarize_config_metrics(per_run_df)
        row = {
            "config_id": config_id,
            "alpha_self": float(alpha_self),
            "beta_neighbor": float(beta_neighbor),
            "prior_boost": float(prior_boost),
            "k_comm_candidates": int(k_comm),
            "fuzzy_iterations": int(fuzzy_iterations),
            "min_comm_size": int(min_comm_size),
            "alpha_threshold_fixed": float(alpha_threshold),
        }
        row.update(summary)
        rows.append(row)

    tuning_df = pd.DataFrame(rows)
    tuning_df = add_selection_composite_to_tuning(tuning_df)
    best_row = tuning_df.iloc[0].to_dict()

    return tuning_df, best_row


# =====================================================
# 8. Save final outputs
# =====================================================
def save_all_run_details(fuzzy_results, outdir, prefix, alpha_threshold):
    """
    Save detailed DPSO-Fuzzy outputs for every final run.

    This file is needed for significant comorbidity analysis because
    centrality must be calculated inside communities for each run.

    Output:
    - <prefix>_dpso_fuzzy_run_details/run_<run>_seed_<seed>.json
    - <prefix>_dpso_fuzzy_20runs_details_index.csv
    """

    details_dir = os.path.join(outdir, f"{prefix}_dpso_fuzzy_run_details")
    os.makedirs(details_dir, exist_ok=True)

    index_rows = []

    for run_data in fuzzy_results:
        run_id = int(run_data.get("run"))
        seed = run_data.get("seed")

        fm = run_data["fuzzy_membership"]

        # Membership per node
        memberships_json = {
            str(node): {
                str(label): float(mu)
                for label, mu in mem.items()
            }
            for node, mem in fm.items()
        }

        # Communities after alpha threshold
        comms = fuzzy_to_communities(
            fm,
            alpha=alpha_threshold,
            min_size=1
        )

        communities_json = {
            str(i + 1): [str(node) for node in nodes]
            for i, nodes in enumerate(comms)
        }

        # Overlapping nodes after alpha threshold
        overlap_rows = overlapping_nodes_from_membership(
            fm,
            alpha=alpha_threshold
        )

        overlapping_nodes = [
            str(row["node"])
            for row in overlap_rows
        ]

        detail = {
            "run": run_id,
            "seed": seed,
            "alpha_threshold": float(alpha_threshold),
            "num_communities": int(len(communities_json)),
            "num_overlapping_nodes": int(len(overlapping_nodes)),
            "communities": communities_json,
            "fuzzy_membership": memberships_json,
            "overlapping_nodes": overlapping_nodes,
            "overlapping_detail": overlap_rows
        }

        detail_path = os.path.join(
            details_dir,
            f"run_{run_id}_seed_{seed}.json"
        )

        save_json(detail, detail_path)

        index_rows.append({
            "run": run_id,
            "seed": seed,
            "num_communities": int(len(communities_json)),
            "num_overlapping_nodes": int(len(overlapping_nodes)),
            "detail_file": detail_path
        })

    index_df = pd.DataFrame(index_rows)
    index_path = os.path.join(
        outdir,
        f"{prefix}_dpso_fuzzy_20runs_details_index.csv"
    )
    index_df.to_csv(index_path, index=False)

    return details_dir, index_path

def save_best_outputs(best_fuzzy_run, outdir, prefix, alpha_threshold):
    fm = best_fuzzy_run["fuzzy_membership"]

    # Best memberships
    memberships_json = {
        str(node): {str(label): float(mu) for label, mu in mem.items()}
        for node, mem in fm.items()
    }
    save_json(memberships_json, os.path.join(outdir, f"{prefix}_best_memberships.json"))

    # Best communities based on fixed alpha threshold
    comms = fuzzy_to_communities(fm, alpha=alpha_threshold, min_size=1)
    communities_json = {
        str(i + 1): [str(node) for node in nodes]
        for i, nodes in enumerate(comms)
    }
    save_json(communities_json, os.path.join(outdir, f"{prefix}_best_communities.json"))

    # Overlapping nodes
    overlap_rows = overlapping_nodes_from_membership(fm, alpha=alpha_threshold)
    pd.DataFrame(overlap_rows).to_csv(
        os.path.join(outdir, f"{prefix}_overlapping_nodes.csv"),
        index=False,
    )


def save_20runs_summary(per_run_df, outdir, prefix):
    summary_cols = [
        "num_hard_communities",
        "num_fuzzy_communities",
        "num_pred_communities_alpha",
        "num_overlapping_nodes",
        "hard_modularity",
        "fuzzy_modularity",
        "stability",
        "entropy",
        "composite",
    ]

    summary_rows = []
    for col in summary_cols:
        summary_rows.append({
            "metric": col,
            "mean": float(per_run_df[col].mean()),
            "std": float(per_run_df[col].std(ddof=0)),
            "min": float(per_run_df[col].min()),
            "max": float(per_run_df[col].max()),
        })

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(os.path.join(outdir, f"{prefix}_dpso_fuzzy_20runs_summary.csv"), index=False)

    compact_summary = {}
    for col in summary_cols:
        compact_summary[f"mean_{col}"] = float(per_run_df[col].mean())
        compact_summary[f"std_{col}"] = float(per_run_df[col].std(ddof=0))

    save_json(compact_summary, os.path.join(outdir, f"{prefix}_dpso_fuzzy_20runs_compact_summary.json"))

    return summary_df, compact_summary


# =====================================================
# 9. Main
# =====================================================

def main():
    parser = argparse.ArgumentParser(description="DPSO-Fuzzy with fixed DPSO params and unsupervised fuzzy refinement tuning")

    parser.add_argument("--graph", required=True, help="Path to disease network CSV")
    parser.add_argument("--name", default="network", help="Network name, e.g., NSCLC or SCLC")
    parser.add_argument("--dpso-params", required=True, help="Path to PSO-LPA best_run_summary.json")
    parser.add_argument("--outdir", default="results_dpso_fuzzy", help="Output directory")

    parser.add_argument("--alpha-threshold", type=float, default=0.4, help="Fixed alpha threshold from benchmark")

    parser.add_argument(
        "--alpha-beta-grid",
        default="0.8:0.2,0.7:0.3,0.6:0.4,0.5:0.5",
        help="Grid for alpha_self:beta_neighbor pairs",
    )
    parser.add_argument("--prior-boosts", default="0.05,0.08,0.10,0.15", help="Comma-separated prior_boost values")
    parser.add_argument("--k-comm-candidates", default="2,3,4", help="Comma-separated k_comm_candidates values")

    parser.add_argument(
        "--fuzzy-iterations",
        "--iter-smooth",
        dest="fuzzy_iterations",
        type=int,
        default=12,
        help=(
            "Number of alternating fuzzy iterations; each iteration performs "
            "one smoothing step and one enhancement step"
        ),
    )
    parser.add_argument("--min-comm-size", type=int, default=1, help="Minimum community size during fuzzy refinement")

    parser.add_argument("--tune-runs", type=int, default=5, help="DPSO runs for fuzzy parameter tuning")
    parser.add_argument("--final-runs", type=int, default=20, help="Final DPSO-Fuzzy runs using best fuzzy parameters")
    parser.add_argument("--seed", type=int, default=1234, help="Base random seed")

    parser.add_argument("--source-col", default=None, help="Optional source column")
    parser.add_argument("--target-col", default=None, help="Optional target column")
    parser.add_argument("--weight-col", default=None, help="Optional weight column")

    parser.add_argument("--verbose-dpso", action="store_true", help="Print DPSO progress")

    args = parser.parse_args()

    if args.fuzzy_iterations < 1:
        parser.error("--fuzzy-iterations must be at least 1")

    os.makedirs(args.outdir, exist_ok=True)
    prefix = args.name.lower()

    print("=" * 80)
    print(f"Network                  : {args.name}")
    print(f"Graph path               : {args.graph}")
    print(f"DPSO parameter JSON      : {args.dpso_params}")
    print(f"Output dir               : {args.outdir}")
    print(f"Fixed alpha threshold    : {args.alpha_threshold}")
    print(f"Fuzzy iterations         : {args.fuzzy_iterations} (1 smoothing + 1 enhancement each)")
    print("=" * 80)

    G = load_graph_from_csv(
        args.graph,
        source_col=args.source_col,
        target_col=args.target_col,
        weight_col=args.weight_col,
    )
    print(f"original graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    largest_nodes = max(
    nx.connected_components(G),
    key=len
    )

    G = G.subgraph(largest_nodes).copy()

    print(
       f"Largest connected component: "
       f"{G.number_of_nodes()} nodes, "
       f"{G.number_of_edges()} edges"
    )
    
    dpso_params = load_dpso_params_from_summary(args.dpso_params)
    print("\nLoaded fixed DPSO parameters:")
    print(json.dumps(dpso_params, indent=2))
    save_json(dpso_params, os.path.join(args.outdir, f"{prefix}_fixed_dpso_params.json"))

    alpha_beta_grid = parse_alpha_beta_grid(args.alpha_beta_grid)
    prior_boost_list = parse_float_list(args.prior_boosts)
    k_comm_candidates_list = parse_int_list(args.k_comm_candidates)

    # -------------------------------------------------
    # Tuning fuzzy refinement parameters
    # -------------------------------------------------
    print("\n" + "=" * 80)
    print("Step 1 - Run fixed DPSO parameters for fuzzy tuning")
    print("=" * 80)

    dpso_tuning_results = run_dpso_fixed_params(
        G,
        dpso_params=dpso_params,
        n_runs=args.tune_runs,
        base_seed=args.seed,
        verbose=True,
    )

    print("\n" + "=" * 80)
    print("Step 2 - Tune fuzzy refinement parameters unsupervised")
    print("=" * 80)

    tuning_df, best_fuzzy_config = tune_fuzzy_params_unsupervised(
        G,
        dpso_results=dpso_tuning_results,
        dpso_params=dpso_params,
        alpha_beta_grid=alpha_beta_grid,
        prior_boost_list=prior_boost_list,
        k_comm_candidates_list=k_comm_candidates_list,
        alpha_threshold=args.alpha_threshold,
        fuzzy_iterations=args.fuzzy_iterations,
        min_comm_size=args.min_comm_size,
        verbose=True,
    )

    tuning_path = os.path.join(args.outdir, f"{prefix}_dpso_fuzzy_tuning.csv")
    tuning_df.to_csv(tuning_path, index=False)

    best_config_path = os.path.join(args.outdir, f"{prefix}_dpso_fuzzy_best_config.json")
    save_json(best_fuzzy_config, best_config_path)

    print("\nBest fuzzy refinement configuration:")
    print(json.dumps(best_fuzzy_config, indent=2))

    # -------------------------------------------------
    # Final 20 runs
    # -------------------------------------------------
    print("\n" + "=" * 80)
    print("Step 3 - Final DPSO-Fuzzy runs using fixed DPSO and best fuzzy parameters")
    print("=" * 80)

    final_dpso_results = run_dpso_fixed_params(
        G,
        dpso_params=dpso_params,
        n_runs=args.final_runs,
        base_seed=args.seed + 10000,
        verbose=True,
    )

    final_fuzzy_params = {
        "alpha_self": float(best_fuzzy_config["alpha_self"]),
        "beta_neighbor": float(best_fuzzy_config["beta_neighbor"]),
        "prior_boost": float(best_fuzzy_config["prior_boost"]),
        "k_comm_candidates": int(best_fuzzy_config["k_comm_candidates"]),
        "fuzzy_iterations": int(best_fuzzy_config["fuzzy_iterations"]),
        "min_comm_size": int(best_fuzzy_config["min_comm_size"]),
    }

    final_fuzzy_results = compute_fuzzy_membership_from_DPSO_MSEFCD_full(
        G,
        final_dpso_results,
        fuzzy_iterations=final_fuzzy_params["fuzzy_iterations"],
        k_comm_candidates=final_fuzzy_params["k_comm_candidates"],
        prior_boost=final_fuzzy_params["prior_boost"],
        min_comm_size=final_fuzzy_params["min_comm_size"],
        alpha_self=final_fuzzy_params["alpha_self"],
        beta_neighbor=final_fuzzy_params["beta_neighbor"],
        verbose=False,
    )

    final_df = evaluate_fuzzy_results(
        G,
        final_fuzzy_results,
        alpha_threshold=args.alpha_threshold,
        dpso_params=dpso_params,
        fuzzy_params=final_fuzzy_params,
    )

    final_path = os.path.join(args.outdir, f"{prefix}_dpso_fuzzy_20runs.csv")
    final_df.to_csv(final_path, index=False)
    
    details_dir, details_index_path = save_all_run_details(
        final_fuzzy_results,
        args.outdir,
        prefix,
        alpha_threshold=args.alpha_threshold,
    )

    summary_df, compact_summary = save_20runs_summary(final_df, args.outdir, prefix)

    # Best run output
    best_run_number = int(final_df.iloc[0]["run"])
    best_fuzzy_run = next(r for r in final_fuzzy_results if int(r["run"]) == best_run_number)

    save_best_outputs(best_fuzzy_run, args.outdir, prefix, alpha_threshold=args.alpha_threshold)

    best_run_summary = final_df.iloc[0].to_dict()
    save_json(best_run_summary, os.path.join(args.outdir, f"{prefix}_best_run_summary.json"))

    print("\nSaved files:")
    print(f"- {tuning_path}")
    print(f"- {best_config_path}")
    print(f"- {final_path}")
    print(f"- {os.path.join(args.outdir, f'{prefix}_dpso_fuzzy_20runs_summary.csv')}")
    print(f"- {os.path.join(args.outdir, f'{prefix}_dpso_fuzzy_20runs_compact_summary.json')}")
    print(f"- {os.path.join(args.outdir, f'{prefix}_best_run_summary.json')}")
    print(f"- {os.path.join(args.outdir, f'{prefix}_best_memberships.json')}")
    print(f"- {os.path.join(args.outdir, f'{prefix}_best_communities.json')}")
    print(f"- {os.path.join(args.outdir, f'{prefix}_overlapping_nodes.csv')}")

    print("\n20-run summary:")
    print(summary_df.to_string(index=False))

    print("\nBest final run summary:")
    print(json.dumps(best_run_summary, indent=2))


if __name__ == "__main__":
    main()
