"""Tune all comparison algorithms except PSO-LPA.

The output JSON is read by ``allmethodyeast.py``. The script supports all five
benchmark datasets and does not execute expensive tuning when imported.
"""

from __future__ import annotations

import argparse
import itertools
import math
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import networkx as nx
import numpy as np
import pandas as pd

from benchmark_common import (
    load_graph,
    resolve_dataset_paths,
    safe_modularity,
    save_json,
    seed_everything,
)

MethodOutput = Tuple[Dict[Any, List[Any]], Dict[Any, Dict[Any, float]]]


def _normalize_membership(values: Mapping[Any, float]) -> Dict[Any, float]:
    total = float(sum(max(0.0, float(v)) for v in values.values()))
    if total <= 0:
        return {}
    return {label: max(0.0, float(value)) / total for label, value in values.items()}


def _memberships_to_communities(
    memberships: Mapping[Any, Mapping[Any, float]],
    threshold: float,
) -> Dict[Any, List[Any]]:
    communities: Dict[Any, List[Any]] = defaultdict(list)
    for node, distribution in memberships.items():
        if not distribution:
            continue
        selected = {
            label: value for label, value in distribution.items() if value >= threshold
        }
        if not selected:
            label = max(distribution, key=distribution.get)
            selected = {label: distribution[label]}
        for label in selected:
            communities[label].append(node)
    return dict(communities)


def fuzzy_bldlp(
    G: nx.Graph,
    alpha: float = 0.5,
    lambd: float = 2.0,
    threshold: float = 0.3,
    max_iter: int = 100,
    seed: int = 123,
) -> MethodOutput:
    rng = random.Random(seed)
    nodes = list(G.nodes())
    edge_weights: Dict[Tuple[Any, Any], float] = {}
    alpha = min(1.0, max(0.0, float(alpha)))

    for u, v in G.edges():
        deg_u, deg_v = G.degree[u], G.degree[v]
        structural = alpha / (deg_u + 1.0) + (1.0 - alpha) / (deg_v + 1.0)
        graph_weight = float(G.get_edge_data(u, v, default={}).get("weight", 1.0))
        edge_weights[(u, v)] = structural * graph_weight
        edge_weights[(v, u)] = structural * graph_weight

    labels = {node: node for node in nodes}
    for _ in range(int(max_iter)):
        changed = False
        order = nodes.copy()
        rng.shuffle(order)
        for node in order:
            scores: Dict[Any, float] = defaultdict(float)
            for neighbor in G.neighbors(node):
                scores[labels[neighbor]] += edge_weights.get((node, neighbor), 0.0)
            if not scores:
                continue
            maximum = max(scores.values())
            candidates = [label for label, score in scores.items() if abs(score - maximum) <= 1e-12]
            new_label = rng.choice(candidates)
            if new_label != labels[node]:
                labels[node] = new_label
                changed = True
        if not changed:
            break

    memberships: Dict[Any, Dict[Any, float]] = {}
    for node in nodes:
        label_weights: Dict[Any, float] = defaultdict(float)
        for neighbor in G.neighbors(node):
            label_weights[labels[neighbor]] += edge_weights.get((node, neighbor), 0.0)
        total = sum(label_weights.values())
        raw: Dict[Any, float] = {}
        if total > 0:
            for label, score in label_weights.items():
                gamma = score / total
                mu = 1.0 - math.exp(-float(lambd) * gamma)
                if mu >= threshold:
                    raw[label] = mu
        if not raw:
            raw[labels[node]] = 1.0
        memberships[node] = _normalize_membership(raw)

    return _memberships_to_communities(memberships, threshold=0.0), memberships


def fuzzy_lpa(
    G: nx.Graph,
    max_iter: int = 100,
    threshold: float = 0.1,
    lambd: float = 2.0,
    seed: int = 123,
) -> MethodOutput:
    rng = random.Random(seed)
    nodes = list(G.nodes())
    labels = {node: node for node in nodes}

    for _ in range(int(max_iter)):
        changes = 0
        order = nodes.copy()
        rng.shuffle(order)
        for node in order:
            counts = Counter(labels[neighbor] for neighbor in G.neighbors(node))
            if not counts:
                continue
            maximum = max(counts.values())
            candidates = [label for label, count in counts.items() if count == maximum]
            new_label = rng.choice(candidates)
            if labels[node] != new_label:
                labels[node] = new_label
                changes += 1
        if changes == 0:
            break

    memberships: Dict[Any, Dict[Any, float]] = {}
    for node in nodes:
        counts = Counter(labels[neighbor] for neighbor in G.neighbors(node))
        total = sum(counts.values())
        raw: Dict[Any, float] = {}
        if total > 0:
            for label, count in counts.items():
                gamma = count / total
                mu = 1.0 - math.exp(-float(lambd) * gamma)
                if mu >= threshold:
                    raw[label] = mu
        if not raw:
            raw[labels[node]] = 1.0
        memberships[node] = _normalize_membership(raw)
    return _memberships_to_communities(memberships, threshold=0.0), memberships


def compute_link_strengths(G: nx.Graph) -> Dict[Any, Dict[Any, float]]:
    strengths: Dict[Any, Dict[Any, float]] = {}
    neighbor_sets = {node: set(G.neighbors(node)) for node in G.nodes()}
    for u in G.nodes():
        raw: Dict[Any, float] = {}
        for v in G.neighbors(u):
            common = len(neighbor_sets[u] & neighbor_sets[v])
            union = len(neighbor_sets[u] | neighbor_sets[v])
            raw[v] = (common + 1.0) / (union + 1.0)
        strengths[u] = _normalize_membership(raw)
    return strengths


def fuzzy_ldpa(
    G: nx.Graph,
    max_iter: int = 100,
    threshold: float = 0.2,
    lambd: float = 2.0,
    merge_threshold: float = 0.3,
    seed: int = 123,
) -> MethodOutput:
    rng = random.Random(seed)
    nodes = list(G.nodes())
    link_strengths = compute_link_strengths(G)
    labels = {node: node for node in nodes}

    for _ in range(int(max_iter)):
        changes = 0
        order = nodes.copy()
        rng.shuffle(order)
        for node in order:
            scores: Dict[Any, float] = defaultdict(float)
            for neighbor in G.neighbors(node):
                scores[labels[neighbor]] += link_strengths[node].get(neighbor, 0.0)
            if not scores:
                continue
            maximum = max(scores.values())
            candidates = [label for label, score in scores.items() if abs(score - maximum) <= 1e-12]
            new_label = rng.choice(candidates)
            if labels[node] != new_label:
                labels[node] = new_label
                changes += 1
        if changes == 0:
            break

    memberships: Dict[Any, Dict[Any, float]] = {}
    for node in nodes:
        scores: Dict[Any, float] = defaultdict(float)
        for neighbor in G.neighbors(node):
            scores[labels[neighbor]] += link_strengths[node].get(neighbor, 0.0)
        total = sum(scores.values())
        raw: Dict[Any, float] = {}
        if total > 0:
            for label, score in scores.items():
                gamma = score / total
                mu = 1.0 - math.exp(-float(lambd) * gamma)
                if mu >= threshold:
                    raw[label] = mu
        if not raw:
            raw[labels[node]] = 1.0
        memberships[node] = _normalize_membership(raw)

    communities = _memberships_to_communities(memberships, threshold=0.0)
    communities = merge_similar_communities(communities, merge_threshold)
    # Rebuild membership labels after merging because old labels no longer match.
    rebuilt: Dict[Any, Dict[int, float]] = defaultdict(dict)
    for community_id, members in communities.items():
        for node in members:
            rebuilt[node][community_id] = 1.0
    for node in rebuilt:
        rebuilt[node] = _normalize_membership(rebuilt[node])
    return communities, dict(rebuilt)


def merge_similar_communities(
    communities: Mapping[Any, Iterable[Any]], threshold: float = 0.3
) -> Dict[int, List[Any]]:
    groups = [set(members) for members in communities.values() if members]
    merged: List[set] = []
    while groups:
        current = groups.pop(0)
        changed = True
        while changed:
            changed = False
            remaining = []
            for other in groups:
                union = current | other
                jaccard = len(current & other) / len(union) if union else 0.0
                if jaccard >= threshold:
                    current |= other
                    changed = True
                else:
                    remaining.append(other)
            groups = remaining
        merged.append(current)
    return {idx: sorted(group, key=str) for idx, group in enumerate(merged) if group}


def cfinder(G: nx.Graph, k: int = 3, seed: int = 123) -> MethodOutput:
    del seed
    k = max(2, int(k))
    cliques = [set(clique) for clique in nx.find_cliques(G) if len(clique) >= k]
    clique_graph = nx.Graph()
    clique_graph.add_nodes_from(range(len(cliques)))
    for i in range(len(cliques)):
        for j in range(i + 1, len(cliques)):
            if len(cliques[i] & cliques[j]) >= k - 1:
                clique_graph.add_edge(i, j)

    communities: Dict[int, List[Any]] = {}
    for component in nx.connected_components(clique_graph):
        members = set().union(*(cliques[index] for index in component))
        if members:
            communities[len(communities)] = sorted(members, key=str)

    memberships: Dict[Any, Dict[int, float]] = defaultdict(dict)
    for label, members in communities.items():
        for node in members:
            memberships[node][label] = 1.0
    return communities, dict(memberships)


def lfk(G: nx.Graph, alpha: float = 1.0, seed: int = 123) -> MethodOutput:
    rng = random.Random(seed)
    nodes = list(G.nodes())

    def fitness(community: set) -> float:
        internal_twice = 0.0
        external = 0.0
        for u in community:
            for v in G.neighbors(u):
                weight = float(G.get_edge_data(u, v, default={}).get("weight", 1.0))
                if v in community:
                    internal_twice += weight
                else:
                    external += weight
        internal = internal_twice / 2.0
        total = internal + external
        return 0.0 if total <= 0 else internal / (total ** float(alpha))

    communities: List[set] = []
    order = nodes.copy()
    rng.shuffle(order)
    for seed_node in order:
        community = {seed_node}
        while True:
            boundary = set().union(*(set(G.neighbors(node)) for node in community)) - community
            current_fitness = fitness(community)
            best_node = None
            best_gain = 0.0
            for candidate in boundary:
                gain = fitness(community | {candidate}) - current_fitness
                if gain > best_gain + 1e-12:
                    best_gain = gain
                    best_node = candidate
            if best_node is None:
                break
            community.add(best_node)

            # Prune members whose removal improves fitness.
            pruned = True
            while pruned and len(community) > 1:
                pruned = False
                current_fitness = fitness(community)
                for member in list(community):
                    candidate = community - {member}
                    if candidate and fitness(candidate) > current_fitness + 1e-12:
                        community = candidate
                        pruned = True
                        break

        if len(community) > 1 and not any(community <= existing for existing in communities):
            communities = [existing for existing in communities if not existing < community]
            communities.append(community)

    output = {idx: sorted(comm, key=str) for idx, comm in enumerate(communities)}
    memberships: Dict[Any, Dict[int, float]] = defaultdict(dict)
    for label, members in output.items():
        for node in members:
            memberships[node][label] = 1.0
    return output, dict(memberships)


def _structural_features(G: nx.Graph, nodes: Sequence[Any]) -> np.ndarray:
    n = len(nodes)
    degree = np.array([G.degree[node] for node in nodes], dtype=float)
    clustering_map = nx.clustering(G)
    clustering = np.array([clustering_map[node] for node in nodes], dtype=float)
    avg_neighbor_map = nx.average_neighbor_degree(G)
    avg_neighbor = np.array([avg_neighbor_map[node] for node in nodes], dtype=float)
    core_map = nx.core_number(G) if G.number_of_edges() else {node: 0 for node in nodes}
    core = np.array([core_map[node] for node in nodes], dtype=float)
    pagerank_map = nx.pagerank(G, alpha=0.85, max_iter=200)
    pagerank = np.array([pagerank_map[node] for node in nodes], dtype=float)
    triangles_map = nx.triangles(G)
    triangles = np.array([triangles_map[node] for node in nodes], dtype=float)

    features = np.column_stack([degree, clustering, avg_neighbor, core, pagerank, triangles])
    for column in range(features.shape[1]):
        minimum = features[:, column].min()
        maximum = features[:, column].max()
        if maximum > minimum:
            features[:, column] = (features[:, column] - minimum) / (maximum - minimum)
        else:
            features[:, column] = 0.0
    return features


def hybrid_cmeans(
    G: nx.Graph,
    K_param: Optional[int] = None,
    m: float = 2.0,
    epsilon: float = 1e-5,
    max_iter: int = 100,
    lower_thr: float = 0.6,
    upper_thr: float = 0.3,
    seed: int = 123,
) -> MethodOutput:
    nodes = list(G.nodes())
    n = len(nodes)
    if n == 0:
        return {}, {}
    K = int(K_param) if K_param is not None else max(2, int(math.sqrt(n)))
    K = max(1, min(K, n))
    m = max(float(m), 1.01)
    rng = np.random.default_rng(seed)
    X = _structural_features(G, nodes)

    initial_indices = rng.choice(n, size=K, replace=False)
    centroids = X[initial_indices].copy()
    U = np.full((n, K), 1.0 / K, dtype=float)

    for _ in range(int(max_iter)):
        distances = np.linalg.norm(X[:, None, :] - centroids[None, :, :], axis=2)
        distances = np.maximum(distances, 1e-12)
        zero_rows = np.where(np.min(distances, axis=1) <= 1e-11)[0]
        U_new = np.zeros_like(U)
        exponent = 2.0 / (m - 1.0)
        for i in range(n):
            zero_clusters = np.where(distances[i] <= 1e-11)[0]
            if zero_clusters.size:
                U_new[i, zero_clusters] = 1.0 / zero_clusters.size
            else:
                ratios = (distances[i, :, None] / distances[i, None, :]) ** exponent
                U_new[i] = 1.0 / np.sum(ratios, axis=1)

        powered = U_new ** m
        denominator = powered.sum(axis=0)[:, None]
        empty = np.where(denominator[:, 0] <= 1e-12)[0]
        denominator = np.maximum(denominator, 1e-12)
        centroids_new = (powered.T @ X) / denominator
        for cluster in empty:
            centroids_new[cluster] = X[rng.integers(0, n)]

        if np.linalg.norm(U_new - U) < float(epsilon):
            U = U_new
            centroids = centroids_new
            break
        U, centroids = U_new, centroids_new

    memberships: Dict[Any, Dict[int, float]] = {}
    communities: Dict[int, List[Any]] = defaultdict(list)
    for idx, node in enumerate(nodes):
        selected = {
            cluster: float(U[idx, cluster])
            for cluster in range(K)
            if U[idx, cluster] >= float(upper_thr)
        }
        if not selected:
            best_cluster = int(np.argmax(U[idx]))
            selected = {best_cluster: float(U[idx, best_cluster])}
        selected = _normalize_membership(selected)
        memberships[node] = selected
        for cluster in selected:
            communities[cluster].append(node)
    return dict(communities), memberships


def nmg(G: nx.Graph, threshold: float = 0.3, seed: int = 123) -> MethodOutput:
    rng = random.Random(seed)
    nodes = list(G.nodes())
    rng.shuffle(nodes)
    neighbor_sets = {node: set(G.neighbors(node)) for node in G.nodes()}
    communities: List[set] = []

    for seed_node in nodes:
        community = {seed_node}
        candidates = neighbor_sets[seed_node].copy()
        for candidate in candidates:
            union = neighbor_sets[seed_node] | neighbor_sets[candidate] | {seed_node, candidate}
            score = len(neighbor_sets[seed_node] & neighbor_sets[candidate]) / len(union) if union else 0.0
            if score >= float(threshold):
                community.add(candidate)
        if len(community) > 1 and not any(community <= old for old in communities):
            communities.append(community)

    output = {idx: sorted(comm, key=str) for idx, comm in enumerate(communities)}
    memberships: Dict[Any, Dict[int, float]] = defaultdict(dict)
    for label, members in output.items():
        for node in members:
            memberships[node][label] = 1.0
    return output, dict(memberships)


METHOD_FUNCTIONS: Dict[str, Callable[..., MethodOutput]] = {
    "fuzzy BLDLP": fuzzy_bldlp,
    "fuzzy LPA": fuzzy_lpa,
    "fuzzy LDPA": fuzzy_ldpa,
    "CFinder": cfinder,
    "LFK": lfk,
    "Hybrid C-Means": hybrid_cmeans,
    "NMG": nmg,
}

METHOD_ALIASES = {
    "bldlp": "fuzzy BLDLP",
    "fuzzy_bldlp": "fuzzy BLDLP",
    "lpa": "fuzzy LPA",
    "fuzzy_lpa": "fuzzy LPA",
    "ldpa": "fuzzy LDPA",
    "fuzzy_ldpa": "fuzzy LDPA",
    "cfinder": "CFinder",
    "lfk": "LFK",
    "hybrid": "Hybrid C-Means",
    "hybrid_cmeans": "Hybrid C-Means",
    "nmg": "NMG",
}


def run_method(
    method_name: str,
    G: nx.Graph,
    params: Optional[Mapping[str, Any]] = None,
    seed: int = 123,
) -> MethodOutput:
    if method_name not in METHOD_FUNCTIONS:
        method_name = METHOD_ALIASES.get(method_name.lower(), method_name)
    if method_name not in METHOD_FUNCTIONS:
        raise ValueError(f"Unknown method: {method_name}")
    return METHOD_FUNCTIONS[method_name](G, seed=seed, **dict(params or {}))


def default_grids(G: nx.Graph, full_grid: bool = False) -> Dict[str, Dict[str, Sequence[Any]]]:
    n = G.number_of_nodes()
    base_k = max(2, int(math.sqrt(max(n, 1))))
    k_values = sorted({max(2, base_k - 1), base_k, min(n, base_k + 1)})
    if full_grid:
        return {
            "fuzzy BLDLP": {
                "alpha": [0.25, 0.5, 0.75], "lambd": [1.0, 1.5, 2.0],
                "threshold": [0.2, 0.3, 0.4, 0.5], "max_iter": [50, 100, 200],
            },
            "fuzzy LPA": {
                "max_iter": [50, 100, 150], "threshold": [0.01, 0.05, 0.1, 0.2],
                "lambd": [1.0, 2.0, 5.0],
            },
            "fuzzy LDPA": {
                "max_iter": [100, 150, 200], "threshold": [0.15, 0.2, 0.25, 0.3],
                "lambd": [1.0, 2.5, 5.0], "merge_threshold": [0.2, 0.3, 0.4],
            },
            "CFinder": {"k": list(range(3, min(10, n) + 1))},
            "LFK": {"alpha": list(np.linspace(0.5, 2.5, 9))},
            "Hybrid C-Means": {
                "K_param": k_values, "m": [1.5, 2.0, 2.5], "epsilon": [1e-4, 1e-5],
                "max_iter": [50, 100], "lower_thr": [0.5, 0.6, 0.7],
                "upper_thr": [0.2, 0.3, 0.4],
            },
            "NMG": {"threshold": [0.1, 0.2, 0.3, 0.4, 0.5]},
        }
    return {
        "fuzzy BLDLP": {
            "alpha": [0.5], "lambd": [1.5, 2.0], "threshold": [0.3, 0.4], "max_iter": [100],
        },
        "fuzzy LPA": {
            "max_iter": [100], "threshold": [0.05, 0.1], "lambd": [2.0, 5.0],
        },
        "fuzzy LDPA": {
            "max_iter": [100], "threshold": [0.2, 0.3], "lambd": [1.0, 2.5],
            "merge_threshold": [0.3],
        },
        "CFinder": {"k": [3, 4, 5] if n >= 5 else [2]},
        "LFK": {"alpha": [0.75, 1.0, 1.25]},
        "Hybrid C-Means": {
            "K_param": k_values, "m": [1.5, 2.0], "epsilon": [1e-4],
            "max_iter": [50], "lower_thr": [0.6], "upper_thr": [0.2, 0.3],
        },
        "NMG": {"threshold": [0.2, 0.3, 0.4]},
    }


def _grid_combinations(grid: Mapping[str, Sequence[Any]]) -> List[Dict[str, Any]]:
    keys = list(grid)
    return [dict(zip(keys, values)) for values in itertools.product(*(grid[key] for key in keys))]


def tune_method(
    method_name: str,
    G: nx.Graph,
    grid: Mapping[str, Sequence[Any]],
    runs_per_config: int,
    base_seed: int,
    max_configs: int,
    verbose: bool,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    configs = _grid_combinations(grid)
    if max_configs > 0 and len(configs) > max_configs:
        configs = random.Random(base_seed).sample(configs, max_configs)

    rows: List[Dict[str, Any]] = []
    best: Optional[Dict[str, Any]] = None
    for index, params in enumerate(configs, start=1):
        scores = []
        counts = []
        for run in range(runs_per_config):
            seed = base_seed + index * 1000 + run
            seed_everything(seed)
            communities, memberships = run_method(method_name, G, params, seed)
            scores.append(safe_modularity(G, communities, memberships))
            counts.append(len(communities))
        row = {
            **params,
            "modularity_mean": float(np.mean(scores)),
            "modularity_std": float(np.std(scores)),
            "communities_mean": float(np.mean(counts)),
        }
        rows.append(row)
        if best is None or row["modularity_mean"] > best["modularity_mean"]:
            best = row.copy()
        if verbose:
            print(
                f"  [{index:03d}/{len(configs):03d}] "
                f"Q={row['modularity_mean']:.6f} ± {row['modularity_std']:.6f}"
            )

    if best is None:
        raise RuntimeError(f"No configuration evaluated for {method_name}")
    frame = pd.DataFrame(rows).sort_values("modularity_mean", ascending=False).reset_index(drop=True)
    best_result = {
        "params": {key: best[key] for key in grid},
        "modularity_mean": best["modularity_mean"],
        "modularity_std": best["modularity_std"],
    }
    return frame, best_result


def parse_method_list(value: str) -> List[str]:
    if value.strip().lower() == "all":
        return list(METHOD_FUNCTIONS)
    selected = []
    for token in value.split(","):
        token = token.strip()
        canonical = METHOD_ALIASES.get(token.lower(), token)
        if canonical not in METHOD_FUNCTIONS:
            raise ValueError(f"Unknown method '{token}'")
        selected.append(canonical)
    return selected


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tune comparison community-detection methods")
    parser.add_argument("--dataset", required=True, choices=["karate", "dolphins", "football", "yeast", "y2h"])
    parser.add_argument("--data-dir", default=".")
    parser.add_argument("--graph", default=None)
    parser.add_argument("--gt", default=None, help="Accepted for CLI consistency; tuning is unsupervised")
    parser.add_argument("--methods", default="all", help="all or comma-separated aliases")
    parser.add_argument("--runs-per-config", type=int, default=2)
    parser.add_argument("--max-configs", type=int, default=30, help="Maximum configurations per method; 0 means all")
    parser.add_argument("--base-seed", type=int, default=123)
    parser.add_argument("--full-grid", action="store_true")
    parser.add_argument("--out-json", default=None)
    parser.add_argument("--out-dir", default=".")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset, graph_path, _ = resolve_dataset_paths(
        args.dataset, args.data_dir, args.graph, args.gt, require_gt=False
    )
    G = load_graph(dataset, graph_path)
    print(f"Dataset={dataset} | nodes={G.number_of_nodes()} | edges={G.number_of_edges()}")

    selected = parse_method_list(args.methods)
    grids = default_grids(G, full_grid=args.full_grid)
    output_dir = Path(args.out_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    best_methods: Dict[str, Any] = {}

    for method_index, method_name in enumerate(selected):
        print(f"\n=== Tuning {method_name} ===")
        frame, best = tune_method(
            method_name,
            G,
            grids[method_name],
            runs_per_config=args.runs_per_config,
            base_seed=args.base_seed + method_index * 100000,
            max_configs=args.max_configs,
            verbose=True,
        )
        best_methods[method_name] = best
        csv_name = method_name.lower().replace(" ", "_").replace("-", "_")
        csv_path = output_dir / f"tuning_{csv_name}_{dataset}.csv"
        frame.to_csv(csv_path, index=False)
        print(f"Best params: {best['params']}")
        print(f"Saved: {csv_path.resolve()}")

    payload = {
        "dataset": dataset,
        "selection_metric": "hard-partition modularity",
        "methods": best_methods,
    }
    out_json = args.out_json or output_dir / f"best_other_methods_{dataset}.json"
    json_path = save_json(out_json, payload)
    print(f"\nSaved best parameters: {json_path.resolve()}")


if __name__ == "__main__":
    main()
