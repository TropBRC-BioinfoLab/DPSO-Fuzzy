"""PSO-LPA training and hyperparameter tuning for benchmark networks.

Despite its historical filename, this script is no longer Yeast-specific. It
supports karate, dolphins, football, Yeast-D2, and Y2H through a common CLI.
The best configuration is written to JSON and is consumed by
``allmethodyeast.py``.
"""

from __future__ import annotations

import argparse
import itertools
import math
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

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


class PSOCommunityDetection:
    """Discrete PSO with categorical community labels.

    Positions are integer community labels aligned to ``node_list``. After each
    position update, disconnected pieces carrying the same label are split so
    every hard community remains connected.
    """

    def __init__(
        self,
        graph: nx.Graph,
        num_particles: int = 50,
        max_iter: int = 100,
        rho: float = 0.5,
        velocity_bound: float = 4.0,
        seed: Optional[int] = None,
    ) -> None:
        if graph.number_of_nodes() == 0:
            raise ValueError("PSO requires a non-empty graph.")
        self.graph = graph
        self.node_list = list(graph.nodes())
        self.node_index = {node: idx for idx, node in enumerate(self.node_list)}
        self.num_nodes = len(self.node_list)
        self.num_particles = int(num_particles)
        self.max_iter = int(max_iter)
        self.rho = float(rho)
        self.velocity_bound = float(velocity_bound)
        self.rng = random.Random(seed)
        self.global_best_position: Optional[List[int]] = None
        self.global_best_modularity = -float("inf")
        self.particles: List[Dict[str, Any]] = []

    @staticmethod
    def _canonicalize(labels: Sequence[int]) -> List[int]:
        mapping: Dict[int, int] = {}
        canonical: List[int] = []
        next_label = 0
        for label in labels:
            label = int(label)
            if label not in mapping:
                mapping[label] = next_label
                next_label += 1
            canonical.append(mapping[label])
        return canonical

    def initialize_particle(self, p: float = 0.5) -> Dict[str, Any]:
        """Neighbor-based initialization used by the supplied DPSO code."""
        labels = [-1] * self.num_nodes
        current_label = 0

        for start_idx in range(self.num_nodes):
            if labels[start_idx] != -1:
                continue
            labels[start_idx] = current_label
            queue = [start_idx]
            while queue:
                idx = queue.pop(0)
                node = self.node_list[idx]
                neighbors = list(self.graph.neighbors(node))
                self.rng.shuffle(neighbors)
                for neighbor in neighbors:
                    j = self.node_index[neighbor]
                    if labels[j] == -1 and self.rng.random() > p:
                        labels[j] = current_label
                        queue.append(j)
            current_label += 1

        labels = self._split_disconnected(labels)
        return {
            "position": labels,
            "velocity": [0.0] * self.num_nodes,
            "best_position": labels.copy(),
            "best_modularity": -float("inf"),
        }

    def _sigmoid(self, velocity: float) -> float:
        value = max(-self.velocity_bound, min(self.velocity_bound, float(velocity)))
        return 1.0 / (1.0 + math.exp(-value))

    def update_velocity(self, particle: Dict[str, Any], w: float, c1: float, c2: float) -> None:
        for idx in range(self.num_nodes):
            current = particle["position"][idx]
            pbest = particle["best_position"][idx]
            gbest = self.global_best_position[idx] if self.global_best_position else current
            cognitive = c1 * self.rng.random() * float(pbest != current)
            social = c2 * self.rng.random() * float(gbest != current)
            value = w * particle["velocity"][idx] + cognitive + social
            particle["velocity"][idx] = max(
                -self.velocity_bound, min(self.velocity_bound, value)
            )

    def update_position(self, particle: Dict[str, Any], c1: float, c2: float) -> None:
        old_position = particle["position"].copy()
        new_position = old_position.copy()

        for idx, node in enumerate(self.node_list):
            activation = self._sigmoid(particle["velocity"][idx])
            if activation < self.rho or self.rng.random() > activation:
                continue

            candidates: List[int] = [old_position[idx]]
            weights: List[float] = [0.10]

            p_label = particle["best_position"][idx]
            if p_label != old_position[idx]:
                candidates.append(p_label)
                weights.append(max(c1, 1e-9))

            if self.global_best_position is not None:
                g_label = self.global_best_position[idx]
                if g_label != old_position[idx]:
                    candidates.append(g_label)
                    weights.append(max(c2, 1e-9))

            neighbor_labels = [
                old_position[self.node_index[nbr]] for nbr in self.graph.neighbors(node)
            ]
            if neighbor_labels:
                counts: Dict[int, int] = defaultdict(int)
                for label in neighbor_labels:
                    counts[label] += 1
                majority = max(counts, key=lambda label: (counts[label], -label))
                candidates.append(majority)
                weights.append(1.0)

            new_position[idx] = self.rng.choices(candidates, weights=weights, k=1)[0]

        particle["position"] = self._split_disconnected(new_position)

    def _split_disconnected(self, labels: Sequence[int]) -> List[int]:
        label_nodes: Dict[int, List[Any]] = defaultdict(list)
        for idx, label in enumerate(labels):
            label_nodes[int(label)].append(self.node_list[idx])

        fixed = list(map(int, labels))
        next_label = max(fixed, default=-1) + 1
        for label, members in label_nodes.items():
            subgraph = self.graph.subgraph(members)
            components = list(nx.connected_components(subgraph))
            if len(components) <= 1:
                continue
            components.sort(key=len, reverse=True)
            for component in components[1:]:
                for node in component:
                    fixed[self.node_index[node]] = next_label
                next_label += 1
        return self._canonicalize(fixed)

    def evaluate_position(self, position: Sequence[int]) -> float:
        communities: Dict[int, List[Any]] = defaultdict(list)
        for idx, label in enumerate(position):
            communities[int(label)].append(self.node_list[idx])
        return safe_modularity(self.graph, communities)

    def run(
        self,
        w: float = 0.7,
        c1: float = 1.5,
        c2: float = 1.0,
        p: float = 0.6,
        verbose: bool = False,
    ) -> List[int]:
        self.particles = [self.initialize_particle(p=p) for _ in range(self.num_particles)]
        self.global_best_position = None
        self.global_best_modularity = -float("inf")

        for particle in self.particles:
            score = self.evaluate_position(particle["position"])
            particle["best_position"] = particle["position"].copy()
            particle["best_modularity"] = score
            if score > self.global_best_modularity:
                self.global_best_modularity = score
                self.global_best_position = particle["position"].copy()

        for iteration in range(self.max_iter):
            for particle in self.particles:
                self.update_velocity(particle, w=w, c1=c1, c2=c2)
                self.update_position(particle, c1=c1, c2=c2)
                score = self.evaluate_position(particle["position"])
                if score > particle["best_modularity"]:
                    particle["best_modularity"] = score
                    particle["best_position"] = particle["position"].copy()
                if score > self.global_best_modularity:
                    self.global_best_modularity = score
                    self.global_best_position = particle["position"].copy()

            if verbose and (
                iteration == self.max_iter - 1
                or iteration % max(1, self.max_iter // 10) == 0
            ):
                print(
                    f"PSO iteration {iteration + 1}/{self.max_iter}: "
                    f"Q={self.global_best_modularity:.6f}"
                )

        if self.global_best_position is None:
            raise RuntimeError("PSO did not produce a valid partition.")
        return self.global_best_position

    def position_to_assignment(self, position: Sequence[int]) -> Dict[Any, int]:
        return {node: int(position[idx]) for idx, node in enumerate(self.node_list)}


def label_propagation_refinement(
    G: nx.Graph,
    pso_assignment: Mapping[Any, int],
    max_iter: int = 50,
    label_threshold: float = 0.5,
    prior_strength: float = 0.25,
    membership_threshold: float = 0.30,
    seed: Optional[int] = None,
) -> Tuple[Dict[int, List[Any]], Dict[Any, Dict[int, float]]]:
    """PSO-seeded balanced multi-label propagation.

    Each node starts with the one-hot PSO label. Neighbor label distributions are
    propagated, labels below ``label_threshold * max_score`` are pruned, and the
    PSO label receives ``prior_strength`` so the optimizer remains a structural
    prior rather than being discarded by LPA.
    """
    rng = random.Random(seed)
    labels = sorted(set(int(label) for label in pso_assignment.values()))
    distributions: Dict[Any, Dict[int, float]] = {
        node: {int(pso_assignment[node]): 1.0} for node in G.nodes()
    }

    for _ in range(int(max_iter)):
        changed = False
        nodes = list(G.nodes())
        rng.shuffle(nodes)
        updated: Dict[Any, Dict[int, float]] = {}

        for node in nodes:
            scores: Dict[int, float] = defaultdict(float)
            total_weight = 0.0
            for neighbor in G.neighbors(node):
                weight = float(G.get_edge_data(node, neighbor, default={}).get("weight", 1.0))
                total_weight += weight
                for label, membership in distributions[neighbor].items():
                    scores[int(label)] += weight * float(membership)

            prior_label = int(pso_assignment[node])
            scores[prior_label] += max(0.0, float(prior_strength)) * max(total_weight, 1.0)

            if not scores:
                updated[node] = {prior_label: 1.0}
                continue

            max_score = max(scores.values())
            retained = {
                label: score
                for label, score in scores.items()
                if score + 1e-12 >= float(label_threshold) * max_score
            }
            if not retained:
                retained = {max(scores, key=scores.get): max_score}

            total = sum(retained.values())
            normalized = {label: score / total for label, score in retained.items()}
            updated[node] = normalized

            old_dominant = max(distributions[node], key=distributions[node].get)
            new_dominant = max(normalized, key=normalized.get)
            if old_dominant != new_dominant or set(distributions[node]) != set(normalized):
                changed = True

        distributions = updated
        if not changed:
            break

    communities: Dict[int, List[Any]] = defaultdict(list)
    memberships: Dict[Any, Dict[int, float]] = {}
    for node, distribution in distributions.items():
        selected = {
            label: value
            for label, value in distribution.items()
            if value >= float(membership_threshold)
        }
        if not selected:
            best_label = max(distribution, key=distribution.get)
            selected = {best_label: distribution[best_label]}
        selected_total = sum(selected.values())
        memberships[node] = {
            label: value / selected_total for label, value in selected.items()
        }
        for label in memberships[node]:
            communities[int(label)].append(node)

    return dict(communities), memberships


def run_pso_lpa(
    G: nx.Graph,
    params: Mapping[str, Any],
    seed: int = 123,
    verbose: bool = False,
) -> Tuple[Dict[int, List[Any]], Dict[Any, Dict[int, float]], Dict[str, Any]]:
    required_defaults = {
        "w": 0.7,
        "c1": 1.5,
        "c2": 1.0,
        "rho": 0.5,
        "p": 0.6,
        "num_particles": 50,
        "max_iter": 100,
        "lpa_max_iter": 50,
        "lpa_threshold": 0.5,
        "prior_strength": 0.25,
        "membership_threshold": 0.30,
    }
    config = {**required_defaults, **dict(params)}
    pso = PSOCommunityDetection(
        G,
        num_particles=int(config["num_particles"]),
        max_iter=int(config["max_iter"]),
        rho=float(config["rho"]),
        seed=seed,
    )
    position = pso.run(
        w=float(config["w"]),
        c1=float(config["c1"]),
        c2=float(config["c2"]),
        p=float(config["p"]),
        verbose=verbose,
    )
    assignment = pso.position_to_assignment(position)
    communities, memberships = label_propagation_refinement(
        G,
        assignment,
        max_iter=int(config["lpa_max_iter"]),
        label_threshold=float(config["lpa_threshold"]),
        prior_strength=float(config["prior_strength"]),
        membership_threshold=float(config["membership_threshold"]),
        seed=seed + 1,
    )
    metadata = {
        "pso_modularity": float(pso.global_best_modularity),
        "num_pso_communities": len(set(position)),
        "num_final_communities": len(communities),
    }
    return communities, memberships, metadata


def _sample_configurations(
    grid: Mapping[str, Sequence[Any]], max_configs: int, seed: int
) -> List[Dict[str, Any]]:
    keys = list(grid)
    combinations = [dict(zip(keys, values)) for values in itertools.product(*(grid[k] for k in keys))]
    if max_configs > 0 and len(combinations) > max_configs:
        rng = random.Random(seed)
        combinations = rng.sample(combinations, max_configs)
    return combinations


def tune_pso_lpa(
    G: nx.Graph,
    grid: Mapping[str, Sequence[Any]],
    runs_per_config: int = 2,
    base_seed: int = 123,
    max_configs: int = 30,
    verbose: bool = True,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    configs = _sample_configurations(grid, max_configs=max_configs, seed=base_seed)
    rows: List[Dict[str, Any]] = []
    best: Optional[Dict[str, Any]] = None

    for index, params in enumerate(configs, start=1):
        scores = []
        for run in range(runs_per_config):
            seed = base_seed + index * 1000 + run
            seed_everything(seed)
            communities, memberships, _ = run_pso_lpa(G, params, seed=seed, verbose=False)
            scores.append(safe_modularity(G, communities, memberships))

        row = {
            **params,
            "modularity_mean": float(np.mean(scores)),
            "modularity_std": float(np.std(scores)),
            "valid_runs": int(sum(score > -1 for score in scores)),
        }
        rows.append(row)
        if best is None or row["modularity_mean"] > best["modularity_mean"]:
            best = row.copy()

        if verbose:
            print(
                f"[{index:03d}/{len(configs):03d}] "
                f"Q={row['modularity_mean']:.6f} ± {row['modularity_std']:.6f}"
            )

    if best is None:
        raise RuntimeError("No PSO-LPA configuration was evaluated.")
    frame = pd.DataFrame(rows).sort_values("modularity_mean", ascending=False).reset_index(drop=True)
    param_keys = list(grid)
    best_payload = {
        "params": {key: best[key] for key in param_keys},
        "modularity_mean": best["modularity_mean"],
        "modularity_std": best["modularity_std"],
    }
    return frame, best_payload


def default_grid(full_grid: bool = False) -> Dict[str, Sequence[Any]]:
    if full_grid:
        return {
            "w": [0.5, 0.7, 0.9],
            "c1": [1.0, 1.5, 2.0],
            "c2": [1.0, 1.5, 2.0],
            "rho": [0.5, 0.7],
            "p": [0.4, 0.6],
            "num_particles": [50, 100],
            "max_iter": [100, 300],
            "lpa_max_iter": [50, 100],
            "lpa_threshold": [0.3, 0.5, 0.7],
            "prior_strength": [0.10, 0.25],
            "membership_threshold": [0.20, 0.30, 0.40],
        }
    return {
        "w": [0.7, 0.9],
        "c1": [1.5, 2.0],
        "c2": [1.0, 1.5],
        "rho": [0.5, 0.7],
        "p": [0.4, 0.6],
        "num_particles": [30, 50],
        "max_iter": [50, 100],
        "lpa_max_iter": [50],
        "lpa_threshold": [0.3, 0.5],
        "prior_strength": [0.10, 0.25],
        "membership_threshold": [0.30],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tune PSO-LPA on a benchmark network")
    parser.add_argument("--dataset", required=True, choices=["karate", "dolphins", "football", "yeast", "y2h"])
    parser.add_argument("--data-dir", default=".")
    parser.add_argument("--graph", default=None, help="Explicit graph path")
    parser.add_argument("--gt", default=None, help="Accepted for a consistent benchmark CLI; not used for tuning")
    parser.add_argument("--runs-per-config", type=int, default=2)
    parser.add_argument("--max-configs", type=int, default=30, help="0 evaluates the entire grid")
    parser.add_argument("--base-seed", type=int, default=123)
    parser.add_argument("--full-grid", action="store_true")
    parser.add_argument("--out-json", default=None)
    parser.add_argument("--out-csv", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset, graph_path, _ = resolve_dataset_paths(
        args.dataset, args.data_dir, args.graph, args.gt, require_gt=False
    )
    G = load_graph(dataset, graph_path)
    print(f"Dataset={dataset} | nodes={G.number_of_nodes()} | edges={G.number_of_edges()}")

    frame, best = tune_pso_lpa(
        G,
        default_grid(full_grid=args.full_grid),
        runs_per_config=args.runs_per_config,
        base_seed=args.base_seed,
        max_configs=args.max_configs,
        verbose=True,
    )

    out_json = args.out_json or f"best_pso_lpa_{dataset}.json"
    out_csv = args.out_csv or f"tuning_pso_lpa_{dataset}.csv"
    payload = {
        "dataset": dataset,
        "method": "PSO-LPA",
        "selection_metric": "hard-partition modularity after LPA refinement",
        **best,
    }
    json_path = save_json(out_json, payload)
    frame.to_csv(out_csv, index=False)
    print("\nBest PSO-LPA parameters:")
    print(payload["params"])
    print(f"Best modularity: {payload['modularity_mean']:.6f}")
    print(f"Saved JSON: {json_path}")
    print(f"Saved CSV : {Path(out_csv).resolve()}")


if __name__ == "__main__":
    main()
