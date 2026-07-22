"""Shared graph, ground-truth, parameter, and evaluation utilities.

The module is intentionally dependency-light. CDlib is optional: when it is not
installed, ONMI is reported as NaN while the remaining metrics are still
computed.
"""

from __future__ import annotations

import json
import math
import os
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import networkx as nx
import numpy as np
from scipy.io import mmread

try:
    from cdlib import NodeClustering, evaluation as cdlib_evaluation
except ImportError:  # CDlib remains optional so the scripts can still run.
    NodeClustering = None
    cdlib_evaluation = None


DATASET_ALIASES = {
    "karate": "karate",
    "zachary": "karate",
    "dolphin": "dolphins",
    "dolphins": "dolphins",
    "soc-dolphins": "dolphins",
    "football": "football",
    "yeast": "yeast",
    "yeast-d2": "yeast",
    "y2h": "y2h",
}

DATASET_DEFAULTS: Dict[str, Dict[str, Sequence[str]]] = {
    "karate": {"graph": (), "gt": ()},
    "dolphins": {"graph": ("soc-dolphins.mtx",), "gt": ()},
    "football": {
        "graph": (
            "football.gml",
            "football.edgelist",
            "football.txt",
            "football.dat",
            "football.net",
            # Supported because this was supplied as the football graph filename.
            "football_GT.txt",
        ),
        # Prefer the supplied misspelled GT filename when both files exist.
        "gt": ("gootball_GT.txt", "football_GT.txt"),
    },
    "yeast": {"graph": ("Yeast_D2.txt",), "gt": ("Yeast_GT.txt",)},
    "y2h": {
        "graph": ("Y2H_reconciled_full.edgelist",),
        "gt": ("Y2H_GT.txt",),
    },
}


def canonical_dataset_name(name: str) -> str:
    key = name.strip().lower()
    if key not in DATASET_ALIASES:
        valid = ", ".join(sorted(set(DATASET_ALIASES.values())))
        raise ValueError(f"Unknown dataset '{name}'. Valid datasets: {valid}")
    return DATASET_ALIASES[key]


def _resolve_existing_file(
    data_dir: Path,
    explicit_path: Optional[str],
    candidates: Sequence[str],
    role: str,
    required: bool,
) -> Optional[Path]:
    if explicit_path:
        path = Path(explicit_path).expanduser()
        if not path.is_absolute():
            path = data_dir / path
        if not path.exists():
            raise FileNotFoundError(f"{role} file not found: {path}")
        return path

    for filename in candidates:
        path = data_dir / filename
        if path.exists():
            return path

    if required:
        tried = ", ".join(str(data_dir / c) for c in candidates) or "<none>"
        raise FileNotFoundError(
            f"Could not find the {role} file. Tried: {tried}. "
            f"Use --{'graph' if role == 'graph' else 'gt'} to set it explicitly."
        )
    return None


def resolve_dataset_paths(
    dataset: str,
    data_dir: str = ".",
    graph_path: Optional[str] = None,
    gt_path: Optional[str] = None,
    require_gt: bool = True,
) -> Tuple[str, Optional[Path], Optional[Path]]:
    dataset = canonical_dataset_name(dataset)
    directory = Path(data_dir).expanduser().resolve()
    defaults = DATASET_DEFAULTS[dataset]

    graph_required = dataset != "karate"
    gt_required = require_gt and dataset not in {"karate", "dolphins"}

    graph = _resolve_existing_file(
        directory, graph_path, defaults["graph"], "graph", graph_required
    )
    gt = _resolve_existing_file(
        directory, gt_path, defaults["gt"], "ground-truth", gt_required
    )
    return dataset, graph, gt


def _clean_graph(G: nx.Graph) -> nx.Graph:
    if G.is_directed():
        G = G.to_undirected()
    else:
        G = nx.Graph(G)
    G.remove_edges_from(nx.selfloop_edges(G))
    isolates = list(nx.isolates(G))
    if isolates:
        G.remove_nodes_from(isolates)
    return G


def load_graph(dataset: str, graph_path: Optional[Path]) -> nx.Graph:
    dataset = canonical_dataset_name(dataset)
    if dataset == "karate":
        return _clean_graph(nx.karate_club_graph())
    if graph_path is None:
        raise ValueError(f"Dataset '{dataset}' requires a graph path.")

    suffix = graph_path.suffix.lower()
    if suffix == ".mtx":
        matrix = mmread(str(graph_path))
        try:
            G = nx.from_scipy_sparse_array(matrix)
        except AttributeError:  # compatibility with older NetworkX
            G = nx.from_scipy_sparse_matrix(matrix)
    elif suffix == ".gml":
        G = nx.read_gml(graph_path)
    elif suffix == ".graphml":
        G = nx.read_graphml(graph_path)
    elif suffix == ".net":
        G = nx.read_pajek(graph_path)
    else:
        # String identifiers preserve protein IDs such as YHR023W.
        G = nx.read_edgelist(
            graph_path,
            comments="#",
            delimiter=None,
            data=False,
            nodetype=str,
        )
    return _clean_graph(G)


def karate_ground_truth(G: nx.Graph) -> Dict[str, List[Any]]:
    communities: Dict[str, List[Any]] = defaultdict(list)
    for node, data in G.nodes(data=True):
        club = data.get("club")
        if club is not None:
            communities[str(club)].append(node)
    return dict(communities)


def dolphin_ground_truth(G: nx.Graph) -> Dict[int, List[Any]]:
    nodes_0 = {
        0, 2, 3, 4, 8, 10, 11, 12, 14, 15, 16, 18, 20, 21, 23, 24,
        28, 29, 30, 33, 34, 35, 36, 37, 38, 39, 40, 42, 43, 44, 45,
        46, 47, 49, 50, 51, 52, 53, 55, 58, 59, 61,
    }
    comm0, comm1 = [], []
    for node in G.nodes():
        # Matrix Market loading creates integer node IDs 0..N-1.
        node_as_int = int(node) if str(node).lstrip("-").isdigit() else node
        (comm0 if node_as_int in nodes_0 else comm1).append(node)
    return {0: comm0, 1: comm1}


def _split_tokens(text: str) -> List[str]:
    return [tok for tok in re.split(r"[\s,;]+", text.strip()) if tok]


def load_ground_truth_file(path: Path, G: nx.Graph) -> Dict[str, List[Any]]:
    """Read common GT formats and align tokens to the graph's node type.

    Supported formats:
      C1: node1 node2 node3
      node community_label                 (one assignment per line)
      C1 node1 node2 node3                 (community label followed by nodes)
      node1 node2 node3                    (one community per line)
    """
    raw_lines = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line and not line.startswith("#"):
                raw_lines.append(line)

    lookup = {str(node): node for node in G.nodes()}
    communities: Dict[str, List[Any]] = defaultdict(list)

    if not raw_lines:
        raise ValueError(f"Ground-truth file is empty: {path}")

    if any(":" in line for line in raw_lines):
        for row_no, line in enumerate(raw_lines):
            if ":" not in line:
                continue
            label, node_text = line.split(":", 1)
            label = label.strip() or str(row_no)
            for token in _split_tokens(node_text):
                if token in lookup:
                    communities[label].append(lookup[token])
    else:
        token_rows = [_split_tokens(line) for line in raw_lines]
        # Detect two-column node -> label assignments.
        two_column_assignments = (
            all(len(row) == 2 for row in token_rows)
            and sum(row[0] in lookup for row in token_rows) >= max(1, int(0.8 * len(token_rows)))
        )
        if two_column_assignments:
            for node_token, label in token_rows:
                if node_token in lookup:
                    communities[label].append(lookup[node_token])
        else:
            for row_no, row in enumerate(token_rows):
                if not row:
                    continue
                if row[0] not in lookup and any(tok in lookup for tok in row[1:]):
                    label, node_tokens = row[0], row[1:]
                else:
                    label, node_tokens = str(row_no), row
                for token in node_tokens:
                    if token in lookup:
                        communities[label].append(lookup[token])

    # Deduplicate and remove empty communities.
    cleaned: Dict[str, List[Any]] = {}
    for label, members in communities.items():
        seen = set()
        unique_members = []
        for node in members:
            if node not in seen:
                seen.add(node)
                unique_members.append(node)
        if unique_members:
            cleaned[str(label)] = unique_members

    if not cleaned:
        example_nodes = list(lookup)[:5]
        raise ValueError(
            f"No GT nodes from {path} matched graph nodes. "
            f"Example graph node IDs: {example_nodes}"
        )
    return cleaned


def load_ground_truth(
    dataset: str,
    G: nx.Graph,
    gt_path: Optional[Path],
) -> Dict[Any, List[Any]]:
    dataset = canonical_dataset_name(dataset)
    if dataset == "karate":
        return karate_ground_truth(G)
    if dataset == "dolphins":
        return dolphin_ground_truth(G)
    if gt_path is None:
        raise ValueError(f"Dataset '{dataset}' requires a ground-truth path.")
    return load_ground_truth_file(gt_path, G)


def normalize_communities(
    communities: Mapping[Any, Iterable[Any]] | Sequence[Iterable[Any]],
    valid_nodes: Optional[set] = None,
    min_size: int = 1,
) -> Dict[int, List[Any]]:
    if isinstance(communities, Mapping):
        iterable = communities.values()
    else:
        iterable = communities

    normalized: Dict[int, List[Any]] = {}
    for members in iterable:
        unique = []
        seen = set()
        for node in members:
            if valid_nodes is not None and node not in valid_nodes:
                continue
            if node not in seen:
                seen.add(node)
                unique.append(node)
        if len(unique) >= min_size:
            normalized[len(normalized)] = unique
    return normalized


def communities_to_hard_partition(
    G: nx.Graph,
    communities: Mapping[Any, Iterable[Any]],
    memberships: Optional[Mapping[Any, Mapping[Any, float]]] = None,
) -> List[set]:
    """Convert overlapping output to a complete disjoint partition for modularity."""
    node_candidates: Dict[Any, List[Tuple[Any, float, int]]] = defaultdict(list)
    for label, members in communities.items():
        size = len(list(members)) if not isinstance(members, list) else len(members)
        for node in members:
            mu = 1.0
            if memberships and node in memberships:
                mu = float(memberships[node].get(label, 0.0))
            node_candidates[node].append((label, mu, size))

    hard: Dict[Any, set] = defaultdict(set)
    for node in G.nodes():
        candidates = node_candidates.get(node, [])
        if candidates:
            label = max(candidates, key=lambda item: (item[1], item[2], str(item[0])))[0]
            hard[label].add(node)
        else:
            # An unassigned node is a singleton, ensuring a valid partition.
            hard[("singleton", node)].add(node)
    return [members for members in hard.values() if members]


def safe_modularity(
    G: nx.Graph,
    communities: Mapping[Any, Iterable[Any]],
    memberships: Optional[Mapping[Any, Mapping[Any, float]]] = None,
) -> float:
    try:
        partition = communities_to_hard_partition(G, communities, memberships)
        if len(partition) <= 1:
            return -1.0
        return float(nx.community.modularity(G, partition, weight="weight"))
    except Exception:
        return -1.0


def _pair_set(communities: Sequence[set]) -> set:
    pairs = set()
    for community in communities:
        ordered = sorted(community, key=str)
        for i in range(len(ordered)):
            for j in range(i + 1, len(ordered)):
                pairs.add((ordered[i], ordered[j]))
    return pairs


def _community_prf(gt_sets: Sequence[set], pred_sets: Sequence[set]) -> Tuple[float, float, float]:
    def set_f1(a: set, b: set) -> float:
        overlap = len(a & b)
        return 0.0 if overlap == 0 else 2.0 * overlap / (len(a) + len(b))

    if not gt_sets or not pred_sets:
        return 0.0, 0.0, 0.0
    precision = float(np.mean([max(set_f1(p, g) for g in gt_sets) for p in pred_sets]))
    recall = float(np.mean([max(set_f1(g, p) for p in pred_sets) for g in gt_sets]))
    f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
    return precision, recall, f1


def _complete_cover(clusters: Sequence[set], universe: set) -> List[List[Any]]:
    completed = [list(c & universe) for c in clusters if c & universe]
    covered = set().union(*(set(c) for c in completed)) if completed else set()
    completed.extend([[node] for node in universe - covered])
    return completed


def evaluate_communities(
    G: nx.Graph,
    ground_truth: Mapping[Any, Iterable[Any]],
    predicted: Mapping[Any, Iterable[Any]],
) -> Dict[str, float]:
    """Evaluate overlapping communities consistently across all algorithms.

    P/R/F are pairwise metrics. Community P/R/F retain the best-community-match
    formulation used by the previous script. ONMI uses CDlib's LFK definition.
    """
    gt_norm = normalize_communities(ground_truth, valid_nodes=set(G.nodes()), min_size=1)
    pred_norm = normalize_communities(predicted, valid_nodes=set(G.nodes()), min_size=1)
    gt_sets = [set(v) for v in gt_norm.values()]
    pred_sets = [set(v) for v in pred_norm.values()]

    # Evaluate only nodes that possess a ground-truth annotation.
    gt_universe = set().union(*gt_sets) if gt_sets else set()
    pred_sets = [p & gt_universe for p in pred_sets if p & gt_universe]

    gt_pairs = _pair_set(gt_sets)
    pred_pairs = _pair_set(pred_sets)
    tp = len(gt_pairs & pred_pairs)
    precision = tp / len(pred_pairs) if pred_pairs else 0.0
    recall = tp / len(gt_pairs) if gt_pairs else 0.0
    f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)

    comm_p, comm_r, comm_f1 = _community_prf(gt_sets, pred_sets)

    onmi = math.nan
    omega = math.nan
    if cdlib_evaluation is not None and NodeClustering is not None and gt_universe:
        try:
            gt_completed = _complete_cover(gt_sets, gt_universe)
            pred_completed = _complete_cover(pred_sets, gt_universe)
            gt_nc = NodeClustering(gt_completed, G.subgraph(gt_universe).copy(), "GT")
            pred_nc = NodeClustering(pred_completed, G.subgraph(gt_universe).copy(), "Pred")
            onmi = float(
                cdlib_evaluation.overlapping_normalized_mutual_information_LFK(
                    pred_nc, gt_nc
                ).score
            )
            omega = float(cdlib_evaluation.omega(pred_nc, gt_nc).score)
        except Exception:
            onmi = math.nan
            omega = math.nan

    return {
        "P": float(precision),
        "R": float(recall),
        "F-Score": float(f1),
        "ONMI": onmi,
        "Omega": omega,
        "Comm_Precision": float(comm_p),
        "Comm_Recall": float(comm_r),
        "Comm_F1": float(comm_f1),
        "Num_Communities": float(len(pred_sets)),
    }


def seed_everything(seed: int) -> None:
    import random

    random.seed(seed)
    np.random.seed(seed)


def save_json(path: str | Path, payload: Mapping[str, Any]) -> Path:
    output = Path(path).expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)

    def convert(value: Any) -> Any:
        if isinstance(value, Mapping):
            return {str(k): convert(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [convert(v) for v in value]
        if isinstance(value, (np.integer,)):
            return int(value)
        if isinstance(value, (np.floating,)):
            return float(value)
        return value

    with output.open("w", encoding="utf-8") as handle:
        json.dump(convert(payload), handle, indent=2, allow_nan=True)
    return output


def load_json(path: str | Path) -> Dict[str, Any]:
    with Path(path).expanduser().open("r", encoding="utf-8") as handle:
        return json.load(handle)
