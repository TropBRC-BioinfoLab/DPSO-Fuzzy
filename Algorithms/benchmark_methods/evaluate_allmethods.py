"""Train and evaluate PSO-LPA and comparison methods on benchmark networks.

Parameter sources:
  * PSO-LPA: JSON produced by ``yeast.py``.
  * Other methods: JSON produced by ``hypeyeast_all.py``.

Supported datasets:
  karate, dolphins, football, yeast, and y2h.
"""

from __future__ import annotations

import argparse
import math
import traceback
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from benchmark_common import (
    evaluate_communities,
    load_graph,
    load_ground_truth,
    load_json,
    resolve_dataset_paths,
    safe_modularity,
    seed_everything,
)
from tune_othermethods import METHOD_ALIASES, METHOD_FUNCTIONS, run_method
from tune_pso_lpa import run_pso_lpa


ALL_METHODS = ["PSO-LPA", *METHOD_FUNCTIONS.keys()]


def _find_parameter_file(
    explicit_path: Optional[str],
    default_name: str,
    data_dir: str,
) -> Path:
    candidates: List[Path] = []
    if explicit_path:
        path = Path(explicit_path).expanduser()
        if not path.is_absolute():
            candidates.extend([Path.cwd() / path, Path(data_dir).expanduser() / path])
        else:
            candidates.append(path)
    else:
        candidates.extend([
            Path.cwd() / default_name,
            Path(data_dir).expanduser() / default_name,
        ])

    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    tried = ", ".join(str(path) for path in candidates)
    raise FileNotFoundError(
        f"Parameter file '{default_name}' was not found. Tried: {tried}. "
        "Run the corresponding tuning script first or provide its path explicitly."
    )


def load_pso_lpa_params(path: Path) -> Dict[str, Any]:
    payload = load_json(path)
    if isinstance(payload.get("params"), Mapping):
        return dict(payload["params"])

    # Compatibility with a direct best-row JSON from an older tuning script.
    allowed = {
        "w", "c1", "c2", "rho", "p", "num_particles", "max_iter",
        "lpa_max_iter", "lpa_threshold", "prior_strength", "membership_threshold",
    }
    params = {key: payload[key] for key in allowed if key in payload}
    if not params:
        raise ValueError(f"No PSO-LPA parameters found in {path}")
    return params


def load_other_params(path: Path) -> Dict[str, Dict[str, Any]]:
    payload = load_json(path)
    methods = payload.get("methods", payload)
    if not isinstance(methods, Mapping):
        raise ValueError(f"Invalid comparison parameter structure in {path}")

    parsed: Dict[str, Dict[str, Any]] = {}
    for method_name, result in methods.items():
        canonical = METHOD_ALIASES.get(str(method_name).lower(), str(method_name))
        if canonical not in METHOD_FUNCTIONS:
            continue
        if isinstance(result, Mapping) and isinstance(result.get("params"), Mapping):
            parsed[canonical] = dict(result["params"])
        elif isinstance(result, Mapping):
            parsed[canonical] = dict(result)
    return parsed


def parse_methods(value: str) -> List[str]:
    if value.strip().lower() == "all":
        return ALL_METHODS.copy()

    selected: List[str] = []
    for token in value.split(","):
        token = token.strip()
        if token.lower() in {"pso-lpa", "psolpa", "pso_lpa"}:
            canonical = "PSO-LPA"
        else:
            canonical = METHOD_ALIASES.get(token.lower(), token)
        if canonical not in ALL_METHODS:
            raise ValueError(f"Unknown method '{token}'. Valid methods: {ALL_METHODS}")
        if canonical not in selected:
            selected.append(canonical)
    return selected


def run_one_method(
    method_name: str,
    G,
    pso_params: Mapping[str, Any],
    other_params: Mapping[str, Mapping[str, Any]],
    seed: int,
    verbose: bool,
):
    if method_name == "PSO-LPA":
        communities, memberships, metadata = run_pso_lpa(
            G, pso_params, seed=seed, verbose=verbose
        )
        return communities, memberships, metadata

    if method_name not in other_params:
        raise KeyError(
            f"Best parameters for '{method_name}' are absent from the comparison JSON. "
            "Run hypeyeast_all.py with this method included."
        )
    communities, memberships = run_method(
        method_name, G, params=other_params[method_name], seed=seed
    )
    return communities, memberships, {}


def evaluate_methods(
    G,
    ground_truth,
    selected_methods: Sequence[str],
    pso_params: Mapping[str, Any],
    other_params: Mapping[str, Mapping[str, Any]],
    n_runs: int,
    base_seed: int,
    verbose_pso: bool,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    per_run_rows: List[Dict[str, Any]] = []

    for method_index, method_name in enumerate(selected_methods):
        print(f"\n=== {method_name} ===")
        for run in range(1, n_runs + 1):
            seed = base_seed + method_index * 100000 + run
            seed_everything(seed)
            row: Dict[str, Any] = {
                "Method": method_name,
                "Run": run,
                "Seed": seed,
                "Status": "ok",
                "Error": "",
            }
            try:
                communities, memberships, metadata = run_one_method(
                    method_name,
                    G,
                    pso_params,
                    other_params,
                    seed=seed,
                    verbose=verbose_pso and method_name == "PSO-LPA",
                )
                metrics = evaluate_communities(G, ground_truth, communities)
                modularity = safe_modularity(G, communities, memberships)
                row.update(metrics)
                row["Modularity"] = modularity
                row["Detected_Communities"] = len(communities)
                row["Overlapping_Nodes"] = sum(
                    1 for distribution in memberships.values() if len(distribution) > 1
                ) if memberships else 0
                for key, value in metadata.items():
                    row[f"Meta_{key}"] = value
                print(
                    f"Run {run:02d}/{n_runs}: Q={modularity:.4f}, "
                    f"ONMI={metrics['ONMI']:.4f}, Omega={metrics['Omega']:.4f}, "
                    f"CommF1={metrics['Comm_F1']:.4f}"
                )
            except Exception as exc:
                row["Status"] = "error"
                row["Error"] = f"{type(exc).__name__}: {exc}"
                for column in [
                    "P", "R", "F-Score", "ONMI", "Omega", "Comm_Precision",
                    "Comm_Recall", "Comm_F1", "Num_Communities", "Modularity",
                    "Detected_Communities", "Overlapping_Nodes",
                ]:
                    row[column] = math.nan
                print(f"Run {run:02d}/{n_runs}: ERROR - {row['Error']}")
            per_run_rows.append(row)

    per_run = pd.DataFrame(per_run_rows)
    metric_columns = [
        "P", "R", "F-Score", "ONMI", "Omega", "Comm_Precision",
        "Comm_Recall", "Comm_F1", "Modularity", "Detected_Communities",
        "Overlapping_Nodes",
    ]
    summary_rows: List[Dict[str, Any]] = []
    for method_name in selected_methods:
        subset = per_run[per_run["Method"] == method_name]
        valid = subset[subset["Status"] == "ok"]
        summary: Dict[str, Any] = {
            "Method": method_name,
            "Requested_Runs": int(len(subset)),
            "Valid_Runs": int(len(valid)),
            "Failed_Runs": int((subset["Status"] != "ok").sum()),
        }
        for column in metric_columns:
            values = pd.to_numeric(valid[column], errors="coerce") if column in valid else pd.Series(dtype=float)
            summary[f"{column}_mean"] = float(values.mean()) if values.notna().any() else math.nan
            summary[f"{column}_std"] = float(values.std(ddof=0)) if values.notna().any() else math.nan
        summary_rows.append(summary)

    summary_frame = pd.DataFrame(summary_rows)
    return per_run, summary_frame


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate PSO-LPA and comparison methods on benchmark data"
    )
    parser.add_argument(
        "--dataset", required=True,
        choices=["karate", "dolphins", "football", "yeast", "y2h"],
    )
    parser.add_argument("--data-dir", default=".")
    parser.add_argument("--graph", default=None, help="Explicit graph path")
    parser.add_argument("--gt", default=None, help="Explicit GT path; ignored for karate/dolphins")
    parser.add_argument("--pso-params", default=None, help="JSON generated by yeast.py")
    parser.add_argument("--other-params", default=None, help="JSON generated by hypeyeast_all.py")
    parser.add_argument("--methods", default="all", help="all or comma-separated method aliases")
    parser.add_argument("--n-runs", type=int, default=20)
    parser.add_argument("--base-seed", type=int, default=123)
    parser.add_argument("--verbose-pso", action="store_true")
    parser.add_argument("--out", default=None, help="Summary CSV path")
    parser.add_argument("--per-run-out", default=None, help="Per-run CSV path")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset, graph_path, gt_path = resolve_dataset_paths(
        args.dataset, args.data_dir, args.graph, args.gt
    )
    G = load_graph(dataset, graph_path)
    ground_truth = load_ground_truth(dataset, G, gt_path)
    selected_methods = parse_methods(args.methods)

    pso_file = None
    other_file = None
    pso_params: Dict[str, Any] = {}
    other_params: Dict[str, Dict[str, Any]] = {}
    if "PSO-LPA" in selected_methods:
        pso_file = _find_parameter_file(
            args.pso_params, f"best_pso_lpa_{dataset}.json", args.data_dir
        )
        pso_params = load_pso_lpa_params(pso_file)
    if any(method != "PSO-LPA" for method in selected_methods):
        other_file = _find_parameter_file(
            args.other_params, f"best_other_methods_{dataset}.json", args.data_dir
        )
        other_params = load_other_params(other_file)

    print(f"Dataset             : {dataset}")
    print(f"Graph               : {graph_path or 'NetworkX karate graph'}")
    print(f"Ground truth        : {gt_path or 'manual definition'}")
    print(f"Nodes / edges       : {G.number_of_nodes()} / {G.number_of_edges()}")
    print(f"GT communities      : {len(ground_truth)}")
    print(f"PSO-LPA parameters  : {pso_file or 'not required'}")
    print(f"Other parameters    : {other_file or 'not required'}")
    print(f"Methods             : {selected_methods}")

    per_run, summary = evaluate_methods(
        G,
        ground_truth,
        selected_methods,
        pso_params,
        other_params,
        n_runs=args.n_runs,
        base_seed=args.base_seed,
        verbose_pso=args.verbose_pso,
    )

    summary_path = Path(args.out or f"results_all_methods_{dataset}.csv").expanduser()
    per_run_path = Path(args.per_run_out or f"results_all_methods_{dataset}_per_run.csv").expanduser()
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    per_run_path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(summary_path, index=False)
    per_run.to_csv(per_run_path, index=False)

    pd.set_option("display.max_columns", None)
    print("\n=== SUMMARY ===")
    print(summary.to_string(index=False))
    print(f"\nSummary CSV : {summary_path.resolve()}")
    print(f"Per-run CSV : {per_run_path.resolve()}")


if __name__ == "__main__":
    main()
