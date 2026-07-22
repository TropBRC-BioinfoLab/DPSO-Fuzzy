# msefcd_hybrid.py

import itertools
from collections import defaultdict
from typing import List, Tuple, Dict  # <=== tambahan

import networkx as nx
import numpy as np
import pandas as pd
try:
    from cdlib.classes import NodeClustering
    from cdlib import evaluation
except ImportError:  # Fuzzy refinement tetap dapat dipakai tanpa modul evaluasi.
    NodeClustering = None
    evaluation = None


def _require_cdlib():
    if NodeClustering is None or evaluation is None:
        raise ImportError(
            "Evaluasi membutuhkan paket 'cdlib'. Install cdlib sebelum "
            "menjalankan mode eval_sup."
        )


def _normalize_rows(U, eps=1e-8):
    """Normalisasi setiap baris membership matrix tanpa menghasilkan NaN."""
    U = np.asarray(U, dtype=float)
    row_sums = U.sum(axis=1, keepdims=True)
    valid = row_sums[:, 0] > eps
    out = U.copy()
    out[valid] = out[valid] / row_sums[valid]
    return out


def _get_assignment_label(assignment, node):
    """Mendukung key node asli maupun key string setelah load dari JSON."""
    if node in assignment:
        return assignment[node]
    node_as_string = str(node)
    if node_as_string in assignment:
        return assignment[node_as_string]
    return None


def compute_fuzzy_membership_from_DPSO_MSEFCD_full(
    G,
    dPSO_results,
    r=2.0,
    iter_smooth=1,
    iter_enhance=1,
    k_comm_candidates=6,
    prior_boost=0.25,
    min_comm_size=5,
    alpha_self=0.7,
    beta_neighbor=0.3,
    eps=1e-8,
    verbose=True,
    membership_threshold=0.45,
):
    """
    Menjalankan tahap fuzzy pada Algorithm 1 DPSO-Fuzzy (langkah 9--24).

    Tahapan:
      9.  Bentuk modularity matrix B dan similarity matrix K.
      10. Inisialisasi U dari label tetangga dan prior partisi DPSO.
      11--15. Membership smoothing, node-to-community distance,
              DPSO-guided enhancement, lalu normalisasi U.
      16--24. Bentuk Omega_i, komunitas Omega, dan overlapping-node set O.

    Parameter tuning dan bentuk keluaran lama tetap dipertahankan. Parameter
    ``iter_smooth`` dan ``iter_enhance`` kini benar-benar digunakan. Jika
    nilainya berbeda, TF = max(iter_smooth, iter_enhance); masing-masing tahap
    aktif sebanyak jumlah iterasinya. Dengan nilai yang sama, setiap iterasi
    menjalankan smoothing dan enhancement seperti pseudocode.

    ``min_comm_size`` dipertahankan pada signature untuk kompatibilitas kode
    tuning lama, tetapi tidak digunakan untuk menggabungkan komunitas karena
    langkah tersebut tidak terdapat dalam pseudocode dan akan mengubah X*.
    """
    if G.number_of_nodes() == 0:
        return []
    if r <= 1.0:
        raise ValueError("Parameter fuzzifier r harus > 1.")
    if not 0.0 <= membership_threshold <= 1.0:
        raise ValueError("membership_threshold harus berada pada [0, 1].")

    nodes = list(G.nodes())
    n = len(nodes)
    node_index = {node: i for i, node in enumerate(nodes)}

    # ------------------------------------------------------------------
    # Step 9: weighted adjacency A, modularity matrix B, similarity K
    # ------------------------------------------------------------------
    A = nx.to_numpy_array(G, nodelist=nodes, weight="weight", dtype=float)
    degree = A.sum(axis=1)
    m = degree.sum() / 2.0
    if m <= eps:
        if verbose:
            print("Graph tidak mempunyai bobot edge positif.")
        return []

    B = A - np.outer(degree, degree) / (2.0 * m)
    B_positive = np.maximum(B, 0.0)

    # Similarity K mengikuti komponen yang telah digunakan pada implementasi
    # sebelumnya: weighted-neighborhood Jaccard, adjacency, dan modularity
    # affinity positif. Semua komponen dinormalisasi ke [0, 1].
    neighbor_weight = [
        {node_index[v]: float(data.get("weight", 1.0)) for v, data in G[node].items()}
        for node in nodes
    ]
    B_scale = B_positive.max()
    B_normalized = B_positive / (B_scale + eps) if B_scale > eps else B_positive

    K = np.zeros((n, n), dtype=float)
    np.fill_diagonal(K, 1.0)

    for i in range(n):
        wi = neighbor_weight[i]
        keys_i = set(wi)
        for j in range(i + 1, n):
            wj = neighbor_weight[j]
            union = keys_i | set(wj)
            if union:
                intersection_weight = sum(min(wi.get(k, 0.0), wj.get(k, 0.0)) for k in union)
                union_weight = sum(max(wi.get(k, 0.0), wj.get(k, 0.0)) for k in union)
                jaccard = intersection_weight / union_weight if union_weight > eps else 0.0
            else:
                jaccard = 0.0

            adjacency_affinity = 1.0 if A[i, j] > 0.0 else 0.0
            modularity_affinity = B_normalized[i, j]
            similarity = (
                0.45 * jaccard
                + 0.35 * adjacency_affinity
                + 0.20 * modularity_affinity
            )
            K[i, j] = K[j, i] = similarity

    K_scale = K.max()
    if K_scale > eps:
        K = K / K_scale
    np.fill_diagonal(K, 1.0)

    results_out = []

    for run_data in dPSO_results:
        run_idx = run_data.get("run", len(results_out) + 1)
        assignment = run_data.get("community_assignment")

        # Fallback untuk hasil DPSO yang hanya menyimpan best_position.
        if assignment is None and run_data.get("best_position") is not None:
            best_position = run_data["best_position"]
            if len(best_position) == n:
                assignment = dict(zip(nodes, best_position))

        if assignment is None:
            if verbose:
                print(f"Run {run_idx}: community_assignment tidak ditemukan; dilewati.")
            continue

        raw_labels = [_get_assignment_label(assignment, node) for node in nodes]
        if any(label is None for label in raw_labels):
            if verbose:
                missing = [nodes[i] for i, label in enumerate(raw_labels) if label is None]
                print(
                    f"Run {run_idx}: label {len(missing)} node tidak ditemukan; "
                    "run dilewati."
                )
            continue

        original_labels = sorted(set(raw_labels), key=lambda value: str(value))
        label_to_index = {label: cidx for cidx, label in enumerate(original_labels)}
        index_to_label = {cidx: label for label, cidx in label_to_index.items()}
        hard_labels = np.array([label_to_index[label] for label in raw_labels], dtype=int)
        c = len(original_labels)

        if c == 0:
            continue

        community_members = {
            cidx: np.where(hard_labels == cidx)[0]
            for cidx in range(c)
        }

        # ------------------------------------------------------------------
        # Step 10 / Eq. (4): neighbor-label initialization + DPSO prior
        # ------------------------------------------------------------------
        U = np.zeros((n, c), dtype=float)

        for i, node in enumerate(nodes):
            neighbor_support = np.zeros(c, dtype=float)
            for neighbor, edge_data in G[node].items():
                j = node_index[neighbor]
                edge_weight = float(edge_data.get("weight", 1.0))
                neighbor_support[hard_labels[j]] += max(edge_weight, 0.0)

            if neighbor_support.sum() > eps:
                U[i] = neighbor_support / neighbor_support.sum()

            # Prior DPSO ditambahkan ke komunitas X*_i.
            U[i, hard_labels[i]] += float(prior_boost)

            if U[i].sum() <= eps:
                U[i, hard_labels[i]] = 1.0

        U = _normalize_rows(U, eps=eps)

        # ------------------------------------------------------------------
        # Steps 11--15: TF iterations
        # ------------------------------------------------------------------
        smooth_iterations = max(0, int(iter_smooth))
        enhance_iterations = max(0, int(iter_enhance))
        TF = max(smooth_iterations, enhance_iterations)

        for t in range(TF):
            # Step 12 / Eq. (6): membership smoothing.
            if t < smooth_iterations:
                U_smoothed = np.zeros_like(U)
                for i in range(n):
                    positive_affinity = B_positive[i]
                    affinity_sum = positive_affinity.sum()

                    if affinity_sum > eps:
                        neighbor_membership = positive_affinity @ U / affinity_sum
                    else:
                        neighbor_membership = U[i]

                    U_smoothed[i] = (
                        float(alpha_self) * U[i]
                        + float(beta_neighbor) * neighbor_membership
                    )

                U = _normalize_rows(U_smoothed, eps=eps)

            # Step 13 / Eq. (7) + DPSO-guided enhancement.
            if t < enhance_iterations:
                distances = np.full((n, c), np.inf, dtype=float)
                for community_idx in range(c):
                    members = community_members[community_idx]
                    if members.size == 0:
                        continue
                    distances[:, community_idx] = 1.0 - K[:, members].mean(axis=1)

                U_enhanced = np.zeros_like(U)
                candidate_count = max(1, min(int(k_comm_candidates), c))
                distance_exponent = 2.0 / (float(r) - 1.0)

                for i in range(n):
                    finite = np.where(np.isfinite(distances[i]))[0]
                    if finite.size == 0:
                        U_enhanced[i] = U[i]
                        continue

                    candidates = finite[
                        np.argsort(distances[i, finite])[:candidate_count]
                    ]
                    candidate_distances = np.maximum(distances[i, candidates], eps)

                    zero_distance = candidates[distances[i, candidates] <= eps]
                    distance_membership = np.zeros(c, dtype=float)
                    if zero_distance.size > 0:
                        distance_membership[zero_distance] = 1.0 / zero_distance.size
                    else:
                        inverse_distance = candidate_distances ** (-distance_exponent)
                        inverse_distance /= inverse_distance.sum() + eps
                        distance_membership[candidates] = inverse_distance

                    # Enhancement mempertahankan informasi U, menguatkan
                    # komunitas yang dekat menurut K, dan memberi guidance
                    # kepada komunitas asal DPSO X*_i.
                    score = U[i] * distance_membership
                    score[hard_labels[i]] *= 1.0 + max(float(prior_boost), 0.0)

                    if score.sum() <= eps:
                        score = distance_membership
                    if score.sum() <= eps:
                        score = U[i]

                    U_enhanced[i] = score

                U = _normalize_rows(U_enhanced, eps=eps)

            # Step 14: explicit normalization after each fuzzy iteration.
            U = _normalize_rows(U, eps=eps)

        # ------------------------------------------------------------------
        # Steps 16--24: membership output, Omega_i, Omega, and O
        # ------------------------------------------------------------------
        fuzzy_membership = {}
        fuzzy_labels = {}
        node_communities = {}
        communities = {label: [] for label in original_labels}
        overlapping_nodes = {}

        for i, node in enumerate(nodes):
            row = U[i]
            if row.sum() <= eps:
                row = np.zeros(c, dtype=float)
                row[hard_labels[i]] = 1.0
            else:
                row = row / row.sum()

            membership = {
                index_to_label[cidx]: float(row[cidx])
                for cidx in range(c)
            }
            fuzzy_membership[node] = membership

            dominant_idx = int(np.argmax(row))
            dominant_label = index_to_label[dominant_idx]
            fuzzy_labels[node] = dominant_label

            omega_i = [
                index_to_label[cidx]
                for cidx in range(c)
                if row[cidx] >= membership_threshold
            ]

            # Steps 18--20: fallback argmax when threshold yields empty set.
            if not omega_i:
                omega_i = [dominant_label]

            node_communities[node] = omega_i
            for label in omega_i:
                communities[label].append(node)

            # Steps 21--23: overlapping-node set O.
            if len(omega_i) > 1:
                overlapping_nodes[node] = omega_i

        communities = {
            label: members for label, members in communities.items() if members
        }

        results_out.append(
            {
                "run": run_idx,
                "fuzzy_membership": fuzzy_membership,
                "fuzzy_labels": fuzzy_labels,
                "num_fuzzy_coms": len(communities),
                "node_communities": node_communities,
                "communities": communities,
                "overlapping_nodes": overlapping_nodes,
                "num_overlapping_nodes": len(overlapping_nodes),
                "membership_threshold": float(membership_threshold),
            }
        )

        if verbose:
            print(
                f"Run {run_idx:02d} selesai — komunitas fuzzy = "
                f"{len(communities)}, overlapping nodes = {len(overlapping_nodes)}"
            )

    if verbose:
        print("\nRingkasan (semua run):")
        for result in results_out:
            print(
                f"Run {result['run']:02d} | "
                f"komunitas fuzzy = {result['num_fuzzy_coms']} | "
                f"overlap = {result['num_overlapping_nodes']}"
            )

    return results_out

def fuzzy_to_communities(fuzzy_membership, alpha=0.45):
    """
    Convert fuzzy_membership (dict node->{label:val}) to list of communities (list of node lists).
    Fallback: node dengan membership < alpha semua → pilih label argmax.
    """
    if not fuzzy_membership:
        return []

    all_labels = sorted({lab for mem in fuzzy_membership.values() for lab in mem.keys()})
    communities = {lab: [] for lab in all_labels}

    for node, mem in fuzzy_membership.items():
        if isinstance(mem, list):
            mem = {all_labels[i]: mem[i] if i < len(mem) else 0.0 for i in range(len(all_labels))}
        assigned = [lab for lab, val in mem.items() if val >= alpha]
        if assigned:
            for lab in assigned:
                communities[lab].append(node)
        else:
            best_lab = max(mem.items(), key=lambda x: x[1])[0]
            communities[best_lab].append(node)

    return [nodes for nodes in communities.values() if len(nodes) > 0]


def detect_overlapping(fuzzy_summary, alpha=0.4):
    """
    Cetak dan kembalikan ringkasan node overlapping per run.
    """
    overlapping_summary = []
    print("\n📌 Node Overlapping per Run:")
    print("=====================================")
    for run_data in fuzzy_summary:
        run_idx = run_data["run"]
        fuzzy_membership = run_data["fuzzy_membership"]
        overlapping_nodes = {}
        for node, memberships in fuzzy_membership.items():
            coms_above_threshold = [lbl for lbl, val in memberships.items() if val >= alpha]
            if len(coms_above_threshold) > 1:
                overlapping_nodes[node] = coms_above_threshold
        overlapping_summary.append({
            "run": run_idx,
            "overlapping_nodes": overlapping_nodes,
            "num_overlapping_nodes": len(overlapping_nodes)
        })
        node_list_display = list(overlapping_nodes.keys())
        print(f"Run {run_idx:02d} — Node overlapping = {len(node_list_display)} | Nodes: {node_list_display[:5]}...")
    return overlapping_summary


def compute_overlapping_metrics(gt_communities, pred_communities):
    """
    Pairwise precision/recall/f1 berdasarkan pasangan node dalam komunitas.
    """
    gt_pairs = set()
    pred_pairs = set()
    for com in gt_communities:
        for i, j in itertools.combinations(sorted(com), 2):
            gt_pairs.add((i, j))
    for com in pred_communities:
        for i, j in itertools.combinations(sorted(com), 2):
            pred_pairs.add((i, j))
    if not gt_pairs or not pred_pairs:
        return 0.0, 0.0, 0.0
    tp = len(gt_pairs & pred_pairs)
    fp = len(pred_pairs - gt_pairs)
    fn = len(gt_pairs - pred_pairs)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    return precision, recall, f1


def search_best_alpha_for_sample(fuzzy_membership_sample, gt_comms, candidates=None):
    """
    Cari alpha terbaik (dengan GT) berdasarkan F1 pada satu sample fuzzy_membership.
    """
    if candidates is None:
        candidates = [0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55]
    best = (None, -1.0)
    for a in candidates:
        pred = fuzzy_to_communities(fuzzy_membership_sample, alpha=a)
        prec, rec, f1 = compute_overlapping_metrics(gt_comms, pred)
        if f1 > best[1]:
            best = (a, f1)
    return best


def evaluate_all(
    fuzzy_results,
    G,
    gt_name_comms,
    auto_alpha=True,
    fixed_alpha=None,
    verbose=True,
    save_csv=None
):
    """
    Evaluasi supervised (BUTUH GT):
    - Precision, Recall, F1 pairwise
    - ONMI (overlapping_normalized_mutual_information_LFK)
    """
    _require_cdlib()
    if not fuzzy_results:
        raise ValueError("fuzzy_results empty")

    print("🔎 Checking fuzzy membership consistency...")
    for i, fr in enumerate(fuzzy_results):
        fm = fr["fuzzy_membership"]
        lengths = sorted({len(v) for v in fm.values()})
        if len(lengths) != 1:
            print(f"❌ Run {i+1}: inconsistent membership length → {set(lengths)}")
        else:
            print(f"✔ Run {i+1}: membership length consistent = {lengths[0]}")

    # Tentukan alpha
    if auto_alpha:
        sample = None
        for fr in fuzzy_results:
            if fr.get("fuzzy_membership"):
                sample = fr["fuzzy_membership"]
                break
        if sample is None:
            raise ValueError("No fuzzy_membership sample found for alpha tuning.")

        all_labels = sorted({lab for mem in sample.values() for lab in mem.keys()})
        for node in list(sample.keys()):
            mem = sample[node]
            if isinstance(mem, list):
                sample[node] = {all_labels[i]: mem[i] if i < len(mem) else 0.0 for i in range(len(all_labels))}
            else:
                for lab in all_labels:
                    mem.setdefault(lab, 0.0)

        a, f1 = search_best_alpha_for_sample(sample, gt_name_comms)
        OPTIMAL_ALPHA = a if a is not None else 0.45
        if verbose:
            print(f"Auto-selected alpha = {OPTIMAL_ALPHA:.2f} (sample F1 = {f1:.4f})")
    else:
        if fixed_alpha is not None:
            OPTIMAL_ALPHA = float(fixed_alpha)
            if verbose:
                print(f"Using fixed alpha from config = {OPTIMAL_ALPHA:.2f}")
        else:
            OPTIMAL_ALPHA = 0.45
            if verbose:
                print(f"Using fixed alpha DEFAULT = {OPTIMAL_ALPHA:.2f}")

    precision_scores, recall_scores, f1_scores, onmi_scores = [], [], [], []
    rows = []
    for i, fr in enumerate(fuzzy_results):
        run_no = fr.get("run", i + 1)
        fm = fr["fuzzy_membership"]

        all_labels = sorted({lab for mem in fm.values() for lab in mem.keys()})
        for node, mem in list(fm.items()):
            if isinstance(mem, list):
                fm[node] = {
                    all_labels[j]: mem[j] if j < len(mem) else 0.0
                    for j in range(len(all_labels))
                }
            else:
                for lab in all_labels:
                    mem.setdefault(lab, 0.0)

        for node in fm:
            s = sum(fm[node].values())
            if s > 0:
                fm[node] = {lab: val / s for lab, val in fm[node].items()}

        pred_comms = fuzzy_to_communities(fm, alpha=OPTIMAL_ALPHA)

        if not pred_comms:
            prec = rec = fsc = onmi_val = 0.0
        else:
            prec, rec, fsc = compute_overlapping_metrics(gt_name_comms, pred_comms)
            gt_nc = NodeClustering(gt_name_comms, G, "GT")
            pred_nc = NodeClustering(pred_comms, G, f"Run_{run_no}")
            onmi_val = evaluation.overlapping_normalized_mutual_information_LFK(
                gt_nc, pred_nc
            ).score

        precision_scores.append(prec)
        recall_scores.append(rec)
        f1_scores.append(fsc)
        onmi_scores.append(onmi_val)

        rows.append({
            "run": run_no,
            "precision": prec,
            "recall": rec,
            "f1": fsc,
            "onmi": onmi_val,
            "num_pred_comms": len(pred_comms)
        })

        if verbose:
            print(
                f"Run {run_no}: Precision={prec:.4f}, Recall={rec:.4f}, "
                f"F1={fsc:.4f}, ONMI={onmi_val:.4f}, pred_comms={len(pred_comms)}"
            )

    df = pd.DataFrame(rows)
    if save_csv:
        df.to_csv(save_csv, index=False)

    summary = {
        "precision_mean": float(np.mean(precision_scores)),
        "precision_std": float(np.std(precision_scores)),
        "recall_mean": float(np.mean(recall_scores)),
        "recall_std": float(np.std(recall_scores)),
        "f1_mean": float(np.mean(f1_scores)),
        "f1_std": float(np.std(f1_scores)),
        "onmi_mean": float(np.mean(onmi_scores)),
        "onmi_std": float(np.std(onmi_scores)),
        "per_run": df
    }

    if verbose:
        print("\n=====================================")
        print("⭐ Rata-rata Metrik Evaluasi (All Runs) ⭐")
        print("=====================================")
        print(f"Precision : {summary['precision_mean']:.4f} ± {summary['precision_std']:.4f}")
        print(f"Recall    : {summary['recall_mean']:.4f} ± {summary['recall_std']:.4f}")
        print(f"F1-score  : {summary['f1_mean']:.4f} ± {summary['f1_std']:.4f}")
        print(f"ONMI      : {summary['onmi_mean']:.4f} ± {summary['onmi_std']:.4f}")
        print("=====================================")

    return summary


# ======================================================================
#  Tambahan: metrik overlapping khusus (ONMI, Omega,
#             Community-level Precision/Recall/F1)
#  (Overlapping_F1 tetap ada sebagai fungsi terpisah, tapi
#   TIDAK dipakai di evaluate_overlapping_all)
# ======================================================================

def to_node_clustering(
    G: nx.Graph,
    communities: List[List[str]],
    name: str = "CLUST",
) -> NodeClustering:
    """
    Helper: konversi list-of-lists -> NodeClustering (CDlib)
    """
    _require_cdlib()
    return NodeClustering(communities, G, name)


def onmi_lfk(
    G: nx.Graph,
    gt_communities: List[List[str]],
    pred_communities: List[List[str]],
) -> float:
    """
    Overlapping Normalized Mutual Information (Lancichinetti–Fortunato–Kertész).
    """
    gt_nc = to_node_clustering(G, gt_communities, "GT")
    pred_nc = to_node_clustering(G, pred_communities, "PRED")

    return evaluation.overlapping_normalized_mutual_information_LFK(
        gt_nc, pred_nc
    ).score


def omega_index(
    G: nx.Graph,
    gt_communities: List[List[str]],
    pred_communities: List[List[str]],
) -> float:
    """
    Omega Index untuk membandingkan dua clustering overlapping.

    cdlib.evaluation.omega mensyaratkan:
      - Kedua partisi mencakup himpunan node yang sama.

    Di sini kita:
      1) Ambil union node dari GT dan prediksi
      2) Untuk setiap partisi, jika ada node yang belum tercakup oleh
         komunitas manapun, kita tambahkan komunitas singleton {node}
         sehingga coverage-nya identik.
    """
    # Konversi ke himpunan untuk perhitungan coverage
    gt_sets = [set(c) for c in gt_communities if len(c) > 0]
    pred_sets = [set(c) for c in pred_communities if len(c) > 0]

    if not gt_sets or not pred_sets:
        return 0.0

    gt_nodes = set().union(*gt_sets) if gt_sets else set()
    pred_nodes = set().union(*pred_sets) if pred_sets else set()

    # Node yang ingin dievaluasi = union (bisa kamu ganti ke intersection kalau mau lebih ketat)
    union_nodes = gt_nodes | pred_nodes

    def extend_partition_to_cover_nodes(comm_sets, target_nodes):
        """
        Pastikan union komunitas = target_nodes.
        Jika ada node yang belum tercakup, tambahkan komunitas singleton.
        """
        covered = set().union(*comm_sets) if comm_sets else set()
        missing = target_nodes - covered

        extended = list(comm_sets)
        for u in missing:
            extended.append({u})
        # Kembalikan dalam bentuk list-of-lists
        return [list(s) for s in extended]

    gt_ext = extend_partition_to_cover_nodes(gt_sets, union_nodes)
    pred_ext = extend_partition_to_cover_nodes(pred_sets, union_nodes)

    gt_nc = to_node_clustering(G, gt_ext, "GT")
    pred_nc = to_node_clustering(G, pred_ext, "PRED")

    try:
        return evaluation.omega(gt_nc, pred_nc).score
    except ValueError as e:
        # fallback aman: kalau masih gagal, jangan bikin program crash
        print(f"[WARNING] Omega computation failed: {e}")
        return 0.0

"""def overlapping_f1(
    G: nx.Graph,
    gt_communities: List[List[str]],
    pred_communities: List[List[str]],
) -> float:
  
    #Overlapping F1 score ala Lancichinetti 2009.
    #Fungsi ini tetap disediakan jika ingin dipakai terpisah,
    #tetapi TIDAK digunakan dalam evaluate_overlapping_all.
  
    gt_nc = to_node_clustering(G, gt_communities, "GT")
    pred_nc = to_node_clustering(G, pred_communities, "PRED")

    return evaluation.overlapping_f1(gt_nc, pred_nc).score
"""

def community_level_prf(
    gt_communities: List[List[str]],
    pred_communities: List[List[str]],
) -> Tuple[float, float, float]:
    """
    Community-level Precision/Recall/F1 seperti yang umum dipakai
    di evaluasi protein complex detection (MCODE, ClusterONE, dll).
    """
    if not gt_communities or not pred_communities:
        return 0.0, 0.0, 0.0

    gt_sets = [set(c) for c in gt_communities if len(c) > 0]
    pred_sets = [set(c) for c in pred_communities if len(c) > 0]

    if not gt_sets or not pred_sets:
        return 0.0, 0.0, 0.0

    # Recall: rata2 best-coverage tiap GT complex
    gt_recalls = []
    for Gc in gt_sets:
        best = 0.0
        for Pc in pred_sets:
            if len(Gc) == 0:
                continue
            overlap = len(Gc & Pc) / len(Gc)
            if overlap > best:
                best = overlap
        gt_recalls.append(best)

    recall = sum(gt_recalls) / len(gt_recalls) if gt_recalls else 0.0

    # Precision: rata2 best-coverage tiap Pred complex
    pred_precs = []
    for Pc in pred_sets:
        best = 0.0
        for Gc in gt_sets:
            if len(Pc) == 0:
                continue
            overlap = len(Gc & Pc) / len(Pc)
            if overlap > best:
                best = overlap
        pred_precs.append(best)

    precision = sum(pred_precs) / len(pred_precs) if pred_precs else 0.0

    # F1
    if precision + recall > 0:
        f1 = 2 * precision * recall / (precision + recall)
    else:
        f1 = 0.0

    return precision, recall, f1


def evaluate_overlapping_all(
    G: nx.Graph,
    gt_communities: List[List[str]],
    pred_communities: List[List[str]],
) -> Dict[str, float]:
    """
    Menghitung:
      - ONMI (LFK)
      - Omega Index
      - Community-level P, R, F1

    *Tidak* lagi mengembalikan Overlapping_F1.
    """
    onmi = onmi_lfk(G, gt_communities, pred_communities)
    omega = omega_index(G, gt_communities, pred_communities)
    comm_p, comm_r, comm_f = community_level_prf(gt_communities, pred_communities)

    return {
        "ONMI_LFK": onmi,
        "Omega": omega,
        "Comm_Precision": comm_p,
        "Comm_Recall": comm_r,
        "Comm_F1": comm_f,
    }
