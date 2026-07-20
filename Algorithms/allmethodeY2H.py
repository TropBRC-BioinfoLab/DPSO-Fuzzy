import networkx as nx
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score
from cdlib import evaluation, NodeClustering
import numpy as np
import os
from collections import defaultdict # Import defaultdict
import math
import random
import urllib.request
import io
import zipfile
from scipy.io import mmread
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize
from cdlib import evaluation
from node2vec import Node2Vec
from cdlib import NodeClustering

def generate_node_embeddings(G, dimensions=64, walk_length=30, num_walks=20, window=10, min_count=1, batch_words=4):
    """
    Menghasilkan embedding node menggunakan Node2Vec.
    G akan diubah menjadi graph untuk Node2Vec.
    """

    # 1. Pra-pemrosesan Node2Vec
    # Menggunakan p=1 dan q=1 secara default, atau bisa disesuaikan
    node2vec = Node2Vec(
        G,
        dimensions=dimensions,
        walk_length=walk_length,
        num_walks=num_walks,
        workers=4,
        quiet=True
    )

    # 2. Pelatihan model Word2Vec
    # Catatan: Node2Vec menggunakan Word2Vec (dari Gensim) untuk menghasilkan embedding
    model = node2vec.fit(
        window=window,
        min_count=min_count,
        batch_words=batch_words
    )

    # 3. Ekstraksi Embedding dalam urutan node yang benar
    nodes = list(G.nodes())
    embedding_list = [model.wv[str(node)] for node in nodes]

    # Hasil A sekarang adalah matriks embedding
    A_embedding = np.array(embedding_list)

    return A_embedding, nodes

# ---------- GET GROUND TRUTH (pastikan node sebagai string) ----------

def get_ground_truth(G, gt_file_path="/home/toto/Eska/Data/Y2H_groundtruth_GO_final_MIN10_sets.txt"):
    """
    Membaca ground truth dan mengembalikan:
      1) partition_dict       -> {node: label}
      2) partition_list       -> [[komunitas 0], [komunitas 1], ...]
      3) raw_communities      -> komunitas apa adanya dari file
    """

    if not os.path.exists(gt_file_path):
        raise FileNotFoundError(f"Ground truth file tidak ditemukan: {gt_file_path}")

    raw_communities = []

    # ----- Load komunitas -----
    with open(gt_file_path) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                nodes = [str(n) for n in line.split()]
                raw_communities.append(nodes)

    print(f"Loaded {len(raw_communities)} GT communities")

    # ----- Buat mapping node → komunitas_label -----
    partition_dict = {}
    for label, comm in enumerate(raw_communities):
        for node in comm:
            if node in G:                     # hanya node yang ada di graph
                partition_dict[node] = label

    # ----- Buat list komunitas final berdasarkan node yg ada di G -----
    partition_list_dict = defaultdict(list)
    for node, lbl in partition_dict.items():
        partition_list_dict[lbl].append(node)

    # convert ke list of lists (urut label)
    partition_list = [partition_list_dict[k] for k in sorted(partition_list_dict.keys())]

    return partition_dict, partition_list, raw_communities

def compute_link_strengths(G):
    link_strengths = {}
    for u in G.nodes():
        strengths = {}
        neighbors_u = set(G.neighbors(u))
        for v in neighbors_u:
            neighbors_v = set(G.neighbors(v))
            common = neighbors_u & neighbors_v
            union = neighbors_u | neighbors_v
            # Link-density weight s_uv
            s_uv = (len(common) + 1) / (len(union) + 1)
            strengths[v] = s_uv

        total_strength = sum(strengths.values())
        if total_strength == 0:
            preferences = {v: 0 for v in strengths}
        else:
            preferences = {v: s_uv / total_strength for v, s_uv in strengths.items()}

        link_strengths[u] = preferences
    return link_strengths

def run_fuzzy_lpa(G, max_iter=50, threshold=0.05, lambd=1.0):
    labels = {u: u for u in G.nodes()}

    for _ in range(max_iter):
        changes = 0
        for u in G.nodes():
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

    community_dict = defaultdict(list)
    for node, labels_mus in memberships.items():
        for label in labels_mus:
            community_dict[label].append(node)

    # OUTPUT BARU: Kembalikan dict membership
    return dict(community_dict), dict(memberships)

def run_fuzzy_ldpa(G, max_iter=100, threshold=0.3, lambd=5.0):
    link_strengths = compute_link_strengths(G)
    labels = {u: u for u in G.nodes()}

    for _ in range(max_iter):
        changes = 0
        for u in G.nodes():
            label_scores = {}
            for v in G.neighbors(u):
                lbl = labels[v]
                pref = link_strengths[u].get(v, 0)
                label_scores[lbl] = label_scores.get(lbl, 0) + pref

            if not label_scores:
                continue

            max_score = max(label_scores.values())
            max_labels = [lbl for lbl, score in label_scores.items() if score == max_score]
            new_label = random.choice(max_labels)

            if labels[u] != new_label:
                labels[u] = new_label
                changes += 1

        if changes == 0:
            break

    # Compute fuzzy memberships
    memberships = {}
    for u in G.nodes():
        label_scores = {}
        for v in G.neighbors(u):
            lbl = labels[v]
            pref = link_strengths[u].get(v, 0)
            label_scores[lbl] = label_scores.get(lbl, 0) + pref

        total_pref = sum(label_scores.values())
        memberships[u] = {}
        for lbl, score in label_scores.items():
            gamma = score / (total_pref + 1e-9)
            mu = 1 - math.exp(-lambd * gamma)
            if mu >= threshold:
                memberships[u][lbl] = mu

    # Bentuk dict komunitas dari membership
    community_dict = defaultdict(list)
    for node, labels_mus in memberships.items():
        for label in labels_mus:
            community_dict[label].append(node)

    # Merge komunitas (Sama seperti kode Anda)
    merged_dict = merge_similar_communities(community_dict, jaccard_threshold=0.3)

    # OUTPUT BARU: Kembalikan dict membership sebelum merging (lebih akurat)
    return dict(merged_dict), dict(memberships)

def merge_similar_communities(community_dict, jaccard_threshold=0.4):
    merged = []
    seen = set()

    communities = list(community_dict.values())

    for i in range(len(communities)):
        if i in seen:
            continue
        group = set(communities[i])
        for j in range(i+1, len(communities)):
            if j in seen:
                continue
            other = set(communities[j])
            intersection = group & other
            union = group | other
            jaccard = len(intersection) / len(union)
            if jaccard >= jaccard_threshold:
                group |= other
                seen.add(j)
        merged.append(list(group))
        seen.add(i)

    # Buat kembali dict dari hasil merge
    # Setelah merging selesai
    final_dict = {i: members for i, members in enumerate(merged) if len(members) > 2}
    return final_dict

# Placeholder fungsi untuk setiap metode
# ... (rest of your functions remain the same as they were already correct)
def fuzzy_BLDLP(G, alpha=1.0, lambd=2.0, threshold=0.3, max_iter=100):
    """
    Implementasi 100% sesuai flowchart BLDLP (Jokar & Mosleh, 2019)
    Output: Fuzzy overlapping communities
    """
    nodes = list(G.nodes())

    # Step 2: Generate edge weights
    edge_weights = {}
    for u, v in G.edges():
        deg_u = G.degree[u]
        deg_v = G.degree[v]
        w = alpha * (1 / (deg_u + 1)) + (1 - alpha) * (1 / (deg_v + 1))
        edge_weights[(u, v)] = w
        edge_weights[(v, u)] = w

    # Step 3: Assign unique label ke tiap node
    labels = {node: node for node in nodes}

    def get_max_neighbor_label(node):
        label_weight = {}
        for nbr in G.neighbors(node):
            lbl = labels[nbr]
            w = edge_weights.get((node, nbr), 0)
            label_weight[lbl] = label_weight.get(lbl, 0) + w
        if not label_weight:
            return labels[node], True
        max_weight = max(label_weight.values())
        max_labels = [lbl for lbl, val in label_weight.items() if val == max_weight]
        # Jika lebih dari satu label maksimum (No), pilih acak (Choose label with highest weight - tie breaking)
        return random.choice(max_labels), len(max_labels) == 1

    def same_as_most_neighbors(node):
        neighbor_labels = [labels[nbr] for nbr in G.neighbors(node)]
        if not neighbor_labels:
            return True
        from collections import Counter
        counts = Counter(neighbor_labels)
        most_common_labels = [lbl for lbl, count in counts.items() if count == max(counts.values())]
        return labels[node] in most_common_labels

    # Step 4: Iteratif hingga kondisi stabil
    for _ in range(max_iter):
        nodes_shuffled = list(nodes)
        random.shuffle(nodes_shuffled)
        changed = False
        for node in nodes_shuffled:
            new_label, unique = get_max_neighbor_label(node)
            if new_label != labels[node]:
                labels[node] = new_label
                changed = True

        # Pengecekan kondisi berhenti kedua: "Each node label is same as that of most neighbors?"
        if all(same_as_most_neighbors(node) for node in nodes):
            break
        if not changed:
            break

    # Step 5: Hitung fuzzy membership
    memberships = {}
    for u in nodes:
        label_weights = {}
        for v in G.neighbors(u):
            lbl = labels[v]
            w = edge_weights.get((u, v), 0)
            label_weights[lbl] = label_weights.get(lbl, 0) + w

        total_w = sum(label_weights.values()) + 1e-9 # Penjumlahan bobot tetangga
        memberships[u] = {}
        for lbl, w in label_weights.items():
            gamma = w / total_w
            # Rumus membership fuzzy: mu = 1 - exp(-lambda * gamma)
            mu = 1 - math.exp(-lambd * gamma)
            if mu >= threshold:
                memberships[u][lbl] = mu

    # Step 6: Bentuk komunitas fuzzy overlapping
    fuzzy_communities = defaultdict(list)
    for node, label_mus in memberships.items():
        for lbl in label_mus:
            fuzzy_communities[lbl].append(node)

    # OUTPUT BARU: Mengembalikan dict komunitas dan dict membership
    return dict(fuzzy_communities), dict(memberships)


def fuzzy_LPA(G):
    communities_dict, memberships_dict = run_fuzzy_lpa(G)
    return communities_dict, memberships_dict

def fuzzy_LDPA(G):
    community_dict, memberships_dict = run_fuzzy_ldpa(G)
    # Tambahkan merging di sini langsung
    # merged_dict = merge_similar_communities(community_dict, jaccard_threshold=0.3)
    return dict (community_dict), dict(memberships_dict)

def CFinder(G, k=3):
    cliques = [clique for clique in nx.find_cliques(G) if len(clique) >= k]

    clique_graph = nx.Graph()
    for i, clique1 in enumerate(cliques):
        clique_graph.add_node(i)
        for j, clique2 in enumerate(cliques[i+1:], start=i+1):
            if len(set(clique1) & set(clique2)) >= k - 1:
                clique_graph.add_edge(i, j)

    communities = []
    for component in nx.connected_components(clique_graph):
        nodes = set()
        for clique_index in component:
            nodes.update(cliques[clique_index])
        communities.append(list(nodes))

    # Build community dict (filter small)
    community_dict = {i: community for i, community in enumerate(communities) if len(community) > 2}

    # Prepare memberships dict as empty dict (CFinder tidak memberikan membership real)
    memberships_pred = {}
    for cid, nodes in community_dict.items():
        for node in nodes:
            memberships_pred.setdefault(node, {})[cid] = 1.0  # hard membership 1.0 (konvensi sederhana)

    # also return list of overlapping nodes if you want (but main caller akan gunakan tuple pertama dua elemen)
    overlapping_nodes = [node for node, coms in defaultdict(list,
                        {n: [cid for cid, members in community_dict.items() if n in members] for n in G.nodes()}).items()
                         if len(coms) > 1]

    return dict(community_dict), dict(memberships_pred)

def LFK(G, alpha=0.75):
    """
    LFK algorithm: Detect overlapping communities using local fitness function.
    Returns a dictionary: {community_id: [list of node_ids]}
    """
    communities = []
    assigned_nodes = set()
    nodes = list(G.nodes())

    def fitness(G, community, alpha):
        """
        Fitness function as defined in LFK paper.
        """
        internal = 0
        external = 0
        for u in community:
            for v in G.neighbors(u):
                if v in community:
                    internal += 1
                else:
                    external += 1
        internal = internal / 2  # each internal edge counted twice
        k_in = internal
        k_out = external
        if (k_in + k_out) == 0:
            return 0
        return k_in / ((k_in + k_out) ** alpha)

    visited = set()
    for seed in nodes:
        if seed in visited:
            continue
        community = set([seed])
        improved = True
        while improved:
            improved = False
            boundary_nodes = set()
            for node in community:
                neighbors = set(G.neighbors(node))
                boundary_nodes.update(neighbors - community)
            best_fitness = fitness(G, community, alpha)
            best_node = None
            for candidate in boundary_nodes:
                temp_community = community | {candidate}
                f = fitness(G, temp_community, alpha)
                if f > best_fitness:
                    best_fitness = f
                    best_node = candidate
            if best_node:
                community.add(best_node)
                improved = True
        # Avoid duplicate communities
        is_duplicate = any(community <= other for other in communities)
        if not is_duplicate:
            communities.append(community)
            visited.update(community)

    # Convert list of sets into a dict {community_id: list of nodes}
    communities_dict = {}
    for idx, comm in enumerate(communities):
        communities_dict[idx] = list(comm)
    return communities_dict, []

def Hybrid_CMeans(G, m=1.5, epsilon=1e-5, max_iter=50, lower_thr=0.6, upper_thr=0.4, K_param=43):
    """
    Hybrid C-Means (Lei et al., 2019)
    Deteksi komunitas fuzzy dengan kombinasi local-global similarity dan teori himpunan kasar.

    Params:
        G : networkx.Graph
        m : fuzzification parameter
        epsilon : konvergensi
        max_iter : iterasi maksimum
        lower_thr : ambang lower approximation
        upper_thr : ambang upper approximation

    Returns:
        communities_dict : {community_id: [list of member nodes]}
        [] : placeholder untuk kompatibilitas
    """

    nodes = list(G.nodes())
    N = len(nodes)
    # Inisialisasi K dari parameter jika ada, jika tidak, gunakan sqrt(N)
    if K_param is not None:
         K = K_param
    else:
         K = int(math.sqrt(N))
    node_index = {node: i for i, node in enumerate(nodes)}

    # Step 4: Compute similarity matrix (menggabungkan lokal dan global)
    def compute_similarity_matrix(G):
        sim = np.zeros((N, N))
        for i in range(N):
            for j in range(N):
                if i == j:
                    sim[i, j] = 1
                else:
                    ni = set(G.neighbors(nodes[i]))
                    nj = set(G.neighbors(nodes[j]))
                    inter = len(ni & nj)
                    union = len(ni | nj)
                    jaccard = inter / union if union > 0 else 0
                    try:
                        sp = nx.shortest_path_length(G, nodes[i], nodes[j])
                        sp_sim = 1 / (1 + sp)
                    except nx.NetworkXNoPath:
                        sp_sim = 0
                    sim[i, j] = 0.5 * jaccard + 0.5 * sp_sim
        return sim

    S = compute_similarity_matrix(G)

    # Step 2: Randomly initialize central nodes (pilih node sebagai pusat komunitas)
    central_nodes = random.sample(nodes, K)

    # Inisialisasi membership matrix (U)
    U = np.zeros((N, K))
    for i in range(N):
        probs = np.random.dirichlet(np.ones(K))
        U[i, :] = probs

    # Step 3: Loop
    for iteration in range(max_iter):
        # Step 7: Compute new central nodes based on max membership contribution
        for k in range(K):
            max_contrib = -1
            best_node = central_nodes[k]
            for i in range(N):
                contrib = U[i][k] ** m * np.mean(S[i])
                if contrib > max_contrib:
                    max_contrib = contrib
                    best_node = nodes[i]
            central_nodes[k] = best_node

        # Step 4 & 5: Compute distances and update membership matrix
        D = np.zeros((N, K))
        for i in range(N):
            for k in range(K):
                idx_c = node_index[central_nodes[k]]
                D[i][k] = 1 - S[i][idx_c]  # jarak berdasarkan similarity

        U_new = np.zeros_like(U)
        for i in range(N):
            for k in range(K):
                denom = sum((D[i][k] / (D[i][j] + 1e-10)) ** (2 / (m - 1)) for j in range(K))
                U_new[i][k] = 1 / denom if denom != 0 else 0

        # Step 6: Rough set theory - lower & upper approximation
        lower_approx = defaultdict(list)
        upper_approx = defaultdict(list)
        for i, node in enumerate(nodes):
            for k in range(K):
                mu = U_new[i][k]
                if mu >= lower_thr:
                    lower_approx[k].append(node)
                elif mu >= upper_thr:
                    upper_approx[k].append(node)

        # Step 8: Check convergence
        if np.linalg.norm(U - U_new) < epsilon:
            break
        U = U_new

    # Gabungkan lower dan upper sebagai komunitas akhir
    communities_dict = defaultdict(list)
    memberships_pred = {} # <--- SIAPKAN DICTIONARY MEMBERSHIP
    for i, node in enumerate(nodes):
        memberships_pred[node] = {} # Inisialisasi

        # Ambil nilai membership terakhir
        U_final = U[i]

        for k in range(K):
            mu = U_final[k]
            # Kita menggunakan ambang batas upper_thr (atau threshold tunggal jika ada)
            if mu >= upper_thr:
                communities_dict[k].append(node)
                memberships_pred[node][k] = mu # Isi membership

    # Output BARU: Kembalikan dict membership
    return dict(communities_dict), dict(memberships_pred)

def NMG(G):
    communities_list = [] # Use a list to store community sets initially
    nodes = list(G.nodes())
    visited = set()

    for u in nodes:
        if u in visited:
            continue
        community = set([u])
        neighbors = set(G.neighbors(u))
        for v in neighbors:
            if v in visited:
                continue
            # Cek jumlah neighbor yang overlap dengan u
            shared_neighbors = neighbors & set(G.neighbors(v))
            jaccard = len(shared_neighbors) / len(neighbors | set(G.neighbors(v)) | {u, v})
            if jaccard > 0.3:  # threshold bisa diatur
                community.add(v)

        if len(community) > 1:
            communities_list.append(community)
            visited |= community

    communities_dict = {idx: list(comm) for idx, comm in enumerate(communities_list)}

    return communities_dict, []


# ---------- EVALUATE COMMUNITIES (versi lengkap: R/P/F1 + ONMI + Omega + Comm-level) ----------
def evaluate_communities(communities_dict, ground_truth_dict, G, memberships_pred=None):
    """
    communities_dict : {orig_label: [node, ...]}
    ground_truth_dict: {gt_label: [node, ...]}
    memberships_pred : optional {node: {orig_label: mu, ...}} untuk fuzzy
    """

    # ---------- 1. Normalisasi label komunitas (GT & Pred) ----------
    def normalize_labels(label_dict):
        # label apapun (int/str) dipetakan ke 0..C-1
        sorted_keys = sorted(label_dict.keys(), key=lambda x: str(x))
        mapping = {label: i for i, label in enumerate(sorted_keys)}
        normalized = {mapping[label]: list(nodes) for label, nodes in label_dict.items()}
        return normalized, mapping

    gt_normalized, gt_mapping     = normalize_labels(ground_truth_dict)
    pred_normalized, pred_mapping = normalize_labels(communities_dict)

    # node -> GT label (normalized)
    node_to_gt_label = {}
    for gt_lbl, nodes in gt_normalized.items():
        for node in nodes:
            node_to_gt_label[str(node)] = gt_lbl

    # ---------- 2. ONMI & Omega (dengan penyamaan himpunan node) ----------
    onmi_value  = 0.0
    omega_value = 0.0

    try:
        # cluster sebagai set
        gt_clusters   = [set(nodes) for nodes in gt_normalized.values()   if len(nodes) > 0]
        pred_clusters = [set(nodes) for nodes in pred_normalized.values() if len(nodes) > 0]

        # himpunan node yang ingin dicakup: pakai semua node di graf
        all_nodes = set(G.nodes())

        def complete_partition(clusters, nodes_universe):
            """
            clusters: list of set(node)
            nodes_universe: set(node) yang seharusnya dicakup
            return: list of list[node] dengan tambahan singleton untuk node yg belum tercakup
            """
            if clusters:
                covered = set().union(*clusters)
            else:
                covered = set()

            missing = nodes_universe - covered

            completed = [list(c) for c in clusters]
            for n in missing:
                completed.append([n])

            return completed

        gt_completed   = complete_partition(gt_clusters,   all_nodes)
        pred_completed = complete_partition(pred_clusters, all_nodes)

        gt_nc   = NodeClustering(gt_completed,   G, "GT")
        pred_nc = NodeClustering(pred_completed, G, "Pred")

        onmi_value  = evaluation.overlapping_normalized_mutual_information_MGH(pred_nc, gt_nc).score
        omega_value = evaluation.omega(pred_nc, gt_nc).score

        # jaga-jaga kalau keluar NaN
        if np.isnan(onmi_value):
            onmi_value = 0.0
        if np.isnan(omega_value):
            omega_value = 0.0

    except Exception:
        onmi_value  = 0.0
        omega_value = 0.0

    # ---------- 3. Node -> predicted labels (pakai membership kalau ada) ----------
    node_to_pred_labels = defaultdict(list)   # node -> list of (pred_label_norm, mu)

    if memberships_pred:
        for node, mus in memberships_pred.items():
            node = str(node)
            for orig_lbl, mu in mus.items():
                mapped_lbl = None
                if orig_lbl in pred_mapping:
                    mapped_lbl = pred_mapping[orig_lbl]
                elif str(orig_lbl) in pred_mapping:
                    mapped_lbl = pred_mapping[str(orig_lbl)]
                if mapped_lbl is not None:
                    node_to_pred_labels[node].append((mapped_lbl, float(mu)))
    else:
        # hard partition: semua mu = 1.0
        for pred_lbl, nodes in pred_normalized.items():
            for node in nodes:
                node_to_pred_labels[str(node)].append((pred_lbl, 1.0))

    # ---------- 4. Node-level R / P / F1 (single-label, hanya node yg punya prediksi) ----------
    y_true_f1, y_pred_f1 = [], []

    for node in G.nodes():
        node_s = str(node)
        gt_label = node_to_gt_label.get(node_s, -1)

        if gt_label == -1:
            # node tidak punya label GT → skip di tahap filter
            y_true_f1.append(-1)
            y_pred_f1.append(-2)
            continue

        pred_list = node_to_pred_labels.get(node_s, [])
        if not pred_list:
            # TIDAK diberi prediksi komunitas → kita anggap unassigned
            y_true_f1.append(gt_label)
            y_pred_f1.append(-2)
            continue

        # pilih label prediksi yang paling “cocok” dengan komunitas GT node tsb
        gt_nodes_set = set(gt_normalized[gt_label])
        best_pred  = -2
        best_score = -1.0

        for pred_lbl, mu in pred_list:
            pred_nodes = set(pred_normalized.get(pred_lbl, []))
            overlap = len(pred_nodes & gt_nodes_set)
            score   = overlap + 1e-4 * mu  # tie-breaking dengan mu
            if score > best_score:
                best_score = score
                best_pred  = pred_lbl

        y_true_f1.append(gt_label)
        y_pred_f1.append(best_pred)

    # filter hanya node yang:
    #   - punya GT (gt != -1)
    #   - DAN punya prediksi (pred != -2)
    valid_indices   = [i for i, (gt, pr) in enumerate(zip(y_true_f1, y_pred_f1)) if gt != -1 and pr != -2]
    y_true_filtered = [y_true_f1[i] for i in valid_indices]
    y_pred_filtered = [y_pred_f1[i] for i in valid_indices]

    r = p = f = 0.0
    if len(y_true_filtered) > 0:
        r = recall_score(y_true_filtered, y_pred_filtered, average='macro', zero_division=0)
        p = precision_score(y_true_filtered, y_pred_filtered, average='macro', zero_division=0)
        f = f1_score(y_true_filtered, y_pred_filtered, average='macro', zero_division=0)

    # ---------- 5. Community-level Precision/Recall/F1 ----------
    def set_f1(a, b):
        a, b = set(a), set(b)
        inter = len(a & b)
        if inter == 0:
            return 0.0
        return 2.0 * inter / (len(a) + len(b))

    gt_clusters   = [set(nodes) for nodes in gt_normalized.values()   if len(nodes) > 0]
    pred_clusters = [set(nodes) for nodes in pred_normalized.values() if len(nodes) > 0]

    comm_precision = comm_recall = comm_f1 = 0.0

    if gt_clusters and pred_clusters:
        # Precision komunitas: rata-rata best F1 tiap komunitas prediksi terhadap semua GT
        best_f1_pred = []
        for P in pred_clusters:
            best = 0.0
            for T in gt_clusters:
                f1_pt = set_f1(P, T)
                if f1_pt > best:
                    best = f1_pt
            best_f1_pred.append(best)
        comm_precision = float(np.mean(best_f1_pred)) if best_f1_pred else 0.0

        # Recall komunitas: rata-rata best F1 tiap komunitas GT terhadap semua prediksi
        best_f1_gt = []
        for T in gt_clusters:
            best = 0.0
            for P in pred_clusters:
                f1_tp = set_f1(T, P)
                if f1_tp > best:
                    best = f1_tp
            best_f1_gt.append(best)
        comm_recall = float(np.mean(best_f1_gt)) if best_f1_gt else 0.0

        if comm_precision + comm_recall > 0:
            comm_f1 = 2.0 * comm_precision * comm_recall / (comm_precision + comm_recall)

    # ---------- 6. Return semua metrik ----------
    return {
        'R': r,
        'P': p,
        'F-Score': f,
        'ONMI': onmi_value,
        'Omega': omega_value,
        'Comm_Precision': comm_precision,
        'Comm_Recall': comm_recall,
        'Comm_F1': comm_f1
    }

# ---------- USAGE DI BAGIAN EKSEKUSI ----------
# -------- Load Data --------
file_path = "/home/toto/Eska/Data/Y2H_reconciled_full.edgelist"
G = nx.read_edgelist(file_path)
mapping = {node: idx for idx, node in enumerate(G.nodes())}
assert G is not None and G.number_of_nodes() > 0, "G tidak valid – jalankan SEL 1 dulu!"
print(f"✅ Graph loaded: Nodes = {G.number_of_nodes()}, Edges = {G.number_of_edges()}")

# Ambil ground truth (pastikan GT file node names cocok; kami pakai str)
ground_truth_partition_dict, ground_truth_gt_partition, _ = get_ground_truth(G)

# Convert to {label: [nodes]} format (nodes already strings)
ground_truth = defaultdict(list)
for node, lbl in ground_truth_partition_dict.items():
    ground_truth[lbl].append(node)

print("Jumlah komunitas ground truth:", len(ground_truth))
print("Contoh ground truth komunitas:")
for lbl, members in list(ground_truth.items())[:2]:
    print(f"Komunitas {lbl}: {members[:5]} ... (total {len(members)})")

# Pastikan Node2Vec menggunakan node strings (generate_node_embeddings Anda sudah menggunakan str(node) untuk lookup)
A_embedding, nodes_order = generate_node_embeddings(G, dimensions=64)
print(f"✅ Embedding berhasil dibuat. Bentuk Matriks Fitur: {A_embedding.shape}")

# Daftar metode
# ---------- DAFTAR METODE YANG DIEVALUASI ----------
methods = {
    "fuzzy BLDLP":      fuzzy_BLDLP,
    "fuzzy LPA":        fuzzy_LPA,
    "fuzzy LDPA":       fuzzy_LDPA,
    "CFinder":          lambda G: CFinder(G, k=3),
    "LFK":              LFK,
    "Hybrid C-Means":   lambda G: Hybrid_CMeans(G, K_param=43),
    "NMG":              NMG,
}

results = []
n_runs = 20  # jumlah pengulangan

print("\n=== MULAI EVALUASI PERULANGAN ===")

for method_name, method_func in methods.items():

    R_list, P_list, F_list, ONMI_list = [], [], [], []
    Omega_list = []
    CommP_list, CommR_list, CommF_list = [], [], []

    for run in range(n_runs):
        try:
            result = method_func(G)

            communities_dict = {}
            memberships_pred = None  # default

            # --- Standarisasi output berbagai metode ---
            if isinstance(result, tuple):
                # (communities_dict, membership_dict/other)
                communities_dict = result[0]
                if len(result) > 1 and isinstance(result[1], dict):
                    memberships_pred = result[1]
            elif isinstance(result, dict):
                communities_dict = result
            elif isinstance(result, list):
                communities_dict = {i: comm for i, comm in enumerate(result) if len(comm) > 2}
            else:
                raise TypeError(f"Output {method_name} tidak valid: {type(result)}")

            # Filter komunitas terlalu kecil (mis: ukuran <=2)
            communities_dict = {k: v for k, v in communities_dict.items() if len(v) > 2}

            if communities_dict:
                metrics = evaluate_communities(communities_dict, ground_truth, G, memberships_pred)

                R_list.append(metrics['R'])
                P_list.append(metrics['P'])
                F_list.append(metrics['F-Score'])
                ONMI_list.append(metrics['ONMI'])
                Omega_list.append(metrics['Omega'])
                CommP_list.append(metrics['Comm_Precision'])
                CommR_list.append(metrics['Comm_Recall'])
                CommF_list.append(metrics['Comm_F1'])
            else:
                # tidak ada komunitas terdeteksi
                R_list.append(0.0)
                P_list.append(0.0)
                F_list.append(0.0)
                ONMI_list.append(0.0)
                Omega_list.append(0.0)
                CommP_list.append(0.0)
                CommR_list.append(0.0)
                CommF_list.append(0.0)

        except Exception as e:
            # kalau error di satu run, treat sebagai 0
            # print(f"❌ Error di {method_name} (run {run+1}): {e}")
            R_list.append(0.0)
            P_list.append(0.0)
            F_list.append(0.0)
            ONMI_list.append(0.0)
            Omega_list.append(0.0)
            CommP_list.append(0.0)
            CommR_list.append(0.0)
            CommF_list.append(0.0)
            continue

    # ---- Hitung rata-rata & standar deviasi untuk SEMUA metrik ----
    mean_R,  std_R  = np.mean(R_list),     np.std(R_list)
    mean_P,  std_P  = np.mean(P_list),     np.std(P_list)
    mean_F,  std_F  = np.mean(F_list),     np.std(F_list)
    mean_ON, std_ON = np.mean(ONMI_list),  np.std(ONMI_list)
    mean_OM, std_OM = np.mean(Omega_list), np.std(Omega_list)

    mean_CP, std_CP = np.mean(CommP_list), np.std(CommP_list)
    mean_CR, std_CR = np.mean(CommR_list), np.std(CommR_list)
    mean_CF, std_CF = np.mean(CommF_list), np.std(CommF_list)

    results.append({
        "Method": method_name,
        "R_mean":       round(mean_R, 4),
        "P_mean":       round(mean_P, 4),
        "F_mean":       round(mean_F, 4),
        "ONMI_mean":    round(mean_ON, 4),
        "Omega_mean":   round(mean_OM, 4),
        "CommP_mean":   round(mean_CP, 4),
        "CommR_mean":   round(mean_CR, 4),
        "CommF_mean":   round(mean_CF, 4),

        "R_std":        round(std_R, 4),
        "P_std":        round(std_P, 4),
        "F_std":        round(std_F, 4),
        "ONMI_std":     round(std_ON, 4),
        "Omega_std":    round(std_OM, 4),
        "CommP_std":    round(std_CP, 4),
        "CommR_std":    round(std_CR, 4),
        "CommF_std":    round(std_CF, 4),
    })

# Buat dataframe hasil akhir
results_df = pd.DataFrame(results)

# Cetak tabel hasil
print("\n=== HASIL RATA-RATA EVALUASI (", n_runs, "RUNS ) ===")
print(results_df.to_string(index=False))

# Simpan hasil ke file
out_path = "/home/toto/Eska/result_fuzzy_methods_y2h.csv"
results_df.to_csv(out_path, index=False)
print(f"\n✅ Hasil evaluasi disimpan ke: {out_path}")

