import networkx as nx
import random, math
import numpy as np
from collections import defaultdict
from itertools import product
from networkx.algorithms.community import modularity
from sklearn.preprocessing import normalize
import itertools
from networkx.algorithms.community.quality import modularity
import os
import io
from scipy.io import mmread
import urllib.request
import zipfile
from scipy.io import mmread
import time

def load_mips_graph(path):
    # 1. Muat graf, Node ID sudah berupa nama protein/string (misal 'YHR023W')
    G_raw = nx.read_edgelist(path, delimiter=None, data=False, comments='#', nodetype=str)
    
    G_raw = G_raw.to_undirected()
    G_raw.remove_edges_from(nx.selfloop_edges(G_raw))
    
    # 2. HAPUS mapping = {old: i for i, old in enumerate(G_raw.nodes())}
    # 3. HAPUS G = nx.relabel_nodes(G_raw, mapping)
    # 4. HAPUS nx.set_node_attributes(G, {i: old for old, i in mapping.items()}, "protein")
    
    # Kunci node G_raw sekarang sudah berupa nama protein
    G = G_raw
    
    # Kembalikan G dan mapping kosong/None (karena tidak dipakai)
    return G, None # G sekarang berisi nama protein

# --- BAGIAN EKSEKUSI ---
txt_path = os.path.join("/home/toto/Eska/Data", "Yeast_D2.txt")
G, mapping = load_mips_graph(txt_path)

# 5. Baris yang Anda panggil sekarang hanya memastikan tipe data string (jika diperlukan)
G = nx.relabel_nodes(G, lambda x: str(x)) # Baris ini sekarang memastikan 'YHR023W' adalah string

# Hasil: list(G.nodes()) akan menghasilkan ['YHR023W', 'YLR274W', ...]
# Jumlah node dan edge
num_nodes = G.number_of_nodes()
num_edges = G.number_of_edges()
print(f"⚽ Total Nodes : {num_nodes}")
print(f"⚽ Total Edges : {num_edges}") 

# 1. Fuzzy BLDLP
def fuzzy_BLDLP(G, alpha=0.5, lambd=2.0, threshold=0.4, max_iter=100):
    """
    Implementasi 100% sesuai flowchart BLDLP (Jokar & Mosleh, 2019)
    Input: Graph G(V,E), balancing parameter α
    Output: Fuzzy overlapping communities + disjoint communities
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

    # Step 3: Assign unique label
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
        if len(max_labels) == 1:
            return max_labels[0], True
        else:
            return random.choice(max_labels), False

    def same_as_most_neighbors(node):
        neighbor_labels = [labels[nbr] for nbr in G.neighbors(node)]
        if not neighbor_labels:
            return True
        most_common = max(set(neighbor_labels), key=neighbor_labels.count)
        return labels[node] == most_common

    # Step 4: Iteratif hingga stabil
    for _ in range(int(max_iter)):
        nodes_shuffled = list(nodes)
        random.shuffle(nodes_shuffled)
        changed = False
        for node in nodes_shuffled:
            new_label, unique = get_max_neighbor_label(node)
            if new_label != labels[node]:
                labels[node] = new_label
                changed = True
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
        total_w = sum(label_weights.values()) + 1e-9
        memberships[u] = {}
        for lbl, w in label_weights.items():
            gamma = w / total_w
            x = -lambd * gamma
            x = max(-700, min(700, x))
            mu = max(0.0, min(1.0, 1 - math.exp(x)))
            if mu >= threshold:
                memberships[u][lbl] = mu

        total_mu = sum(memberships[u].values()) + 1e-9
        for lbl in memberships[u]:
            memberships[u][lbl] /= total_mu

    # Step 6: Bentuk fuzzy communities
    fuzzy_communities = defaultdict(list)
    for node, label_mus in memberships.items():
        for lbl in label_mus:
            fuzzy_communities[lbl].append(node)

    # Step 7: Bentuk disjoint communities (hard assignment)
    hard_assignment = defaultdict(list)
    for node, label_mus in memberships.items():
        if label_mus:
            best_label = max(label_mus.items(), key=lambda x: x[1])[0]
            hard_assignment[best_label].append(node)
        else:
            hard_assignment[random.choice(list(fuzzy_communities.keys()))].append(node)

    return dict(fuzzy_communities), list(hard_assignment.values())


# ===============================
# GRID SEARCH PARAMETER TERBAIK
# ===============================
G, mapping = load_mips_graph(txt_path)
# Jumlah node dan edge
num_nodes = G.number_of_nodes()
num_edges = G.number_of_edges()
print(f"⚽ Total Nodes : {num_nodes}")
print(f"⚽ Total Edges : {num_edges}")

alpha_values = [0.5, 1.0, 2.0, 4.0, 6.0]
max_iter_values = [50, 100, 200]
threshold_values = [0.3, 0.4, 0.5]
lambd_values = [1.0, 1.5, 2.0]

best_params = None
best_modularity = -1
results = []

for alpha, max_iter, threshold, lambd in product(alpha_values, max_iter_values, threshold_values, lambd_values):
    fuzzy_coms, disjoint_coms = fuzzy_BLDLP(G, alpha, lambd, threshold, max_iter)
    try:
        if not disjoint_coms:
            raise ValueError("Komunitas kosong")
        mod = modularity(G, disjoint_coms)
    except Exception:
        mod = -1
    results.append((mod, alpha, max_iter, threshold, lambd))
    if mod > best_modularity:
        best_modularity = mod
        best_params = (alpha, max_iter, threshold, lambd)



print("=== HASIL PARAMETER TERBAIK ===")
if best_params is None:
    print("❌ Tidak ada parameter valid (semua kombinasi gagal menghasilkan komunitas).")
else:
    print(f"Alpha     : {best_params[0]}")
    print(f"Max_iter  : {best_params[1]}")
    print(f"Threshold : {best_params[2]}")
    print(f"Lambda    : {best_params[3]}")
    print(f"Modularity: {best_modularity:.4f}")
    
# 2. Fuzzy LPA
def run_fuzzy_lpa(G, max_iter=150, threshold=0.01, lambd=5.0):
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

    return memberships


# =============================
# Hyperparameter tuning section
# =============================

def get_communities_disjoint(memberships):
    node_to_best = {}
    for node, labels in memberships.items():
        if not labels:
            continue
        best_label = max(labels.items(), key=lambda x: x[1])[0]
        node_to_best[node] = best_label

    communities = {}
    for node, lbl in node_to_best.items():
        communities.setdefault(lbl, []).append(node)
    return list(communities.values())

def compute_modularity(G, communities):
    if not communities:
        return 0.0
    return nx.algorithms.community.modularity(G, communities)


def tune_fuzzy_lpa(G):
    max_iter_list = [50, 100, 150]
    threshold_list = [0.01, 0.05, 0.1]
    lambd_list = [1.0, 2.0, 5.0]

    best_params = None
    best_modularity = -1

    for max_iter, threshold, lambd in product(max_iter_list, threshold_list, lambd_list):
        memberships = run_fuzzy_lpa(G, max_iter=max_iter, threshold=threshold, lambd=lambd)
        communities = get_communities_disjoint(memberships)
        modularity = compute_modularity(G, communities)

        #print(f"max_iter={max_iter}, threshold={threshold}, lambda={lambd} => modularity={modularity:.4f}")

        if modularity > best_modularity:
            best_modularity = modularity
            best_params = (max_iter, threshold, lambd)

    print("=== PARAMETER TERBAIK FUZZY LPA ===")
    print(f"max_iter  : {best_params[0]}")
    print(f"threshold : {best_params[1]}")
    print(f"lambda    : {best_params[2]}")
    print(f"modularity: {best_modularity:.4f}")

    return best_params, best_modularity

# Pemanggilan
if __name__ == "__main__":
    G, mapping = load_mips_graph(txt_path)
    tune_fuzzy_lpa(G)
  
# 3. Fuzzy LDPA
# ==========================================
# 1. Fungsi Link Strengths (Link Density)
# ==========================================
def compute_link_strengths(G):
    link_strengths = {}
    for u in G.nodes():
        strengths = {}
        neighbors_u = set(G.neighbors(u))
        for v in neighbors_u:
            neighbors_v = set(G.neighbors(v))
            common = neighbors_u & neighbors_v
            union = neighbors_u | neighbors_v
            s_uv = (len(common) + 1) / (len(union) + 1)  # Link-density weight
            strengths[v] = s_uv

        total_strength = sum(strengths.values())
        if total_strength == 0:
            preferences = {v: 0 for v in strengths}
        else:
            preferences = {v: s_uv / total_strength for v, s_uv in strengths.items()}
        link_strengths[u] = preferences
    return link_strengths

# ==========================================
# 2. Fuzzy LDPA
# ==========================================
def run_fuzzy_ldpa(G, max_iter=150, threshold=0.75, lambd=2.5):
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

    return memberships

# ==========================================
# 3. Konversi ke komunitas disjoint
# ==========================================
def get_communities_disjoint(memberships):
    node_to_best = {}
    for node, labels in memberships.items():
        if not labels:
            continue
        best_label = max(labels.items(), key=lambda x: x[1])[0]
        node_to_best[node] = best_label

    communities = defaultdict(list)
    for node, lbl in node_to_best.items():
        communities[lbl].append(node)
    return list(communities.values())

# ==========================================
# 4. Hitung modularity (NetworkX)
# ==========================================
def compute_modularity(G, communities):
    try:
        from networkx.algorithms.community import modularity
        return modularity(G, communities)
    except Exception:
        return -1

# ==========================================
# 5. Grid Search Hyperparameter
# ==========================================
def tune_fuzzy_ldpa(G):
    max_iter_values = [100, 150, 200]
    threshold_values = [0.2, 0.25, 0.3, 0.35]
    lambd_values = [1.0, 2.5, 5.0, 10.0]

    best_mod = -1
    best_params = None

    for max_iter, threshold, lambd in product(max_iter_values, threshold_values, lambd_values):
        memberships = run_fuzzy_ldpa(G, max_iter=max_iter, threshold=threshold, lambd=lambd)
        communities = get_communities_disjoint(memberships)
        if len(communities) <= 1:
            mod = -1
        else:
            mod = compute_modularity(G, communities)
        #print(f"Tes param: iter={max_iter}, thr={threshold}, λ={lambd} → mod={mod:.4f}")

        if mod > best_mod:
            best_mod = mod
            best_params = (max_iter, threshold, lambd)

    print("=== PARAMETER TERBAIK FUZZY LDPA ===")
    if best_params:
        print(f"max_iter  : {best_params[0]}")
        print(f"threshold : {best_params[1]}")
        print(f"lambda    : {best_params[2]}")
        print(f"modularity: {best_mod:.4f}")
    else:
        print("⚠️ Tidak ada kombinasi parameter yang menghasilkan komunitas valid.")

# ==========================================
# 6. Jalankan contoh
# ==========================================
if __name__ == "__main__":
    G, mapping = load_mips_graph(txt_path)
    tune_fuzzy_ldpa(G)

# 4. Fungsi Fuzzy C-Means
def fuzzy_CMeans(G, k=2, m=1.5, max_iter=100, epsilon=1e-5, threshold=0.5):
    nodes = list(G.nodes())
    n = len(nodes)
    node_idx = {node: i for i, node in enumerate(nodes)}
    A = nx.to_numpy_array(G, nodelist=nodes)
    A = normalize(A, axis=1)

    U = np.random.dirichlet(np.ones(k), size=n)

    for iteration in range(max_iter):
        centroids = []
        for j in range(k):
            numerator = np.sum((U[:, j] ** m)[:, np.newaxis] * A, axis=0)
            denominator = np.sum(U[:, j] ** m)
            centroids.append(numerator / (denominator + 1e-9))
        centroids = np.array(centroids)

        dist = np.zeros((n, k))
        for i in range(n):
            for j in range(k):
                dist[i, j] = np.linalg.norm(A[i] - centroids[j]) + 1e-9

        new_U = np.zeros((n, k))
        for i in range(n):
            for j in range(k):
                denom = np.sum([(dist[i, j] / dist[i, l]) ** (2 / (m - 1)) for l in range(k)])
                new_U[i, j] = 1.0 / denom

        if np.linalg.norm(new_U - U) < epsilon:
            break
        U = new_U

    communities = [[] for _ in range(k)]
    overlapping_nodes = []
    for i, node in enumerate(nodes):
        memberships = U[i]
        strong_affiliations = [j for j, val in enumerate(memberships) if val >= threshold]
        if len(strong_affiliations) > 1:
            overlapping_nodes.append(node)
        for j in strong_affiliations:
            communities[j].append(node)

    return communities, overlapping_nodes, U


# ============================================================
# Fungsi evaluasi & tuning hyperparameter
# ============================================================
def evaluate_modularity(G, communities, U):
    """
    Evaluasi modularitas, ubah komunitas fuzzy → hard partition (disjoint)
    """
    nodes = list(G.nodes())
    n = len(nodes)

    # Jika U tidak disediakan, abaikan
    if U is None:
        comms = [set(c) for c in communities if len(c) > 0]
        return modularity(G, comms)

    # Ambil komunitas dominan tiap node (hard partition)
    hard_assign = np.argmax(U, axis=1)
    k = np.max(hard_assign) + 1
    comms = [set() for _ in range(k)]
    for i, node in enumerate(nodes):
        comms[hard_assign[i]].add(node)

    comms = [c for c in comms if len(c) > 0]
    if len(comms) <= 1:
        return -1
    return modularity(G, comms)


def tune_fuzzy_CMeans(G):
    k_values = [100, 150, 180, 200]
    m_values = [1.3, 1.5, 1.7, 2.0]
    thresholds = [0.2, 0.3, 0.4, 0.5]
    max_iters = [50, 100, 150]
    epsilon_values = [1e-4, 1e-5, 1e-6]

    best_mod = -1
    best_params = None
    results = []

    # Grid Search: Kombinasi semua parameter
    param_combinations = itertools.product(
        k_values, m_values, thresholds, max_iters, epsilon_values
    )

    total_combinations = len(k_values) * len(m_values) * len(thresholds) * len(max_iters) * len(epsilon_values)
    print(f"Memulai tuning dengan {total_combinations} kombinasi...")

    for k, m, threshold, max_iter, epsilon in param_combinations:
        try:
            # Panggil fungsi fuzzy_CMeans, sekarang menyertakan epsilon
            communities, overlapping_nodes, U = fuzzy_CMeans(
                G, k=k, m=m, max_iter=max_iter, epsilon=epsilon, threshold=threshold
            )
            mod = evaluate_modularity(G, communities, U)

            # Catat hasil
            results.append((mod, k, m, threshold, max_iter, epsilon))

            if mod > best_mod:
                best_mod = mod
                best_params = (k, m, threshold, max_iter, epsilon)

        except Exception as e:
            # Jika terjadi error saat eksekusi, lewati kombinasi ini
            # print(f"⚠️ Error di k={k}, m={m}, threshold={threshold}, iter={max_iter}, eps={epsilon}: {e}")
            results.append((-1, k, m, threshold, max_iter, epsilon))
            continue

    print("=== PARAMETER TERBAIK FUZZY C-MEANS ===")
    if best_params:
        print(f"k          : {best_params[0]}")
        print(f"m          : {best_params[1]}")
        print(f"threshold  : {best_params[2]}")
        print(f"max_iter   : {best_params[3]}")
        print(f"epsilon    : {best_params[4]}")
        print(f"Modularity : {best_mod:.4f}")
    else:
        print("Tidak ada kombinasi parameter yang valid.")

    return best_params, best_mod, results

# ============================================================
# Contoh penggunaan
# ============================================================
if __name__ == "__main__":
    G, mapping = load_mips_graph(txt_path)
    tune_fuzzy_CMeans(G)

# 5. CFinder

def CFinder(G, k=3):
    # Temukan semua k-clique
    cliques = [clique for clique in nx.find_cliques(G) if len(clique) >= k]

    # Buat graph antar clique jika mereka memiliki k-1 node yang sama
    clique_graph = nx.Graph()
    for i, clique1 in enumerate(cliques):
        clique_graph.add_node(i)
        for j, clique2 in enumerate(cliques[i + 1:], start=i + 1):
            if len(set(clique1) & set(clique2)) >= k - 1:
                clique_graph.add_edge(i, j)

    # Temukan komponen terhubung di clique graph sebagai komunitas
    communities = []
    for component in nx.connected_components(clique_graph):
        nodes = set()
        for clique_index in component:
            nodes.update(cliques[clique_index])
        communities.append(list(nodes))

    # Identifikasi node overlapping
    node_to_communities = defaultdict(list)
    for idx, community in enumerate(communities):
        for node in community:
            node_to_communities[node].append(idx)

    overlapping_nodes = [node for node, coms in node_to_communities.items() if len(coms) > 1]

    # Format dictionary untuk kompatibilitas
    community_dict = {i: community for i, community in enumerate(communities) if len(community) > 2}

    return community_dict, overlapping_nodes, []


# ============================================================
# Evaluasi modularitas (disjoint sementara)
# ============================================================
def evaluate_modularity(G, community_dict):
    # Konversi ke bentuk partisi disjoint (node hanya masuk 1 komunitas)
    assigned = set()
    hard_partition = []
    for community in community_dict.values():
        comm = set([n for n in community if n not in assigned])
        if len(comm) > 0:
            hard_partition.append(comm)
            assigned |= comm

    # Sisa node tanpa komunitas (isolated atau tidak masuk clique)
    unassigned = set(G.nodes()) - assigned
    if unassigned:
        hard_partition.append(unassigned)

    if len(hard_partition) <= 1:
        return -1  # modularitas tidak relevan

    return modularity(G, hard_partition)


# ============================================================
# Hyperparameter tuning CFinder
# ============================================================
def tune_CFinder(G):
    k_values = range(3, 100)  # biasanya 3-6 cukup
    best_mod = -1
    best_k = None

    for k in k_values:
        community_dict, overlapping_nodes, _ = CFinder(G, k=k)
        mod = evaluate_modularity(G, community_dict)
        print(f"k={k} → modularity={mod:.4f}, communities={len(community_dict)}, overlapping_nodes={len(overlapping_nodes)}")

        if mod > best_mod:
            best_mod = mod
            best_k = k

    print("\n=== PARAMETER TERBAIK CFINDER ===")
    if best_k is not None:
        print(f"k          : {best_k}")
        print(f"modularity : {best_mod:.4f}")
    else:
        print("Tidak ada kombinasi parameter yang valid.")


# ============================================================
# Contoh penggunaan
# ============================================================
if __name__ == "__main__":
    G, mapping = load_mips_graph(txt_path)
    tune_CFinder(G)

# 6. LFK
def LFK(G, alpha=1.0):
    """
    LFK algorithm: Detect overlapping communities using local fitness function.
    Returns a dictionary: {community_id: [list of node_ids]}, [] for overlapping nodes (belum dihitung)
    """
    communities = []
    assigned_nodes = set()
    nodes = list(G.nodes())

    def fitness(G, community, alpha):
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
        # Avoid duplicate or subset communities
        is_duplicate = any(community <= other for other in communities)
        if not is_duplicate:
            communities.append(community)
            visited.update(community)

    communities_dict = {i: list(comm) for i, comm in enumerate(communities)}
    return communities_dict, []


# ============================================================
# Evaluasi modularitas (disjoint sementara)
# ============================================================
def evaluate_modularity(G, community_dict):
    assigned = set()
    hard_partition = []
    for community in community_dict.values():
        comm = set([n for n in community if n not in assigned])
        if len(comm) > 0:
            hard_partition.append(comm)
            assigned |= comm

    unassigned = set(G.nodes()) - assigned
    if unassigned:
        hard_partition.append(unassigned)

    if len(hard_partition) <= 1:
        return -1  # modularitas tidak relevan

    return modularity(G, hard_partition)


# ============================================================
# Hyperparameter tuning LFK
# ============================================================
def tune_LFK(G):
    alpha_values = np.linspace(0.5, 2.5, 9)  # coba alpha 0.5, 0.75, 1.0, ..., 2.5
    best_alpha = None
    best_mod = -1

    print("Mulai tuning hyperparameter LFK...\n")
    for alpha in alpha_values:
        community_dict, _ = LFK(G, alpha=alpha)
        mod = evaluate_modularity(G, community_dict)
        #print(f"alpha={alpha:.2f} → modularity={mod:.4f}, communities={len(community_dict)}")

        if mod > best_mod:
            best_mod = mod
            best_alpha = alpha

    print("=== PARAMETER TERBAIK LFK ===")
    if best_alpha is not None:
        print(f"alpha       : {best_alpha:.2f}")
        print(f"modularity  : {best_mod:.4f}")
    else:
        print("Tidak ada kombinasi parameter yang valid.")


# ============================================================
# Contoh penggunaan
# ============================================================
if __name__ == "__main__":
    G, mapping = load_mips_graph(txt_path)
    tune_LFK(G)

# 7. HYBRID C-MEANS
def Hybrid_CMeans(G, m=2.0, epsilon=1e-5, max_iter=100, lower_thr=0.6, upper_thr=0.4, K_param=None):
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
        K_param : Jumlah komunitas (K). Jika None, dihitung sebagai max(2, int(sqrt(N)))

    Returns:
        communities_dict : {community_id: [list of member nodes]}
        U : np.ndarray (N x K) - Final membership matrix
    """
    nodes = list(G.nodes())
    N = len(nodes)

    # Inisialisasi K dari parameter jika ada, jika tidak, gunakan default
    if K_param is not None:
         K = K_param
    else:
         # Minimal 2 komunitas
         K = max(2, int(math.sqrt(N)))

    # Guardrail: K tidak boleh lebih besar dari N
    if K > N:
        K = N

    node_index = {node: i for i, node in enumerate(nodes)}

    # Step 4: Compute similarity matrix (menggabungkan lokal dan global)
    def compute_similarity_matrix(G):
        sim = np.zeros((N, N))
        for i in range(N):
            for j in range(i, N): # Hanya hitung segitiga atas
                if i == j:
                    sim[i, j] = 1
                else:
                    ni = set(G.neighbors(nodes[i]))
                    nj = set(G.neighbors(nodes[j]))
                    inter = len(ni & nj)
                    union = len(ni | nj)
                    jaccard = inter / union if union > 0 else 0
                    try:
                        # Hanya hitung jika ada jalur (lebih efisien)
                        if nodes[i] in G and nodes[j] in G and nx.has_path(G, nodes[i], nodes[j]):
                             sp = nx.shortest_path_length(G, nodes[i], nodes[j])
                             sp_sim = 1 / (1 + sp)
                        else:
                             sp_sim = 0
                    except nx.NetworkXNoPath:
                         sp_sim = 0
                    except Exception:
                         sp_sim = 0 # Safety catch

                    sim[i, j] = 0.5 * jaccard + 0.5 * sp_sim
                    sim[j, i] = sim[i, j] # Simetri

        return sim

    S = compute_similarity_matrix(G)

    # Step 2: Randomly initialize central nodes
    # Pastikan K tidak melebihi jumlah node
    K = min(K, N)
    if K < 2:
        K = 2 # Minimal 2 komunitas

    try:
        central_nodes = random.sample(nodes, K)
    except ValueError:
        # Jika N terlalu kecil
        K = N
        central_nodes = nodes

    # Inisialisasi membership matrix (U)
    U = np.zeros((N, K))
    for i in range(N):
        # Pastikan inisialisasi Dirichlet bekerja meskipun K berubah
        probs = np.random.dirichlet(np.ones(K))
        U[i, :] = probs

    # Step 3: Iterasi
    for iteration in range(max_iter):
        # Step 7: Update central nodes
        for k in range(K):
            max_contrib = -1
            best_node = central_nodes[k]
            for i in range(N):
                # Central node update: maximize contribution (membership * average similarity)
                contrib = (U[i][k] ** m) * np.mean(S[i])
                if contrib > max_contrib:
                    max_contrib = contrib
                    best_node = nodes[i]
            central_nodes[k] = best_node

        # Step 4 & 5: Compute distances and update membership matrix (FCM update)
        D = np.zeros((N, K))
        for i in range(N):
            for k in range(K):
                idx_c = node_index[central_nodes[k]]
                D[i][k] = 1 - S[i][idx_c] # Jarak = 1 - Similarity

        U_new = np.zeros_like(U)
        m_factor = 2 / (m - 1)

        for i in range(N):
            for k in range(K):
                # Pastikan distance tidak nol atau sangat kecil di pembilang
                dist_ik = D[i][k] + 1e-10
                denom = sum((dist_ik / (D[i][j] + 1e-10)) ** m_factor for j in range(K))
                U_new[i][k] = 1 / denom if denom != 0 else 0

        # Step 6: Rough set theory (Implementasi ini biasanya untuk penentuan final communities,
        # namun perhitungannya (lower/upper approx) bisa dilakukan setiap iterasi)

        # Step 8: Check convergence
        if np.linalg.norm(U - U_new) < epsilon:
            break
        U = U_new

    # Final: Gabungkan lower dan upper sebagai komunitas akhir & kembalikan U
    communities_dict = defaultdict(list)

    # Penerapan Rough Set Theory untuk hasil akhir
    for i, node in enumerate(nodes):
        for k in range(K):
            mu = U[i][k]
            # Lower approximation (strong membership)
            if mu >= lower_thr:
                communities_dict[k].append(node)
            # Boundary (Upper approximation but not in Lower)
            elif mu >= upper_thr:
                communities_dict[k].append(node)

    # Filter komunitas yang kosong
    final_communities = {k: sorted(v) for k, v in communities_dict.items() if v}

    return dict(final_communities), U


def fuzzy_modularity(G, U):
    """
    Menghitung fuzzy modularity (Nicosia et al., 2008; Chen et al., 2012)
    Params:
        G : networkx.Graph
        U : np.ndarray (N x K), membership matrix
    Returns:
        Q_fuzzy : float
    """
    # Guardrail
    if U is None or U.size == 0:
        return -1.0

    A = nx.to_numpy_array(G)
    k = np.sum(A, axis=1) # Degree of each node
    m_total = np.sum(k) / 2 # Total number of edges
    N, K = U.shape

    # s_ij = sum_c (u_ic * u_jc)
    S = U @ U.T

    Q = 0.0
    for i in range(N):
        for j in range(N):
            # Q = 1/(2m) * sum_i sum_j [ A_ij - (k_i * k_j)/(2m) ] * S_ij
            Q += (A[i, j] - (k[i] * k[j]) / (2 * m_total)) * S[i, j]

    Q /= (2 * m_total) if m_total > 0 else 0
    return Q


# ==========================================================
# 🚀 GRID SEARCH PARAMETER
# ==========================================================

def tune_Hybrid(G):
    """
    Melakukan Grid Search untuk parameter Hybrid C-Means, termasuk K_param.
    """
    nodes = list(G.nodes())
    N = len(nodes)

    # --- Hyperparameter Grid ---
    m_values = [1.5, 2.0, 2.5]
    epsilon_values = [1e-4, 1e-5]
    max_iter_values = [50, 100]
    lower_thr_values = [0.5, 0.6, 0.7] # Ditambah 0.7 untuk eksplorasi
    upper_thr_values = [0.3, 0.4]

    # Menentukan K_param. Default adalah sqrt(N). Kita tes di sekitarnya.
    default_K = max(2, int(math.sqrt(N)))
    K_values = sorted(list(set([default_K, max(2, default_K - 1), default_K + 1])))
    # ---------------------------

    best_params = None
    best_modularity = -1
    results = []

    print(f"Nodes: {N}, Default K: {default_K}. Testing K in: {K_values}")

    # Kombinasi semua parameter
    param_combinations = product(
        K_values, m_values, epsilon_values, max_iter_values, lower_thr_values, upper_thr_values
    )

    start_time = time.time()

    for K_param, m, epsilon, max_iter, lower_thr, upper_thr in param_combinations:
        # Skip jika lower_thr < upper_thr (secara teori, lower harus lebih ketat/tinggi)
        if lower_thr < upper_thr:
            continue

        try:
            # Panggil Hybrid_CMeans dengan K_param
            coms, U = Hybrid_CMeans(G, m=m, epsilon=epsilon, max_iter=max_iter,
                                    lower_thr=lower_thr, upper_thr=upper_thr, K_param=K_param)

            if U is None or U.shape[0] == 0:
                mod = -1
            else:
                mod = fuzzy_modularity(G, U)

        except Exception as e:
            # print(f"⚠️ Error at params (K={K_param}, m={m}, eps={epsilon}, iter={max_iter}, lower={lower_thr}, upper={upper_thr}): {e}")
            mod = -1

        results.append((mod, K_param, m, epsilon, max_iter, lower_thr, upper_thr))

        if mod > best_modularity:
            best_modularity = mod
            best_params = (K_param, m, epsilon, max_iter, lower_thr, upper_thr)

    end_time = time.time()

    # ==========================================================
    # CETAK HASIL PARAMETER TERBAIK
    # ==========================================================
    print("\n" + "="*40)
    print("=== HASIL PARAMETER TERBAIK (Hybrid C-Means) ===")
    print("="*40)

    if best_params is not None:
        print(f"Modularity Terbaik : {best_modularity:.4f}")
        print(f"K (Komunitas) : {best_params[0]}")
        print(f"m : {best_params[1]}")
        print(f"epsilon : {best_params[2]}")
        print(f"max_iter : {best_params[3]}")
        print(f"lower_thr : {best_params[4]}")
        print(f"upper_thr : {best_params[5]}")
        print(f"Total Uji : {len(results)} kombinasi")
        print(f"Waktu Eksekusi  : {end_time - start_time:.2f} detik")
    else:
        print("Tidak ada parameter yang menghasilkan komunitas valid.")

    return best_params, best_modularity, results

# ==========================================================
# MAIN EXECUTION
# ==========================================================

if __name__ == "__main__":
    G, mapping = load_mips_graph(txt_path)
    tune_Hybrid(G)
