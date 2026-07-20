# msefcd.py

import numpy as np
import networkx as nx
from scipy.linalg import expm


# ============================================================
# BASIC UTILITIES
# ============================================================

def normalize_columns(U, eps=1e-12):
    """
    Normalisasi membership matrix agar setiap kolom berjumlah 1.
    Kolom = node, baris = komunitas.
    """
    col_sum = U.sum(axis=0, keepdims=True)
    col_sum[col_sum < eps] = 1.0
    return U / col_sum


def modularity_matrix(A):
    """
    B_ij = A_ij - deg(i)deg(j) / sum(A)
    Untuk graf undirected, sum(A) = 2m.
    """
    deg = A.sum(axis=1)
    total_weight = A.sum()

    if total_weight == 0:
        return np.zeros_like(A)

    expected = np.outer(deg, deg) / total_weight
    return A - expected


def generalized_modularity(U, A):
    """
    Generalized / fuzzy modularity Qg:
    Qg = trace(U B U^T) / ||W||
    """
    total_weight = A.sum()

    if total_weight == 0:
        return 0.0

    B = modularity_matrix(A)
    qg = np.trace(U @ B @ U.T) / total_weight

    return float(qg)


# ============================================================
# MSEFCD CENTRALITY
# ============================================================

def msefcd_centrality(A):
    """
    Centrality sesuai MSEFCD:

    centrality(v) = deg(v)
                    + sum_{u in N(v)} (deg(u)
                    + sum_{w in N(u)} deg(w))
    """
    n = A.shape[0]
    deg = A.sum(axis=1)
    neighbors = [np.where(A[i] > 0)[0] for i in range(n)]

    centrality = np.zeros(n)

    for v in range(n):
        value = deg[v]

        for u in neighbors[v]:
            value += deg[u]
            value += deg[neighbors[u]].sum()

        centrality[v] = value

    return centrality


def select_non_adjacent_centers(A, max_centers=None):
    """
    Memilih center nodes berdasarkan centrality tertinggi.
    Diutamakan center yang tidak saling adjacent.
    """
    n = A.shape[0]
    cent = msefcd_centrality(A)

    # Urutan deterministik:
    # centrality terbesar dulu, jika sama pilih indeks kecil.
    order = np.lexsort((np.arange(n), -cent))

    centers = []

    for node in order:
        if len(centers) == 0:
            centers.append(node)
        else:
            adjacent = any(A[node, c] > 0 for c in centers)

            if not adjacent:
                centers.append(node)

        if max_centers is not None and len(centers) >= max_centers:
            break

    # Fallback jika center non-adjacent kurang dari max_centers
    if max_centers is not None and len(centers) < max_centers:
        for node in order:
            if node not in centers:
                centers.append(node)

            if len(centers) >= max_centers:
                break

    return centers


# ============================================================
# DIFFUSION KERNEL DISTANCE
# ============================================================

def diffusion_kernel_distance(A, beta=1.0, eps=1e-12):
    """
    K = exp(-beta * L)
    K_ij normalized = K_ij / sqrt(K_ii * K_jj)
    DK = max(K) - K

    DK digunakan sebagai jarak antar node.
    """
    deg = A.sum(axis=1)
    L = np.diag(deg) - A

    K = expm(-beta * L)

    diag = np.diag(K).copy()
    diag[diag < eps] = eps

    K_norm = K / np.sqrt(np.outer(diag, diag))
    DK = np.max(K_norm) - K_norm

    return DK


# ============================================================
# MEMBERSHIP INITIALIZATION
# ============================================================

def crisp_membership_from_centers(DK, centers):
    """
    Membership hard/crisp untuk mengevaluasi jumlah komunitas
    saat interval reduction.
    """
    n = DK.shape[0]
    c = len(centers)

    U = np.zeros((c, n))

    for j in range(n):
        dists = np.array([DK[center, j] for center in centers])
        nearest = int(np.argmin(dists))
        U[nearest, j] = 1.0

    return U


def fuzzy_membership_from_centers(DK, centers, r=2.0, eps=1e-12):
    """
    Initial fuzzy membership matrix.

    u_ij = 1 / sum_k ((d_ij / d_kj) ** (2 / (r - 1)))

    Jika jarak node ke center = 0, membership ke center tersebut = 1.
    """
    n = DK.shape[0]
    c = len(centers)

    U = np.zeros((c, n))
    exponent = 2.0 / (r - 1.0)

    for j in range(n):
        dists = np.array([DK[center, j] for center in centers], dtype=float)

        zero_idx = np.where(dists < eps)[0]

        if len(zero_idx) > 0:
            U[zero_idx[0], j] = 1.0
        else:
            for i in range(c):
                ratio = (dists[i] / (dists + eps)) ** exponent
                U[i, j] = 1.0 / np.sum(ratio)

    return normalize_columns(U)


def interval_reduction_initialization(
    A,
    DK,
    init_iter=5,
    max_communities=None,
    r=2.0
):
    """
    Inisialisasi membership matrix menggunakan interval reduction.
    Tujuannya menentukan jumlah komunitas awal secara otomatis.
    """
    n = A.shape[0]

    if n <= 1:
        return np.ones((1, n)), [0], 1

    all_centers = select_non_adjacent_centers(A)

    if max_communities is None:
        max_communities = len(all_centers)

    max_communities = max(2, min(int(max_communities), n))

    cl = 2
    cr = max_communities

    def evaluate_c(c):
        c = int(max(1, min(c, n)))
        centers = select_non_adjacent_centers(A, max_centers=c)
        U_crisp = crisp_membership_from_centers(DK, centers)
        qg = generalized_modularity(U_crisp, A)
        return qg, centers

    for _ in range(init_iter):
        if cr - cl <= 1:
            break

        cm = int(round((cl + cr) / 2))

        ql, _ = evaluate_c(cl)
        qm, _ = evaluate_c(cm)
        qr, _ = evaluate_c(cr)

        if qm > ql and qm < qr:
            cl = cm

        elif qm < ql and qm > qr:
            cr = cm

        elif qm >= ql and qm >= qr:
            new_cl = int(round((cl + cm) / 2))
            new_cr = int(round((cm + cr) / 2))

            if new_cl == cl and new_cr == cr:
                break

            cl, cr = new_cl, new_cr

        else:
            break

    candidates = sorted(set([cl, int(round((cl + cr) / 2)), cr]))

    best_q = -np.inf
    best_c = None
    best_centers = None

    for c in candidates:
        q, centers = evaluate_c(c)

        if q > best_q:
            best_q = q
            best_c = c
            best_centers = centers

    U_init = fuzzy_membership_from_centers(
        DK=DK,
        centers=best_centers,
        r=r
    )

    return U_init, best_centers, best_c


# ============================================================
# MEMBERSHIP SMOOTHING
# ============================================================

def membership_smoothing(U, A):
    """
    Membership smoothing:
    membership node diperhalus berdasarkan consensus nodes.

    Consensus nodes:
    Nco = {j | B_pj > 0}
    """
    B = modularity_matrix(A)

    U_old = U.copy()
    U_new = np.zeros_like(U_old)

    n = U.shape[1]

    for p in range(n):
        consensus_nodes = np.where(B[p, :] > 0)[0]

        vec = U_old[:, p].copy()

        if len(consensus_nodes) > 0:
            vec += U_old[:, consensus_nodes].sum(axis=1)

        U_new[:, p] = vec

    return normalize_columns(U_new)


# ============================================================
# MEMBERSHIP ENHANCEMENT
# ============================================================

def membership_enhancement(U, DK, k=6, eps=1e-12):
    """
    Membership enhancement:
    - Ambil k-nearest nodes dari node p.
    - Ambil komunitas dominan dari k-nearest nodes.
    - Membership komunitas lain dihapus.
    - Membership yang tersisa dinormalisasi ulang.
    """
    c, n = U.shape
    k = max(1, min(int(k), n))

    U_old = U.copy()
    U_new = np.zeros_like(U_old)

    for p in range(n):
        nearest_nodes = np.argsort(DK[p, :])[:k]

        selected_communities = set()

        for node in nearest_nodes:
            selected_communities.add(int(np.argmax(U_old[:, node])))

        selected_communities = sorted(selected_communities)

        denom = U_old[selected_communities, p].sum()

        if denom < eps:
            best = int(np.argmax(U_old[:, p]))
            U_new[best, p] = 1.0
        else:
            U_new[selected_communities, p] = (
                U_old[selected_communities, p] / denom
            )

    # Hapus community yang seluruh membership-nya nol
    nonzero_rows = np.where(U_new.sum(axis=1) > eps)[0]
    U_new = U_new[nonzero_rows, :]

    return normalize_columns(U_new)


# ============================================================
# MAIN CLASS MSEFCD
# ============================================================

class MSEFCD:
    def __init__(
        self,
        init_iter=5,
        max_iter=100,
        k=6,
        beta=1.0,
        r=2.0,
        max_communities=None,
        tol=1e-8,
        verbose=False
    ):
        self.init_iter = init_iter
        self.max_iter = max_iter
        self.k = k
        self.beta = beta
        self.r = r
        self.max_communities = max_communities
        self.tol = tol
        self.verbose = verbose

    def fit(self, G):
        """
        Menjalankan MSEFCD pada graph NetworkX.

        Input:
            G : networkx.Graph

        Output:
            dictionary berisi:
            - nodes
            - adjacency
            - membership
            - hard_labels
            - centers
            - n_communities
            - qg
            - qg_history
        """
        nodes = list(G.nodes())
        A = nx.to_numpy_array(
            G,
            nodelist=nodes,
            weight="weight",
            dtype=float
        )

        if A.shape[0] == 0:
            raise ValueError("Graph kosong. Periksa file input.")

        DK = diffusion_kernel_distance(
            A=A,
            beta=self.beta
        )

        U, centers, selected_c = interval_reduction_initialization(
            A=A,
            DK=DK,
            init_iter=self.init_iter,
            max_communities=self.max_communities,
            r=self.r
        )

        qg_history = []

        for it in range(1, self.max_iter + 1):
            U_before = U.copy()

            U = membership_smoothing(U, A)
            U = membership_enhancement(U, DK, k=self.k)

            qg = generalized_modularity(U, A)
            qg_history.append(qg)

            diff = np.inf

            if U.shape == U_before.shape:
                diff = np.max(np.abs(U - U_before))

            if self.verbose:
                print(
                    f"Iter {it:03d} | "
                    f"communities={U.shape[0]} | "
                    f"Qg={qg:.6f} | "
                    f"diff={diff:.3e}"
                )

            if diff < self.tol:
                break

        hard_labels = np.argmax(U, axis=0)

        return {
            "nodes": nodes,
            "adjacency": A,
            "membership": U,
            "hard_labels": hard_labels,
            "centers": centers,
            "initial_communities": selected_c,
            "n_communities": int(U.shape[0]),
            "qg": generalized_modularity(U, A),
            "qg_history": qg_history
        }


# ============================================================
# CONVERT MEMBERSHIP TO OVERLAPPING COMMUNITIES
# ============================================================

def membership_to_communities(
    nodes,
    U,
    alpha=0.5,
    include_argmax=True
):
    """
    Mengubah membership matrix menjadi komunitas overlapping.

    Node dimasukkan ke komunitas k jika:
        U[k, node] >= alpha

    include_argmax=True memastikan setiap node tetap masuk
    minimal ke satu komunitas, yaitu komunitas dengan membership terbesar.
    """
    communities = []

    for k in range(U.shape[0]):
        comm = set()

        for j, node in enumerate(nodes):
            if U[k, j] >= alpha:
                comm.add(str(node))

        if len(comm) > 0:
            communities.append(comm)

    if include_argmax:
        covered = set().union(*communities) if communities else set()

        for j, node in enumerate(nodes):
            node = str(node)

            if node not in covered:
                k_best = int(np.argmax(U[:, j]))

                while len(communities) <= k_best:
                    communities.append(set())

                communities[k_best].add(node)

    # Hapus komunitas kosong dan duplikat
    clean = []
    seen = set()

    for comm in communities:
        key = tuple(sorted(comm))

        if len(comm) > 0 and key not in seen:
            clean.append(comm)
            seen.add(key)

    return clean
