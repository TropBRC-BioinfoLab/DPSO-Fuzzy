import math
import random
from collections import defaultdict

import networkx as nx
import numpy as np


class PSOCommunityDetection:
    """
    Discrete Particle Swarm Optimization (DPSO) untuk deteksi komunitas.

    Alur implementasi mengikuti langkah 1--8 pada pseudocode DPSO-Fuzzy:
      1. Inisialisasi posisi partikel dari struktur tetangga dan kecepatan nol.
      2. Evaluasi modularitas serta inisialisasi personal/global best.
      3. Pembaruan kecepatan diskrit.
      4. Pembaruan label komunitas berbasis sigmoid dan threshold ``rho``.
      5. Pemisahan komunitas yang tidak terhubung.
      6. Evaluasi modularitas dan pembaruan personal/global best.

    Catatan representasi:
      - Satu dimensi partikel mewakili satu node.
      - Nilai posisi adalah label komunitas node tersebut.
      - Selisih posisi pada DPSO bersifat kategorikal: 0 bila label sama dan
        1 bila label berbeda. Dengan demikian, nomor label tidak diperlakukan
        sebagai jarak numerik.
    """

    def __init__(
        self,
        graph,
        num_particles=30,
        max_iter=100,
        rho=0.5,
        sig_variant="paper",
    ):
        if graph.number_of_nodes() == 0:
            raise ValueError("Graph tidak memiliki node.")

        self.graph = graph
        self.node_list = list(graph.nodes())
        self.num_nodes = len(self.node_list)
        self.num_particles = int(num_particles)
        self.max_iter = int(max_iter)
        self.rho = float(rho)
        self.sig_variant = sig_variant

        self.node_index = {node: idx for idx, node in enumerate(self.node_list)}
        self.adj_index = {
            self.node_index[node]: [self.node_index[nbr] for nbr in graph.neighbors(node)]
            for node in self.node_list
        }

        self.global_best_position = None
        self.global_best_modularity = -float("inf")
        self.particles = []

    # ------------------------------------------------------------------
    # Eq. (1): position initialization from graph neighborhoods
    # ------------------------------------------------------------------
    def initialize_particle(self, p=0.5):
        """
        Membentuk partisi awal dengan propagasi label lokal.

        Node awal membentuk komunitas baru. Untuk setiap tetangga yang belum
        diberi label, label saat ini diwariskan dengan peluang ``1-p``.
        Node yang tetap belum terlabel menjadi seed komunitas berikutnya.
        """
        n = self.num_nodes
        labels = [0] * n
        visited = [False] * n
        current_label = 0

        while any(label == 0 for label in labels):
            seed = next(i for i, label in enumerate(labels) if label == 0)
            current_label += 1
            labels[seed] = current_label
            visited[seed] = True

            queue = [seed]
            while queue:
                j = queue.pop(0)
                for i in self.adj_index[j]:
                    if visited[i]:
                        continue
                    visited[i] = True
                    if random.random() > p:
                        labels[i] = labels[j]
                        queue.append(i)

            # Node yang dikunjungi tetapi tidak menerima label harus dapat
            # dipilih kembali sebagai seed pada putaran berikutnya.
            visited = [label != 0 for label in labels]

        labels = self._split_disconnected_communities(labels)
        labels = self._canonicalize_labels(labels)

        return {
            "position": labels,
            "velocity": [0.0] * n,
            "best_position": labels.copy(),
            "best_modularity": -float("inf"),
            # Bobot target disimpan agar update posisi konsisten dengan
            # komponen cognitive/social pada update kecepatan.
            "target_weights": [(0.0, 0.0)] * n,
        }

    # ------------------------------------------------------------------
    # Sigmoid for discrete label update
    # ------------------------------------------------------------------
    def _sigmoid(self, velocity):
        """Mengubah kecepatan menjadi probabilitas perubahan label [0, 1]."""
        v = float(np.clip(velocity, -4.0, 4.0))

        # ``paper`` diperlakukan sebagai logistic sigmoid. Implementasi lama
        # memakai tanh yang menghasilkan nilai negatif sehingga tidak valid
        # sebagai probabilitas perubahan label.
        if self.sig_variant in {"paper", "logistic"}:
            return 1.0 / (1.0 + math.exp(-v))
        if self.sig_variant == "tanh":
            return 0.5 * (math.tanh(0.5 * v) + 1.0)
        if self.sig_variant == "leaky":
            return 1.0 / (1.0 + math.exp(-0.5 * v))
        raise ValueError(f"sig_variant tidak dikenal: {self.sig_variant}")

    @staticmethod
    def _categorical_difference(target_label, current_label):
        """Selisih kategorikal untuk label komunitas."""
        return 0.0 if target_label == current_label else 1.0

    # ------------------------------------------------------------------
    # Eq. (3): velocity update
    # ------------------------------------------------------------------
    def update_velocity(self, particle, w, c1, c2):
        """
        v_i(t+1) = w v_i(t)
                   + c1 r1 delta(pbest_i, x_i)
                   + c2 r2 delta(gbest_i, x_i)

        ``delta`` bersifat kategorikal karena label komunitas tidak mempunyai
        makna ordinal. Tidak ada faktor tambahan atau random-exploration di
        luar persamaan DPSO.
        """
        target_weights = []

        for i in range(self.num_nodes):
            current = particle["position"][i]
            pbest = particle["best_position"][i]
            gbest = (
                self.global_best_position[i]
                if self.global_best_position is not None
                else current
            )

            cognitive = (
                float(c1)
                * random.random()
                * self._categorical_difference(pbest, current)
            )
            social = (
                float(c2)
                * random.random()
                * self._categorical_difference(gbest, current)
            )

            velocity = float(w) * particle["velocity"][i] + cognitive + social
            particle["velocity"][i] = float(np.clip(velocity, -4.0, 4.0))
            target_weights.append((cognitive, social))

        particle["target_weights"] = target_weights

    # ------------------------------------------------------------------
    # Sigmoid-based discrete position update
    # ------------------------------------------------------------------
    def update_position(self, particle):
        """
        Mengubah label hanya ketika nilai sigmoid melewati ``rho``.

        Label baru dipilih dari exemplar personal-best/global-best, sebanding
        dengan kontribusinya pada pembaruan kecepatan. Jika kedua exemplar
        sama dengan posisi saat ini, posisi dipertahankan.
        """
        position = particle["position"].copy()

        for i in range(self.num_nodes):
            change_probability = self._sigmoid(particle["velocity"][i])
            if change_probability <= self.rho:
                continue

            current = position[i]
            pbest = particle["best_position"][i]
            gbest = (
                self.global_best_position[i]
                if self.global_best_position is not None
                else current
            )
            cognitive, social = particle["target_weights"][i]

            candidates = []
            weights = []
            if pbest != current and cognitive > 0.0:
                candidates.append(pbest)
                weights.append(cognitive)
            if gbest != current and social > 0.0:
                candidates.append(gbest)
                weights.append(social)

            if candidates:
                position[i] = random.choices(candidates, weights=weights, k=1)[0]

        particle["position"] = position

    # ------------------------------------------------------------------
    # Step 5: split disconnected communities
    # ------------------------------------------------------------------
    def _split_disconnected_communities(self, labels):
        """Memecah setiap label yang mengandung lebih dari satu komponen."""
        label_nodes = defaultdict(list)
        for idx, label in enumerate(labels):
            label_nodes[label].append(idx)

        corrected = list(labels)
        next_label = max(labels, default=0)

        for label, indices in label_nodes.items():
            if label == 0 or len(indices) <= 1:
                continue

            graph_nodes = [self.node_list[i] for i in indices]
            subgraph = self.graph.subgraph(graph_nodes)
            components = sorted(
                nx.connected_components(subgraph),
                key=lambda comp: min(self.node_index[node] for node in comp),
            )

            for component in components[1:]:
                next_label += 1
                for node in component:
                    corrected[self.node_index[node]] = next_label

        return corrected

    def _canonicalize_labels(self, labels):
        """Menomori ulang label berdasarkan urutan kemunculan node."""
        mapping = {}
        next_label = 1
        canonical = []
        for label in labels:
            if label not in mapping:
                mapping[label] = next_label
                next_label += 1
            canonical.append(mapping[label])
        return canonical

    # Nama lama dipertahankan agar kode eksternal tetap kompatibel.
    def _fix_invalid_partition(self, labels):
        return self._canonicalize_labels(
            self._split_disconnected_communities(labels)
        )

    def position_to_assignment(self, position, node_list=None):
        nodes = self.node_list if node_list is None else node_list
        return {node: int(label) for node, label in zip(nodes, position)}

    # ------------------------------------------------------------------
    # Eq. (2): modularity evaluation
    # ------------------------------------------------------------------
    def evaluate_particle(self, particle):
        partition = defaultdict(set)
        for node_idx, label in enumerate(particle["position"]):
            partition[label].add(self.node_list[node_idx])

        communities = [nodes for nodes in partition.values() if nodes]
        if len(communities) <= 1:
            return -1.0

        try:
            return float(
                nx.community.modularity(
                    self.graph,
                    communities,
                    weight="weight",
                )
            )
        except (nx.NetworkXError, ZeroDivisionError, ValueError):
            return -1.0

    # ------------------------------------------------------------------
    # Steps 1--8: run DPSO
    # ------------------------------------------------------------------
    def run(
        self,
        w=0.9,
        c1=2.0,
        c2=2.0,
        p=0.6,
        verbose=True,
        num_particles=None,
        max_iter=None,
    ):
        if num_particles is not None:
            self.num_particles = int(num_particles)
        if max_iter is not None:
            self.max_iter = int(max_iter)

        self.particles = [
            self.initialize_particle(p=p) for _ in range(self.num_particles)
        ]
        self.global_best_modularity = -float("inf")
        self.global_best_position = None

        # Step 2: initialize personal/global best.
        for particle in self.particles:
            score = self.evaluate_particle(particle)
            particle["best_modularity"] = score
            particle["best_position"] = particle["position"].copy()
            if score > self.global_best_modularity:
                self.global_best_modularity = score
                self.global_best_position = particle["position"].copy()

        if verbose:
            print(f"Initial global modularity: {self.global_best_modularity:.4f}")

        # Steps 3--7.
        for iteration in range(self.max_iter):
            for particle in self.particles:
                self.update_velocity(particle, w=w, c1=c1, c2=c2)
                self.update_position(particle)

                # Step 5 must occur after every label update.
                particle["position"] = self._fix_invalid_partition(
                    particle["position"]
                )

                score = self.evaluate_particle(particle)
                if score > particle["best_modularity"]:
                    particle["best_modularity"] = score
                    particle["best_position"] = particle["position"].copy()

                if score > self.global_best_modularity:
                    self.global_best_modularity = score
                    self.global_best_position = particle["position"].copy()

            if verbose and (
                iteration % max(1, self.max_iter // 10) == 0
                or iteration == self.max_iter - 1
            ):
                print(
                    f"Iter {iteration + 1}/{self.max_iter} - "
                    f"Global best modularity: {self.global_best_modularity:.4f}"
                )

        # Step 8: X* <- global best.
        return (
            self.global_best_position.copy()
            if self.global_best_position is not None
            else None
        )


def run_dpso_multiple(
    G,
    num_runs=20,
    num_particles=100,
    max_iter=300,
    w=0.7,
    c1=1.5,
    c2=1.0,
    p=0.6,
    rho=0.5,
    verbose=True,
):
    """Menjalankan DPSO berulang tanpa mengubah skema tuning/evaluasi."""
    results = []
    node_list = list(G.nodes())

    for run_idx in range(int(num_runs)):
        if verbose:
            print("\n============================")
            print(f"Percobaan DPSO ke-{run_idx + 1}/{num_runs}")
            print("============================")

        pso = PSOCommunityDetection(
            graph=G,
            num_particles=num_particles,
            max_iter=max_iter,
            rho=rho,
        )

        best_position = pso.run(
            w=w,
            c1=c1,
            c2=c2,
            p=p,
            verbose=False,
        )
        if best_position is None:
            if verbose:
                print(f"Run {run_idx + 1} tidak menghasilkan posisi terbaik.")
            continue

        community_assignment = pso.position_to_assignment(
            best_position,
            node_list,
        )
        num_communities = len(set(community_assignment.values()))
        best_modularity = pso.global_best_modularity

        if verbose:
            print(f"Hasil run {run_idx + 1}:")
            print(f"  - Jumlah komunitas : {num_communities}")
            print(f"  - Modularitas      : {best_modularity:.4f}")

        results.append(
            {
                "run": run_idx + 1,
                "best_position": [int(x) for x in best_position],
                "community_assignment": community_assignment,
                "num_communities": int(num_communities),
                "modularity": float(best_modularity),
            }
        )

    if verbose:
        print("\nRingkasan DPSO:")
        for result in results:
            print(
                f"Run {result['run']:02d}: "
                f"Modularitas = {result['modularity']:.4f}, "
                f"Komunitas = {result['num_communities']}"
            )

    return results
