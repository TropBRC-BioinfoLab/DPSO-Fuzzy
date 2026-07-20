# main.py

import argparse
import os
import re
from collections import defaultdict

import pandas as pd
import networkx as nx

from tuning_unsupervised import (
    tune_msefcd_params_unsupervised,
    run_final_msefcd_with_best_params
)


# ============================================================
# PARSER PARAMETER
# ============================================================

def parse_int_list(text):
    return [
        int(x.strip())
        for x in text.split(",")
        if x.strip() != ""
    ]


def parse_float_list(text):
    return [
        float(x.strip())
        for x in text.split(",")
        if x.strip() != ""
    ]


def parse_optional_int_list(text):
    values = []

    for x in text.split(","):
        x = x.strip()

        if x == "":
            continue

        if x.lower() in ["none", "null"]:
            values.append(None)
        else:
            values.append(int(x))

    return values


# ============================================================
# LOAD GRAPH: KARATE / GML / CSV / TXT / TSV / EDGELIST / MTX
# ============================================================

def load_graph(
    graph_path,
    source_col=None,
    target_col=None,
    weight_col=None
):
    graph_path_str = str(graph_path)
    graph_arg = graph_path_str.lower()

    # ========================================================
    # 0. BUILT-IN KARATE CLUB GRAPH
    # Bisa dipanggil dengan:
    # --graph karate
    # ========================================================
    if graph_arg in ["karate", "karate_club", "zachary"]:
        G = nx.karate_club_graph()

        # Relabel node menjadi string agar konsisten dengan evaluasi
        G = nx.relabel_nodes(G, lambda x: str(x))

        for u, v in G.edges():
            if "weight" not in G[u][v]:
                G[u][v]["weight"] = 1.0

        return G

    ext = os.path.splitext(graph_path_str)[1].lower()

    # ========================================================
    # 1. GML
    # Contoh:
    # football.gml
    # ========================================================
    if ext == ".gml":
        G = nx.read_gml(graph_path_str, label="label")

        if G.is_directed():
            G = G.to_undirected()

        for u, v in G.edges():
            if "weight" not in G[u][v]:
                G[u][v]["weight"] = 1.0

        return G

    # ========================================================
    # 2. CSV EDGE LIST
    # Contoh:
    # source,target,weight
    # disease1,disease2,similarity
    # ========================================================
    elif ext == ".csv":
        df = pd.read_csv(graph_path_str)

        if df.shape[1] < 2:
            raise ValueError(
                "File CSV graph minimal harus memiliki 2 kolom: source dan target."
            )

        if source_col is None:
            source_col = df.columns[0]

        if target_col is None:
            target_col = df.columns[1]

        if weight_col is None:
            for col in [
                "weight",
                "similarity",
                "score",
                "wang_similarity"
            ]:
                if col in df.columns:
                    weight_col = col
                    break

        G = nx.Graph()

        for _, row in df.iterrows():
            u = str(row[source_col]).strip()
            v = str(row[target_col]).strip()

            if u == "" or v == "":
                continue

            if u.lower() == "nan" or v.lower() == "nan":
                continue

            if u == v:
                continue

            w = 1.0

            if weight_col is not None:
                try:
                    w = float(row[weight_col])
                except Exception:
                    w = 1.0

            G.add_edge(u, v, weight=w)

        return G

    # ========================================================
    # 3. TXT / TSV / EDGELIST
    #
    # Contoh MIPS:
    # YBL093C    YDL005C
    #
    # Contoh Y2H:
    # YDR330W YKR094C {}
    # YDR330W YOL133W {}
    #
    # Kolom ketiga "{}" akan diabaikan.
    # Jika kolom ketiga numerik, dianggap weight.
    # ========================================================
    elif ext in [".txt", ".tsv", ".edges", ".edgelist"]:
        G = nx.Graph()

        with open(graph_path_str, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                line = line.strip()

                if line == "":
                    continue

                if line.startswith("#") or line.startswith("%"):
                    continue

                parts = line.split()

                if len(parts) < 2:
                    continue

                u = parts[0].strip()
                v = parts[1].strip()

                if u == "" or v == "":
                    continue

                if u.lower() == "nan" or v.lower() == "nan":
                    continue

                if u == v:
                    continue

                w = 1.0

                if len(parts) >= 3:
                    try:
                        w = float(parts[2])
                    except Exception:
                        w = 1.0

                G.add_edge(u, v, weight=w)

        return G

    # ========================================================
    # 4. MTX / MATRIX MARKET
    #
    # Contoh Dolphin:
    # soc-dolphins.mtx
    #
    # Format Matrix Market biasanya:
    # %%MatrixMarket matrix coordinate pattern symmetric
    # 62 62 159
    # 1 9
    # 1 10
    #
    # Node Matrix Market biasanya 1-based.
    # Diubah menjadi 0-based agar cocok dengan GT manual Dolphin.
    # ========================================================
    elif ext == ".mtx":
        G = nx.Graph()

        raw_edges = []
        n_rows = None
        n_cols = None

        has_matrix_market_header = False
        size_line_read = False

        with open(graph_path_str, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                line = line.strip()

                if line == "":
                    continue

                if line.lower().startswith("%%matrixmarket"):
                    has_matrix_market_header = True
                    continue

                if line.startswith("%") or line.startswith("#"):
                    continue

                parts = line.split()

                # Baris ukuran Matrix Market:
                # jumlah_baris jumlah_kolom jumlah_edge
                if not size_line_read and len(parts) >= 3:
                    try:
                        a = int(parts[0])
                        b = int(parts[1])
                        c = int(parts[2])

                        # Jika ada header Matrix Market, pasti ini size line.
                        # Jika tidak ada header, tetap aman untuk file .mtx umum.
                        if has_matrix_market_header or len(raw_edges) == 0:
                            n_rows = a
                            n_cols = b
                            size_line_read = True
                            continue
                    except Exception:
                        pass

                if len(parts) < 2:
                    continue

                try:
                    u = int(parts[0])
                    v = int(parts[1])
                except Exception:
                    continue

                if u == v:
                    continue

                w = 1.0

                if len(parts) >= 3:
                    try:
                        w = float(parts[2])
                    except Exception:
                        w = 1.0

                raw_edges.append((u, v, w))

        if len(raw_edges) == 0:
            raise ValueError(
                "File .mtx terbaca, tetapi tidak ada edge yang valid."
            )

        all_ids = []

        for u, v, _ in raw_edges:
            all_ids.append(u)
            all_ids.append(v)

        min_id = min(all_ids)
        max_id = max(all_ids)

        # Jika Matrix Market 1-based, ubah ke 0-based.
        # Ini cocok dengan GT Dolphin manual yang memakai node 0-61.
        use_one_based = False

        if n_rows is not None:
            if min_id >= 1 and max_id <= n_rows:
                use_one_based = True
        else:
            if min_id >= 1:
                use_one_based = True

        # Tambahkan semua node supaya node isolated tetap ada jika ada di file ukuran.
        if n_rows is not None:
            for i in range(n_rows):
                G.add_node(str(i))

        for u, v, w in raw_edges:
            if use_one_based:
                u = u - 1
                v = v - 1

            G.add_edge(str(u), str(v), weight=w)

        return G

    else:
        raise ValueError(
            f"Format graph tidak didukung: {ext}. "
            "Gunakan .gml, .csv, .txt, .tsv, .edges, .edgelist, atau .mtx"
        )


# ============================================================
# LOAD GROUND TRUTH: TXT / CSV
# ============================================================

def load_ground_truth(
    ground_truth_path,
    node_col="node",
    comm_col="community"
):
    ext = os.path.splitext(ground_truth_path)[1].lower()

    # ========================================================
    # 1. TXT
    #
    # Format football / MIPS:
    # C1: node node node
    # C2: node node node
    #
    # Format Y2H GO:
    # GO:0005634<TAB>node node node
    #
    # Format alternatif:
    # GO:0005634 node node node
    # ========================================================
    if ext == ".txt":
        communities = []

        with open(ground_truth_path, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                line = line.strip()

                if line == "":
                    continue

                if line.startswith("#") or line.startswith("%"):
                    continue

                nodes_text = ""

                # Format GO_ID<TAB>node node node
                if "\t" in line:
                    parts = line.split("\t", 1)

                    if len(parts) < 2:
                        continue

                    nodes_text = parts[1].strip()

                # Format C1: node node node
                elif ":" in line and re.match(r"^C\d+\s*:", line):
                    parts = line.split(":", 1)

                    if len(parts) < 2:
                        continue

                    nodes_text = parts[1].strip()

                # Format GO:0005634 node node node
                elif line.startswith("GO:"):
                    parts = line.split(None, 1)

                    if len(parts) < 2:
                        continue

                    nodes_text = parts[1].strip()

                else:
                    continue

                nodes = [
                    node.strip()
                    for node in nodes_text.split()
                    if node.strip() != ""
                ]

                if len(nodes) > 0:
                    communities.append(set(nodes))

        return communities

    # ========================================================
    # 2. CSV
    #
    # Format:
    # node,community
    # A,1
    # B,1
    # C,2
    #
    # Overlap bisa:
    # C,1
    # C,2
    # atau:
    # C,"1,2"
    # C,"1|2"
    # C,"1;2"
    # ========================================================
    elif ext == ".csv":
        df = pd.read_csv(ground_truth_path)

        if df.shape[1] < 2:
            raise ValueError(
                "File CSV ground truth minimal harus memiliki 2 kolom: node dan community."
            )

        if node_col not in df.columns:
            node_col = df.columns[0]

        if comm_col not in df.columns:
            comm_col = df.columns[1]

        comm_dict = defaultdict(set)

        for _, row in df.iterrows():
            node = str(row[node_col]).strip()
            raw_comm = str(row[comm_col]).strip()

            if node == "" or raw_comm == "":
                continue

            if node.lower() == "nan" or raw_comm.lower() == "nan":
                continue

            # Support:
            # 1
            # 1,2
            # 1|2
            # 1;2
            parts = re.split(r"[|;,]", raw_comm)

            for comm in parts:
                comm = comm.strip()

                if comm != "":
                    comm_dict[comm].add(node)

        return [
            set(nodes)
            for nodes in comm_dict.values()
            if len(nodes) > 0
        ]

    else:
        raise ValueError(
            f"Format ground truth tidak didukung: {ext}. "
            "Gunakan .txt atau .csv"
        )


# ============================================================
# BUILT-IN KARATE GROUND TRUTH
# ============================================================

def load_karate_ground_truth(G):
    """
    Ground truth Karate Club berdasarkan atribut 'club' dari NetworkX.
    Komunitas:
        Mr. Hi
        Officer
    """

    comm_dict = defaultdict(set)

    for node, data in G.nodes(data=True):
        club = data.get("club", None)

        if club is not None:
            comm_dict[club].add(str(node))

    return [
        set(nodes)
        for nodes in comm_dict.values()
        if len(nodes) > 0
    ]


# ============================================================
# BUILT-IN DOLPHIN GROUND TRUTH MANUAL
# ============================================================

def load_dolphin_ground_truth_manual(G):
    """
    Ground truth Dolphin manual berdasarkan daftar yang kamu punya.

    Return:
        true_communities = [comm0, comm1]
    """

    ground_truth_nodes_0 = {
        0, 2, 3, 4, 8, 10, 11, 12, 14, 15, 16, 18, 20, 21, 23, 24,
        28, 29, 30, 33, 34, 35, 36, 37, 38, 39, 40, 42, 43, 44, 45,
        46, 47, 49, 50, 51, 52, 53, 55, 58, 59, 61
    }

    comm0 = []
    comm1 = []

    for node in G.nodes():
        node_str = str(node)

        try:
            node_id = int(node_str)
        except Exception:
            continue

        if node_id in ground_truth_nodes_0:
            comm0.append(node_str)
        else:
            comm1.append(node_str)

    true_communities = [
        set(comm0),
        set(comm1)
    ]

    return true_communities


# ============================================================
# CHECK NODE CONSISTENCY
# ============================================================

def check_ground_truth_nodes(G, true_communities):
    graph_nodes = set(str(n) for n in G.nodes())
    gt_nodes = set()

    for comm in true_communities:
        gt_nodes.update(str(n) for n in comm)

    missing_in_graph = gt_nodes - graph_nodes
    missing_in_gt = graph_nodes - gt_nodes

    print("\n============================================")
    print("CHECK GROUND TRUTH")
    print("============================================")
    print(f"Nodes in graph       : {len(graph_nodes)}")
    print(f"Nodes in ground truth: {len(gt_nodes)}")

    if len(missing_in_graph) > 0:
        print(f"[WARNING] Ada {len(missing_in_graph)} node GT tidak ada di graph.")
        print("Contoh:", list(sorted(missing_in_graph))[:10])

    if len(missing_in_gt) > 0:
        print(f"[WARNING] Ada {len(missing_in_gt)} node graph tidak ada di GT.")
        print("Contoh:", list(sorted(missing_in_gt))[:10])

    if len(missing_in_graph) == 0 and len(missing_in_gt) == 0:
        print("[OK] Semua node graph dan ground truth cocok.")


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Grid search MSEFCD berdasarkan fuzzy modularity Qg, "
            "lalu final run dan tuning alpha threshold berdasarkan ground truth."
        )
    )

    # ========================================================
    # INPUT - OUTPUT
    # ========================================================
    parser.add_argument(
        "--graph",
        type=str,
        required=True,
        help=(
            "Path file graph: .gml, .csv, .txt, .tsv, .edges, "
            ".edgelist, .mtx, atau keyword karate"
        )
    )

    parser.add_argument(
        "--out",
        type=str,
        required=True,
        help="Folder output hasil tuning dan evaluasi."
    )

    # ========================================================
    # ARGUMEN CSV GRAPH
    # ========================================================
    parser.add_argument(
        "--source-col",
        type=str,
        default=None,
        help="Nama kolom source untuk graph CSV."
    )

    parser.add_argument(
        "--target-col",
        type=str,
        default=None,
        help="Nama kolom target untuk graph CSV."
    )

    parser.add_argument(
        "--weight-col",
        type=str,
        default=None,
        help="Nama kolom weight untuk graph CSV."
    )

    # ========================================================
    # GROUND TRUTH
    # ========================================================
    parser.add_argument(
        "--ground-truth",
        type=str,
        default=None,
        help=(
            "Path file ground truth: .txt atau .csv. "
            "Untuk Dolphin manual, gunakan: dolphin_manual"
        )
    )

    parser.add_argument(
        "--gt-node-col",
        type=str,
        default="node",
        help="Nama kolom node untuk ground truth CSV."
    )

    parser.add_argument(
        "--gt-comm-col",
        type=str,
        default="community",
        help="Nama kolom community untuk ground truth CSV."
    )

    # ========================================================
    # GRID SEARCH PARAMETER MSEFCD
    # ========================================================
    parser.add_argument(
        "--k-list",
        type=str,
        default="3,4,5,6,7,8,9,10",
        help="Daftar nilai k untuk MSEFCD, pisahkan dengan koma."
    )

    parser.add_argument(
        "--beta-list",
        type=str,
        default="0.5,1.0,1.5,2.0",
        help="Daftar nilai beta untuk diffusion kernel."
    )

    parser.add_argument(
        "--r-list",
        type=str,
        default="1.5,2.0,2.5,3.0",
        help="Daftar nilai r untuk fuzzy membership initialization."
    )

    parser.add_argument(
        "--init-iter-list",
        type=str,
        default="5",
        help="Daftar nilai init_iter."
    )

    parser.add_argument(
        "--max-iter-list",
        type=str,
        default="100",
        help="Daftar nilai max_iter."
    )

    parser.add_argument(
        "--max-communities-list",
        type=str,
        default="None",
        help="Daftar batas jumlah komunitas. Contoh: None atau 10,20,30"
    )

    # ========================================================
    # THRESHOLD OVERLAPPING COMMUNITY
    # ========================================================
    parser.add_argument(
        "--alpha-threshold",
        type=float,
        default=0.3,
        help=(
            "Threshold manual jika ground truth tidak diberikan. "
            "Jika ground truth diberikan, threshold terbaik akan dicari dari alpha-threshold-list."
        )
    )

    parser.add_argument(
        "--alpha-threshold-list",
        type=str,
        default=(
            "0.10,0.15,0.20,0.25,0.30,"
            "0.35,0.40,0.45,0.50,0.55,"
            "0.60,0.65,0.70,0.75,0.80"
        ),
        help=(
            "Daftar alpha threshold untuk tuning setelah best parameter MSEFCD diperoleh."
        )
    )

    parser.add_argument(
        "--threshold-objective",
        type=str,
        default="comm_F1",
        choices=[
            "onmi",
            "omega",
            "comm_recall",
            "comm_precision",
            "comm_F1",
            "average_score"
        ],
        help=(
            "Metrik untuk memilih threshold terbaik. "
            "Default: comm_F1."
        )
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Tampilkan detail proses tuning."
    )

    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)

    # ========================================================
    # LOAD GRAPH
    # ========================================================
    print("============================================")
    print("LOAD GRAPH")
    print("============================================")

    G = load_graph(
        graph_path=args.graph,
        source_col=args.source_col,
        target_col=args.target_col,
        weight_col=args.weight_col
    )

    print(f"Graph path: {args.graph}")
    print(f"Nodes     : {G.number_of_nodes()}")
    print(f"Edges     : {G.number_of_edges()}")

    if G.number_of_nodes() == 0 or G.number_of_edges() == 0:
        raise ValueError(
            "Graph kosong atau tidak memiliki edge. "
            "Periksa format file graph."
        )

    # ========================================================
    # LOAD GROUND TRUTH
    # ========================================================
    true_communities = None

    graph_arg = str(args.graph).lower()
    graph_basename = os.path.basename(str(args.graph)).lower()

    ground_truth_arg = (
        None
        if args.ground_truth is None
        else str(args.ground_truth).lower()
    )

    # ========================================================
    # Ground truth otomatis untuk Karate
    # Bisa dipanggil dengan:
    # --graph karate
    # ========================================================
    if graph_arg in ["karate", "karate_club", "zachary"]:
        print("\n============================================")
        print("LOAD GROUND TRUTH")
        print("============================================")
        print("[INFO] Menggunakan ground truth bawaan Karate Club dari NetworkX.")

        true_communities = load_karate_ground_truth(G)

        print(f"Ground truth communities: {len(true_communities)}")

        check_ground_truth_nodes(G, true_communities)

    # ========================================================
    # Ground truth manual untuk Dolphin
    # Bisa dipanggil dengan:
    # --ground-truth dolphin_manual
    #
    # Atau otomatis jika nama graph mengandung dolphin
    # dan ground truth tidak diberikan.
    # ========================================================
    elif (
        ground_truth_arg in [
            "dolphin",
            "dolphins",
            "dolphin_manual",
            "soc-dolphins"
        ]
        or (args.ground_truth is None and "dolphin" in graph_basename)
    ):
        print("\n============================================")
        print("LOAD GROUND TRUTH")
        print("============================================")
        print("[INFO] Menggunakan ground truth manual Dolphin.")

        true_communities = load_dolphin_ground_truth_manual(G)

        print(f"Ground truth communities: {len(true_communities)}")

        check_ground_truth_nodes(G, true_communities)

    # ========================================================
    # Ground truth dari file
    # Contoh:
    # football_GT.txt
    # Detected Complexes.txt
    # Y2H_groundtruth_GO_combined_v2.txt
    # ========================================================
    elif args.ground_truth is not None:
        print("\n============================================")
        print("LOAD GROUND TRUTH")
        print("============================================")

        true_communities = load_ground_truth(
            ground_truth_path=args.ground_truth,
            node_col=args.gt_node_col,
            comm_col=args.gt_comm_col
        )

        print(f"Ground truth path       : {args.ground_truth}")
        print(f"Ground truth communities: {len(true_communities)}")

        if len(true_communities) == 0:
            print("[WARNING] Ground truth terbaca 0 komunitas.")
            print("[WARNING] Evaluasi ONMI/Omega/comm_recall/comm_precision/comm_F1 mungkin tidak valid.")

        check_ground_truth_nodes(G, true_communities)

    # ========================================================
    # Tanpa ground truth
    # ========================================================
    else:
        print("\n[INFO] Ground truth tidak diberikan.")
        print("[INFO] Threshold tuning berbasis ground truth tidak dilakukan.")
        print("[INFO] ONMI, Omega, comm_recall, comm_precision, dan comm_F1 tidak dihitung.")
        print(f"[INFO] Final community akan dibentuk memakai alpha_threshold={args.alpha_threshold}")

    # ========================================================
    # PARSE PARAMETER LIST
    # ========================================================
    k_list = parse_int_list(args.k_list)
    beta_list = parse_float_list(args.beta_list)
    r_list = parse_float_list(args.r_list)
    init_iter_list = parse_int_list(args.init_iter_list)
    max_iter_list = parse_int_list(args.max_iter_list)
    max_communities_list = parse_optional_int_list(args.max_communities_list)
    alpha_threshold_list = parse_float_list(args.alpha_threshold_list)

    # ========================================================
    # GRID SEARCH MSEFCD
    # ========================================================
    print("\n============================================")
    print("GRID SEARCH MSEFCD")
    print("Objective: maximize fuzzy modularity Qg")
    print("============================================")

    df_tuning, best_row, best_result, csv_path, cfg_path = tune_msefcd_params_unsupervised(
        G=G,
        k_list=k_list,
        beta_list=beta_list,
        r_list=r_list,
        init_iter_list=init_iter_list,
        max_iter_list=max_iter_list,
        max_communities_list=max_communities_list,
        save_csv_path=os.path.join(args.out, "tuning_unsupervised_msefcd.csv"),
        best_config_json_path=os.path.join(args.out, "best_config_msefcd.json"),
        verbose=True
    )

    if best_row is None:
        raise ValueError(
            "Best parameter tidak ditemukan. "
            "Periksa daftar parameter tuning."
        )

    print("\n============================================")
    print("BEST PARAMETER")
    print("============================================")
    print(best_row)

    # ========================================================
    # FINAL RUN + THRESHOLD TUNING
    # ========================================================
    print("\n============================================")
    print("FINAL RUN WITH BEST PARAMETER")
    print("============================================")

    if true_communities is not None:
        print("Threshold tuning: aktif")
        print(f"Threshold objective: {args.threshold_objective}")
        print(f"Alpha threshold list: {alpha_threshold_list}")
    else:
        print("Threshold tuning: tidak aktif karena ground truth tidak diberikan")
        print(f"Manual alpha threshold: {args.alpha_threshold}")

    final_result, pred_communities, final_eval = run_final_msefcd_with_best_params(
        G=G,
        best_row=best_row,
        true_communities=true_communities,
        alpha_threshold=args.alpha_threshold,
        alpha_threshold_list=alpha_threshold_list,
        threshold_objective=args.threshold_objective,
        out_dir=args.out,
        verbose=True
    )

    print("\n============================================")
    print("SELESAI")
    print("============================================")
    print(f"Output folder: {args.out}")


if __name__ == "__main__":
    main()
