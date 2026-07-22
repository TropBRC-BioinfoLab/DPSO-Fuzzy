# main.py

import argparse
import json
import os
from datetime import datetime

import networkx as nx
import pandas as pd

from dpso import run_dpso_multiple
from utils_loader import load_disease_csv
from tuning_dpso_disease import tune_dpso_hyperparams_disease
from msefcd_hybrid import (
    compute_fuzzy_membership_from_DPSO_MSEFCD_full,
    detect_overlapping,
    evaluate_all,
    fuzzy_to_communities,
    evaluate_overlapping_all,
)
from tuning_unsupervised import tune_hybrid_msefcd_params_unsupervised


# =========================================================
#  Helper I/O
# =========================================================

def load_graph(path):
    """
    Loader otomatis berdasarkan ekstensi file:
    - .csv       -> disease CSV (load_disease_csv)
    - .gml       -> standard GML
    - lainnya    -> edgelist
    """
    ext = os.path.splitext(path)[1].lower()

    # === FILE CSV (Dataset penyakit) ===
    if ext == ".csv":
        from utils_loader import load_disease_csv
        print(f"📂 Memuat graph CSV penyakit: {path}")
        G = load_disease_csv(path)

    # === FILE GML ===
    elif ext == ".gml":
        print(f"📂 Memuat graph GML: {path}")
        G = nx.read_gml(path)

    # === FILE EDGE LIST (default) ===
    else:
        print(f"📂 Memuat graph edge list: {path}")
        G = nx.read_edgelist(path)

    # Hapus node terisolasi
    isolates = list(nx.isolates(G))
    if isolates:
        G.remove_nodes_from(isolates)

    # Hasil komponen terbesar saja
    components = list(nx.connected_components(G))
    components_sorted = sorted(components, key=len, reverse=True)

    print(f"Jumlah komponen: {len(components_sorted)}")
    print("Ukuran tiap komponen:", [len(c) for c in components_sorted])

    largest_component = components_sorted[0]
    G_big = G.subgraph(largest_component).copy()

    print(
        f"Gunakan komponen terbesar: "
        f"{G_big.number_of_nodes()} nodes, "
        f"{G_big.number_of_edges()} edges"
    )

    return G_big



def save_dpso_results(path, results):
    """
    Simpan hasil multi-run DPSO ke JSON.
    """
    with open(path, "w") as f:
        json.dump(results, f)
    print(f"💾 DPSO results disimpan ke: {path}")


def load_dpso_results(path):
    """
    Load hasil multi-run DPSO dari JSON.
    File harus berisi list of dict seperti keluaran run_dpso_multiple.
    """
    with open(path, "r") as f:
        data = json.load(f)
    print(f"📂 DPSO results loaded dari: {path} (runs = {len(data)})")
    return data


def load_gt_communities(path, G):
    """
    Memuat ground truth kompleks protein dengan format:

    C1:  YHR023W YLR274W YAL029C ...
    C2:  YGL048C YDL007W YDL097C ...
    C3:  YOR304W YGL133W YJL065C ...

    Return: list of list node names yang ada di G.
    """
    comms = []

    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            # Harus ada ':'
            if ":" not in line:
                continue

            # Pisahkan sebelum dan sesudah colon
            _, nodes_str = line.split(":", 1)

            # Node dipisah spasi
            parts = nodes_str.strip().split()

            # Filter node yang tidak ada di graph
            filtered = [u for u in parts if u in G.nodes()]

            if len(filtered) > 0:
                comms.append(filtered)

    print(f"📂 Ground truth loaded dari: {path}")
    print(f"📊 Total complexes (non-empty & in-graph): {len(comms)}")
    return comms


# =========================================================
#  MAIN LOGIC
# =========================================================

def parse_args():
    p = argparse.ArgumentParser(
        description="Hybrid DPSO + MSEFCD Community Detection Framework"
    )
    p.add_argument("--graph", required=True,
                   help="Path graph (GML atau edgelist)")

    p.add_argument("--mode", required=True,
                   choices=["dpso", "tune_unsup", "dpso_plus_tune_unsup", "eval_sup", "tune_dpso_disease","dpso_best"],
                   help=(
                       "dpso: hanya jalankan DPSO\n"
                       "tune_unsup: tuning unsupervised (butuh --dpsofile)\n"
                       "dpso_plus_tune_unsup: jalankan DPSO lalu tuning unsupervised\n"
                       "eval_sup: evaluasi supervised dengan GT"
                   ))

    p.add_argument("--best_dpso", default=None,
               help="Path JSON hasil tuning DPSO (dpso_tuning_*.json)")
    # DPSO related
    p.add_argument("--num-runs", type=int, default=20,
                   help="Jumlah run DPSO (default: 20)")
    p.add_argument("--num-particles", type=int, default=100,
                   help="Jumlah partikel per run DPSO (default: 100)")
    p.add_argument("--max-iter", type=int, default=100,
                   help="Jumlah iterasi per run DPSO (default: 100)")
    p.add_argument("--save-dpso", default=None,
                   help="Path JSON untuk menyimpan hasil DPSO")

    # Untuk mode yang butuh DPSO hasil terdahulu
    p.add_argument("--dpsofile", default=None,
                   help="Path JSON hasil DPSO yang sudah pernah dihitung")

    # Supervised evaluation
    p.add_argument("--best_config", default=None,
                   help="JSON konfigurasi terbaik hasil tuning_unsupervised (dipakai di eval_sup)")
    p.add_argument("--gt", default=None,
                   help="Path file ground truth komunitas (untuk eval_sup)")
    p.add_argument("--out", default=None,
                   help="Path CSV untuk simpan hasil per-run evaluasi (eval_sup)")
    p.add_argument("--no-auto-alpha", action="store_true",
                   help="Nonaktifkan auto search alpha (eval_sup)")

    return p.parse_args()


def main():
    args = parse_args()

    # 1. Load graph
    G = load_graph(args.graph)

    # 2. Mode pemilihan
    mode = args.mode

    if mode == "dpso":
        # -----------------------------------------
        # MODE 1: Hanya jalankan DPSO multi-run
        # -----------------------------------------
        print("\n=== MODE: DPSO ONLY ===")
        results = run_dpso_multiple(
            G,
            num_runs=args.num_runs,
            num_particles=args.num_particles,
            max_iter=args.max_iter,
            verbose=True
        )

        if args.save_dpso:
            save_dpso_results(args.save_dpso, results)

        print("\n✅ Selesai DPSO only.")
        return

    elif mode == "dpso_best":
        print("\n=== MODE: DPSO BEST CONFIG (dari JSON tuning) ===")
    
        if args.best_dpso is None:
            raise ValueError("Mode 'dpso_best' membutuhkan --best_dpso (json hasil tuning).")
    
        with open(args.best_dpso, "r") as f:
            best = json.load(f)
    
        # Ambil parameter best
        w   = float(best["w"])
        c1  = float(best["c1"])
        c2  = float(best["c2"])
        rho = float(best["rho"])
        p_  = float(best["p"])
        num_particles = int(float(best["num_particles"]))
        max_iter      = int(float(best["max_iter"]))
    
        print("📌 Best DPSO params:")
        print(f"  w={w}, c1={c1}, c2={c2}, rho={rho}, p={p_}, "
              f"num_particles={num_particles}, max_iter={max_iter}")
    
        # Jalankan DPSO (seed RANDOM karena tidak diset)
        results = run_dpso_multiple(
            G,
            num_runs=args.num_runs,
            num_particles=num_particles,
            max_iter=max_iter,
            w=w, c1=c1, c2=c2,
            rho=rho, p=p_,
            verbose=True
        )
    
        if args.save_dpso:
            save_dpso_results(args.save_dpso, results)
        else:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            auto_path = f"dpso_best_{ts}.json"
            save_dpso_results(auto_path, results)
    
        print("\n✅ Selesai DPSO BEST CONFIG.")
        return


    elif mode == "tune_unsup":
        # -----------------------------------------
        # MODE 2: Tuning UNSUPERVISED (butuh dpsofile)
        # -----------------------------------------
        print("\n=== MODE: UNSUPERVISED TUNING (pakai DPSO results dari file) ===")

        if args.dpsofile is None:
            raise ValueError("Mode 'tune_unsup' membutuhkan --dpsofile (hasil DPSO).")

        results = load_dpso_results(args.dpsofile)

        # Path folder tempat menyimpan hasil tuning
        output_folder = "/home/toto/Eska/HybridMSEFCD"

        # Pastikan folder ada
        os.makedirs(output_folder, exist_ok=True)

        # Buat full path untuk CSV dan JSON
        csv_path = os.path.join(output_folder, "tuning_unsupervised_hybrid_dpso_msefcd.csv")
        json_path = os.path.join(output_folder, "best_config.json")

        # Jalankan tuning unsupervised
        df_unsup, best_cfg, final_csv_path, final_json_path = tune_hybrid_msefcd_params_unsupervised(
            G=G,
            dpso_results=results,
            iter_smooth=12,
            iter_enhance=7,
            min_comm_size=1,
            save_csv_path=csv_path,          # simpan ke folder yang kamu mau
            best_config_json_path=json_path  # simpan best config di folder itu
        )

        print("\n📊 Ringkasan konfigurasi terbaik (UNSUPERVISED):")
        print(best_cfg)

        print(f"\n📄 File CSV tuning: {final_csv_path}")
        print(f"📄 File best_config JSON: {final_json_path}")

        print("\n✅ Selesai tuning unsupervised.")
        return

    elif mode == "dpso_plus_tune_unsup":
        # -----------------------------------------
        # MODE 3: DPSO + Tuning UNSUPERVISED dalam satu kali jalan
        # -----------------------------------------
        print("\n=== MODE: DPSO + UNSUPERVISED TUNING ===")

        # 3.1 Jalankan DPSO dulu
        results = run_dpso_multiple(
            G,
            num_runs=args.num_runs,
            num_particles=args.num_particles,
            max_iter=args.max_iter,
            verbose=True
        )

        # Opsional simpan hasil DPSO
        if args.save_dpso:
            save_dpso_results(args.save_dpso, results)
            dpso_path = args.save_dpso
        else:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            dpso_path = f"dpso_results_{ts}.json"
            save_dpso_results(dpso_path, results)

        print(f"\n💾 DPSO results disimpan di: {dpso_path}")

        # 3.2 Tuning unsupervised di atas hasil DPSO tersebut
        output_folder = "/home/toto/Eska/HybridMSEFCD"
        os.makedirs(output_folder, exist_ok=True)
        csv_path = os.path.join(output_folder, "tuning_unsupervised_hybrid_dpso_msefcd.csv")
        json_path = os.path.join(output_folder, "best_config.json")

        df_unsup, best_cfg, final_csv_path, final_json_path = tune_hybrid_msefcd_params_unsupervised(
            G=G,
            dpso_results=results,
            iter_smooth=12,
            iter_enhance=7,
            min_comm_size=1,
            save_csv_path=csv_path,
            best_config_json_path=json_path,
        )

        print("\n📊 Ringkasan konfigurasi terbaik (UNSUPERVISED):")
        print(best_cfg)
        print(f"\n📄 File CSV tuning: {final_csv_path}")
        print(f"📄 File best_config JSON: {final_json_path}")
        print("\n✅ Selesai DPSO + tuning unsupervised.")
        return
    
    elif mode == "tune_dpso_disease":
        # -----------------------------------------
        # MODE BARU: Tuning DPSO untuk dataset penyakit (unsupervised)
        # -----------------------------------------
        print("\n=== MODE: TUNING DPSO UNTUK DATASET PENYAKIT (unsupervised) ===")

        # Load graph penyakit
        # Catatan: file CSV disease harus dibaca dengan fungsi loader khusus
        ext = os.path.splitext(args.graph)[1].lower()
        
        if ext == ".csv":
            print("📂 Memuat dataset penyakit (CSV)...")
            G = load_disease_csv(args.graph)
        else:
            print("📂 Memuat graph biasa...")
            G = load_graph(args.graph)
    
        print(f"Graph loaded: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    
        # Ambil only komponen terbesar
        components = list(nx.connected_components(G))
        components_sorted = sorted(components, key=len, reverse=True)
        
        print(f"Jumlah komponen: {len(components_sorted)}")
        print("Ukuran tiap komponen:", [len(c) for c in components_sorted])
    
        largest_component = components_sorted[0]
        G_big = G.subgraph(components_sorted[0]).copy()
        print(f"Gunakan komponen terbesar: {G_big.number_of_nodes()} nodes")
    
        # ==== TUNING DPSO ====
        from tuning_dpso_disease import tune_dpso_hyperparams_disease
        
        # Jalankan tuning
        df_tune, best_cfg = tune_dpso_hyperparams_disease(
            G=G_big,
            runs_per_cfg=2,
            base_seed=123,
            verbose=True
        )
    
        # Simpan OUT
        if args.out:
            df_tune.to_csv(args.out, index=False)
            print(f"\n💾 Hasil tuning DPSO disimpan ke: {args.out}")
    
        # Simpan hyperparameter terbaik
        best_json = "best_dpso_config.json"
        if args.out:
            # simpan berdampingan
            best_json = args.out.replace(".csv", "_best.json")
    
        with open(best_json, "w") as f:
            json.dump(best_cfg, f, indent=2)
        print(f"💾 Hyperparameter terbaik disimpan ke: {best_json}")
    
        print("\n🏁 Tuning DPSO selesai.\n")
        return

    elif mode == "eval_sup":
        # -----------------------------------------
        # MODE 4: Evaluasi SUPERVISED (butuh GT + DPSO results)
        # -----------------------------------------
        print("\n=== MODE: SUPERVISED EVALUATION (dengan ground truth) ===")

        if args.dpsofile is None:
            raise ValueError("Mode 'eval_sup' membutuhkan --dpsofile (hasil DPSO).")
        if args.gt is None:
            raise ValueError("Mode 'eval_sup' membutuhkan --gt (ground truth communities).")

        # 4.1 Load DPSO results
        results = load_dpso_results(args.dpsofile)

        # 4.2 Load parameter dari best_config JSON (kalau ada)
        if args.best_config is not None:
            with open(args.best_config, "r") as f:
                best_cfg = json.load(f)

            print(f"\n📂 Best config loaded dari: {args.best_config}")
            for k, v in best_cfg.items():
                print(f"  - {k}: {v}")

            alpha_self        = best_cfg.get("alpha_self", 0.6)
            beta_neighbor     = best_cfg.get("beta_neighbor", 0.3)
            prior_boost       = best_cfg.get("prior_boost", 0.08)
            k_comm_candidates = best_cfg.get("k_comm_candidates", 3)
            iter_smooth       = best_cfg.get("iter_smooth", 12)
            iter_enhance      = best_cfg.get("iter_enhance", 7)
            alpha_threshold   = best_cfg.get("alpha_threshold", 0.45)
        else:
            # fallback ke default bila best_config tidak diberikan
            print("\n⚠️ best_config JSON tidak diberikan, pakai parameter DEFAULT.")
            alpha_self        = 0.6
            beta_neighbor     = 0.3
            prior_boost       = 0.08
            k_comm_candidates = 3
            iter_smooth       = 12
            iter_enhance      = 7
            alpha_threshold   = 0.45

        # 4.3 Hitung fuzzy membership (Hybrid MSEFCD) dengan parameter di atas
        print("\n== Running compute_fuzzy_membership_from_DPSO_MSEFCD_full ==")
        fuzzy_results = compute_fuzzy_membership_from_DPSO_MSEFCD_full(
            G, results,
            r=2.0,
            iter_smooth=iter_smooth,
            iter_enhance=iter_enhance,
            k_comm_candidates=k_comm_candidates,
            prior_boost=prior_boost,
            min_comm_size=1,
            alpha_self=alpha_self,
            beta_neighbor=beta_neighbor,
            verbose=True,
            membership_threshold=alpha_threshold,
        )

        # 4.4 Load ground truth
        gt_name_comms = load_gt_communities(args.gt, G)
        print(f"Loaded {len(gt_name_comms)} GT communities (node names).")

        # 4.5 Optional: cek overlapping nodes
        _ = detect_overlapping(fuzzy_results, alpha=alpha_threshold)

        # 4.6 Evaluasi (precision, recall, F1, ONMI pairwise) TANPA langsung save CSV
        summary = evaluate_all(
            fuzzy_results,
            G,
            gt_name_comms,
            auto_alpha=False,                 # jangan cari alpha lagi
            fixed_alpha=alpha_threshold,      # pakai alpha dari best_config
            verbose=True,
            save_csv=None                     # <=== jangan tulis CSV di sini
        )

        print("\n📊 Ringkasan (supervised - pairwise + ONMI_LFK):")
        for k, v in summary.items():
            if k != "per_run":
                print(f"{k}: {v}")

        # DataFrame per-run dari evaluate_all (precision, recall, f1, onmi, num_pred_comms)
        df_pairwise = summary["per_run"].copy()

        # 4.7 Evaluasi OVERLAPPING: ONMI_LFK, Omega, Comm_P/R/F1 (community-level)
        print("\n=== Evaluasi Overlapping (ONMI_LFK, Omega, Community-level P/R/F1) ===")
        overlap_metrics_per_run = []

        for fr in fuzzy_results:
            run_no = fr.get("run", len(overlap_metrics_per_run) + 1)
            fm = fr["fuzzy_membership"]

            # Pastikan membership berbentuk dict label->value lengkap & ter-normalisasi
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

            # Konversi ke komunitas hard dengan threshold fuzzy
            pred_comms = fuzzy_to_communities(fm, alpha=alpha_threshold)
            if not pred_comms:
                print(f"Run {run_no:02d}: tidak ada komunitas prediksi (skip overlapping metrics)")
                continue

            scores = evaluate_overlapping_all(G, gt_name_comms, pred_comms)

            # simpan juga nomor run di dict
            scores_with_run = {"run": run_no}
            scores_with_run.update(scores)
            overlap_metrics_per_run.append(scores_with_run)

            print(f"\nRun {run_no:02d} Overlapping metrics:")
            for k, v in scores.items():
                print(f"  {k}: {v:.4f}")

        # 4.8 Gabungkan semua metrik ke SATU CSV
        # 4.8 Gabungkan semua metrik ke SATU CSV (tanpa ONMI_LFK duplikat)
        if overlap_metrics_per_run:
            df_overlap = pd.DataFrame(overlap_metrics_per_run)

            # Hapus ONMI_LFK karena sudah sama dengan kolom "onmi" di pairwise metrics
            if "ONMI_LFK" in df_overlap.columns:
                df_overlap = df_overlap.drop(columns=["ONMI_LFK"])

            # Merge berdasarkan "run"
            df_merged = pd.merge(df_pairwise, df_overlap, on="run", how="left")

            # Hitung mean & std untuk semua kolom numerik
            numeric_cols = [c for c in df_merged.columns if c != "run"]

            mean_row = {"run": "mean"}
            std_row  = {"run": "std"}

            for col in numeric_cols:
                mean_row[col] = df_merged[col].mean()
                std_row[col]  = df_merged[col].std()

            df_merged = pd.concat([df_merged, 
                                   pd.DataFrame([mean_row, std_row])],
                                   ignore_index=True)

            # Simpan ke CSV
            if args.out is not None:
                df_merged.to_csv(args.out, index=False)
                print(f"\n💾 Semua metrik lengkap disimpan ke: {args.out}")

        else:
            print("\n⚠️ Tidak ada run dengan komunitas prediksi untuk overlapping metrics.")
            if args.out is not None:
                df_pairwise.to_csv(args.out, index=False)
                print(f"\n💾 Hanya metrik pairwise disimpan ke: {args.out}")


        print("\n✅ Selesai evaluasi supervised.")
        return


    else:
        raise ValueError(f"Mode tidak dikenal: {mode}")


if __name__ == "__main__":
    main()
