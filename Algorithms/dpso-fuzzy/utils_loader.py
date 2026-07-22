import pandas as pd
import networkx as nx

def load_disease_csv(path):
    df = pd.read_csv(path)

    required = ["txt", "txt.1", "num", "num.1", "num.2"]
    for c in required:
        if c not in df.columns:
            raise ValueError(f"Kolom '{c}' hilang dari CSV kamu.")

    G = nx.Graph()

    for _, row in df.iterrows():
        src = str(row["txt"])
        dst = str(row["txt.1"])
        weight = float(row["num.2"]) if not pd.isna(row["num.2"]) else 1.0
        G.add_edge(src, dst, weight=weight)

    return G
