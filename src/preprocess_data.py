# src/preprocess_data.py
import argparse
import yaml
import os
import pandas as pd
import numpy as np
import json
from sklearn.preprocessing import StandardScaler
from src.utils import makedirs, set_seed


def build_sample_csv(path, n=50000, seed=0):
    rng = np.random.RandomState(seed)
    df = pd.DataFrame({
        "C1": rng.randint(0, 1000, size=n).astype(str),
        "C2": rng.randint(0, 500, size=n).astype(str),
        "I1": rng.randn(n),
        "I2": rng.randn(n),
        "action": rng.randint(0, 10, size=n),
        "click": rng.binomial(1, 0.05, size=n).astype(int)
    })
    logits = rng.randn(n, 10)
    probs = np.exp(logits) / np.exp(logits).sum(axis=1, keepdims=True)
    df["p_log"] = probs[np.arange(n), df["action"]]
    makedirs(os.path.dirname(path) or ".")
    df.to_csv(path, index=False)
    print(f"Sample csv generated at {path}")


def main(config_path):
    print("Preprocess: config path:", os.path.abspath(config_path))
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    set_seed(cfg.get("seed", 42))
    csv_path = cfg["data"]["csv_path"]
    if not os.path.exists(csv_path):
        print("No csv found at", csv_path, "- generating a sample.")
        build_sample_csv(csv_path)
    df = pd.read_csv(csv_path)
    ss = cfg["data"].get("subsample", None)
    if ss:
        frac = min(1.0, float(ss) / len(df))
        df = df.sample(frac=frac, random_state=cfg.get("seed", 42)).reset_index(drop=True)
    # encode categoricals
    cat_map = {}
    for c in cfg["data"]["categorical"]:
        df[c] = df[c].astype(str)
        cats = df[c].unique().tolist()
        cat_map[c] = {v: i for i, v in enumerate(cats)}
        df[c] = df[c].map(cat_map[c]).astype(int)
    # numerical scaling
    if cfg["data"]["numerical"]:
        scaler = StandardScaler()
        df[cfg["data"]["numerical"]] = scaler.fit_transform(df[cfg["data"]["numerical"]])
    out_dir = os.path.join("data", "processed")
    makedirs(out_dir)
    df.to_parquet(os.path.join(out_dir, "logs.parquet"), index=False)
    meta = {
        "cat_sizes": {c: int(df[c].max() + 1) for c in cfg["data"]["categorical"]},
        "num_features": len(cfg["data"]["numerical"]),
        "columns": cfg["data"]["categorical"] + cfg["data"]["numerical"]
    }
    with open(os.path.join(out_dir, "meta.json"), "w") as f:
        json.dump(meta, f)
    print("Preprocessing done. Processed data at data/processed")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/config.yaml")
    args = p.parse_args()
    main(args.config)