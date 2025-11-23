# src/evaluate_policy.py
import argparse
import yaml
import os
import json
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from src.utils import set_seed, makedirs

# robust import: relative preferred, fallback to absolute if necessary
try:
    from .models import PolicyNet
except Exception:
    from models import PolicyNet


def load(cfg):
    df = pd.read_parquet("data/processed/logs.parquet")
    meta = json.load(open("data/processed/meta.json"))
    cats = np.stack([df[c].values for c in cfg["data"]["categorical"]], axis=1).astype(np.int64)
    nums = df[cfg["data"]["numerical"]].values.astype(np.float32) if cfg["data"]["numerical"] else None
    a_log = df[cfg["data"]["action_col"]].values.astype(np.int64)
    r = df[cfg["data"]["reward_col"]].values.astype(np.float32)
    p_log = df[cfg["data"]["propensity_col"]].values.astype(np.float32) if cfg["data"].get("propensity_col") in df.columns else None
    return cats, nums, a_log, r, p_log, meta


def ips_wis_estimate(model, cats, nums, a_log, r, p_log, cfg, device):
    model.eval()
    with torch.no_grad():
        N = len(a_log)
        batch = 2048
        pis = []
        for i in range(0, N, batch):
            cats_b = torch.from_numpy(cats[i:i + batch]).to(device)
            nums_b = torch.from_numpy(nums[i:i + batch]).to(device) if nums is not None else None
            logits, _ = model(cats_b, nums_b)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            pis.append(probs)
        pis = np.vstack(pis)
        pi_a = pis[np.arange(len(a_log)), a_log]
        assert p_log is not None, "propensity p_log required for IPS/WIS evaluation"
        w = (pi_a / p_log).clip(max=cfg["eval"]["ips_clip"])
        ips = (w * r).mean()
        w_norm = w / (w.sum() + 1e-12)
        wis = (w_norm * r).sum()
        return ips, wis, pis


def bootstrap_ci(vals, iters=500, alpha=0.05, seed=0):
    rng = np.random.RandomState(seed)
    n = len(vals)
    boot = []
    for _ in range(iters):
        idx = rng.randint(0, n, size=n)
        boot.append(np.mean(vals[idx]))
    lo = np.percentile(boot, 100 * alpha / 2)
    hi = np.percentile(boot, 100 * (1 - alpha / 2))
    return lo, hi


def main(config_path):
    print("Evaluate: config path:", os.path.abspath(config_path))
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    set_seed(cfg.get("seed", 42))
    device = torch.device(cfg.get("device", "cpu"))
    makedirs(cfg.get("plots_dir", "plots"))

    cats, nums, a_log, r, p_log, meta = load(cfg)

    # init model
    model = PolicyNet([meta["cat_sizes"][c] for c in cfg["data"]["categorical"]], meta["num_features"], cfg["model"]["emb_dim"], cfg["model"]["hidden"], cfg["data"]["n_actions"]).to(device)

    # load final checkpoint (try final.pt then epoch)
    ckpt_final = os.path.join(cfg.get("checkpoints_dir", "checkpoints"), "final.pt")
    if os.path.exists(ckpt_final):
        state = torch.load(ckpt_final, map_location=device, weights_only=False)
        model.load_state_dict(state.get("model_state", state))
    else:
        last_epoch_ck = os.path.join(cfg.get("checkpoints_dir", "checkpoints"), f"model_epoch{cfg['train']['epochs']}.pt")
        if os.path.exists(last_epoch_ck):
            model.load_state_dict(torch.load(last_epoch_ck, map_location=device))
        else:
            raise FileNotFoundError("No checkpoint found. Run training first.")

    ips, wis, pis = ips_wis_estimate(model, cats, nums, a_log, r, p_log, cfg, device)
    print(f"IPS: {ips:.6f}  WIS: {wis:.6f}")

    # per-sample ips contributions and bootstrap CI
    pi_a = pis[np.arange(len(a_log)), a_log]
    w = (pi_a / p_log).clip(max=cfg["eval"]["ips_clip"])
    ipss = (w * r)
    lo, hi = bootstrap_ci(ipss, iters=cfg["eval"]["bootstrap_iters"], seed=cfg.get("seed", 0))
    print(f"IPS bootstrap CI: [{lo:.6f}, {hi:.6f}]")

    # simple bar plot
    plt.figure(figsize=(6, 4))
    plt.bar(["IPS", "WIS"], [ips, wis])
    plt.title("Offline evaluation estimates")
    plt.ylabel("Estimated CTR")
    plt.savefig(os.path.join(cfg.get("plots_dir", "plots"), "ips_wis.png"))
    plt.close()
    print("Evaluation finished. Plots saved to", cfg.get("plots_dir", "plots"))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/config.yaml")
    args = p.parse_args()
    main(args.config)