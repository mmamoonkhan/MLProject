import argparse, yaml, os, json, math
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd

from torch.utils.data import DataLoader, TensorDataset

from .utils import set_seed, makedirs
from .models import PolicyNet

import matplotlib.pyplot as plt
from tqdm import tqdm


def load_processed(cfg):
    df = pd.read_parquet("data/processed/logs.parquet")
    meta = json.load(open("data/processed/meta.json"))

    cats = np.stack(
        [df[c].values for c in cfg["data"]["categorical"]], axis=1
    ).astype(np.int64)

    nums = (
        df[cfg["data"]["numerical"]].values.astype(np.float32)
        if cfg["data"]["numerical"]
        else None
    )

    a_log = df[cfg["data"]["action_col"]].values.astype(np.int64)
    r = df[cfg["data"]["reward_col"]].values.astype(np.float32)

    p_log = (
        df[cfg["data"]["propensity_col"]].values.astype(np.float32)
        if cfg["data"]["propensity_col"] in df.columns
        else None
    )

    return cats, nums, a_log, r, p_log, meta


def reinforce_step(model, opt, cats, nums, a, ret, ent_coef, grad_clip):
    logits, values = model(cats, nums)
    pi = torch.distributions.Categorical(logits=logits)

    logp = pi.log_prob(a)
    ent = pi.entropy().mean()

    # advantage
    adv = (ret - values).detach()
    adv = (adv - adv.mean()) / (adv.std() + 1e-8)

    pg = -(logp * adv).mean()
    v = 0.5 * (values - ret).pow(2).mean()

    loss = pg + v - ent_coef * ent

    opt.zero_grad()
    loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    opt.step()

    return {"loss": float(loss.item()), "entropy": float(ent.item())}


def plot_curves(logs, out_dir):
    plt.figure(figsize=(8, 5))
    plt.plot(logs["loss"], label="loss")
    plt.plot(logs["entropy"], label="entropy")
    plt.xlabel("epoch")
    plt.legend()
    plt.grid()
    plt.savefig(os.path.join(out_dir, "training_curves.png"))
    plt.close()


def main(config_path):
    # Load config
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    print("Loaded config:", os.path.abspath(config_path))
    print("Learning rate type:", type(cfg["train"]["lr"]))

    set_seed(cfg.get("seed", 42))
    device = torch.device(cfg.get("device", "cpu"))

    makedirs(cfg.get("plots_dir", "plots"))
    makedirs(cfg.get("checkpoints_dir", "checkpoints"))

    cats, nums, a_log, r, p_log, meta = load_processed(cfg)

    # Dataset
    X_c = torch.from_numpy(cats)
    X_n = torch.from_numpy(nums) if nums is not None else None
    A = torch.from_numpy(a_log)
    R = torch.from_numpy(r)

    dataset = (
        TensorDataset(X_c, X_n, A, R)
        if X_n is not None
        else TensorDataset(X_c, A, R)
    )
    loader = DataLoader(
        dataset, batch_size=cfg["train"]["batch_size"], shuffle=True
    )

    # Model
    model = PolicyNet(
        [meta["cat_sizes"][c] for c in cfg["data"]["categorical"]],
        meta["num_features"],
        cfg["model"]["emb_dim"],
        cfg["model"]["hidden"],
        cfg["data"]["n_actions"],
    ).to(device)

    # Optimizer
    lr = float(cfg["train"]["lr"])  # <- safety: ensure float
    opt = optim.Adam(model.parameters(), lr=lr)

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=cfg["train"]["epochs"]
    )

    logs = {"loss": [], "entropy": []}

    # Train loop
    for epoch in range(cfg["train"]["epochs"]):
        model.train()
        bs_loss = []
        bs_ent = []

        for batch in loader:
            if X_n is not None:
                cats_b, nums_b, a_b, r_b = batch
                nums_b = nums_b.to(device)
            else:
                cats_b, a_b, r_b = batch
                nums_b = None

            cats_b = cats_b.to(device)
            a_b = a_b.to(device)
            r_b = r_b.to(device)

            out = reinforce_step(
                model,
                opt,
                cats_b,
                nums_b,
                a_b,
                r_b,
                cfg["train"]["entropy_coef"],
                cfg["train"]["grad_clip"],
            )

            bs_loss.append(out["loss"])
            bs_ent.append(out["entropy"])

        logs["loss"].append(np.mean(bs_loss))
        logs["entropy"].append(np.mean(bs_ent))

        scheduler.step()

        if (epoch + 1) % 10 == 0:
            print(
                f"Epoch {epoch+1}/{cfg['train']['epochs']} "
                f"loss={logs['loss'][-1]:.4f} "
                f"ent={logs['entropy'][-1]:.4f}"
            )

        if (epoch + 1) % cfg["train"]["checkpoint_every"] == 0:
            torch.save(
                model.state_dict(),
                f"{cfg.get('checkpoints_dir','checkpoints')}/model_epoch{epoch+1}.pt",
            )

    # Save training curve
    plot_curves(logs, cfg.get("plots_dir", "plots"))

    # Final model
    torch.save(
        {"model_state": model.state_dict(), "logs": logs},
        os.path.join(
            cfg.get("checkpoints_dir", "checkpoints"), "final.pt"
        ),
    )

    print("Training finished. Checkpoints and plots saved.")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/config.yaml")
    args = p.parse_args()
    main(args.config)