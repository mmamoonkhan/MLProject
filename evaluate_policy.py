import torch, pandas as pd, numpy as np, yaml
from train_policy import PolicyNet

def evaluate(config):
    df = pd.read_parquet("data/processed/val.parquet")
    feature_cols = ["hour", "banner_pos", "site_id", "app_id", "device_model", "day"]

    X = torch.tensor(df[feature_cols].values, dtype=torch.float32)
    model = PolicyNet(X.shape[1], config["n_actions"], config["hidden_sizes"])
    model.load_state_dict(torch.load("data/processed/policy.pt"))
    model.eval()

    with torch.no_grad():
        logits, _ = model(X)
        probs = torch.softmax(logits, dim=1)
    pi_a = probs[np.arange(len(df)), df["action"].values].numpy()
    w = np.clip(pi_a / df["p_log"].values, None, config["clip_weight"])
    ips = np.mean(w * df["reward"].values)
    wis = np.sum(w * df["reward"].values) / np.sum(w)
    print(f"✅ IPS = {ips:.6f}, WIS = {wis:.6f}")

if __name__ == "__main__":
    config = yaml.safe_load(open("config.yaml"))
    evaluate(config)
