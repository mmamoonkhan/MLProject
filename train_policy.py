import torch, torch.nn as nn, torch.optim as optim
import pandas as pd, yaml
from tqdm import tqdm
from utils import set_seed

class PolicyNet(nn.Module):
    def __init__(self, input_dim, n_actions, hidden):
        super().__init__()
        layers, last = [], input_dim
        for h in hidden:
            layers += [nn.Linear(last, h), nn.ReLU()]
            last = h
        self.backbone = nn.Sequential(*layers)
        self.logits = nn.Linear(last, n_actions)
        self.value = nn.Linear(last, 1)

    def forward(self, x):
        z = self.backbone(x)
        return self.logits(z), self.value(z).squeeze(-1)

def train(config):
    set_seed(config["seed"])
    device = config["device"]

    df = pd.read_parquet("data/processed/train.parquet")
    feature_cols = ["hour", "banner_pos", "site_id", "app_id", "device_model", "day"]
    X = torch.tensor(df[feature_cols].values, dtype=torch.float32).to(device)
    A = torch.tensor(df["action"].values, dtype=torch.long).to(device)
    R = torch.tensor(df["reward"].values, dtype=torch.float32).to(device)

    model = PolicyNet(X.shape[1], config["n_actions"], config["hidden_sizes"]).to(device)
    opt = optim.Adam(model.parameters(), lr=float(config["lr"]))
    for epoch in range(config["epochs"]):
        total_loss = 0
        for i in tqdm(range(0, len(X), config["batch_size"]), desc=f"Epoch {epoch+1}"):
            xb = X[i:i+config["batch_size"]]
            ab, rb = A[i:i+config["batch_size"]], R[i:i+config["batch_size"]]
            logits, v = model(xb)
            dist = torch.distributions.Categorical(logits=logits)
            logp = dist.log_prob(ab)
            adv = rb - v.detach()
            loss_pg = -(logp * adv).mean()
            loss_v = nn.functional.mse_loss(v, rb)
            loss = loss_pg + 0.5*loss_v - config["entropy_coef"]*dist.entropy().mean()

            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), config["clip_grad_norm"])
            opt.step()
            total_loss += loss.item()
        print(f"Loss @ epoch {epoch+1}: {total_loss:.4f}")

    torch.save(model.state_dict(), "data/processed/policy.pt")
    print("Model trained and saved at data/processed/policy.pt")

if __name__ == "__main__":
    config = yaml.safe_load(open("config.yaml"))
    train(config)
