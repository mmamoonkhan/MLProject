# src/models.py
import torch
import torch.nn as nn
from torch.distributions import Categorical


class PolicyNet(nn.Module):
    def __init__(self, cat_sizes, num_features, emb_dim, hidden, n_actions):
        """
        cat_sizes: list of ints (vocab sizes for each categorical column)
        num_features: number of numerical features (int)
        emb_dim: embedding dimension per categorical
        hidden: list of hidden dims for trunk
        n_actions: number of discrete actions (K)
        """
        super().__init__()
        # embeddings for each categorical
        self.embs = nn.ModuleList([nn.Embedding(s, emb_dim) for s in cat_sizes]) if len(cat_sizes) > 0 else nn.ModuleList()
        trunk_in = emb_dim * len(cat_sizes) + (num_features or 0)
        layers = []
        prev = trunk_in
        for h in hidden:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU())
            prev = h
        if len(layers) > 0:
            self.trunk = nn.Sequential(*layers)
        else:
            # trivial identity trunk if no hidden layers
            self.trunk = nn.Identity()
        self.logits = nn.Linear(prev, n_actions)
        self.value = nn.Linear(prev, 1)

    def forward(self, cats, nums=None):
        # cats: long tensor [B, n_cat]
        # nums: float tensor [B, n_num] or None
        emb_list = []
        if len(self.embs) > 0:
            # cats may be on CPU or device, embeddings will handle device
            for i, e in enumerate(self.embs):
                emb_list.append(e(cats[:, i]))
        x_parts = emb_list + ([nums] if nums is not None else [])
        if len(x_parts) == 0:
            x = nums  # both None unlikely, but safe
        else:
            x = torch.cat(x_parts, dim=1)
        z = self.trunk(x)
        return self.logits(z), self.value(z).squeeze(-1)


def select_action_from_logits(logits, invalid_mask=None):
    """
    logits: [B, K] tensor
    invalid_mask: optional boolean tensor same shape where True -> invalid
    Returns: action tensor [B], log_prob tensor [B], entropy tensor [B]
    """
    if invalid_mask is not None:
        logits = logits.masked_fill(invalid_mask, float("-1e9"))
    pi = Categorical(logits=logits)
    a = pi.sample()
    return a, pi.log_prob(a), pi.entropy()