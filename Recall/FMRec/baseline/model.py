"""FMRec: flow-matching sequential recommender (Liu et al., IJCAI 2025).

Follows the authors' code (github.com/FengLiu-1/FMRec, commit 1c39b17) unless a
deviation is named in the variant README:

* x0-prediction: the network predicts the clean target embedding from
  z_t = t * x_target + (1 - t) * z_noise, so t = 1 is data and t = 0 is noise.
* One Transformer stack split into Decoder1 (first blocks, feeds the history
  reconstruction MLP) and Decoder2 (remaining blocks, feeds the flow output).
* No positional embeddings (they are commented out upstream).
* The noisy target and time embedding enter every history position scaled by a
  per-element weight drawn from N(lambda, lambda) ("uncertainty" injection).
* Loss = L_FM (MSE to target embedding) + alpha * L_CE (softmax over item
  embeddings) + beta * L_MSE (balanced MSE reconstruction of the history set).
* Inference: Euler ODE with velocity pred - z_0 for `sample_steps` steps; rank
  items by inner product of the result with item embeddings.
"""

import math

import torch
from torch import nn
from torch.nn import functional as F

TIME_SCHEDULES = ("mode", "uniform", "logit_normal", "cosmap")
ATTENTION_MODES = ("bidirectional", "causal")


def attention_allowed(sequence, attention):
    """Boolean [B, 1, L, L] mask: query i may attend key j.

    Padding keys are never attended; every query may attend itself so that rows
    of left padding stay finite under causal masking.
    """
    length = sequence.shape[1]
    allowed = sequence.gt(0)[:, None, None, :].expand(-1, 1, length, -1)
    if attention == "causal":
        allowed = allowed & torch.ones(length, length, dtype=torch.bool,
                                       device=sequence.device).tril()
    eye = torch.eye(length, dtype=torch.bool, device=sequence.device)
    return allowed | eye


class SelfAttention(nn.Module):
    def __init__(self, hidden, heads, dropout):
        super().__init__()
        if hidden % heads:
            raise ValueError("hidden size must be divisible by heads")
        self.heads = heads
        self.qkv = nn.Linear(hidden, 3 * hidden)
        self.output = nn.Linear(hidden, hidden)
        self.dropout = dropout

    def forward(self, hidden, allowed):
        batch, length, width = hidden.shape
        q, k, v = self.qkv(hidden).view(batch, length, 3, self.heads, -1).permute(2, 0, 3, 1, 4)
        attended = F.scaled_dot_product_attention(
            q, k, v, attn_mask=allowed, dropout_p=self.dropout if self.training else 0.0)
        return self.output(attended.transpose(1, 2).reshape(batch, length, width))


class Block(nn.Module):
    """Pre-norm Transformer block with a trailing dropout, as upstream."""

    def __init__(self, hidden, heads, dropout):
        super().__init__()
        self.attention_norm = nn.LayerNorm(hidden, eps=1e-12)
        self.attention = SelfAttention(hidden, heads, dropout)
        self.feedforward_norm = nn.LayerNorm(hidden, eps=1e-12)
        self.feedforward = nn.Sequential(nn.Linear(hidden, 4 * hidden), nn.GELU(approximate="tanh"),
                                         nn.Dropout(dropout), nn.Linear(4 * hidden, hidden))
        self.dropout = nn.Dropout(dropout)

    def forward(self, hidden, allowed):
        hidden = hidden + self.dropout(self.attention(self.attention_norm(hidden), allowed))
        hidden = hidden + self.dropout(self.feedforward(self.feedforward_norm(hidden)))
        return self.dropout(hidden)


def timestep_embedding(t, dim, max_period=10000):
    half = dim // 2
    frequencies = torch.exp(-math.log(max_period) * torch.arange(half, device=t.device) / half)
    angles = t[:, None].float() * frequencies[None]
    embedding = torch.cat([torch.cos(angles), torch.sin(angles)], dim=-1)
    return F.pad(embedding, (0, dim % 2))


class FMRec(nn.Module):
    def __init__(self, num_items, hidden=128, heads=4, decoder1_layers=2,
                 decoder2_layers=2, dropout=0.1, embedding_dropout=0.3,
                 lambda_uncertainty=0.001, eps=0.001, time_schedule="mode",
                 mode_scale=1.0, logit_normal_mean=1.0, logit_normal_std=0.6,
                 sample_steps=30, reconstruction_dims=(512, 2048),
                 attention="bidirectional"):
        super().__init__()
        if num_items < 1 or min(hidden, heads, decoder1_layers, decoder2_layers, sample_steps) < 1:
            raise ValueError("sizes must be positive")
        if time_schedule not in TIME_SCHEDULES:
            raise ValueError("unknown time schedule: {}".format(time_schedule))
        if attention not in ATTENTION_MODES:
            raise ValueError("unknown attention mode: {}".format(attention))
        if not 0 < eps < 1:
            raise ValueError("eps must be in (0, 1)")
        self.num_items = num_items
        self.hidden = hidden
        self.lambda_uncertainty = lambda_uncertainty
        self.eps = eps
        self.time_schedule = time_schedule
        self.mode_scale = mode_scale
        self.logit_normal_mean = logit_normal_mean
        self.logit_normal_std = logit_normal_std
        self.sample_steps = sample_steps
        self.attention = attention
        # Row 0 is padding; real items are 1..num_items.
        self.item_embedding = nn.Embedding(num_items + 1, hidden)
        self.embedding_dropout = nn.Dropout(embedding_dropout)
        self.embedding_norm = nn.LayerNorm(hidden, eps=1e-12)
        self.time_mlp = nn.Sequential(nn.Linear(hidden, 4 * hidden), nn.SiLU(),
                                      nn.Linear(4 * hidden, hidden))
        self.decoder1 = nn.ModuleList(Block(hidden, heads, dropout) for _ in range(decoder1_layers))
        self.decoder2 = nn.ModuleList(Block(hidden, heads, dropout) for _ in range(decoder2_layers))
        self.output_dropout = nn.Dropout(dropout)
        self.output_norm = nn.LayerNorm(hidden, eps=1e-12)
        dims = [hidden, *reconstruction_dims, num_items + 1]
        layers = []
        for d_in, d_out in zip(dims[:-1], dims[1:]):
            linear = nn.Linear(d_in, d_out)
            nn.init.xavier_normal_(linear.weight)
            nn.init.zeros_(linear.bias)
            layers += [linear, nn.Tanh()]
        self.reconstruction = nn.Sequential(*layers[:-1])

    def _noise(self, shape, generator, device):
        if generator is None:
            return torch.randn(shape, device=device)
        return torch.randn(shape, generator=generator).to(device)

    def encode_history(self, sequence):
        return self.embedding_norm(self.embedding_dropout(self.item_embedding(sequence)))

    def denoise(self, history, sequence, x_t, t, generator=None):
        """Predict the clean target embedding; also return Decoder1's last state."""
        time = self.time_mlp(timestep_embedding(t * (1 / self.eps - 1), self.hidden))
        weight = self.lambda_uncertainty * (1 + self._noise(history.shape, generator, history.device))
        hidden = history + weight * (x_t + time)[:, None, :]
        allowed = attention_allowed(sequence, self.attention)
        for block in self.decoder1:
            hidden = block(hidden, allowed)
        middle = hidden[:, -1]
        for block in self.decoder2:
            hidden = block(hidden, allowed)
        return self.output_norm(self.output_dropout(hidden))[:, -1], middle

    def sample_time(self, batch, device):
        u = torch.rand(batch, device=device)
        if self.time_schedule == "mode":
            t = 1 - u - self.mode_scale * (torch.cos(math.pi / 2 * u) ** 2 - 1 + u)
        elif self.time_schedule == "uniform":
            t = u
        elif self.time_schedule == "logit_normal":
            t = torch.sigmoid(self.logit_normal_mean + self.logit_normal_std
                              * torch.randn(batch, device=device))
        else:
            t = 1 - 1 / (torch.tan(math.pi / 2 * u) + 1)
        return t * (1 - self.eps) + self.eps

    def scores(self, representation):
        scores = representation @ self.item_embedding.weight.t()
        scores[:, 0] = float("-inf")
        return scores

    def history_targets(self, sequence):
        target = torch.zeros(sequence.shape[0], self.num_items + 1, device=sequence.device)
        target.scatter_(1, sequence, 1.0)
        target[:, 0] = 0
        return target

    def balanced_mse(self, sequence, reconstruction, mask_ratio):
        """MSE over history positives plus as many random cells (upstream: random zeros)."""
        target = self.history_targets(sequence)
        positives = target.bool()
        selected = positives.clone().view(-1)
        count = int(positives.sum().item() * mask_ratio)
        if count:
            selected[torch.randint(selected.numel(), (count,), device=sequence.device)] = True
        selected = selected.view_as(target).float()
        return F.mse_loss(reconstruction * selected, target * selected)

    def loss(self, sequence, target, alpha=0.2, beta=0.4, mask_ratio=1.0):
        history = self.encode_history(sequence)
        target_embedding = self.item_embedding(target)
        noise = torch.randn_like(target_embedding)
        t = self.sample_time(target.shape[0], target.device)
        x_t = t[:, None] * target_embedding + (1 - t[:, None]) * noise
        prediction, middle = self.denoise(history, sequence, x_t, t)
        flow = F.mse_loss(prediction, target_embedding)
        cross_entropy = F.cross_entropy(self.scores(prediction), target)
        reconstruction = self.balanced_mse(sequence, self.reconstruction(middle), mask_ratio)
        total = flow + alpha * cross_entropy + beta * reconstruction
        return {"loss": total, "loss_fm": flow, "loss_ce": cross_entropy,
                "loss_mse": reconstruction}

    @torch.no_grad()
    def sample(self, sequence, steps=None, generator=None):
        """Euler-integrate from noise (t=eps) to data (t=1); return z_1."""
        steps = steps or self.sample_steps
        history = self.encode_history(sequence)
        shape = (sequence.shape[0], self.hidden)
        start = self._noise(shape, generator, sequence.device)
        x = start
        for index in range(steps):
            t = torch.full((shape[0],), index / steps * (1 - self.eps) + self.eps,
                           device=sequence.device)
            prediction, _ = self.denoise(history, sequence, x, t, generator)
            x = x + (prediction - start) / steps
        return x
