"""fb_core.py — FrequencyBench sprint core: data, models, training, probes, diagnostics.

Self-contained: only torch / numpy needed. Designed to run on a single GPU.

Tasks (symbols Q_t in Z_M, activations x_t = m_t * u_{Q_t} + sigma * eps_t):
  dc        : hidden class C ~ U([K]); Q_t = C w.p. p, else uniform wrong symbol.
  ac_sign   : hidden sign S ~ U({-1,+1}); Q_t = B + S*v*t mod M  (high-freq v).
  multifreq : hidden velocity Y ~ U(Omega); Q_t = B + Y*t mod M.

Architectures (matched total atoms H and window-level L0 = k_win):
  token_sae : TopK SAE per token (k_win/W per token), probe on stacked window codes.
  txc       : window crosscoder, full temporal band (SpectralTXC with one full band).
  dcac      : SpectralTXC with bands {0} and {1..W-1}, half atoms/budget each.
  multiband : SpectralTXC with 4 DCT bands, quarter atoms/budget each.
  conv      : localized convolutional dictionary, filter length 3, TopK over (h, T).
"""
from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass, field, asdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ─── data ────────────────────────────────────────────────────────────────────


def make_embedding(M: int, d: int, seed: int) -> torch.Tensor:
    """(M, d) orthonormal rows (requires d >= M)."""
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((d, M))
    Q, _ = np.linalg.qr(A)
    return torch.tensor(Q.T[:M], dtype=torch.float32)  # (M, d)


def gen_dc(n: int, W: int, K: int, p: float, rng) -> tuple[np.ndarray, np.ndarray]:
    C = rng.integers(0, K, size=n)
    correct = rng.random((n, W)) < p
    # uniformly random *wrong* symbol where not correct
    offset = rng.integers(1, K, size=(n, W))
    Q = np.where(correct, C[:, None], (C[:, None] + offset) % K)
    return Q.astype(np.int64), C.astype(np.int64)


def gen_cyclic(n: int, W: int, M: int, velocities: np.ndarray, rng):
    """Q_t = B + v*t mod M with v drawn uniformly from `velocities`. Returns Q, label idx."""
    lab = rng.integers(0, len(velocities), size=n)
    v = velocities[lab]
    B = rng.integers(0, M, size=n)
    t = np.arange(W)
    Q = (B[:, None] + v[:, None] * t[None, :]) % M
    return Q.astype(np.int64), lab.astype(np.int64)


@dataclass
class TaskSpec:
    name: str
    M: int                      # alphabet size
    W: int                      # window length
    n_classes: int              # hidden-label cardinality
    sigma: float = 0.25
    d: int = 0                  # embedding dim (defaults to M)
    # dc params
    K: int = 8
    p: float = 0.6
    # cyclic params
    velocities: tuple = ()
    a_loc_star: float = 0.0     # theoretical single-token Bayes accuracy
    embedding: str = "random"   # "random" (orthonormal) | "circle" (R^2 torus)

    def __post_init__(self):
        if self.d == 0:
            self.d = self.M


def make_task(name: str, W: int = 16, sigma: float = 0.25) -> TaskSpec:
    if name == "dc":
        K = 8
        return TaskSpec("dc", M=K, W=W, n_classes=K, sigma=sigma, K=K, p=0.6,
                        a_loc_star=0.6)
    if name == "ac_sign":
        M, v = 17, 8
        return TaskSpec("ac_sign", M=M, W=W, n_classes=2, sigma=sigma,
                        velocities=(v, M - v), a_loc_star=0.5)
    if name == "multifreq":
        M = 101
        omegas = (0, 1, 2, 4, 8, 16, 24, 32, 40, 50)
        return TaskSpec("multifreq", M=M, W=W, n_classes=10, sigma=sigma,
                        velocities=omegas, a_loc_star=0.1)
    if name == "multifreq_circle":
        # circle embedding: velocity y == temporal tone at y/M cycles/token
        M = 101
        omegas = (0, 1, 2, 4, 8, 16, 24, 32, 40, 50)
        return TaskSpec("multifreq_circle", M=M, W=W, n_classes=10, sigma=sigma,
                        d=8, velocities=omegas, a_loc_star=0.1,
                        embedding="circle")
    if name == "phasepair":
        # velocities come in +/- pairs: identical power spectra within a pair
        M = 101
        omegas = (3, 98, 12, 89, 30, 71)  # (3,-3),(12,-12),(30,-30) mod 101
        return TaskSpec("phasepair", M=M, W=W, n_classes=6, sigma=sigma,
                        velocities=omegas, a_loc_star=1.0 / 6)
    raise ValueError(name)


class TaskData:
    """Holds embedding + can sample (X, y, Q) batches on device."""

    def __init__(self, spec: TaskSpec, seed: int):
        self.spec = spec
        self.rng = np.random.default_rng(seed)
        if spec.embedding == "circle":
            rng = np.random.default_rng(1000 + seed)
            A = rng.standard_normal((spec.d, 2))
            R, _ = np.linalg.qr(A)                       # (d, 2) isometry
            ang = 2 * np.pi * np.arange(spec.M) / spec.M
            V = np.stack([np.cos(ang), np.sin(ang)], 1)  # (M, 2)
            self.R = torch.tensor(R, dtype=torch.float32, device=DEVICE)
            self.U = torch.tensor(V @ R.T, dtype=torch.float32, device=DEVICE)
        else:
            self.R = None
            self.U = make_embedding(spec.M, spec.d, seed=1000 + seed).to(DEVICE)

    def sample_symbols(self, n: int):
        s = self.spec
        if s.name == "dc":
            return gen_dc(n, s.W, s.K, s.p, self.rng)
        return gen_cyclic(n, s.W, s.M, np.array(s.velocities), self.rng)

    def embed(self, Q: np.ndarray) -> torch.Tensor:
        Qt = torch.tensor(Q, device=DEVICE)
        X = self.U[Qt]  # (n, W, d)
        X = X + self.spec.sigma * torch.randn_like(X)
        return X

    def sample(self, n: int):
        Q, y = self.sample_symbols(n)
        return self.embed(Q), torch.tensor(y, device=DEVICE), Q


# ─── temporal bases ──────────────────────────────────────────────────────────


def dct_basis(W: int) -> torch.Tensor:
    """(W, W) orthonormal DCT-II basis; row w = frequency index w (0 = DC)."""
    tau = np.arange(W)
    psi = np.zeros((W, W))
    for w in range(W):
        if w == 0:
            psi[w] = np.sqrt(1.0 / W)
        else:
            psi[w] = np.sqrt(2.0 / W) * np.cos(np.pi * (tau + 0.5) * w / W)
    return torch.tensor(psi, dtype=torch.float32)


# ─── models ──────────────────────────────────────────────────────────────────


class TokenSAE(nn.Module):
    """TopK SAE on single tokens."""

    def __init__(self, d: int, h: int, k: int):
        super().__init__()
        self.k = k
        self.b_dec = nn.Parameter(torch.zeros(d))
        self.W_enc = nn.Parameter(torch.randn(h, d) / math.sqrt(d))
        self.b_enc = nn.Parameter(torch.zeros(h))
        self.W_dec = nn.Parameter(torch.randn(d, h) / math.sqrt(h))
        self._normalize_decoder()

    @torch.no_grad()
    def _normalize_decoder(self):
        self.W_dec.data /= self.W_dec.data.norm(dim=0, keepdim=True).clamp(min=1e-8)

    def encode(self, x):  # x: (..., d) -> (..., h)
        pre = (x - self.b_dec) @ self.W_enc.T + self.b_enc
        vals, idx = pre.topk(self.k, dim=-1)
        u = torch.zeros_like(pre)
        u.scatter_(-1, idx, F.relu(vals))
        return u

    def decode(self, u):
        return u @ self.W_dec.T + self.b_dec

    def forward(self, x):  # (B, W, d): treat tokens independently
        u = self.encode(x)
        x_hat = self.decode(u)
        return x_hat, u

    def window_codes(self, x):  # (B, W, d) -> (B, W*h)
        u = self.encode(x)
        return u.flatten(1)


class SpectralTXC(nn.Module):
    """Window crosscoder with per-branch DCT-band-constrained encoder/decoder kernels.

    bands: list of lists of frequency indices. Branch b has h_b atoms, TopK budget k_b.
    Encoder kernels e_{j,tau} = sum_{w in band} C[j,w,:] psi_w(tau); same for decoder.
    Vanilla TXC == single branch with the full band (DCT basis is an orthonormal
    rotation of the tau basis, so this is exactly the standard window crosscoder).
    """

    def __init__(self, d: int, W: int, bands: list[list[int]],
                 h_per_branch: list[int], k_per_branch: list[int]):
        super().__init__()
        self.d, self.W = d, W
        self.bands = bands
        self.h_per_branch = h_per_branch
        self.k_per_branch = k_per_branch
        psi = dct_basis(W)
        self.enc_coef = nn.ParameterList()
        self.dec_coef = nn.ParameterList()
        self.b_enc = nn.ParameterList()
        for band, h in zip(bands, h_per_branch):
            nb = len(band)
            self.enc_coef.append(nn.Parameter(
                torch.randn(h, nb, d) / math.sqrt(nb * d)))
            self.dec_coef.append(nn.Parameter(
                torch.randn(h, nb, d) / math.sqrt(nb * d)))
            self.b_enc.append(nn.Parameter(torch.zeros(h)))
            self.register_buffer(f"psi_{len(self.enc_coef)-1}",
                                 psi[band], persistent=False)
        self.b_dec = nn.Parameter(torch.zeros(W, d))
        self._normalize_decoder()

    def psi_b(self, b: int) -> torch.Tensor:
        return getattr(self, f"psi_{b}")

    @torch.no_grad()
    def _normalize_decoder(self):
        # Parseval: normalizing coefficients == normalizing time-domain kernels
        for D in self.dec_coef:
            n = D.data.flatten(1).norm(dim=1).clamp(min=1e-8)
            D.data /= n[:, None, None]

    def enc_kernels(self, b: int) -> torch.Tensor:
        """(h_b, W, d) time-domain encoder kernels for branch b."""
        return torch.einsum("wt,hwd->htd", self.psi_b(b), self.enc_coef[b])

    def dec_kernels(self, b: int) -> torch.Tensor:
        return torch.einsum("wt,hwd->htd", self.psi_b(b), self.dec_coef[b])

    def encode(self, x):  # (B, W, d) -> list of (B, h_b)
        zs = []
        for b in range(len(self.bands)):
            E = self.enc_kernels(b)                       # (h, W, d)
            pre = torch.einsum("btd,htd->bh", x, E) + self.b_enc[b]
            k = self.k_per_branch[b]
            vals, idx = pre.topk(k, dim=-1)
            z = torch.zeros_like(pre)
            z.scatter_(-1, idx, F.relu(vals))
            zs.append(z)
        return zs

    def decode(self, zs):
        x_hat = self.b_dec.unsqueeze(0)
        for b, z in enumerate(zs):
            D = self.dec_kernels(b)                       # (h, W, d)
            x_hat = x_hat + torch.einsum("bh,htd->btd", z, D)
        return x_hat

    def forward(self, x):
        zs = self.encode(x)
        return self.decode(zs), zs

    def window_codes(self, x):  # (B, W, d) -> (B, sum h_b)
        return torch.cat(self.encode(x), dim=-1)

    def branch_codes(self, x):
        return self.encode(x)


class ConvDict(nn.Module):
    """Localized convolutional dictionary: filters of length L, codes per (atom, pos).

    TopK is over the flattened (h * T) preactivation per window (budget k_win).
    Decode via conv_transpose with per-atom temporal templates.
    """

    def __init__(self, d: int, W: int, h: int, k_win: int, L: int = 3):
        super().__init__()
        self.d, self.W, self.h, self.k_win, self.L = d, W, h, k_win, L
        self.W_enc = nn.Parameter(torch.randn(h, d, L) / math.sqrt(d * L))
        self.b_enc = nn.Parameter(torch.zeros(h))
        self.W_dec = nn.Parameter(torch.randn(h, d, L) / math.sqrt(L))
        self.b_dec = nn.Parameter(torch.zeros(d))
        self._normalize_decoder()

    @torch.no_grad()
    def _normalize_decoder(self):
        n = self.W_dec.data.flatten(1).norm(dim=1).clamp(min=1e-8)
        self.W_dec.data /= n[:, None, None]

    def encode(self, x):  # (B, W, d) -> (B, h, T)
        xc = x.transpose(1, 2)  # (B, d, W)
        pre = F.conv1d(xc, self.W_enc, padding=self.L // 2) + self.b_enc[None, :, None]
        B = pre.shape[0]
        flat = pre.flatten(1)                              # (B, h*W)
        vals, idx = flat.topk(self.k_win, dim=-1)
        z = torch.zeros_like(flat)
        z.scatter_(-1, idx, F.relu(vals))
        return z.view(B, self.h, self.W)

    def decode(self, z):  # (B, h, T) -> (B, W, d)
        # conv_transpose1d weight wants (in_ch=h, out_ch=d, L); W_dec is (h, d, L),
        # giving x_hat[:, :, s] = sum_{j,t} z[:, j, t] * W_dec[j, :, s - t + L//2],
        # i.e. each activation places its temporal template around its center.
        x_hat = F.conv_transpose1d(z, self.W_dec, padding=self.L // 2)
        return x_hat.transpose(1, 2) + self.b_dec

    def forward(self, x):
        z = self.encode(x)
        return self.decode(z), z

    def window_codes(self, x):  # (B, h*W)
        return self.encode(x).flatten(1)


def build_model(arch: str, d: int, W: int, H: int, k_win: int) -> nn.Module:
    if arch == "token_sae":
        return TokenSAE(d, H, max(1, k_win // W))
    if arch == "txc":
        return SpectralTXC(d, W, [list(range(W))], [H], [k_win])
    if arch == "dcac":
        return SpectralTXC(d, W, [[0], list(range(1, W))],
                           [H // 2, H // 2], [k_win // 2, k_win // 2])
    if arch == "multiband":
        # 4 bands: DC, low, mid, high
        edges = [1, 1 + (W - 1) // 3, 1 + 2 * (W - 1) // 3, W]
        bands = [[0], list(range(1, edges[1])),
                 list(range(edges[1], edges[2])), list(range(edges[2], W))]
        return SpectralTXC(d, W, bands, [H // 4] * 4, [k_win // 4] * 4)
    if arch == "conv":
        return ConvDict(d, W, H, k_win)
    if arch == "conv7":
        return ConvDict(d, W, H, k_win, L=7)
    raise ValueError(arch)


# ─── training ────────────────────────────────────────────────────────────────


@dataclass
class TrainResult:
    fvu: float
    l0: float
    steps: int
    seconds: float


def train_model(model: nn.Module, data: TaskData, steps: int = 4000,
                batch: int = 512, lr: float = 1e-3, log_every: int = 500,
                pregen: int = 100_000) -> TrainResult:
    model.to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    # pregenerate a training pool, sample batches from it
    Xpool, _, _ = data.sample(pregen)
    t0 = time.time()
    fvu = float("nan")
    for step in range(steps):
        idx = torch.randint(0, Xpool.shape[0], (batch,), device=DEVICE)
        x = Xpool[idx]
        x_hat, z = model(x)
        mse = (x_hat - x).pow(2).sum(-1).mean()
        loss = mse
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        model._normalize_decoder()
        if step % log_every == 0 or step == steps - 1:
            with torch.no_grad():
                var = (x - x.mean(dim=0, keepdim=True)).pow(2).sum(-1).mean()
                fvu = (mse / var).item()
    with torch.no_grad():
        x = Xpool[:4096]
        x_hat, z = model(x)
        mse = (x_hat - x).pow(2).sum(-1).mean()
        var = (x - x.mean(dim=0, keepdim=True)).pow(2).sum(-1).mean()
        fvu = (mse / var).item()
        if isinstance(z, list):
            l0 = sum((zz > 0).float().sum(-1).mean().item() for zz in z)
        else:
            # (B, W, h) token codes or (B, h, T) conv codes: window-level L0
            l0 = (z > 0).float().flatten(1).sum(-1).mean().item()
    return TrainResult(fvu=fvu, l0=l0, steps=steps, seconds=time.time() - t0)


# ─── probes ──────────────────────────────────────────────────────────────────


def _standardize(train: torch.Tensor, test: torch.Tensor):
    mu = train.mean(0, keepdim=True)
    sd = train.std(0, keepdim=True).clamp(min=1e-6)
    return (train - mu) / sd, (test - mu) / sd


def fit_probe(feats_tr, y_tr, feats_te, y_te, n_classes: int,
              hidden: int = 0, epochs: int = 300, lr: float = 1e-2,
              wd: float = 1e-4, seed: int = 0) -> dict:
    """Linear (hidden=0) or 1-hidden-layer MLP probe, full-batch Adam, on device.

    Returns dict with train/test acc and per-class test acc.
    """
    torch.manual_seed(seed)
    ftr, fte = _standardize(feats_tr.float(), feats_te.float())
    D = ftr.shape[1]
    if hidden:
        probe = nn.Sequential(nn.Linear(D, hidden), nn.ReLU(),
                              nn.Linear(hidden, n_classes)).to(DEVICE)
    else:
        probe = nn.Linear(D, n_classes).to(DEVICE)
    opt = torch.optim.Adam(probe.parameters(), lr=lr, weight_decay=wd)
    for _ in range(epochs):
        logits = probe(ftr)
        loss = F.cross_entropy(logits, y_tr)
        opt.zero_grad()
        loss.backward()
        opt.step()
    with torch.no_grad():
        acc_tr = (probe(ftr).argmax(-1) == y_tr).float().mean().item()
        pred_te = probe(fte).argmax(-1)
        acc_te = (pred_te == y_te).float().mean().item()
        per_class = []
        for c in range(n_classes):
            m = y_te == c
            per_class.append(((pred_te[m] == c).float().mean().item()
                              if m.any() else float("nan")))
        # confusion counts: rows = true class, cols = predicted
        conf = torch.zeros(n_classes, n_classes, dtype=torch.long)
        for c in range(n_classes):
            m = y_te == c
            if m.any():
                conf[c] = torch.bincount(pred_te[m].cpu(),
                                         minlength=n_classes)
    return {"acc_train": acc_tr, "acc_test": acc_te, "per_class": per_class,
            "confusion": conf.tolist()}


# ─── oracles & ceilings ──────────────────────────────────────────────────────


def decode_symbols(X: torch.Tensor, U: torch.Tensor) -> torch.Tensor:
    """Nearest-embedding symbol decode. X: (n, W, d) -> (n, W) ints."""
    return torch.einsum("nwd,md->nwm", X, U).argmax(-1)


def oracle_accuracy(spec: TaskSpec, X, y, U, R=None) -> float:
    """Strong reference pipeline on noisy activations.

    dc: nearest-symbol decode then majority vote (ML for this channel).
    random-embedding cyclic: nearest-symbol decode, consecutive-difference votes.
    circle cyclic: DFT matched filter — project onto the circle plane, take
    |DFT| at each candidate velocity's frequency, argmax (ML tone detection).
    """
    if spec.embedding == "circle":
        proj = X @ R                                    # (n, W, 2)
        c = torch.complex(proj[..., 0], proj[..., 1])   # (n, W)
        t = torch.arange(spec.W, device=X.device, dtype=torch.float32)
        vel = torch.tensor(spec.velocities, device=X.device,
                           dtype=torch.float32)
        phase = torch.exp(-2j * np.pi * vel[:, None] * t[None, :] / spec.M)
        scores = torch.einsum("nw,vw->nv", c, phase).abs()
        pred = scores.argmax(-1)
        return (pred == y).float().mean().item()
    Qh = decode_symbols(X, U)  # (n, W)
    if spec.name == "dc":
        counts = F.one_hot(Qh, spec.M).sum(1)  # (n, M)
        pred = counts.argmax(-1)
        return (pred == y).float().mean().item()
    vel = torch.tensor(spec.velocities, device=X.device)
    diffs = (Qh[:, 1:] - Qh[:, :-1]) % spec.M          # (n, W-1)
    votes = (diffs.unsqueeze(-1) == vel[None, None, :]).sum(1)  # (n, n_classes)
    pred = votes.argmax(-1)
    return (pred == y).float().mean().item()


# ─── diagnostics ─────────────────────────────────────────────────────────────


def freq_profile(model: nn.Module, W: int) -> np.ndarray | None:
    """(H_total, W) FreqFrac per atom (encoder kernels in DCT basis); None if N/A."""
    psi = dct_basis(W).to(DEVICE)
    if isinstance(model, SpectralTXC):
        rows = []
        for b in range(len(model.bands)):
            E = model.enc_kernels(b)                  # (h, W, d)
            coef = torch.einsum("wt,htd->hwd", psi, E)
            en = coef.pow(2).sum(-1)                  # (h, W)
            rows.append(en / en.sum(-1, keepdim=True).clamp(min=1e-12))
        return torch.cat(rows).detach().cpu().numpy()
    if isinstance(model, ConvDict):
        # zero-pad filters to W, center them, measure DCT energy
        E = model.W_enc                               # (h, d, L)
        pad = torch.zeros(E.shape[0], E.shape[1], W, device=E.device)
        s = (W - model.L) // 2
        pad[:, :, s:s + model.L] = E
        coef = torch.einsum("wt,hdt->hwd", psi, pad)
        en = coef.pow(2).sum(-1)
        return (en / en.sum(-1, keepdim=True).clamp(min=1e-12)).detach().cpu().numpy()
    return None


def project_to_band(model: SpectralTXC, band: list[int]) -> SpectralTXC:
    """Return a copy of `model` with encoder kernels projected onto DCT band
    `band` (coefficients at frequencies outside the band zeroed)."""
    import copy
    m2 = copy.deepcopy(model)
    with torch.no_grad():
        for b, mband in enumerate(m2.bands):
            keep = torch.tensor([w in band for w in mband], device=DEVICE)
            m2.enc_coef[b].data[:, ~keep, :] = 0.0
    return m2


def shuffle_windows(X: torch.Tensor, seed: int = 0) -> torch.Tensor:
    """Independent random within-window permutation per window (destroys order
    per-sample; a single shared permutation would be learnable by the probe)."""
    g = torch.Generator(device="cpu")
    g.manual_seed(seed)
    r = torch.rand(X.shape[0], X.shape[1], generator=g).to(X.device)
    idx = torch.argsort(r, dim=1)
    return torch.gather(X, 1, idx[..., None].expand_as(X))


def shuffle_sensitivity(model, X, n_classes, y, probe_seed=0) -> dict:
    """Code displacement under per-window shuffle / reversal."""
    with torch.no_grad():
        z = model.window_codes(X[:2048])
        z_sh = model.window_codes(shuffle_windows(X[:2048], seed=7))
        z_rev = model.window_codes(X[:2048].flip(1))
        denom = z.norm(dim=-1).clamp(min=1e-8)
        return {
            "shuffle_rel_change": ((z - z_sh).norm(dim=-1) / denom).mean().item(),
            "reversal_rel_change": ((z - z_rev).norm(dim=-1) / denom).mean().item(),
        }


# ─── utility ─────────────────────────────────────────────────────────────────


def s_temp(acc, a_loc, a_oracle):
    if a_oracle - a_loc < 1e-9:
        return float("nan")
    return (acc - a_loc) / (a_oracle - a_loc)
