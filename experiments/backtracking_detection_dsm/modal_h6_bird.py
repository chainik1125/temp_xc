"""H6 -- BIRD memorization-phase test on real activation windows, plus the
exemplar-memorization diagnostic from the 2026-08-13 proposal response.

Implements the reviewer note's own estimators
(experiments/dsm-txc/theory/bird_diffusion_sae_txc_proposal.tex Sec. 7):

  channel        C_gamma = sqrt(gamma) P X + eps,  eps ~ N(0, I_r)
                 (eq. linear-gaussian-restriction; P fixed random
                 orthonormal rows, r in {8, 32, 128})
  posterior      p(i|C) prop exp(-||C - sqrt(gamma) P X^(i)||^2 / 2)
                 (the empirical posterior of Sec. 7.1); mean posterior
                 entropy over many draws with C generated from a random
                 training example
  (a) Gaussian   MI <= 1/2 log det(I_r + gamma P Sigma P^T)
                 (eq. gaussian-mi-bound; empirical covariance of the n
                 training windows, eigenvalues via SVD of the projected,
                 centered n x r matrix -- the full 24576^2 covariance is
                 never formed)
  (b) I-MMSE     MI(gamma) = 1/2 int_0^gamma mmse(u) du
                 (eq. i-mmse-integral; mmse of the EMPIRICAL posterior-mean
                 denoiser of U = PX, Monte-Carlo on the same grid)
  (c) isotropic  gamma_c = (n^{2/r} - 1) / nu   (eq. critical-snr)

Data: distill-model (DeepSeek-R1-Distill-Llama-8B) resid_L10 T=6 windows over
the ward traces -- capture recipe identical to modal_p1p3.py (prompt +
full_response, truncation at 1200 tokens, hidden_states[11]).  To respect the
note's i.i.d. caveat the n-window training sets are drawn from DISTINCT
traces: one window per trace for n <= 300, and evenly-strided windows with
guaranteed >= 6-token separation (typically ~85-300 tokens) otherwise.

Exemplar diagnostic (response doc "mutual test" section): the w6mix
dictionaries trained on ALL 300 traces (psc_train_sae.trace_docs cycles the
whole file), so held-out trace windows are obtained by REGENERATING traces
with the distill model on a subset of the same prompts -- the reference
recipe of src/bench/venhoff/generate_traces.py::generate_traces_transformers
(chat template + decode lines copied verbatim), with the single deliberate
divergence that sampling is on (temperature 0.6, seed 20260813): the original
traces were greedy, and a greedy rerun would reproduce the training set
instead of holding it out.  Codes are TopK-96 on base-Llama resid_L10
windows of the TRAINING doc convention (prompt + "\\n\\n" + full_response).

Modal launch traps (all five bit this project before):
  1. spawn-style entrypoints still need --detach;
  2. name the entrypoint explicitly (::main) -- bare `modal run file.py`
     with >= 2 entrypoints prints a list and exits 1, which looks launched;
  3. ONE spawned function per detached invocation (detach keeps only the
     last-triggered function);
  4. the volume artifact is the only honest launch evidence -- monitors must
     check backtracking_eval/h6_bird_phase.json, not process/log state;
  5. pollers do not notify a stopped session -- the caller drives the
     pipeline with sleeps in its own shell.

    uvx modal run --detach \
        experiments/backtracking_detection_dsm/modal_h6_bird.py::main
"""

import json
import pathlib

import modal

try:
    ROOT = pathlib.Path(__file__).resolve().parents[2]
except IndexError:
    ROOT = pathlib.Path("/work")

app = modal.App("h6-bird-phase")
vol = modal.Volume.from_name("diffusion-txc", create_if_missing=True)
hf_cache = modal.Volume.from_name("sae-deadlatent-hf-cache", create_if_missing=True)
hf_secret = modal.Secret.from_name("hf-token")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("torch==2.5.1", "numpy>=2.0", "transformers==4.46.2",
                 "accelerate", "sentencepiece", "huggingface_hub", "hf_transfer")
    .env({"HF_HOME": "/hf", "HF_HUB_ENABLE_HF_TRANSFER": "1"})
    .add_local_file(str(ROOT / "results/ward_backtracking/traces.json"),
                    "/work/traces.json")
)

T, K, H, D = 6, 96, 16384, 4096
D_EFF = T * D                                     # 24576
DISTILL = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
BASE_CANDIDATES = ["NousResearch/Meta-Llama-3.1-8B", "meta-llama/Llama-3.1-8B"]
N_GRID = [64, 256, 1024, 4096]
R_GRID = [8, 32, 128]
G_GRID_LOG10 = (-4.0, 5.0, 55)                    # gamma*nu, 6 pts/decade
N_DRAWS = 1024                                    # MC draws per gamma point
MAX_TOK = 1200                                    # p1p3 capture truncation
GEN_EVERY = 5                                     # fresh traces: prompts [::5]
GEN_MAX_NEW = 1500
GEN_TEMP = 0.6
GEN_SEED = 20260813
DIAG_MAX_TOK = 2048
REF_STRIDE = 8                                    # reference windows
TRAIN_Q_PER_TRACE = 10
TEST_Q_PER_TRACE = 50
W6MIX = {"dsm": "/vol/txc_w6_mix/txc_dsm/dsm_s0.pt",
         "recon": "/vol/txc_w6_mix/txc_recon/recon_s0.pt"}
OUT = "/vol/backtracking_eval/h6_bird_phase.json"


def _flush(out: dict) -> None:
    p = pathlib.Path(OUT)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, indent=1, default=str))
    vol.commit()


def _crossing(xs, ys, target, decreasing):
    """gamma at which ys crosses target, linear interp in log10(gamma)."""
    import numpy as np
    xs, ys = np.asarray(xs), np.asarray(ys)
    yy = -ys if decreasing else ys
    tt = -target if decreasing else target
    idx = np.where((yy[:-1] < tt) & (yy[1:] >= tt))[0]
    if len(idx) == 0:
        return None
    i = int(idx[0])
    f = (tt - yy[i]) / (yy[i + 1] - yy[i] + 1e-30)
    return float(10 ** (np.log10(xs[i]) + f * (np.log10(xs[i + 1]) -
                                               np.log10(xs[i]))))


@app.function(image=image, gpu="A100-40GB", timeout=14400, memory=65536,
              volumes={"/vol": vol, "/hf": hf_cache}, secrets=[hf_secret],
              cpu=8)
def run_h6() -> dict:
    import time

    import numpy as np
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dev = "cuda"
    t0 = time.time()
    out: dict = {"meta": {
        "capture": f"distill resid_L10 (hidden_states[11]) T={T}, prompt+"
                   f"full_response, truncation max_length={MAX_TOK} "
                   "(identical to modal_p1p3.py)",
        "channel": "C = sqrt(gamma) P X + N(0, I_r); P random orthonormal "
                   "rows, seed 1000+r; X raw (uncentered) window, d_eff "
                   f"= {D_EFF}",
        "posterior": "p(i|C) prop exp(-||C - sqrt(gamma) P X_i||^2/2) over "
                     "the n training windows; C drawn from a uniformly "
                     f"random training example; {N_DRAWS} draws per gamma",
        "gamma_grid": f"gamma = g/nu, log10 g in [{G_GRID_LOG10[0]}, "
                      f"{G_GRID_LOG10[1]}], {G_GRID_LOG10[2]} points; nu = "
                      "tr(P Sigma_n P^T)/r per cell (empirical, biased /n)",
        "mi_gauss": "0.5 * sum log(1 + gamma * lambda_i), lambda_i "
                    "eigenvalues of the projected centered n-window "
                    "covariance via SVD (Gram trick; full covariance never "
                    "formed)",
        "mi_immse": "0.5 * int_0^gamma mmse(u) du; mmse = MC MSE of the "
                    "empirical posterior-mean denoiser of U = PX; "
                    "trapezoid on the log grid + exact mmse(0)=tr segment",
        "gamma_c_iso": "(n^{2/r} - 1)/nu (note eq. critical-snr)",
        "selection": "distinct traces; 1 window/trace for n<=300, else "
                     "even strided windows per trace, min separation "
                     ">= 6 tokens (recorded per cell)",
        "n_grid": N_GRID, "r_grid": R_GRID,
    }}

    # ---------------- stage 1: distill capture (p1p3 recipe) ----------------
    tok = AutoTokenizer.from_pretrained(DISTILL)
    model = AutoModelForCausalLM.from_pretrained(
        DISTILL, torch_dtype=torch.bfloat16).to(dev).eval()
    traces = json.loads(pathlib.Path("/work/traces.json").read_text())
    texts = [t["prompt"] + t["full_response"] for t in traces]
    per_trace: list[np.ndarray] = []
    with torch.no_grad():
        for i, tx in enumerate(texts):
            ids = tok(tx, return_tensors="pt", truncation=True,
                      max_length=MAX_TOK)["input_ids"].to(dev)
            hs = model(ids, output_hidden_states=True).hidden_states[11][0]
            per_trace.append(hs.float().to(torch.float16).cpu().numpy())
            if i % 50 == 0:
                print(f"[distill capture] {i}/300 ({time.time()-t0:.0f}s)",
                      flush=True)
    lens = np.array([a.shape[0] for a in per_trace])
    out["meta"]["n_traces"] = len(per_trace)
    out["meta"]["trace_token_lens"] = {
        "min": int(lens.min()), "median": float(np.median(lens)),
        "max": int(lens.max())}
    print(f"[distill capture] done, tokens min/med/max "
          f"{lens.min()}/{np.median(lens):.0f}/{lens.max()}", flush=True)

    # ---------------- stage 2: phase test -----------------------------------
    def build_windows(sel):
        """sel: list[(trace_idx, start)] -> (len(sel), D_EFF) float32 GPU."""
        Xn = torch.empty(len(sel), D_EFF, dtype=torch.float32)
        for j, (ti, s) in enumerate(sel):
            Xn[j] = torch.from_numpy(
                per_trace[ti][s:s + T].astype(np.float32)).reshape(-1)
        return Xn.to(dev)

    def select_windows(n):
        rng = np.random.default_rng(500 + n)
        n_tr = len(per_trace)
        sel, seps = [], []
        if n <= n_tr:
            for ti in rng.choice(n_tr, size=n, replace=False):
                L = per_trace[ti].shape[0]
                sel.append((int(ti), int(rng.integers(0, L - T + 1))))
        else:
            base, rem = divmod(n, n_tr)
            order = rng.permutation(n_tr)
            for pos, ti in enumerate(order):
                m = base + (1 if pos < rem else 0)
                L = per_trace[ti].shape[0]
                seg = (L - T) / m
                starts = sorted(
                    int(round(j * seg + rng.uniform(0, max(seg - T, 1.0))))
                    for j in range(m))
                starts = [min(s, L - T) for s in starts]
                sel += [(int(ti), s) for s in starts]
                if m > 1:
                    seps += list(np.diff(sorted(starts)))
        info = {"n": n, "n_distinct_traces": len({t for t, _ in sel}),
                "windows_per_trace_max": int(max(
                    [sum(1 for t, _ in sel if t == ti)
                     for ti in {t for t, _ in sel}])),
                "min_sep_tokens": int(min(seps)) if seps else None,
                "median_sep_tokens": float(np.median(seps)) if seps else None}
        return sel[:n], info

    projections = {}
    for r in R_GRID:
        g = torch.Generator(device=dev).manual_seed(1000 + r)
        Gm = torch.randn(D_EFF, r, generator=g, device=dev)
        Q, _ = torch.linalg.qr(Gm)
        projections[r] = Q.T.contiguous()          # (r, D_EFF)

    g_lo, g_hi, g_npt = G_GRID_LOG10
    g_grid = np.logspace(g_lo, g_hi, int(g_npt))
    out["phase"] = {"cells": {}}
    for n in N_GRID:
        sel, sel_info = select_windows(n)
        Xn = build_windows(sel)
        for r in R_GRID:
            cell = f"n{n}_r{r}"
            Y = (Xn @ projections[r].T).double()               # (n, r)
            Yc = Y - Y.mean(0, keepdim=True)
            lam = (torch.linalg.svdvals(Yc) ** 2 / n).cpu().numpy()
            tr_cov = float(lam.sum())
            nu = tr_cov / r
            gammas = g_grid / nu
            log_n = float(np.log(n))
            gen = torch.Generator(device=dev).manual_seed(9000 + 17 * n + r)
            ent_mean, ent_p90, top1_mass, mmse = [], [], [], []
            for gam in gammas:
                sg = float(np.sqrt(gam))
                idx = torch.randint(0, n, (N_DRAWS,), generator=gen,
                                    device=dev)
                C = sg * Y[idx] + torch.randn(N_DRAWS, r, generator=gen,
                                              device=dev, dtype=torch.float64)
                d2 = torch.cdist(C, sg * Y) ** 2                # (B, n)
                logits = -0.5 * d2
                logp = logits - torch.logsumexp(logits, dim=1, keepdim=True)
                p = logp.exp()
                Hb = -torch.special.xlogy(p, p).sum(1)
                Uhat = p @ Y
                err = (Y[idx] - Uhat).pow(2).sum(1)
                ent_mean.append(float(Hb.mean()))
                ent_p90.append(float(Hb.quantile(0.9)))
                top1_mass.append(float(p.max(1).values.mean()))
                mmse.append(float(err.mean()))
            ent_mean = np.array(ent_mean)
            mmse_a = np.array(mmse)
            mi_gauss = 0.5 * np.log1p(np.outer(gammas, lam)).sum(1)
            # I-MMSE cumulative integral; exact mmse(0) = tr(Cov)
            gam_ext = np.concatenate([[0.0], gammas])
            mmse_ext = np.concatenate([[tr_cov], mmse_a])
            mi_immse = 0.5 * np.concatenate(
                [[0.0], np.cumsum(np.diff(gam_ext) *
                                  (mmse_ext[1:] + mmse_ext[:-1]) / 2)])[1:]
            pred_g = np.clip(log_n - mi_gauss, 0, None)
            pred_i = np.clip(log_n - mi_immse, 0, None)
            cellres = {
                "n": n, "r": r, "log_n_nats": log_n, "nu": nu,
                "tr_proj_cov": tr_cov,
                "lambda_top8": [float(x) for x in
                                np.sort(lam)[::-1][:8]],
                "selection": sel_info,
                "gamma": [float(x) for x in gammas],
                "gamma_nu": [float(x) for x in g_grid],
                "entropy_mean_nats": [round(float(x), 5) for x in ent_mean],
                "entropy_p90_nats": [round(float(x), 5) for x in ent_p90],
                "top1_posterior_mass": [round(float(x), 5)
                                        for x in top1_mass],
                "mmse": [float(x) for x in mmse_a],
                "mi_gauss_nats": [round(float(x), 5) for x in mi_gauss],
                "mi_immse_nats": [round(float(x), 5) for x in mi_immse],
                "crossings": {
                    "gamma_entropy_half_logn": _crossing(
                        gammas, ent_mean, log_n / 2, decreasing=True),
                    "gamma_entropy_1nat": _crossing(
                        gammas, ent_mean, 1.0, decreasing=True),
                    "gamma_mi_gauss_eq_logn": _crossing(
                        gammas, mi_gauss, log_n, decreasing=False),
                    "gamma_mi_immse_eq_logn": _crossing(
                        gammas, mi_immse, log_n, decreasing=False),
                    "gamma_c_isotropic": float((n ** (2 / r) - 1) / nu),
                },
                "collapse_residual_mean_abs": {
                    "vs_gauss": float(np.mean(np.abs(ent_mean - pred_g))),
                    "vs_immse": float(np.mean(np.abs(ent_mean - pred_i))),
                },
                "collapse_residual_max_abs": {
                    "vs_gauss": float(np.max(np.abs(ent_mean - pred_g))),
                    "vs_immse": float(np.max(np.abs(ent_mean - pred_i))),
                },
                "grid_crossed_transition": bool(
                    ent_mean[0] > 0.98 * log_n and
                    ent_mean[-1] < 0.05 * log_n),
            }
            out["phase"]["cells"][cell] = cellres
            c = cellres["crossings"]
            print(f"[phase {cell}] nu={nu:.4g} ent {ent_mean[0]:.3f}->"
                  f"{ent_mean[-1]:.4f} | g_ent_half={c['gamma_entropy_half_logn']}"
                  f" g_gauss={c['gamma_mi_gauss_eq_logn']} "
                  f"g_immse={c['gamma_mi_immse_eq_logn']} "
                  f"g_iso={c['gamma_c_isotropic']:.4g}", flush=True)
            _flush(out)
        del Xn
        torch.cuda.empty_cache()

    # ---------------- stage 3: fresh held-out traces -------------------------
    # generate_traces_transformers recipe (src/bench/venhoff/generate_traces
    # .py), chat-template + decode lines verbatim; divergence: sampling on.
    gen_prompts = [(t["question_id"], t["prompt"])
                   for t in traces[::GEN_EVERY]]
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id
    torch.manual_seed(GEN_SEED)
    fresh = []
    bs = 6
    for i in range(0, len(gen_prompts), bs):
        chunk = gen_prompts[i:i + bs]
        prompts = [
            tok.apply_chat_template(
                [{"role": "user", "content": p}],
                tokenize=False,
                add_generation_prompt=True,
            )
            for _, p in chunk
        ]
        inputs = tok(prompts, return_tensors="pt", padding=True,
                     truncation=True).to(dev)
        with torch.no_grad():
            out_ids = model.generate(
                **inputs,
                max_new_tokens=GEN_MAX_NEW,
                do_sample=True,
                temperature=GEN_TEMP,
                top_p=1.0,
                pad_token_id=tok.pad_token_id,
            )
        for (qid, prompt), in_ids, full_ids in zip(chunk, inputs.input_ids,
                                                   out_ids):
            new_tokens = full_ids[in_ids.shape[0]:]
            full_response = tok.decode(new_tokens, skip_special_tokens=True)
            fresh.append({"question_id": qid, "prompt": prompt,
                          "full_response": full_response})
        print(f"[gen] {min(i + bs, len(gen_prompts))}/{len(gen_prompts)} "
              f"({time.time()-t0:.0f}s)", flush=True)
    fl = np.array([len(f["full_response"]) for f in fresh])
    out["fresh_traces"] = {
        "n": len(fresh), "seed": GEN_SEED, "temperature": GEN_TEMP,
        "top_p": 1.0, "max_new_tokens": GEN_MAX_NEW,
        "recipe": "generate_traces_transformers verbatim (chat template + "
                  "skip_special_tokens decode); sampled because the original "
                  "traces were greedy (temp 0.0) and a greedy rerun would "
                  "reproduce the training set",
        "resp_chars": {"min": int(fl.min()), "median": float(np.median(fl)),
                       "max": int(fl.max())},
        "frac_with_end_think": float(np.mean(
            ["</think>" in f["full_response"] for f in fresh])),
        "orig_frac_with_end_think": float(np.mean(
            ["</think>" in t["full_response"] for t in traces])),
    }
    pathlib.Path("/vol/backtracking_eval/h6_fresh_traces.json").write_text(
        json.dumps(fresh, indent=1))
    _flush(out)
    del model
    torch.cuda.empty_cache()
    del per_trace

    # ---------------- stage 4: base capture + exemplar diagnostic ------------
    base_model = None
    last_err = None
    for mid in BASE_CANDIDATES:
        try:
            btok = AutoTokenizer.from_pretrained(mid, use_fast=True)
            base_model = AutoModelForCausalLM.from_pretrained(
                mid, torch_dtype=torch.bfloat16).to(dev).eval()
            out["exemplar_capture_model"] = mid
            break
        except Exception as e:                                 # noqa: BLE001
            last_err = e
    if base_model is None:
        raise RuntimeError(f"no base Llama: {last_err}")

    def base_capture(docs):
        arrs = []
        with torch.no_grad():
            for i, tx in enumerate(docs):
                ids = btok(tx, return_tensors="pt", truncation=True,
                           max_length=DIAG_MAX_TOK,
                           add_special_tokens=False)["input_ids"].to(dev)
                hs = base_model(ids, output_hidden_states=True
                                ).hidden_states[11][0]
                arrs.append(hs.float().to(torch.float16).cpu().numpy())
                if i % 60 == 0:
                    print(f"[base capture] {i}/{len(docs)} "
                          f"({time.time()-t0:.0f}s)", flush=True)
        return arrs

    orig_docs = [t["prompt"] + "\n\n" + t["full_response"] for t in traces]
    fresh_docs = [f["prompt"] + "\n\n" + f["full_response"] for f in fresh]
    orig_acts = base_capture(orig_docs)
    fresh_acts = base_capture(fresh_docs)
    del base_model
    torch.cuda.empty_cache()

    # dictionaries: dsm + recon from the volume, random init as the floor
    dicts = {}
    for kind, path in W6MIX.items():
        sd = torch.load(path, map_location=dev, weights_only=True)
        dicts[kind] = (sd["W_enc"].float(), sd["b_enc"].float(),
                       sd["b_dec"].float())
    torch.manual_seed(0)                      # run_controls random_dict_seed
    dicts["random"] = (
        (torch.randn(D_EFF, H) / D_EFF ** 0.5).to(dev),
        torch.zeros(H, device=dev), torch.zeros(D_EFF, device=dev))

    @torch.no_grad()
    def codes_for(acts, sel, W_enc, b_enc, b_dec):
        """sel: list[(idx, start)] -> unit-norm TopK-96 codes fp16 (N, H)."""
        Z = torch.empty(len(sel), H, dtype=torch.float16, device=dev)
        n_zero = 0
        for s0 in range(0, len(sel), 1024):
            block = sel[s0:s0 + 1024]
            Xb = torch.empty(len(block), D_EFF, dtype=torch.float32)
            for j, (ti, s) in enumerate(block):
                Xb[j] = torch.from_numpy(
                    acts[ti][s:s + T].astype(np.float32)).reshape(-1)
            Xb = Xb.to(dev)
            p = (Xb - b_dec) @ W_enc + b_enc
            v, i = p.topk(K, dim=1)
            z = torch.zeros_like(p).scatter(1, i, torch.relu(v))
            nrm = z.norm(dim=1, keepdim=True)
            n_zero += int((nrm.squeeze(1) == 0).sum())
            Z[s0:s0 + len(block)] = (z / nrm.clamp_min(1e-8)).half()
        return Z, n_zero

    rng = np.random.default_rng(GEN_SEED + 1)
    ref_sel, ref_ti, ref_start = [], [], []
    for ti, a in enumerate(orig_acts):
        for s in range(0, a.shape[0] - T + 1, REF_STRIDE):
            ref_sel.append((ti, s))
            ref_ti.append(ti)
            ref_start.append(s)
    trainq_sel = []
    for ti, a in enumerate(orig_acts):
        starts = np.arange(REF_STRIDE // 2, a.shape[0] - T + 1, REF_STRIDE)
        take = rng.choice(len(starts), size=min(TRAIN_Q_PER_TRACE,
                                                len(starts)), replace=False)
        trainq_sel += [(ti, int(starts[j])) for j in take]
    testq_sel = []
    for fi, a in enumerate(fresh_acts):
        starts = np.arange(0, a.shape[0] - T + 1, REF_STRIDE)
        take = rng.choice(len(starts), size=min(TEST_Q_PER_TRACE,
                                                len(starts)), replace=False)
        testq_sel += [(fi, int(starts[j])) for j in take]
    fresh_qid_to_orig_ti = {}
    for fi, f in enumerate(fresh):
        for ti, t in enumerate(traces):
            if t["question_id"] == f["question_id"]:
                fresh_qid_to_orig_ti[fi] = ti
                break
    ref_ti_t = torch.tensor(ref_ti, device=dev)
    ref_start_t = torch.tensor(ref_start, device=dev)
    out["exemplar"] = {"protocol": {
        "codes": f"TopK-{K} on base-Llama resid_L10 T={T} windows, doc = "
                 "prompt + '\\n\\n' + full_response (the psc mixed-stream "
                 f"training convention), truncation {DIAG_MAX_TOK} tokens",
        "reference_set": f"stride-{REF_STRIDE} windows of all 300 original "
                         f"(training) traces, n_ref={len(ref_sel)}",
        "train_queries": f"{TRAIN_Q_PER_TRACE}/trace at stride offset "
                         f"{REF_STRIDE//2} (mod {REF_STRIDE})",
        "test_queries": f"{TEST_Q_PER_TRACE}/fresh-trace from "
                        f"{len(fresh_acts)} held-out regenerated traces",
        "metric": "top1 = max cosine to allowed refs; top1_mass = top1 / "
                  "sum relu(cos). Variants: overlap_excluded (train "
                  "queries drop same-trace refs with |dstart| < 6; test "
                  "queries keep all refs) and trace_excluded (train: whole "
                  "same trace removed; test: whole same-question original "
                  "trace removed)",
        "n_train_q": len(trainq_sel), "n_test_q": len(testq_sel),
    }, "arms": {}}

    @torch.no_grad()
    def query_stats(Zq, q_ti, q_start, Zref, variant, is_test):
        stats = {"top1": [], "mass": [], "margin": [], "same_trace_top1": []}
        q_ti_t = torch.tensor(q_ti, device=dev)
        q_start_t = torch.tensor(q_start, device=dev)
        for s0 in range(0, Zq.shape[0], 512):
            # half-precision GEMM (fp32 accumulate); cast the (b, n_ref)
            # result only -- never the full reference matrix
            S = (Zq[s0:s0 + 512] @ Zref.T).float()             # (b, n_ref)
            if is_test:
                omap = torch.tensor(
                    [fresh_qid_to_orig_ti.get(int(x), -1)
                     for x in q_ti[s0:s0 + 512]], device=dev)
                same = ref_ti_t[None, :] == omap[:, None]
                mask = same if variant == "trace_excluded" else \
                    torch.zeros_like(same)
            else:
                same = ref_ti_t[None, :] == q_ti_t[s0:s0 + 512, None]
                if variant == "trace_excluded":
                    mask = same
                else:
                    close = (ref_start_t[None, :] -
                             q_start_t[s0:s0 + 512, None]).abs() < T
                    mask = same & close
            S = S.masked_fill(mask, float("-inf"))
            top2 = S.topk(2, dim=1).values
            pos_sum = torch.relu(S).sum(1)     # relu clamps -inf to 0
            stats["top1"] += top2[:, 0].tolist()
            stats["margin"] += (top2[:, 0] - top2[:, 1]).tolist()
            mass = top2[:, 0] / pos_sum.clamp_min(1e-8)
            stats["mass"] += mass.tolist()
            arg = S.argmax(1)
            stats["same_trace_top1"] += (
                ref_ti_t[arg] == (q_ti_t[s0:s0 + 512] if not is_test else
                                  omap)).float().tolist()
            del S
        def agg(v):
            v = np.array(v)
            return {"mean": float(v.mean()), "median": float(np.median(v)),
                    "p10": float(np.quantile(v, .1)),
                    "p90": float(np.quantile(v, .9))}
        return {"top1_cos": agg(stats["top1"]),
                "top1_mass": agg(stats["mass"]),
                "top1_margin": agg(stats["margin"]),
                "frac_top1_same_trace": float(np.mean(
                    stats["same_trace_top1"]))}

    for kind, (W_enc, b_enc, b_dec) in dicts.items():
        Zref, z0r = codes_for(orig_acts, ref_sel, W_enc, b_enc, b_dec)
        Ztr, z0t = codes_for(orig_acts, trainq_sel, W_enc, b_enc, b_dec)
        Zte, z0e = codes_for(fresh_acts, testq_sel, W_enc, b_enc, b_dec)
        arm = {"n_zero_codes": {"ref": z0r, "train_q": z0t, "test_q": z0e}}
        for variant in ("overlap_excluded", "trace_excluded"):
            tr = query_stats(Ztr, [t for t, _ in trainq_sel],
                             [s for _, s in trainq_sel], Zref, variant, False)
            te = query_stats(Zte, [t for t, _ in testq_sel],
                             [s for _, s in testq_sel], Zref, variant, True)
            arm[variant] = {
                "train": tr, "test": te,
                "gap_top1_cos_mean": round(tr["top1_cos"]["mean"] -
                                           te["top1_cos"]["mean"], 5),
                "gap_top1_mass_mean": round(tr["top1_mass"]["mean"] -
                                            te["top1_mass"]["mean"], 6),
            }
            print(f"[exemplar {kind}/{variant}] train top1 "
                  f"{tr['top1_cos']['mean']:.4f} test "
                  f"{te['top1_cos']['mean']:.4f} gap "
                  f"{arm[variant]['gap_top1_cos_mean']:+.4f}", flush=True)
        out["exemplar"]["arms"][kind] = arm
        _flush(out)
        del Zref, Ztr, Zte
        torch.cuda.empty_cache()

    out["_runtime_s"] = round(time.time() - t0, 1)
    _flush(out)
    print(f"H6 ALL DONE in {out['_runtime_s']}s", flush=True)
    return out


@app.local_entrypoint()
def main():
    call = run_h6.spawn()
    print("SPAWNED h6:", call.object_id, flush=True)
