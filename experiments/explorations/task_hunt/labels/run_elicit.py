"""Pod-side ELICITATION RUNNER — generates a scaffold's corpus with a
pinned open-weights model and writes the standard stream + receipt.

    python -m experiments.explorations.task_hunt.labels.run_elicit \
        --scaffold evalage --model meta-llama/Llama-3.1-8B-Instruct \
        --n-docs 400 --out-tag v1

Backend: vLLM if importable (batched, fast), else transformers. Both
paths are seeded and record the resolved model commit sha, so the
corpus is exactly reproducible from the receipt.

Turn-major batching: all N documents advance ONE turn per step, so
generation is batched across documents while staying causally correct
within each document. Event turns are inserted by the scaffold and are
never model-authored — the event can therefore never be "the model
noticed", which is the visible-cue trap `retryesc`'s
``is_failure_turn`` rule already refuses.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from . import elicit_lib as el
from . import evalage_lib as ev
from . import sycgen_lib as sg
from .lib import doc_split

HERE = Path(__file__).resolve().parent
SYS_ASSISTANT = "You are a helpful assistant. Answer concisely."
USER_PROMPT = ("Continue the conversation as the USER. Ask one short "
               "follow-up question about {topic}. Output only the "
               "question.")


def _resolve_sha(model_id: str) -> str:
    try:
        from huggingface_hub import HfApi
        return HfApi().model_info(model_id).sha
    except Exception:
        return "unresolved"


class AnthropicBackend:
    """API backend (MATS key, authorized `a74f52cb9` — mac-c may switch
    per-card). Provenance differs from open weights and the card must
    say so: the pin is model-id + API version, NOT a weight sha, so the
    corpus is reproducible-in-expectation, not bit-exact. Key is read
    from the macOS keychain and never printed, filed, or passed as argv.
    """

    def __init__(self, model_id: str, seed: int, temperature: float,
                 top_p: float, tokenizer_id: str = "gpt2"):
        import subprocess as sp
        from anthropic import Anthropic
        from transformers import AutoTokenizer
        key = sp.run(["security", "find-generic-password", "-s",
                      "dmitry-mats-claude-api-key", "-w"],
                     capture_output=True, text=True,
                     check=True).stdout.strip()
        self.client = Anthropic(api_key=key)
        self.model_id, self.seed = model_id, seed
        self.temperature, self.top_p = temperature, top_p
        self.kind = "anthropic"
        self.tok = AutoTokenizer.from_pretrained(tokenizer_id)

    def chat(self, conversations: list[list[dict]],
             max_new_tokens: int) -> list[str]:
        from concurrent.futures import ThreadPoolExecutor
        def one(conv):
            sys_txt = "".join(m["content"] for m in conv
                              if m["role"] == "system")
            msgs = [{"role": m["role"], "content": m["content"]}
                    for m in conv if m["role"] != "system"]
            if not msgs or msgs[0]["role"] != "user":
                msgs = [{"role": "user", "content": "Begin."}] + msgs
            for attempt in range(4):
                try:
                    r = self.client.messages.create(
                        model=self.model_id, max_tokens=max_new_tokens,
                        temperature=self.temperature,
                        system=sys_txt or "You are a helpful assistant.",
                        messages=msgs)
                    return "".join(b.text for b in r.content
                                   if b.type == "text").strip()
                except Exception as e:              # rate limit / transient
                    if attempt == 3:
                        print(f"[api] giving up: {type(e).__name__}",
                              flush=True)
                        return ""
                    import time as _t
                    _t.sleep(2 ** attempt)
            return ""
        with ThreadPoolExecutor(max_workers=8) as ex:
            return list(ex.map(one, conversations))


class Backend:
    def __init__(self, model_id: str, seed: int, temperature: float,
                 top_p: float):
        self.model_id, self.seed = model_id, seed
        self.temperature, self.top_p = temperature, top_p
        self.kind = None
        try:
            from vllm import LLM, SamplingParams          # noqa: F401
            self.kind = "vllm"
        except Exception:
            self.kind = "transformers"
        if self.kind == "vllm":
            from vllm import LLM
            self.llm = LLM(model=model_id, seed=seed,
                           gpu_memory_utilization=0.90,
                           enable_prefix_caching=True)
            self.tok = self.llm.get_tokenizer()
        else:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
            self.tok = AutoTokenizer.from_pretrained(model_id)
            if self.tok.pad_token is None:
                self.tok.pad_token = self.tok.eos_token
            self.model = AutoModelForCausalLM.from_pretrained(
                model_id, torch_dtype=torch.bfloat16, device_map="auto")
            torch.manual_seed(seed)

    def chat(self, conversations: list[list[dict]],
             max_new_tokens: int) -> list[str]:
        """One batched turn for many conversations."""
        prompts = [self.tok.apply_chat_template(
            c, tokenize=False, add_generation_prompt=True)
            for c in conversations]
        if self.kind == "vllm":
            from vllm import SamplingParams
            sp = SamplingParams(temperature=self.temperature,
                                top_p=self.top_p,
                                max_tokens=max_new_tokens, seed=self.seed)
            outs = self.llm.generate(prompts, sp)
            return [o.outputs[0].text.strip() for o in outs]
        import torch
        res = []
        for i in range(0, len(prompts), 16):
            batch = prompts[i:i + 16]
            enc = self.tok(batch, return_tensors="pt", padding=True,
                           truncation=True, max_length=4096).to(
                self.model.device)
            with torch.no_grad():
                out = self.model.generate(
                    **enc, max_new_tokens=max_new_tokens, do_sample=True,
                    temperature=self.temperature, top_p=self.top_p,
                    pad_token_id=self.tok.pad_token_id)
            for j in range(len(batch)):
                gen = out[j][enc["input_ids"].shape[1]:]
                res.append(self.tok.decode(gen,
                                           skip_special_tokens=True).strip())
        return res


def run_evalage(be: Backend, n_docs: int, seed: int, max_new: int):
    rng = np.random.default_rng(seed)
    plans = ev.evalage_plan(rng, n_docs)
    # turn-major: each doc keeps its own transcript of (role, text, is_event)
    docs = [{"plan": p, "turns": []} for p in plans]
    max_turns = max(p["n_turns"] for p in plans)
    for t in range(max_turns):
        live = [d for d in docs if t < d["plan"]["n_turns"]]
        if not live:
            break
        # ── USER turn: a cue turn (exact template) or a topic question
        u_needed, u_idx = [], []
        for d in live:
            if t in d["plan"]["cue_turns"]:
                d["turns"].append(("user", ev.cue_text(rng), True))
            else:
                u_idx.append(d)
                u_needed.append([{"role": "user",
                                  "content": USER_PROMPT.format(
                                      topic=d["plan"]["topic"])}])
        if u_needed:
            for d, txt in zip(u_idx, be.chat(u_needed, 48)):
                d["turns"].append(("user", txt or
                                   f"Tell me more about "
                                   f"{d['plan']['topic']}.", False))
        # ── ASSISTANT turn, conditioned on the full transcript
        convs = [[{"role": "system", "content": SYS_ASSISTANT}]
                 + [{"role": r, "content": x} for r, x, _ in d["turns"]]
                 for d in live]
        for d, txt in zip(live, be.chat(convs, max_new)):
            d["turns"].append(("assistant", txt, False))
        print(f"[turn {t + 1}/{max_turns}] {len(live)} docs live",
              flush=True)
    return docs


# ── sycgen (generation card sycgen/GENERATION_CARD.md; mac-d executor,
# assignment ca7ed51eb, ratified 58a92efbf) ──────────────────────────
# Scaffold constants live FROZEN in sycgen_lib. Question seeds are the
# vendored pinned file below (meg-tong/sycophancy-eval @ 9a16942…,
# datasets/are_you_sure.jsonl) — sha asserted at load, so the corpus
# can never silently drift from the pin.
SYCGEN_SEEDS_FILE = "sycgen_seeds_are_you_sure.jsonl"
SYCGEN_SEEDS_SHA256 = ("16e034c2ec6a6145c0058863a7c0f41fee5ffa7f"
                       "9f0391547ae3685e713f115f")
SYCGEN_FILLER = "I see."          # alternation-preserving hard-fail filler


def _load_sycgen_seeds() -> dict[str, list[str]]:
    import hashlib
    p = HERE / SYCGEN_SEEDS_FILE
    digest = hashlib.sha256(p.read_bytes()).hexdigest()
    assert digest == SYCGEN_SEEDS_SHA256, \
        f"seeds sha mismatch: {digest} != pinned {SYCGEN_SEEDS_SHA256}"
    by_dom: dict[str, list[str]] = {}
    with open(p) as f:
        for line in f:
            r = json.loads(line)
            by_dom.setdefault(r["base"].get("dataset", "?"), []).append(
                r["prompt"][0]["content"])
    return by_dom


def run_sycgen(be, n_docs: int, seed: int, max_new: int):
    """Generator-mode sycophancy corpus. USER side is ALL scaffold
    material (pinned seed questions + the frozen challenge template) —
    the model authors ONLY assistant turns, so the event can never be
    model-authored. Per-conversation draw order: plan (n_ex, challenge
    mask, lengths — sycgen_lib's frozen order), THEN domain, THEN
    questions; the domain/question draws never consult the mask, which
    is the evalage independent-topic rule in this scaffold's terms."""
    rng = np.random.default_rng(seed)
    ch_len = len(be.tok(sg.SYCGEN_CHALLENGE_TEXT,
                        add_special_tokens=False)["input_ids"])
    plans = sg.sycgen_plan(rng, ch_len, n_convs=n_docs)
    by_dom = _load_sycgen_seeds()
    doms = sorted(by_dom)
    docs = []
    for msgs in plans:
        n_q = sum(1 for m in msgs if not m["assistant"]
                  and not m["challenge"])
        dom = doms[int(rng.integers(len(doms)))]
        pool = by_dom[dom]
        idx = rng.choice(len(pool), size=n_q, replace=n_q > len(pool))
        docs.append({"plan": {"topic": dom, "msgs": msgs},
                     "qs": [pool[int(i)] for i in idx], "qi": 0,
                     "turns": []})
    max_msgs = max(len(d["plan"]["msgs"]) for d in docs)
    fillers = 0
    for step in range(max_msgs):
        live = [d for d in docs if step < len(d["plan"]["msgs"])]
        if not live:
            break
        gen_docs, gen_convs, caps = [], [], []
        for d in live:
            m = d["plan"]["msgs"][step]
            if m["challenge"]:
                d["turns"].append(("user", sg.SYCGEN_CHALLENGE_TEXT, True))
            elif not m["assistant"]:
                d["turns"].append(("user", d["qs"][d["qi"]], False))
                d["qi"] += 1
            else:
                gen_docs.append(d)
                gen_convs.append(
                    [{"role": "system", "content": SYS_ASSISTANT}]
                    + [{"role": r, "content": x} for r, x, _ in d["turns"]])
                caps.append(m["n"])
        if gen_convs:
            cap = int(min(max(caps), max_new))
            for d, txt in zip(gen_docs, be.chat(gen_convs, cap)):
                if not txt:
                    txt, fillers = SYCGEN_FILLER, fillers + 1
                d["turns"].append(("assistant", txt, False))
        print(f"[msg {step + 1}/{max_msgs}] {len(live)} docs live, "
              f"{len(gen_convs)} generated", flush=True)
    return docs, fillers


def build_stream(docs, tok):
    ids_l, first_l, mask_l, elig_l, off = [], [], [], [], [0]
    topics, texts = [], []
    for d in docs:
        pi, pf, pm, pe = [], [], [], []
        for role, text, is_ev in d["turns"]:
            enc = np.asarray(tok(text, add_special_tokens=False)
                             ["input_ids"], dtype=np.int32)
            if enc.size == 0:
                continue
            f = np.zeros(enc.size, dtype=np.int8)
            m = np.zeros(enc.size, dtype=np.int8)
            e = np.zeros(enc.size, dtype=np.int8)
            if is_ev:
                f[0] = 1
                m[:] = 1                      # cue text never probe-eligible
            elif role == "assistant":
                e[:] = 1                      # probe on assistant tokens
            pi.append(enc); pf.append(f); pm.append(m); pe.append(e)
        if not pi:
            continue
        ids_l.append(np.concatenate(pi)); first_l.append(np.concatenate(pf))
        mask_l.append(np.concatenate(pm)); elig_l.append(np.concatenate(pe))
        off.append(off[-1] + len(ids_l[-1]))
        topics.append(d["plan"]["topic"])
        texts.append(" ".join(x for _, x, _ in d["turns"]))
    return (np.concatenate(ids_l), np.array(off, dtype=np.int64),
            np.concatenate(first_l), np.concatenate(mask_l),
            np.concatenate(elig_l), topics, texts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scaffold", default="evalage",
                    choices=["evalage", "sycgen"])
    ap.add_argument("--model", default="meta-llama/Llama-3.1-8B-Instruct")
    ap.add_argument("--n-docs", type=int, default=ev.N_DOCS)
    ap.add_argument("--seed", type=int, default=ev.SEED)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--max-new", type=int, default=200)
    ap.add_argument("--out-tag", default="v1")
    ap.add_argument("--backend", default="local",
                    choices=["local", "anthropic"])
    a = ap.parse_args()

    if a.backend == "anthropic":
        be = AnthropicBackend(a.model, a.seed, a.temperature, a.top_p)
    else:
        be = Backend(a.model, a.seed, a.temperature, a.top_p)
    print(f"[backend] {be.kind} | {a.model}", flush=True)
    fillers = 0
    if a.scaffold == "sycgen":
        docs, fillers = run_sycgen(be, a.n_docs, a.seed, a.max_new)
    else:
        docs = run_evalage(be, a.n_docs, a.seed, a.max_new)
    ids, doc_off, first, mask, elig, topics, texts = build_stream(docs,
                                                                  be.tok)
    n_docs = len(doc_off) - 1
    split = doc_split(n_docs, seed=a.seed)
    tag = f"{a.scaffold}_{a.out_tag}"
    el.write_stream(HERE / f"elicit_{tag}.npz", token_ids=ids,
                    doc_off=doc_off, event_first=first, event_mask=mask,
                    probe_eligible=elig, doc_split=split)
    # raw text beside the npz — the fix to my own defect. Without this,
    # re-tokenizing for the 3-tokenizer rule needs a decode-recovery
    # step, which is exactly the blocker evalage v1 hit.
    el.save_transcripts(HERE / f"elicit_{tag}_transcripts.json", docs)
    gaps = el.realised_gaps(first, doc_off)
    vc = el.vocabulary_control_check(ids, first, doc_off, topics)
    rec = el.corpus_receipt(
        scaffold_name=a.scaffold, model_id=a.model,
        model_sha=_resolve_sha(a.model), seed=a.seed,
        temperature=a.temperature, top_p=a.top_p, n_docs=n_docs,
        n_tokens=int(ids.size), gaps=gaps, vocab_check=vc,
        first_doc_text=texts[0],
        backend_note=("Anthropic API (MATS key; pin = model-id + API "
                      "version, NOT a weight sha ⇒ reproducible-in-"
                      "expectation, not bit-exact)" if a.backend ==
                      "anthropic" else
                      "pod-hosted open weights (HF cache; weight sha "
                      "pinned ⇒ bit-exact reproducible)"),
        extra={"scaffold_constants": (
            {"gap_turns": [ev.GAP_TURNS_LO, ev.GAP_TURNS_HI],
             "n_cues": [ev.N_CUES_LO, ev.N_CUES_HI],
             "n_topics": len(ev.TOPICS),
             "n_cue_templates": len(ev.EVAL_CUES)}
            if a.scaffold == "evalage" else
            {"n_exchanges": [sg.SYCGEN_N_EX_LO, sg.SYCGEN_N_EX_HI - 1],
             "p_challenge": sg.SYCGEN_P_CHALLENGE,
             "challenge_text": sg.SYCGEN_CHALLENGE_TEXT,
             "seeds_file": SYCGEN_SEEDS_FILE,
             "seeds_sha256": SYCGEN_SEEDS_SHA256,
             "seeds_source": f"{sg.SYCGEN_SOURCE_REPO} @ "
                             f"{sg.SYCGEN_SOURCE_COMMIT} "
                             f"{sg.SYCGEN_SOURCE_DATA}",
             "filler_turns": fillers}),
            "generation_tokenizer": a.model})
    el.save_receipt(HERE / f"elicit_{tag}_receipt.json", rec)
    print(json.dumps({"gaps": gaps, "vocab_control": vc,
                      "n_tokens": int(ids.size), "n_docs": n_docs},
                     indent=1))
    print(f"-> elicit_{tag}.npz + receipt", flush=True)


if __name__ == "__main__":
    main()
