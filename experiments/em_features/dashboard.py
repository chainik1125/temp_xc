"""Dashboard for emergent-misalignment steering experiments.

Two classes:
  * ``ExperimentSelector`` discovers frontier-sweep outputs and selects defaults
    (which experiment, which α, which questions).
  * ``SteeringDashboard`` collects example generations and renders the HTML.

The two classes are separated so that the GPU-bound generation step (``collect``)
runs on an h100, dumps a portable JSON, and the rendering step runs anywhere.

RNG matching: ``collect_examples`` reseeds before each generation pass with the
same seed, so at α=0 the steered call produces output bit-identical to the
unsteered call (since 0 × direction = 0). At α≠0 they diverge from the first
token where the steering vector changes the residual stream.

Usage on GPU box:

    python -m experiments.em_features.dashboard collect \\
        --results_root /root/em_features/results \\
        --ckpt /root/em_features/checkpoints/v2_qwen_l15_sae_arditi_k128_step100000.pt \\
        --experiment wang/sae_bundle30_frontier.json \\
        --out /root/em_features/dashboards/sae_wang_bundle30.json

Local rendering:

    python -m experiments.em_features.dashboard render \\
        --data sae_wang_bundle30.json \\
        --out sae_wang_bundle30.html
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import random
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
VENDOR_SRC = REPO_ROOT / "experiments" / "separation_scaling" / "vendor" / "src"
for p in (str(VENDOR_SRC), str(REPO_ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)


def _seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _enable_determinism() -> None:
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _is_finite(x) -> bool:
    if x is None:
        return False
    if isinstance(x, float) and math.isnan(x):
        return False
    return True


def _html_escape(s: str) -> str:
    return (s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
            .replace("\n", "<br>"))


@dataclass
class Experiment:
    name: str
    arch: str           # 'sae' / 'custom_sae' / 'han'
    ckpt_path: Path
    feature_ids: list[int]
    layer: int
    frontier_path: Path
    frontier_rows: list[dict] = field(default_factory=list)

    @classmethod
    def from_frontier_json(cls, frontier_path: Path, ckpt_path: Path) -> "Experiment":
        d = json.loads(Path(frontier_path).read_text())
        meta = d.get("meta", {})
        steerer = meta.get("steerer") or "sae"
        return cls(
            name=Path(frontier_path).stem,
            arch=steerer,
            ckpt_path=Path(ckpt_path),
            feature_ids=list(meta.get("feature_ids", [])),
            layer=int(meta.get("layer", 15)),
            frontier_path=Path(frontier_path),
            frontier_rows=list(d.get("rows", [])),
        )


class ExperimentSelector:
    """Discovers frontier-sweep outputs and picks the default α + question set."""

    def __init__(self, results_root: Path, ckpt_root: Optional[Path] = None):
        self.results_root = Path(results_root)
        self.ckpt_root = Path(ckpt_root) if ckpt_root else None

    def list(self) -> list[str]:
        """Return relative-path names for every frontier JSON found."""
        out = []
        for f in self.results_root.glob("**/*frontier.json"):
            out.append(str(f.relative_to(self.results_root)))
        return sorted(out)

    def get(self, name_or_path: str, ckpt_path: Optional[Path] = None) -> Experiment:
        """Resolve a frontier JSON by name or path. If ``ckpt_path`` is None,
        try to guess from filename; otherwise the caller's value is used."""
        path = Path(name_or_path)
        if not path.is_absolute() and not path.exists():
            cand = self.results_root / name_or_path
            if cand.exists():
                path = cand
            else:
                matches = list(self.results_root.glob(f"**/{name_or_path}"))
                if not matches:
                    matches = list(self.results_root.glob(f"**/{name_or_path}*frontier.json"))
                if not matches:
                    raise FileNotFoundError(f"No frontier JSON matching '{name_or_path}'")
                path = matches[0]
        ckpt = Path(ckpt_path) if ckpt_path else self._guess_ckpt(path)
        return Experiment.from_frontier_json(path, ckpt)

    def _guess_ckpt(self, frontier_path: Path) -> Path:
        if not self.ckpt_root:
            raise ValueError("ckpt_root not set; pass --ckpt explicitly")
        stem = frontier_path.stem
        for suf in ("_frontier", "_encoder_frontier", "_encoder_k1_frontier",
                    "_bundle30_frontier", "_alpha0"):
            if stem.endswith(suf):
                stem = stem[: -len(suf)]
                break
        cand = self.ckpt_root / f"{stem}.pt"
        if cand.exists():
            return cand
        matches = list(self.ckpt_root.glob(f"**/{stem}.pt"))
        if matches:
            return matches[0]
        raise FileNotFoundError(f"No ckpt for {frontier_path}; tried {cand}")

    def best_alpha(self, exp: Experiment, *, coh_floor_frac: float = 0.9,
                   prefer_negative: bool = True) -> float:
        rows = [r for r in exp.frontier_rows if _is_finite(r.get("mean_alignment"))]
        if not rows:
            raise ValueError(f"no valid rows in {exp.name}")
        a0 = next((r for r in rows if abs(r["alpha"]) < 0.01), None)
        if a0 and _is_finite(a0.get("mean_coherence")):
            coh_floor = coh_floor_frac * a0["mean_coherence"]
        else:
            coh_vals = [r["mean_coherence"] for r in rows if _is_finite(r.get("mean_coherence"))]
            coh_floor = coh_floor_frac * (max(coh_vals) if coh_vals else 0.0)
        candidates = [r for r in rows if _is_finite(r.get("mean_coherence"))
                      and r["mean_coherence"] >= coh_floor]
        if not candidates:
            return max(rows, key=lambda r: r["mean_alignment"])["alpha"]
        if prefer_negative:
            neg = [r for r in candidates if r["alpha"] < 0]
            if neg:
                return max(neg, key=lambda r: r["mean_alignment"])["alpha"]
        return max(candidates, key=lambda r: r["mean_alignment"])["alpha"]


class SteeringDashboard:
    """Collects example generations and renders an HTML dashboard."""

    def __init__(self, exp: Experiment, *, subject_model: str = "andyrdt/Qwen2.5-7B-Instruct_bad-medical",
                 base_model: str = "Qwen/Qwen2.5-7B-Instruct", device: str = "cuda"):
        self.exp = exp
        self.subject_model = subject_model
        self.base_model = base_model
        self.device = device
        self._model = None
        self._tok = None
        self._direction = None

    def _ensure_model(self):
        if self._model is not None:
            return
        from open_source_em_features.utils.model_loading import load_model_and_tokenizer
        self._model, self._tok = load_model_and_tokenizer(
            self.subject_model, base_model_id=self.base_model,
            torch_dtype=torch.bfloat16, device=self.device,
        )
        if hasattr(self._model, "merge_and_unload"):
            self._model = self._model.merge_and_unload()
        self._model.eval()

    def _ensure_direction(self):
        if self._direction is not None:
            return
        if self.exp.arch in ("sae", "custom_sae"):
            from sae_day.sae import TopKSAE
            ckpt = torch.load(self.exp.ckpt_path, map_location=self.device, weights_only=False)
            cfg = ckpt["config"]
            sae = TopKSAE(d_in=cfg["d_in"], d_sae=cfg["d_sae"], k=cfg["k"]).to(self.device)
            sae.load_state_dict(ckpt["state_dict"])
            sae.eval()
            rows = sae.W_dec[self.exp.feature_ids].detach()  # (k, d_in)
        elif self.exp.arch == "han":
            from experiments.em_features.architectures.txc_bare_multidistance_contrastive_antidead import (
                TXCBareMultiDistanceContrastiveAntidead,
            )
            ckpt = torch.load(self.exp.ckpt_path, map_location=self.device, weights_only=False)
            cfg = ckpt["config"]
            m = TXCBareMultiDistanceContrastiveAntidead(
                d_in=cfg["d_in"], d_sae=cfg["d_sae"], T=cfg["T"], k=cfg["k"],
                shifts=tuple(cfg.get("shifts", (1, 2))),
                matryoshka_h_size=cfg.get("matryoshka_h_size", cfg["d_sae"] // 5),
                alpha=cfg.get("alpha_contrastive", 1.0),
                aux_k=cfg.get("aux_k", 512),
                dead_threshold_tokens=cfg.get("dead_threshold_tokens", 640_000),
                auxk_alpha=cfg.get("auxk_alpha", 1.0 / 32.0),
            ).to(self.device)
            m.load_state_dict(ckpt["state_dict"])
            m.eval()
            rows = m.W_dec[self.exp.feature_ids, -1, :].detach()  # (k, d_in)
        else:
            raise ValueError(f"unsupported arch '{self.exp.arch}'")
        self._direction = rows.sum(dim=0).to(torch.bfloat16)

    def collect_examples(self, *, alpha_steered: float, n_questions: int = 5,
                         n_rollouts: int = 1, seed: int = 42, judge: bool = True,
                         judge_temp: float = 0.5, max_new_tokens: int = 200) -> dict:
        _enable_determinism()
        self._ensure_model()
        self._ensure_direction()
        from open_source_em_features.pipeline.longform_steering import (
            generate_longform_completions, load_em_dataset,
        )
        em = load_em_dataset()
        questions = [d["messages"][0]["content"] for d in em][:n_questions]

        def _gen(alpha: float):
            _seed_all(seed)
            return generate_longform_completions(
                model=self._model, tokenizer=self._tok, questions=questions,
                steering_direction=self._direction, magnitude=float(alpha),
                layer_idx=int(self.exp.layer), n_generations=int(n_rollouts),
                max_new_tokens=int(max_new_tokens), temperature=1.0,
            )

        gens_unsteered = _gen(0.0)
        gens_steered = _gen(alpha_steered)

        align_un = align_st = coh_un = coh_st = None
        if judge:
            from experiments.em_features.gemini_judge import evaluate_generations_with_gemini
            align_un, coh_un = asyncio.run(evaluate_generations_with_gemini(
                gens_unsteered, temperature=judge_temp,
            ))
            align_st, coh_st = asyncio.run(evaluate_generations_with_gemini(
                gens_steered, temperature=judge_temp,
            ))

        pairs = []
        for i, (gu, gs) in enumerate(zip(gens_unsteered, gens_steered)):
            pairs.append({
                "question": gu["question"],
                "rollout_idx": i % max(1, n_rollouts),
                "unsteered_answer": gu["answer"],
                "steered_answer": gs["answer"],
                "unsteered_align": align_un[i] if align_un else None,
                "unsteered_coh": coh_un[i] if coh_un else None,
                "steered_align": align_st[i] if align_st else None,
                "steered_coh": coh_st[i] if coh_st else None,
            })

        return {
            "experiment": {
                "name": self.exp.name,
                "arch": self.exp.arch,
                "ckpt": str(self.exp.ckpt_path),
                "feature_ids": list(self.exp.feature_ids),
                "layer": self.exp.layer,
                "frontier_path": str(self.exp.frontier_path),
            },
            "alpha_unsteered": 0.0,
            "alpha_steered": float(alpha_steered),
            "seed": int(seed),
            "n_questions": int(n_questions),
            "n_rollouts": int(n_rollouts),
            "frontier_rows": list(self.exp.frontier_rows),
            "pairs": pairs,
        }

    @staticmethod
    def render_html(data: dict, out_path: Path) -> None:
        import plotly.graph_objects as go

        rows = [r for r in data.get("frontier_rows", [])
                if _is_finite(r.get("mean_alignment")) and _is_finite(r.get("mean_coherence"))]
        alphas = [r["alpha"] for r in rows]
        aligns = [r["mean_alignment"] for r in rows]
        cohs = [r["mean_coherence"] for r in rows]
        a_steered = data["alpha_steered"]

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=cohs, y=aligns, mode="markers",
            marker=dict(size=10, color=alphas, colorscale="RdBu_r",
                        cmin=-100, cmid=0, cmax=100,
                        showscale=True, colorbar=dict(title="α"),
                        line=dict(color="black", width=0.5)),
            text=[f"α={a:+g}" for a in alphas],
            hovertemplate="<b>%{text}</b><br>align=%{y:.2f}<br>coh=%{x:.2f}<extra></extra>",
            name="frontier",
        ))
        a0 = next((r for r in rows if abs(r["alpha"]) < 0.01), None)
        ab = next((r for r in rows if abs(r["alpha"] - a_steered) < 0.01), None)
        if a0:
            fig.add_trace(go.Scatter(
                x=[a0["mean_coherence"]], y=[a0["mean_alignment"]],
                mode="markers+text", marker=dict(size=22, symbol="star", color="black",
                                                  line=dict(color="white", width=2)),
                text=["α=0"], textposition="bottom center",
                name="α=0 (unsteered)", showlegend=False,
            ))
        if ab and (a0 is None or abs(ab["alpha"]) > 0.01):
            fig.add_trace(go.Scatter(
                x=[ab["mean_coherence"]], y=[ab["mean_alignment"]],
                mode="markers+text", marker=dict(size=22, symbol="star", color="gold",
                                                  line=dict(color="black", width=2)),
                text=[f"α={ab['alpha']:+g}"], textposition="top center",
                name=f"α={ab['alpha']:+g} (steered)", showlegend=False,
            ))
        fig.update_layout(
            title=f"{data['experiment']['name']} — frontier (Gemini judge)",
            xaxis_title="mean coherence", yaxis_title="mean alignment",
            template="simple_white", height=520, width=900,
        )
        fig_html = fig.to_html(include_plotlyjs="cdn", full_html=False, div_id="frontier")

        rows_html = []
        for i, p in enumerate(data["pairs"]):
            au = p.get("unsteered_align"); cu = p.get("unsteered_coh")
            as_ = p.get("steered_align"); cs = p.get("steered_coh")
            def _fmt(x):
                return f"{x:.0f}" if _is_finite(x) else "—"
            rows_html.append(f"""
            <tr>
              <td class="qcell"><b>Q{i+1}.</b><br>{_html_escape(p['question'])}</td>
              <td class="acell unsteered">
                <div class="meta">α=0 · align={_fmt(au)} · coh={_fmt(cu)}</div>
                <div class="answer">{_html_escape(p['unsteered_answer'])}</div>
              </td>
              <td class="acell steered">
                <div class="meta">α={a_steered:+g} · align={_fmt(as_)} · coh={_fmt(cs)}</div>
                <div class="answer">{_html_escape(p['steered_answer'])}</div>
              </td>
            </tr>
            """)
        examples_html = f"""
        <h2>Examples (n={len(data['pairs'])}, seed={data.get('seed')})</h2>
        <p class="small">Same RNG seed for both columns. At α=0 unsteered ≡ steered (steering vector × 0 = 0).</p>
        <table class="examples">
          <thead><tr><th>Question</th>
            <th>Unsteered (α=0)</th>
            <th>Steered (α={a_steered:+g})</th></tr></thead>
          <tbody>{''.join(rows_html)}</tbody>
        </table>
        """

        meta_html = f"""
        <h1>{data['experiment']['name']}</h1>
        <ul class="meta-list">
          <li>arch: <code>{data['experiment']['arch']}</code></li>
          <li>layer: <code>{data['experiment']['layer']}</code></li>
          <li>ckpt: <code>{data['experiment']['ckpt']}</code></li>
          <li>k features: <code>{len(data['experiment']['feature_ids'])}</code></li>
          <li>α_unsteered = {data['alpha_unsteered']}, α_steered = {a_steered:+g}</li>
        </ul>
        """

        css = """
        body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
               max-width: 1200px; margin: 1em auto; padding: 0 1em; color: #222; }
        h1 { border-bottom: 2px solid #ccc; padding-bottom: 0.3em; }
        h2 { margin-top: 2em; }
        .meta-list { color: #555; }
        .meta-list code { background: #f0f0f0; padding: 1px 5px; border-radius: 3px; }
        table.examples { width: 100%; border-collapse: collapse; margin-top: 1em; }
        table.examples th, table.examples td { border: 1px solid #ddd; padding: 0.6em; vertical-align: top; }
        table.examples th { background: #f5f5f5; }
        td.qcell { width: 25%; background: #fafafa; }
        td.acell { width: 37.5%; font-size: 0.9em; }
        td.acell .meta { color: #888; font-size: 0.85em; margin-bottom: 0.3em; }
        td.acell .answer { white-space: pre-wrap; }
        td.unsteered { background: #f5fff5; }
        td.steered { background: #fff5f5; }
        .small { color: #777; font-style: italic; }
        """

        full = (f"<!DOCTYPE html><html><head><meta charset='utf-8'>"
                f"<title>{data['experiment']['name']}</title>"
                f"<style>{css}</style></head>"
                f"<body>{meta_html}{fig_html}{examples_html}</body></html>")
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(full)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sp = p.add_subparsers(dest="cmd", required=True)

    pc = sp.add_parser("collect", help="generate examples on a GPU box")
    pc.add_argument("--results_root", type=Path, required=True)
    pc.add_argument("--ckpt", type=Path, default=None,
                    help="explicit ckpt path (overrides --ckpt_root guessing)")
    pc.add_argument("--ckpt_root", type=Path, default=None)
    pc.add_argument("--experiment", required=True,
                    help="frontier JSON path or basename inside --results_root")
    pc.add_argument("--alpha_steered", type=float, default=None,
                    help="if omitted, picks best -α with coh ≥ 90%% baseline")
    pc.add_argument("--n_questions", type=int, default=5)
    pc.add_argument("--n_rollouts", type=int, default=1)
    pc.add_argument("--seed", type=int, default=42)
    pc.add_argument("--out", type=Path, required=True)
    pc.add_argument("--no_judge", action="store_true")

    pr = sp.add_parser("render", help="render HTML from collected JSON")
    pr.add_argument("--data", type=Path, required=True)
    pr.add_argument("--out", type=Path, required=True)

    args = p.parse_args()

    if args.cmd == "collect":
        sel = ExperimentSelector(args.results_root, args.ckpt_root)
        exp = sel.get(args.experiment, ckpt_path=args.ckpt)
        alpha = args.alpha_steered if args.alpha_steered is not None else sel.best_alpha(exp)
        print(f"[dashboard] experiment={exp.name} arch={exp.arch} k={len(exp.feature_ids)} "
              f"alpha_steered={alpha:+g}", flush=True)
        dash = SteeringDashboard(exp)
        data = dash.collect_examples(
            alpha_steered=alpha, n_questions=args.n_questions,
            n_rollouts=args.n_rollouts, seed=args.seed, judge=not args.no_judge,
        )
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(data, indent=2))
        print(f"[dashboard] wrote {args.out}", flush=True)
    elif args.cmd == "render":
        data = json.loads(Path(args.data).read_text())
        SteeringDashboard.render_html(data, args.out)
        print(f"[dashboard] wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
