"""Judge a wave's steering rows: genuine-backtracking count + coherence grade.

    uvx modal run --detach experiments/backtracking_steering_dsm/modal_judge.py \
        --wave wave1

Both judges are the reference scripts' own single-row coroutines, imported
unmodified: grade_backtracking.judge_one (genuine-event COUNT) and
grade_sonnet.grade_one (0-3 coherence). We call them directly rather than via
their `grade_rows` wrappers because those wrappers skip rows whose keyword_rate
is within 0.005 of the 0.007 baseline and write genuine_count = 0 for them --
the published c7 headline judged every row (n_judge_calls = 1525 = 61 prompts x
25 magnitudes), so a kw prefilter here would put our numbers on a different
footing from the reference before any arm is compared.

Writes backtracking_eval/steering/<wave>/judged__<tag>.json:
  {row_idx: {"genuine_count": int, "raw": str, "grade": int, "grade_raw": str}}
"""

import json
import pathlib

import modal

try:
    ROOT = pathlib.Path(__file__).resolve().parents[2]
except IndexError:
    ROOT = pathlib.Path("/work")

app = modal.App("backtracking-steering-judge")
vol = modal.Volume.from_name("diffusion-txc", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("anthropic", "pyyaml")
    .add_local_file(
        str(ROOT / "experiments" / "ward_backtracking_txc" / "grade_backtracking.py"),
        "/work/experiments/ward_backtracking_txc/grade_backtracking.py")
    .add_local_file(
        str(ROOT / "experiments" / "ward_backtracking_txc" / "grade_sonnet.py"),
        "/work/experiments/ward_backtracking_txc/grade_sonnet.py")
    .add_local_file(str(ROOT / "results" / "ward_backtracking" / "prompts.json"),
                    "/work/data/prompts.json")
)


@app.function(image=image, timeout=10800, memory=8192, volumes={"/vol": vol},
              secrets=[modal.Secret.from_name("em-sprint-judges")])
def judge_one_file(args: tuple[str, str], concurrency: int = 8) -> dict:
    import asyncio
    import os
    import sys
    import time

    sys.path.insert(0, "/work")
    from anthropic import AsyncAnthropic
    from experiments.ward_backtracking_txc.grade_backtracking import judge_one
    from experiments.ward_backtracking_txc.grade_sonnet import grade_one

    wave, tag = args
    t0 = time.time()
    d = pathlib.Path("/vol/backtracking_eval/steering") / wave
    rows = json.loads((d / f"rows__{tag}.json").read_text())["rows"]
    out_path = d / f"judged__{tag}.json"
    judged: dict = {}
    if out_path.exists():
        judged = json.loads(out_path.read_text())
        print(f"[resume] {len(judged)} existing", flush=True)

    stage_a = json.loads(pathlib.Path("/work/data/prompts.json").read_text())
    prompts = {p["id"]: p.get("question") or p.get("prompt") or p.get("text", "")
               for p in stage_a}

    # A row that errors is written with genuine_count = -1 and then dropped from
    # every mean downstream, so a burst of 429s would quietly bias an arm rather
    # than fail loudly. Leaning on the SDK's own backoff plus one outer retry
    # keeps transient failures from turning into missing data.
    client = AsyncAnthropic(api_key=os.environ["ANTHROPIC_API_KEY"], max_retries=8)
    sem = asyncio.Semaphore(concurrency)
    done = [0]

    async def _both(ptext: str, text: str):
        last = None
        for attempt in range(3):
            try:
                return await asyncio.gather(judge_one(client, ptext, text),
                                            grade_one(client, ptext, text))
            except Exception as e:                                # noqa: BLE001
                last = e
                await asyncio.sleep(2.0 * (attempt + 1))
        raise last

    async def worker(idx: int, row: dict):
        key = str(idx)
        prev = judged.get(key, {})
        if prev.get("genuine_count", -1) >= 0 and prev.get("grade", -1) >= 0:
            return
        text = (row.get("text") or "").strip()
        ptext = prompts.get(row.get("prompt_id", ""), "(unknown prompt)")
        if not text:
            judged[key] = {"genuine_count": 0, "raw": "(empty)",
                           "grade": 0, "grade_raw": "(empty)"}
            return
        async with sem:
            try:
                bt, coh = await _both(ptext, text)
                judged[key] = {"genuine_count": bt["genuine_count"], "raw": bt["raw"],
                               "grade": coh["grade"], "grade_raw": coh["raw"]}
            except Exception as e:
                judged[key] = {"genuine_count": -1, "raw": f"(error: {e})",
                               "grade": -1, "grade_raw": f"(error: {e})"}
        done[0] += 1
        if done[0] % 100 == 0:
            print(f"[judge] {tag}: {done[0]} done ({time.time() - t0:.0f}s)",
                  flush=True)

    async def run():
        await asyncio.gather(*[worker(i, r) for i, r in enumerate(rows)])

    asyncio.run(run())
    out_path.write_text(json.dumps(judged))
    vol.commit()
    n_bad = sum(1 for v in judged.values() if v.get("genuine_count", -1) < 0)
    print(f"[done] {tag}: {len(judged)} judged, {n_bad} errored, "
          f"{time.time() - t0:.0f}s", flush=True)
    return {"tag": tag, "n": len(judged), "n_errored": n_bad,
            "runtime_s": round(time.time() - t0, 1)}


@app.function(image=image, timeout=600, volumes={"/vol": vol})
def list_tags(wave: str) -> list[str]:
    d = pathlib.Path("/vol/backtracking_eval/steering") / wave
    return sorted(p.name[len("rows__"):-len(".json")]
                  for p in d.glob("rows__*.json"))


@app.local_entrypoint()
def main(wave: str = "wave1"):
    tags = list_tags.remote(wave)
    print(f"[judge] {wave}: {len(tags)} files -> {tags}")
    res = list(judge_one_file.map([(wave, t) for t in tags]))
    print(json.dumps(res, indent=2))
