"""Read-only CPU audit of the frozen search; no fitting or selection changes."""
from collections import defaultdict
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
import statistics as st
import sys

import torch

root = Path(sys.argv[1]); full = root/"full"
torch.set_num_threads(1)
digest = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
selection = json.loads((full/"selection.json").read_text())
frozen = datetime.fromisoformat(selection["frozen_at"])
completion = json.loads((full/"completion.json").read_text())
assert completion["state"]=="complete"
assert json.loads((root/"launcher_exit.json").read_text())["returncode"]==0
manifest = json.loads((root/"source_manifest.json").read_text())
for item in manifest["files"]: assert digest(root/"source"/item["path"])==item["sha256"]
audit = json.loads((full/"split_audit.json").read_text())
heldout = {}
for data in audit["datasets"]:
    ids = [set(x) for x in data["split_ids"].values()]
    assert all(len(x)==4096 for x in ids)
    assert all(not a.intersection(b) for i,a in enumerate(ids) for b in ids[i+1:])
    seed = data["seed"]
    assert digest(full/"data"/f"tuning_seed{seed}.pt")==data["tuning_sha256"]
    assert digest(full/"data"/f"heldout_seed{seed}.pt")==data["heldout_sha256"]
    heldout[seed] = torch.load(full/"data"/f"heldout_seed{seed}.pt", map_location="cpu", weights_only=False)
results = [json.loads(p.read_text()) for p in sorted((full/"jobs").glob("*/result.json"))]
assert len(results)==46
screen = [r for r in results if r["spec"]["phase"]=="screen"]
confirm = [r for r in results if r["spec"]["phase"]=="confirm"]
assert len(screen)==16 and len(confirm)==30
assert sum(len(r["rows"]) for r in screen)==48
assert sum(len(r["rows"]) for r in confirm)==30
normal_probes = controls = models = 0
for result in results:
    directory = full/"jobs"/result["spec"]["id"]
    for row in result["rows"]:
        spec=row["spec"]; steps=row["steps"]
        assert (spec["h"],spec["q"],spec["k"],spec["width"])==(3,11,5,2048)
        assert row["n_train"]==row["n_val"]==4096 and row["l0_per_window"]<=5
        model=torch.load(directory/f"model_step{steps}.pt",map_location="cpu",weights_only=False)
        norms=model["state_dict"]["W_dec"].norm(dim=(1,2))
        assert torch.allclose(norms,torch.ones_like(norms),atol=2e-6,rtol=0)
        models+=1
        probe=torch.load(directory/f"probe_{row['split']}_step{steps}.pt",map_location="cpu",weights_only=False)
        score=float((probe["eval_labels"]==probe["eval_predictions"]).float().mean())
        assert score==row["accuracy"]
        normal_probes+=1
        if spec["phase"]=="confirm":
            assert datetime.fromisoformat(row["evaluated"])>frozen
            assert torch.equal(probe["eval_labels"],heldout[spec["seed"]]["anchors"][spec["W"]]["Y"])
            for name,key in [("raw_linear_probe.pt","raw_linear_control"),("shuffled_label_probe.pt","shuffled_label_control")]:
                if key not in row: continue
                p=torch.load(directory/name,map_location="cpu",weights_only=False)
                assert float((p["eval_labels"]==p["eval_predictions"]).float().mean())==row[key]["val_accuracy"]
                assert torch.equal(p["eval_labels"],probe["eval_labels"])
                controls+=1
        else:
            assert row["split"]=="validation"

rows=[r for result in confirm for r in result["rows"]]
groups=defaultdict(list)
for r in rows: groups[(r["spec"]["recipe"]["name"],r["spec"]["W"])].append(r)
assert len(groups)==10 and all(sorted(r["spec"]["seed"] for r in rs)==[0,1,2] for rs in groups.values())
chance=1/11
chance_sd=lambda n: math.sqrt(chance*(1-chance)/n)
negative={recipe:dict(mean=st.fmean(r["accuracy"] for r in rs),pooled_chance_z=(st.fmean(r["accuracy"] for r in rs)-chance)/chance_sd(3*4096)) for (recipe,W),rs in groups.items() if W==3}
raw=[]; shuffled=[]
for r in rows:
    # Raw linear controls are identical across recipes, so count each once.
    if r["spec"]["recipe"]["name"]=="baseline": raw.append(r["raw_linear_control"]["val_accuracy"])
    if "shuffled_label_control" in r: shuffled.append(r["shuffled_label_control"]["val_accuracy"])
assert all(abs(v["pooled_chance_z"])<6 for v in negative.values())
assert max(abs((x-chance)/chance_sd(4096)) for x in raw+shuffled)<6
assert all(r["oracle"]["symbolic_accuracy"]==r["oracle"]["observation_accuracy"]==1 for r in rows if r["spec"]["W"]>=4)
selected=selection["selected"]["recipe"]["name"]
acc={(r["spec"]["recipe"]["name"],r["spec"]["seed"],r["spec"]["W"]):r["accuracy"] for r in rows}
endpoint=[acc[(selected,s,10)]-acc[(selected,s,4)] for s in range(3)]
gain=[acc[(selected,s,10)]-acc[("baseline",s,10)] for s in range(3)]
selected_means=[st.fmean(acc[(selected,s,W)] for s in range(3)) for W in [3,4,5,6,10]]
report=dict(passed=True, source_hashes_verified=len(manifest["files"]), dataset_hashes_verified=6,
            episode_splits_disjoint=True, all_test_evaluations_after_selection=True,
            screening_jobs=16, screening_validation_rows=48, confirmation_jobs=30,
            model_checkpoints_verified=models, normal_probes_verified=normal_probes, control_probes_verified=controls,
            decoder_unit_norms_verified=True, common_seeds=[0,1,2], negative_control=negative,
            raw_linear_control_range=[min(raw),max(raw)], shuffled_label_control_range=[min(shuffled),max(shuffled)],
            symbolic_and_observation_oracles_exact=True, selected=selection["selected"],
            selected_W10_minus_W4_by_seed=endpoint, selected_minus_baseline_W10_by_seed=gain,
            selected_minus_baseline_W10_mean=st.fmean(gain), selected_minus_baseline_W10_sd=st.stdev(gain),
            monotonic_mean_curve=all(a<=b for a,b in zip(selected_means,selected_means[1:])),
            interpretation="Modest improvement, with a small W5-to-W6 decline; no strict monotonicity or high recovery accuracy established.",
            elapsed_seconds=completion["elapsed_seconds"])
(full/"audit_report.json").write_text(json.dumps(report,indent=2)+"\n")
print(json.dumps(report,indent=2))
