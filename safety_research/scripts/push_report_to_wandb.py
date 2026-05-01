"""Push the generated REPORT.md and all figures to a wandb run."""
import os
import sys
from pathlib import Path

import wandb

SAFETY_DIR = Path("/home/cs29824/andre/temp_xc/safety_research")
FIG = SAFETY_DIR / "figures"


def main() -> None:
    run = wandb.init(project="temporal-crosscoders-safety",
                     name="report",
                     tags=["safety", "report"],
                     reinit=True)
    print(f"wandb run: {run.url}")

    artifact = wandb.Artifact("safety_research_report", type="report")
    artifact.add_file(str(SAFETY_DIR / "REPORT.md"))
    for png in sorted(FIG.glob("*.png")):
        artifact.add_file(str(png), name=f"figures/{png.name}")
    for js in (SAFETY_DIR / "results").rglob("*.json"):
        artifact.add_file(str(js), name=f"results/{js.relative_to(SAFETY_DIR / 'results')}")
    run.log_artifact(artifact)

    for png in sorted(FIG.glob("*.png")):
        run.log({f"figures/{png.stem}": wandb.Image(str(png))})
    run.summary["report_path"] = str(SAFETY_DIR / "REPORT.md")
    run.finish()


if __name__ == "__main__":
    main()
