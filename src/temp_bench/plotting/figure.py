"""Figure save helper.

Produces both a high-resolution PNG (for the paper) and a thumbnail
(``.thumb.png``, ≤288px wide @ 48 DPI) for AI-agent inspection.
Loading a multi-MB PNG into an LLM context is expensive — agents must
read the thumbnail.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.figure as _mpl_fig

THUMB_DPI = 48
THUMB_MAX_WIDTH_PX = 288
HIRES_DPI = 150


def save_figure(fig: _mpl_fig.Figure, path: str | Path, **savefig_kwargs: Any) -> Path:
    """Save ``fig`` to ``path`` and write a sibling ``.thumb.png``.

    Returns the high-res path.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() != ".png":
        raise ValueError(f"save_figure requires a .png path, got {path.suffix}")

    # high-res
    fig.savefig(path, dpi=HIRES_DPI, bbox_inches="tight", **savefig_kwargs)

    # thumb — re-render at low DPI then resize to width cap
    thumb_path = path.with_suffix(".thumb.png")
    fig.savefig(thumb_path, dpi=THUMB_DPI, bbox_inches="tight")

    # Optional resize to enforce width cap. We try Pillow but degrade gracefully.
    try:
        from PIL import Image
        with Image.open(thumb_path) as im:
            w, h = im.size
            if w > THUMB_MAX_WIDTH_PX:
                new_h = int(h * THUMB_MAX_WIDTH_PX / w)
                im.resize((THUMB_MAX_WIDTH_PX, new_h)).save(thumb_path)
    except ImportError:
        pass

    return path
