"""Plotting utilities.

Provides save_figure(), which saves both a high-res version (for docs /
readers) and a low-res thumbnail (for AI agent inspection without blowing
up context windows).

Convention:
    save_figure(fig, "results/foo/bar.png")
    → results/foo/bar.png        (high-res, 150 DPI)
    → results/foo/bar.thumb.png  (thumbnail, ≤480px wide, 48 DPI)
"""

import os

import matplotlib.pyplot as plt
from matplotlib.figure import Figure

THUMB_MAX_WIDTH_IN = 6.0  # inches – at 48 DPI → 288px wide
THUMB_DPI = 48


def save_figure(
    fig: Figure,
    path: str,
    *,
    dpi: int = 150,
    bbox_inches: str = "tight",
    **kwargs,
) -> None:
    """Save *fig* at full resolution **and** as a small thumbnail.

    Parameters
    ----------
    fig : matplotlib Figure
    path : str
        Destination for the high-res image (e.g. ``results/foo/plot.png``).
        The thumbnail is saved alongside it with a ``.thumb.png`` suffix
        (e.g. ``results/foo/plot.thumb.png``).
    dpi : int
        DPI for the high-res version (default 150).
    bbox_inches, **kwargs
        Forwarded to ``fig.savefig``.
    """
    # ── high-res ──────────────────────────────────────────────────────
    fig.savefig(path, dpi=dpi, bbox_inches=bbox_inches, **kwargs)

    # ── thumbnail ─────────────────────────────────────────────────────
    base, ext = os.path.splitext(path)
    thumb_path = f"{base}.thumb{ext}"

    # Temporarily shrink the figure for the thumbnail
    orig_size = fig.get_size_inches()
    w, h = orig_size
    if w > THUMB_MAX_WIDTH_IN:
        scale = THUMB_MAX_WIDTH_IN / w
        fig.set_size_inches(w * scale, h * scale)

    fig.savefig(thumb_path, dpi=THUMB_DPI, bbox_inches=bbox_inches, **kwargs)

    # Restore original size so caller isn't surprised
    fig.set_size_inches(orig_size)
