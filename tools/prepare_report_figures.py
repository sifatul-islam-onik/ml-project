"""
Select the figures that appear in the report, strip their baked-in titles, and
copy them into outputs/report_figures/ numbered in the order they appear.

The analysis pipeline writes 30 diagnostic figures into outputs/figures/, each
with a "Figure N" title drawn inside the image, where N is the pipeline's own
number. The report uses a curated subset in a different order, so those printed
numbers would contradict the report's captions. This script crops the title band
off each selected figure and renames it to its position in the report, leaving
the caption as the single source of the number.

    python tools/prepare_report_figures.py

Source figures are never modified.
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "outputs" / "figures"
DST = ROOT / "outputs" / "report_figures"

# The figures the report actually argues with, in the order they appear.
# Each entry: (source filename, slug used in the copied filename).
SELECTED = [
    ("fig03_likert_distribution.png", "likert_distribution"),
    ("fig05_interitem_correlations.png", "interitem_correlations"),
    ("fig09_efa_loadings.png", "efa_loadings"),
    ("fig11_validity_indices.png", "validity_indices"),
    ("fig13_gap_statistic.png", "gap_statistic"),
    ("fig14_bootstrap_stability.png", "bootstrap_stability"),
    ("fig16_consensus.png", "consensus_matrix"),
    ("fig28_decision_tree.png", "decision_tree"),
    ("fig18_persona_profiles.png", "persona_profiles"),
    ("fig30_persona_cards.png", "persona_cards"),
    ("fig17_holdout_centroids.png", "holdout_centroids"),
    ("fig27_unasked_stressors.png", "unasked_stressors"),
]

INK_THRESHOLD = 250   # grey level below which a pixel counts as ink
LINE_GAP = 9          # blank runs this short are gaps *within* a title block
MAX_TITLE_FRAC = 0.30 # a title must sit within the top 30% of the image
PAD = 14              # whitespace to leave above the plot after cropping


def title_crop_row(path: Path) -> int:
    """Row at which content below the baked-in title begins (0 if none found)."""
    grey = np.array(Image.open(path).convert("L"))
    ink = (grey < INK_THRESHOLD).sum(axis=1)
    height = len(ink)
    limit = int(height * MAX_TITLE_FRAC)

    row = 0
    while row < limit and ink[row] == 0:  # skip the top margin
        row += 1
    if row >= limit:
        return 0

    # Walk through the title's text lines, allowing short gaps between them,
    # and stop at the first wide gap — that is the break before the plot.
    while row < limit:
        while row < limit and ink[row] > 0:
            row += 1
        gap_start = row
        while row < limit and ink[row] == 0:
            row += 1
        gap = row - gap_start
        if gap == 0:
            break
        if gap > LINE_GAP:
            return max(0, min(gap_start + PAD, row))
    return 0


def main() -> int:
    if not SRC.is_dir():
        print(f"missing {SRC}", file=sys.stderr)
        return 1

    if DST.exists():
        shutil.rmtree(DST)
    DST.mkdir(parents=True)

    missing = [name for name, _ in SELECTED if not (SRC / name).exists()]
    if missing:
        print(f"missing source figures: {missing}", file=sys.stderr)
        return 1

    for index, (name, slug) in enumerate(SELECTED, start=1):
        src = SRC / name
        out = DST / f"figure_{index:02d}_{slug}.png"

        cut = title_crop_row(src)
        image = Image.open(src)
        if cut > 0:
            image = image.crop((0, cut, image.width, image.height))
        image.save(out)

        note = f"cropped {cut}px" if cut else "no title found — copied as-is"
        print(f"  figure_{index:02d}  {name:<34} {note}")

    print(f"\nwrote {len(SELECTED)} figures to {DST.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
