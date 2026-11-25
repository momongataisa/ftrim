"""Fig. 2 of the paper from results/mechanism.json: what the models learn to skip.

    python scripts/make_figures.py        # -> assets/skip.png   (needs the `figures` extra: matplotlib)

For every fine-tuned method, the change against the untrained model of the log-odds of a *skip*: at the first line
that would be deleted, the model writes the token after the line instead of the line's first token. One bar for the
first format-talk line of an output, one for random other lines of the same length. Mean of the four settings, seed 17.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from ftrim import config  # noqa: E402
from ftrim.utils import read_json  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
LABELS = {"ftrim": "FTrim", "random_deletion": "Random deletion", "rationalization": "+ Rationalization", "stm": "STM"}
# blue: what the format instructions cause; grey: everything else (the colours of Fig. 1)
SERIES = (("skip_format_talk", "first format-talk line", "#2a78d6"),
          ("skip_random", "random other lines of the same length", "#94938c"))
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"


def skip_figure(seed: int, dest: Path) -> None:
    probes = read_json(config.RESULTS / "mechanism.json")["probes"][f"s{seed}"]
    if "pooled" not in probes:
        raise SystemExit("results/mechanism.json holds the probes of some settings only")
    methods = sorted(LABELS, key=lambda name: -probes["pooled"]["methods"][name]["skip_format_talk"])
    plt.rcParams.update({"font.size": 9, "axes.linewidth": 0.6, "axes.edgecolor": MUTED, "xtick.color": MUTED,
                         "ytick.color": INK, "axes.labelcolor": INK})
    fig, ax = plt.subplots(figsize=(7.2, 2.7))
    height, gap = 0.34, 0.05
    for k, name in enumerate(methods):
        y = len(methods) - 1 - k
        for j, (key, _, colour) in enumerate(SERIES):
            value = probes["pooled"]["methods"][name][key]
            at = y + (0.5 - j) * (height + gap)
            ax.barh(at, value, height=height, color=colour, linewidth=0, zorder=3)
            ax.text(value + (0.25 if value >= 0 else -0.25), at, f"{value:+.1f}".replace("-", "−"),
                    va="center", ha="left" if value >= 0 else "right", fontsize=8, color=INK, zorder=5)
    ax.axvline(0, color=MUTED, linewidth=0.8, zorder=4)
    ax.grid(axis="x", color=GRID, linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)
    ax.set_yticks([len(methods) - 1 - k for k in range(len(methods))], [LABELS[name] for name in methods])
    ax.tick_params(axis="y", length=0, pad=6)
    ax.set_xlim(-13.5, 13.5)
    ax.set_xticks(range(-10, 11, 5), [f"{x:+d}".replace("-", "−") if x else "0" for x in range(-10, 11, 5)])
    ax.set_xlabel("change in the log-odds of a skip against the untrained model (nats)\n"
                  "← skips the line less often          skips it more often →", color=MUTED, fontsize=8.5)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    handles = [plt.Rectangle((0, 0), 1, 1, color=colour) for _, _, colour in SERIES]
    ax.legend(handles, [label for _, label, _ in SERIES], loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2,
              frameon=False, fontsize=8.5, handlelength=1.2, columnspacing=2.0)
    fig.tight_layout()
    dest.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(dest, dpi=200, facecolor="white")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seed", type=int, default=17, help="the seed of the fine-tuned models (17 or 42)")
    ap.add_argument("--out", default=str(ROOT / "assets" / "skip.png"))
    a = ap.parse_args()
    skip_figure(a.seed, Path(a.out))
    print(a.out)


if __name__ == "__main__":
    main()
