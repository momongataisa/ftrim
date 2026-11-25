"""Fill the generated tables of README.md and docs/REPRODUCTION.md.

    python scripts/update_docs.py            # rewrite the blocks <!-- results:NAME --> ... <!-- /results:NAME -->
    python scripts/update_docs.py --check    # fail if a file is not up to date

README.md shows the main tables of the paper's runs (results/tables.json, written by `python -m ftrim report`; the
same tables and the others are in results/RESULTS.md). docs/REPRODUCTION.md compares the paper's runs with a run of
this repository from scratch (results/reproduction/, exported with FTRIM_RESULTS=results/reproduction).
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np

from ftrim import config, report
from ftrim.report import BENCH, HEADS, NAMES, TABLE, num, table
from ftrim.results import Results

ROOT = Path(__file__).resolve().parents[1]
SETTINGS = config.SETTINGS
THRESHOLD_SEED = report.THRESHOLD_SEED          # the seed of the one-run rows and of the reproduction


# ------------------------------------------------------------------------------------------------- the reproduction
class Reproduction:
    """The paper's runs (results/) and the from-scratch run of this repository (results/reproduction/)."""

    def __init__(self) -> None:
        self.dir = config.RESULTS / "reproduction"
        self.paper, self.new = Results(), Results(self.dir / "per_question")
        self.unseen = config.experiment()["unseen"]

    def settings(self) -> list[tuple[str, str]]:
        return [s for s in SETTINGS if self.runs(*s)]

    def runs(self, m: str, ds: str) -> list[tuple[str, int]]:
        """The fine-tuned (method, seed) runs of a setting that exist in the reproduction, in the order of Table 1."""
        methods = dict.fromkeys([*TABLE, "normalized", *report.THRESHOLDS])       # "ftrim" is in both lists
        return [(method, seed) for method in methods
                if method not in config.UNTRAINED for seed in self.new.seeds(m, ds, method)]

    def pair(self, m: str, ds: str, method: str, bench: str, seed: int | None, field: str = "correct"):
        """(paper, reproduction) values per question of one run."""
        seeds = None if seed is None else [seed]
        return [r.matrix(m, ds, method, bench, field, seeds=seeds)[0] for r in (self.paper, self.new)]

    def unseen_mean(self, m: str, ds: str, method: str, seed: int | None) -> list[float]:
        both = [self.pair(m, ds, method, b, seed) for b in self.unseen]
        return [float(np.mean([pair[k].mean() for pair in both])) for k in (0, 1)]

    def paper_seeds(self, m: str, ds: str, method: str, benches: list[str]) -> str:
        """Lowest and highest value over the paper's seeds of the mean accuracy on `benches`."""
        values = np.mean([self.paper.matrix(m, ds, method, b).mean(axis=1) for b in benches], axis=0)
        return f"{values.min():.2f}–{values.max():.2f}"


def _delta(a: float, b: float, nd: int = 2) -> str:
    return num(b - a, nd, True)


def repro_untrained(r: Reproduction) -> list[str]:
    rows = []
    for m in config.experiment()["models"]:
        for bench in [*config.experiment()["datasets"], *r.unseen]:
            if r.new.matrix(m, bench, "untrained", bench) is None:
                continue
            a, b = r.pair(m, bench, "untrained", bench, None)
            rows.append([config.model(m)["name"], BENCH[bench], f"{len(a):,}", num(a.mean()), num(b.mean()),
                         _delta(a.mean(), b.mean()), num(100 * float((a == b).mean()), 1)])
    return table(["Untrained model", "Benchmark", "Questions", "Paper", "Reproduction", "Difference",
                  "Same score (%)"], rows, "llrrrrr")


def repro_finetuned(r: Reproduction) -> list[str]:
    """Every fine-tuned run of the reproduction next to the paper's run with the same seed: target accuracy,
    generalization accuracy and output length."""
    rows = []
    for m, ds in r.settings():
        for method, seed in r.runs(m, ds):
            a, b = r.pair(m, ds, method, ds, seed)
            ua, ub = r.unseen_mean(m, ds, method, seed)
            ta, tb = r.pair(m, ds, method, ds, seed, "tokens")
            rows.append([HEADS[m, ds], NAMES[method], num(a.mean()), num(b.mean()), r.paper_seeds(m, ds, method, [ds]),
                         num(ua), num(ub), r.paper_seeds(m, ds, method, r.unseen), num(ta.mean(), 0),
                         num(tb.mean(), 0)])
    head = ["Setting", f"Method (seed {THRESHOLD_SEED})", "Target accuracy: paper", "Reproduction",
            "Paper's three seeds", "Generalization accuracy: paper", "Reproduction", "Paper's three seeds",
            "Tokens: paper", "Reproduction"]
    return table(head, rows, "ll" + "r" * 8)


PAPER_BLOCKS = {"target": report.target_table, "generalization": report.generalization_table,
                "gains": report.gains_table, "threshold": report.threshold_table}
REPRODUCTION_BLOCKS = {"repro-untrained": repro_untrained, "repro-finetuned": repro_finetuned}
FILES = [ROOT / "README.md", ROOT / "docs" / "REPRODUCTION.md"]
BLOCK = r"(<!-- results:{name} -->\n).*?(<!-- /results:{name} -->)"


def render(text: str) -> str:
    """The text with every generated block rebuilt."""
    names = re.findall(r"<!-- results:([a-z0-9-]+) -->", text)
    paper = report.load_tables() if any(n in PAPER_BLOCKS for n in names) else None
    reproduction = Reproduction() if any(n in REPRODUCTION_BLOCKS for n in names) else None
    for name in names:
        if name in PAPER_BLOCKS:
            lines = PAPER_BLOCKS[name](paper)
        elif name in REPRODUCTION_BLOCKS:
            lines = REPRODUCTION_BLOCKS[name](reproduction)
        else:
            raise SystemExit(f"unknown block results:{name}")
        body = "\n" + "\n".join(lines).rstrip("\n") + "\n\n"          # blank lines: the markers are HTML blocks
        text = re.sub(BLOCK.format(name=re.escape(name)), lambda m, body=body: m.group(1) + body + m.group(2), text,
                      flags=re.S)
    return text


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true", help="do not write; fail if a file is not up to date")
    a = ap.parse_args()
    stale = []
    for path in FILES:
        current = path.read_text(encoding="utf-8")
        updated = render(current)
        if current != updated:
            stale.append(str(path.relative_to(ROOT)))
            if not a.check:
                path.write_text(updated, encoding="utf-8")
    if a.check and stale:
        print("not up to date (run `python scripts/update_docs.py`): " + ", ".join(stale))
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
