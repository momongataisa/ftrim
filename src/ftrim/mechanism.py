"""What the models learn (Sect. 5 and Fig. 2 of the paper). Forward passes only, no training.

1. `format_talk` (GPU): FTrim's rule applied to the untrained model's own test outputs. The untrained model scores
   every reasoning token of an output under the prompt with the format instructions and under the counterfactual
   prompt; the lines with m(l) >= 1 bit per token are the output's *format talk*.
2. `probe` (GPU): the untrained model and each fine-tuned model score every such output three times: as written
   (T0), without its format-talk lines (T1), and without random other lines of the same token count (T2, chosen as
   random deletion chooses them).
3. `summary` (CPU): for each fine-tuned model, against the untrained model,
   * the change of the log-likelihood of a format-talk line and of another line (nats per line), and
   * the change of the log-odds of a *skip* at the first deleted line: the model writes the token that follows the
     line instead of the line's first token. Fig. 2 shows it for the first format-talk line and for random lines.
     The log-odds are taken at the first token where the shortened output differs from the output as written. This
     is the line's first token unless the line and the text after it begin with the same tokens (for example a list
     marker); then it is the first token where they differ.
"""
from __future__ import annotations

import time

import numpy as np

from . import chat, config, core, data
from .prompts import build_prompt
from .utils import provenance, read_json, read_jsonl, rng_for, save_npz, write_json, write_jsonl

RESULT = config.RESULTS / "mechanism.json"


def _talk_file(m: str, ds: str):
    return config.mechanism_dir(m, ds) / "format_talk.jsonl"


# ----------------------------------------------------------------------------------- 1. format talk in the test outputs
def format_talk(m: str, ds: str) -> dict:
    from .engine import Engine
    dest = _talk_file(m, ds)
    if dest.with_suffix(".summary.json").exists():          # written last: the stage is complete
        return read_json(dest.with_suffix(".summary.json"))
    rule = config.experiment()["edits"]["ftrim"]
    examples = data.load_index(ds, "test")
    items, skipped = [], {"no_tag": 0, "unsplit": 0}
    for r in read_jsonl(config.eval_dir(m, "untrained") / f"{ds}-test.jsonl"):
        if "<answer>" not in r["text"]:
            skipped["no_tag"] += 1
            continue
        try:        # the token view of an output, as for a teacher
            tt = chat.teacher_tokens(m, build_prompt(examples[r["id"]]), r["text"])
        except ValueError:
            skipped["unsplit"] += 1
            continue
        items.append((r, tt))
    seqs, starts = [], []
    for r, tt in items:
        bare = chat.pair(m, build_prompt(examples[r["id"]], bare=True), "x")      # the prompt part only
        seqs += [list(tt.pair.input_ids), list(bare.input_ids[:bare.prompt_len]) + list(tt.pair.response)]
        starts += [tt.pair.prompt_len, bare.prompt_len]
    engine = Engine(m, kind="score")
    lps = []
    for c in range(0, len(seqs), 2000):
        lps += engine.token_logprobs(seqs[c:c + 2000], starts[c:c + 2000])
        print(f"  scored {min(c + 2000, len(seqs))}/{len(seqs)}", flush=True)
    rows = []
    for k, (r, tt) in enumerate(items):
        full, bare = (lps[2 * k + i][:tt.n_reason].astype(np.float32).astype(np.float64) for i in (0, 1))
        rows.append({"id": r["id"], "document_id": r["document_id"], "prompt_len": tt.pair.prompt_len,
                     "response": list(tt.pair.response), "n_reason": tt.n_reason, "lines": [list(x) for x in tt.lines],
                     "line_scores": core.line_scores(full, bare, tt.lines),
                     "format_talk": sorted(core.format_talk_lines(full, bare, tt.lines,
                                                                  threshold_bits=rule["threshold_bits"]))})
    write_jsonl(dest, rows)
    summary = {"model": m, "dataset": ds, **talk_share(rows), **skipped, **provenance()}
    write_json(dest.with_suffix(".summary.json"), summary)
    return summary


def talk_share(rows: list[dict]) -> dict:
    tokens = sum(r["n_reason"] for r in rows)
    talk = sum(r["lines"][i][1] - r["lines"][i][0] for r in rows for i in r["format_talk"])
    return {"outputs": len(rows), "with_format_talk": sum(bool(r["format_talk"]) for r in rows),
            "token_share": talk / max(tokens, 1)}


# ----------------------------------------------------------------------------------------------------------- 2. probes
def other_random_lines(r: dict, m: str, ds: str) -> set[int]:
    """Lines outside the format talk with about its token count, in a seeded random order (as random deletion)."""
    talk = set(r["format_talk"])
    sizes = [b - a for a, b in r["lines"]]
    free = [i for i in range(len(sizes)) if i not in talk]
    rng_for("mech-RDK", m, ds, r["id"]).shuffle(free)
    return core.random_lines(free, sizes, sum(sizes[i] for i in talk))[0]


def without(r: dict, drop: set[int]) -> list[int]:
    """The response tokens without the given reasoning lines (the answer line is kept)."""
    return core.delete(r["response"][:r["n_reason"]], r["lines"], drop) + r["response"][r["n_reason"]:]


def probe_items(m: str, ds: str) -> list[dict]:
    """The outputs with format talk and at least one other line; DROP is subsampled with a seeded order."""
    rows = [r for r in read_jsonl(_talk_file(m, ds)) if r["format_talk"] and len(r["format_talk"]) < len(r["lines"])]
    rows.sort(key=lambda r: rng_for("mech-lines", m, ds, r["id"]).random())
    return sorted(rows[:config.experiment()["mechanism"]["max_outputs"][ds]], key=lambda r: r["id"])


def probe(m: str, ds: str, seed: int) -> dict:
    from .engine import Engine
    out_dir = config.mechanism_dir(m, ds) / f"probe_s{seed}"
    if (out_dir / "done.json").exists():
        return read_json(out_dir / "done.json")
    methods = config.experiment()["mechanism"]["methods"]
    rows = probe_items(m, ds)
    examples = data.load_index(ds, "test")
    variants, prompts = [], {}
    for r in rows:
        random = other_random_lines(r, m, ds)
        variants.append((r, [r["response"], without(r, set(r["format_talk"])), without(r, random)], sorted(random)))
        p = chat.pair(m, build_prompt(examples[r["id"]]), "x")
        if p.prompt_len != r["prompt_len"]:
            raise ValueError(f"{r['id']}: the prompt length differs from the scored output's")
        prompts[r["id"]] = list(p.input_ids[:p.prompt_len])
    out_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(out_dir / "items.jsonl", [
        {"id": r["id"], "document_id": r["document_id"], "n_reason": r["n_reason"], "lines": r["lines"],
         "format_talk": r["format_talk"], "random": rnd, "lens": [len(t) for t in ts]} for r, ts, rnd in variants])
    seqs = [prompts[r["id"]] + t for r, ts, _ in variants for t in ts]
    starts = [len(prompts[r["id"]]) for r, ts, _ in variants for _ in ts]
    engine = Engine(m, lora=True, kind="score")
    for name in ["untrained", *methods]:
        dest = out_dir / f"{name}.npz"
        if dest.exists():
            continue
        adapter = None
        if name != "untrained":
            adapter_dir = config.fit_dir(m, ds, name, seed) / "adapter"
            if not (adapter_dir / "adapter_config.json").exists():
                raise SystemExit(f"{adapter_dir} is missing")
            adapter = str(adapter_dir)
        started, lps = time.time(), []
        for c in range(0, len(seqs), 3000):
            lps += engine.token_logprobs(seqs[c:c + 3000], starts[c:c + 3000], lora=adapter)
        save_npz(dest, lens=np.array([len(x) for x in lps]), lp=np.concatenate(lps).astype(np.float32))
        print(f"  {name}: {len(seqs)} sequences in {time.time() - started:.0f}s", flush=True)
    result = {"model": m, "dataset": ds, "seed": seed, "outputs": len(rows), "methods": methods, **provenance()}
    write_json(out_dir / "done.json", result)
    return result


# ---------------------------------------------------------------------------------------------------------- 3. summary
def measures(m: str, ds: str, seed: int) -> dict | None:
    """Per method, the mean over the probed outputs of each measure (against the untrained model)."""
    d = config.mechanism_dir(m, ds) / f"probe_s{seed}"
    if not (d / "done.json").exists():
        return None
    items = read_jsonl(d / "items.jsonl")
    responses = {r["id"]: r["response"] for r in read_jsonl(_talk_file(m, ds))}
    names = ["untrained", *config.experiment()["mechanism"]["methods"]]
    scored = {}
    for name in names:
        z = np.load(d / f"{name}.npz")
        lp, off = z["lp"].astype(np.float64), np.concatenate([[0], np.cumsum(z["lens"])])
        arrs = [lp[a:b] for a, b in zip(off[:-1], off[1:])]
        scored[name] = [arrs[3 * k:3 * k + 3] for k in range(len(items))]
    raw = {name: {k: [] for k in ("talk", "other", "skip_talk", "skip_random")} for name in names}
    for k, it in enumerate(items):
        seq0, talk = responses[it["id"]], set(it["format_talk"])
        first = []          # the first position where the output and each shortened variant differ
        for cut in (it["format_talk"], it["random"]):
            seq = core.delete(seq0[:it["n_reason"]], it["lines"], set(cut)) + seq0[it["n_reason"]:]
            first.append(next((i for i, (a, b) in enumerate(zip(seq0, seq)) if a != b), None))
        for name in names:
            t0, t1, t2 = scored[name][k]
            delta = t0 - scored["untrained"][k][0]
            per_line = [delta[a:b].sum() for a, b in it["lines"]]
            raw[name]["talk"].append(np.mean([x for i, x in enumerate(per_line) if i in talk]))
            raw[name]["other"].append(np.mean([x for i, x in enumerate(per_line) if i not in talk]))
            for key, arr, j in (("skip_talk", t1, first[0]), ("skip_random", t2, first[1])):
                # log p(the token after the line) - log p(the line's first token); nothing removed: undefined
                raw[name][key].append(np.nan if j is None else arr[j] - t0[j])
    base = {k: np.array(v) for k, v in raw["untrained"].items()}
    out = {"outputs": len(items), "methods": {}}
    for name in names:
        v = {k: np.array(x) for k, x in raw[name].items()}
        out["methods"][name] = {
            "format_talk_line_dll": float(v["talk"].mean()), "other_line_dll": float(v["other"].mean()),
            "skip_format_talk": float(np.nanmean(v["skip_talk"] - base["skip_talk"])),
            "skip_random": float(np.nanmean(v["skip_random"] - base["skip_random"])),
            "skip_format_talk_logodds": float(np.nanmean(v["skip_talk"])),
            "skip_random_logodds": float(np.nanmean(v["skip_random"]))}
    return out


def summary() -> dict:
    """results/mechanism.json: format talk in the untrained models' test outputs, and the probes per setting and seed
    with the mean over the four settings (`pooled`)."""
    result: dict = {"format_talk": {}, "probes": {}}
    for m, ds in config.SETTINGS:
        if _talk_file(m, ds).exists():
            share = talk_share(read_jsonl(_talk_file(m, ds)))
            result["format_talk"][f"{m}/{ds}"] = {
                "outputs": share["outputs"], "with_format_talk_pct": 100 * share["with_format_talk"] / share["outputs"],
                "token_share_pct": 100 * share["token_share"]}
    for seed in config.experiment()["mechanism"]["seeds"]:
        cells = {f"{m}/{ds}": measures(m, ds, seed) for m, ds in config.SETTINGS}
        cells = {key: value for key, value in cells.items() if value is not None}
        entry: dict = {"settings": cells}
        if len(cells) == len(config.SETTINGS):
            names = list(next(iter(cells.values()))["methods"])
            entry["pooled"] = {"outputs": sum(c["outputs"] for c in cells.values()), "methods": {
                name: {k: float(np.mean([c["methods"][name][k] for c in cells.values()]))
                       for k in cells[f"{config.SETTINGS[0][0]}/{config.SETTINGS[0][1]}"]["methods"][name]}
                for name in names}}
        if cells:
            result["probes"][f"s{seed}"] = entry
    write_json(RESULT, result)
    return {"format_talk": result["format_talk"],
            "pooled": {seed: e["pooled"]["methods"] for seed, e in result["probes"].items() if "pooled" in e}}
