"""GPU generation stages: the evaluation of a method on its benchmarks, and the sampled solutions of the untrained
model (steps 1 and 2 of Fig. 1)."""
from __future__ import annotations

import time
from collections import Counter
from pathlib import Path

from . import config, data, scoring
from .prompts import build_prompt, hint_prompt
from .teachers import draw_flags
from .utils import provenance, read_json, read_jsonl, stable_seed, write_json, write_jsonl

# The names of the per-request seed streams of the paper's runs (see utils.stable_seed).
SEED_STREAM = {"sample": "own", "rationalize": "card"}


# ---------------------------------------------------------------------------------------------------------- evaluation
def eval_requests(model_key: str, examples: list[dict], *, thinking: bool, adapter: str | None):
    from .engine import Request
    exp, cfg = config.experiment(), config.model(model_key)
    if thinking and examples:
        from .chat import render
        probe = build_prompt(examples[0])
        if render(model_key, probe, thinking=True) == render(model_key, probe, thinking=False):
            raise ValueError("the chat template renders the same prompt with thinking on and off")
    reqs = []
    for ex in examples:
        if thinking:
            s = cfg["thinking_sampling"]
            reqs.append(Request(ex["id"], build_prompt(ex), exp["evaluation"]["thinking_max_new_tokens"],
                                temperature=s["temperature"], top_p=s["top_p"], top_k=s.get("top_k", -1),
                                min_p=s.get("min_p", 0.0), presence_penalty=s.get("presence_penalty", 0.0),
                                seed=stable_seed("thinking", model_key, ex["id"]), thinking=True, lora=adapter))
        else:
            reqs.append(Request(ex["id"], build_prompt(ex), exp["evaluation"]["max_new_tokens"][ex["dataset"]],
                                seed=stable_seed("eval", ex["id"]), lora=adapter))
    return reqs


def score_rows(model_key: str, examples: list[dict], outs: dict, *, thinking: bool) -> list[dict]:
    rows = []
    for ex in examples:
        o = outs[ex["id"]][0]
        text = scoring.thinking_final_answer(o["text"], config.model(model_key)["hf_id"]) if thinking else o["text"]
        rows.append({"id": ex["id"], "document_id": ex["document_id"], "text": o["text"], "tokens": o["tokens"],
                     "finish": o["finish"], "prompt_tokens": o["prompt_tokens"],
                     **({"final": text} if thinking else {}), "score": scoring.score(ex, text)})
    return rows


def summarize(rows: list[dict]) -> dict:
    n = len(rows)
    keys = sorted({k for r in rows for k, v in r["score"].items()
                   if isinstance(v, (int, float, bool)) and v is not None})
    return {"questions": n, **{k: round(100 * sum(float(r["score"].get(k) or 0) for r in rows) / n, 3) for k in keys},
            "truncated": sum(r["finish"] == "length" for r in rows),
            "context_overflow": sum(r["finish"] == "context" for r in rows),
            "mean_output_tokens": round(sum(r["tokens"] for r in rows) / n, 1)}


def run_eval(model_key: str, method: str, *, ds: str | None = None, seed: int | None = None) -> dict:
    """Evaluate the untrained model (`untrained`, `thinking`: every target task and unseen benchmark) or a fine-tuned
    adapter (its target task and the unseen benchmarks): greedy decoding, one answer per question."""
    from .engine import Engine
    thinking = method == "thinking"
    out_dir: Path = config.eval_dir(model_key, method, ds=ds, seed=seed)
    adapter = None
    if method not in config.UNTRAINED:
        adapter_dir = config.fit_dir(model_key, ds, method, seed) / "adapter"
        if not (adapter_dir / "adapter_config.json").exists():
            raise SystemExit(f"{adapter_dir} is missing: run `python -m ftrim fit` first")
        adapter = str(adapter_dir)
    benches = config.eval_benchmarks(None if method in config.UNTRAINED else ds)
    todo = [b for b in benches if not (out_dir / f"{b}-test.jsonl").exists()]
    if todo:
        examples = {b: data.load(b, "test") for b in todo}
        engine = Engine(model_key, lora=adapter is not None)
        reqs = [r for b in todo for r in eval_requests(model_key, examples[b], thinking=thinking, adapter=adapter)]
        started = time.time()
        outs = engine.generate(reqs, log_every=2000)
        seconds = time.time() - started
        for b in todo:
            rows = score_rows(model_key, examples[b], outs, thinking=thinking)
            write_jsonl(out_dir / f"{b}-test.jsonl", rows)
            write_json(out_dir / f"{b}-test.summary.json",
                       {**summarize(rows), "method": method, "model": model_key, "dataset": ds, "seed": seed,
                        "generation_seconds": seconds, **provenance()})
    result = {b: read_json(out_dir / f"{b}-test.summary.json") for b in benches}
    write_json(out_dir / "done.json", {"benchmarks": sorted(result), **provenance()})
    return {b: v.get("primary") for b, v in result.items()}


# ---------------------------------------------------------------------------------------------------- sampled solutions
def unsolved_ids(model_key: str, ds: str) -> list[str]:
    """Training questions without a usable sampled solution: rationalization samples again with a hint."""
    rows = read_jsonl(config.draw_file(model_key, ds, "sample"))
    return [r["id"] for r in rows if not any(d["usable"] for d in r["draws"])]


def run_draws(model_key: str, ds: str, kind: str) -> dict:
    """`sample`: K solutions y ~ p0(. | x) per training question. `rationalize`: K solutions per unsolved question,
    sampled with the gold answer as a hint in the prompt."""
    from .engine import Engine, Request
    dest = config.draw_file(model_key, ds, kind)
    if dest.with_suffix(".summary.json").exists():          # written last: the stage is complete
        return read_json(dest.with_suffix(".summary.json"))
    s = config.experiment()["sampling"]
    pool = data.load(ds, data.TRAIN_SPLIT[ds])
    if kind == "sample":
        examples, make = pool, build_prompt
    elif kind == "rationalize":
        unsolved = set(unsolved_ids(model_key, ds))
        examples, make = [ex for ex in pool if ex["id"] in unsolved], hint_prompt
    else:
        raise ValueError(kind)
    k = s["k"]
    reqs = [Request(f"{ex['id']}#{d}", make(ex), s["max_new_tokens"], temperature=s["temperature"], top_p=s["top_p"],
                    seed=stable_seed(SEED_STREAM[kind], model_key, ex["id"], d))
            for ex in examples for d in range(k)]
    started = time.time()
    outs = Engine(model_key).generate(reqs, log_every=5000)
    seconds = time.time() - started
    rows = []
    for ex in examples:
        draws = []
        for d in range(k):
            o = outs[f"{ex['id']}#{d}"][0]
            draws.append({"draw": d, "text": o["text"], "tokens": o["tokens"], "finish": o["finish"],
                          **draw_flags(ex, o["text"])})
        rows.append({"id": ex["id"], "draws": draws})
    write_jsonl(dest, rows)
    flat = [d for r in rows for d in r["draws"]]
    summary = {"model": model_key, "dataset": ds, "kind": kind, "questions": len(rows), "draws": len(flat),
               "correct": sum(d["correct"] for d in flat), "usable": sum(d["usable"] for d in flat),
               "questions_with_usable": sum(any(d["usable"] for d in r["draws"]) for r in rows),
               "excluded": dict(Counter(f for d in flat for f in ("mentions_hint", "control_tokens", "no_reasoning")
                                        if d[f])),
               "truncated": sum(d["finish"] == "length" for d in flat), "generation_seconds": seconds, **provenance()}
    write_json(dest.with_suffix(".summary.json"), summary)
    return summary
