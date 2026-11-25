"""Format-talk deletion (lines 7 and 8 of Algorithm 1) on the normalized teachers, and the comparison edits.

`score` (GPU) records, for every normalized teacher, the untrained model's log-probability of each response token
under the prompt with the format instructions and under the counterfactual prompt, teacher-forced on the exact
training token sequence: two forward passes per teacher. `build` (CPU) derives every edited pool from these numbers:

* ftrim            delete the reasoning lines with m(l) >= 1 bit per token;
* ftrim_*          the same rule with another threshold (configs/experiment.yaml);
* random_deletion  for each teacher that FTrim edits, delete as many tokens in other, random lines;
* stm              leave the reasoning tokens with perplexity above tau out of the loss (Wu et al., 2025).

The answer normalization is a part of FTrim. Random deletion and STM are therefore made on the rationalization
teachers: the same reasoning as the normalized teacher (the two differ only in the answer line), with the answer line
as the model wrote it. The reasoning tokens are the same in both teachers, so the lines that random deletion removes
and the tokens that STM masks do not depend on the answer line.

An edit never touches the answer line, and it is not made if the edited teacher would not have exactly one answer
marker or would not pass the verifier.
"""
from __future__ import annotations

from collections import Counter
from collections.abc import Iterator

import numpy as np

from . import chat, config, core, data, scoring, teachers
from .prompts import build_prompt
from .utils import provenance, read_json, rng_for, save_npz, write_json, write_jsonl


# ---------------------------------------------------------------------------------------------------------- scoring
def score(model_key: str, ds: str, *, chunk: int = 1000) -> dict:
    """Token log-probabilities of the normalized teachers under both prompts (resumable by chunk)."""
    from .engine import Engine
    dest = config.scores_file(model_key, ds)
    if dest.with_suffix(".summary.json").exists():          # written last: the stage is complete
        return read_json(dest.with_suffix(".summary.json"))
    rows = teachers.load(model_key, ds, "normalized")
    examples = data.load_index(ds, data.TRAIN_SPLIT[ds])
    limit = config.experiment()["sampling"]["max_pair_tokens"]
    part_dir = dest.parent / "parts"
    part_dir.mkdir(parents=True, exist_ok=True)
    engine = None
    for c in range(0, len(rows), chunk):
        part = part_dir / f"{c // chunk:04d}.npz"
        if part.exists():
            continue
        seqs, starts, ids = [], [], []
        for r in rows[c:c + chunk]:
            ex = examples[r["id"]]
            full = chat.pair(model_key, build_prompt(ex), r["target"], max_tokens=limit)
            bare = chat.pair(model_key, build_prompt(ex, bare=True), r["target"])
            if full.response != bare.response:
                raise ValueError(f"{r['id']}: the response tokens differ between the two prompts")
            seqs += [list(full.input_ids), list(bare.input_ids)]
            starts += [full.prompt_len, bare.prompt_len]
            ids.append(r["id"])
        engine = engine or Engine(model_key, kind="score", max_model_len=limit + 512)
        lps = engine.token_logprobs(seqs, starts)
        lens = np.array([len(x) for x in lps[0::2]], np.int64)
        save_npz(part, ids=np.array(ids), lens=lens, lp_full=np.concatenate(lps[0::2]).astype(np.float32),
                 lp_bare=np.concatenate(lps[1::2]).astype(np.float32))
        print(f"  scored {min(c + chunk, len(rows))}/{len(rows)}", flush=True)
    parts = [np.load(p) for p in sorted(part_dir.glob("*.npz"))]
    ids = np.concatenate([p["ids"] for p in parts])
    if list(ids) != [r["id"] for r in rows]:
        raise ValueError("the score parts do not cover the normalized teachers in order")
    save_npz(dest, ids=ids, lens=np.concatenate([p["lens"] for p in parts]),
             lp_full=np.concatenate([p["lp_full"] for p in parts]),
             lp_bare=np.concatenate([p["lp_bare"] for p in parts]))
    summary = {"model": model_key, "dataset": ds, "teachers": len(rows), **provenance()}
    write_json(dest.with_suffix(".summary.json"), summary)
    return summary


def load_scores(model_key: str, ds: str) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """{question id: (log p under the full prompt, under the counterfactual prompt)} per response token, in nats."""
    z = np.load(config.scores_file(model_key, ds))
    full, bare = z["lp_full"].astype(np.float64), z["lp_bare"].astype(np.float64)
    off = np.concatenate([[0], np.cumsum(z["lens"])])
    return {str(i): (full[a:b], bare[a:b]) for i, a, b in zip(z["ids"], off[:-1], off[1:])}


def scored_teachers(model_key: str,
                    ds: str) -> Iterator[tuple[dict, chat.TeacherTokens | None, np.ndarray, np.ndarray]]:
    """Every normalized teacher with its token view and its reasoning log-probabilities under the two prompts.
    The view is None for a teacher whose reasoning cannot be split from its answer line (it stays unedited)."""
    examples = data.load_index(ds, data.TRAIN_SPLIT[ds])
    lp = load_scores(model_key, ds)
    for r in teachers.load(model_key, ds, "normalized"):
        try:
            tt = chat.teacher_tokens(model_key, build_prompt(examples[r["id"]]), r["target"])
        except ValueError:
            yield r, None, np.empty(0), np.empty(0)
            continue
        full, bare = lp[r["id"]]
        if len(full) != len(tt.pair.response):
            raise ValueError(f"{r['id']}: the scores do not match the tokenization")
        yield r, tt, full[:tt.n_reason], bare[:tt.n_reason]


# ------------------------------------------------------------------------------------------------------------ editing
def passes_check(ex: dict, target: str) -> bool:
    """The edited teacher has exactly one answer marker and passes the verifier."""
    return target.count("<answer>") == 1 and scoring.score(ex, target)["primary"] == 1


def reasoning_text(target: str) -> str:
    return target[:target.rfind("<answer>")]


def delete_checked(model_key: str, ex: dict, teacher: dict, tt: chat.TeacherTokens,
                   lines: set[int]) -> tuple[dict, set[int]]:
    """The teacher without the given reasoning lines, and the lines deleted: none if the check fails."""
    target = chat.delete_lines(model_key, tt, teacher["target"], lines) if lines else teacher["target"]
    if lines and not passes_check(ex, target):
        lines, target = set(), teacher["target"]
    tokens = sum(tt.lines[k][1] - tt.lines[k][0] for k in lines)
    return {**teacher, "target": target, "edited": bool(lines), "deleted_tokens": tokens,
            "deleted_lines": len(lines)}, lines


def answer_line(target: str) -> str:
    return target[target.rfind("<answer>"):]


def on_rationalization(ex: dict, edited: dict, source: dict) -> dict:
    """A deletion made on a normalized teacher, moved to its rationalization teacher `source`: the edited reasoning,
    then the answer line of `source`. The deletion is not made if this teacher fails the check."""
    target = reasoning_text(edited["target"]) + answer_line(source["target"])
    if not edited["edited"] or not passes_check(ex, target):
        return {**source, "edited": False, "deleted_tokens": 0, "deleted_lines": 0}
    return {**source, "target": target, "edited": True, "deleted_tokens": edited["deleted_tokens"],
            "deleted_lines": edited["deleted_lines"]}


def build(model_key: str, ds: str) -> dict:
    """Write every edited pool of configs/experiment.yaml (`edits`) and teachers/edits.json with their statistics."""
    out = config.setting_dir(model_key, ds) / "teachers"
    if (out / "edits.json").exists():
        return read_json(out / "edits.json")
    cfg = config.experiment()["edits"]
    thresholds = {name: rule for name, rule in cfg.items() if "threshold_bits" in rule}
    randoms = {name: rule for name, rule in cfg.items() if "like" in rule}
    maskers = {name: rule for name, rule in cfg.items() if "tau" in rule}
    examples = data.load_index(ds, data.TRAIN_SPLIT[ds])
    plain = teachers.load(model_key, ds, "rationalization")     # the teachers before the answer normalization
    pools = {name: [] for name in cfg}
    stats = {name: Counter() for name in cfg}
    total = n_teachers = 0

    for (r, tt, full, bare), r0 in zip(scored_teachers(model_key, ds), plain, strict=True):
        if r0["id"] != r["id"] or reasoning_text(r0["target"]) != reasoning_text(r["target"]):
            raise ValueError(f"{r['id']}: the normalized and the rationalization teacher differ before the answer")
        n_teachers += 1
        if tt is None:
            for name in pools:
                pools[name].append({**(r if name in thresholds else r0), "edited": False})
            continue
        ex = examples[r["id"]]
        sizes = [b - a for a, b in tt.lines]
        total += tt.n_reason
        deleted = {}
        for name, rule in thresholds.items():
            chosen = core.format_talk_lines(full, bare, tt.lines, threshold_bits=rule["threshold_bits"],
                                            strict=rule.get("strict", False))
            edited, deleted[name] = delete_checked(model_key, ex, r, tt, chosen)
            pools[name].append(edited)
            stats[name]["tokens_before_check"] += sum(sizes[k] for k in chosen)
            stats[name]["kept_by_check"] += bool(chosen) and not deleted[name]
        for name, rule in randoms.items():
            base = deleted[rule["like"]]
            goal = sum(sizes[k] for k in base)
            chosen: set[int] = set()
            if base:
                free = [k for k in range(len(tt.lines)) if k not in base]
                rng_for(rule["stream"], model_key, ds, r["id"]).shuffle(free)
                chosen, _ = core.random_lines(free, sizes, goal)
            edited, _ = delete_checked(model_key, ex, r, tt, chosen)
            edited = on_rationalization(ex, edited, r0)
            pools[name].append(edited)
            stats[name]["goal_tokens"] += goal
            stats[name]["tokens_before_check"] += sum(sizes[k] for k in chosen)
            stats[name]["kept_by_check"] += bool(chosen) and not edited["edited"]
        for name, rule in maskers.items():
            mask = core.stm_mask(full, tt.n_reason, rule["tau"])
            pools[name].append({**r0, "edited": bool(mask), "loss_mask": mask})     # positions in the reasoning

    summary = {"model": model_key, "dataset": ds, "teachers": n_teachers, "reasoning_tokens": total, "edits": cfg,
               "pools": {}}
    for name, pool in pools.items():
        removed = sum(len(p["loss_mask"]) if "loss_mask" in p else p.get("deleted_tokens", 0) for p in pool)
        info = {"teachers": len(pool), "edited": sum(p["edited"] for p in pool), "tokens": removed,
                "share_of_reasoning": round(removed / total, 6)}
        if name not in maskers:
            info.update(share_before_check=round(stats[name]["tokens_before_check"] / total, 6),
                        kept_by_check=stats[name]["kept_by_check"],
                        no_reasoning=sum(not reasoning_text(p["target"]).strip() for p in pool))
        if name in randoms:
            info["goal_share_of_reasoning"] = round(stats[name]["goal_tokens"] / total, 6)
        summary["pools"][name] = info
        write_jsonl(config.teacher_file(model_key, ds, name), pool)
    summary.update(provenance())
    write_json(out / "edits.json", summary)
    return summary
