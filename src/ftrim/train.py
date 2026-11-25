"""LoRA fine-tuning on one teacher pool (Sect. 3, "Training"). One process, one GPU.

Every method is trained in the same way: AdamW, learning rate 2e-5 with a linear decay to zero over three passes of
which only the first is trained, batches of 32 questions, gradient clipping at norm 1. The loss of a batch is the mean
over its sequences of the per-sequence mean negative log-likelihood of the response tokens that are not masked.
"""
from __future__ import annotations

import json
import math
import random
import time
from pathlib import Path

from . import chat, config, data, teachers
from .prompts import build_prompt
from .utils import provenance, sha256_file, write_json


def tokenized(model_key: str, ds: str, method: str) -> tuple[list[dict], list[str]]:
    """[{id, input_ids, labels}] of the method's teachers; pairs over the context are excluded and returned."""
    limit = config.experiment()["sampling"]["max_pair_tokens"]
    examples = data.load_index(ds, data.TRAIN_SPLIT[ds])
    items, over = [], []
    for t in teachers.load(model_key, ds, method):
        try:
            p = chat.pair(model_key, build_prompt(examples[t["id"]]), t["target"], max_tokens=limit)
        except chat.PairTooLong:
            over.append(t["id"])
            continue
        labels = [-100] * p.prompt_len + list(p.response)
        mask = t.get("loss_mask") or []
        if mask:
            if len(p.response) != t["response_tokens"]:
                raise ValueError(f"{t['id']}: the response tokenization differs from the one the mask was made on")
            for i in mask:
                labels[p.prompt_len + i] = -100
        items.append({"id": t["id"], "input_ids": list(p.input_ids), "labels": labels,
                      "loss_tokens": sum(x != -100 for x in labels)})
    if len({i["id"] for i in items}) != len(items):
        raise ValueError("a question has two teachers")
    return items, over


def step_order(ids: list[str], seed: int, batch: int) -> list[list[str]]:
    """One pass: the sorted question ids shuffled with the seed and cut into batches (an incomplete last batch is not
    used)."""
    order = sorted(ids)
    random.Random(seed + 1).shuffle(order)
    steps = len(order) // batch
    return [order[s * batch:(s + 1) * batch] for s in range(steps)]


def lr_factor(step: int, horizon: int) -> float:
    """Linear decay to zero over `horizon` steps, no warm-up."""
    return max(0.0, (horizon - step) / horizon)


def micro_batches(items: list[dict], max_tokens: int) -> list[list[dict]]:
    """Right-padded groups (longest first) whose padded size stays within max_tokens (a longer pair is alone). The
    loss is computed per sequence, so the grouping changes memory use and not the update."""
    groups, current = [], []
    for it in sorted(items, key=lambda x: (-len(x["input_ids"]), x["id"])):
        width = len(current[0]["input_ids"]) if current else len(it["input_ids"])
        if current and width * (len(current) + 1) > max_tokens:
            groups.append(current)
            current = []
        current.append(it)
    if current:
        groups.append(current)
    return groups


def lora_targets(model, suffixes: list[str]) -> list[str]:
    """The text-decoder leaf modules with the given suffixes (vision / audio towers are left out). A layer without
    such a module is not adapted there: the linear-attention layers of Qwen3.5 keep their attention unchanged."""
    leaves = [name for name, module in model.named_modules()
              if not any(True for _ in module.children()) and any(name.endswith("." + s) for s in suffixes)]
    text = sorted(n for n in leaves if "language_model" in n.split("."))
    if not text:
        raise ValueError("no LoRA target in the text decoder")
    return text


def fit(model_key: str, ds: str, method: str, seed: int, *, max_steps: int | None = None) -> dict:
    """Train one adapter on the teachers of `method`; `max_steps` cuts the pass short (smoke tests only). A cut run is
    written next to the real one, to fits/<method>/s<seed>-max<steps>, so that the pipeline never takes it for a
    finished fit."""
    import numpy as np
    import torch
    import torch.nn.functional as F
    import transformers
    from peft import LoraConfig, get_peft_model

    out: Path = config.fit_dir(model_key, ds, method, seed)
    if max_steps is not None:
        out = out.with_name(f"{out.name}-max{max_steps}")
    if (out / "done.json").exists():
        return json.loads((out / "done.json").read_text())
    exp, cfg = config.experiment(), config.model(model_key)
    tc = exp["training"]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    items, over = tokenized(model_key, ds, method)
    by_id = {i["id"]: i for i in items}
    steps = step_order(list(by_id), seed, tc["batch"])
    horizon = tc["horizon_factor"] * len(steps)
    if max_steps is not None:
        steps = steps[:max_steps]

    model = getattr(transformers, cfg["loader"]).from_pretrained(
        config.snapshot(model_key), dtype=torch.bfloat16, local_files_only=True, trust_remote_code=False,
        attn_implementation="sdpa")
    lc = tc["lora"]
    model = get_peft_model(model, LoraConfig(r=lc["r"], lora_alpha=lc["alpha"], lora_dropout=lc["dropout"],
                                             target_modules=lora_targets(model, lc["targets"]), bias="none",
                                             task_type="CAUSAL_LM"))
    device = torch.device("cuda")
    model.to(device)
    model.config.use_cache = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    trainable = [p for p in model.parameters() if p.requires_grad]
    if not trainable or any(p.dtype != torch.float32 for p in trainable):
        raise ValueError("the LoRA parameters must be trainable fp32 tensors")
    opt = torch.optim.AdamW(trainable, lr=tc["lr"], betas=tuple(tc["betas"]), eps=tc["eps"],
                            weight_decay=tc["weight_decay"])
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: lr_factor(s, horizon))
    pad = cfg["pad_token_id"]
    out.mkdir(parents=True, exist_ok=True)
    log = (out / "log.jsonl").open("w", encoding="utf-8")
    started = time.time()
    model.train()
    for s, batch_ids in enumerate(steps):
        t0 = time.time()
        opt.zero_grad(set_to_none=True)
        total, tokens = 0.0, 0
        for group in micro_batches([by_id[i] for i in batch_ids], tc["micro_batch_tokens"]):
            width = max(len(g["input_ids"]) for g in group)
            ids = torch.full((len(group), width), pad, dtype=torch.long)
            labels = torch.full((len(group), width), -100, dtype=torch.long)
            attn = torch.zeros((len(group), width), dtype=torch.long)
            for j, g in enumerate(group):
                n = len(g["input_ids"])
                ids[j, :n] = torch.tensor(g["input_ids"])
                labels[j, :n] = torch.tensor(g["labels"])
                attn[j, :n] = 1
            ids, labels, attn = ids.to(device), labels.to(device), attn.to(device)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = model(input_ids=ids, attention_mask=attn).logits
            shift = labels[:, 1:]
            sel = shift != -100
            losses = F.cross_entropy(logits[:, :-1][sel].float(), shift[sel], reduction="none")
            rows = sel.nonzero()[:, 0]
            per_seq = torch.zeros(len(group), device=device, dtype=torch.float32).index_add_(0, rows, losses)
            per_seq = per_seq / sel.sum(dim=1).clamp(min=1)
            loss = per_seq.sum() / tc["batch"]
            loss.backward()
            total += float(loss.detach())
            tokens += int(attn.sum())
            del logits, losses, loss
        norm = float(torch.nn.utils.clip_grad_norm_(trainable, tc["grad_clip"]))
        opt.step()
        sched.step()
        if not math.isfinite(total):
            raise FloatingPointError(f"non-finite loss at step {s + 1}")
        record = {"step": s + 1, "loss": round(total, 6), "grad_norm": round(norm, 4), "lr": sched.get_last_lr()[0],
                  "tokens": tokens, "seconds": round(time.time() - t0, 2)}
        log.write(json.dumps(record) + "\n")
        log.flush()
    log.close()
    adapter = out / "adapter"
    model.save_pretrained(adapter, safe_serialization=True)
    result = {"model": model_key, "dataset": ds, "method": method, "seed": seed, "teachers": len(items),
              "steps": len(steps), "horizon": horizon, "max_steps": max_steps,
              "questions_trained": sum(len(b) for b in steps), "over_length": over,
              "loss_tokens": sum(by_id[i]["loss_tokens"] for b in steps for i in b),
              "seconds": round(time.time() - started), "adapter": str(adapter),
              "adapter_sha256": sha256_file(adapter / "adapter_model.safetensors"),
              "teacher_file_sha256": sha256_file(config.teacher_file(model_key, ds, method)), **provenance()}
    write_json(out / "done.json", result)
    return result
