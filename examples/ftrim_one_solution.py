"""FTrim on one solution, with plain Hugging Face transformers.

The script scores a solution under two prompts with one forward pass each, prints the line scores m(l) and deletes
the lines with m(l) >= 1 bit per token. Of this package it uses only `ftrim.core` (the rule) and `ftrim.config` (the
pinned model revisions); it needs the `gpu` extra for torch, transformers and accelerate. The scores differ slightly
from the pipeline's, which scores with vLLM, and the edited solution is not checked by the verifier here.

    python examples/ftrim_one_solution.py                                  # the teacher of Fig. 1 of the paper
    python examples/ftrim_one_solution.py --example my.json --model <hf id>

An example file is a JSON object with `prompt` (with the format instructions), `bare_prompt` (the counterfactual
prompt: the same prompt with only the answer tags described) and `solution` (the model's own solution, ending with
its answer line). The default
example needs google/gemma-4-E4B-it: about 18 GB of GPU memory in bfloat16, so one GPU with 24 GB.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import transformers

from ftrim import config, core

HERE = Path(__file__).resolve().parent


def load(name: str):
    """A model of configs/experiment.yaml (`gemma`, `qwen`: the pinned revision) or any causal LM of the Hub."""
    known = config.experiment()["models"]
    if name in known:
        cfg = known[name]
        loader, name, revision = getattr(transformers, cfg["loader"]), cfg["hf_id"], cfg["revision"]
    else:
        loader, revision = transformers.AutoModelForCausalLM, None
    tok = transformers.AutoTokenizer.from_pretrained(name, revision=revision)
    model = loader.from_pretrained(name, revision=revision, dtype=torch.bfloat16, device_map="auto")
    return tok, model.eval()


@torch.no_grad()
def response_logprobs(tok, model, prompt: str, solution: str) -> tuple[list[int], np.ndarray]:
    """The tokens of the assistant turn that holds `solution`, and log p(token | prompt, tokens before)."""
    user = [{"role": "user", "content": prompt}]
    kwargs = dict(tokenize=True, return_dict=False, enable_thinking=False)
    prefix = tok.apply_chat_template(user, add_generation_prompt=True, **kwargs)
    full = tok.apply_chat_template([*user, {"role": "assistant", "content": solution}], add_generation_prompt=False,
                                   **kwargs)
    if full[:len(prefix)] != prefix:
        raise ValueError("the chat template does not render the assistant turn after the generation prompt")
    ids = torch.tensor([full], device=model.device)
    logp = torch.log_softmax(model(input_ids=ids).logits[0].float(), dim=-1)
    token_logp = logp[:-1].gather(1, ids[0, 1:, None])[:, 0]              # position i predicts token i + 1
    return full[len(prefix):], token_logp[len(prefix) - 1:].double().cpu().numpy()


def reasoning_lines(tok, response: list[int], solution: str) -> tuple[int, list[tuple[int, int]]]:
    """The number of reasoning tokens (those before the token in which the last answer tag begins) and the token
    ranges of the reasoning lines."""
    at = solution.rfind("<answer>")
    n = next(i for i in range(len(response)) if len(tok.decode(response[:i + 1])) > at)
    return n, core.split_lines(["\n" in tok.decode([t]) for t in response[:n]])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--example", default=str(HERE / "tatqa_example.json"))
    ap.add_argument("--model", help="gemma, qwen or a Hugging Face id (default: the model named in the example)")
    ap.add_argument("--threshold-bits", type=float, default=1.0)
    a = ap.parse_args()
    example = json.loads(Path(a.example).read_text(encoding="utf-8"))
    tok, model = load(a.model or example["model"])

    response, lp_full = response_logprobs(tok, model, example["prompt"], example["solution"])
    response_bare, lp_bare = response_logprobs(tok, model, example["bare_prompt"], example["solution"])
    assert response == response_bare
    n, lines = reasoning_lines(tok, response, example["solution"])
    scores = core.line_scores(lp_full[:n], lp_bare[:n], lines)
    deleted = core.format_talk_lines(lp_full[:n], lp_bare[:n], lines, threshold_bits=a.threshold_bits)

    print(f"{'m(l)':>7}  line   (bits per token; deleted lines are marked with x)")
    for k, ((start, end), m) in enumerate(zip(lines, scores)):
        text = tok.decode(response[start:end]).rstrip("\n")
        print(f"{m:+7.2f}  {'x' if k in deleted else ' '} {text}")
    kept = core.delete(response[:n], lines, deleted)
    share = 1 - len(kept) / n
    print(f"\nFTrim deletes {len(deleted)} of {len(lines)} lines ({100 * share:.0f} % of the reasoning tokens).\n")
    print(tok.decode(kept) + example["solution"][example["solution"].rfind("<answer>"):])


if __name__ == "__main__":
    main()
