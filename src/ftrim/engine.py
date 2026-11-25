"""vLLM engines: generation (greedy evaluation, sampled solutions, thinking mode) and teacher-forced token
log-probabilities (FTrim's scoring, STM's perplexities, the mechanism analysis)."""
from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from . import config
from .chat import prompt_ids


@dataclass
class Request:
    key: str                               # unique id of the request (question id)
    prompt: str
    max_tokens: int
    n: int = 1
    temperature: float = 0.0
    top_p: float = 1.0
    top_k: int = -1
    min_p: float = 0.0
    presence_penalty: float = 0.0
    seed: int = 0
    thinking: bool = False
    lora: str | None = None                # adapter directory


class Engine:
    def __init__(self, model_key: str, *, lora: bool = False, max_loras: int = 1, kind: str = "generate",
                 max_model_len: int | None = None):
        from vllm import LLM
        self.model_key = model_key
        self.cfg = config.model(model_key)
        exp = config.experiment()
        settings = dict(exp["engines"][kind])
        self.max_model_len = max_model_len or exp["evaluation"]["max_model_len"]
        snap = str(config.snapshot(model_key))
        kwargs = dict(model=snap, tokenizer=snap, dtype="bfloat16", trust_remote_code=False, seed=0,
                      max_model_len=self.max_model_len, enable_prefix_caching=False,
                      enable_chunked_prefill=kind == "score", limit_mm_per_prompt={"image": 0, "audio": 0, "video": 0},
                      **settings)
        if lora:
            kwargs.update(enable_lora=True, max_lora_rank=exp["training"]["lora"]["r"], max_loras=max_loras)
        if kind == "score":
            kwargs.update(max_logprobs=1)
        self.llm = LLM(**kwargs)
        self._lora_ids: dict[str, int] = {}

    def _lora(self, path: str | None):
        if path is None:
            return None
        from vllm.lora.request import LoRARequest
        if path not in self._lora_ids:
            self._lora_ids[path] = len(self._lora_ids) + 1
        return LoRARequest(f"adapter-{self._lora_ids[path]}", self._lora_ids[path], path)

    def generate(self, requests: list[Request], *, log_every: int = 0) -> dict[str, list[dict]]:
        """{request key: [{"text", "tokens", "finish"} per sample]}; requests whose prompt does not fit the context are
        reported with finish = "context" and no generation (never truncated)."""
        from vllm import SamplingParams
        from vllm.inputs import TokensPrompt
        out: dict[str, list[dict]] = {}
        engine = self.llm.llm_engine
        pending = {}
        for r in requests:
            ids = prompt_ids(self.model_key, r.prompt, thinking=r.thinking)
            if len(ids) + r.max_tokens > self.max_model_len:
                out[r.key] = [{"text": "", "tokens": 0, "finish": "context", "prompt_tokens": len(ids)}] * r.n
                continue
            params = SamplingParams(n=r.n, temperature=r.temperature, top_p=r.top_p, top_k=r.top_k, min_p=r.min_p,
                                    presence_penalty=r.presence_penalty, max_tokens=r.max_tokens, seed=r.seed,
                                    stop_token_ids=list(self.cfg["stop_token_ids"]),
                                    skip_special_tokens=not r.thinking)
            if r.key in pending or r.key in out:
                raise ValueError(f"duplicate request {r.key}")
            pending[r.key] = len(ids)
            engine.add_request(r.key, TokensPrompt(prompt_token_ids=ids), params, lora_request=self._lora(r.lora))
        started, done = time.time(), 0
        while engine.has_unfinished_requests():
            for o in engine.step():
                if not o.finished:
                    continue
                if o.request_id not in pending:
                    raise ValueError(f"unknown request {o.request_id}")
                out[o.request_id] = [{"text": c.text, "tokens": len(c.token_ids), "finish": c.finish_reason,
                                      "prompt_tokens": pending[o.request_id]}
                                     for c in sorted(o.outputs, key=lambda c: c.index)]
                done += 1
                if log_every and done % log_every == 0:
                    print(f"  {done}/{len(pending)} done, {time.time() - started:.0f}s", flush=True)
        missing = set(pending) - set(out)
        if missing:
            raise RuntimeError(f"{len(missing)} requests did not finish")
        return out

    def token_logprobs(self, sequences: list[list[int]], starts: list[int], *,
                       lora: str | None = None) -> list[np.ndarray]:
        """Natural-log p(token_i | tokens_<i) for i >= start of each sequence (teacher forcing, no generation), under
        the base model or under the adapter `lora`."""
        from vllm import SamplingParams
        from vllm.inputs import TokensPrompt
        params = SamplingParams(max_tokens=1, temperature=0.0, prompt_logprobs=1, detokenize=False, flat_logprobs=True)
        outs = self.llm.generate([TokensPrompt(prompt_token_ids=list(s)) for s in sequences], params, use_tqdm=False,
                                 lora_request=self._lora(lora))
        result = []
        for seq, start, o in zip(sequences, starts, outs, strict=True):
            plp = o.prompt_logprobs
            st, ti, lp = plp.start_indices, plp.token_ids, plp.logprobs
            row = np.empty(len(seq) - start, np.float64)
            for i in range(start, len(seq)):
                a = st[i]
                if ti[a] != seq[i]:
                    raise ValueError("prompt logprobs are not aligned with the sequence")
                row[i - start] = lp[a]
            result.append(row)
        return result
