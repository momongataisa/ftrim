"""Chat rendering and the token view of a teacher, shared by generation, the edits and training.

A training pair is the chat-template rendering of [user: prompt, assistant: target]; its response tokens y_1..y_T are
the tokens after the generation prompt, ending with the template's assistant terminator. The *reasoning* of a teacher
is the response tokens before the token in which the last "<answer>" begins; the rest is the *answer line*. A *line*
of the reasoning ends with a token whose text contains a line break (or with the last reasoning token).
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

from . import config


@lru_cache(maxsize=4)
def tokenizer(model_key: str):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(config.snapshot(model_key), local_files_only=True, trust_remote_code=False)
    if not tok.chat_template:
        raise SystemExit(f"{model_key}: the tokenizer has no chat template")
    return tok


def render(model_key: str, prompt: str, *, thinking: bool = False) -> str:
    return tokenizer(model_key).apply_chat_template([{"role": "user", "content": prompt}], tokenize=False,
                                                    add_generation_prompt=True, enable_thinking=thinking)


def prompt_ids(model_key: str, prompt: str, *, thinking: bool = False) -> list[int]:
    return tokenizer(model_key).encode(render(model_key, prompt, thinking=thinking), add_special_tokens=False)


class PairTooLong(ValueError):
    """The pair does not fit the training context; it is excluded and counted, never truncated."""


@dataclass(frozen=True)
class Pair:
    input_ids: tuple[int, ...]
    prompt_len: int                       # response tokens are input_ids[prompt_len:]

    @property
    def response(self) -> tuple[int, ...]:
        return self.input_ids[self.prompt_len:]


def pair(model_key: str, prompt: str, target: str, *, max_tokens: int | None = None) -> Pair:
    tok, cfg = tokenizer(model_key), config.model(model_key)
    user = [{"role": "user", "content": prompt}]
    prefix = tok.apply_chat_template(user, tokenize=True, add_generation_prompt=True, return_dict=False,
                                     enable_thinking=False)
    full = tok.apply_chat_template(user + [{"role": "assistant", "content": target}], tokenize=True,
                                   add_generation_prompt=False, return_dict=False, enable_thinking=False)
    if full[:len(prefix)] != prefix:
        raise ValueError("the training rendering does not start with the generation prompt")
    term = list(cfg["assistant_terminator"])
    if full[-len(term):] != term:
        raise ValueError("the rendering does not end with the assistant terminator")
    if max_tokens is not None and len(full) > max_tokens:
        raise PairTooLong(f"{len(full)} tokens > {max_tokens}")
    return Pair(tuple(full), len(prefix))


def answer_token(model_key: str, body: list[int] | tuple[int, ...], text: str) -> int | None:
    """Index of the first body token whose decoded prefix reaches the last "<answer>" of `text` (binary search)."""
    tok = tokenizer(model_key)
    at = text.rfind("<answer>")
    if at < 0:
        return None
    lo, hi = 0, len(body)
    while lo < hi:
        mid = (lo + hi) // 2
        if len(tok.decode(list(body[:mid + 1]))) > at:
            hi = mid
        else:
            lo = mid + 1
    return lo if lo < len(body) else None


@dataclass(frozen=True)
class TeacherTokens:
    pair: Pair
    n_reason: int                          # reasoning = response[:n_reason]
    lines: tuple[tuple[int, int], ...]     # [start, end) ranges over the reasoning tokens

    @property
    def reasoning(self) -> tuple[int, ...]:
        return self.pair.response[:self.n_reason]


@lru_cache(maxsize=1 << 16)
def _has_newline(model_key: str, token: int) -> bool:
    return "\n" in tokenizer(model_key).decode([token])


def teacher_tokens(model_key: str, prompt: str, target: str, *, max_tokens: int | None = None) -> TeacherTokens:
    p = pair(model_key, prompt, target, max_tokens=max_tokens)
    term = len(config.model(model_key)["assistant_terminator"])
    body = p.response[:len(p.response) - term]
    n = answer_token(model_key, body, target)
    if n is None:
        raise ValueError("the target has no answer tag")
    tok = tokenizer(model_key)
    reason = tok.decode(list(body[:n]))
    if not target.startswith(reason) or not target[len(reason):].lstrip(" ").startswith("<answer>"):
        raise ValueError("the decoded reasoning does not reproduce the target")
    lines, start = [], 0
    for i in range(n):
        if _has_newline(model_key, body[i]) or i == n - 1:
            lines.append((start, i + 1))
            start = i + 1
    return TeacherTokens(p, n, tuple(lines))


def delete_lines(model_key: str, tt: TeacherTokens, target: str, deleted: set[int]) -> str:
    """The target with the given reasoning lines (indices into tt.lines) removed; the answer line is kept."""
    tok = tokenizer(model_key)
    drop = {i for k in deleted for i in range(*tt.lines[k])}
    reason = tok.decode(list(tt.reasoning))
    kept = tok.decode([t for i, t in enumerate(tt.reasoning) if i not in drop])
    return kept + target[len(reason):]
