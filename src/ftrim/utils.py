"""Small helpers shared by every stage: hashing, atomic JSON / JSONL / npz files, seeded random streams, provenance."""
from __future__ import annotations

import hashlib
import json
import os
import random
import subprocess
import time
from collections.abc import Iterable
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def stable_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def digest(value: Any) -> str:
    return sha256_text(stable_json(value))


def stable_seed(*parts: Any, bits: int = 32) -> int:
    """A reproducible integer seed from any JSON-serialisable parts (the per-request seeds of generation)."""
    return int(digest(list(parts))[: bits // 4], 16)


def rng_for(*parts: Any) -> random.Random:
    """A reproducible random stream named by its parts (random deletion, the probes of the mechanism analysis)."""
    return random.Random(int(digest(list(parts)), 16))


def write_json(path: str | Path, value: Any, *, indent: int | None = 1) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=indent) + "\n", encoding="utf-8")
    tmp.replace(path)


def read_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_jsonl(path: str | Path, rows: Iterable[Any]) -> int:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    n = 0
    with tmp.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            n += 1
    tmp.replace(path)
    return n


def save_npz(path: str | Path, **arrays: Any) -> None:
    """`np.savez` through a temporary file: a job that is killed while writing leaves no partial file under the
    final name, so the file's existence means that it is complete."""
    import numpy as np
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with tmp.open("wb") as f:
        np.savez(f, **arrays)
    tmp.replace(path)


def read_jsonl(path: str | Path) -> list[Any]:
    with Path(path).open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def git_state() -> dict:
    """Commit and dirtiness of this repository, recorded with every output."""
    def run(*args):
        try:
            return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, timeout=20).stdout.strip()
        except (OSError, subprocess.SubprocessError):
            return ""
    return {"commit": run("rev-parse", "HEAD"), "dirty": bool(run("status", "--porcelain", "--untracked-files=no"))}


def provenance(**extra: Any) -> dict:
    return {"created": now(), "git": git_state(), **extra}
