"""Run the stages of the experiment as a dependency graph over the free GPUs.

Each job is one `python -m ftrim <stage> ...` subprocess on one GPU, with its own log under runs/logs/. A job is done
when its marker file exists, so the pipeline can be stopped and restarted at any time. A GPU is used only while no
other process holds memory on it. Host-specific environment variables for the jobs (for example LD_LIBRARY_PATH) can
be put in an untracked configs/host.local.yaml; `FTRIM_NVIDIA_SMI` names another nvidia-smi binary.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from . import config, data
from .utils import ROOT, now

POOL_STAGE = {"sft": "teachers", "rejection": "teachers", "rationalization": "teachers", "normalized": "teachers"}


@dataclass
class Job:
    name: str
    args: list[str]
    done: Path
    gpu: bool = True
    deps: list[str] = field(default_factory=list)
    priority: int = 50


def _teacher_jobs(m: str, ds: str, data_dep: list[str], *, edit: bool = True) -> list[Job]:
    """Sample -> rationalize -> pools -> FTrim's scoring (-> edited pools) of one model and dataset."""
    cell, base = config.setting_dir(m, ds), ["--model", m, "--dataset", ds]
    out = [
        Job(f"{m}:{ds}:sample", ["sample", *base], cell / "draws/sample.summary.json", deps=data_dep, priority=10),
        Job(f"{m}:{ds}:rationalize", ["rationalize", *base], cell / "draws/rationalize.summary.json",
            deps=[f"{m}:{ds}:sample"], priority=10),
        Job(f"{m}:{ds}:teachers", ["teachers", *base], cell / "teachers/summary.json", gpu=False,
            deps=[f"{m}:{ds}:rationalize"], priority=10),
        Job(f"{m}:{ds}:score", ["score", *base], cell / "scores/logprobs.summary.json", deps=[f"{m}:{ds}:teachers"],
            priority=10),
    ]
    if edit:
        out.append(Job(f"{m}:{ds}:edit", ["edit", *base], cell / "teachers/edits.json", gpu=False,
                       deps=[f"{m}:{ds}:score"], priority=10))
    return out


def jobs(settings: list[tuple[str, str]], methods: dict[str, list[int]], *, untrained: bool = True,
         thinking: bool = True, mechanism: bool = False, mechanism_seeds: list[int] | None = None) -> list[Job]:
    """The experiment on the given (model, target task) settings: the evaluations of the untrained model (without and
    with its thinking mode), teachers, edits, one fit and one evaluation per method and seed, and (optionally) the
    mechanism analysis of Sect. 5 for `mechanism_seeds` (default: the seeds of configs/experiment.yaml)."""
    exp = config.experiment()
    for m, ds in settings:
        if m not in exp["models"] or ds not in exp["datasets"]:
            raise ValueError(f"unknown setting {m}/{ds}")
    out = [Job("prepare-data", ["prepare-data"], data.PROCESSED / "summary.json", gpu=False, priority=0)]
    data_dep = ["prepare-data"]
    mech = exp["mechanism"]
    mech_seeds = list(mech["seeds"] if mechanism_seeds is None else mechanism_seeds)
    for m in dict.fromkeys(m for m, _ in settings):
        if untrained or mechanism:
            out.append(Job(f"{m}:untrained", ["eval", "--model", m, "--method", "untrained"],
                           config.eval_dir(m, "untrained") / "done.json", deps=data_dep, priority=20))
        if thinking:        # long: start it before the fits, not at the end
            out.append(Job(f"{m}:thinking", ["eval", "--model", m, "--method", "thinking"],
                           config.eval_dir(m, "thinking") / "done.json", deps=data_dep, priority=35))
    for m, ds in dict.fromkeys(settings):
        base = ["--model", m, "--dataset", ds]
        out += _teacher_jobs(m, ds, data_dep)
        wanted = {name: list(seeds) for name, seeds in methods.items()}
        if mechanism:   # the probes need the adapters of these methods
            for name in mech["methods"]:
                wanted[name] = sorted(set(wanted.get(name, [])) | set(mech_seeds))
        for method, seeds in wanted.items():
            pool = f"{m}:{ds}:{POOL_STAGE.get(method, 'edit')}"
            for k, seed in enumerate(seeds):     # the first seed of every method is trained first
                fit = f"{m}:{ds}:{method}:s{seed}:fit"
                out.append(Job(fit, ["fit", *base, "--method", method, "--seed", str(seed)],
                               config.fit_dir(m, ds, method, seed) / "done.json", deps=[pool], priority=40 + k))
                if seed in methods.get(method, []):
                    out.append(Job(f"{m}:{ds}:{method}:s{seed}:eval",
                                   ["eval", *base, "--method", method, "--seed", str(seed)],
                                   config.eval_dir(m, method, ds=ds, seed=seed) / "done.json", deps=[fit],
                                   priority=30))
        if mechanism:
            talk = f"{m}:{ds}:format-talk"
            out.append(Job(talk, ["mechanism", "format-talk", *base],
                           config.mechanism_dir(m, ds) / "format_talk.summary.json", deps=[f"{m}:untrained"],
                           priority=25))
            for seed in mech_seeds:
                out.append(Job(f"{m}:{ds}:probe:s{seed}", ["mechanism", "probe", *base, "--seed", str(seed)],
                               config.mechanism_dir(m, ds) / f"probe_s{seed}" / "done.json",
                               deps=[talk] + [f"{m}:{ds}:{name}:s{seed}:fit" for name in mech["methods"]],
                               priority=45))
    return out


def simple_format_jobs(models: list[str]) -> list[Job]:
    """Sect. 5, "Where FTrim helps less": teachers and FTrim's scoring on the datasets with simple answer formats
    (no fine-tuning here); `python -m ftrim report` gives the share of the reasoning that FTrim would delete."""
    out = [Job("prepare-simple-data", ["prepare-data", "--simple-formats"],
               data.PROCESSED / "simple_formats_summary.json", gpu=False, priority=0)]
    for m in models:
        for ds in config.experiment()["simple_formats"].get(m, []):
            out += _teacher_jobs(m, ds, ["prepare-simple-data"], edit=False)
    return out


# --------------------------------------------------------------------------------------------------------- GPU control
def host_env() -> dict[str, str]:
    path = ROOT / "configs" / "host.local.yaml"
    if not path.exists():
        return {}
    return {k: str(v) for k, v in (yaml.safe_load(path.read_text()) or {}).get("env", {}).items()}


def gpu_memory(env: dict[str, str]) -> dict[int, int]:
    """{GPU index: memory in use (MiB)}."""
    cmd = os.environ.get("FTRIM_NVIDIA_SMI", "nvidia-smi")
    q = subprocess.run([cmd, "--query-gpu=index,memory.used", "--format=csv,noheader,nounits"],
                       capture_output=True, text=True, env={**os.environ, **env}, timeout=60)
    if q.returncode:
        raise RuntimeError(f"nvidia-smi failed: {(q.stderr or q.stdout).strip()}")
    return {int(i): int(used) for i, used in (line.split(",") for line in q.stdout.strip().splitlines())}


# ----------------------------------------------------------------------------------------------------------------- loop
def run(graph: list[Job], gpus: list[int], *, dry_run: bool = False, poll: float = 20.0, max_cpu_jobs: int = 4,
        free_below_mib: int = 1024) -> bool:
    """Run every job that is not done; returns True when all jobs are done."""
    jobs_by_name = {j.name: j for j in graph}
    for j in graph:
        for d in j.deps:
            if d not in jobs_by_name:
                raise ValueError(f"{j.name} depends on the unknown job {d}")
    env = host_env()
    logs = config.RUNS / "logs"
    running: dict[str, tuple[subprocess.Popen, int | None, float]] = {}
    failed: dict[str, int] = {}

    def done(name: str) -> bool:
        return jobs_by_name[name].done.exists()

    def ready(j: Job) -> bool:
        return not done(j.name) and j.name not in running and j.name not in failed and all(done(d) for d in j.deps)

    if dry_run:
        for j in sorted(graph, key=lambda j: (j.priority, j.name)):
            print(("done   " if done(j.name) else "todo   ") + j.name)
        return all(done(j.name) for j in graph)
    logs.mkdir(parents=True, exist_ok=True)
    print(f"[{now()}] pipeline: {sum(not done(j.name) for j in graph)} of {len(graph)} jobs to run on GPUs {gpus}",
          flush=True)
    while True:
        for name, (proc, gpu, started) in list(running.items()):
            code = proc.poll()
            if code is None:
                continue
            del running[name]
            ok = code == 0 and done(name)
            if not ok:
                failed[name] = code
            status = "done " if ok else "FAIL "
            print(f"[{now()}] {status} {name} (gpu {gpu}, {time.time() - started:.0f}s, exit {code})",
                  flush=True)
        todo = [j for j in graph if not done(j.name)]
        if not todo and not running:
            print(f"[{now()}] pipeline: all jobs done", flush=True)
            return True
        if not running and not any(ready(j) for j in todo):
            print(f"[{now()}] pipeline: stopped; failed = {sorted(failed)}; blocked = {len(todo)}", flush=True)
            return False
        busy = {g for _, g, _ in running.values() if g is not None}
        free: list[int] = []
        if any(ready(j) and j.gpu for j in todo):
            try:
                used = gpu_memory(env)
                free = [g for g in gpus if g not in busy and used.get(g, free_below_mib) < free_below_mib]
            except (RuntimeError, OSError, subprocess.SubprocessError) as e:
                print(f"[{now()}] {e}", flush=True)
        cpu_running = sum(g is None for _, g, _ in running.values())
        for j in sorted((j for j in todo if ready(j)), key=lambda j: (j.priority, j.name)):
            if j.gpu and not free:
                continue
            if not j.gpu and cpu_running >= max_cpu_jobs:
                continue
            gpu = free.pop(0) if j.gpu else None
            # the venv's bin directory holds ninja for vLLM's JIT kernels; CUDA numbers the GPUs as nvidia-smi does
            job_env = {**os.environ, **env, "PYTHONUNBUFFERED": "1",
                       "PATH": str(Path(sys.executable).parent) + os.pathsep + os.environ.get("PATH", ""),
                       "CUDA_DEVICE_ORDER": "PCI_BUS_ID", "CUDA_VISIBLE_DEVICES": "" if gpu is None else str(gpu)}
            if gpu is None:
                cpu_running += 1
            with (logs / f"{j.name.replace(':', '_')}.log").open("a", encoding="utf-8") as log:
                log.write(f"\n=== {now()} {' '.join(j.args)} (gpu {gpu})\n")
                log.flush()
                proc = subprocess.Popen([sys.executable, "-m", "ftrim", *j.args], cwd=ROOT, env=job_env, stdout=log,
                                        stderr=subprocess.STDOUT, start_new_session=True)
            running[j.name] = (proc, gpu, time.time())
            print(f"[{now()}] start {j.name} (gpu {gpu})", flush=True)
        (config.RUNS / "pipeline_state.json").write_text(json.dumps(
            {"at": now(), "running": {n: g for n, (_, g, _) in running.items()}, "failed": failed,
             "remaining": len(todo)}, indent=1))
        time.sleep(poll)
