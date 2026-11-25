"""Dataset acquisition and conversion.

Every source is pinned by URL / revision and sha256 and downloaded from its original location; nothing is
redistributed with this repository. Converted examples are JSON lines with the fields
  id, dataset, split, document_id (bootstrap cluster), question, table, paragraphs, gold [, choices, prompt]
under data/processed/<dataset>/<split>.jsonl. The converted files are compared with the checksums of the paper's
runs (configs/data_checksums.json). Nothing here depends on a model.

Splits. TAT-QA: the official train / dev / test. DROP: test = the official dev (the official test is hidden); the
official train passages whose text also appears in the official dev are dropped, the others are ordered by a salted
hash and assigned to a held-out dev part until it holds 1,000 questions, then to train until it holds 10,000. The dev
parts are not used in the paper.
"""
from __future__ import annotations

import json
import zipfile
from collections import Counter
from pathlib import Path
from urllib.request import urlopen

from .config import DATA
from .utils import ROOT, provenance, read_json, read_jsonl, sha256_file, sha256_text, write_json, write_jsonl

RAW = DATA / "raw"
PROCESSED = DATA / "processed"
CHECKSUMS = ROOT / "configs" / "data_checksums.json"

TATQA_REV = "870accc41953dcde885aabeb963d94aabdc0fbc3"
TATQA_FILES = {  # official split -> (path in the repository, sha256)
    "train": ("dataset_raw/tatqa_dataset_train.json", "2df6e722cdbaaa37efcbfb280f5c9a15be29a6ec18f618ef936fe63cc6d07c69"),
    "dev": ("dataset_raw/tatqa_dataset_dev.json", "8da095a819af6db3c14877c6df2d4d29960e41d1a63dd1fa853507bd2a616af5"),
    "test": ("dataset_raw/tatqa_dataset_test_gold.json", "c4d08418359c1d76468dec420ee748a37f48c06b63cb8ec2766f19d5d314b597"),
}
DROP_URL = "https://s3-us-west-2.amazonaws.com/allennlp/datasets/drop/drop_dataset.zip"
DROP_SHA256 = "39d2278a29fd729de301b111a45f434c24834f40df8f4ff116d864589e3249d6"
DROP_SALT = "20261024"          # salt of the passage order that splits the official train
DROP_DEV_MIN, DROP_FIT_MIN = 1000, 10000
FR_REV = "b0fe6455396f831955e4eb988472b4a563403bc5"
FR_FILES = {"easy": "53890ee130f33759cf53cc71943248671e27e6035e75e4234373587784e56dd9",
            "medium": "98098f9b54c0a8c6192b3f3f8b7afe514b0fef2b0972ff0fe63ff43202c20e22",
            "hard": "ca8c055ef104176131da2e0f533897350cde3b93c4754575b09c7c7fbd046d9f"}
HF_FILES = {  # dataset -> (repo, revision, file, sha256)
    "gsm8k": ("openai/gsm8k", "e53f048856ff4f594e959d75785d2c2d37b678ee", "main/test-00000-of-00001.parquet",
              "ee7b8da9e381df27b9e3f7758a159ab2bdaa4dbaa910546cbbc47e0cb44e4f59"),
    "arc_challenge": ("allenai/ai2_arc", "210d026faf9955653af8916fad021475a3f00453", "ARC-Challenge/test-00000-of-00001.parquet",
                      "62f03257e737aed263f55c6abf87c7bb0028a44a6bdd2a26eb1279eb42c1d1e9"),
    "ifeval": ("google/IFEval", "966cd89545d6b6acfd7638bc708b98261ca58e84", "ifeval_input_data.jsonl",
               "6a85310ca8ce15eff755aa08a3a4ff931c7e273e7515ebb3c492ea85fd8288f2"),
}
# Simple answer formats (Sect. 5 of the paper): a single number, expression or option letter
SPLIT_SALT = "20261024"         # salt of the order that takes a dev part out of an official train split
MATH_REPO, MATH_REV = "EleutherAI/hendrycks_math", "21a5633873b6a120296cce3e2df9d5550074f4a3"
MATH_FILES = {  # (subject, official split) -> sha256 of <subject>/<split>-00000-of-00001.parquet
    ("algebra", "train"): "73347f5b55869f21ff094361025dbd12d291e971aa4d72142eacee10438e449f",
    ("algebra", "test"): "0e04a850dac9cfc758eac0c80afd2d0cc65e29ea72f916c8ac1045b12af937f5",
    ("counting_and_probability", "train"): "d37ae017930df41678edee50dcde18c17fb06afaff9eb505c1b95f1002bfcb31",
    ("counting_and_probability", "test"): "c324818949c75a6227c0328a8499d58023cbbf6757cfc1736f5ab24cc8d87966",
    ("geometry", "train"): "9dbf5e5b55499e696e0ba72da9d85d4f0be3acbf7eaa6cb014b0fc78bb63784f",
    ("geometry", "test"): "98720f3a8134ad9784fd8951195245264495df03ffc5c9485a9a54ff5ae2431a",
    ("intermediate_algebra", "train"): "0c87f3798ebdfb915b1f9b6fc13c1a8d064c3cb91631ccbe85b83ed5d71150f6",
    ("intermediate_algebra", "test"): "e859f72486c951c2017da880541ca729940eaa986b0364eb23e238983df9bfeb",
    ("number_theory", "train"): "81c08649149b39811f2d1d7ddf5dfccf6c4c8044a2e169cfb24db09160c941f4",
    ("number_theory", "test"): "ba156c9c9fa2040836ba12135fd964e6e67d6bccd8c469ec4a3ad941f8f6fbe9",
    ("prealgebra", "train"): "765b526fb5c0b8385cb7be2c16bf4f6f28cf71f2bc15c826abfbb5fa7f8616cc",
    ("prealgebra", "test"): "301e5d8ade5c7b5038032532e498bdc9dca799e7f1061a63f2e7d980803240cc",
    ("precalculus", "train"): "28bb1ff95525c8d32521509bef91de76339659f5467a381e6e8deea204538f0e",
    ("precalculus", "test"): "da37474ba3761ae5727af3844a7ba963a735bd6a14c65e6137593669a4fe3b9c",
}
GSM8K_FILES = {"train": ("main/train-00000-of-00001.parquet", "ea82612ea9582142387730c793eb67d3b12849002bc0b7fa6f8efafa7351419d"),
               "test": ("main/test-00000-of-00001.parquet", "ee7b8da9e381df27b9e3f7758a159ab2bdaa4dbaa910546cbbc47e0cb44e4f59")}
MEDCALC_REPO, MEDCALC_REV = "nsk7153/MedCalc-Bench-Verified", "58fc3de568bfc53a68f32327e58f60dc6eaa17f8"
MEDCALC_FILES = {"train": ("data/train-00000-of-00001.parquet", "963b55c822b48ce14a1cb4a06a9633108fbf7b4def8e5d54f027a0f04ecfaf79"),
                 "test": ("data/test-00000-of-00001.parquet", "c28f77c97a2639bad71f2d9c4ef5f3ed41e9863bb7877e307bf7d1bfc180dfd2")}
MEDQA_REPO, MEDQA_REV = "bigbio/med_qa", "484a6c066fe8e75c83edea0c88b5169316714fcd"
MEDQA_FILES = {  # official split -> (file, sha256); the original splits of Jin et al. 2021
    "train": ("med_qa_en_4options_source/train-00000-of-00001.parquet", "b8355af7b58b9de2bf95751b1f7c3ae30be4b4df0c2f7ae0998c0866a7fd5e2b"),
    "validation": ("med_qa_en_4options_source/validation-00000-of-00001.parquet",
                   "6962f5537d8a349244208f02583ee9bc5ef2807e4aee912c33dfbd240feb5620"),
    "test": ("med_qa_en_4options_source/test-00000-of-00001.parquet", "b50b49fad2642c10ed343c6174d5ab0cb8efad434e7b66465ac471094eb57deb"),
}
SIMPLE_FORMAT_DATASETS = ("math", "gsm8k_strict", "medcalc", "medqa")
TRAIN_SPLIT = {"tatqa": "train", "drop": "train",   # the questions that self-training samples solutions for
               **{ds: "train" for ds in SIMPLE_FORMAT_DATASETS}}


def _fetch(url: str, dest: Path, sha256: str | None) -> Path:
    if not dest.exists():
        dest.parent.mkdir(parents=True, exist_ok=True)
        with urlopen(url, timeout=120) as r:
            data = r.read()
        tmp = dest.with_suffix(dest.suffix + ".part")
        tmp.write_bytes(data)
        tmp.replace(dest)
    if sha256 and sha256_file(dest) != sha256:
        raise SystemExit(f"{dest}: sha256 differs from the pinned source")
    return dest


def acquire() -> dict:
    """Download every pinned source into data/raw (skips files already present and verified)."""
    got = {}
    for split, (path, sha) in TATQA_FILES.items():
        url = f"https://raw.githubusercontent.com/NExTplusplus/TAT-QA/{TATQA_REV}/{path}"
        got[f"tatqa/{split}"] = str(_fetch(url, RAW / "tatqa" / Path(path).name, sha))
    _fetch(f"https://raw.githubusercontent.com/NExTplusplus/TAT-QA/{TATQA_REV}/LICENSE", RAW / "tatqa" / "LICENSE", None)
    got["drop"] = str(_fetch(DROP_URL, RAW / "drop" / "drop_dataset.zip", DROP_SHA256))
    for level, sha in FR_FILES.items():
        url = f"https://raw.githubusercontent.com/BUPT-Reasoning-Lab/FinanceReasoning/{FR_REV}/data/FinanceReasoning/{level}.json"
        got[f"finance_reasoning/{level}"] = str(_fetch(url, RAW / "finance_reasoning" / f"{level}.json", sha))
    for name, (repo, rev, fn, sha) in HF_FILES.items():
        got[name] = _hf(repo, rev, fn, sha)
    return got


# ----------------------------------------------------------------------------------------------------------------- TAT-QA

def convert_tatqa(path: Path, split: str) -> list[dict]:
    rows, out = json.loads(path.read_text(encoding="utf-8")), []
    for row in rows:
        table = row["table"]
        cells = table["table"]
        if not isinstance(cells, list) or any(not isinstance(c, str) for r in cells for c in r):
            raise ValueError("TAT-QA table cells must be text")
        paragraphs = [{"source_id": f"paragraph:{p['order']}", "text": p["text"]} for p in row["paragraphs"]]
        for qa in row["questions"]:
            kind = qa["answer_type"]
            answer = qa["answer"]
            if kind in {"span", "multi-span"}:
                if not isinstance(answer, list) or not all(isinstance(a, str) for a in answer):
                    raise ValueError("TAT-QA span answers are lists of text")
            elif kind not in {"arithmetic", "count"}:
                raise ValueError(f"unsupported TAT-QA answer type {kind}")
            out.append({"id": f"tatqa:{qa['uid']}", "dataset": "tatqa", "split": split,
                        "document_id": f"tatqa:{table['uid']}", "question": qa["question"], "table": cells,
                        "paragraphs": paragraphs,
                        "gold": {"answer": answer, "answer_type": kind, "scale": qa.get("scale") or "none",
                                 "derivation": qa.get("derivation", ""), "answer_from": qa.get("answer_from", "")}})
    return out


# ------------------------------------------------------------------------------------------------------------------- DROP

def drop_canonical(answer: dict) -> tuple[str | None, str | None]:
    """(answer type, canonical gold string) of a DROP answer json in the order of the official answer_json_to_strings
    (number, spans, date): a number as released (a leading-dot decimal gets its 0), one span, several spans as a JSON
    array, or the date's non-empty parts as "day month year"; (None, None) when the answer is empty."""
    if answer.get("number"):
        n = str(answer["number"]).strip()
        return "number", ("0" + n if n.startswith(".") else n)
    if answer.get("spans"):
        spans = [str(x) for x in answer["spans"]]
        return ("span", spans[0]) if len(spans) == 1 else ("spans", json.dumps(spans, ensure_ascii=False))
    date = answer.get("date") or {}
    parts = [str(date.get(k, "")).strip() for k in ("day", "month", "year")]
    text = " ".join(p for p in parts if p)
    return ("date", text) if text else (None, None)


def convert_drop(zip_path: Path) -> dict[str, list[dict]]:
    with zipfile.ZipFile(zip_path) as zf:
        data = {sp: json.loads(zf.read(f"drop_dataset/drop_dataset_{sp}.json").decode("utf-8")) for sp in ("train", "dev")}
    norm = lambda text: " ".join(str(text).split())  # noqa: E731
    dev_texts = {norm(a["passage"]) for a in data["dev"].values()}
    shared = {pid for pid, a in data["train"].items() if norm(a["passage"]) in dev_texts}
    order = sorted((pid for pid in data["train"] if pid not in shared),
                   key=lambda pid: sha256_text(DROP_SALT + "drop:train/" + pid))
    part_of, held = {}, {"dev": 0, "train": 0}
    for pid in order:
        part = "dev" if held["dev"] < DROP_DEV_MIN else "train" if held["train"] < DROP_FIT_MIN else None
        if part is None:
            break
        part_of[pid] = part
        held[part] += sum(drop_canonical(qa["answer"])[0] is not None for qa in data["train"][pid]["qa_pairs"])
    out = {"train": [], "dev": [], "test": []}
    for source in ("train", "dev"):
        for pid, a in data[source].items():
            split = "test" if source == "dev" else part_of.get(pid)
            if split is None:
                continue
            for i, qa in enumerate(a["qa_pairs"]):
                kind, canonical = drop_canonical(qa["answer"])
                if kind is None:
                    if source == "dev":
                        raise ValueError(f"official dev question {pid}/{i} has no answer")
                    continue
                answers = [qa["answer"]] + list(qa.get("validated_answers") or [])
                out[split].append({"id": f"drop:{source}/{pid}/{i}", "dataset": "drop", "split": split,
                                   "document_id": f"drop:passage/{pid}", "question": str(qa["question"]).strip(),
                                   "table": [], "paragraphs": [{"source_id": "passage", "text": str(a["passage"]).strip()}],
                                   "gold": {"answer": canonical, "answer_type": kind, "answers": answers,
                                            "query_id": qa["query_id"]}})
    return out


# ---------------------------------------------------------------------------------------------------- unseen benchmarks

def convert_finance_reasoning(raw_dir: Path) -> list[dict]:
    out = []
    for level in ("easy", "medium", "hard"):
        for row in json.loads((raw_dir / f"{level}.json").read_text(encoding="utf-8")):
            answer = row["ground_truth"]
            if not isinstance(answer, (int, float, bool)):
                raise ValueError("FinanceReasoning gold must be numeric or boolean")
            out.append({"id": f"finance_reasoning:{level}:{row['question_id']}", "dataset": "finance_reasoning",
                        "split": "test", "document_id": f"finance_reasoning:source:{row['source']}",
                        "question": row["question"], "table": [],
                        "paragraphs": [{"source_id": "context", "text": row["context"]}],
                        "gold": {"answer": answer, "answer_type": "boolean" if isinstance(answer, bool) else "arithmetic",
                                 "level": level, "source": row["source"]}})
    return out


def convert_gsm8k(path: str) -> list[dict]:
    import pandas as pd
    out = []
    for i, r in pd.read_parquet(path).iterrows():
        gold = float(str(r["answer"]).split("####")[-1].strip().replace(",", ""))
        out.append({"id": f"gsm8k:{i}", "dataset": "gsm8k", "split": "test", "document_id": f"gsm8k:{i}",
                    "question": r["question"], "gold": {"answer": gold}})
    return out


def convert_arc(path: str) -> list[dict]:
    import pandas as pd
    out = []
    for _, r in pd.read_parquet(path).iterrows():
        choices = [{"label": str(lab), "text": str(txt)} for lab, txt in zip(r["choices"]["label"], r["choices"]["text"])]
        out.append({"id": f"arc:{r['id']}", "dataset": "arc_challenge", "split": "test", "document_id": f"arc:{r['id']}",
                    "question": r["question"], "choices": choices, "gold": {"answer": str(r["answerKey"])}})
    return out


def convert_ifeval(path: str) -> list[dict]:
    out = []
    for r in read_jsonl(path):
        out.append({"id": f"ifeval:{r['key']}", "dataset": "ifeval", "split": "test", "document_id": f"ifeval:{r['key']}",
                    "prompt": r["prompt"], "question": r["prompt"],
                    "gold": {"key": r["key"], "instruction_id_list": r["instruction_id_list"], "kwargs": r["kwargs"]}})
    return out


def _verify(summary: dict) -> list[str]:
    """Compare the converted files with the checksums of the paper's runs; returns the files that differ."""
    expected = read_json(CHECKSUMS)
    differ = []
    for name, splits in summary["datasets"].items():
        for split, info in splits.items():
            if not isinstance(info, dict) or "sha256" not in info:
                continue
            info["matches_paper"] = info["sha256"] == expected.get(f"{name}/{split}", {}).get("sha256")
            if not info["matches_paper"]:
                differ.append(f"{name}/{split}")
    if differ:
        print("WARNING: these converted files differ from the paper's (configs/data_checksums.json): "
              + ", ".join(differ), flush=True)
    return differ


def prepare() -> dict:
    """Acquire and convert the target tasks and the unseen benchmarks; the summary is also written to
    data/processed/summary.json."""
    got = acquire()
    summary = {"datasets": {}}

    def save(name, split, rows):
        write_jsonl(PROCESSED / name / f"{split}.jsonl", rows)
        summary["datasets"].setdefault(name, {})[split] = {
            "questions": len(rows), "documents": len({r["document_id"] for r in rows}),
            "answer_types": dict(Counter(r["gold"].get("answer_type", "-") for r in rows))}

    for split in ("train", "dev", "test"):
        save("tatqa", split, convert_tatqa(RAW / "tatqa" / Path(TATQA_FILES[split][0]).name, split))
    for split, rows in convert_drop(RAW / "drop" / "drop_dataset.zip").items():
        save("drop", split, rows)
    save("finance_reasoning", "test", convert_finance_reasoning(RAW / "finance_reasoning"))
    save("gsm8k", "test", convert_gsm8k(got["gsm8k"]))
    save("arc_challenge", "test", convert_arc(got["arc_challenge"]))
    save("ifeval", "test", convert_ifeval(got["ifeval"]))
    for name in summary["datasets"]:
        for split in summary["datasets"][name]:
            summary["datasets"][name][split]["sha256"] = sha256_file(PROCESSED / name / f"{split}.jsonl")
    summary["differ_from_paper"] = _verify(summary)
    write_json(PROCESSED / "summary.json", {**summary, **provenance()})
    return summary


# ------------------------------------------------------------------------------------------------ simple answer formats

def _hf(repo: str, revision: str, filename: str, sha256: str) -> str:
    from huggingface_hub import hf_hub_download
    path = hf_hub_download(repo_id=repo, repo_type="dataset", revision=revision, filename=filename)
    if sha256_file(path) != sha256:
        raise SystemExit(f"{repo}/{filename}: sha256 differs from the pinned source")
    return path


def split_order(key: str) -> str:
    """The order that takes a dev part out of an official train split: sha256 of the salt and the key."""
    return sha256_text(SPLIT_SALT + key)


def remove_boxed(s: str | None) -> str | None:
    """`remove_boxed` of hendrycks/math (modeling/evaluate_gpt3.py): the content of "\\boxed{...}", else None."""
    left = "\\boxed{"
    if s is None or s[:len(left)] != left or s[-1:] != "}":
        return None
    return s[len(left):-1]


def _first_train_to_dev(rows: list[dict], key, size: int) -> None:
    """Official train rows in the split order: the first `size` go to dev, the rest stay train."""
    train = sorted((r for r in rows if r["split"] == "train"), key=lambda r: split_order(key(r)))
    for r in train[:size]:
        r["split"] = "dev"


def convert_math(files: dict[tuple[str, str], str]) -> tuple[list[dict], dict]:
    import pandas as pd

    from .vendor.math_util import last_boxed_only_string
    rows, dropped = [], Counter()
    for (subject, source), path in sorted(files.items()):
        for i, r in pd.read_parquet(path).iterrows():
            gold = remove_boxed(last_boxed_only_string(r["solution"]))
            if gold is None:
                dropped[f"{source}/{subject}"] += 1
                continue
            qid = f"math:{source}/{subject}/{i}"
            rows.append({"id": qid, "dataset": "math", "split": "test" if source == "test" else "train",
                         "document_id": qid, "question": r["problem"], "table": [], "paragraphs": [],
                         "gold": {"answer": gold, "answer_type": "expression", "subject": subject, "level": r["level"]}})
    _first_train_to_dev(rows, lambda r: r["id"], 500)
    return rows, {"dropped_without_boxed": dict(dropped)}


def convert_gsm8k_strict(files: dict[str, str]) -> tuple[list[dict], dict]:
    import pandas as pd
    rows = []
    for source in ("train", "test"):
        for i, r in pd.read_parquet(files[source]).iterrows():
            qid = f"gsm8k_strict:{source}/{i}"
            rows.append({"id": qid, "dataset": "gsm8k_strict", "split": source, "document_id": qid,
                         "question": r["question"], "table": [], "paragraphs": [],
                         "gold": {"answer": str(r["answer"]).split("####")[-1].strip().replace(",", ""),
                                  "answer_type": "integer"}})
    _first_train_to_dev(rows, lambda r: r["id"].replace("gsm8k_strict:", "gsm8k:", 1), 500)   # ordered by the short id
    return rows, {}


def convert_medcalc(files: dict[str, str]) -> tuple[list[dict], dict]:
    from collections import defaultdict

    import pandas as pd
    frames = {source: pd.read_parquet(files[source]) for source in ("train", "test")}
    group = lambda note: sha256_text(" ".join(str(note).split()))   # noqa: E731 - one patient note = one cluster
    groups = defaultdict(list)
    for i, r in frames["train"].iterrows():
        groups[group(r["Patient Note"])].append(i)
    dev: set = set()
    for g in sorted(groups, key=split_order):
        if len(dev) >= 550:
            break
        dev.update(groups[g])
    rows = []
    for source, df in frames.items():
        for i, r in df.iterrows():
            split = "test" if source == "test" else "dev" if i in dev else "train"
            rows.append({"id": f"medcalc:{source}/{int(r['Row Number'])}", "dataset": "medcalc", "split": split,
                         "document_id": "medcalc:note/" + group(r["Patient Note"])[:16], "question": str(r["Question"]),
                         "table": [], "paragraphs": [{"source_id": "note", "text": str(r["Patient Note"])}],
                         "gold": {"answer": str(r["Ground Truth Answer"]), "answer_type": str(r["Output Type"]),
                                  "lower_limit": str(r["Lower Limit"]), "upper_limit": str(r["Upper Limit"]),
                                  "calculator_id": int(r["Calculator ID"]), "calculator_name": str(r["Calculator Name"]),
                                  "category": str(r["Category"])}})
    test_groups = {group(n) for n in frames["test"]["Patient Note"]}
    return rows, {"test_notes_also_in_official_train": len(test_groups & set(groups))}


def convert_medqa(files: dict[str, str]) -> tuple[list[dict], dict]:
    import pandas as pd
    rows = []
    for source, split in (("train", "train"), ("validation", "dev"), ("test", "test")):
        for i, r in pd.read_parquet(files[source]).iterrows():
            qid = f"medqa:{source}/{i}"
            options = list(r["options"])
            keys = [o["key"] for o in options]
            if keys != ["A", "B", "C", "D"] or r["answer_idx"] not in keys:
                raise ValueError(f"{qid}: unexpected options {keys} / {r['answer_idx']}")
            text = "\n".join(f"{o['key']}. {o['value']}" for o in options)
            rows.append({"id": qid, "dataset": "medqa", "split": split, "document_id": qid, "question": str(r["question"]),
                         "table": [], "paragraphs": [{"source_id": "options", "text": text}],
                         "gold": {"answer": str(r["answer_idx"]), "answer_type": "option", "answer_text": str(r["answer"])}})
    return rows, {}


def prepare_simple_formats() -> dict:
    """Acquire and convert the four datasets with simple answer formats; the summary is also written to
    data/processed/simple_formats_summary.json."""
    math_files = {key: _hf(MATH_REPO, MATH_REV, f"{key[0]}/{key[1]}-00000-of-00001.parquet", sha)
                  for key, sha in MATH_FILES.items()}
    gsm_repo, gsm_rev = HF_FILES["gsm8k"][:2]
    converted = {
        "math": convert_math(math_files),
        "gsm8k_strict": convert_gsm8k_strict({s: _hf(gsm_repo, gsm_rev, f, sha) for s, (f, sha) in GSM8K_FILES.items()}),
        "medcalc": convert_medcalc({s: _hf(MEDCALC_REPO, MEDCALC_REV, f, sha) for s, (f, sha) in MEDCALC_FILES.items()}),
        "medqa": convert_medqa({s: _hf(MEDQA_REPO, MEDQA_REV, f, sha) for s, (f, sha) in MEDQA_FILES.items()}),
    }
    summary = {"sources": {"math": f"{MATH_REPO}@{MATH_REV}", "gsm8k_strict": f"{gsm_repo}@{gsm_rev}",
                           "medcalc": f"{MEDCALC_REPO}@{MEDCALC_REV}", "medqa": f"{MEDQA_REPO}@{MEDQA_REV}"}, "datasets": {}}
    for name, (rows, notes) in converted.items():
        if len({r["id"] for r in rows}) != len(rows):
            raise ValueError(f"{name}: question ids are not unique")
        info = summary["datasets"].setdefault(name, {"notes": notes})
        for split in ("train", "dev", "test"):
            part = [r for r in rows if r["split"] == split]
            path = PROCESSED / name / f"{split}.jsonl"
            write_jsonl(path, part)
            info[split] = {"questions": len(part), "documents": len({r["document_id"] for r in part}),
                           "answer_types": dict(Counter(r["gold"]["answer_type"] for r in part)), "sha256": sha256_file(path)}
    summary["differ_from_paper"] = _verify(summary)
    write_json(PROCESSED / "simple_formats_summary.json", {**summary, **provenance()})
    return summary


def load(dataset: str, split: str) -> list[dict]:
    path = PROCESSED / dataset / f"{split}.jsonl"
    if not path.exists():
        raise SystemExit(f"{path} is missing; run `python -m ftrim prepare-data` first")
    return read_jsonl(path)


def load_index(dataset: str, split: str) -> dict[str, dict]:
    return {r["id"]: r for r in load(dataset, split)}
