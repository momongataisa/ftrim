"""Download the vendored official evaluators again and show that they differ from the originals only as
src/ftrim/vendor/NOTICE.md says (an import line).

    python scripts/check_vendor.py
"""
import difflib
import hashlib
import sys
from pathlib import Path
from urllib.request import urlopen

VENDOR = Path(__file__).resolve().parents[1] / "src" / "ftrim" / "vendor"
TATQA = "https://raw.githubusercontent.com/NExTplusplus/TAT-QA/870accc41953dcde885aabeb963d94aabdc0fbc3/"
DROP = "https://raw.githubusercontent.com/allenai/allennlp-models/02480d86450dff1dfc2003f4ebd93d0347532a20/"
MATH = "https://raw.githubusercontent.com/hendrycks/math/985bdc1696e88e8643f081a0ff4719da39f2ae2a/"
SOURCES = {
    "tatqa_metric.py": TATQA + "tatqa_metric.py",
    "tatqa_utils.py": TATQA + "tatqa_utils.py",
    "drop_eval.py": DROP + "allennlp_models/rc/tools/drop.py",
    "math_equivalence.py": MATH + "modeling/math_equivalence.py",
    "math_util.py": MATH + "modeling/dataset/util.py",
}


def main() -> int:
    ok = True
    for name, url in SOURCES.items():
        original = urlopen(url, timeout=60).read().decode("utf-8")
        local = (VENDOR / name).read_text(encoding="utf-8")
        print(f"{name}: sha256 of the original {hashlib.sha256(original.encode()).hexdigest()[:12]}")
        changed = [line for line in difflib.unified_diff(original.splitlines(), local.splitlines(), lineterm="", n=0)
                   if line[:1] in "+-" and not line.startswith(("+++", "---"))]
        for line in changed:
            print("   ", line)
        ok &= all("import" in line for line in changed)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
