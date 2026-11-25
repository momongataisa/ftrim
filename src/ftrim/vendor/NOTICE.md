# Vendored third-party evaluators

The official evaluators of the datasets are vendored so that the scores do not depend on a package version.

| File | Source | License | Change |
|---|---|---|---|
| `tatqa_metric.py`, `tatqa_utils.py` | https://github.com/NExTplusplus/TAT-QA @ `870accc41953dcde885aabeb963d94aabdc0fbc3` | MIT (`LICENSE.tatqa`) | `from tatqa_utils import *` → package-relative import |
| `drop_eval.py` | https://github.com/allenai/allennlp-models @ `02480d86450dff1dfc2003f4ebd93d0347532a20` (v2.10.1), `allennlp_models/rc/tools/drop.py` (the official DROP evaluator) | Apache-2.0 (`LICENSE.allennlp-models`) | none; renamed |
| `math_equivalence.py`, `math_util.py` | https://github.com/hendrycks/math @ `985bdc1696e88e8643f081a0ff4719da39f2ae2a`, `modeling/math_equivalence.py` and `modeling/dataset/util.py` (MATH's answer equivalence and `\boxed{}` extraction) | MIT (`LICENSE.hendrycks_math`) | `util.py` renamed to `math_util.py` |

`python scripts/check_vendor.py` downloads the originals again and prints every difference.
