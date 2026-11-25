# Data and models

`python -m ftrim prepare-data` downloads every dataset from its source at a pinned revision, checks the SHA-256 of
each file and compares the converted files with the checksums of the paper's runs (`configs/data_checksums.json`).
The pipeline samples and fine-tunes on the `train` split of a target task and evaluates on the `test` split of every
benchmark; it does not use the `dev` splits.

## Target tasks (strict answer formats)

| Dataset | Source | Splits (questions / documents) | License |
|:--|:--|:--|:--|
| TAT-QA (Zhu et al., 2021) | [NExTplusplus/TAT-QA](https://github.com/NExTplusplus/TAT-QA) @ `870accc4`, `dataset_raw/tatqa_dataset_{train,dev,test_gold}.json` | train 13,215 / 2,201 · dev 1,668 / 278 · test 1,663 / 277 (official splits) | data CC BY 4.0; code and evaluator MIT |
| DROP (Dua et al., 2019) | [`drop_dataset.zip`](https://s3-us-west-2.amazonaws.com/allennlp/datasets/drop/drop_dataset.zip) of AllenNLP, sha256 `39d2278a…` | train 10,006 / 722 · dev 1,017 / 74 · test 9,536 / 582 | CC BY-SA 4.0 |

A *document* is a hybrid context (table and paragraphs) of TAT-QA or a passage of DROP. The bootstrap of the paper
resamples whole documents.

**DROP splits.** The official test answers are hidden, so the test split here is the official development set. The
official training passages whose text also appears in the development set are dropped. The other training passages
are ordered by the SHA-256 of a fixed salt and the passage id; passages are taken in this order for `dev` until it
has 1,000 answerable questions, then for `train` until it has 10,000. No passage is in two splits.

## Unseen benchmarks (evaluation only)

| Benchmark | Source | Test questions | Scoring | License |
|:--|:--|--:|:--|:--|
| FinanceReasoning (Tang et al., 2025) | [BUPT-Reasoning-Lab/FinanceReasoning](https://github.com/BUPT-Reasoning-Lab/FinanceReasoning) @ `b0fe6455`, `data/FinanceReasoning/{easy,medium,hard}.json` | 2,238 | answer and unit tags as on TAT-QA; a number within 0.2 % of the reference (the benchmark's tolerance), or yes/no | CC BY 4.0 on the dataset card of the Hugging Face copy; the GitHub repository has no license file |
| GSM8K (Cobbe et al., 2021) | [openai/gsm8k](https://huggingface.co/datasets/openai/gsm8k) @ `e53f0488`, `main` test | 1,319 | the number in the answer tag, else the last number of the output | MIT |
| IFEval (Zhou et al., 2023) | [google/IFEval](https://huggingface.co/datasets/google/IFEval) @ `966cd895` | 541 | prompt-level strict accuracy, with the checker of lm-evaluation-harness 0.4.12 | Apache-2.0 |
| ARC-Challenge (Clark et al., 2018) | [allenai/ai2_arc](https://huggingface.co/datasets/allenai/ai2_arc) @ `210d026f`, `ARC-Challenge` test | 1,172 | the option in the answer tag, else the last option named in the output | CC BY-SA 4.0 |

FinanceReasoning uses the answer-and-unit format instructions of TAT-QA without the TAT-QA note; GSM8K and
ARC-Challenge ask for the answer in `<answer></answer>`; IFEval uses its own prompts (`src/ftrim/prompts.py`). The
benchmarks are never used for training or for any choice. The *generalization accuracy* of the paper is the mean
accuracy on these four benchmarks.

## Simple answer formats (Sect. 5, teachers only)

`python -m ftrim prepare-data --simple-formats`

| Dataset | Source | Splits (train / dev / test) | License |
|:--|:--|:--|:--|
| MATH (Hendrycks et al., 2021) | [EleutherAI/hendrycks_math](https://huggingface.co/datasets/EleutherAI/hendrycks_math) @ `21a56338` | 6,998 / 500 / 5,000 | MIT |
| GSM8K (Cobbe et al., 2021) | [openai/gsm8k](https://huggingface.co/datasets/openai/gsm8k) @ `e53f0488` | 6,973 / 500 / 1,319 | MIT |
| MedCalc-Bench Verified (Khandekar et al., 2024) | [nsk7153/MedCalc-Bench-Verified](https://huggingface.co/datasets/nsk7153/MedCalc-Bench-Verified) @ `58fc3de5` | 9,988 / 550 / 1,100 | CC BY-SA 4.0 |
| MedQA, four options (Jin et al., 2021) | [bigbio/med_qa](https://huggingface.co/datasets/bigbio/med_qa) @ `484a6c06`, `med_qa_en_4options_source` | 10,178 / 1,272 / 1,273 (official splits) | MIT (the original repository [jind11/MedQA](https://github.com/jind11/MedQA); the dataset card says unknown) |

For MATH and GSM8K the `dev` part is taken out of the official training split in a fixed order (SHA-256 of a salt
and the question id). For MedCalc-Bench whole patient notes are taken in such an order, so that no note is in both
parts. The test splits are the official ones.

## Evaluators

The *target accuracy* of the paper is the *strict* exact match (EM) on TAT-QA and DROP. It uses the official
evaluators, vendored in `src/ftrim/vendor/` (sources, commits and licenses in `src/ftrim/vendor/NOTICE.md`). An
output counts only if it ends with exactly one answer tag that follows the format; otherwise it is wrong. The check
of the format differs between the two tasks (`src/ftrim/scoring.py`):

* **TAT-QA**: the full grammar of the format instructions. The output ends with
  `<answer>value</answer><unit>unit</unit>`; the value is a plain number, lowercase yes or no, a quoted span or a
  JSON array, and the unit is one of the 24 allowed words. The unit gives the scale for the official evaluator.
  TAT-QA has five scales: none, thousand, million, billion and percent. `none` and the currency, count and time
  labels count as no scale. The other five allowed units (`ratio`, `percentage_point`, `basis_point`, `hundred`,
  `trillion`) have no TAT-QA scale, and an answer with one of them counts as wrong, in the teachers and in the
  evaluation. [The results](../results/RESULTS.md#scoring-of-the-tat-qa-units) also score the test outputs with
  two of these units mapped.
* **DROP**: the structure only. The output ends with exactly one `<answer>...</answer>` with non-empty content. The
  official evaluator then scores the content; the rules about its form, such as digits only or dates as day month
  year, are not checked separately.

So "follows the format" checks more rules on TAT-QA than on DROP. The untrained models follow the format in 86–88 %
of their outputs on TAT-QA and in 98.6–99.9 % on DROP.

*Last-tag* EM (key `lenient` in the result files) is strict EM of the output cut at its last answer tag: everything
before that tag, for example an earlier answer tag, is ignored. The last tag must still follow the format and end
the output. Sect. 4 of the paper reports it as "if we score only the last answer tag".

## The example question

`examples/tatqa_example.json` and the figure `assets/overview.png` (Fig. 1 of the paper) contain one training
question of TAT-QA (Zhu et al., 2021; uid `c3f0531e-c9a7-4e52-ad37-02e9d1c61864`) with its table and paragraphs,
under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Changes: the question is written into our prompt,
and the solution that follows it was sampled by Gemma-4-E4B-it; its answer line is the gold answer in the form of the
format instructions.

## Models

| Key | Hugging Face id | Revision | License |
|:--|:--|:--|:--|
| `qwen` | [Qwen/Qwen3.5-4B](https://huggingface.co/Qwen/Qwen3.5-4B) | `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a` | Apache-2.0 |
| `gemma` | [google/gemma-4-E4B-it](https://huggingface.co/google/gemma-4-E4B-it) | `ee0ef6023621cff504d758262d4e04895a5af4a2` | Apache-2.0 |

The licenses are those of the model cards at the pinned revisions. LoRA adapts the attention and MLP projections
named in `configs/experiment.yaml`. In Qwen3.5-4B these are the MLP of all 32 layers and the attention of its 8
full-attention layers; its 24 linear-attention layers have other projection names and are not adapted.
