<div align="center">

 # Multiple LLM Agents Debate for Equitable <br> Cultural Alignment

<p align="center">
  <img src="https://github.com/user-attachments/assets/74b6e33b-2eea-4aed-8db5-d6f89945c1b6" width="800">
</p>


<a href=https://dayeonki.github.io/>Dayeon Ki</a>, <a href=https://rudinger.github.io/>Rachel Rudinger</a>, <a href=https://tianyizhou.github.io/>Tianyi Zhou</a>, <a href=https://www.cs.umd.edu/~marine/>Marine Carpuat<a> <br>
University of Maryland
<br>

This repository contains the code and dataset for our ACL 2025 Main paper <br> **Multiple LLM Agents Debate for Equitable Cultural Alignment**.

<p>
  <a href="https://aclanthology.org/2025.acl-long.1210/" target="_blank" style="text-decoration:none">
    <img src="https://img.shields.io/badge/arXiv-Paper-b31b1b?style=flat&logo=arxiv" alt="arXiv">
  </a>
</p>

</div>

---

## 👾 TL;DR
While previous efforts in cultural alignment have focused on single-model, single-turn approaches, we propose to exploit the complementary strengths of multiple LLMs to promote cultural adaptability. We introduce a Multi-Agent Debate framework, where two LLM-based agents debate over a cultural scenario and collaboratively reach a final decision, which improves both (i) overall accuracy and (ii) cultural group parity over single-model baselines.

## 📰 News
- **`2025-07-10`** Our paper has been selected for an **oral** presentation — top 8% of accepted papers!
- **`2025-05-15`** Our paper is accepted to **ACL 2025**! See you in Vienna!

## 👥 Team (This Project Fork)
- Sushil Bohara
- Leo Ding


## ✏️ Content
- [🗺️ Overview](#overview)
- [🚀 Quick Start](#quick_start)
  - [Data Preparation](#data-preparation)
  - [Single LLM](#single-llm)
  - [Multiple LLM](#multiple-llm)
  - [Evaluation](#evaluation)
- [🤲 Citation](#citation)
- [📧 Contact](#contact)

---

<a id="overview"></a>
## 🗺️ Overview

How can multiple LLMs collaborate toward equitable alignment across cultures? We investigate a common form of multi-LLM collaboration: _debate_. We propose a **Multi-Agent Debate framework**, where two LLM agents debate over the given scenario and collaboratively arrive at a final decision with a judge LLM. We introduce two key variants as illustrated in the above figure:
1. **Debate-Only:** multiple LLM agents exclusively engage in debate with a discussant
2. **Self-Reflect+Debate:** each LLM agent dynamically choose between self-reflection and debating during its turn

For more comprehensive comparison study, we investigate two additional strategies based on single-LLM:

3. **Single Model:** a single LLM generates outputs
4. **Self-Reflection:** an LLM generates verbal self-reflections on its own outputs and incorporate them in subsequent iterations

### Results

<div align="center">
<img width="900" height="279" alt="Screenshot 2026-04-03 at 1 44 40 PM" src="https://github.com/user-attachments/assets/602d2b6b-59db-4715-aa8f-2895311f6f9c" />
</div>


<a id="quick_start"></a>
## 🚀 Quick Start

### Data Preparation

<div align="center">
<img width="850" height="489" alt="Screenshot 2026-04-03 at 1 20 57 PM" src="https://github.com/user-attachments/assets/4ece461f-3875-42ac-9571-cdb6eb3882df" />
</div>


We use [NORMAD-ETI](https://github.com/Akhila-Yerukola/NormAd) dataset for evaluation, a benchmark designed to assess the cultural adaptability of LLMs. The dataset contains 2.6K stories reflecting social and cultural norms from 75 countries, derived from the social-etiquette norms outlined in the [Cultural Atlas](https://culturalatlas.sbs.com.au/). Each story is associated with a country, a rule-of-thumb, and a ternary ground truth label in {Yes, No, Neither} as shown in the figure above. We categorize a total of 75 countries according to the Inglehart-Welzel cultural map and show the label and country distribution for each bin.

- Raw data: `data/normad_raw.csv`
- Country distribution: `data/normad_country_dist.csv`
- Refined data: `data/normad.jsonl`


### Single LLM
#### (1) Single Model
We first investigate the effect of adding relevant cultural context in enhancing cultural alignment of LLMs. We test two variants: without and with the rule-of-thumb (RoT) information in the prompts. (`single_llm/single_model/`)

For running **without** RoT prompting,

```bash
python -u sinlge_llm/single_model/{$LLM}.py \
  --input_path $PATH_TO_INPUT_FILE \
  --output_path $PATH_TO_OUTPUT_FILE \
  --type without_rot
```

For running **with** RoT prompting,

```bash
python -u sinlge_llm/single_model/{$LLM}.py \
  --input_path $PATH_TO_INPUT_FILE \
  --output_path $PATH_TO_OUTPUT_FILE \
  --type with_rot
```

Arguments for the prompting code are as follows:
  - `$LLM`: Name of the LLM (specific names can be found in the directory).
  - `--input_path`: Path to input data file (`data/normad.jsonl`).
  - `--output_path`: Save path of output file.
  - `--type`: Without or with RoT information.


#### (2) Self-Reflection
Building on previous works that showed that LLMs can evaluate their outputs and learn from their own feedback, we explore self-reflection for each LLM. (`single_llm/self_reflection/`)

```bash
python -u sinlge_llm/self_reflection/{$LLM}.py \
  --input_path $PATH_TO_INPUT_FILE \
  --output_path $PATH_TO_OUTPUT_FILE \
```

Arguments for the prompting code are as follows:
  - `$LLM`: Name of the LLM (specific names can be found in the directory).
  - `--input_path`: Path to input data file (`data/normad.jsonl`).
  - `--output_path`: Save path of output file.



### Multiple LLM
LLMs often exhibit varying knowledge coverage, with the potential to complement each other due to differences in training data distributions and alignment processes. We tap into this _knowledge complementarity_ through multi-LLM collaboration, debate, where two LLM-based agents debate and collaboratively evaluate the given scenario.

```bash
python -u multi_llm/{$FIRST_LLM}_$SECOND_LLM.py \
  --input_path $PATH_TO_INPUT_FILE \
  --output_path $PATH_TO_OUTPUT_FILE \
```

Arguments for the prompting code are as follows:
  - `$FIRST_LLM`: Name of the first participant LLM (specific names can be found in the directory).
  - `$SECOND_LLM`: Name of the second participant LLM (specific names can be found in the directory).
  - `--input_path`: Path to input data file (`data/normad.jsonl`).
  - `--output_path`: Save path of output file.


### Evaluation

- For evaluating single LLM baselines, use `evaluate/accuracy_single.py`. Add the model names to test in the `MODEL_NAMES` variable and run the code: `python evaluate/accuracy_single.py`.

- For evaluating multi LLM baselines, use `evaluate/accuracy_multi.py`. Add the name of the first model as `FIRST_MODEL` and the name of the second model as `SECOND_MODEL` variables and run the code: `python evaluate/accuracy_multi.py`.


### Scripted Runs (Recommended)

To make experiments easier to reproduce, this repo now includes reusable runners in `scripts/`:

- `scripts/run_single.py`: run Single Model or Self-Reflection
- `scripts/run_multi.py`: run Multi-LLM debate
- `scripts/evaluate_outputs.py`: evaluate any generated `.jsonl` output file

#### 1) Single Model

```bash
python scripts/run_single.py \
  --variant single_model \
  --model gemma \
  --prompt_type without_rot \
  --input_path data/normad.jsonl \
  --run_eval
```

For RoT prompting, change `--prompt_type with_rot`.

#### 2) Self-Reflection

```bash
python scripts/run_single.py \
  --variant self_reflection \
  --model gemma \
  --input_path data/normad.jsonl \
  --run_eval
```

#### 3) Multi-LLM Debate

```bash
python scripts/run_multi.py \
  --pair gemma_aya \
  --input_path data/normad.jsonl \
  --run_eval
```

#### 4) Evaluate Existing Output Files

Single-output evaluation:

```bash
python scripts/evaluate_outputs.py \
  --mode single \
  --input_file results/single/single_model/gemma_without_rot.jsonl \
  --model gemma
```

Multi-output evaluation:

```bash
python scripts/evaluate_outputs.py \
  --mode multi \
  --input_file results/multi/gemma_aya.jsonl \
  --first_model gemma \
  --second_model aya
```

### Novel Cultural Benchmarking

We provide two additional scripts for novelty-focused analysis:

- `scripts/benchmark_cultural.py`: computes fairness-aware cultural metrics on one output file
- `scripts/analyze_interventions.py`: computes Debate Parity Gain (DPG), Partner Stability Index (PSI), and partner selection by parity-adjusted score
- `scripts/analyze_drift.py`: computes turn-to-turn drift, neutral collapse, and country-level drift from debate outputs

#### Final Run Configuration (Current)

The final full-dataset run configuration uses:

- `Qwen/Qwen2.5-3B-Instruct` (alias `qwen3`)
- `Qwen/Qwen2.5-1.5B-Instruct` (alias `qwen15`)

We did run initial mini-batch tests with smaller open models to verify pipeline stability and throughput before selecting this final 3B setup.

#### 1) Benchmark One Output File (CPG / PAA)

```bash
python scripts/benchmark_cultural.py \
  --input_file results/multi/gemma_aya.jsonl \
  --mode multi \
  --model gemma \
  --parity_lambda 0.25
```

This writes:

- `*_benchmark.json` summary
- `*_country_metrics.csv` per-country performance

#### 2) Run Intervention Analysis Across Existing Result Folders

```bash
python scripts/analyze_interventions.py \
  --single_dir results/single/single_model \
  --multi_dir results/multi \
  --output_dir results/analysis \
  --parity_lambda 0.25
```

This writes:

- `results/analysis/debate_parity_gain.csv`
- `results/analysis/partner_stability_index.csv`
- `results/analysis/partner_selection_by_paa.csv`
- `results/analysis/intervention_summary.json`

#### 3) Analyze Debate Drift

```bash
python scripts/analyze_drift.py \
  --input_file results/multi/open_qwen3_qwen15_full.jsonl \
  --output_dir results/analysis_drift_qwen3 \
  --models qwen3 qwen15
```

This writes:

- `results/analysis_drift_qwen3/drift_summary.json`
- `results/analysis_drift_qwen3/drift_model_metrics.csv`
- `results/analysis_drift_qwen3/drift_country_metrics.csv`

Notes:

- For `benchmark_cultural.py`, use `--prediction_field` if your file uses a non-standard key.
- `parity_lambda` controls how strongly parity gap is penalized in `PAA = accuracy - lambda * CPG`.

#### 4) Significance Report (Paired Bootstrap + Permutation)

```bash
python scripts/significance_report.py \
  --input_file results/multi/open_qwen3_qwen15_full.jsonl \
  --first_model qwen3 \
  --second_model qwen15 \
  --parity_lambda 0.25 \
  --bootstrap_samples 1500 \
  --permutation_samples 2000 \
  --output_json results/analysis_qwen3/significance_report.json
```

This writes:

- `results/analysis_qwen3/significance_report.json`

#### 5) Output Integrity Validation (Duplicates / Coverage)

```bash
python scripts/validate_results_integrity.py \
  --input_file results/multi/open_qwen3_qwen15_full.jsonl \
  --reference_file data/normad.jsonl \
  --expected_rows 2633 \
  --output_json results/analysis_integrity_qwen3/integrity_report.json \
  --duplicates_csv results/analysis_integrity_qwen3/duplicate_rows.csv
```

This writes:

- `results/analysis_integrity_qwen3/integrity_report.json`
- `results/analysis_integrity_qwen3/duplicate_rows.csv` (only when duplicates exist)

#### 6) Judge Sensitivity Check

```bash
python scripts/judge_sensitivity_check.py \
  --input_file results/multi/open_qwen3_qwen15_full.jsonl \
  --model qwen3 \
  --output_json results/analysis_sensitivity_qwen3/qwen3_judge_sensitivity.json \
  --output_csv results/analysis_sensitivity_qwen3/qwen3_judge_disagreements.csv
```

Run again with `--model qwen15` for the second debater.

By default this compares a lenient label parser vs a stricter parser on the same raw generations. You can also provide `--alternate_file` and `--alternate_field` to compare against an externally re-judged output file.

### Crash Recovery Safeguard (Auto-Resume)

For long runs, use the resilient supervisor so generation restarts automatically after crashes or stalls and continues from the last completed row.

```bash
python scripts/resilient_resume_runner.py \
  --input_path data/normad.jsonl \
  --output_path results/multi/open_qwen3_qwen15_full.jsonl \
  --expected_rows 2633 \
  --model_a Qwen/Qwen2.5-3B-Instruct \
  --model_b Qwen/Qwen2.5-1.5B-Instruct \
  --alias_a qwen3 \
  --alias_b qwen15 \
  --max_new_tokens 48 \
  --poll_seconds 90 \
  --stall_seconds 1800 \
  --log_file results/logs/qwen3_resilient_runner.log
```

What this does:
- starts `scripts/run_multi_open_debate.py` with `--resume`,
- watches output row growth,
- restarts automatically if the worker crashes,
- restarts automatically if no new rows are written for a long stall window,
- exits when `expected_rows` are reached.

Optional: queue all downstream analysis automatically when the 3B run reaches 2633 rows.

```bash
python scripts/postprocess_when_complete.py \
  --target_file results/multi/open_qwen3_qwen15_full.jsonl \
  --expected_rows 2633 \
  --first_model qwen3 \
  --second_model qwen15 \
  --analysis_dir results/analysis_qwen3 \
  --drift_dir results/analysis_drift_qwen3 \
  --figures_dir results/figures_qwen3 \
  --ml_dir results/analysis_ml_qwen3 \
  --integrity_dir results/analysis_integrity_qwen3 \
  --sensitivity_dir results/analysis_sensitivity_qwen3
```


<a id="citation"></a>
## 🤲 Citation
If you find our work useful in your research, please consider citing our work:
```
@inproceedings{ki-etal-2025-multiple,
    title = "Multiple {LLM} Agents Debate for Equitable Cultural Alignment",
    author = "Ki, Dayeon  and
      Rudinger, Rachel  and
      Zhou, Tianyi  and
      Carpuat, Marine",
    editor = "Che, Wanxiang  and
      Nabende, Joyce  and
      Shutova, Ekaterina  and
      Pilehvar, Mohammad Taher",
    booktitle = "Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)",
    month = jul,
    year = "2025",
    address = "Vienna, Austria",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2025.acl-long.1210/",
    doi = "10.18653/v1/2025.acl-long.1210",
    pages = "24841--24877",
    ISBN = "979-8-89176-251-0",
}
```

<a id="contact"></a>
## 📧 Contact
For questions, issues, or collaborations, please reach out to [dayeonki@umd.edu](mailto:dayeonki@umd.edu).
