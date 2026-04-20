# CS534 Project Proposal (Adjusted)
## Causal and Fairness-Aware Benchmarking of Cultural Alignment in Multi-LLM Debate

### Team
Sushil Bohara, Paulo Sergio Condori, Leo Ding

## 1. Summary
This adjusted project focuses on **cultural alignment benchmarking** using the codebase we already have in this repository. Instead of building a new generic conversation simulator, we will use the existing pipelines for:
- single-model inference,
- self-reflection,
- multi-agent debate,
- and evaluation on NORMAD-ETI-style cultural scenarios.

Our goal is to quantify whether multi-agent debate improves:
1. overall prediction quality, and
2. fairness across cultural groups/countries,
compared to single-model baselines.

The repository already supports most generation/evaluation workflows, so this project is designed to contribute **new science and new methods**, not only reruns:
1. a new fairness-oriented metric suite for cultural alignment,
2. intervention-based analysis to isolate what aspect of debate causes gains,
3. and a partner-selection policy that optimizes both average performance and cultural parity.

## 2. Problem Statement
Large language models often perform unevenly across cultural contexts. A model can achieve good overall accuracy while still failing systematically for specific countries or cultural regions.

This project asks:
1. Does multi-agent debate improve cultural alignment relative to single-model prompting?
2. Does self-reflection help, and how does it compare to debate?
3. Are gains consistent across countries, or concentrated in already easy regions?
4. What is the trade-off between average performance and cross-cultural parity?

We also ask a stronger causal question:
5. Which component of debate (partner identity, response order, or feedback structure) actually drives parity gains?

## 3. Core Novelty Claims
This project makes three explicit novel contributions:

1. **New evaluation objective for cultural alignment**
   - We introduce a fairness-aware score that jointly evaluates global quality and country-level parity.

2. **Intervention-based causal analysis of debate**
   - We move beyond descriptive comparison by systematically intervening on debate structure and estimating effect sizes.

3. **A practical policy for partner selection**
   - We propose a selection algorithm that chooses debate pairs to maximize parity-adjusted utility, not only raw accuracy.

## 3.1 What Makes This Project More Unique
The original paper already studies single-model prompting, self-reflection, debate-only, and self-reflect+debate. To move beyond that baseline line of work, this project adds a second layer of analysis that the prior setup does not capture:

1. **Cultural drift across turns, not just final correctness**
   - We will measure whether a model's stance or label changes from initial answer to final answer, and whether that change is systematic by country or cultural region.

2. **Counterfactual culture transfer**
   - We will reuse the same story while varying the country prompt to test how sensitive a model is to cultural context shifts.

3. **Calibration and abstention behavior**
   - We will study whether debate makes models more confident, less overconfident, or more likely to collapse to neutral responses in difficult countries.

4. **Disagreement topology**
   - We will analyze which countries or regions produce the most disagreement between agents, and whether disagreement concentrates in culturally distant groups.

5. **Judge sensitivity and partner effects**
   - Instead of treating debate as a single black box, we will separate what comes from the speaker, the partner, and the judge stage.

These additions make the project about more than "does debate help?". It becomes a study of how cultural judgments move, stabilize, or break under collaboration.

## 4. Scope (Adjusted to Existing Code)
### In scope
- Dataset: `data/normad.jsonl`
- Single-model baselines (`single_llm/single_model`): with and without rule-of-thumb (RoT)
- Self-reflection baselines (`single_llm/self_reflection`)
- Multi-LLM debate baselines (`multi_llm/*`)
- Evaluation scripts in `evaluate/` and new scripts in `scripts/`

### Out of scope
- Building a new external simulator framework
- Human annotation collection
- Full retraining/fine-tuning of base models

## 5. What We Need to Run This
To execute the full benchmark plan in this repository, we need the following minimum setup:

### A. Python Environment
- Python 3.10+ with a clean virtual environment
- Required packages for generation and analysis:
   - `torch`
   - `transformers`
   - `huggingface_hub`
   - `accelerate`
   - `sentencepiece`
   - `jsonlines`

### B. Model Access
- Access to at least one open-weight model or authenticated access to gated Hugging Face models
- A working Hugging Face login/token if using restricted checkpoints such as Gemma
- A fallback model list that is accessible in the current environment so the benchmark can still run if a gated model is unavailable

Recommended open-model paths to start with:
- `Qwen/Qwen2.5-1.5B-Instruct` for the best quality-speed balance in the pilot tests
- `Qwen/Qwen2.5-0.5B-Instruct` for a very fast smoke test
- `TinyLlama/TinyLlama-1.1B-Chat-v1.0` as a lightweight baseline

Pilot takeaway:
- On a 3-example sanity check, `Qwen/Qwen2.5-1.5B-Instruct` was the strongest of the tested open models.
- The smaller `Qwen/Qwen2.5-0.5B-Instruct` was much faster but too weak on this task.
- `TinyLlama/TinyLlama-1.1B-Chat-v1.0` was slower than the 0.5B Qwen model and also weaker on the pilot.

### C. Compute and Storage
- A GPU machine with enough memory to run the target model pair
- Sufficient disk space for model cache, generated outputs, and benchmark summaries
- A stable working directory for `results/` so all runs write outputs in a consistent location

### D. Benchmark Artifacts
- The refined dataset: `data/normad.jsonl`
- Single-run outputs for each model and prompting condition
- Multi-run debate outputs for each model pair
- Evaluation outputs generated by the new scripts in `scripts/`

### E. Execution Workflow
1. Generate single-model outputs.
2. Generate debate outputs for selected model pairs.
3. Run cultural benchmark metrics on each output file.
4. Run intervention analysis across the saved result folders.
5. Compare baseline, RoT, self-reflection, and debate using the new parity-aware metrics.

### F. Estimated Runtime
- Single-model pilot on 3 examples: about 1-2 minutes per model after the first download.
- Single full-dataset run for a small open model: roughly 10-30 minutes depending on model size and hardware.
- Debate pair full-dataset run: roughly 2x a single-model run because two models are invoked.
- The first time you use a model, add download time for its weights; Qwen 1.5B took about 24 seconds to download in the pilot environment, while larger models can take much longer.

## 6. Methods and Benchmark Design
We organize benchmarks into seven families. In addition to the parity-focused metrics already introduced, we add a classical machine-learning layer to study conversational dynamics directly.

### A. Core Accuracy Benchmarks
- **Single model (without RoT)**
- **Single model (with RoT)**
- **Self-reflection**
- **Multi-agent debate**

Primary metric:
- Exact-label accuracy on `{yes, no, neutral}` after normalization.

### B. Cultural Fairness / Parity Benchmarks
For each model condition, compute:
- per-country accuracy,
- macro-average across countries,
- variance/standard deviation across countries,
- worst-country accuracy,
- performance gap between best and worst country.

These metrics capture whether methods improve equity across cultural contexts, not just the average.

### C. Debate Gain Benchmarks
For each model pair in debate:
- compare each debater's final answer against its own single-model baseline,
- compute per-country and global delta,
- identify where debate helps or hurts.

This isolates the value of interaction beyond raw model quality.

### D. Prompting Strategy Benchmarks
For single-model baselines:
- compare `with_rot` vs `without_rot` globally and per-country,
- analyze whether RoT helps underrepresented/hard countries more than easy ones.

### E. New Composite Novelty Metrics
We introduce the following new metrics:

- **Cultural Parity Gap (CPG)**: best-country accuracy minus worst-country accuracy.
- **Parity-Adjusted Accuracy (PAA)**: global accuracy minus `lambda * CPG`.
- **Debate Parity Gain (DPG)**: `CPG(single) - CPG(debate)`.
- **Partner Stability Index (PSI)**: variance of a model's per-country accuracy across different debate partners.

These metrics directly test whether a method is robust and equitable across cultures.

### F. Intervention and Causal Benchmark
We run controlled interventions on debate structure:

- **Partner swap intervention**: hold first model fixed, vary second model.
- **Order intervention**: reverse speaking order where supported by scripts.
- **Feedback ablation**: compare full debate pipeline against reduced interaction variants (initial-only vs full feedback turns when script fields allow analysis).
- **Culture swap intervention**: keep the same story but substitute different country prompts to measure label sensitivity.
- **Judge swap intervention**: compare the final label under different judging or aggregation policies when available.

Analysis plan:
- estimate average treatment effect on PAA and CPG,
- use bootstrap confidence intervals across countries,
- run permutation tests for significance of parity improvements.

### G. Drift and Calibration Benchmarks
We add a new benchmark family that is not present in the original debate-only framing:

- **Opinion drift rate**: fraction of examples where the final answer differs from the first answer.
- **Culture-sensitive drift**: drift rate broken down by country, region, and label type.
- **Neutral collapse rate**: how often debate ends in a neutral answer compared with the initial stance.
- **Confidence proxy**: whether models become more or less decisive after debate, based on output form and response consistency.
- **Cross-country flip rate**: how often the same story changes label when the country changes.

### H. Machine Learning Conversation Analysis
To complement the parity and drift metrics, we will analyze conversational trajectories using clustering and supervised learning on turn-level features.

#### Turn-level feature extraction
For each conversation turn, we will derive features such as:
- sentence-transformer embeddings for the turn text,
- cosine similarity to the previous turn,
- stance label or stance-change indicator,
- sentiment score,
- conversation position / turn index,
- model speaker identity,
- whether the turn is part of debate or collaboration.

#### K-means clustering
We will cluster conversations or conversation trajectories using a compact feature vector formed from:
- mean embedding statistics,
- embedding variance across turns,
- sentiment mean and variance,
- stance-change rate,
- coherence statistics.

This is useful for discovering latent conversational regimes such as:
- stable agreement,
- polarized debate,
- high-drift / low-coherence conversations,
- collaboration that converges quickly.

#### Random Forest classification
We will train a Random Forest classifier to predict conversation type, such as:
- debate vs collaboration,
- high-drift vs low-drift,
- high-coherence vs low-coherence.

Feature importance will help us identify which characteristics most strongly separate interaction styles. We expect the most informative features to include:
- semantic similarity between turns,
- sentiment variance,
- stance-change frequency,
- trajectory length,
- and speaker-role patterns.

#### Dimensionality reduction and visualization
We will use PCA or a similar reduction method to project turn embeddings or conversation summaries into 2D for visualization. This provides a qualitative view of conversation trajectories and cluster separation.

## 7. Experimental Plan
### Models/conditions to run first
- Single: `gemma`, `llama3`, `aya`, `internlm`, `yi`, `exaone`, `seallm` (as available)
- Self-reflection: same set where scripts exist
- Multi pairs: start with existing pairs in `multi_llm/` and prioritize diverse pairs (e.g., `gemma_aya`, `llama3_yi`, `internlm_qwen`)

### Data split
- Use full provided evaluation file (`data/normad.jsonl`) for benchmark comparison.
- If needed for ablation speed, run pilot subset first, then full set.

### Reproducibility
- Use centralized runners in `scripts/`.
- Save outputs in structured folders under `results/`.
- Keep one command per run type and one report format per benchmark.

## 8. What I Will Do (Implementation Commitments)
I will complete the following in this repo:

1. **Finalize run orchestration**
   - finish README command documentation and validation for `scripts/run_single.py` and `scripts/run_multi.py`.

2. **Build benchmark report scripts**
   - add a benchmark script that produces:
     - global accuracy,
     - per-country table,
     - macro/micro summaries,
     - fairness gaps.

3. **Add novelty metric implementation**
   - implement CPG, PAA, DPG, PSI in a single analysis pipeline.

4. **Add intervention analysis**
   - implement partner-swap and order-based effect estimation with bootstrap confidence intervals and permutation testing.

5. **Add debate-gain analysis**
   - compare debate outputs to corresponding single-model outputs.

6. **Add a drift and calibration analysis layer**
   - quantify opinion drift, neutral collapse, and culture-sensitive flips across turns.

7. **Add clustering and classification analysis**
   - implement K-means clustering and Random Forest classification on turn-level conversation features.

8. **Add a consolidated benchmark output format**
   - generate machine-readable `.jsonl`/`.csv` summaries for easy plotting and write-up.

9. **Run initial benchmark matrix**
   - execute a minimal but representative set of models/pairs,
   - provide a first result table and interpretation.

10. **Build a parity-aware partner-selection baseline**
   - select partner model for each base model by maximizing validation PAA,
   - compare against fixed partner choice and strongest-accuracy-only partner choice.

11. **Write final project report structure**
   - methods,
   - benchmark definitions,
   - novelty metrics and intervention protocol,
   - results and failure cases,
   - limitations and next steps.

## 9. Expected Contributions
1. A reproducible benchmark framework for cultural alignment using existing multi-agent debate scripts.
2. New fairness-aware cultural metrics (CPG, PAA, DPG, PSI) for multi-agent evaluation.
3. A drift and calibration analysis layer that shows how opinions move across turns and cultures.
4. A machine-learning conversation analysis layer using K-means and Random Forest to identify latent interaction regimes.
5. Causal evidence about which debate factors improve or degrade parity.
6. A parity-aware partner-selection policy that can be reused in future multi-agent systems.
7. Practical guidance on when debate helps cultural alignment, when it merely stabilizes uncertainty, and when it fails.

## 10. Risks and Mitigations
- **Compute cost / runtime**: start with pilot subset and prioritized model list.
- **Output format inconsistencies**: use centralized normalization and evaluation scripts.
- **Metric ambiguity**: fix one canonical benchmark schema early and keep all reports aligned to it.
- **Intervention validity constraints**: where scripts do not expose direct control knobs, use post-hoc structured ablations with transparent caveats.

## 11. Deliverables
- Updated proposal (this file)
- Benchmark scripts in `scripts/`
- Reproducible run commands in `README.md`
- Results tables for baseline, intervention, and debate comparisons
- A parity-aware partner-selection baseline report
- Final analysis narrative focused on cultural alignment, parity, and causal mechanisms
