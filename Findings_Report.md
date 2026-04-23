# Findings Report: Cultural Alignment, Debate Dynamics, and Fairness-Aware Evaluation

This document reviews the current project results and interprets them against the goals in `Proposal_CS534_adjusted.md`. The proposal frames the project around a specific scientific question: whether multi-agent debate improves both overall cultural-alignment accuracy and fairness across countries compared with single-model baselines. It also proposes a deeper mechanism-level analysis through parity metrics, drift, intervention tests, partner selection, judge sensitivity, and machine-learning analysis of conversation trajectories.

The current results are valuable, but the main finding is cautionary: the full Qwen debate run is complete and internally consistent, yet the final decisions collapse to `yes` for every example for both debaters. As a result, the current full-run benchmark does not demonstrate successful debate-based cultural alignment. Instead, it reveals an important failure mode in the debate or final-label extraction pipeline that must be resolved before making claims about debate improving cultural fairness.

## Source Artifacts Reviewed

The review is grounded in the proposal and the following result artifacts:

| Artifact | Role in this review |
| --- | --- |
| `Proposal_CS534_adjusted.md` | Defines the project goals, novelty claims, metrics, and intended contribution. |
| `results/multi/open_qwen3_qwen15_full.jsonl` | Full Qwen 3B/Qwen 1.5B debate output over the NORMAD-style dataset. |
| `results/analysis_integrity_qwen3/integrity_report.json` | Confirms full-run coverage and duplicate checks. |
| `results/analysis_qwen3/qwen3_benchmark.json` | Full-run benchmark metrics for `qwen3_final`. |
| `results/analysis_qwen3/qwen15_benchmark.json` | Full-run benchmark metrics for `qwen15_final`. |
| `results/analysis_qwen3/significance_report.json` | Paired bootstrap/permutation comparison between final debater outputs. |
| `results/analysis_drift_qwen3/drift_summary.json` | Initial-to-final stance drift summary for the full run. |
| `results/analysis_drift_qwen3/drift_country_metrics.csv` | Country-level drift and accuracy-change metrics. |
| `results/analysis_sensitivity_qwen3/*.json` | Judge/parser sensitivity checks for saved final labels. |
| `results/analysis_ml_qwen3/ml_summary.json` | K-means and Random Forest conversation-feature analysis. |
| `results/analysis/*.json` and `results/analysis/*.csv` | Pilot intervention and partner-selection outputs. |

## Project Goals From the Proposal

The proposal identifies two primary evaluation questions:

1. Does multi-agent debate improve overall prediction quality?
2. Does it improve fairness across cultural groups and countries?

It also adds stronger goals that go beyond simple accuracy:

1. Measure country-level parity through metrics such as Cultural Parity Gap (CPG), Parity-Adjusted Accuracy (PAA), Debate Parity Gain (DPG), and Partner Stability Index (PSI).
2. Analyze how judgments move across debate turns through drift and neutral-collapse metrics.
3. Separate speaker, partner, and judge effects through intervention-style analysis.
4. Use machine-learning methods to identify conversational regimes and features associated with drift or stability.
5. Develop a parity-aware partner-selection baseline.

These goals matter because cultural-alignment systems can look good on global accuracy while still failing unevenly across countries. The proposal correctly treats fairness as a first-class benchmark target rather than an afterthought.

## Data Integrity Finding

The full debate output passes the basic integrity checks:

| Check | Result |
| --- | ---: |
| Total rows | 2,633 |
| Unique rows | 2,633 |
| Expected rows met | Yes |
| Duplicate rows | 0 |
| Missing required rows | 0 |
| Unexpected rows vs. reference | 0 |
| Missing reference rows | 0 |

This is significant because the main full-run anomaly is not caused by missing data, duplicate examples, or incomplete generation. The output appears complete relative to `data/normad.jsonl`, so the observed behavior should be interpreted as a modeling, prompting, or parsing failure rather than a file coverage failure.

## Main Full-Run Finding: Final Answer Collapse

The full debate output contains 2,633 examples. Both final prediction fields contain only `yes`:

| Field | `yes` | `no` | `neutral` |
| --- | ---: | ---: | ---: |
| Gold labels | 943 | 875 | 815 |
| `qwen3_final` | 2,633 | 0 | 0 |
| `qwen15_final` | 2,633 | 0 | 0 |

This means the final accuracy is exactly the prevalence of the `yes` class in the dataset:

`943 / 2633 = 0.358147`

The benchmark summaries therefore report the same final metrics for both debaters:

| Metric | `qwen3_final` | `qwen15_final` |
| --- | ---: | ---: |
| Total | 2,633 | 2,633 |
| Correct | 943 | 943 |
| Global accuracy | 0.358147 | 0.358147 |
| Macro country accuracy | 0.359134 | 0.359134 |
| Best-country accuracy | 0.470588 | 0.470588 |
| Worst-country accuracy | 0.305556 | 0.305556 |
| CPG | 0.165033 | 0.165033 |
| PAA, lambda = 0.25 | 0.316888 | 0.316888 |

### Why This Is Significant

This directly bears on the proposal's first two research questions. The current full-run result does not support the claim that debate improves prediction quality. It also does not provide meaningful evidence that debate improves fairness. The final outputs behave like an always-`yes` classifier.

This is an important finding rather than just a failed run. It shows that multi-agent debate can introduce a collapse mode where the final decision loses the label diversity present in the task. For a cultural-alignment benchmark, that is especially serious: a system that always answers `yes` may appear moderately accurate on aggregate if `yes` is common, but it is not adapting to cultural context.

It also shows why the proposal's parity-aware metrics are useful. Global accuracy alone reports 35.8%, but CPG and PAA expose that the system has country-level variation even under a trivial policy. However, in this collapsed run, CPG mostly reflects differences in each country's gold-label distribution, not meaningful differences in cultural reasoning.

## Country-Level Findings

Because both final predictors always output `yes`, each country's final accuracy is equal to that country's fraction of gold `yes` labels. The best and worst countries under the final benchmark are:

| Rank | Country | Total | Correct | Final accuracy |
| ---: | --- | ---: | ---: | ---: |
| 1 | Palestinian Territories | 34 | 16 | 0.471 |
| 2 | Brazil | 32 | 15 | 0.469 |
| 3 | Fiji | 21 | 9 | 0.429 |
| 4 | Laos | 33 | 14 | 0.424 |
| 5 | Afghanistan | 41 | 17 | 0.415 |
| 71 | Israel | 42 | 13 | 0.310 |
| 72 | India | 29 | 9 | 0.310 |
| 73 | Thailand | 39 | 12 | 0.308 |
| 74 | Serbia | 39 | 12 | 0.308 |
| 75 | Hungary | 36 | 11 | 0.306 |

The measured Cultural Parity Gap is:

`0.470588 - 0.305556 = 0.165033`

### Why This Is Significant

The country-level table demonstrates a subtle but important evaluation risk. A collapsed classifier can still produce a nonzero country-level gap. If interpreted naively, this could be mistaken for cultural performance differences. In this run, the country pattern is mostly an artifact of label distribution: countries with more gold `yes` examples look easier, and countries with fewer gold `yes` examples look harder.

This validates the proposal's decision to include fairness-aware analysis, but it also shows that fairness metrics must be interpreted alongside prediction-distribution diagnostics. CPG and PAA are necessary, but they are not sufficient when a model has collapsed to one label.

## Drift Findings

The drift analysis compares each model's initial answer with its final answer.

| Model | Total | Drift rate | Neutral collapse rate | Initial accuracy | Final accuracy | Accuracy gain |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `qwen3` | 2,633 | 0.535891 | 0.000000 | 0.527535 | 0.358147 | -0.169389 |
| `qwen15` | 2,633 | 0.099506 | 0.000000 | 0.410179 | 0.358147 | -0.052032 |

The most important pattern is that final debate reduces measured accuracy for both models, especially `qwen3`.

For `qwen3`, 74 of 75 countries have negative initial-to-final accuracy gain. The largest drops include:

| Country | Drift rate | Initial accuracy | Final accuracy | Gain |
| --- | ---: | ---: | ---: | ---: |
| Hong Kong | 0.600 | 0.657 | 0.371 | -0.286 |
| Türkiye | 0.600 | 0.629 | 0.343 | -0.286 |
| Malta | 0.583 | 0.611 | 0.333 | -0.278 |
| France | 0.647 | 0.618 | 0.353 | -0.265 |
| Zimbabwe | 0.412 | 0.618 | 0.353 | -0.265 |

For `qwen15`, 59 of 75 countries have negative initial-to-final accuracy gain, 14 are unchanged, and only 2 improve.

### Why This Is Significant

The proposal explicitly adds cultural drift as a novelty layer. These results show why that layer is necessary. If we only looked at final answers, we would know the final system performed poorly. Drift tells us more: the debate process appears to move answers away from more accurate initial judgments and toward a uniform final label.

This is the opposite of the desired mechanism. The intended debate behavior is that agents use discussion to correct mistakes, incorporate cultural context, and converge to better final judgments. The observed behavior suggests convergence, but toward a degenerate answer rather than toward correctness.

The neutral-collapse rate is zero, so the failure is not a collapse to uncertainty. It is better described as `yes` collapse or affirmational collapse. Future analysis should add an explicit label-collapse metric for each final label, not only neutral collapse.

One caveat: the current drift scripts normalize long prose answers with a permissive parser. In some scripts, text containing "socially acceptable" can be mapped to `yes` before checking for negation. This can affect initial-answer accuracy and drift estimates. The all-`yes` final-label finding is more robust because the saved final fields are already exact labels, but initial-vs-final drift should be recomputed with a stricter label parser before being treated as final.

## Significance Test Finding

The paired significance report compares `qwen3_final` and `qwen15_final` over the full debate output.

| Quantity | Value |
| --- | ---: |
| Accuracy difference | 0.000000 |
| CPG difference | 0.000000 |
| PAA difference | 0.000000 |
| Accuracy-difference CI | [0.000000, 0.000000] |
| CPG-difference CI | [0.000000, 0.000000] |
| PAA-difference CI | [0.000000, 0.000000] |
| Two-sided permutation p-values | 1.000000 |

### Why This Is Significant

The equality between debaters is not evidence that the two models are equally good cultural reasoners. It is a consequence of both final prediction fields being identical for every row. The significance test is functioning as a diagnostic: it confirms there is no measurable difference between debaters in the saved final labels, but that sameness reflects collapse rather than robust agreement.

This matters for the proposal's judge-sensitivity and partner-effect goals. The current full-run artifact cannot separate model-specific effects, partner effects, or judge-stage effects because both final outputs are identical and constant.

## Judge Sensitivity Finding

The judge sensitivity checks report perfect agreement between the base parser and stricter parser for the saved final labels:

| Model | Agreement | Cohen's kappa | Disagreements | Accuracy delta |
| --- | ---: | ---: | ---: | ---: |
| `qwen3` | 1.000 | 1.000 | 0 | 0.000 |
| `qwen15` | 1.000 | 1.000 | 0 | 0.000 |

### Why This Is Significant

This result says the saved final labels are stable under the parser comparison that was run. It does not prove the final-label generation process was correct. The final raw text was not saved in the full output, so we cannot distinguish among these possibilities:

1. Both models actually generated `yes` as the final answer for every row.
2. The final generation echoed prompt text containing "Yes" and the extractor selected `yes`.
3. The extractor's ordering favored `yes` when ambiguous output contained multiple label words.
4. The prompt or chat-template handling caused final outputs to be parsed incorrectly.

The current output examples show initial responses beginning with `assistant`, which suggests chat-template stripping is imperfect. Because raw final responses are not preserved, this should be treated as a high-priority reproducibility issue.

## Pilot Intervention and Partner-Selection Findings

The intervention outputs in `results/analysis/` should be interpreted as pilot evidence, not as final full-dataset evidence.

The available file counts are:

| Output | Rows |
| --- | ---: |
| `results/single/single_model/qwen15_without_rot.jsonl` | 40 |
| `results/single/single_model/qwen3_without_rot.jsonl` | 100 |
| `results/multi/qwen3_qwen15.jsonl` | 40 |
| `results/multi/qwen15_qwen3.jsonl` | 40 |
| `results/multi/open_qwen3_qwen15_full.jsonl` | 2,633 |

The pilot intervention table reports:

| Base model | Single accuracy | Debate accuracy | Single CPG | Debate CPG | DPG | Single PAA | Debate PAA | Delta PAA |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `qwen15` | 0.650 | 0.650 | 0.095 | 0.095 | 0.000 | 0.626 | 0.626 | 0.000 |
| `qwen3` | 0.150 | 0.100 | 0.500 | 0.060 | 0.440 | 0.025 | 0.085 | 0.060 |

The partner-selection file selects `qwen15` as the partner for `qwen3` because it improves pilot PAA despite reducing pilot accuracy from 0.150 to 0.100.

### Why This Is Significant

The pilot results illustrate the value of PAA as a decision metric: a method can lose raw accuracy but gain parity-adjusted utility if it substantially reduces CPG. That is exactly the kind of trade-off the proposal aims to study.

However, the pilot intervention evidence is currently too small and mismatched to support a final claim. The single baselines and mini debate runs do not cover the same full 2,633-row benchmark as the full debate run. The PSI result is also not meaningful yet: it reports `0.0` with only two evaluated pair files that are effectively the same model pair in reversed order.

The significance of this result is methodological rather than substantive. The analysis pipeline exists, but it must be rerun on matched full-dataset baselines and multiple genuinely distinct partners before it can support causal or partner-selection claims.

## Machine-Learning Conversation Analysis Finding

The ML summary reports K-means clusters and Random Forest classification for both models. Cluster counts are:

| Model | Cluster 0 | Cluster 1 | Cluster 2 | Cluster 3 |
| --- | ---: | ---: | ---: | ---: |
| `qwen3` | 487 | 598 | 747 | 801 |
| `qwen15` | 358 | 920 | 970 | 385 |

The Random Forest reports 1.0 training accuracy for both models. The largest feature importances are:

| Model | Most important features |
| --- | --- |
| `qwen3` | `initial_label_id` = 0.443, `stance_changed` = 0.403 |
| `qwen15` | `initial_label_id` = 0.452, `stance_changed` = 0.422 |

### Why This Is Significant

The ML layer is currently useful as implementation scaffolding, but not as scientific evidence. The Random Forest target is whether stance changed, and `stance_changed` is also included as an input feature. That creates direct label leakage. The model is also evaluated on the same data it is trained on, which explains the perfect accuracy.

The proposal's idea remains strong: clustering and supervised learning can help identify stable agreement, high-drift, low-coherence, or convergence regimes. But the current ML output should be rewritten before being used in a final report:

1. Remove target-leaking features from the input.
2. Use a train/test split or cross-validation.
3. Predict an outcome that is not already encoded in the features.
4. Add semantic or embedding-based features if the goal is to study conversation content rather than label bookkeeping.

## Overall Interpretation

The current full-run results do not validate the main expected benefit of debate. Instead, they identify a central failure mode:

> The Qwen 3B/Qwen 1.5B full debate pipeline converges to identical final `yes` labels for every example, reducing final accuracy below measured initial accuracy and making fairness metrics reflect label-distribution artifacts rather than cultural reasoning.

This finding is significant for three reasons.

First, it directly challenges the project's starting hypothesis. The proposal asks whether debate improves accuracy and fairness. The current full run shows that debate can degrade accuracy and erase label diversity.

Second, it demonstrates why fairness-aware evaluation is necessary. A single global score would miss the mechanism of failure. Country-level accuracy, CPG, PAA, drift, and prediction-distribution checks together reveal that the final system is not culturally adaptive.

Third, it improves the project by identifying exactly what needs to be fixed before the final claims are written. The result is not merely "bad accuracy"; it is a reproducible collapse pattern over a complete dataset.

## Recommended Claims for the Current Stage

The following claims are supported by the current artifacts:

1. The full Qwen debate run completed over all 2,633 reference rows with no duplicate or missing-row integrity violations.
2. Both saved final debater outputs collapse to `yes` on every full-run example.
3. The resulting final accuracy, 0.358147, equals the gold `yes` prevalence in the dataset.
4. The measured final CPG, 0.165033, reflects cross-country variation under an always-`yes` policy and should not be interpreted as meaningful cultural parity.
5. The drift layer reveals substantial movement from initial to final answers, especially for `qwen3`, and that movement is mostly harmful under current metrics.
6. Judge/parser sensitivity on saved final labels shows no disagreement, but raw final generations are needed to validate the final extraction stage.
7. Pilot intervention and partner-selection scripts are implemented but not yet supported by matched full-dataset baselines.
8. The ML conversation-analysis pipeline is present but currently has target leakage and should not be used for substantive conclusions without revision.

The following claims are not yet supported:

1. Multi-agent debate improves overall cultural-alignment accuracy.
2. Multi-agent debate improves cross-country fairness on the full benchmark.
3. A specific partner-selection policy improves cultural parity on a validated full-dataset setting.
4. Partner identity, response order, or judge policy has a causal effect on parity under the current full-run evidence.
5. The Random Forest or K-means results reveal validated conversational regimes.

## Recommended Next Steps

Before writing final project conclusions, the most important next step is to repair and rerun the final decision pipeline.

1. Save raw final generations in addition to parsed labels, for example `qwen3_final_raw` and `qwen15_final_raw`.
2. Fix label extraction to check explicit labels and negation before broad substring matches. In particular, avoid mapping any text containing "socially acceptable" to `yes` before checking "not socially acceptable".
3. Add unit tests for label parsing with cases such as `No, this is not socially acceptable`, `Neither`, prompt echoes, and outputs containing all three label words.
4. Verify chat-template stripping so saved generations do not include assistant markers or echoed prompt text.
5. Add prediction-distribution diagnostics to every benchmark summary: counts for `yes`, `no`, `neutral`, invalid, and missing.
6. Add explicit collapse metrics, including always-label rate and final-label entropy.
7. Rerun full single-model baselines for `qwen3` and `qwen15` on the same 2,633 rows.
8. Rerun the full debate after parser fixes and compare against matched full single baselines.
9. Rerun drift with a stricter parser and add `yes_collapse_rate`, not just `neutral_collapse_rate`.
10. Rerun intervention and partner-selection analysis only on matched full-dataset outputs.
11. Revise the ML pipeline to remove target leakage and evaluate on held-out data.

## Bottom Line

The current results are scientifically useful because they expose a concrete failure mode in multi-agent cultural debate. The full run is complete, so the failure is not a missing-data artifact. But the final predictions collapse to `yes`, which means the current full-run metrics cannot be used to claim that debate improves cultural alignment or fairness.

Grounded in the proposal's goals, the strongest current conclusion is:

> The benchmark and analysis framework is in place, and it successfully detects a high-impact debate-stage failure. The next project milestone should be a corrected full rerun with raw final outputs, stricter label parsing, matched single baselines, and renewed fairness/drift/intervention analysis.
