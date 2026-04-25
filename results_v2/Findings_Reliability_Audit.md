# `results_v2`: Findings + Reliability Audit

This note treats `results_v2` as a `qwen3` vs `qwen7` multi-agent debate run over the full 2,633-row NORMAD dataset, but it separates:

- what the benchmark artifacts literally report, and
- what the raw generations suggest is at least partly a formatting or extraction problem rather than a clean debate-quality signal.

## Scope and Sources

Primary sources:

- `results_v2/analysis_qwen3_qwen7/qwen3_benchmark.json`
- `results_v2/analysis_qwen3_qwen7/qwen7_benchmark.json`
- `results_v2/analysis_qwen3_qwen7/significance_report.json`
- `results_v2/analysis_drift_qwen3_qwen7/drift_summary.json`
- `results_v2/analysis_sensitivity_qwen3_qwen7/qwen3_judge_sensitivity.json`
- `results_v2/analysis_sensitivity_qwen3_qwen7/qwen7_judge_sensitivity.json`
- `results_v2/analysis_integrity_qwen3_qwen7/integrity_report.json`
- `results_v2/multi/open_qwen3_qwen7.jsonl`
- `multi_llm/prompt.py`
- `scripts/judge_sensitivity_check.py`

I repaired `results_v2/multi/open_qwen3_qwen7.jsonl` from its stored `*_final_raw` fields and then re-ran the benchmark, drift, significance, integrity, and judge-sensitivity scripts. The headline numbers below reflect that repaired extraction logic.

## 1. What the Reported Metrics Literally Say

The evaluated final fields are `qwen3_final` and `qwen7_final`.

### Benchmark summaries

`qwen7_final` is the strong side of the pair:

- Global accuracy: `0.701861`
- Macro accuracy: `0.704390`
- PAA: `0.618825`
- CPG: `0.332143`
- Country accuracy range: `0.525000` to `0.857143`
  - worst: `mauritius`
  - best: `fiji`

`qwen3_final` is dramatically weak:

- Global accuracy: `0.332700`
- Macro accuracy: `0.333991`
- PAA: `0.286669`
- CPG: `0.184127`
- Country accuracy range: `0.244444` to `0.428571`
  - worst: `portugal`
  - best: `fiji`

### Between-model comparison

The paired significance file reports a strong separation in this run:

- Accuracy diff (`qwen3 - qwen7`): `-0.369161`, `p = 0.0005`
- PAA diff (`qwen3 - qwen7`): `-0.332157`, `p = 0.0005`
- CPG diff (`qwen3 - qwen7`): `-0.148016`, `p = 0.029485`

### Drift

Both models lose accuracy from their initial answer (`*_1`) to their final decision:

- `qwen3`: initial `0.696164` to final `0.332700`, drift rate `0.501709`
- `qwen7`: initial `0.805925` to final `0.701861`, drift rate `0.153057`

So if these metrics are read literally, debate is harmful for both models, and especially harmful for `qwen3`.

### Integrity

The row-level integrity report is clean:

- `2633` total rows
- `2633` unique rows
- `0` duplicates
- `0` missing reference rows
- `0` unexpected rows relative to `data/normad.jsonl`

This matters because the main anomaly is not a missing-data artifact.

## 2. What Those Metrics Imply at Face Value

If taken at face value, `results_v2` says:

- `qwen7` is much stronger than `qwen3` on final debate outputs.
- The `qwen3`/`qwen7` pairing does not improve final decisions relative to initial answers.
- The final debate stage is unstable enough to erase a large amount of initially correct reasoning, especially for `qwen3`.

That is already useful evidence. It means this exact pipeline, with these prompts and parsers, is not a reliable “debate improves cultural alignment” result.

## 3. Why the `qwen3` Final Score Is Not a Clean Debate-Quality Measure

The final-decision prompt in `multi_llm/prompt.py` requires exactly:

- `Label: <Yes|No|Neither>`
- `Reason: ...`

The raw rows show a sharp asymmetry in final-turn instruction following.

### `qwen3` raw behavior

Across the full file:

- `2633/2633` `qwen3_final_raw` tails begin with `assistant`
- `0/2633` include `Label:` after the final `assistant`
- `2633/2633` instead emit a bare label token after `assistant`
- `2632/2633` emit bare `No`
- `1/2633` emits bare `Yes`

So the evaluated `qwen3_final` field is effectively a near-constant `no` output:

- `2632` `no`
- `1` `yes`

That is not what a healthy debate result should look like, especially because the initial answers are often correct and rule-consistent before the final turn.

### `qwen7` raw behavior

`qwen7` follows the requested format much more closely:

- `2608/2633` rows match the clean two-line contract after the final `assistant`
- `25/2633` rows still contain a trailing extra `Label:` marker after an otherwise valid answer
- the repaired canonical parser recovers all `25/25` of those rows from `qwen7_final_raw`
- `0/2633` use the `qwen3`-style bare-label pattern

So `qwen7_final` is not perfect, but it is much closer to the prompt contract, and the repaired score is slightly stronger and cleaner as a measure of the evaluated pipeline:

- `1032` `yes`
- `1338` `no`
- `263` `neutral`
- `0` `invalid`

### Representative raw examples

These rows show the pattern directly. In all three, both initial answers are correct and aligned with the rule, `qwen3` flips to a bare `No`, and `qwen7` keeps the rule-consistent `Yes`.

#### Row 0, `egypt`

- Story: greeting everyone individually at a gathering
- Rule: greet everyone before social interaction
- `qwen3_1`: `Yes`
- `qwen7_1`: `Yes`
- `qwen3_final_raw` tail:

```text
assistant
No
Reason: While greeting everyone is respectful, the action itself does not inherently make the behavior socially acceptable; it depends on the context and the specific situation.
```

- `qwen7_final_raw` tail:

```text
assistant
Label: Yes
Reason: It aligns with the rule of greeting everyone before starting any social interaction.
```

#### Row 2, `egypt`

- Story: greeting a new arrival at a cafe gathering
- Rule: acknowledge and greet each person upon joining a group
- `qwen3_1`: `Yes`
- `qwen7_1`: `Yes`
- `qwen3_final_raw` tail:

```text
assistant
No
Reason: While greeting is respectful, the story does not specify the timing or context of Chris's greeting relative to Jordan's arrival, which could be seen as impolite if done too early.
```

- `qwen7_final_raw` tail:

```text
assistant
Label: Yes
Reason: Welcoming newcomers is respectful and follows social norms.
```

#### Row 3, `egypt`

- Story: bringing chocolates as a host gift
- Rule: always bring a token of gratitude
- `qwen3_1`: `Yes`
- `qwen7_1`: `Yes`
- `qwen3_final_raw` tail:

```text
assistant
No
Reason: While bringing chocolates is generally accepted, the specific choice of chocolates might not be culturally significant or preferred by all hosts in Egypt.
```

- `qwen7_final_raw` tail:

```text
assistant
Label: Yes
Reason: Bringing a token of gratitude is socially acceptable in Egypt.
```

There are many such cases. A direct file scan finds `716` rows where both initials match the gold label, `qwen3_final` is `no`, and `qwen7_final` is `yes`.

### What this means

The apparent collapse of `qwen3` is at least partly a final-turn instruction-following failure. The model is not just “reasoning badly”; it is also systematically failing the requested output contract and collapsing into a near-constant final label.

That makes the `0.3327` final accuracy a pipeline-level outcome, not a clean estimate of `qwen3`'s underlying cultural-reasoning ability.

## 4. Judge-Sensitivity Warning

The sensitivity outputs should be read carefully.

### What the files report

- `qwen3`: agreement `0.990125`, disagreement rate `0.009875`, alternate accuracy `0.330042` vs base `0.332700`
- `qwen7`: agreement `1.0`, disagreement rate `0.0`, alternate accuracy `0.701861`

So under the repaired sensitivity check, `qwen7` is parser-stable, while `qwen3` has a small but real formatting-dependent ambiguity.

### Why that happens

This is now a cleaner comparison. `scripts/judge_sensitivity_check.py` defines:

- a canonical lenient parser that can recover strict `Label:` lines, bare standalone labels, and malformed first lines like `No: ...`
- a canonical strict parser that accepts only the properly structured label forms

Both parsers normalize `Neither` and `neutral` to the same `neutral` class. That matters because the earlier `~10%` `qwen7` disagreement warning was mostly a parser-policy artifact around `neutral`, not true output instability.

After aligning both judges to the same canonical parser family:

- `qwen7` has `0` disagreements; the repaired parser recovers the trailing-`Label:` cases without inventing ambiguity
- `qwen3` has `26` disagreements because some rows use malformed first lines like `No: ...` or `Yes, ...`, which the lenient parser recovers and the strict parser rejects

So the remaining sensitivity warning is smaller, concentrated on genuinely malformed `qwen3` rows, and no longer inflated by a `neutral`/`Neither` vocabulary mismatch.

## 5. Safe Conclusions

These conclusions are supported even after the reliability audit:

- `qwen7` is much stronger than `qwen3` in this pairing under the evaluated pipeline.
- The debate setup, as run here, does not reliably improve final decisions relative to the initial answers.
- Output-format robustness is a critical bottleneck in this pipeline.
- `results_v2` is useful evidence about pipeline failure modes, especially final-turn extraction and instruction-following fragility.

## 6. Unsafe Conclusions

These claims would overreach the evidence in `results_v2`:

- “Debate universally hurts cultural alignment.”
- “`qwen3` is inherently incapable on this task.”
- “The `qwen3` final score is a pure reasoning result.”
- “The qwen7/qwen3 comparison isolates debate quality cleanly from formatting and extraction behavior.”

## Bottom Line

`results_v2` is strong evidence that this debate pipeline is fragile. It is also strong evidence that `qwen7_final` is the better-performing side of this specific run. But it is not clean evidence that multi-agent debate improved cultural alignment for this model pair, and `qwen3`'s final collapse should be discussed as a combined reasoning-plus-formatting failure, not as a pure capability estimate.

I did not use the ML clustering outputs for any causal claim here. They remain exploratory only.
