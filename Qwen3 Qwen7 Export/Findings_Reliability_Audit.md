# `Qwen3 Qwen7 Export`: Findings + Reliability Audit

This note evaluates the export bundle using raw-label recomputation as the primary source of truth and treats the saved benchmark artifacts as supporting evidence.

## Data Integrity

- Rows: `2633` total, `2633` unique
- Expected row count met: `True`
- Violations: duplicates `0`, missing required `0`, unexpected vs reference `0`, missing reference `0`

## Core Metrics

| Model | Final Accuracy | Initial Accuracy | Stance Change | Final Label Distribution |
| --- | ---: | ---: | ---: | --- |
| `qwen3` | `0.332700` | `0.696164` | `0.501709` | `yes=1, no=2632, neutral=0` |
| `qwen7` | `0.701861` | `0.805925` | `0.153057` | `yes=1032, no=1338, neutral=263` |

- Gold label distribution: `yes=943`, `no=875`, `neutral=815`
- Pairwise final agreement: `0.508545`; Cohen's kappa: `0.000862`

## Findings

- `qwen7` is the stronger final model in this export, with recomputed final accuracy `0.701861`.
- `qwen3` collapses to `no` much more often than its partner. In the raw final labels it emits `no` on `2632` of `2633` rows.
- Interaction hurts `qwen3` more than `qwen7` by raw recomputation. The final-stage stance-change rates are `0.501709` vs `0.153057`.
- The saved paired comparison also favors `qwen7`: accuracy diff `-0.369161` with p-value `0.0005`.
- Drift artifacts align with the raw summary: `qwen3` moves from `0.696164` initial accuracy to `0.332700` final accuracy, while `qwen7` moves from `0.805925` to `0.701861`.

## Reliability Notes

- `qwen3` judge-sensitivity agreement is `0.990125` with accuracy delta `-0.002659`.
- `qwen7` judge-sensitivity agreement is `1.000000` with accuracy delta `0.000000`.
- Confidence-interval inconsistencies were detected in saved artifacts, so raw recomputation should be treated as primary evidence:
  - `benchmark:qwen3` metric `cpg` has point estimate `0.184127` outside CI `0.327586` to `0.542125`.
  - `benchmark:qwen3` metric `paa` has point estimate `0.286669` outside CI `0.193876` to `0.257849`.
  - `benchmark:qwen7` metric `cpg` has point estimate `0.332143` outside CI `0.384615` to `0.608597`.
  - `benchmark:qwen7` metric `paa` has point estimate `0.618825` outside CI `0.548812` to `0.613635`.

## Bottom Line

`qwen7` is the more reliable side of this pair on the final judged labels, while `qwen3` shows a much less stable final-stage behavior. The export is structurally complete, but any substantive conclusion should prioritize raw-label recomputation over any saved artifact whose intervals are internally inconsistent.
