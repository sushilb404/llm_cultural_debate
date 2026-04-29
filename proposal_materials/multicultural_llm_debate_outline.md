# Multicultural LLM Debate: Proposal Analysis and Structure

## Core Takeaway

The experiments suggest that multi-agent debate can improve multicultural acceptability judgments, but the benefit depends strongly on the quality and behavior of the participating models. The best pairing was Qwen7 + Gemma4, while Qwen7 + Ollama Llama3.1 8B was weaker but still usable, and Qwen7 + Qwen1.5 showed that a weak debate partner can introduce noisy disagreement rather than useful correction.

This means the paper should not argue that more agents automatically improve cultural reasoning. The stronger claim is that debate can help when both agents are capable enough to critique, revise, and converge on culturally relevant evidence.

## Main Results

| Run | Model | Accuracy | Notes |
| --- | --- | ---: | --- |
| Qwen7 + Gemma4 | Gemma4 | 78.7% | Best final accuracy |
| Qwen7 + Gemma4 | Qwen7 | 75.4% | Strong paired performance |
| Qwen7 + Ollama Llama3.1 8B | Qwen7 | 71.3% | Qwen7 significantly ahead |
| Qwen7 + Ollama Llama3.1 8B | Ollama Llama3.1 8B | 66.9% | Lower but coherent debate partner |
| Qwen7 + Qwen1.5 | Qwen7 | 73.1% | Strong asymmetry |
| Qwen7 + Qwen1.5 | Qwen1.5 | 43.3% | Weak debate partner |

Pairwise agreement also separates the runs:

| Run | Agreement |
| --- | ---: |
| Qwen7 + Gemma4 | 84.2% |
| Qwen7 + Ollama Llama3.1 8B | 83.8% |
| Qwen7 + Qwen1.5 | 40.1% |

The Qwen7 + Gemma4 and Qwen7 + Ollama runs had high agreement, but the former paired high agreement with stronger accuracy. Qwen7 + Qwen1.5 had low agreement and poor partner accuracy, indicating unstable interaction.

## What This Says About Multicultural LLMs

1. Cultural reasoning is not solved by scale alone or by adding a second model. Debate helps only when the partner model can provide meaningful critique.

2. Neutral or context-dependent cases are the hardest. The ML cluster analysis found low-performing clusters dominated by neutral gold labels. This suggests that models often overcommit to yes or no when the correct answer requires uncertainty, ambiguity, or insufficient cultural evidence.

3. Country-level performance varies substantially. Some countries and norm types are consistently easier, while others produce more errors. This supports the need for country-specific evaluation rather than only aggregate accuracy.

4. Model pair composition matters. Gemma4 provided the strongest debate partner in these experiments, while Qwen1.5 produced noisy disagreement. Ollama Llama3.1 8B was coherent but weaker than Qwen7 in the paired run.

5. Debate should be evaluated with both outcome and interaction metrics. Accuracy alone misses whether models agree, drift, correct one another, or reinforce mistakes.

## Recommended Figures

### Figure 1: Overall Accuracy by Run and Model

Use a grouped bar chart showing final accuracy for each model in each debate pair:

- Qwen7 + Gemma4
- Qwen7 + Ollama Llama3.1 8B
- Qwen7 + Qwen1.5

Purpose: establish the headline result that Qwen7 + Gemma4 is the strongest pairing.

### Figure 2: Pairwise Agreement Across Debate Runs

Use a bar chart with:

- Qwen7 + Gemma4: 84.2%
- Qwen7 + Ollama Llama3.1 8B: 83.8%
- Qwen7 + Qwen1.5: 40.1%

Purpose: show that useful debate requires stable interaction. Low agreement in Qwen7 + Qwen1.5 reflects noise, not productive disagreement.

### Figure 3: Country-Level Accuracy Heatmap

Use:

- `results/figures_qwen7_gemma4/drift_country_heatmap.png`
- `results/figures_qwen7_ollama_llama3_8b/drift_country_heatmap.png`

Purpose: show that multicultural performance varies across countries and cannot be captured by aggregate accuracy alone.

### Figure 4: Debate Drift and Accuracy Gain

Use:

- `results/figures_qwen7_gemma4/drift_accuracy_gain.png`
- `results/figures_qwen7_ollama_llama3_8b/drift_accuracy_gain.png`
- optionally `drift_overview.png`

Purpose: show how answers changed through debate and whether those changes improved correctness.

### Figure 5: Scenario Difficulty Clusters

Use:

- `results/analysis_cross_run_ml/scenario_clusters.csv`
- `results/analysis_cross_run_ml/cross_run_ml_summary.json`

Purpose: show that the hardest clusters are dominated by neutral or ambiguous scenarios, while high-performing clusters are mostly clearer yes/no cases.

## Suggested Paper Structure

### 1. Introduction

Introduce the problem of multicultural norm reasoning. LLMs are increasingly used in culturally diverse contexts, but social acceptability judgments depend on local norms, context, and ambiguity. A single model may overgeneralize or default to majority-culture assumptions.

End the introduction with the central question: can multi-agent debate improve culturally situated judgment, and when does it fail?

### 2. Research Questions

RQ1: Does multi-agent debate improve cultural acceptability classification?

RQ2: How does debate partner choice affect performance?

RQ3: Which cultural judgment categories are most difficult for LLMs?

RQ4: Do agreement and answer drift explain when debate helps or hurts?

### 3. Dataset and Task

Describe the cultural scenario format:

- country
- story
- rule of thumb
- gold label: yes, no, or neutral

The task is to classify whether the action described in the story is socially acceptable according to the relevant cultural norm.

### 4. Method

Describe the debate protocol:

1. Each model gives an initial answer.
2. Each model sees or responds to the other model's reasoning.
3. Each model gives a final label.
4. Final answers are evaluated against the gold label.

Describe the compared pairings:

- Qwen7 + Gemma4
- Qwen7 + Ollama Llama3.1 8B
- Qwen7 + Qwen1.5

Describe evaluation metrics:

- final accuracy
- pairwise agreement
- country-level accuracy
- drift and answer revision
- cultural parity and performance spread
- bootstrap and permutation significance tests
- ML clustering and random forest analysis

### 5. Results

Start with the main accuracy table. Emphasize that Qwen7 + Gemma4 produced the strongest performance. Then compare against Qwen7 + Ollama and Qwen7 + Qwen1.5 to show that debate effectiveness depends on model choice.

Report that Qwen7 significantly outperformed Ollama Llama3.1 8B in the Qwen7 + Ollama run, while Gemma4 significantly outperformed Qwen7 in the Qwen7 + Gemma4 run.

### 6. Interaction Analysis

Use agreement and drift results to explain the differences between runs. High agreement in Qwen7 + Gemma4 suggests stable convergence. Low agreement in Qwen7 + Qwen1.5 suggests disagreement without reliable correction.

This section should argue that debate quality depends on whether the models can challenge each other productively.

### 7. Multicultural Error Analysis

Use the country heatmaps and ML clusters. Emphasize that the hardest examples are often neutral or context-dependent. These cases require recognizing ambiguity rather than forcing a yes/no answer.

This is the most important analysis section for the multicultural framing.

### 8. Limitations

Mention:

- several experiments are single runs rather than repeated seeds
- prompt format may affect comparability between single-model and debate settings
- gold labels simplify complex cultural situations
- high accuracy does not prove deep cultural understanding
- model outputs may reflect label-pattern learning as well as reasoning

### 9. Conclusion

Conclude that multi-agent debate is a promising method for multicultural LLM evaluation, but only when the debate partners are sufficiently capable. The results show that model pairing, agreement, and neutral-case handling are central to whether debate improves culturally grounded reasoning.

## Recommended Thesis Statement

Multi-agent debate can improve LLM performance on multicultural acceptability judgments, but its effectiveness is conditional: strong model pairings produce stable and accurate judgments, while weak pairings introduce noisy disagreement. The hardest failures occur in neutral or culturally ambiguous scenarios, showing that future multicultural LLM evaluation must measure not only accuracy but also agreement, answer drift, and scenario-level difficulty.
