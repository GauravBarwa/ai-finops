# Historical router KPI replay

## Experiment scope

This report preserves the nine-task historical replay evaluated on 5 November 2026.
It used two Ollama models and the cost-first historical task-average routing mode.
The current benchmark dataset contains 16 tasks, and later model coverage expanded
to six active candidates.

The fixed baseline was `gpt-oss-120b-cloud`. Constraints required minimum reference
quality 0.50 and minimum reliability 0. Cost and latency were unbounded for the selection.
The comparison used configured Method A shadow token rates in USD.

Source filenames were `benchmark_20261003_055819.jsonl` and
`benchmark_20261003_055940.jsonl`. They are local experiment artifacts. A fresh clone
collects its own measurements before replaying the evaluation.

## Method

Each evaluated task was excluded from its routing profiles. The remaining task records
produced the model/task priors used for selection. Saved baseline and selected-model
outcomes were paired on the same held-out task.

Repeated trials were averaged within each task/model group. Failed trials received zero
quality and remained in latency/reliability metrics. KPI denominators used the paired
workload. The script reports skipped tasks and coverage when a comparison is unavailable.

## Recorded results

All nine observed tasks were paired, covering nine of the 16 tasks in the expanded dataset.

- Estimated shadow cost reduction was **95.96%**.
- Retained reference-quality score was **108.93%** of the fixed baseline.
- Median task latency increased by **1325.33%**.
- Baseline mean reference quality was **0.699**, while routed quality was **0.762**.
- Baseline estimated workload cost was **$0.00228340**, while routed cost was **$0.00009231**.
- Baseline median task latency was **840.0ms**, while routed latency was **11973.3ms**.
- Both paths recorded an observed call-success fraction of **1.000**.

Displayed figures are rounded. The median latency is calculated across per-task trial
means, which differs from taking the median of all raw calls in a task-type profile.
Quality retention divides the routed score by the baseline score, allowing a ratio
above 100% when the routed reference score is higher.

## Interpretation

The cost-first configuration reduced estimated token cost under the selected shadow
assumptions and retained the scorer's reference quality. Response latency increased
substantially. The experiment demonstrates the tradeoff that explicit latency limits
and alternative policies are intended to manage.

Shadow costs are experimental planning values. Actual billing, infrastructure cost,
semantic quality, provider load, and larger workload coverage require separate evidence.
The current prompt-aware estimator also needs independent held-out validation beyond
this historical task-average replay.

## Reproduce with your measurements

Collect and evaluate compatible benchmark data, then run the replay.

```bash
python -m scripts.evaluate_router_kpi \
  --baseline gpt-oss-120b-cloud \
  --policy cost_first \
  --min-quality 0.5 \
  --output data/results/kpi_report.md
```

`--results-dir` selects a separate local experiment. The command recomputes results
from its available runs and current shadow registry, so a new experiment can produce
different coverage and values. The generated report records its source filenames.
The implementation explicitly uses `task_average` routing for the replay.
