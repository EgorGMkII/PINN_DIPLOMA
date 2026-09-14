# EXP-NNN: Short name

Status: `proposed`

Owner:
Created:
Related global stage:
Related ADRs:

## Question

One precise question answered by this experiment.

## Hypothesis

Statement written before observing results, including the expected mechanism.

## Scientific Value

What conclusion would be useful if the hypothesis is confirmed, and what would
still be learned if it is rejected?

## Methods Compared

List the baseline and variants. State exactly one primary independent variable.

## Controlled Variables

- dataset and split
- observations
- model architecture/parameter-matching rule
- point budgets and sampler
- optimizer, scheduler and precision
- loss weighting/reduction
- number of updates and wall-time budget
- seeds and hardware class

## Data Contract

Dataset fingerprint, columns allowed for training, split ID, normalization and
physical parameters. Link `docs/data_spec.md` and record deviations.

## Configurations

Paths to versioned resolved configs. Do not describe an unversioned command as a
configuration.

## Metrics

Primary metric, secondary quality metrics, compute metrics and uncertainty
summary. Define pressure gauge and normalization.

## Acceptance And Stop Criteria

Numerical condition for success, failure, early termination, and escalation to a
full run. Set before results are read.

## Run Procedure

Exact safe command(s), required environment and expected artifact directory.

## Expected Artifacts

Run manifest, resolved config, metrics table, checkpoints if needed, profiler
summary and figures. State retention policy.

## Results

Append after execution. Include run IDs, commit hashes and failed runs.

## Interpretation And Limitations

Separate observed evidence from proposed mechanism. Record confounders and what
the experiment does not establish.

## Decision

What project decision follows, or why no decision can yet be made.

