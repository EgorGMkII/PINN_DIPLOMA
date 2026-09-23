# AGENTS.md

Instructions for coding agents working on DiplomaPINN.

## Read First

1. `docs/experiment_plan.md`
2. `docs/framework_spec.md`
3. `docs/data_spec.md`
4. `docs/external_services.md` for logging or remote execution work
5. The specification under `docs/experiments/` for the experiment in scope
6. `docs/specifications/architecture.md` and the relevant focused specification
7. The active source file and its tests

## Project Objective

Build a small PyTorch research framework for controlled comparison of PDE
formulations in sparse 3D inverse Rayleigh--Benard reconstruction. The framework
is an experimental instrument, not the thesis contribution by itself.

The primary scientific comparison is `VP` versus `VV` versus `FO` under matched
data, sampling, optimizer, model, and compute budgets. Do not add new experiment
axes without updating `docs/experiment_plan.md` and writing an experiment spec.

## Source Of Truth

- Scientific questions and stage gates: `docs/experiment_plan.md`
- Architecture and interfaces: `docs/framework_spec.md`
- Dataset meaning, normalization, and splits: `docs/data_spec.md`
- Exact run configuration and acceptance criteria: `docs/experiments/EXP-*.md`
- Lasting design choices: `docs/decisions/ADR-*.md`
- Executable defaults: versioned configuration files, once implemented

When prose and code disagree, report the discrepancy. Do not silently choose one.

## Safety And Reproducibility

- Do not commit datasets, checkpoints, W&B directories, secrets, or generated
  field exports.
- Do not install dependencies, launch training, use a GPU, or download data
  without explicit approval.
- Do not run Git write operations unless explicitly requested.
- Never use T or p from the DNS training rows unless the experiment spec allows
  them. They are hidden ground truth by default.
- The PTV CSV columns T and p are placeholders, not labels.
- Every completed run must record commit, config, seed, data fingerprint, split,
  hardware, software versions, point counts, optimizer updates, wall time, and
  output paths.
- Never put W&B or DataSphere credentials in source, configs, documentation,
  shell history examples, or run manifests.
- For a remote job expected to last an hour or more, create a timer-backed
  follow-up before ending the turn. Check its status no later than the stated
  estimate, report completion/failure promptly, and keep checking at a useful
  cadence until results are downloaded or the job needs intervention.

## Implementation Rules

- Keep observation loss, PDE formulation, sampling, optimization, and evaluation
  as independent modules.
- A formulation returns named residual tensors; it must not select points, apply
  weights, call `backward`, or step the optimizer.
- The trainer performs exactly one explicitly documented optimizer update per
  train step.
- Compute only derivatives required by active residuals.
- Avoid `.item()`, CPU transfers, logging, or file writes inside the hot step.
- Add analytical residual tests before using a new formulation in training.
- Compare methods by optimizer steps and wall time, never epoch labels alone.

## Scope Control

Current mandatory formulations: strong velocity--pressure (`VP`), strong
velocity--vorticity (`VV`), and mixed first-order (`FO`). Weak forms, SPAV,
trajectory integration losses, KANs, and a general-purpose geometry framework
are out of scope until the main comparison reaches its decision gate.

## Current Implementation Stage

EXP-001 VP parity is complete. The active sequence is EXP-002 matched VP--VV,
EXP-003 mode observability, then EXP-004 FO. Do not launch full VP/VV/FO or
sparse/noisy matrices before their documented pilot gates. The parity sampler
remains global shuffle; post-parity experiments use stratified-time sampling.

There are no periodic checkpoints. Preserve one complete `checkpoints/final.pt`
per run. DNS `T,p` may be read only by evaluation; PTV `T,p` columns are zero
placeholders. The labels `001` and `005` are not percentages: they contain 8,100
and 40,500 observations per time respectively.
