# AGENTS.md

Instructions for coding agents working on DiplomaPINN.

## Read First

1. `docs/experiment_plan.md`
2. `docs/framework_spec.md`
3. `docs/data_spec.md`
4. `docs/external_services.md` for logging or remote execution work
5. The specification under `docs/experiments/` for the experiment in scope
6. The active source file and its tests

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
