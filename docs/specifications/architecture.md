# Architecture implementation contract

Status: skeleton for `EXP-001`; implementation owner: coding agent.

## Vertical slice

The first implementation supports only the strong VP formulation for
`RBC_PTV_1E6_07`. The package is modular at construction time and specialized at
execution time. It is not a general geometry or condition framework.

```text
RBCDNSDataset -> ObservationSampler -----+
DomainSampler ---------------------------+-> PointBatch
BoundarySampler -------------------------+
PointBatch -> VP loss kernel -> LossAssembler -> one backward -> one Adam update
```

## Ownership boundaries

- `data`: validates schemas and prevents accidental T,p training-label access.
- `sampling`: chooses points; it never evaluates the model.
- `models`: maps `(t,x,y,z)` to `(u,v,w,T,p)`.
- `formulations`: computes named, unreduced PDE residuals.
- `losses`: owns reductions and scalar weights.
- `training`: owns zero-grad, one backward, and one optimizer update.
- `evaluation`: may access complete DNS truth and pressure gauge alignment.
- `instrumentation`: runs outside the hot graph.

The production loss kernel is a concrete callable assembled once. It may call
modules, but it must not iterate over generic conditions or discover active
residuals dynamically during a step.

## Implementation order

1. Dataset schema and leakage tests.
2. Global-shuffle and stratified-time observation samplers.
3. Uniform domain and RBC boundary samplers.
4. Sine MLP plus TensorFlow weight mapping fixture.
5. VP derivatives and analytical residual tests.
6. Literal reference reductions and boundary terms.
7. One-step TensorFlow/PyTorch parity fixture.
8. Eager benchmark, then compiled and fused-Adam profiles.

Terra must not implement later stages until the preceding correctness tests pass.

## Complete module map

```text
config/           typed resolved configuration
data/             guarded DNS loading and schema validation
sampling/         observation, domain, and boundary point policies
models/           field networks and initialization
operators/        coordinate derivative graph
formulations/     unreduced named PDE residuals
observations/     velocity-only data operator
boundaries/       active RBC boundary losses
losses/           reductions, weights, and concrete VP kernel
training/         optimizer factories, one step, lifecycle, assembly
evaluation/       chunked fields, metrics, pressure gauge
instrumentation/  manifest, checkpoints, phase timing, memory
integrations/     optional W&B boundary
parity/           TensorFlow fixture and weight mapping
runtime/          seeds, determinism, and device policy
cli.py            validate/parity/train/benchmark/evaluate entry points
```
