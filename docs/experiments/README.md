# Experiment Specifications

## Active specifications

- [`EXP-001-tf-torch-vp-parity.md`](EXP-001-tf-torch-vp-parity.md) -- reproduce
  TensorFlow VP semantics and benchmark the new PyTorch step.

## Naming

Every experiment begins as a versioned Markdown specification:

```text
EXP-001-tf-torch-vp-parity.md
EXP-010-vp-vv-fo-pilot.md
EXP-020-formulation-main-study.md
EXP-030-observability-sparsity-noise.md
EXP-040-ptv-transfer.md
```

Use [`TEMPLATE.md`](TEMPLATE.md). One document should test one primary hypothesis.
If two independent variables cannot be explained as one controlled experiment,
split the document.

Lifecycle:

1. `proposed`: hypothesis and protocol written before implementation/run.
2. `ready`: data, code, config and acceptance criteria reviewed.
3. `running`: run IDs appended; original hypothesis remains unchanged.
4. `complete`: results, failures and interpretation added.
5. `superseded`: replacement document linked; history is retained.

Do not paste large logs or generated tables into Git. Link artifact identifiers
and commit the small aggregated result table needed to support the conclusion.
