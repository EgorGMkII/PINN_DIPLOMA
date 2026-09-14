# Documentation Map

| Document | Question it answers | Update trigger |
| --- | --- | --- |
| [`experiment_plan.md`](experiment_plan.md) | What scientific claims do the experiments test? | Research scope or stage gate changes |
| [`framework_spec.md`](framework_spec.md) | How must the experimental software behave? | Public interface or training semantics change |
| [`data_spec.md`](data_spec.md) | What does each dataset contain and how may it be used? | Dataset, normalization, or split changes |
| [`external_services.md`](external_services.md) | How do W&B and DataSphere participate safely and reproducibly? | Logging or remote execution changes |
| [`experiments/README.md`](experiments/README.md) | How is one experiment specified and recorded? | Experiment workflow changes |
| [`experiments/TEMPLATE.md`](experiments/TEMPLATE.md) | Which fields are required for a new experiment? | Required evidence changes |
| [`decisions/README.md`](decisions/README.md) | Where are durable design decisions recorded? | Decision process changes |

Current experiment: [`EXP-001: TensorFlow--PyTorch VP parity`](experiments/EXP-001-tf-torch-vp-parity.md).

Documentation roles are intentionally separated:

- The global plan contains hypotheses and comparisons, not every hyperparameter.
- Framework documentation contains contracts and invariants, not research results.
- External-service documentation defines W&B/DataSphere integration; credentials
  and machine-specific authentication remain outside Git.
- Experiment specifications contain exact configurations and acceptance criteria.
- Run artifacts contain machine-generated measurements and are not committed.
- ADRs explain choices that would otherwise be repeatedly reconsidered.
