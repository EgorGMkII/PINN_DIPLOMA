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
| [`specifications/architecture.md`](specifications/architecture.md) | How is the first VP vertical slice divided into modules? | Module ownership changes |
| [`specifications/sampling.md`](specifications/sampling.md) | How are observation, collocation, and boundary points sampled? | Sampling semantics change |
| [`specifications/implementation_tasks.md`](specifications/implementation_tasks.md) | What should an implementation agent build next? | A focused task passes its tests |
| [`specifications/autograd-vp.md`](specifications/autograd-vp.md) | Which derivatives and reductions must VP compute? | VP equations or AD backend changes |
| [`specifications/training-lifecycle.md`](specifications/training-lifecycle.md) | What happens during construction, a step, and a run? | Training semantics change |
| [`specifications/evaluation.md`](specifications/evaluation.md) | How are hidden fields and pressure evaluated? | Evaluation protocol changes |
| [`specifications/config-runtime.md`](specifications/config-runtime.md) | What belongs in config, manifests, and execution profiles? | Runtime contract changes |
| [`specifications/parity-and-benchmark.md`](specifications/parity-and-benchmark.md) | How is correctness and speed compared? | Parity or timing protocol changes |

Current experiment: [`EXP-001: TensorFlow--PyTorch VP parity`](experiments/EXP-001-tf-torch-vp-parity.md).

Primary dataset profile: [`RBC_PTV_1E6_07`](datasets/RBC_PTV_1E6_07.md).

Documentation roles are intentionally separated:

- The global plan contains hypotheses and comparisons, not every hyperparameter.
- Framework documentation contains contracts and invariants, not research results.
- External-service documentation defines W&B/DataSphere integration; credentials
  and machine-specific authentication remain outside Git.
- Experiment specifications contain exact configurations and acceptance criteria.
- Run artifacts contain machine-generated measurements and are not committed.
- ADRs explain choices that would otherwise be repeatedly reconsidered.
