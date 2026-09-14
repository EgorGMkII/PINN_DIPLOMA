# Architecture Decision Records

Use an ADR when a durable choice affects several experiments or modules and would
otherwise be reopened repeatedly.

Naming:

```text
ADR-0001-specialized-step-builder.md
ADR-0002-canonical-coordinate-order.md
```

Each ADR contains:

```text
Status: proposed | accepted | superseded
Context
Decision
Alternatives considered
Consequences
Validation or rollback condition
```

Experiment outcomes belong in `docs/experiments/`, not in ADRs. An ADR may cite
the experiment that justified the decision.

