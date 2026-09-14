# DiplomaPINN

Research code for a master's thesis on reconstruction of three-dimensional
velocity, temperature, and pressure fields in Rayleigh--Benard convection from
sparse Lagrangian velocity observations.

The project is currently in the specification phase. The first implementation
goal is a compact PyTorch training kernel that preserves the useful training
semantics of the TensorFlow RBC-PINN reference while allowing controlled changes
of PDE formulation, sampling, and optimization.

Start with:

1. [`docs/experiment_plan.md`](docs/experiment_plan.md) -- scientific questions
   and experiment sequence.
2. [`docs/framework_spec.md`](docs/framework_spec.md) -- required software
   boundaries and training-step semantics.
3. [`docs/experiments/README.md`](docs/experiments/README.md) -- how individual
   experiments are specified and recorded.
4. [`docs/external_services.md`](docs/external_services.md) -- W&B logging and
   DataSphere job contract.
5. [`AGENTS.md`](AGENTS.md) -- rules for coding agents.

The source datasets and the old implementations are intentionally not copied
into this repository. Their locations and roles are documented in
[`docs/data_spec.md`](docs/data_spec.md).
