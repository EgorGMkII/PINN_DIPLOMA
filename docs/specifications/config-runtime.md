# Configuration and runtime specification

Versioned YAML contains scientific and execution settings but no secrets or
machine-specific dataset path. A local ignored overlay or environment value
supplies the path. Every run writes the fully resolved configuration and its hash.

Profiles are introduced in order: `parity`, `eager`, `profile`, `compiled`, then
`fused`. Compilation and fused Adam are enabled only after their correctness
parity checks.

The manifest records Git commit and dirty flag, dataset fingerprint, split,
seeds, point counts, model schema, precision, backend versions, hardware, profile,
optimizer updates, timing policy, output paths, and termination status.

W&B is optional and lazily imported. Disabled and offline modes must work without
network access. DataSphere jobs invoke the same CLI/config contract as local runs.
