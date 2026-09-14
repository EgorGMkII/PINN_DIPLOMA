# DataSphere job specifications

Do not add runnable job YAML until the corresponding CLI command has passed a
local smoke test and the environment is locked. Planned jobs:

```text
exp-001-tf-reference.yaml
exp-001-torch-parity.yaml
exp-001-torch-benchmark.yaml
```

Each job validates the dataset fingerprint and exports the resolved config,
manifest, summary, and profiler report. Credentials use DataSphere secrets.
