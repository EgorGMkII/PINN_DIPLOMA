# DataSphere jobs

## EXP-001 PyTorch smoke

The smoke job runs three VP optimizer updates on GPU. It verifies dataset
delivery, package imports, CUDA, sampling, autograd, finite losses/gradients,
Adam parameter updates, and output collection. It is not a convergence run.

Prepare the ignored time-stratified dataset from the article snapshots:

```powershell
python scripts/build_smoke_dataset.py `
  --raw-dir "C:\path\to\RBC_PTV_1E6_07\input\raw_data" `
  --output data\smoke\RBC_PTV_1E6_07_smoke.npz `
  --rows-per-time 256 `
  --seed 2204
```

If the builder prints a different SHA-256, update
`configs/datasphere/exp001_torch_smoke.yaml` before submission.

Validate locally:

```powershell
$env:PYTHONPATH = "src"
python -m diploma_pinn.cli validate-data --config configs/exp001_torch_smoke_cpu.yaml
python -m diploma_pinn.cli train --config configs/exp001_torch_smoke_cpu.yaml --smoke
```

Submit and inspect the DataSphere job:

```powershell
datasphere project job execute -p <project_id> -c datasphere/exp-001-torch-smoke.yaml
datasphere project job list -p <project_id>
datasphere project job download-files --id <job_id>
```

Successful output under `outputs/EXP-001-smoke/<run_id>/` contains:

```text
manifest.json
resolved_config.yaml
summary.json
```

The job uses Python 3.10.13 and `requirements-datasphere.txt`. Dataset files,
outputs, and credentials remain outside Git.

## Later jobs

TensorFlow reference, full PyTorch training, and performance benchmark jobs are
added only after the smoke job succeeds. Credentials use DataSphere secrets.
