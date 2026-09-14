from pathlib import Path

import numpy as np

from diploma_pinn.config.schema import DataConfig, ExperimentConfig, ModelConfig, RuntimeConfig
from diploma_pinn.training.factory import build_experiment


def test_factory_runs_one_cpu_vp_update(tmp_path: Path) -> None:
    path = tmp_path / "dns.npz"
    inputs = np.array([[time, 0.2, 0.3, 0.4] for time in (0.0, 0.5) for _ in range(4)], dtype=np.float32)
    outputs = np.zeros((len(inputs), 5), dtype=np.float32)
    np.savez(path, inputs=inputs, outputs=outputs)
    config = ExperimentConfig(
        experiment_id="test",
        data=DataConfig(path=path, expected_sha256="", batch_size=4),
        model=ModelConfig(widths=(4, 8, 5)),
        runtime=RuntimeConfig(device="cpu", max_steps=1),
    )
    metrics = build_experiment(config).trainer.run()
    assert metrics.step == 0
    assert metrics.total >= 0
