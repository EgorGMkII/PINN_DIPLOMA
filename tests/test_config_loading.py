from pathlib import Path

import pytest

from diploma_pinn.config import load_config, write_resolved_config


def test_config_resolves_paths_and_round_trips(tmp_path: Path) -> None:
    data = tmp_path / "data" / "smoke.npz"
    config_path = tmp_path / "configs" / "smoke.yaml"
    config_path.parent.mkdir(parents=True)
    config_path.write_text(
        """
experiment_id: smoke
data:
  path: ../data/smoke.npz
runtime:
  execution_profile: smoke
  max_steps: 3
  output_dir: ../outputs
""".strip(),
        encoding="utf-8",
    )

    config = load_config(config_path)
    assert config.data.path == data.resolve()
    assert config.runtime.output_dir == (tmp_path / "outputs").resolve()
    resolved_path = tmp_path / "resolved.yaml"
    write_resolved_config(config, resolved_path)
    resolved = load_config(resolved_path)
    assert resolved == config


def test_config_rejects_unknown_nested_key(tmp_path: Path) -> None:
    config_path = tmp_path / "bad.yaml"
    config_path.write_text(
        "experiment_id: smoke\ndata:\n  path: data.npz\n  batch_szie: 32\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="batch_szie"):
        load_config(config_path)
