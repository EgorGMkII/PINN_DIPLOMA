import torch

from diploma_pinn.data import RBCDNSDataset
from diploma_pinn.sampling import (
    GlobalShuffleSampler,
    RBCBoundarySampler,
    StratifiedTimeSampler,
    UniformDomainSampler,
)


def _dataset(tmp_path) -> RBCDNSDataset:
    import numpy as np

    inputs = np.array(
        [[time, row / 10, 0.25, 0.75] for time in (0.0, 0.5, 1.0) for row in range(6)],
        dtype=np.float32,
    )
    outputs = np.arange(inputs.shape[0] * 5, dtype=np.float32).reshape(-1, 5)
    path = tmp_path / "dns.npz"
    np.savez(path, inputs=inputs, outputs=outputs)
    return RBCDNSDataset(path)


def test_global_shuffle_is_seeded_and_preserves_velocity_pairs(tmp_path) -> None:
    dataset = _dataset(tmp_path)
    first = GlobalShuffleSampler(dataset, batch_size=7, seed=23).sample()
    second = GlobalShuffleSampler(dataset, batch_size=7, seed=23).sample()
    torch.testing.assert_close(first.points, second.points)
    torch.testing.assert_close(first.velocity, second.velocity)


def test_stratified_sampler_rotates_remainder_across_times(tmp_path) -> None:
    sampler = StratifiedTimeSampler(_dataset(tmp_path), batch_size=5, seed=7)
    one = torch.unique(sampler.sample().points[:, 0], return_counts=True)[1].tolist()
    two = torch.unique(sampler.sample().points[:, 0], return_counts=True)[1].tolist()
    assert sorted(one) == [1, 2, 2]
    assert sorted(two) == [1, 2, 2]
    assert one != two


def test_domain_and_boundary_sampling_reuse_time_and_respect_cube() -> None:
    times = torch.tensor([0.0, 0.25, 0.5])
    domain = UniformDomainSampler(device=torch.device("cpu"), dtype=torch.float32, seed=1).sample(times)
    boundaries = RBCBoundarySampler(device=torch.device("cpu"), dtype=torch.float32, seed=2).sample(times)
    torch.testing.assert_close(domain[:, 0], times)
    assert torch.all((domain[:, 1:] >= 0) & (domain[:, 1:] <= 1))
    assert set(boundaries) == {"x", "y", "z0", "z1"}
    assert torch.all((boundaries["x"][:, 1] == 0) | (boundaries["x"][:, 1] == 1))
    assert torch.all((boundaries["y"][:, 2] == 0) | (boundaries["y"][:, 2] == 1))
    assert torch.all(boundaries["z0"][:, 3] == 0)
    assert torch.all(boundaries["z1"][:, 3] == 1)
