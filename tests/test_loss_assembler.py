import torch

from diploma_pinn.losses import LossAssembler, LossWeights


def test_loss_assembler_matches_reference_weights() -> None:
    components = {
        "data": torch.tensor(2.0),
        "momentum": torch.tensor(3.0),
        "energy": torch.tensor(5.0),
        "continuity": torch.tensor(7.0),
        "boundary": torch.tensor(11.0),
    }
    report = LossAssembler(LossWeights())(components)
    expected = 2.0 + 0.1 * 3.0 + 0.01 * 5.0 + 0.001 * 7.0 + 0.0001 * 11.0
    torch.testing.assert_close(report.total, torch.tensor(expected))
