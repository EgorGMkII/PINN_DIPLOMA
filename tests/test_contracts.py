import torch

from diploma_pinn.models import SineMLP


def test_reference_model_shape_and_initial_biases() -> None:
    model = SineMLP(widths=(4, 8, 5))
    output = model(torch.zeros(7, 4))
    assert output.shape == (7, 5)
    assert all(torch.count_nonzero(layer.bias) == 0 for layer in model.layers)
