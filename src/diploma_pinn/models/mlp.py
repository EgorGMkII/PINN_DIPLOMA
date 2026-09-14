"""Reference-compatible field network."""

import torch
from torch import Tensor, nn


class SineMLP(nn.Module):
    """Dense ``(t,x,y,z) -> (u,v,w,T,p)`` network with sine activations."""

    def __init__(self, widths: tuple[int, ...] = (4, *(256,) * 10, 5)) -> None:
        super().__init__()
        if len(widths) < 2:
            raise ValueError("widths must contain input and output dimensions")
        self.layers = nn.ModuleList(
            nn.Linear(input_width, output_width)
            for input_width, output_width in zip(widths, widths[1:])
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Match Keras Dense defaults: Glorot uniform kernels and zero biases."""
        for layer in self.layers:
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)

    def forward(self, points: Tensor) -> Tensor:
        values = points
        for layer in self.layers[:-1]:
            values = torch.sin(layer(values))
        return self.layers[-1](values)
