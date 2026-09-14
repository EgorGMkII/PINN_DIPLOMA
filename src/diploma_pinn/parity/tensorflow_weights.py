"""TensorFlow-to-PyTorch parameter mapping for EXP-001 fixtures."""

from pathlib import Path

from diploma_pinn.models import SineMLP


def import_keras_dense_weights(model: SineMLP, fixture: Path) -> None:
    """Load kernels with ``torch_weight = tf_kernel.T`` and copy biases."""
    raise NotImplementedError("define a framework-neutral fixture schema")
