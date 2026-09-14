import numpy as np
import pytest

from diploma_pinn.data.validation import sha256_file, validate_rbc_dns


def test_validation_accepts_expected_rbc_schema(tmp_path) -> None:
    path = tmp_path / "dns.npz"
    np.savez(
        path,
        inputs=np.array([[0.0, 0.0, 0.5, 1.0], [0.5, 1.0, 0.5, 0.0]]),
        outputs=np.zeros((2, 5)),
    )
    report = validate_rbc_dns(path, sha256_file(path))
    assert report.rows == 2
    assert report.unique_times == (0.0, 0.5)


def test_validation_rejects_wrong_fingerprint(tmp_path) -> None:
    path = tmp_path / "dns.npz"
    np.savez(path, inputs=np.zeros((1, 4)), outputs=np.zeros((1, 5)))
    with pytest.raises(ValueError, match="fingerprint mismatch"):
        validate_rbc_dns(path, "0" * 64)
