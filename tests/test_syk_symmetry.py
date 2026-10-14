"""SYK symmetry classes and sector-resolved level statistics (PLAN P1-2, P1-3)."""

import numpy as np
import pytest

from quantum_chaos.hamiltonians import syk_hamiltonian as syk_module
from quantum_chaos.hamiltonians.syk_hamiltonian import SYKHamiltonian
from quantum_chaos.spectral import R_MEAN


@pytest.mark.xfail(reason="P1-2: exact degeneracies are silently dropped", strict=True)
def test_level_spacing_ratio_keeps_degeneracies():
    from quantum_chaos.utils.helpers import level_spacing_ratio

    with pytest.warns(UserWarning):
        assert level_spacing_ratio(np.array([0.0, 1.0, 1.0, 3.0])) == pytest.approx(0.0)


@pytest.mark.xfail(reason="P1-3: every N is compared with GOE", strict=True)
@pytest.mark.parametrize("N, expected", [(8, "GOE"), (10, "GUE"), (12, "GSE"), (14, "GUE"),
                                         (16, "GOE")])
def test_disorder_average_reports_class_and_reference(N, expected):
    stats = syk_module.disorder_average_lsr(N, n_realizations=2)
    assert stats["symmetry_class"] == expected
    assert stats["reference"] == pytest.approx(R_MEAN[expected.lower()])
    assert SYKHamiltonian(N, seed=0).summary()["symmetry_class"] == expected


@pytest.mark.xfail(reason="P1-3: 1/N extrapolation mixes GOE, GUE and GSE sizes", strict=True)
def test_cross_class_finite_size_extrapolation_removed():
    assert not hasattr(syk_module, "finite_size_scaling")
