import json
from pathlib import Path

import numpy as np
import pytest

import monols

CASES = json.loads((Path(__file__).parents[2] / "tests" / "fixtures" / "cases.json").read_text())
SUPPORTED = lambda c: True  # noqa: E731


def arr(v):
    return None if v is None else np.array([np.nan if e is None else e for e in v], dtype=float)


@pytest.mark.parametrize("case", CASES, ids=[c["name"] for c in CASES])
def test_fixture(case):
    if not SUPPORTED(case):
        pytest.skip("not implemented yet")
    y = arr(case["y"])
    f = monols.fit(y, arr(case["x"]), weights=arr(case["weights"]), **case["options"])
    exp = case["expected"]
    z = arr(exp["fitted"])
    yv = y[~np.isnan(y)]
    atol = 1e-8 * (np.ptp(yv) if yv.size and np.ptp(yv) > 0 else 1.0)
    np.testing.assert_array_equal(np.isnan(f.fitted), np.isnan(z))
    if case["options"]["loss"] == "l2":  # L1 minimizers need not be unique: compare the loss only
        np.testing.assert_allclose(f.fitted[~np.isnan(z)], z[~np.isnan(z)], atol=atol)
    assert (f.direction, f.curvature) == (exp["direction"], exp["curvature"])
    if case["options"]["loss"] == "l2":
        assert f.loss_value == pytest.approx(exp["loss_value"], rel=1e-7, abs=1e-12)
    else:
        assert f.loss_value == pytest.approx(exp["loss_value"], rel=1e-6, abs=1e-12)
