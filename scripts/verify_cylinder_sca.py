"""Print target-wise independent surface errors and refinement for T-C."""

import runpy
from pathlib import Path

import numpy as np

from gragra.analytic import cylinder_sca_wall_coupling_tensor_g_free


def main():
    oracle = runpy.run_path(
        str(Path(__file__).resolve().parents[1] / "tests/test_cylinder_sca_tensor.py")
    )
    targets = oracle["TARGETS"]
    actual = cylinder_sca_wall_coupling_tensor_g_free([0, 0, 0], 2.0, 3.0, targets)
    low = oracle["surface_tensor"](targets, order=48)
    high = oracle["surface_tensor"](targets, order=80)
    errors = np.linalg.norm(actual - high, axis=(1, 2))
    delta = np.linalg.norm(high - low, axis=(1, 2))
    width = 1e-12 + 1e-9 * np.linalg.norm(high, axis=(1, 2))
    assert np.all(errors <= width)
    assert np.all(delta <= width / 10)
    print("target |C-reference80| |reference80-reference48| acceptance_width")
    for target, error, ref_error, limit in zip(
        targets, errors, delta, width, strict=True
    ):
        print(target, f"{error:.6e} {ref_error:.6e} {limit:.6e}")
    print("center tensor:", actual[0])
    print("general off-axis tensor:", actual[1])


if __name__ == "__main__":
    main()
