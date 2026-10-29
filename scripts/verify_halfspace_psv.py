"""Print target-wise W errors and independent Green-reference refinement.

Run using the branch source: PYTHONPATH=src python scripts/verify_halfspace_psv.py.
SciPy/pytest are needed for the independent test oracle; no output is committed.
"""

import runpy
from pathlib import Path

import numpy as np

from gragra.analytic import halfspace_psv_observables_g_free


def main():
    oracle = runpy.run_path(
        str(Path(__file__).resolve().parents[1] / "tests/test_halfspace_psv.py")
    )
    points = np.array([[3.0, -4.0, 20.0], [-10.0, 30.0, -25.0], [4.0, 5.0, -80.0]])
    amplitude = 1e-6 + 2e-6j
    print("material mode offset target observable error/Qscale reference_delta/Qscale")
    maximum = np.zeros(2)
    for vp, vs in oracle["MATERIALS"]:
        scale = (
            2700
            * abs(amplitude)
            * np.array([vs / (2 * np.pi * 5), 1.0, 2 * np.pi * 5 / vs])
        )
        for mode, offset in [
            ("P", 0.1),
            ("SV", -0.01),
            ("SV", -1e-8),
            ("SV", 0.0),
            ("SV", 1e-8),
            ("SV", 0.01),
            ("SV", 0.4),
        ]:
            theta = np.arcsin(vs / vp) + offset
            kw = {
                "vp": vp,
                "vs": vs,
                "theta": theta,
                "incident": mode,
                "omega": 2 * np.pi * 5,
                "rho": 2700.0,
                "amplitude": amplitude,
                "azimuth": 0.7,
            }
            low = oracle["green_reference"](points, **kw, tolerance=2e-10)
            high = oracle["green_reference"](points, **kw, tolerance=2e-12)
            actual = halfspace_psv_observables_g_free(
                points,
                wave_type=mode,
                density_kg_m3=2700.0,
                v_p_m_s=vp,
                v_s_m_s=vs,
                frequency_hz=5.0,
                incidence_angle_rad=theta,
                azimuth_rad=0.7,
                incident_displacement_m=amplitude,
            )
            for i, name in enumerate(("potential", "acceleration", "gradient")):
                err = (
                    np.linalg.norm(
                        (getattr(actual, name) - high[i]).reshape(3, -1), axis=1
                    )
                    / scale[i]
                )
                delta = (
                    np.linalg.norm((high[i] - low[i]).reshape(3, -1), axis=1) / scale[i]
                )
                ref_norm = np.linalg.norm(high[i].reshape(3, -1), axis=1) / scale[i]
                rtol = 1e-7 if mode == "SV" and abs(offset) < 1e-7 else 1e-10
                width = 1e-11 + rtol * ref_norm
                assert np.all(err <= width)
                assert np.all(delta <= width / 10)
                maximum = np.maximum(maximum, [err.max(), delta.max()])
                for target in range(3):
                    print(
                        f"{vp:g}/{vs:g} {mode} {offset:g} {target} {name} "
                        f"{err[target]:.5e} {delta[target]:.5e}"
                    )
    print("maximum normalized error, reference delta:", maximum)


if __name__ == "__main__":
    main()
