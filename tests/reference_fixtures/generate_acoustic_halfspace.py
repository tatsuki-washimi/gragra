"""Regenerate the high-precision G-free acoustic test fixture.

Run from the repository root with ``python
tests/reference_fixtures/generate_acoustic_halfspace.py``. This is a software
regression fixture, not the independent scientific oracle: mpmath differentiates
the single scalar potential directly so the vector and tensor values do not
reuse the production component formulas.
"""

from __future__ import annotations

import mpmath as mp


def main() -> None:
    mp.mp.dps = 80
    pressure = mp.mpc(2, 3)
    speed = mp.mpf(340)
    frequency = mp.mpf(3)
    theta = mp.pi / 4
    azimuth = mp.pi / 6
    phase_origin = (mp.mpf("0.1"), mp.mpf("0.2"))
    k = 2 * mp.pi * frequency / speed
    q = k * mp.sin(theta)
    p = k * mp.cos(theta)
    h = (q * mp.cos(azimuth), q * mp.sin(azimuth))
    coefficient = 4 * mp.pi * pressure / (speed**2 * k**2)
    positions = (
        (mp.mpf("0.8"), mp.mpf("-0.3"), mp.mpf("0.5")),
        (mp.mpf("0.8"), mp.mpf("-0.3"), mp.mpf("-0.5")),
    )

    def potential(x: tuple[mp.mpf, mp.mpf, mp.mpf]) -> mp.mpc:
        phase = mp.exp(
            1j * (h[0] * (x[0] - phase_origin[0]) + h[1] * (x[1] - phase_origin[1]))
        )
        if x[2] > 0:
            vertical = mp.exp(-q * x[2]) / 2 - mp.cos(p * x[2])
        else:
            vertical = -mp.exp(q * x[2]) / 2
        return coefficient * phase * vertical

    def partial(x: tuple[mp.mpf, mp.mpf, mp.mpf], axis: int) -> mp.mpc:
        return mp.diff(
            lambda value: potential(x[:axis] + (value,) + x[axis + 1 :]),
            x[axis],
        )

    def second_partial(x: tuple[mp.mpf, mp.mpf, mp.mpf], i: int, j: int) -> mp.mpc:
        return mp.diff(
            lambda value_j: partial(x[:j] + (value_j,) + x[j + 1 :], i),
            x[j],
        )

    for target_index, target in enumerate(positions):
        phi = potential(target)
        acceleration = tuple(-partial(target, i) for i in range(3))
        gradient = tuple(
            tuple(-second_partial(target, i, j) for j in range(3)) for i in range(3)
        )
        print(f"target {target_index} potential:", mp.nstr(phi, 45))
        print("acceleration:", tuple(mp.nstr(value, 45) for value in acceleration))
        print(
            "gradient:",
            tuple(tuple(mp.nstr(value, 45) for value in row) for row in gradient),
        )


if __name__ == "__main__":
    main()
