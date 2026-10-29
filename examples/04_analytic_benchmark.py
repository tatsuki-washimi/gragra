"""Example 04: compare analytic and midpoint uniform-sphere evaluations."""

from astropy import units as u

from gragra.analytic.objects import uniform_sphere
from gragra.geometry import Frame, Sphere
from gragra.gravity_sources import discretize, point_sum
from gragra.results import compare
from gragra.targets import TargetPoints


def main() -> None:
    frame = Frame("world")
    sphere = Sphere([0.0, 0.0, 0.0] * u.m, 1.0 * u.m, frame)
    density = 1000.0 * u.kg / u.m**3
    targets = TargetPoints(
        [
            [2.0, 0.0, 0.0],
            [3.0, 0.0, 0.0],
            [5.0, 0.0, 0.0],
            [0.0, 2.0, 0.0],
            [1.0, 1.0, 1.0],
        ]
        * u.m,
        ("2R-x", "3R-x", "5R-x", "2R-y", "diagonal"),
        frame,
    )
    source = discretize(sphere, density, 0.1 * u.m)

    discrete_setup = point_sum(source).at(targets)
    validation = discrete_setup.validate()
    figure = discrete_setup.plot_setup(max_sources=5000)
    discrete = discrete_setup.run()

    analytic_setup = uniform_sphere(sphere, density).at(targets)
    analytic_setup.validate()
    analytic = analytic_setup.run()
    comparison = compare(reference=analytic, candidate=discrete)

    print(f"Sphere discretized into {len(source)} point masses")
    print(f"Validation checked {validation.checked_pairs} source-target pairs")
    print(f"Plot axes: {len(figure.axes)}")
    print(comparison.summary)


if __name__ == "__main__":
    main()
