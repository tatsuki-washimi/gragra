"""Example 03: build and evaluate a uniform sphere with the C5 object API."""

from astropy import units as u

from gragra.geometry import Frame, Sphere
from gragra.gravity_sources import discretize, point_sum
from gragra.targets import TargetPoints


def main() -> None:
    frame = Frame("world")
    sphere = Sphere([0.0, 0.0, 0.0] * u.m, 1.0 * u.m, frame)
    density = 1000.0 * u.kg / u.m**3
    targets = TargetPoints([[3.0, 0.0, 0.0]] * u.m, ("A",), frame)
    source = discretize(sphere, density, 0.2 * u.m)

    request = point_sum(source).at(targets)
    validation = request.validate()
    figure = request.plot_setup()
    result = request.run()

    print("=== Uniform sphere ===")
    print(f"  Frame                  : {frame.name}")
    print(f"  Number of point masses : {len(source)}")
    print(f"  Discrete mass          : {source.mass.to_value(u.kg):.4e} kg")
    print(f"  Validation             : {validation.valid}")
    print(f"  Potential at A         : {result.potential[0]:.6e}")
    print(f"  Acceleration at A      : {result.acceleration[0]}")
    print(f"  Plot axes              : {len(figure.axes)}")


if __name__ == "__main__":
    main()
