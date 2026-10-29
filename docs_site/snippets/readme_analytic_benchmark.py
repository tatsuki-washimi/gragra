from astropy import units as u

from gragra.analytic.objects import uniform_sphere
from gragra.geometry import Frame, Sphere
from gragra.gravity_sources import discretize, point_sum
from gragra.results import compare
from gragra.targets import TargetPoints

frame = Frame("world")
sphere = Sphere([0, 0, 0] * u.m, 1 * u.m, frame)
density = 1000 * u.kg / u.m**3
targets = TargetPoints([[3, 0, 0]] * u.m, ("A",), frame)

source = discretize(sphere, density, 0.1 * u.m)
discrete = point_sum(source).at(targets).run()
analytic = uniform_sphere(sphere, density).at(targets).run()
comparison = compare(reference=analytic, candidate=discrete)
print(comparison.summary)
