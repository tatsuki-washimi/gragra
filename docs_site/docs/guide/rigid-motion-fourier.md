# Rigid motion: continuous Fourier analysis

`rigid_motion` builds a rigid-motion model of a point-mass cloud. It has two
forms. This guide covers the **continuous form**, in which the body pose is a
function of time. That form gives Fourier coefficients directly, without
sampling the motion on a grid first.

## Choosing a form

| Second argument | Required extra argument | Evaluation |
|---|---|---|
| `PrescribedRigidTransform` (sampled motion) | `sampling=TimeSampling(...)` | `.at(targets).run()` returns a time-series `EvaluationResult` |
| callable pose (continuous motion) | `epoch=` (GPS time of pose time zero) | `.at(targets)`, then `fourier_series` or `fourier_transform`, below |

The factory selects the form from the second argument. Mixing the arguments
(for example `epoch=` with a sampled transform, or `sampling=` with a callable
pose) raises `TypeError`. `provenance` is not supported in either form.

## Units and conventions

- Times are in seconds, relative to `epoch` for the pose. Positions are in
  metres, and accelerations in m/s². Fourier frequencies are in Hz.
- The pose maps body coordinates to the computation frame as
  `x = R x_body + c`. It returns one rotation per time and a translation
  `Quantity` of shape `(N, 3)` in metres.
- `epoch` is the absolute GPS time of pose time zero. Pass a `LIGOTimeGPS`,
  an astropy `Time`, or a `(seconds, nanoseconds)` tuple of integers.
- Fourier coefficients use the phase convention
  `exp[-2*pi*i*f*(t - tref)]`. `tref` is an explicit argument of each Fourier
  method and has no default. The epoch is never used as an implicit phase
  reference.

```python
import numpy as np
from astropy import units as u
from scipy.spatial.transform import Rotation

from gragra.constants import G_SI
from gragra.geometry import Frame
from gragra.gravity_sources import PointMassCloud, rigid_motion
from gragra.targets import TargetPoints

EPOCH = (1_000_000_000, 0)
world = Frame("world")
cloud = PointMassCloud(np.array([1.0]), np.array([[0.0, 0.0, 0.0]]))
targets = TargetPoints(np.zeros((1, 3)) * u.m, ids=("centre",), frame=world)


def circular_pose(times):
    t = times.to_value(u.s)
    angle = 2.0 * np.pi * t
    translation = np.stack([np.cos(angle), np.sin(angle), np.zeros_like(angle)], axis=1)
    return Rotation.identity(len(t)), translation * u.m


circle = rigid_motion(cloud, circular_pose, frame=world, epoch=EPOCH)
```

## Fourier series of periodic motion

`fourier_series` evaluates the periodic coefficients. It requires the
period, the harmonic orders, `tref`, `tau_min` and `atol`.

```python
series = circle.at(targets, observables=("acceleration",)).fourier_series(
    period=1.0 * u.s,
    orders=[0, 1, 2],
    tref=EPOCH,
    tau_min=0.125 * u.s,
    atol=1e-9 * G_SI * u.m / u.s**2,
)
ax = np.asarray(series.acceleration[0, 0])  # x component, one value per order
```

The amplitudes are stored one-sided: `A_0 = C_0` and `A_n = 2 C_n` for
`n > 0`. For the circle above, the first harmonic is `A_1x = G` and
`A_1y = -iG`, and the other orders vanish. The example reproduces this closed
form.

## Fourier transform of a finite window

`fourier_transform` integrates over `[t_start, t_stop]` at the requested
frequencies. Its coefficients are one-sided and are **not** doubled. The
absolute tolerance has the units of acceleration times seconds, so write it
as `atol=... * u.m / u.s`.

```python
def passage_pose(times):
    t = times.to_value(u.s)
    translation = np.stack([t, np.ones_like(t), np.zeros_like(t)], axis=1)
    return Rotation.identity(len(t)), translation * u.m


passage = rigid_motion(cloud, passage_pose, frame=world, epoch=EPOCH)
spectrum = passage.at(targets, observables=("acceleration",)).fourier_transform(
    t_start=-2.0 * u.s,
    t_stop=2.0 * u.s,
    frequencies=np.array([0.0, 0.5, 1.0]) * u.Hz,
    tref=EPOCH,
    tau_min=0.5 * u.s,
    atol=1e-9 * G_SI * u.m / u.s,
    rtol=1e-10,
)
```

## Result and metadata

Both methods return an `EvaluationResult`, so `.select(...)` and the usual
accessors apply. `series.context.representation` is an amplitude spectrum with
the `PERIODIC` interpretation, and `spectrum.context.representation` is a
one-sided Fourier transform. `result.evaluation.parameters` records the
quadrature settings, including `atol`, `rtol`, `tref`, and `epoch`.

The metadata does **not** record the pose callable or the point-mass cloud.
Keep both alongside the result if you need to reproduce the physics.

## Applicability

- `fourier_series` applies to periodic motion only. Aliasing is limited by
  `tau_min` and by the finite sampling of the quadrature. Choose `tau_min`
  small enough for the highest frequency you need.
- For `fourier_transform`, the phase range must satisfy
  `f_max * |epoch - tref| <= 1e6` cycles. Place `tref` near the window to keep
  the phase accurate.
- `frequencies` for `fourier_transform` must be finite, nonnegative and
  strictly increasing, because the result's frequency axis requires this order.
  An unsorted or repeated sequence raises `ValueError`; sort it first.
- `tau_min` and `atol` are required. No physical scale is assumed.
- `tau_min` must not be larger than the narrowest structure of the response.
  The coarse grid and the initial partition are built from `tau_min`. If it is
  too large, the quadrature can still succeed and return a wrong value, because
  the narrow structure falls between the sample points. Choose `tau_min` no
  larger than that structure.
- `rtol` defaults to `1e-10`, `max_nodes` to `65536` and `max_evaluations`
  to `100000`.
- Only the computation frame is supported. Passing `output_frame` raises
  `ValueError`, and the evaluation view offers no output-frame change.

## Memory budget

`memory_budget_bytes` defaults to 64 MiB. It is a **per-lane** budget for the
arrays that the library manages. The value is not a limit on process memory
(RSS), and the lanes do not sum it. Temporary arrays created inside the pose
callable are outside this estimate.

## Gravitational constant and complex outputs

The gravitational constant `G` is applied exactly once, in the observable layer.
The kernels are G-free.

Complex Fourier coefficients are linear complex amplitudes or
transfer-function-like responses. They are not real-valued time-domain
gravitational fields, so do not read their real part as a time series of the
gravity field.
