# Water Newtonian noise models

This guide describes the analytical models implemented in `gragra` for estimating Newtonian noise (NN) induced by water flow near gravitational-wave detector test masses.

Water flow in drainage channels, pipes, and natural cascades produces mass redistribution that couples gravitationally to detector test masses. Several idealized analytical configurations provide rapid and reliable benchmarks.

## Physical models

### 1. Frozen advection in narrow channels

Consider a straight channel of width $w$ running along the $x$-axis, with mean water surface at $z = 0$. A test mass is located at $(0, 0, D)$, directly above the channel at perpendicular distance $D > 0$. Water flows in the $+x$ direction with mean flow speed $v$.

Surface displacement perturbations $\eta(x, t) = \eta_0 \exp(-i \Omega (t - x / v))$ advect along the channel with wavenumber $k = \Omega / v = 2\pi f / v$. The resulting gravitational potential, acceleration, and gravity-gradient perturbations are expressed in closed form using modified Bessel functions of the second kind ($K_0$ and $K_1$):

$$
\frac{T_{\Phi\eta}(2\pi f)}{G} = -2 \rho w K_0\left(\frac{2\pi f D}{v}\right)
$$

$$
\frac{T_{a_x\eta}(2\pi f)}{G} = 2i \rho w \left(\frac{2\pi f}{v}\right) K_0\left(\frac{2\pi f D}{v}\right)
$$

$$
\frac{T_{a_z\eta}(2\pi f)}{G} = -2 \rho w \left(\frac{2\pi f}{v}\right) K_1\left(\frac{2\pi f D}{v}\right)
$$

The full gravity gradient tensor satisfies the Laplace vacuum condition $\operatorname{Tr}(\mathbf{\Gamma}) = \Gamma_{xx} + \Gamma_{yy} + \Gamma_{zz} = 0$.

### 2. Vertical waterfall

A free-falling water stream flows vertically along the $z$-axis with velocity $v$, with a test mass located at $(D, 0, 0)$ horizontally. Cross-sectional area perturbations $A(z, t) = A_0 \exp(-i\Omega(t - z/v))$ produce horizontal and vertical gravitational accelerations:

$$
\frac{T_{a_x A}(2\pi f)}{G} = -2 \rho \left(\frac{2\pi f}{v}\right) K_1\left(\frac{2\pi f D}{v}\right)
$$

$$
\frac{T_{a_z A}(2\pi f)}{G} = 2i \rho \left(\frac{2\pi f}{v}\right) K_0\left(\frac{2\pi f D}{v}\right)
$$

### 3. Spatially uncorrelated surface cells

When surface waves are turbulent and incoherently distributed across small cells of area $A_{\mathrm{cell}}$, the gravitational acceleration ASD is computed by integrating uncorrelated cell contributions across the channel length:

$$
\int_{-\infty}^\infty \frac{x^2}{(D^2 + x^2)^3} dx = \frac{\pi}{8 D^3}
$$

> [!NOTE] Model Provenance and Derivation Status
> - **`INDEPENDENT_DERIVATION`**: The narrow-channel uncorrelated cell model, geometric coupling integral $\pi / (8 D^3)$, and linear width scaling implemented in `gragra` are an independent analytical derivation from first principles.
> - **`ORIGINAL_SOURCE_UNVERIFIED`**: Historical presentation formulas from external workshops (e.g. Chen 2016 notes / Somiya TAUP 2019 slides) remain unverified primary sources; `gragra` does not assert identity with unverified external notebooks.

### 4. Channel inflow edge step

A sudden step change in water volume $\Delta V$ at the channel entrance produces a transient Heaviside response $\Theta(t)$. The resulting deterministic Fourier spectrum of acceleration is given by:

$$
\frac{\Delta a_x(2\pi f)}{G} = \rho \Delta V \frac{x_0}{(D^2 + x_0^2)^{3/2}} \frac{i}{2\pi f}
$$

This is a deterministic transient Fourier spectrum with units $(\mathrm{m/s^2})\cdot\mathrm{s} = \mathrm{m/s}$, distinct from a stationary spectral density.

## Usage example

```python
import numpy as np
from gragra.observables import (
    water_channel_advection_asd,
    water_channel_advection_strain_asd,
)

# Frequency grid from 1 to 50 Hz
freqs = np.linspace(1.0, 50.0, 50)

# Physical parameters: distance 2 m, width 0.5 m, flow velocity 1.5 m/s
distance_m = 2.0
width_m = 0.5
velocity_mps = 1.5
arm_length_m = 3000.0

# Surface displacement ASD: 1 mm / sqrt(Hz) white noise
surface_asd = np.full_like(freqs, 1e-3)

# Compute acceleration ASD (m/s^2 / sqrt(Hz)) and equivalent strain ASD (1 / sqrt(Hz))
acc_asd = water_channel_advection_asd(
    frequency_hz=freqs,
    velocity_mps=velocity_mps,
    distance_m=distance_m,
    width_m=width_m,
    surface_asd_m_per_rt_hz=surface_asd,
)
strain_asd = water_channel_advection_strain_asd(
    frequency_hz=freqs,
    velocity_mps=velocity_mps,
    distance_m=distance_m,
    width_m=width_m,
    surface_asd_m_per_rt_hz=surface_asd,
    arm_length_m=arm_length_m,
)
```

## Low-level G-free functions

For custom integrations or dimensionless scaling, the low-level functions in `gragra.analytic` return transfer coefficients with $G$ divided out. To obtain physical SI quantities, multiply by $G_{\mathrm{SI}} \approx 6.6743 \times 10^{-11} \ \mathrm{m^3 / (kg \cdot s^2)}$.
