import numpy as np

from gragra import WeightedPointSource
from gragra.coherent import directional_power_average, plane_wave_mass_weights
from gragra import coherent_point_acceleration
from gragra.waves import normalize_directions

# Mass model and plane-wave kinematics
source = WeightedPointSource(
    positions_m=[[0.5, 0.0, 0.0], [-0.5, 0.0, 0.0]],
    weights_kg=[10.0, 10.0],
)
directions = normalize_directions([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])  # (Ndir, 3)
wavenumber_rad_m = np.array([2 * np.pi * 10.0, 2 * np.pi * 50.0])  # (Nf,)

# Coherent complex mass weights: shape (N, Ndir, Nf)
weights = plane_wave_mass_weights(
    base_weights_kg=source.weights_kg,
    positions_m=source.positions_m,
    directions=directions,
    wavenumber_rad_m=wavenumber_rad_m,
)

# Coherent acceleration (complex valued): shape (M, Ndir, Nf, 3)
targets_m = np.array([[0.0, 0.0, 5.0]])
coherent_acc = coherent_point_acceleration(
    source_positions_m=source.positions_m,
    weights_kg=weights,
    targets_m=targets_m,
    softening_m=1e-6,
)
print("Coherent acceleration (complex):", coherent_acc[0, 0, 0])

# Directional power average over the direction axis: shape (M, Nf, 3)
direction_weights = np.ones(len(directions)) / len(directions)
power_avg = directional_power_average(
    response=coherent_acc,
    direction_weights=direction_weights,
    direction_axis=1,
)
