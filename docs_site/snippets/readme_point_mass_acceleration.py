import numpy as np

from gragra import PointMass, point_acceleration

# Single point mass: 1 kg at origin
mass = PointMass(mass_kg=1.0, position_m=[0.0, 0.0, 0.0])

# Calculate acceleration at target
target = [1.0, 0.0, 0.0]
acc = point_acceleration(mass.as_weighted_source(), [target])
print("Acceleration shape:", acc.shape)
print("Magnitude:", np.linalg.norm(acc[0]))
