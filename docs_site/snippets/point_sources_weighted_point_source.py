from gragra import WeightedPointSource

source = WeightedPointSource(
    positions_m=[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
    weights_kg=[2.0, -1.5],
)
assert source.positions_m.shape == (2, 3)
assert source.weights_kg[0] == 2.0
assert source.weights_kg[1] == -1.5
