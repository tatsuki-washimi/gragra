"""Explicit finite material regions and Q1 hex8 source quadratures (SI).

The exterior is explicitly vacuum or defined contrast background. Background density
is cell-constant and non-negative; bulk perturbations may be signed or complex.
A hex's nodal displacement is interpolated via Q1 shape functions.
"""

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

import numpy as np

from gragra._arrays import as_xyz_table
from gragra.fields.evaluation import _density, _displacement, _frame, _source_ids
from gragra.fields.finite_body_p1 import piecewise_tetra_p1_density_sources
from gragra.fields.hex8_geometry import (
    check_hex8_jacobians,
    hex8_gauss_rule,
    hex8_jacobians,
    hex8_shape_and_derivative,
)
from gragra.fields.unstructured import UnstructuredDisplacementField
from gragra.gravity_sources import VolumeElementSource


def _readonly(value, dtype=None):
    result = np.array(value, dtype=dtype, copy=True)
    result.setflags(write=False)
    return result


_rule = hex8_gauss_rule
_shape = hex8_shape_and_derivative


@dataclass(frozen=True, init=False)
class MaterialRegions:
    """Finite cell-constant background with explicit cell/region identities.

    ``region_ids`` is in ``cell_ids`` order. The density map must contain
    exactly the used integer region IDs. Zero-density cells represent cavities.
    """

    cell_ids: tuple[str, ...]
    region_ids: np.ndarray
    density_by_region_kg_m3: Mapping
    cell_density_kg_m3: np.ndarray
    exterior_density_kg_m3: float
    coordinate_frame: str
    unit_system: str
    mode: str  # "finite_domain_total" or "background_contrast"

    def __init__(
        self,
        cell_ids,
        region_ids,
        density_by_region_kg_m3,
        *,
        exterior_density_kg_m3=0.0,
        mode: str = "finite_domain_total",
        coordinate_frame="world",
        unit_system="SI",
    ):
        regions = np.asarray(region_ids)
        if regions.ndim != 1 or regions.dtype.kind not in "iu":
            raise TypeError("region_ids must be a one-dimensional integer array")
        if np.any(regions > np.iinfo(np.int64).max):
            raise ValueError("region_ids must fit the stored int64 representation")
        ids = _source_ids(cell_ids, len(regions))
        if not isinstance(density_by_region_kg_m3, Mapping):
            raise TypeError("density_by_region_kg_m3 must be a mapping")
        densities = {}
        for key, value in density_by_region_kg_m3.items():
            if isinstance(key, (bool, np.bool_)) or not isinstance(
                key, (int, np.integer)
            ):
                raise TypeError("region density keys must be integer IDs")
            densities[int(key)] = _density(value)
        used = set(map(int, regions))
        if used != set(densities):
            raise ValueError(
                "region mapping mismatch: "
                f"missing={sorted(used - set(densities))}, "
                f"unused={sorted(set(densities) - used)}"
            )
        exterior = _density(exterior_density_kg_m3)
        if mode not in {"finite_domain_total", "background_contrast"}:
            raise ValueError(
                f"mode must be 'finite_domain_total' or 'background_contrast', "
                f"got '{mode}'"
            )
        if mode == "finite_domain_total" and exterior != 0.0:
            raise ValueError(
                "exterior_density_kg_m3 must explicitly be zero "
                "(vacuum) in finite_domain_total mode"
            )
        if unit_system != "SI":
            raise ValueError("material regions require unit_system='SI'")

        cell_densities = np.array(
            [densities[int(k)] for k in regions], dtype=np.float64
        )

        for name, value in {
            "cell_ids": ids,
            "region_ids": _readonly(regions, np.int64),
            "density_by_region_kg_m3": MappingProxyType(densities),
            "cell_density_kg_m3": _readonly(cell_densities),
            "exterior_density_kg_m3": exterior,
            "coordinate_frame": _frame(coordinate_frame),
            "unit_system": unit_system,
            "mode": mode,
        }.items():
            object.__setattr__(self, name, value)

    def check_mesh(self, cell_ids, coordinate_frame, unit_system="SI"):
        """Reject reordered, missing, duplicate IDs or mixed frame/units."""
        if _source_ids(cell_ids, len(self.cell_ids)) != self.cell_ids:
            raise ValueError("cell_ids must match material cell_ids in the same order")
        if coordinate_frame != self.coordinate_frame:
            raise ValueError("coordinate frame does not match material regions")
        if unit_system != self.unit_system:
            raise ValueError("unit_system must match material SI units")

    def tetra_sources(
        self,
        field,
        *,
        cell_ids,
        coordinate_frame="world",
        unit_system="SI",
        quadrature_order=None,
    ):
        """Reuse P1 bulk/interface topology: jump=rho_negative-rho_positive."""
        if not isinstance(field, UnstructuredDisplacementField):
            raise TypeError("field must be an UnstructuredDisplacementField")
        self.check_mesh(cell_ids, coordinate_frame, unit_system)
        if len(field.tetra) != len(self.cell_ids):
            raise ValueError("cell_ids count must match tetra cells")
        density = np.repeat(self.cell_density_kg_m3[:, None], 4, axis=1)
        return piecewise_tetra_p1_density_sources(
            field, density, quadrature_order=quadrature_order
        )


@dataclass(frozen=True, init=False)
class Hex8Sources:
    """Q1 Gaussian volume source views, built only by a validated hex field.

    ``bulk_source`` omits interfaces. ``mass_kg, displacement_m`` represents
    complete linearized finite-body mass motion, including those interfaces,
    through the existing displacement integrator. Do not add the two views.
    """

    positions_m: np.ndarray
    volumes_m3: np.ndarray
    mass_kg: np.ndarray
    displacement_m: np.ndarray
    bulk_source: VolumeElementSource
    cell_index: np.ndarray
    source_ids: tuple[str, ...]

    def __init__(self):
        raise TypeError("Hex8Sources is builder-only")


@dataclass(frozen=True, init=False)
class Hex8DisplacementField:
    """Isoparametric Q1 displacement on hex8 cells with VTK node order.

    Jacobians are checked at 27 reference lattice points and at every selected
    quadrature point. Coordinates and displacement use metres.
    """

    vertices_m: np.ndarray
    hex8: np.ndarray
    displacement_m: np.ndarray
    coordinate_frame: str

    def __init__(self, vertices_m, hex8, displacement_m, *, coordinate_frame="world"):
        vertices = as_xyz_table("vertices_m", vertices_m)
        connectivity = np.asarray(hex8)
        if connectivity.ndim != 2 or connectivity.shape[1] != 8:
            raise ValueError("hex8 must have shape (Nc,8)")
        if connectivity.dtype.kind not in "iu":
            raise TypeError("hex8 must have integer dtype")
        if len(connectivity) == 0:
            raise ValueError("hex8 must contain at least one cell")
        if np.any(connectivity < 0) or np.any(connectivity >= len(vertices)):
            raise ValueError("hex8 contains an invalid node index")
        if any(len(set(row)) != 8 for row in connectivity):
            raise ValueError("hex8 Jacobian is degenerate: repeated nodes")
        if len({tuple(sorted(row)) for row in connectivity}) != len(connectivity):
            raise ValueError("hex8 contains duplicate cells")

        # 27-point Jacobian positivity check
        check_hex8_jacobians(vertices, connectivity)

        object.__setattr__(self, "vertices_m", vertices)
        object.__setattr__(self, "hex8", _readonly(connectivity, np.int64))
        object.__setattr__(
            self, "displacement_m", _displacement(displacement_m, len(vertices))
        )
        object.__setattr__(self, "coordinate_frame", _frame(coordinate_frame))

    def sources(self, materials, *, quadrature_order=3, cell_ids=None):
        """Build Gaussian mass-motion and bulk-density views (G-free)."""
        if not isinstance(materials, MaterialRegions):
            raise TypeError("materials must be MaterialRegions")
        materials.check_mesh(
            materials.cell_ids if cell_ids is None else cell_ids, self.coordinate_frame
        )
        if len(materials.cell_ids) != len(self.hex8):
            raise ValueError("material cell_ids count must match hex8 cells")

        points, weights = hex8_gauss_rule(quadrature_order)
        shape, derivative = hex8_shape_and_derivative(points)
        nodes = self.vertices_m[self.hex8]
        jac, det = hex8_jacobians(self.vertices_m, self.hex8, derivative)

        # grad_x N = grad_reference N @ J^-1
        gradients = np.einsum("qnj,cqji->cqni", derivative, np.linalg.inv(jac))
        displacement_nodes = self.displacement_m[self.hex8]
        divergence = np.einsum("cni,cqni->cq", displacement_nodes, gradients)
        positions = np.einsum("qn,cni->cqi", shape, nodes)
        displacement = np.einsum("qn,cni->cqi", shape, displacement_nodes)
        volume = det * weights
        density = materials.cell_density_kg_m3[:, None]
        bulk = (-density * divergence).reshape(-1)

        result = object.__new__(Hex8Sources)
        for name, value in {
            "positions_m": _readonly(positions.reshape(-1, 3)),
            "volumes_m3": _readonly(volume.reshape(-1)),
            "mass_kg": _readonly((density * volume).reshape(-1)),
            "displacement_m": _readonly(displacement.reshape(-1, 3)),
            "bulk_source": VolumeElementSource(
                positions.reshape(-1, 3), volume.reshape(-1), bulk
            ),
            "cell_index": _readonly(np.repeat(np.arange(len(nodes)), len(points))),
            "source_ids": tuple(
                f"{len(cid)}:{cid}:q{q}"
                for cid in materials.cell_ids
                for q in range(len(points))
            ),
        }.items():
            object.__setattr__(result, name, value)
        return result
