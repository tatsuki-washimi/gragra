import numpy as np
import pytest

from gragra.geometry import (
    HalfCylinder,
    HalfSphere,
    MeshResolution,
    PrimitiveSurfaceMesh,
)
from gragra.geometry.primitives import Cuboid, Cylinder, Sphere


# Test cases for the 5 shapes
def get_shapes():
    return [
        Sphere(
            center_m=[1.0, -2.0, 3.0], radius_m=2.5, density_kg_m3=2700.0, spacing_m=0.2
        ),
        Cuboid(
            center_m=[-1.0, 2.0, -0.5],
            lengths_m=[2.0, 3.0, 4.0],
            density_kg_m3=2700.0,
            spacing_m=0.2,
        ),
        Cylinder(
            center_m=[0.5, 0.5, 0.5],
            radius_m=1.8,
            height_m=4.5,
            axis="z",
            density_kg_m3=2700.0,
            spacing_m=0.2,
        ),
        Cylinder(
            center_m=[0.5, 0.5, 0.5],
            radius_m=1.8,
            height_m=4.5,
            axis="x",
            density_kg_m3=2700.0,
            spacing_m=0.2,
        ),
        Cylinder(
            center_m=[0.5, 0.5, 0.5],
            radius_m=1.8,
            height_m=4.5,
            axis="y",
            density_kg_m3=2700.0,
            spacing_m=0.2,
        ),
        HalfSphere(
            center_m=[2.0, 1.0, -1.0], radius_m=2.2, density_kg_m3=2700.0, spacing_m=0.2
        ),
        HalfCylinder(
            center_m=[-1.0, -1.0, 0.0],
            radius_m=1.5,
            length_m=3.5,
            density_kg_m3=2700.0,
            spacing_m=0.2,
        ),
    ]


def get_interior_point(shape):
    """Retrieve the interior reference point for a given geometry primitive."""
    if isinstance(shape, (Sphere, Cuboid, Cylinder)):
        return shape.center_m
    if isinstance(shape, HalfSphere):
        return shape.center_m + np.array([0.0, 0.0, shape.radius_m / 2.0])
    if isinstance(shape, HalfCylinder):
        return shape.center_m + np.array([0.0, 0.0, shape.radius_m / 4.0])
    return shape.center_m


@pytest.mark.parametrize("shape", get_shapes())
def test_divergence_theorem_gate_1a(shape):
    """Verify volume computed via divergence theorem is within <0.5%."""
    res = MeshResolution()
    mesh = shape.surface_mesh(res)

    # Volume via divergence theorem: (1/3) * sum( (centroids - ip) · normals * areas )
    # using a vector field F = x - ip, where div(F) = 3.
    ip = get_interior_point(shape)

    local_centroids = mesh.centroids_m - ip
    dot_products = np.sum(local_centroids * mesh.normals, axis=1)
    div_volume = (1.0 / 3.0) * np.sum(dot_products * mesh.areas_m2)
    analytic_vol = shape.analytic_volume_m3

    rel_err = abs(div_volume - analytic_vol) / analytic_vol
    assert rel_err < 0.005, (
        f"Divergence volume err {rel_err:.5f} exceeds 0.5% for {shape}"
    )


@pytest.mark.parametrize("shape", get_shapes())
def test_closed_surface_identity(shape):
    """Verify that sum( normals * areas ) is approximately 0."""
    res = MeshResolution()
    mesh = shape.surface_mesh(res)

    flux = np.sum(mesh.normals * mesh.areas_m2[:, np.newaxis], axis=0)
    total_area = shape.analytic_area_m2
    atol = 1e-12 * total_area

    np.testing.assert_allclose(
        flux, 0.0, atol=atol, err_msg=f"Closed surface identity failed for {shape}"
    )


@pytest.mark.parametrize("shape", get_shapes())
def test_outward_normals(shape):
    """Verify that normals point outward by checking offset points.

    For each cell, moving along the normal should go outside the shape,
    while moving against the normal should go inside the shape.
    """
    res = MeshResolution()
    mesh = shape.surface_mesh(res)

    # eps is chosen to be larger than sagitta (approx 0.002 * R at resolution 50)
    # and smaller than cell half-width (approx 0.06 * R), ensuring the offset
    # point is strictly inside/outside without crossing adjacent faces.
    eps = 1e-6 if isinstance(shape, Cuboid) else 0.01 * shape.radius_m

    pts_out = mesh.centroids_m + eps * mesh.normals
    pts_in = mesh.centroids_m - eps * mesh.normals

    is_out = shape.contains(pts_out)
    is_in = shape.contains(pts_in)

    # All inward offset points must be inside the shape
    assert is_in.all(), f"Some inward points are outside for {shape}"
    # All outward offset points must be outside the shape
    assert not is_out.any(), f"Some outward points are inside for {shape}"


@pytest.mark.parametrize("shape", get_shapes())
def test_area_consistency(shape):
    """Verify that sum( areas ) is within <0.5% of analytic area."""
    res = MeshResolution()
    mesh = shape.surface_mesh(res)

    computed_area = np.sum(mesh.areas_m2)
    analytic_area = shape.analytic_area_m2

    rel_err = abs(computed_area - analytic_area) / analytic_area
    assert rel_err < 0.005, (
        f"Area relative error {rel_err:.5f} exceeds 0.5% for {shape}"
    )


@pytest.mark.parametrize("shape", get_shapes())
def test_resolution_convergence(shape):
    """Verify resolution convergence (n=50 error vs n=25 error ratio > 2.5)."""
    # Cuboid is flat-faced, error is 0, so skip it for convergence test
    if isinstance(shape, Cuboid):
        pytest.skip(
            "Cuboid is flat-faced with zero volume error, skip convergence check."
        )

    # Define convergence helper for a given resolution
    def get_volume_err(n):
        res = MeshResolution(n_u=n, n_v=n, n_cap=n)
        mesh = shape.surface_mesh(res)
        ip = get_interior_point(shape)

        local_centroids = mesh.centroids_m - ip
        dot_products = np.sum(local_centroids * mesh.normals, axis=1)
        div_volume = (1.0 / 3.0) * np.sum(dot_products * mesh.areas_m2)
        return abs(div_volume - shape.analytic_volume_m3)

    err_25 = get_volume_err(25)
    err_50 = get_volume_err(50)

    # Avoid division by very small errors
    if err_50 < 1e-12:
        pytest.skip(
            "Volume error at resolution 50 is too close to numerical precision limit."
        )

    ratio = err_25 / err_50
    assert ratio > 2.5, f"Convergence ratio {ratio:.2f} is not > 2.5 for {shape}"


@pytest.mark.parametrize("shape", get_shapes())
def test_readonly_and_frozen(shape):
    """Verify that PrimitiveSurfaceMesh fields are read-only and frozen."""
    res = MeshResolution()
    mesh = shape.surface_mesh(res)

    assert isinstance(mesh, PrimitiveSurfaceMesh)
    assert not mesh.vertices_m.flags.writeable
    assert not mesh.triangles.flags.writeable
    assert not mesh.centroids_m.flags.writeable
    assert not mesh.normals.flags.writeable
    assert not mesh.areas_m2.flags.writeable

    with pytest.raises(AttributeError):
        mesh.vertices_m = np.zeros((3, 3))
