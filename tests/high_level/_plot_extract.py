"""Artist data extraction utilities for plot value verification.

Extracts underlying numerical data (coordinates, scalar values, histogram bins,
and polygon vertices) directly from Matplotlib artists without pixel or image
comparisons.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from matplotlib.axes import Axes


def extract_step_histogram(ax: Axes) -> tuple[np.ndarray, np.ndarray]:
    """Extract (values, edges) from a StepPatch in ax.patches.

    Parameters
    ----------
    ax : Axes
        Axes containing a StepPatch (e.g. from ax.stairs).

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        (values, edges) arrays representing the histogram.
    """
    for patch in ax.patches:
        if hasattr(patch, "get_data"):
            data = patch.get_data()
            return np.asarray(data.values, dtype=np.float64), np.asarray(
                data.edges, dtype=np.float64
            )
    raise ValueError("No StepPatch with get_data() found in ax.patches")


def extract_scatter_values(ax: Axes) -> tuple[np.ndarray, np.ndarray]:
    """Extract coordinates and scalar color values from scatter collections.

    Parameters
    ----------
    ax : Axes
        Axes containing a PathCollection (e.g. from ax.scatter).

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        (coords, values) where coords is (N, D) and values is (N,).
    """
    if not ax.collections:
        raise ValueError("No collections found in ax.collections")

    coll = ax.collections[0]
    vals = coll.get_array()
    if vals is not None:
        values = np.asarray(vals, dtype=np.float64)
    else:
        values = np.array([], dtype=np.float64)

    # 3D scatter
    if hasattr(coll, "_offsets3d"):
        x, y, z = coll._offsets3d
        coords = np.column_stack([np.asarray(x), np.asarray(y), np.asarray(z)])
    else:
        # 2D scatter
        offsets = coll.get_offsets()
        coords = np.asarray(offsets, dtype=np.float64)

    return coords, values


def extract_line_data(ax: Axes, line_index: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Extract (x, y) data from a Line2D in ax.lines.

    Parameters
    ----------
    ax : Axes
        Axes containing lines.
    line_index : int, default=0
        Index of the line artist.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        (x_data, y_data) arrays.
    """
    if not ax.lines or line_index >= len(ax.lines):
        raise ValueError(f"Line index {line_index} not found in ax.lines")
    line = ax.lines[line_index]
    return (
        np.asarray(line.get_xdata(), dtype=np.float64),
        np.asarray(line.get_ydata(), dtype=np.float64),
    )


def extract_3d_faces(ax: Axes) -> list[np.ndarray]:
    """Extract 3D polygon vertex arrays from Poly3DCollection artists.

    Parameters
    ----------
    ax : Axes
        Axes containing Poly3DCollection artists.

    Returns
    -------
    list[np.ndarray]
        List of vertex arrays, each of shape (V_face, 3).
    """
    all_faces: list[np.ndarray] = []
    for coll in ax.collections:
        faces = getattr(coll, "_faces", None)
        if faces is not None:
            for face in faces:
                all_faces.append(np.asarray(face, dtype=np.float64))
    if not all_faces:
        raise ValueError("No Poly3DCollection with _faces found in ax.collections")
    return all_faces


def extract_image_data(ax: Axes) -> np.ndarray:
    """Extract 2D scalar array from AxesImage in ax.images.

    Parameters
    ----------
    ax : Axes
        Axes containing an AxesImage.

    Returns
    -------
    np.ndarray
        2D array of image values.
    """
    if not ax.images:
        raise ValueError("No AxesImage found in ax.images")
    return np.asarray(ax.images[0].get_array(), dtype=np.float64)
