#!/usr/bin/env python3
"""Build the exporter partition map from the new run's rank-local VTK meshes."""

from pathlib import Path
import re

CASE = Path("/home/washimi/work/specfem3d-rayleigh-init-20260907/research_cases/rayleigh_b5_200m_12km_8rank_dt_half")
DB = CASE / "OUTPUT_FILES/DATABASES_MPI_200M_DT_HALF"
OUT = DB / "part_array.vtk"


def read_vtk(path: Path):
    lines = path.read_text().splitlines()
    p = next(i for i, line in enumerate(lines) if line.startswith("POINTS "))
    npoint = int(lines[p].split()[1])
    vals = []
    i = p + 1
    while len(vals) < 3 * npoint:
        vals.extend(float(v) for v in lines[i].split())
        i += 1
    points = [tuple(vals[j:j + 3]) for j in range(0, len(vals), 3)]
    c = next(i for i, line in enumerate(lines) if line.startswith("CELLS "))
    ncells = int(lines[c].split()[1])
    cells = []
    for i in range(c + 1, c + 1 + ncells):
        row = [int(v) for v in lines[i].split()]
        if row[0] != 8:
            raise ValueError(f"{path}: non-hexahedral cell")
        cells.append(tuple(row[1:]))
    return points, cells


def main() -> None:
    point_by_xyz = {}
    elements = {}
    for rank in range(8):
        points, cells = read_vtk(DB / f"proc{rank:06d}_mesh.vtk")
        for cell in cells:
            xyz = tuple(sorted(points[node] for node in cell))
            if xyz in elements and elements[xyz] != rank:
                raise ValueError("element assigned to multiple ranks")
            elements[xyz] = rank
            for point in xyz:
                point_by_xyz.setdefault(point, len(point_by_xyz))
    ordered = sorted(elements)
    cells_global = [[point_by_xyz[point] for point in xyz] for xyz in ordered]
    OUT.write_text(
        "# vtk DataFile Version 3.1\n"
        "part array reconstructed from new rank-local meshes\n"
        "ASCII\nDATASET UNSTRUCTURED_GRID\n"
        f"POINTS {len(point_by_xyz)} float\n"
        + "".join(f"{x:.12g} {y:.12g} {z:.12g}\n" for x, y, z in
                  sorted(point_by_xyz, key=point_by_xyz.get))
        + f"CELLS {len(cells_global)} {9 * len(cells_global)}\n"
        + "".join("8 " + " ".join(map(str, cell)) + "\n" for cell in cells_global)
        + f"CELL_TYPES {len(cells_global)}\n"
        + "".join("12\n" for _ in cells_global)
        + f"CELL_DATA {len(cells_global)}\nSCALARS elem_flag int 1\nLOOKUP_TABLE default\n"
        + "".join(f"{elements[xyz]}\n" for xyz in ordered)
    )
    print(f"{OUT} elements={len(cells_global)} points={len(point_by_xyz)}")


if __name__ == "__main__":
    main()
