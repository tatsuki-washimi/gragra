"""Small legacy-VTK fixtures for the bounded OpenFOAM profile."""

from pathlib import Path


def write_two_hex(
    path: Path,
    *,
    density=(-1.0, 2.0),
    centres=None,
    ids=(0, 1),
    extra: bool = False,
    x_offset: float = 0.0,
    scalar_centres: bool = False,
) -> Path:
    if centres is None:
        centres = ((0.5 + x_offset, 0.5, 0.5), (1.5 + x_offset, 0.5, 0.5))
    points = "\n".join(
        f"{x} {y} {z}"
        for x, y, z in (
            (0, 0, 0),
            (1, 0, 0),
            (1, 1, 0),
            (0, 1, 0),
            (0, 0, 1),
            (1, 0, 1),
            (1, 1, 1),
            (0, 1, 1),
            (2, 0, 0),
            (2, 1, 0),
            (2, 0, 1),
            (2, 1, 1),
        )
        for x in (x + x_offset,)
    )
    cells = "0 1 2 3 4 5 6 7\n1 8 9 2 5 10 11 6"

    def scalar(name, values, kind="float"):
        return f"SCALARS {name} {kind} 1\nLOOKUP_TABLE default\n" + "\n".join(
            map(str, values)
        )

    if scalar_centres:
        c = "SCALARS C float 1\nLOOKUP_TABLE default\n0.5\n1.5"
    else:
        c = "VECTORS C float\n" + "\n".join(
            " ".join(map(str, row)) for row in centres
        )
    text = "# vtk DataFile Version 3.0\nOpenFOAM\nASCII\nDATASET UNSTRUCTURED_GRID\n"
    text += (
        f"POINTS 12 float\n{points}\nCELLS 2 18\n8 {cells.splitlines()[0]}\n"
        f"8 {cells.splitlines()[1]}\nCELL_TYPES 2\n12\n12\nCELL_DATA 2\n"
    )
    if extra:
        text += scalar("p", (1, 2)) + "\n" + scalar("p_rgh", (3, 4)) + "\n"
        text += "VECTORS U float\n0 0 0\n1 2 3\n" + scalar("T", (280, 281)) + "\n"
    text += (
        scalar("cellID", ids, "int")
        + "\n"
        + scalar("rho", (999, 1002))
        + "\n"
        + scalar("delta_rho", density)
        + "\n"
        + c
        + "\n"
        + scalar("Vc", (1, 1))
        + "\n"
    )
    path.write_text(text, encoding="ascii")
    return path
