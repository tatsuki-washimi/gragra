"""Public high-level plotting and source-diagnostic API.

Names are resolved on first access (PEP 562), so importing this package does
not load Matplotlib or GWexpy.
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from gragra._plotting._contribution import (
        plot_contribution_hist,
        plot_cumulative_contribution,
    )
    from gragra._plotting._field import plot_field
    from gragra._plotting._formula_latex import formula_to_latex
    from gragra._plotting._hist import hist, plot_hist
    from gragra._plotting._map import plot_map
    from gragra._plotting._result_field import plot_result
    from gragra._plotting._setup_shapes import plot_shape

__all__ = [
    "formula_to_latex",
    "hist",
    "plot_contribution_hist",
    "plot_cumulative_contribution",
    "plot_field",
    "plot_hist",
    "plot_map",
    "plot_result",
    "plot_shape",
]

_LAZY_SOURCES = {
    "formula_to_latex": "gragra._plotting._formula_latex",
    "hist": "gragra._plotting._hist",
    "plot_contribution_hist": "gragra._plotting._contribution",
    "plot_cumulative_contribution": "gragra._plotting._contribution",
    "plot_field": "gragra._plotting._field",
    "plot_hist": "gragra._plotting._hist",
    "plot_map": "gragra._plotting._map",
    "plot_result": "gragra._plotting._result_field",
    "plot_shape": "gragra._plotting._setup_shapes",
}


def __getattr__(name: str) -> object:
    module_name = _LAZY_SOURCES.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(importlib.import_module(module_name), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
