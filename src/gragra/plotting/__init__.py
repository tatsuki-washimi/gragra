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
    from gragra._plotting._dynamic_contribution import (
        ContributionFrequencyStatistic,
        ContributionRMSStatistics,
        ContributionStatistics,
        ContributionValidityMask,
        DirectionalContributionStatistics,
        contribution_statistics,
        directional_contribution_statistics,
        source_contributions,
    )
    from gragra._plotting._field import plot_field
    from gragra._plotting._formula_latex import formula_to_latex
    from gragra._plotting._hist import hist, plot_hist
    from gragra._plotting._map import plot_map
    from gragra._plotting._radial_histogram import (
        DynamicRadialHistogramResult,
        GravityGradientRadialHistogramResult,
        RadialHistogramResult,
        dynamic_radial_histogram,
        gravity_gradient_radial_histogram,
        radial_histogram,
    )
    from gragra._plotting._result_field import plot_result
    from gragra._plotting._setup_shapes import plot_shape
    from gragra._plotting._source_contracts import SourceContributionResult
    from gragra._plotting._source_field import (
        SourceFieldResult,
        source_field,
        source_field_result,
    )

__all__ = [
    "ContributionFrequencyStatistic",
    "ContributionRMSStatistics",
    "ContributionStatistics",
    "ContributionValidityMask",
    "DirectionalContributionStatistics",
    "DynamicRadialHistogramResult",
    "GravityGradientRadialHistogramResult",
    "RadialHistogramResult",
    "SourceContributionResult",
    "SourceFieldResult",
    "contribution_statistics",
    "directional_contribution_statistics",
    "dynamic_radial_histogram",
    "formula_to_latex",
    "gravity_gradient_radial_histogram",
    "hist",
    "plot_contribution_hist",
    "plot_cumulative_contribution",
    "plot_field",
    "plot_hist",
    "plot_map",
    "plot_result",
    "plot_shape",
    "radial_histogram",
    "source_contributions",
    "source_field",
    "source_field_result",
]

_LAZY_SOURCES = {
    "ContributionFrequencyStatistic": "gragra._plotting._dynamic_contribution",
    "ContributionRMSStatistics": "gragra._plotting._dynamic_contribution",
    "ContributionStatistics": "gragra._plotting._dynamic_contribution",
    "ContributionValidityMask": "gragra._plotting._dynamic_contribution",
    "DirectionalContributionStatistics": "gragra._plotting._dynamic_contribution",
    "DynamicRadialHistogramResult": "gragra._plotting._radial_histogram",
    "GravityGradientRadialHistogramResult": "gragra._plotting._radial_histogram",
    "RadialHistogramResult": "gragra._plotting._radial_histogram",
    "SourceContributionResult": "gragra._plotting._source_contracts",
    "SourceFieldResult": "gragra._plotting._source_field",
    "contribution_statistics": "gragra._plotting._dynamic_contribution",
    "directional_contribution_statistics": "gragra._plotting._dynamic_contribution",
    "dynamic_radial_histogram": "gragra._plotting._radial_histogram",
    "formula_to_latex": "gragra._plotting._formula_latex",
    "gravity_gradient_radial_histogram": "gragra._plotting._radial_histogram",
    "hist": "gragra._plotting._hist",
    "plot_contribution_hist": "gragra._plotting._contribution",
    "plot_cumulative_contribution": "gragra._plotting._contribution",
    "plot_field": "gragra._plotting._field",
    "plot_hist": "gragra._plotting._hist",
    "plot_map": "gragra._plotting._map",
    "plot_result": "gragra._plotting._result_field",
    "plot_shape": "gragra._plotting._setup_shapes",
    "radial_histogram": "gragra._plotting._radial_histogram",
    "source_contributions": "gragra._plotting._dynamic_contribution",
    "source_field": "gragra._plotting._source_field",
    "source_field_result": "gragra._plotting._source_field",
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
