"""Golden tests for formula_to_latex conversion.

Tests conversion of verified symbol formulas from ResolvedEvaluation.formulas
into LaTeX representations.
"""

from __future__ import annotations

import pytest

from gragra._plotting._formula_latex import formula_to_latex

# Golden correspondence table for the 6 verified symbol formulas:
# Source: src/gragra/_evaluation_setup.py:492-527
GOLDEN_FORMULAS = [
    # (1) Discrete mass source - potential (src/gragra/_evaluation_setup.py:502)
    (
        "-G sum_j m_j/|r_j|",
        r"-G \sum_j \frac{m_j}{|\mathbf{r}_j|}",
    ),
    # (2) Discrete mass source - acceleration (src/gragra/_evaluation_setup.py:503)
    (
        "-G sum_j m_j r_j/|r_j|^3",
        r"-G \sum_j \frac{m_j \mathbf{r}_j}{|\mathbf{r}_j|^3}",
    ),
    # (3) Discrete mass source - gradient (src/gragra/_evaluation_setup.py:504)
    (
        "G sum_j m_j (3 r_j r_j^T/|r_j|^5-I/|r_j|^3)",
        (
            r"G \sum_j m_j \left( "
            r"\frac{3 \mathbf{r}_j \mathbf{r}_j^T}{|\mathbf{r}_j|^5} - "
            r"\frac{\mathbf{I}}{|\mathbf{r}_j|^3} \right)"
        ),
    ),
    # (4) Weighted source - potential (src/gragra/_evaluation_setup.py:512)
    (
        "-G sum_j w_j/|r_j|",
        r"-G \sum_j \frac{w_j}{|\mathbf{r}_j|}",
    ),
    # (5) Weighted source - acceleration (src/gragra/_evaluation_setup.py:513)
    (
        "-G sum_j w_j r_j/|r_j|^3",
        r"-G \sum_j \frac{w_j \mathbf{r}_j}{|\mathbf{r}_j|^3}",
    ),
    # (6) Weighted source - gradient (src/gragra/_evaluation_setup.py:514)
    (
        "G sum_j w_j (3 r_j r_j^T/|r_j|^5-I/|r_j|^3)",
        (
            r"G \sum_j w_j \left( "
            r"\frac{3 \mathbf{r}_j \mathbf{r}_j^T}{|\mathbf{r}_j|^5} - "
            r"\frac{\mathbf{I}}{|\mathbf{r}_j|^3} \right)"
        ),
    ),
]

UNSUPPORTED_PROSE_AND_EXPRESSIONS = [
    # Sphere branch prose (src/gragra/_evaluation_setup.py:496-498)
    "exterior -GM/r; interior -GM(3R^2-r^2)/(2R^3)",
    "exterior -GM r_vec/r^3; interior -GM r_vec/R^3",
    "exterior GM(3 rr^T/r^5-I/r^3); interior -GM I/R^3",
    # Analytic model notes
    "Existing spherical-shell solid-harmonic series",
    "Existing spherical-cavity solid-harmonic series",
    # Random prose or unsupported symbols
    "unknown formula expression",
    "",
    "   ",
]


@pytest.mark.parametrize("expression, expected_latex", GOLDEN_FORMULAS)
def test_golden_formula_latex_matches(expression: str, expected_latex: str) -> None:
    """Check that verified ASCII formulas match golden LaTeX representations."""
    actual_latex = formula_to_latex(expression)
    assert actual_latex == expected_latex


@pytest.mark.parametrize("unsupported_expression", UNSUPPORTED_PROSE_AND_EXPRESSIONS)
def test_unsupported_expressions_raise_value_error(unsupported_expression: str) -> None:
    """Check that unsupported prose or unknown formulas raise ValueError."""
    with pytest.raises(ValueError, match="Unsupported formula expression"):
        formula_to_latex(unsupported_expression)
