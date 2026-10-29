"""Display converter from verified ASCII formula expressions to LaTeX representations.

Provides formula_to_latex for turning verified ASCII formulas stored in
ResolvedEvaluation.formulas into LaTeX markup suitable for documentation and plotting.
"""

from __future__ import annotations

_VERIFIED_FORMULA_LATEX_MAP: dict[str, str] = {
    # Discrete mass source (m_j)
    "-G sum_j m_j/|r_j|": r"-G \sum_j \frac{m_j}{|\mathbf{r}_j|}",
    "-G sum_j m_j r_j/|r_j|^3": (
        r"-G \sum_j \frac{m_j \mathbf{r}_j}{|\mathbf{r}_j|^3}"
    ),
    "G sum_j m_j (3 r_j r_j^T/|r_j|^5-I/|r_j|^3)": (
        r"G \sum_j m_j \left( "
        r"\frac{3 \mathbf{r}_j \mathbf{r}_j^T}{|\mathbf{r}_j|^5} - "
        r"\frac{\mathbf{I}}{|\mathbf{r}_j|^3} \right)"
    ),
    # Weighted static source (w_j)
    "-G sum_j w_j/|r_j|": r"-G \sum_j \frac{w_j}{|\mathbf{r}_j|}",
    "-G sum_j w_j r_j/|r_j|^3": (
        r"-G \sum_j \frac{w_j \mathbf{r}_j}{|\mathbf{r}_j|^3}"
    ),
    "G sum_j w_j (3 r_j r_j^T/|r_j|^5-I/|r_j|^3)": (
        r"G \sum_j w_j \left( "
        r"\frac{3 \mathbf{r}_j \mathbf{r}_j^T}{|\mathbf{r}_j|^5} - "
        r"\frac{\mathbf{I}}{|\mathbf{r}_j|^3} \right)"
    ),
}


def formula_to_latex(expression: str) -> str:
    """Convert a verified ASCII formula expression into its LaTeX representation.

    Parameters
    ----------
    expression : str
        The ASCII formula string, exactly as recorded in ResolvedEvaluation.formulas.

    Returns
    -------
    str
        The LaTeX representation.

    Raises
    ------
    ValueError
        If the expression is not one of the verified symbol formula expressions,
        such as model notes, prose descriptions, or unknown formulas.
    """
    if isinstance(expression, str) and expression in _VERIFIED_FORMULA_LATEX_MAP:
        return _VERIFIED_FORMULA_LATEX_MAP[expression]
    raise ValueError(
        f"Unsupported formula expression: {expression!r}. "
        "Only verified symbol formula expressions from "
        "ResolvedEvaluation.formulas are supported."
    )

