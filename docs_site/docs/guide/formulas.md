# Formula LaTeX conversion

> This helper is present in the package but outside the documented public API
> scope for this release. Its interface carries no compatibility promise.

This guide demonstrates how to convert verified analytical formula strings into LaTeX markup using `gragra.plotting.formula_to_latex`.

## Overview

When models record analytical expressions (for instance, in evaluation metadata), they are stored as normalized ASCII formula strings. The `formula_to_latex` utility translates these verified expressions into LaTeX math strings suitable for display in notebooks, reports, or documentation.

Math rendering on the documentation site uses `pymdownx.arithmatex` with MathJax, following the `$..$` inline and `$$..$$` display math conventions.

## Supported formula expressions

Conversion is strictly restricted to verified closed-form symbol formulas. Arbitrary mathematical text, prose notes, or unverified expressions raise a `ValueError`.

The six verified canonical formulas currently supported are:

| Observable | Source representation | ASCII formula expression | Rendered LaTeX |
|---|---|---|---|
| Potential | Mass elements | `-G sum_j m_j/|r_j|` | $-G \sum_j \frac{m_j}{\lvert\mathbf{r}_j\rvert}$ |
| Acceleration | Mass elements | `-G sum_j m_j r_j/|r_j|^3` | $-G \sum_j \frac{m_j \mathbf{r}_j}{\lvert\mathbf{r}_j\rvert^3}$ |
| Gravity Gradient | Mass elements | `G sum_j m_j (3 r_j r_j^T/|r_j|^5-I/|r_j|^3)` | $G \sum_j m_j \left( \frac{3 \mathbf{r}_j \mathbf{r}_j^T}{\lvert\mathbf{r}_j\rvert^5} - \frac{\mathbf{I}}{\lvert\mathbf{r}_j\rvert^3} \right)$ |
| Potential | Weighted sources | `-G sum_j w_j/|r_j|` | $-G \sum_j \frac{w_j}{\lvert\mathbf{r}_j\rvert}$ |
| Acceleration | Weighted sources | `-G sum_j w_j r_j/|r_j|^3` | $-G \sum_j \frac{w_j \mathbf{r}_j}{\lvert\mathbf{r}_j\rvert^3}$ |
| Gravity Gradient | Weighted sources | `G sum_j w_j (3 r_j r_j^T/|r_j|^5-I/|r_j|^3)` | $G \sum_j w_j \left( \frac{3 \mathbf{r}_j \mathbf{r}_j^T}{\lvert\mathbf{r}_j\rvert^5} - \frac{\mathbf{I}}{\lvert\mathbf{r}_j\rvert^3} \right)$ |

## Example usage

```python
from gragra.plotting import formula_to_latex

# Convert acceleration formula for discrete mass elements
ascii_expr = "-G sum_j m_j r_j/|r_j|^3"
latex_math = formula_to_latex(ascii_expr)
print(latex_math)
# Output: -G \sum_j \frac{m_j \mathbf{r}_j}{|\mathbf{r}_j|^3}
```

## Error handling

If an unverified expression or prose string is passed to `formula_to_latex`, a `ValueError` is raised immediately:

```python
import pytest
from gragra.plotting import formula_to_latex

# Arbitrary text is rejected
try:
    formula_to_latex("Gravitational field of a point mass")
except ValueError as exc:
    assert "Unsupported formula expression" in str(exc)
```
