# Executable documentation snippets

Every Python code example shown on the documentation site lives here as a
standalone, runnable `.py` file and is embedded into a page with a
`pymdownx.snippets` include:

    ```python
    --8<-- "snippets/observables_gradient_fixed_point.py"
    ```

`tests/test_docs_snippets.py` executes each file in this directory and fails
the suite if a page contains a hand-written Python block that is not one of
these files. This directory is **not** part of `docs_dir`, so nothing here is
copied into the built site; only the include is.

## Rules for a snippet file

1. **Whole-file includes only.** Do not use `pymdownx.snippets` section
   markers (`# --8<-- [start:...]`). A section include shows a strict subset
   of what actually runs, so a reader who copies the visible code gets a
   `NameError` on the imports that were left outside the section. Each file
   must be exactly what a reader can paste into a fresh interpreter.

2. **Imports are restricted to `numpy`, `gragra`, and `astropy.units`.**
   Astropy is a runtime dependency, and the C5 public geometry API takes
   Quantity inputs. Other Astropy modules and optional extras (`h5py`,
   `meshio`, `scipy`, `trimesh`, `numba`, `cupy`, ...) remain disallowed. A
   snippet that needs an optional extra belongs in `examples/` and is linked
   from the page as a non-executed block with an explicit label.

3. **Assertions must state their tolerance explicitly.** NumPy's default
   `atol=1e-8` is meaningless at gragra's output scale (a gravitational
   acceleration of order 1e-11 compares equal to zero, and equal to its own
   negation). Every `np.allclose` / `np.isclose` call in a snippet must pass
   an explicit `atol=`; the test suite enforces this by parsing the source.
   For a quantity whose true value is zero, do not compare against zero with
   a bare tolerance at all -- normalise by the largest term that went into
   the cancellation and bound the normalised residual instead.

4. **No file writes and no network.** Snippets run inside a temporary working
   directory and the test asserts that the directory is still empty
   afterwards.

5. **Print sparingly.** The snippet is read as documentation first and run as
   a test second; its output is not compared against anything, so an
   assertion is worth more than a `print`.

## Naming

Prefix by the page family that owns the snippet, so an orphaned file is
obvious: `quickstart_*`, `observables_*`, `sources_*`, `geometry_*`,
`continuous_*`, `fields_*`, `waves_*`, `coherent_*`, `analytic_*`,
`backends_*`.
