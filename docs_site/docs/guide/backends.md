
# Compute backends

Most functions that integrate over sources accept a `backend` keyword. It selects *which implementation* evaluates the kernel; it never selects a different physical model, a different sign convention or a different set of guarantees. Everything else on this site — units, axis order, softening semantics, complex-output meaning — is identical on all four paths.

## The reference implementation

The NumPy path (`backend="numpy"`, the default) is the reference implementation. It is the one the analytic fixed points, the independent oracles and the convergence sweeps are asserted against, it is the only one that is exercised on every supported Python version in continuous integration, and it is the one whose result the other three are compared to.

The other three paths are **experimental performance selectors**. They exist to make a large problem finish sooner, and their claim to correctness is entirely derivative: each is asserted to reproduce the NumPy result at a stated tolerance on the inputs the test suite covers. That is a real check, and it is not the same thing as an independent derivation. If an expression is wrong in the reference, it is wrong identically in all four.

| Backend | Requires | Typical reason to use it |
|---|---|---|
| `"numpy"` | nothing beyond the base install | Correctness, portability, reproducibility. The default |
| `"numba"` | the `numba` extra | JIT-compiled CPU loops; avoids materializing the full source × target intermediate |
| `"cpp"` | a build that compiled the bundled C++ extension | Compiled CPU loops with OpenMP threading |
| `"cupy"` | a CuPy wheel matching the local CUDA or ROCm runtime, **and** a usable device | Large source counts on an NVIDIA GPU |

The `numba` and `cpp` paths accept `chunk_size` and validate it, but do not use it for tiling. On the GPU path `chunk_size` genuinely bounds device memory; see [Reproducibility across backends](#reproducibility-across-backends).

## Selecting a backend

Pass the keyword. Nothing else changes:

```python
--8<-- "snippets/backends_error_contract.py"
```

Running that file prints one line per path. On a machine with no accelerators installed, the last three lines report that the path refused — which is the contract, not a failure of the example.

## Backends never fall back

Backends never fall back. If the backend you asked for is unavailable, gragra raises an exception; it does not quietly compute the answer on NumPy.

This is worth stating because the opposite convention is common, and because a silent fallback is undetectable from the result: you would receive a numerically correct array while believing you had measured a GPU timing, or while believing a deployment had the compiled extension it was supposed to have. Here, an unavailable path is an exception at the call that requested it.

## What happens when a backend is unavailable

| You ask for | And | You get |
|---|---|---|
| `"numba"` | numba not installed | `ImportError` with an install hint |
| `"cpp"` | no compiled backend present | `ImportError` ("not built") with the rebuild command |
| `"cpp"` | built before a required symbol existed | a **different** `ImportError` naming the missing symbols (stale binary) |
| `"cupy"` | cupy not installed | an exception that is both an `ImportError` and a `ValueError` (catch either; the class itself is defined locally and is **not** importable from `gragra`) |
| `"cupy"` | cupy present, no usable CUDA device | `RuntimeError`, at call time |
| `"cupy"` | cupy present, device memory insufficient for the requested tile | `MemoryError`, raised from the size estimate before any allocation is attempted |
| an unrecognised name | — | `ValueError` listing the four accepted names |

Three details in that table decide whether your `except` clause actually catches anything.

**The missing-dependency error for the GPU path is not one class.** It is declared inside the function that raises it, separately in each of the four modules that can reach a GPU kernel — the point-direct dispatch, the coherent dispatch, the plane-wave weight dispatch and the volume-transfer kernel. Each definition produces a distinct class object, so catching one of them does not catch another, and there is no name you can import for any of them. What is stable is the base classes: every one of them derives from both `ImportError` and `ValueError`, so `except ImportError` works, `except ValueError` works, and both are documented behaviour.

**A stale compiled extension is a separate failure from a missing one.** The C++ path checks, at import, that the built module exports every symbol the current Python source expects. A binary compiled before a symbol was added therefore raises an `ImportError` that names the missing symbols and asks for a rebuild, rather than failing later with an obscure attribute error.

**Absence of a device is discovered at call time, not at import.** Importing gragra never imports CuPy — the default path is checked to stay clean of it — and a CuPy installation without a working driver or a visible device is only detected when a kernel is about to run. Argument validation happens first, so an invalid `softening_m` or `chunk_size` still raises its own `ValueError` on a machine with no GPU.

## Capability boundaries

All four paths cover the point-direct observables (potential, acceleration, projected acceleration, gravity gradient), the coherent parametric observables, the plane-wave mass weights, the volume transfer kernel and the displacement-transfer (dipole) path. Selecting `"cupy"` for a coherent or parametric call is supported and dispatches to a GPU kernel; it is not rejected as an unsupported combination.

!!! warning "This corrects an earlier statement"

    An earlier internal note recorded that the GPU path covered only the point-direct kernels and that a coherent call would be rejected with a dedicated `ValueError`. That is no longer true: the coherent, parametric, plane-wave and dipole GPU kernels were subsequently implemented, and no such rejection exists in the code. A few source docstrings still carry the older wording; where they disagree with this page, this page describes the behaviour that is actually implemented.

What is genuinely outside every accelerated path:

- **Everything that is not an integration over sources.** The analytic closed forms, the geometry and meshing helpers, the field-to-source conversions and the external-data adapters have no `backend` keyword at all. They run on NumPy.
- **Device-resident results.** The GPU path always returns a host-side, read-only NumPy array. There is no way to keep a result on the device and chain another call against it, so a pipeline of small calls pays the transfer cost every time and can be slower than the CPU paths.
- **Single precision.** Coordinates stay float64 and amplitudes complex128 everywhere, including on the GPU, where double-precision throughput on consumer hardware is a small fraction of single-precision throughput.
- **Multiple devices.** One device per call, selected by the usual CuPy mechanisms. There is no multi-GPU decomposition.

## Reproducibility across backends

Two different questions are worth separating.

**Does a backend depend on `chunk_size`?** On `numpy`, `numba` and `cpp`, no: the result is unchanged for any `chunk_size`, and the compiled paths do not even use the value for tiling. On `cupy`, yes: `chunk_size` bounds what is materialized in device memory, and because a parallel reduction is not associative, changing it changes the summation grouping and therefore the last bits of the answer. Reproducibility there is "same device, same library version, same `chunk_size`", not "same everywhere". [Chunking, softening and tolerances](../concepts/numerics.md) states the full contract.

**Do the backends agree with each other?** They are asserted to, at the tolerances recorded in the blocks below, on the configurations the test suite covers. "Asserted to agree at `rtol=1e-12`" is a weaker and more useful statement than "identical": it is a bound on divergence between implementations of the same formula, and it says nothing at all about whether that formula is right. [How gragra is verified](../verification/index.md) explains why a comparison between implementations is structurally blind to a shared error.

The GPU comparison carries one further limit that no tolerance expresses: it is a local, single-machine measurement, and there is no GPU runner in continuous integration, so nothing re-runs it when the code changes.

## Status and verification

### `backend="numpy"` { #status-backend-numpy }

!!! abstract "Status & verification"

    |  |  |
    |---|---|
    | **Status** | Stable |
    | **Verified against** | closed-form analytic solution; independent numerical oracle (tests/oracles/); invariance property |
    | **Agreement** | Exact at the point-mass fixed points (unit mass, unit distance, zero softening); `rtol=1e-15` for chunk-size invariance; oracle agreements are quoted per symbol on the page that documents it |
    | **Test evidence** | `tests/test_observables.py::test_acceleration_analytic_value`, `tests/test_observables.py::test_potential_analytic_value`, `tests/test_kernel_contract.py::test_kernel_vs_observable_real`, `tests/test_kernel_contract.py::test_chunk_size_invariance` |
    | **Conditions** | The default path; no optional dependency involved. Coordinates float64, amplitudes complex128 |
    | **Not verified** | That the formulas themselves are right beyond the cases with a closed form or an oracle. Two of the project's own sign-convention fixed points are currently asserted at a tolerance that cannot fail — see [Limitations and non-guarantees](../verification/limitations.md) |
    | **Known limits** | Single-threaded; memory is bounded by `chunk_size` rather than by the problem size, so a large source count is slow rather than impossible |
    | **Scope** | none |

### `backend="numba"` { #status-backend-numba }

!!! abstract "Status & verification"

    |  |  |
    |---|---|
    | **Status** | Experimental |
    | **Verified against** | backend equivalence; invariance property |
    | **Agreement** | `rtol=1e-12`, `atol=1e-15` against the NumPy path, real and complex weights; `rtol=1e-15` for chunk-size invariance |
    | **Test evidence** | `tests/test_numba_backend_available.py::test_numpy_numba_equivalence_real`, `tests/test_numba_backend_available.py::test_numpy_numba_equivalence_complex`, `tests/test_numba_backend_available.py::test_numba_chunk_independence`, `tests/test_numba_backend_missing.py::test_numba_backend_missing_raises_importerror` |
    | **Conditions** | Requires the `numba` extra. Comparisons are run on small, well-conditioned point configurations, not on production-sized problems |
    | **Not verified** | Timing or memory behaviour; agreement outside the tested configurations; anything about the correctness of the shared formula, which is identical to the reference |
    | **Known limits** | First call pays JIT compilation. `chunk_size` is validated but not used for tiling |
    | **Scope** | The `backend` keyword itself is an experimental performance selector; availability and behaviour of this path may change without a deprecation period |

### `backend="cpp"` { #status-backend-cpp }

!!! abstract "Status & verification"

    |  |  |
    |---|---|
    | **Status** | Experimental |
    | **Verified against** | backend equivalence; invariance property |
    | **Agreement** | `rtol=1e-12`, `atol=1e-15` against the NumPy path, real and complex weights, including a three-way comparison with the JIT path |
    | **Test evidence** | `tests/test_cpp_backend_available.py::test_numpy_cpp_equivalence_real`, `tests/test_cpp_backend_available.py::test_numpy_numba_cpp_three_way_equivalence`, `tests/test_cpp_backend_available.py::test_cpp_openmp_determinism`, `tests/test_cpp_backend_missing.py::test_cpp_backend_missing_raises_import_error`, `tests/test_cpp_backend_missing.py::test_stale_backend_point_potential` |
    | **Conditions** | Requires a source build that compiled the bundled extension; a plain wheel install does not have it. Thread-count determinism is asserted, so an OpenMP thread count change must not change the result |
    | **Not verified** | Behaviour of compilers, standard libraries or OpenMP runtimes other than the one used in continuous integration; agreement outside the tested configurations |
    | **Known limits** | `chunk_size` is validated but not used for tiling. A binary built against an older source tree is rejected at import rather than used |
    | **Scope** | The `backend` keyword itself is an experimental performance selector; availability and behaviour of this path may change without a deprecation period |

### `backend="cupy"` { #status-backend-cupy }

!!! abstract "Status & verification"

    |  |  |
    |---|---|
    | **Status** | Experimental |
    | **Verified against** | backend equivalence |
    | **Agreement** | `rtol=1e-12`, `atol=1e-15` against the NumPy path — measured on the maintainer's local GPU only, and **not run in CI**, because hosted runners have no GPU |
    | **Test evidence** | `tests/test_cupy_backend_available.py::test_numpy_cupy_equivalence_real`, `tests/test_cupy_backend_available.py::test_multi_backend_equivalence`, `tests/test_cupy_backend_available.py::test_cupy_chunk_invariance_small_n`, `tests/test_cupy_backend_missing.py::test_cupy_backend_missing_raises_import_error`, `tests/test_cupy_backend_missing.py::test_cupy_device_unavailable_raises_runtime_error` |
    | **Conditions** | One device, one driver, one library version, on one machine. The comparison covers point-direct, parametric, plane-wave and dipole kernels at the sizes the local suite runs |
    | **Not verified** | Every configuration other than the maintainer's: no continuous check exists at all for this path. Agreement on other devices, drivers or library versions is unmeasured, as is behaviour at problem sizes that approach device memory |
    | **Known limits** | Results are not bit-reproducible across `chunk_size`, device or library version, because the in-chunk reduction is parallel and non-associative. Output is always copied back to the host; there is no device-resident API. Double precision is used throughout, which on consumer hardware is a small fraction of peak throughput |
    | **Scope** | The `backend` keyword itself is an experimental performance selector; availability and behaviour of this path may change without a deprecation period |
