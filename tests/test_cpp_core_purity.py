"""Purity tests for core gragra.

Asserts compiled backend is not eagerly loaded on import.
"""

import subprocess
import sys


def test_core_purity_cpp_backend_lazy_loaded():
    """Verify that importing gragra and executing basic numpy calculations
    does not load C++ backend.
    """
    script = """
import sys
import gragra
from gragra.gravity_sources import PointMass
from gragra.observables import point_potential

# Run potential calculation with numpy (default)
source = PointMass(1.0, [0.0, 0.0, 0.0])
target = [[1.0, 0.0, 0.0]]
point_potential(source, target) # Should use numpy default backend

# Verify _cpp_backend is not loaded in sys.modules
if "gragra._cpp_backend" in sys.modules:
    sys.exit(1)
sys.exit(0)
"""
    res = subprocess.run([sys.executable, "-c", script], capture_output=True)
    assert res.returncode == 0, (
        "Core purity test failed: C++ backend was loaded eagerly. "
        f"Stderr: {res.stderr.decode()}"
    )
