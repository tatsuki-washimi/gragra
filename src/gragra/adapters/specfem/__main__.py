"""Allows ``python -m gragra.adapters.specfem <command> ...`` invocation."""

import sys

from gragra.adapters.specfem.cli import main

if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
