"""Import torch before numpy/pandas.

On Windows builds where numpy ships MKL's OpenMP runtime and torch ships its
own, whichever library initialises second can fail to load its DLLs
(``WinError 1114`` on ``c10.dll``). Importing torch first makes its runtime
the one already in the process, which both libraries then share.

Every module in this package reaches torch through the package ``__init__``,
which imports this module on its first line -- so the ordering holds no
matter which submodule a caller touches first. Applications that import
pandas before ``iagnn`` on such a machine should import ``torch`` (or this
module) at their own entry point first; `tests/conftest.py` shows the shape.
"""
from __future__ import annotations

import torch as _torch  # noqa: F401  -- the import itself is the point

__all__ = ["_torch"]
