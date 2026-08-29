"""Canonical progress-reporting type.

Formalizes the `progress: Callable[[str], None] = print` convention
already used in `sam3_cropmarks` into one shared import, so newly
migrated modules can adopt the same signature instead of hardcoding
`print()` calls.
"""
from __future__ import annotations

from typing import Callable

Progress = Callable[[str], None]
DEFAULT_PROGRESS: Progress = print
