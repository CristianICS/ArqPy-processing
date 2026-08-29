"""Structural contract for a GUI-invokable long-running job.

This is a `Protocol` (structural typing), not an ABC — new orchestrator
classes such as `atmo_correction.job.AtmoCorrectionJob` can satisfy it
without inheriting from anything, and existing well-factored classes
(`highpass.GDALHighPassFilter`, `sensors.WorldView` subclasses) are not
retrofitted into a forced base class.
"""
from __future__ import annotations

from typing import Protocol, runtime_checkable

from .progress import DEFAULT_PROGRESS, Progress


@runtime_checkable
class Processor(Protocol):
    """Error-handling contract binding for new code written against this
    Protocol: `validate()` raises a typed exception (FileNotFoundError,
    ValueError, NotADirectoryError, or a domain-specific subclass) the
    instant an input is invalid — it never silently returns/no-ops to mean
    "skipped". `run()` either completes and returns a result, or raises;
    it does not catch and return an exception as a value (that conversion
    happens once, at the GUI `run_job` boundary — see `app/gui/framework.py`).
    """

    def validate(self) -> None: ...

    def run(self, progress: Progress = DEFAULT_PROGRESS) -> object: ...
