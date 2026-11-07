"""Patience-based early stopping for a scalar validation metric."""

from __future__ import annotations

import math


class EarlyStopping:
    """Track the best value of a metric and say when training should stop.

    ``patience`` is the number of consecutive updates without a sufficient
    improvement that are allowed before ``should_stop`` becomes true. The
    first update is always an improvement. In ``min`` mode a later update
    improves when ``value < best - min_delta``. In ``max`` mode it improves
    when ``value > best + min_delta``.
    """

    def __init__(self, *, patience: int, min_delta: float = 0.0, mode: str = "min") -> None:
        if patience < 1:
            raise ValueError("patience must be an integer greater than or equal to 1.")
        if not math.isfinite(min_delta) or min_delta < 0:
            raise ValueError("min_delta must be a finite number greater than or equal to zero.")
        if mode not in {"min", "max"}:
            raise ValueError("mode must be 'min' or 'max'.")
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.best: float | None = None
        self.best_step: int | None = None
        self.steps_without_improvement = 0
        self.should_stop = False

    def update(self, value: float, step: int) -> bool:
        """Record one observation. Return whether it is the new best value."""
        if not math.isfinite(value):
            raise ValueError("EarlyStopping metric must be finite.")
        best = self.best
        if best is None or self._is_improvement(value, best):
            self.best = value
            self.best_step = step
            self.steps_without_improvement = 0
            return True
        self.steps_without_improvement += 1
        self.should_stop = self.steps_without_improvement >= self.patience
        return False

    def _is_improvement(self, value: float, best: float) -> bool:
        if self.mode == "min":
            return value < best - self.min_delta
        return value > best + self.min_delta
