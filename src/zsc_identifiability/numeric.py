"""Exact and tolerance-aware scalar helpers."""

from __future__ import annotations

from fractions import Fraction
from typing import Literal

Backend = Literal["fraction", "float"]
type Number = Fraction | float
FLOAT_TOLERANCE = 1e-12


def parse_rational(value: str) -> Fraction:
    """Parse a canonical integer or rational string."""
    if not isinstance(value, str):
        raise TypeError("canonical game scalars must be JSON strings")
    try:
        result = Fraction(value)
    except (ValueError, ZeroDivisionError) as exc:
        raise ValueError(f"invalid rational string: {value!r}") from exc
    return result


def number(value: str | Fraction | int, backend: Backend) -> Number:
    exact = value if isinstance(value, Fraction) else Fraction(value)
    return exact if backend == "fraction" else float(exact)


def zero(backend: Backend) -> Number:
    return Fraction(0) if backend == "fraction" else 0.0


def one(backend: Backend) -> Number:
    return Fraction(1) if backend == "fraction" else 1.0


def close(left: Number, right: Number, tolerance: float = FLOAT_TOLERANCE) -> bool:
    if isinstance(left, Fraction) and isinstance(right, Fraction):
        return left == right
    return abs(float(left) - float(right)) <= tolerance


def less(left: Number, right: Number, tolerance: float = FLOAT_TOLERANCE) -> bool:
    if isinstance(left, Fraction) and isinstance(right, Fraction):
        return left < right
    return float(left) < float(right) - tolerance


def serialize_number(value: Number | None) -> str | float | None:
    if value is None:
        return None
    if isinstance(value, Fraction):
        return str(value)
    return float(value)
