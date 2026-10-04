"""Exact forward-mode automatic differentiation via dual numbers.

A dual number is written a + b*ε where ε^2 = 0. The non-tuple part is the
value; the coefficient of ε is the derivative. Because the ring quotient by
ε^2 is exact on the operations defined here, the derivative computed for any
composition of them is exact (not a finite-difference approximation).

Implementation note: each Dual stores (value, deriv) as a plain two-tuple on
the instance (``self._pair``). The public ``pair`` property returns a copy so
external mutation of the underlying tuple cannot desync a Dual's identity.
All arithmetic on Duals produces new Duals; nothing is mutated in place.
"""

from __future__ import annotations

import math
from typing import Tuple, Union

Number = Union[int, float]
Pair = Tuple[float, float]


class Dual:
    """A dual number a + b*ε with ε^2 = 0.

    Construction is via class methods (``Dual(value, deriv)`` or the ``dual``
    helper) rather than overloading ``__new__`` because we want strict type
    checking: a Dual is never a no-op wrapper around another Dual.
    """

    __slots__ = ("_pair",)

    __slots__ = ("_pair",)

    def __init__(self, value: Number, deriv: Number = 0.0) -> None:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError("Dual value must be a real number (int or float), got " + type(value).__name__)
        if isinstance(deriv, bool) or not isinstance(deriv, (int, float)):
            raise TypeError("Dual deriv must be a real number (int or float), got " + type(deriv).__name__)
        object.__setattr__(self, "_pair", (float(value), float(deriv)))

    # --- properties -------------------------------------------------------

    @property
    def value(self) -> float:
        return self._pair[0]

    @property
    def deriv(self) -> float:
        return self._pair[1]

    @property
    def pair(self) -> Pair:
        # Copy so callers cannot mutate the internal tuple in place.
        return (self._pair[0], self._pair[1])

    # --- representation ---------------------------------------------------

    def __repr__(self) -> str:
        return f"Dual(value={self._pair[0]!r}, deriv={self._pair[1]!r})"

    def __str__(self) -> str:
        sign = "+" if self._pair[1] >= 0 else "-"
        return f"{self._pair[0]} {sign} {abs(self._pair[1])}*e"

    # --- equality / hashing ----------------------------------------------

    def __eq__(self, other: object) -> bool:
        if isinstance(other, Dual):
            return self._pair == other._pair
        if isinstance(other, (int, float)) and not isinstance(other, bool):
            return self._pair == (float(other), 0.0)
        return NotImplemented

    def __hash__(self) -> int:
        return hash(self._pair)

    # --- helpers ----------------------------------------------------------

    @staticmethod
    def _to_pair(x: Union["Dual", Number]) -> Pair:
        if isinstance(x, Dual):
            return x._pair
        if isinstance(x, bool) or not isinstance(x, (int, float)):
            raise TypeError("operand must be a Dual or real number, got " + type(x).__name__)
        return (float(x), 0.0)

    def _coerce_addition_operand(self, other: Union["Dual", Number]) -> Pair:
        # For + and -, numeric operands add zero derivative; Dual operands add
        # both components. This is the only coercion that accepts scalars.
        return self._to_pair(other)

    def _coerce_mul_operand(self, other: Union["Dual", Number]) -> Tuple[Pair, Pair]:
        # For *, /, //, %, ** a scalar operand contributes value=itself,
        # deriv=0. A Dual operand contributes its own pair. Crucially, a Dual
        # in the base/exponent position of pow is only meaningful for non-dual
        # exponents (see __pow__).
        return self._pair, self._to_pair(other)

    # --- arithmetic: addition & subtraction ------------------------------

    def __add__(self, other: Union["Dual", Number]) -> "Dual":
        a, b = self._coerce_addition_operand(other)
        return Dual(self._pair[0] + a, self._pair[1] + b)

    def __radd__(self, other: Number) -> "Dual":
        a, b = self._to_pair(other)
        return Dual(self._pair[0] + a, self._pair[1] + b)

    def __sub__(self, other: Union["Dual", Number]) -> "Dual":
        a, b = self._coerce_addition_operand(other)
        return Dual(self._pair[0] - a, self._pair[1] - b)

    def __rsub__(self, other: Number) -> "Dual":
        a, b = self._to_pair(other)
        return Dual(a - self._pair[0], b - self._pair[1])

    # --- arithmetic: multiplication & division ---------------------------

    def __mul__(self, other: Union["Dual", Number]) -> "Dual":
        (av, ad), (bv, bd) = self._coerce_mul_operand(other)
        return Dual(av * bv, ad * bv + av * bd)

    def __rmul__(self, other: Number) -> "Dual":
        (av, ad), (bv, bd) = self._coerce_mul_operand(other)
        return Dual(bv * av, bd * av + bv * ad)

    def __truediv__(self, other: Union["Dual", Number]) -> "Dual":
        (av, ad), (bv, bd) = self._coerce_mul_operand(other)
        if bv == 0.0:
            raise ZeroDivisionError("dual division by zero value component")
        return Dual(av / bv, (ad * bv - av * bd) / (bv * bv))

    def __rtruediv__(self, other: Number) -> "Dual":
        (av, ad), (bv, bd) = self._coerce_mul_operand(other)
        if av == 0.0:
            raise ZeroDivisionError("dual division by zero value component")
        return Dual(bv / av, (bd * av - bv * ad) / (av * av))

    def __floordiv__(self, other: Union["Dual", Number]) -> "Dual":
        # Floor division is not smooth, so the derivative is the derivative of
        # the floor of the value (constant, hence 0). We compute the floored
        # value and report derivative 0, mirroring how floor behaves at points
        # away from integer jumps. This is the pragmatic, stated interpretation.
        (av, ad), (bv, bd) = self._coerce_mul_operand(other)
        if bv == 0.0:
            raise ZeroDivisionError("dual floordiv by zero value component")
        return Dual(float(math.floor(av / bv)), 0.0)

    def __rfloordiv__(self, other: Number) -> "Dual":
        a, b = self._to_pair(other)
        if self._pair[0] == 0.0:
            raise ZeroDivisionError("dual floordiv by zero value component")
        return Dual(float(math.floor(a / self._pair[0])), 0.0)

    def __mod__(self, other: Union["Dual", Number]) -> "Dual":
        # Mirrors floordiv's stance: mod is piecewise, derivative is 0 at
        # generic points. We return the value-mod and a zero derivative.
        (av, ad), (bv, bd) = self._coerce_mul_operand(other)
        if bv == 0.0:
            raise ZeroDivisionError("dual mod by zero value component")
        return Dual(float(math.fmod(av, bv)), 0.0)

    def __rmod__(self, other: Number) -> "Dual":
        a, b = self._to_pair(other)
        if self._pair[0] == 0.0:
            raise ZeroDivisionError("dual mod by zero value component")
        return Dual(float(math.fmod(a, self._pair[0])), 0.0)

    # --- arithmetic: power -----------------------------------------------

    def __pow__(self, other: Union["Dual", Number]) -> "Dual":
        (av, ad), (bv, bd) = self._coerce_mul_operand(other)
        if av == 0.0 and bv <= 0.0:
            raise ValueError("0**non-positive is undefined for duals")
        if isinstance(other, Dual) and bd != 0.0:
            # Only meaningful when base is strictly positive; the value of
            # a**b is exp(b*ln(a)) and its derivative in a is b*a**(b-1),
            # in b is a**b*ln(a).
            if av <= 0.0:
                raise ValueError("dual base with nonzero exponent derivative requires positive base")
            ln_a = math.log(av)
            val = av ** bv
            deriv = val * (bd * ln_a + (ad * bv / av))
            return Dual(val, deriv)
        # Scalar (or zero-derivative) exponent: standard power rule, with
        # av**(bv-1) computed once.
        if av < 0.0 and float(bv) != int(bv):
            raise ValueError("negative base with fractional exponent is undefined for duals")
        return Dual(av ** bv, ad * bv * (av ** (bv - 1)))

    def __rpow__(self, other: Number) -> "Dual":
        a, b = self._to_pair(other)
        if a == 0.0 and self._pair[0] <= 0.0:
            raise ValueError("0**non-positive is undefined for duals")
        if a <= 0.0:
            raise ValueError("scalar base ** Dual exponent requires positive scalar base")
        val = a ** self._pair[0]
        deriv = val * self._pair[1] * math.log(a)
        return Dual(val, deriv)

    # --- unary -----------------------------------------------------------

    def __neg__(self) -> "Dual":
        return Dual(-self._pair[0], -self._pair[1])

    def __pos__(self) -> "Dual":
        return Dual(self._pair[0], self._pair[1])

    def __abs__(self) -> "Dual":
        # |x| is non-smooth at 0; pick the sign of the value and zero derivative
        # there. This is the pragmatic single interpretation, stated plainly.
        v = self._pair[0]
        if v < 0.0:
            return Dual(-v, -self._pair[1])
        if v > 0.0:
            return Dual(v, self._pair[1])
        return Dual(0.0, 0.0)


# Module-level functional helpers ------------------------------------------------


def dual(value: Number, deriv: Number = 0.0) -> Dual:
    """Construct a Dual. Functional alias for ``Dual(value, deriv)``."""
    return Dual(value, deriv)


def from_value(value: Number) -> Dual:
    """Embed a real number as a Dual with zero derivative."""
    return Dual(value, 0.0)


def to_pair(x: Union[Dual, Number]) -> Pair:
    """Return the ``(value, deriv)`` pair; reals contribute deriv 0."""
    return Dual._to_pair(x)


D = Dual


# Elementary functions on Duals ---------------------------------------------------


def _apply(fn_deriv, value: float, deriv: float) -> Dual:
    val, d = fn_deriv(value, deriv)
    return Dual(val, d)


def exp(x: Union[Dual, Number]) -> Dual:
    v, d = to_pair(x)
    val = math.exp(v)
    return Dual(val, d * val)


def log(x: Union[Dual, Number]) -> Dual:
    v, d = to_pair(x)
    if v <= 0.0:
        raise ValueError("log domain error: dual value component must be positive")
    return Dual(math.log(v), d / v)


def sqrt(x: Union[Dual, Number]) -> Dual:
    v, d = to_pair(x)
    if v < 0.0:
        raise ValueError("sqrt domain error: dual value component must be non-negative")
    if v == 0.0:
        # Derivative of sqrt at 0 is infinite; report 0 as the pragmatic choice
        # so callers get a finite result, consistent with abs at zero.
        return Dual(0.0, 0.0)
    val = math.sqrt(v)
    return Dual(val, d / (2.0 * val))


def sin(x: Union[Dual, Number]) -> Dual:
    v, d = to_pair(x)
    return Dual(math.sin(v), d * math.cos(v))


def cos(x: Union[Dual, Number]) -> Dual:
    v, d = to_pair(x)
    return Dual(math.cos(v), -d * math.sin(v))


def tan(x: Union[Dual, Number]) -> Dual:
    v, d = to_pair(x)
    c = math.cos(v)
    if abs(c) < 1e-12:
        raise ValueError("tan undefined at odd multiples of pi/2")
    t = math.tan(v)
    return Dual(t, d * (1.0 + t * t))


def sinh(x: Union[Dual, Number]) -> Dual:
    v, d = to_pair(x)
    return Dual(math.sinh(v), d * math.cosh(v))


def cosh(x: Union[Dual, Number]) -> Dual:
    v, d = to_pair(x)
    return Dual(math.cosh(v), d * math.sinh(v))


def tanh(x: Union[Dual, Number]) -> Dual:
    v, d = to_pair(x)
    t = math.tanh(v)
    return Dual(t, d * (1.0 - t * t))


def asin(x: Union[Dual, Number]) -> Dual:
    v, d = to_pair(x)
    if v <= -1.0 or v >= 1.0:
        raise ValueError("asin domain error: value must be in (-1, 1)")
    return Dual(math.asin(v), d / math.sqrt(1.0 - v * v))


def acos(x: Union[Dual, Number]) -> Dual:
    v, d = to_pair(x)
    if v <= -1.0 or v >= 1.0:
        raise ValueError("acos domain error: value must be in (-1, 1)")
    return Dual(math.acos(v), -d / math.sqrt(1.0 - v * v))


def atan(x: Union[Dual, Number]) -> Dual:
    v, d = to_pair(x)
    return Dual(math.atan(v), d / (1.0 + v * v))
