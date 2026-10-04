# dual-number-arithmetic

Exact forward-mode automatic differentiation on scalar functions using dual numbers stored as two-tuples.

## Usage

```python
from dual_number_arithmetic import Dual, dual, from_value, to_pair, D
from dual_number_arithmetic import core

# f(x) = x**3 + 2x**2 + 3x + 4 at x=2; f(2)=26, f'(2)=23
x = Dual(2, 1)            # value 2, seed derivative 1
y = x**3 + 2*x**2 + 3*x + 4
assert y.value == 26.0
assert y.deriv == 23.0

# Functions compose exactly:
z = core.exp(core.sin(x)) # f(x)=exp(sin(x)); at x=0, f'=cos(0)*exp(sin(0))=1
x0 = Dual(0, 1)
assert core.exp(core.sin(x0)).deriv == 1.0
```

## Why this exists

This library computes exact derivatives of scalar functions without symbolic CAS or finite-difference approximation. A dual number `a + b*ε` (where `ε² = 0`) carries a value and a derivative in one arithmetic; composing operations composes derivatives by the chain rule, exactly. The trade-off is scope: only the smooth elementary functions in `core` are supported, and the derivative is first-order only.

## Exports

- `Dual` — the dual-number class. Construct as `Dual(value, deriv=0.0)`.
- `dual(value, deriv=0.0)` — functional alias for `Dual(...)`.
- `from_value(x)` — embed a real number as `Dual(x, 0.0)`.
- `to_pair(x)` — return `(value, deriv)` for a `Dual` or real.
- `D` — short alias for `Dual`.
- `core.exp`, `core.log`, `core.sqrt`, `core.sin`, `core.cos`, `core.tan`, `core.sinh`, `core.cosh`, `core.tanh`, `core.asin`, `core.acos`, `core.atan`.

Operators supported on `Dual`: `+`, `-`, `*`, `/`, `//`, `%`, `**`, unary `-`, `+`, `abs()`. Scalars on either side are accepted for `+`, `-`, `*`, `/`, `//`, `%`, `**`.

## Edge you will hit

`//`, `%`, and `abs` are not differentiable everywhere. This library picks one stated interpretation: their derivative is reported as `0` (the derivative of the value-component only, ignoring the `ε`-component). At `x=0`, `abs` reports `(0, 0)` rather than an infinite derivative. `sqrt(0)` likewise returns `(0, 0)`. If you need sub-gradient behaviour, this is not the right tool.

`**` with a `Dual` exponent whose derivative is nonzero requires a strictly positive base (so `log` is defined). `0 ** 0` and `0 ** negative` raise `ValueError`.

Booleans are rejected as operands despite `bool` being a subtype of `int`.
