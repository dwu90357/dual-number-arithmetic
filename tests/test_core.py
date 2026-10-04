import math
import unittest

from dual_number_arithmetic import Dual, dual, from_value, to_pair, D
from dual_number_arithmetic import core


class TestConstruction(unittest.TestCase):
    def test_basic_construction(self):
        x = Dual(3.0, 2.0)
        self.assertEqual(x.value, 3.0)
        self.assertEqual(x.deriv, 2.0)
        self.assertEqual(x.pair, (3.0, 2.0))

    def test_int_inputs_promoted_to_float(self):
        x = Dual(1, 0)
        self.assertIsInstance(x.value, float)
        self.assertIsInstance(x.deriv, float)

    def test_default_deriv_is_zero(self):
        x = Dual(5.0)
        self.assertEqual(x.deriv, 0.0)

    def test_helpers(self):
        self.assertEqual(dual(1.0, 2.0).pair, (1.0, 2.0))
        self.assertEqual(from_value(7.0).pair, (7.0, 0.0))
        self.assertEqual(to_pair(9.0), (9.0, 0.0))
        self.assertEqual(to_pair(Dual(1, 2)), (1.0, 2.0))
        self.assertEqual(to_pair(Dual(1, 2)), (1.0, 2.0))
        self.assertIs(D, Dual)

    def test_pair_is_a_copy(self):
        x = Dual(1.0, 2.0)
        p = x.pair
        # Mutating the returned tuple is impossible (tuples are immutable),
        # but confirm a fresh tuple object each call.
        self.assertIsNot(p, x.pair)
        self.assertEqual(p, x.pair)

    def test_rejects_non_real_value(self):
        with self.assertRaises(TypeError):
            Dual("3", 0.0)
        with self.assertRaises(TypeError):
            Dual(None)
        with self.assertRaises(TypeError):
            Dual([1.0], 0.0)

    def test_rejects_non_real_deriv(self):
        with self.assertRaises(TypeError):
            Dual(1.0, "2")
        with self.assertRaises(TypeError):
            Dual(1.0, None)

    def test_rejects_bool(self):
        with self.assertRaises(TypeError):
            Dual(True, 0.0)
        with self.assertRaises(TypeError):
            Dual(1.0, False)


class TestEquality(unittest.TestCase):
    def test_dual_equality(self):
        self.assertEqual(Dual(1.0, 2.0), Dual(1, 2))
        self.assertNotEqual(Dual(1.0, 2.0), Dual(1.0, 3.0))

    def test_equality_with_scalar(self):
        self.assertEqual(Dual(5.0, 0.0), 5)
        self.assertEqual(5, Dual(5.0, 0.0))
        self.assertNotEqual(Dual(5.0, 1.0), 5)

    def test_not_equal_to_unrelated(self):
        self.assertNotEqual(Dual(1.0, 0.0), "1")
        self.assertNotEqual(Dual(1.0, 0.0), None)

    def test_hashable(self):
        s = {Dual(1.0, 2.0), Dual(1.0, 2.0), Dual(3.0, 0.0)}
        self.assertEqual(len(s), 2)


class TestAddSub(unittest.TestCase):
    def test_add_duals(self):
        self.assertEqual((Dual(1, 2) + Dual(3, 4)).pair, (4.0, 6.0))

    def test_add_scalar(self):
        self.assertEqual((Dual(1, 2) + 3).pair, (4.0, 2.0))
        self.assertEqual((3 + Dual(1, 2)).pair, (4.0, 2.0))
        self.assertEqual((3.0 + Dual(1, 2)).pair, (4.0, 2.0))

    def test_sub_duals(self):
        self.assertEqual((Dual(5, 7) - Dual(2, 3)).pair, (3.0, 4.0))

    def test_rsub(self):
        self.assertEqual((10 - Dual(2, 1)).pair, (8.0, -1.0))

    def test_chain(self):
        x = Dual(1.0, 1.0)
        y = Dual(2.0, 3.0)
        z = x + y - 1
        self.assertEqual(z.pair, (2.0, 4.0))


class TestMulDiv(unittest.TestCase):
    def test_mul_duals(self):
        # (a + b e)(c + d e) = ac + (ad + bc) e
        self.assertEqual((Dual(2, 3) * Dual(5, 7)).pair, (10.0, 14.0 + 15.0))

    def test_mul_scalar(self):
        self.assertEqual((Dual(2, 3) * 4).pair, (8.0, 12.0))
        self.assertEqual((4 * Dual(2, 3)).pair, (8.0, 12.0))

    def test_truediv(self):
        self.assertEqual((Dual(6, 2) / Dual(2, 1)).pair, (3.0, (2*2 - 6*1) / 4.0))

    def test_truediv_by_scalar(self):
        self.assertEqual((Dual(6, 2) / 2).pair, (3.0, 1.0))

    def test_rtruediv(self):
        self.assertEqual((10 / Dual(2, 1)).pair, (5.0, -10*1/4.0))

    def test_division_by_zero_raises(self):
        with self.assertRaises(ZeroDivisionError):
            Dual(1, 0) / Dual(0, 1)
        with self.assertRaises(ZeroDivisionError):
            Dual(1, 0) / 0
        with self.assertRaises(ZeroDivisionError):
            1 / Dual(0, 1)


class TestFloorDivMod(unittest.TestCase):
    def test_floordiv_value_and_zero_deriv(self):
        self.assertEqual((Dual(7, 5) // Dual(2, 9)).pair, (3.0, 0.0))
        self.assertEqual((Dual(7, 5) // 2).pair, (3.0, 0.0))
        self.assertEqual((7 // Dual(2, 9)).pair, (3.0, 0.0))

    def test_mod_value_and_zero_deriv(self):
        self.assertEqual((Dual(7, 5) % Dual(3, 9)).pair, (1.0, 0.0))
        self.assertEqual((Dual(7, 5) % 3).pair, (1.0, 0.0))
        self.assertEqual((7 % Dual(3, 9)).pair, (1.0, 0.0))

    def test_floordiv_by_zero(self):
        with self.assertRaises(ZeroDivisionError):
            Dual(1, 0) // Dual(0, 0)
        with self.assertRaises(ZeroDivisionError):
            Dual(1, 0) // 0
        with self.assertRaises(ZeroDivisionError):
            1 // Dual(0, 0)

    def test_mod_by_zero(self):
        with self.assertRaises(ZeroDivisionError):
            Dual(1, 0) % Dual(0, 0)
        with self.assertRaises(ZeroDivisionError):
            1 % Dual(0, 0)


class TestPow(unittest.TestCase):
    def test_pow_scalar_exponent(self):
        # d/dx x**3 = 3x**2
        self.assertEqual((Dual(2, 1) ** 3).pair, (8.0, 12.0))

    def test_pow_negative_base(self):
        # (-2)**3 = -8; derivative 3 * (-2)**2 = 12
        self.assertEqual((Dual(-2, 1) ** 3).pair, (-8.0, 12.0))

    def test_pow_fractional_exponent_positive_base(self):
        # sqrt: (4)**0.5 = 2, deriv 0.5 * 4**-0.5 = 0.25
        self.assertEqual((Dual(4, 1) ** 0.5).pair, (2.0, 0.25))

    def test_pow_fractional_exponent_negative_base_raises(self):
        with self.assertRaises(ValueError):
            Dual(-2, 1) ** 0.5

    def test_pow_zero_to_nonpositive_raises(self):
        with self.assertRaises(ValueError):
            Dual(0, 0) ** 0
        with self.assertRaises(ValueError):
            Dual(0, 0) ** -1

    def test_rpow(self):
        # 2**(1 + 1*e) = 2 * 2**e ... value 2, deriv 2*ln(2)
        result = 2 ** Dual(1, 1)
        self.assertAlmostEqual(result.value, 2.0)
        self.assertAlmostEqual(result.deriv, 2.0 * math.log(2.0))

    def test_rpow_negative_base_raises(self):
        with self.assertRaises(ValueError):
            (-2) ** Dual(1, 1)

    def test_dual_exponent_positive_base(self):
        # x**y with both dual: value 2**3=8,
        # deriv = 8 * (1*ln(2) + 1*3/2) = 8*(ln2 + 1.5)
        result = Dual(2, 1) ** Dual(3, 1)
        self.assertAlmostEqual(result.value, 8.0)
        self.assertAlmostEqual(result.deriv, 8.0 * (math.log(2.0) + 1.5))

    def test_dual_exponent_nonpositive_base_raises(self):
        with self.assertRaises(ValueError):
            Dual(-2, 1) ** Dual(3, 1)

    def test_dual_exponent_zero_deriv_is_scalar_path(self):
        # Exponent Dual with deriv 0 should behave as scalar exponent.
        self.assertEqual((Dual(2, 1) ** Dual(3, 0)).pair, (8.0, 12.0))


class TestUnary(unittest.TestCase):
    def test_neg(self):
        self.assertEqual((-Dual(3, 4)).pair, (-3.0, -4.0))

    def test_pos(self):
        self.assertEqual((+Dual(3, 4)).pair, (3.0, 4.0))
        self.assertIsNot(+Dual(3, 4), Dual(3, 4))

    def test_abs_positive(self):
        self.assertEqual(abs(Dual(3, 4)).pair, (3.0, 4.0))

    def test_abs_negative(self):
        self.assertEqual(abs(Dual(-3, 4)).pair, (3.0, -4.0))

    def test_abs_zero_value(self):
        self.assertEqual(abs(Dual(0, 4)).pair, (0.0, 0.0))


class TestElementary(unittest.TestCase):
    def test_exp(self):
        # exp(1 + e): value e, deriv e
        r = core.exp(Dual(1, 1))
        self.assertAlmostEqual(r.value, math.e)
        self.assertAlmostEqual(r.deriv, math.e)

    def test_exp_scalar(self):
        self.assertEqual(core.exp(0).pair, (1.0, 0.0))

    def test_log(self):
        # d/dx ln(x) = 1/x; at x=2, deriv=0.5
        r = core.log(Dual(2, 1))
        self.assertAlmostEqual(r.value, math.log(2))
        self.assertAlmostEqual(r.deriv, 0.5)

    def test_log_nonpositive_raises(self):
        with self.assertRaises(ValueError):
            core.log(Dual(0, 1))
        with self.assertRaises(ValueError):
            core.log(Dual(-1, 1))

    def test_sqrt(self):
        r = core.sqrt(Dual(4, 1))
        self.assertAlmostEqual(r.value, 2.0)
        self.assertAlmostEqual(r.deriv, 0.25)

    def test_sqrt_zero(self):
        self.assertEqual(core.sqrt(Dual(0, 1)).pair, (0.0, 0.0))

    def test_sqrt_negative_raises(self):
        with self.assertRaises(ValueError):
            core.sqrt(Dual(-1, 1))

    def test_sin_cos(self):
        rs = core.sin(Dual(0, 1))
        rc = core.cos(Dual(0, 1))
        self.assertAlmostEqual(rs.value, 0.0)
        self.assertAlmostEqual(rs.deriv, 1.0)
        self.assertAlmostEqual(rc.value, 1.0)
        self.assertAlmostEqual(rc.deriv, 0.0)

    def test_tan(self):
        r = core.tan(Dual(0, 1))
        self.assertAlmostEqual(r.value, 0.0)
        self.assertAlmostEqual(r.deriv, 1.0)

    def test_tan_at_pi_over_2_raises(self):
        with self.assertRaises(ValueError):
            core.tan(Dual(math.pi / 2, 1))

    def test_sinh_cosh_tanh(self):
        rs = core.sinh(Dual(0, 1))
        rc = core.cosh(Dual(0, 1))
        rt = core.tanh(Dual(0, 1))
        self.assertAlmostEqual(rs.pair, (0.0, 1.0))
        self.assertAlmostEqual(rc.pair, (1.0, 0.0))
        self.assertAlmostEqual(rt.pair, (0.0, 1.0))

    def test_asin(self):
        r = core.asin(Dual(0.5, 1))
        self.assertAlmostEqual(r.value, math.asin(0.5))
        self.assertAlmostEqual(r.deriv, 1.0 / math.sqrt(1.0 - 0.25))

    def test_asin_domain(self):
        with self.assertRaises(ValueError):
            core.asin(Dual(1, 1))
        with self.assertRaises(ValueError):
            core.asin(Dual(-1, 1))
        with self.assertRaises(ValueError):
            core.asin(Dual(2, 1))

    def test_acos(self):
        r = core.acos(Dual(0.5, 1))
        self.assertAlmostEqual(r.value, math.acos(0.5))
        self.assertAlmostEqual(r.deriv, -1.0 / math.sqrt(1.0 - 0.25))

    def test_acos_domain(self):
        with self.assertRaises(ValueError):
            core.acos(Dual(1, 1))

    def test_atan(self):
        r = core.atan(Dual(1, 1))
        self.assertAlmostEqual(r.value, math.atan(1))
        self.assertAlmostEqual(r.deriv, 0.5)


class TestDifferentiation(unittest.TestCase):
    def test_polynomial_derivative(self):
        # f(x) = x**3 + 2x**2 + 3x + 4; f'(2) = 3*4 + 4*2 + 3 = 23
        x = Dual(2, 1)
        y = x**3 + 2 * x**2 + 3 * x + 4
        self.assertAlmostEqual(y.value, 8 + 8 + 6 + 4)
        self.assertAlmostEqual(y.deriv, 12 + 8 + 3)

    def test_composition(self):
        # f(x) = exp(sin(x)); f'(0) = cos(0)*exp(sin(0)) = 1
        x = Dual(0, 1)
        y = core.exp(core.sin(x))
        self.assertAlmostEqual(y.value, 1.0)
        self.assertAlmostEqual(y.deriv, 1.0)

    def test_product_rule(self):
        x = Dual(3, 1)
        # f = x * x; f'(3) = 6
        y = x * x
        self.assertAlmostEqual(y.deriv, 6.0)

    def test_quotient_rule(self):
        x = Dual(2, 1)
        # f = x / (x+1); f'(2) = 1/9
        y = x / (x + 1)
        self.assertAlmostEqual(y.deriv, 1.0 / 9.0)

    def test_chain_with_log_pow(self):
        # f(x) = (log(x))**2; f'(e) = 2*log(e)/e = 2/e
        x = Dual(math.e, 1)
        y = core.log(x) ** 2
        self.assertAlmostEqual(y.value, 1.0)
        self.assertAlmostEqual(y.deriv, 2.0 / math.e)


class TestRejectsBadOperands(unittest.TestCase):
    def test_add_rejects_string(self):
        with self.assertRaises(TypeError):
            Dual(1, 0) + "2"
        with self.assertRaises(TypeError):
            "2" + Dual(1, 0)

    def test_mul_rejects_list(self):
        with self.assertRaises(TypeError):
            Dual(1, 0) * [1]


class TestRepr(unittest.TestCase):
    def test_repr_roundtrip_safe(self):
        x = Dual(1.5, 2.5)
        r = repr(x)
        self.assertIn("Dual(value=", r)
        self.assertIn("1.5", r)
        self.assertIn("2.5", r)

    def test_str(self):
        self.assertEqual(str(Dual(1.0, 2.0)), "1.0 + 2.0*e")
        self.assertEqual(str(Dual(1.0, -2.0)), "1.0 - 2.0*e")


if __name__ == "__main__":
    unittest.main()
