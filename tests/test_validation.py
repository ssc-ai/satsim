import pytest

from satsim.util.validation import (
    finite_number,
    integer,
    nonnegative_number,
    optional_finite_number,
    optional_unit_interval,
    positive_integer,
    positive_number,
    unit_interval,
)


@pytest.mark.parametrize('value', [0, 1.5])
def test_finite_number_accepts_real_numbers(value):
    assert finite_number('field', value) == pytest.approx(float(value))


@pytest.mark.parametrize('value', [True, '1.0', float('inf')])
def test_finite_number_rejects_non_numeric_or_non_finite_values(value):
    with pytest.raises(ValueError, match='field must be a finite number'):
        finite_number('field', value)


def test_integer_requires_integral_values():
    assert integer('field', 3.0) == 3

    with pytest.raises(ValueError, match='field must be an integer'):
        integer('field', 3.5)


def test_positive_integer_requires_positive_integral_values():
    assert positive_integer('field', 2) == 2

    with pytest.raises(ValueError, match='field must be positive'):
        positive_integer('field', 0)


@pytest.mark.parametrize('value', [0.0, 0.5, 1.0])
def test_unit_interval_accepts_closed_unit_range(value):
    assert unit_interval('field', value) == pytest.approx(value)


@pytest.mark.parametrize('value', [-0.1, 1.1])
def test_unit_interval_rejects_out_of_range_values(value):
    with pytest.raises(ValueError, match='field must be in the range'):
        unit_interval('field', value)


def test_positive_and_nonnegative_number_bounds():
    assert positive_number('field', 0.1) == pytest.approx(0.1)
    assert nonnegative_number('field', 0.0) == pytest.approx(0.0)

    with pytest.raises(ValueError, match='field must be positive'):
        positive_number('field', 0.0)

    with pytest.raises(ValueError, match='field must be nonnegative'):
        nonnegative_number('field', -0.1)


def test_optional_validators_allow_none():
    assert optional_finite_number('field', None) is None
    assert optional_unit_interval('field', None) is None
    assert optional_finite_number('field', 2.5) == pytest.approx(2.5)
    assert optional_unit_interval('field', 0.25) == pytest.approx(0.25)
