"""Tests for `satsim.time` package."""
from dateutil import parser
from satsim.time import utc, utc_from_list, utc_from_list_or_scalar, to_astropy, from_astropy, linspace, delta_sec, from_datetime


def test_utc():

    t = utc(2020,1,1,0,0,0.0)
    assert(t.tt == 2458849.500800741)

    t = utc(2020,1,1,0,0,0.0 + 1)
    assert(t.tt == 2458849.500812315)


def test_utc_from_list():

    t = utc_from_list([2020,1,1,0,0,0.0])
    assert(t.tt == 2458849.500800741)

    t = utc_from_list([2020,1,1,0,0,0.0], 1)
    assert(t.tt == 2458849.500812315)


def test_utc_from_scalar():

    t = utc_from_list_or_scalar([2020,1,1,0,0,0.0], 0, None)
    assert(t.tt == 2458849.500800741)

    t = utc_from_list_or_scalar([2020,1,1,0,0,0.0], 1, None)
    assert(t.tt == 2458849.500812315)

    t = utc_from_list_or_scalar(None, default_t=[2020,1,1,0,0,0.0])
    assert(t.tt == 2458849.500800741)

    t = utc_from_list_or_scalar(0, default_t=[2020,1,1,0,0,0.0])
    assert(t.tt == 2458849.500800741)

    t = utc_from_list_or_scalar(0, 1, default_t=[2020,1,1,0,0,0.0])
    assert(t.tt == 2458849.500812315)

    t = utc_from_list_or_scalar(1, default_t=[2020,1,1,0,0,0.0])
    assert(t.tt == 2458849.500812315)


def test_from_datetime():

    dt = from_datetime(parser.isoparse('2020-01-01T00:00:00Z'))
    t = utc_from_list_or_scalar([2020,1,1,0,0,0.0], 0, None)

    assert(delta_sec(dt, t) == 0)


def test_astropy():

    t = utc_from_list_or_scalar([2020,1,1,0,0,0.0], 0, None)

    ta = to_astropy(t)
    t2 = from_astropy(ta)

    assert(abs(delta_sec(t, t2)) < 2e-5)


def test_linspace():

    t0 = utc_from_list_or_scalar([2020,1,1,0,0,0.0], 0, None)
    t1 = utc_from_list_or_scalar([2020,1,1,0,1,0.0], 0, None)

    tt = linspace(t0, t1, 61)

    for ii in range(1, 60):
        assert(abs(1.0 - delta_sec(tt[ii], tt[ii - 1])) < 1e-10)
