import os

import numpy as np

from satsim.geometry import gcvs5


FIXTURE = os.path.join(os.path.dirname(__file__), "data", "gcvs5_fixture.txt")


def _make_catalog(**overrides):
    data = {
        "ra_deg": np.array([0.0], dtype=float),
        "dec_deg": np.array([0.0], dtype=float),
        "var_type": np.array(["EA"], dtype=object),
        "max_mag": np.array([10.0], dtype=float),
        "max_mag_limit": np.array([""], dtype=object),
        "min_mag": np.array([12.0], dtype=float),
        "min_mag_limit": np.array([""], dtype=object),
        "sec_min_mag": np.array([np.nan], dtype=float),
        "sec_min_mag_limit": np.array([""], dtype=object),
        "mag_system": np.array(["V"], dtype=object),
        "epoch": np.array([np.nan], dtype=float),
        "period": np.array([np.nan], dtype=float),
        "period_limit": np.array([""], dtype=object),
        "period_flag": np.array([""], dtype=object),
        "period_note": np.array([""], dtype=object),
        "rise_eclipse_time": np.array([np.nan], dtype=float),
        "rise_eclipse_note": np.array([""], dtype=object),
        "name": np.array(["TEST"], dtype=object),
        "unit_vectors": np.array([[1.0, 0.0, 0.0]], dtype=float),
        "kd_tree": None,
    }
    data.update(overrides)
    return gcvs5.Gcvs5Catalog(**data)


def test_load_gcvs5_fixture_parsing():
    catalog = gcvs5.load_gcvs5_catalog(FIXTURE, None)
    assert len(catalog.ra_deg) == 5

    ra_expected = (0.0 + 24.0 / 60.0 + 1.95 / 3600.0) * 15.0
    dec_expected = 38.0 + 34.0 / 60.0 + 37.3 / 3600.0
    assert np.isclose(catalog.ra_deg[0], ra_expected, atol=1e-6)
    assert np.isclose(catalog.dec_deg[0], dec_expected, atol=1e-6)

    assert np.isclose(catalog.max_mag[0], 5.8)
    assert np.isclose(catalog.min_mag[0], 15.2)
    assert np.isclose(catalog.sec_min_mag[1], 9.28)
    assert np.isclose(catalog.period[1], 0.6289216)
    assert np.isclose(catalog.rise_eclipse_time[1], 17.0)
    assert catalog.rise_eclipse_note[1] == "*"

    assert catalog.var_type[2] == "SR:"
    assert catalog.mag_system[2] == "P"


def test_auto_eclipsing_detects_composite_type():
    catalog = gcvs5.load_gcvs5_catalog(FIXTURE, None)
    idx = 3  # CI Aql, type NR+EA
    lc_type = gcvs5.get_lightcurve_type(catalog.var_type[idx], catalog.rise_eclipse_time[idx], "auto")
    assert lc_type == "eclipse"


def test_rv_epoch_min_sawtooth_shape():
    catalog = gcvs5.load_gcvs5_catalog(FIXTURE, None)
    idx = 4  # GK Cyg, type RV with rise time
    cfg = gcvs5.normalize_config(
        {
            "enabled": True,
            "match_radius_arcsec": 5.0,
            "template": "auto",
        }
    )
    ra = np.array([catalog.ra_deg[idx]])
    dec = np.array([catalog.dec_deg[idx]])
    base_mag = np.array([(catalog.max_mag[idx] + catalog.min_mag[idx]) / 2.0])
    state = gcvs5.prepare_gcvs5_state(ra, dec, base_mag, catalog, cfg)

    epoch_jd = catalog.epoch[idx] + 2400000.0
    m = gcvs5.apply_gcvs5_variability(base_mag, ra, dec, epoch_jd, catalog, state, cfg, frame_index=0)
    assert np.isclose(m[0], catalog.min_mag[idx])


def test_eclipse_v_shape_and_secondary_minimum():
    catalog = gcvs5.load_gcvs5_catalog(FIXTURE, None)
    idx = 1  # RT And, eclipsing with '*' note and secondary minimum
    cfg = gcvs5.normalize_config(
        {
            "enabled": True,
            "match_radius_arcsec": 5.0,
            "template": "auto",
        }
    )
    ra = np.array([catalog.ra_deg[idx]])
    dec = np.array([catalog.dec_deg[idx]])
    base_mag = np.array([(catalog.max_mag[idx] + catalog.min_mag[idx]) / 2.0])
    state = gcvs5.prepare_gcvs5_state(ra, dec, base_mag, catalog, cfg)

    epoch_jd = catalog.epoch[idx] + 2400000.0
    period = catalog.period[idx]
    d_frac = catalog.rise_eclipse_time[idx] / 100.0
    mid_phase = 0.25 * d_frac

    t_mid = epoch_jd + period * mid_phase
    m_mid = gcvs5.apply_gcvs5_variability(base_mag, ra, dec, t_mid, catalog, state, cfg, frame_index=0)
    assert m_mid[0] < catalog.min_mag[idx]
    assert m_mid[0] > catalog.max_mag[idx]

    t_sec = epoch_jd + period * 0.5
    m_sec = gcvs5.apply_gcvs5_variability(base_mag, ra, dec, t_sec, catalog, state, cfg, frame_index=0)
    assert np.isclose(m_sec[0], catalog.sec_min_mag[idx])


def test_missing_period_constant_is_deterministic():
    catalog = gcvs5.load_gcvs5_catalog(FIXTURE, None)
    cfg = gcvs5.normalize_config(
        {
            "enabled": True,
            "match_radius_arcsec": 5.0,
            "missing_period_mode": "constant",
            "missing_period_seed": 123,
        }
    )

    ra = np.array([catalog.ra_deg[3]])
    dec = np.array([catalog.dec_deg[3]])
    base_mag = np.array([12.0])

    state = gcvs5.prepare_gcvs5_state(ra, dec, base_mag, catalog, cfg)
    m1 = gcvs5.apply_gcvs5_variability(base_mag, ra, dec, 2450000.0, catalog, state, cfg, frame_index=0)
    m2 = gcvs5.apply_gcvs5_variability(base_mag, ra, dec, 2450000.5, catalog, state, cfg, frame_index=1)

    assert np.isclose(m1[0], m2[0])


def test_eclipse_default_duration_when_missing():
    max_mag = 10.0
    min_mag = 12.0
    m_out = gcvs5._template_mag(
        0.06,
        max_mag,
        min_mag,
        np.nan,
        "",
        True,
        np.nan,
        "",
        "EA",
        "auto",
    )
    m_in = gcvs5._template_mag(
        0.04,
        max_mag,
        min_mag,
        np.nan,
        "",
        True,
        np.nan,
        "",
        "EA",
        "auto",
    )
    assert np.isclose(m_out, max_mag)
    assert np.isclose(m_in, min_mag)


def test_saw_fallback_defaults_to_symmetric():
    max_mag = 10.0
    min_mag = 14.0
    mean_mag = 12.0
    m = gcvs5._template_mag(
        0.25,
        max_mag,
        min_mag,
        np.nan,
        "",
        False,
        np.nan,
        "",
        "SR",
        "saw",
    )
    assert np.isfinite(m)
    assert np.isclose(m, mean_mag)


def test_missing_period_random_varies_by_frame():
    catalog = gcvs5.load_gcvs5_catalog(FIXTURE, None)
    idx = 3  # CI Aql, missing period
    cfg = gcvs5.normalize_config(
        {
            "enabled": True,
            "match_radius_arcsec": 5.0,
            "template": "sin",
            "missing_period_mode": "random",
            "missing_period_seed": 123,
        }
    )

    ra = np.array([catalog.ra_deg[idx]])
    dec = np.array([catalog.dec_deg[idx]])
    base_mag = np.array([12.0])

    state = gcvs5.prepare_gcvs5_state(ra, dec, base_mag, catalog, cfg)
    m1 = gcvs5.apply_gcvs5_variability(base_mag, ra, dec, 2450000.0, catalog, state, cfg, frame_index=0)
    m2 = gcvs5.apply_gcvs5_variability(base_mag, ra, dec, 2450000.0, catalog, state, cfg, frame_index=1)

    assert not np.isclose(m1[0], m2[0])


def test_normalize_config_mag_systems():
    cfg = gcvs5.normalize_config({"mag_system": "v"})
    assert cfg["mag_systems"] == ["V"]
    cfg = gcvs5.normalize_config({"mag_systems": ["", "  "]})
    assert cfg["mag_systems"] is None


def test_get_field_positions_defaults(tmp_path):
    missing = tmp_path / "missing.txt"
    assert gcvs5._get_field_positions(str(missing)) == gcvs5._DEFAULT_SEP_POSITIONS
    bad = tmp_path / "bad.txt"
    bad.write_text("no separators\n", encoding="ascii")
    assert gcvs5._get_field_positions(str(bad)) == gcvs5._DEFAULT_SEP_POSITIONS


def test_split_fixed_short_line():
    parts = gcvs5._split_fixed("x", (3,))
    assert len(parts) == 2
    assert parts[0] == "x  "


def test_parse_ra_dec_invalid_inputs():
    ra, dec = gcvs5._parse_ra_dec("")
    assert np.isnan(ra)
    assert np.isnan(dec)

    ra, dec = gcvs5._parse_ra_dec("123")
    assert np.isnan(ra)
    assert np.isnan(dec)

    ra, dec = gcvs5._parse_ra_dec("1234 12")
    assert np.isnan(ra)
    assert np.isnan(dec)

    ra, dec = gcvs5._parse_ra_dec("...... +......")
    assert np.isnan(ra)
    assert np.isnan(dec)


def test_parse_mag_field_flags():
    value, limit, uncertain = gcvs5._parse_mag_field("<12.3:")
    assert np.isclose(value, 12.3)
    assert limit == "<"
    assert uncertain is True

    value, limit, uncertain = gcvs5._parse_mag_field("(")
    assert np.isnan(value)
    assert limit == "("
    assert uncertain is False


def test_parse_epoch_field_flags():
    value, flag = gcvs5._parse_epoch_field(":")
    assert np.isnan(value)
    assert flag == ":"

    value, flag = gcvs5._parse_epoch_field("+123.4")
    assert np.isclose(value, 123.4)
    assert flag == "+"


def test_parse_period_field_flags():
    value, limit, flag, note = gcvs5._parse_period_field("<MN")
    assert np.isnan(value)
    assert limit == "<"
    assert note == "MN"
    assert flag == ""

    value, limit, flag, note = gcvs5._parse_period_field("1.23M2")
    assert np.isclose(value, 1.23)
    assert note == "M2"
    assert limit == ""
    assert flag == ""


def test_parse_float_field_note_only():
    value, flag, note = gcvs5._parse_float_field("*")
    assert np.isnan(value)
    assert flag == ""
    assert note == "*"


def test_epoch_to_jd_special_cases():
    assert np.isnan(gcvs5._epoch_to_jd(np.nan))
    assert gcvs5._epoch_to_jd(237249.0, "WY Sge") == 237249.0


def test_is_epoch_minimum_defaults():
    assert gcvs5._is_epoch_minimum("") is False
    assert gcvs5._is_epoch_minimum("DSCT") is False


def test_template_sawtooth_branches():
    max_mag = 10.0
    min_mag = 14.0
    rise_frac = 0.3
    fall_frac = 1.0 - rise_frac
    phase = 0.9

    m = gcvs5._template_mag(
        phase,
        max_mag,
        min_mag,
        np.nan,
        "",
        True,
        30.0,
        "",
        "SR",
        "saw",
    )
    expected = max_mag + (min_mag - max_mag) * ((phase - rise_frac) / fall_frac)
    assert np.isclose(m, expected)

    m = gcvs5._template_mag(
        phase,
        max_mag,
        min_mag,
        np.nan,
        "",
        False,
        30.0,
        "",
        "SR",
        "saw",
    )
    expected = min_mag + (max_mag - min_mag) * ((phase - fall_frac) / rise_frac)
    assert np.isclose(m, expected)


def test_template_eclipse_explicit():
    max_mag = 10.0
    min_mag = 12.0
    m = gcvs5._template_mag(
        0.2,
        max_mag,
        min_mag,
        np.nan,
        "",
        True,
        np.nan,
        "",
        "EA",
        "eclipse",
    )
    assert np.isclose(m, max_mag)


def test_template_auto_branches():
    max_mag = 10.0
    min_mag = 12.0
    m = gcvs5._template_mag(
        0.0,
        max_mag,
        min_mag,
        np.nan,
        "",
        True,
        np.nan,
        "",
        "ELL",
        "auto",
    )
    assert np.isclose(m, min_mag)

    m = gcvs5._template_mag(
        0.0,
        max_mag,
        min_mag,
        np.nan,
        "",
        False,
        np.nan,
        "",
        "DSCT",
        "auto",
    )
    assert np.isclose(m, max_mag)


def test_get_lightcurve_type_branches():
    assert gcvs5.get_lightcurve_type("EA", np.nan, "sin") == "sin"
    assert gcvs5.get_lightcurve_type("ELL", np.nan, "auto") == "sin"
    assert gcvs5.get_lightcurve_type("SR", 12.0, "auto") == "saw"
    assert gcvs5.get_lightcurve_type("SR", np.nan, "auto") == "sin"


def test_get_lightcurve_info_valid_and_invalid():
    info = gcvs5.get_lightcurve_info("SR", np.nan, "auto", 10.0, 10.0, 9.0, "", "")
    assert info["lc_status"] == "invalid_mag_range"

    info = gcvs5.get_lightcurve_info("SR", np.nan, "auto", 10.0, 10.0, 12.0, "", "")
    assert info["lc_status"] == "ok"


def test_apply_period_note_and_derive_mag_range():
    assert gcvs5._apply_period_note(1.0, "M2") == 2.0

    max_mag, min_mag, source = gcvs5._derive_mag_range(10.0, 2.0, np.nan, "(", "")
    assert np.isclose(max_mag, 9.0)
    assert np.isclose(min_mag, 11.0)
    assert source == "amplitude"

    max_mag, min_mag, source = gcvs5._derive_mag_range(10.0, np.nan, 2.0, "", "(")
    assert np.isclose(max_mag, 9.0)
    assert np.isclose(min_mag, 11.0)
    assert source == "amplitude"

    max_mag, min_mag, source = gcvs5._derive_mag_range(10.0, 1.0, 2.0, "<", "")
    assert np.isnan(max_mag)
    assert np.isnan(min_mag)
    assert source == "invalid"

    max_mag, min_mag, source = gcvs5._derive_mag_range(10.0, np.nan, 2.0, "", "")
    assert np.isnan(max_mag)
    assert np.isnan(min_mag)
    assert source == "invalid"

    max_mag, min_mag, source = gcvs5._derive_mag_range(10.0, 5.0, 5.0, "", "")
    assert np.isnan(max_mag)
    assert np.isnan(min_mag)
    assert source == "invalid"


def test_load_gcvs5_catalog_mag_system_filter():
    catalog = gcvs5.load_gcvs5_catalog(FIXTURE, ["Z"])
    assert len(catalog.ra_deg) == 0
    assert catalog.kd_tree is None


def test_load_gcvs5_catalog_skips_invalid_ra_dec(tmp_path):
    with open(FIXTURE, "r", encoding="ascii") as handle:
        line = handle.readline().rstrip("\n")
    positions = gcvs5._get_field_positions(FIXTURE)
    start = positions[1] + 1
    end = positions[2]
    bad_line = line[:start] + (" " * (end - start)) + line[end:]

    path = tmp_path / "mixed.txt"
    path.write_text(bad_line + "\n" + line + "\n", encoding="ascii")

    catalog = gcvs5.load_gcvs5_catalog(str(path), None)
    assert len(catalog.ra_deg) == 1


def test_match_gcvs5_no_catalog():
    idx = gcvs5.match_gcvs5([0.0], [0.0], None, 5.0)
    assert idx[0] == -1


def test_match_gcvs5_bruteforce_fallback():
    catalog = gcvs5.load_gcvs5_catalog(FIXTURE, None)
    catalog_no_tree = gcvs5.Gcvs5Catalog(**{**catalog.__dict__, "kd_tree": None})
    idx = gcvs5.match_gcvs5([catalog.ra_deg[0]], [catalog.dec_deg[0]], catalog_no_tree, 5.0)
    assert idx[0] == 0


def test_prepare_gcvs5_state_none_and_no_match():
    cfg = gcvs5.normalize_config({"enabled": True})
    assert gcvs5.prepare_gcvs5_state([0.0], [0.0], [10.0], None, cfg) is None

    catalog = gcvs5.load_gcvs5_catalog(FIXTURE, None)
    cfg["match_radius_arcsec"] = 0.1
    state = gcvs5.prepare_gcvs5_state([0.0], [0.0], [10.0], catalog, cfg)
    assert state.match_idx[0] == -1


def test_prepare_gcvs5_state_period_without_epoch():
    catalog = gcvs5.load_gcvs5_catalog(FIXTURE, None)
    idx = 2  # AW And
    cfg = gcvs5.normalize_config({"enabled": True})
    ra = np.array([catalog.ra_deg[idx]])
    dec = np.array([catalog.dec_deg[idx]])
    base_mag = np.array([12.0])
    state = gcvs5.prepare_gcvs5_state(ra, dec, base_mag, catalog, cfg)
    assert np.isfinite(state.phase_offset[0])


def test_apply_gcvs5_variability_no_catalog():
    base_mag = np.array([11.0])
    out = gcvs5.apply_gcvs5_variability(base_mag, [0.0], [0.0], 2450000.0, None, None, {}, frame_index=0)
    assert np.isclose(out[0], base_mag[0])


def test_apply_gcvs5_variability_no_match():
    catalog = gcvs5.load_gcvs5_catalog(FIXTURE, None)
    state = gcvs5.Gcvs5State(
        match_idx=np.array([-1], dtype=int),
        phase_offset=np.array([0.0], dtype=float),
        missing_phase=np.array([np.nan], dtype=float),
    )
    cfg = gcvs5.normalize_config({"enabled": True})
    base_mag = np.array([11.0])
    out = gcvs5.apply_gcvs5_variability(base_mag, [0.0], [0.0], 2450000.0, catalog, state, cfg, frame_index=0)
    assert np.isclose(out[0], base_mag[0])


def test_apply_gcvs5_variability_invalid_mag_range():
    catalog = _make_catalog(max_mag_limit=np.array(["<"], dtype=object))
    state = gcvs5.Gcvs5State(
        match_idx=np.array([0], dtype=int),
        phase_offset=np.array([0.0], dtype=float),
        missing_phase=np.array([np.nan], dtype=float),
    )
    cfg = gcvs5.normalize_config({"enabled": True})
    base_mag = np.array([11.0])
    out = gcvs5.apply_gcvs5_variability(base_mag, [0.0], [0.0], 2450000.0, catalog, state, cfg, frame_index=0)
    assert np.isclose(out[0], base_mag[0])


def test_apply_gcvs5_variability_period_valid_epoch_missing():
    catalog = gcvs5.load_gcvs5_catalog(FIXTURE, None)
    idx = 2  # AW And
    cfg = gcvs5.normalize_config({"enabled": True})
    ra = np.array([catalog.ra_deg[idx]])
    dec = np.array([catalog.dec_deg[idx]])
    base_mag = np.array([12.0])
    state = gcvs5.prepare_gcvs5_state(ra, dec, base_mag, catalog, cfg)
    out = gcvs5.apply_gcvs5_variability(base_mag, ra, dec, 2450000.0, catalog, state, cfg, frame_index=0)
    assert np.isfinite(out[0])


def test_apply_gcvs5_variability_missing_phase_seeded():
    catalog = _make_catalog(period=np.array([np.nan], dtype=float))
    state = gcvs5.Gcvs5State(
        match_idx=np.array([0], dtype=int),
        phase_offset=np.array([0.0], dtype=float),
        missing_phase=np.array([np.nan], dtype=float),
    )
    cfg = gcvs5.normalize_config({"enabled": True, "missing_period_mode": "constant"})
    base_mag = np.array([11.0])
    gcvs5.apply_gcvs5_variability(base_mag, [0.0], [0.0], 2450000.0, catalog, state, cfg, frame_index=0)
    assert np.isfinite(state.missing_phase[0])


def test_apply_gcvs5_variability_missing_mode_skip():
    catalog = _make_catalog(period=np.array([np.nan], dtype=float))
    state = gcvs5.Gcvs5State(
        match_idx=np.array([0], dtype=int),
        phase_offset=np.array([0.0], dtype=float),
        missing_phase=np.array([np.nan], dtype=float),
    )
    cfg = gcvs5.normalize_config({"enabled": True, "missing_period_mode": "skip"})
    base_mag = np.array([11.0])
    out = gcvs5.apply_gcvs5_variability(base_mag, [0.0], [0.0], 2450000.0, catalog, state, cfg, frame_index=0)
    assert np.isclose(out[0], base_mag[0])


def test_get_variable_flags_basic():
    base_mag = np.array([10.0, 11.0])
    out = gcvs5.get_variable_flags(base_mag, None, None, None)
    assert (out == np.array([False, False], dtype=object)).all()


def test_get_variable_flags_with_catalog():
    catalog = gcvs5.load_gcvs5_catalog(FIXTURE, None)
    idx = 1  # RT And, eclipsing with valid range
    base_mag = np.array([(catalog.max_mag[idx] + catalog.min_mag[idx]) / 2.0])
    state = gcvs5.Gcvs5State(
        match_idx=np.array([idx], dtype=int),
        phase_offset=np.array([0.0], dtype=float),
        missing_phase=np.array([np.nan], dtype=float),
    )
    cfg = gcvs5.normalize_config({"enabled": True, "template": "auto"})
    out = gcvs5.get_variable_flags(base_mag, catalog, state, cfg)
    assert out[0] == "eclipse"


def test_get_variable_flags_invalid_range():
    catalog = _make_catalog(max_mag_limit=np.array(["<"], dtype=object))
    base_mag = np.array([11.0])
    state = gcvs5.Gcvs5State(
        match_idx=np.array([0], dtype=int),
        phase_offset=np.array([0.0], dtype=float),
        missing_phase=np.array([np.nan], dtype=float),
    )
    out = gcvs5.get_variable_flags(base_mag, catalog, state, {"template": "auto"})
    assert out[0] is False


def test_get_variable_flags_explicit_template():
    catalog = gcvs5.load_gcvs5_catalog(FIXTURE, None)
    idx = 1  # RT And
    base_mag = np.array([(catalog.max_mag[idx] + catalog.min_mag[idx]) / 2.0])
    state = gcvs5.Gcvs5State(
        match_idx=np.array([idx], dtype=int),
        phase_offset=np.array([0.0], dtype=float),
        missing_phase=np.array([np.nan], dtype=float),
    )
    out = gcvs5.get_variable_flags(base_mag, catalog, state, {"template": "sin"})
    assert out[0] == "sin"
