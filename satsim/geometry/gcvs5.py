from __future__ import division, print_function, absolute_import

import logging
import math
import os
import re
import zlib
from dataclasses import dataclass
from functools import lru_cache
from typing import Optional

import numpy as np

from scipy.spatial import cKDTree

logger = logging.getLogger(__name__)

DEFAULT_GCVS5_PATH = os.environ.get(
    "SATSIM_GCVS5_PATH",
    os.path.abspath("./gcvs5.txt"),
)

# Fixed-width separator positions inferred from gcvs5.txt (first record).
_DEFAULT_SEP_POSITIONS = (
    7, 19, 40, 51, 61, 74, 87, 90, 103, 109, 130, 136, 154, 166, 178, 192,
    201, 203, 213, 224, 235,
)


@dataclass
class Gcvs5Catalog:
    ra_deg: np.ndarray
    dec_deg: np.ndarray
    var_type: np.ndarray
    max_mag: np.ndarray
    max_mag_limit: np.ndarray
    min_mag: np.ndarray
    min_mag_limit: np.ndarray
    sec_min_mag: np.ndarray
    sec_min_mag_limit: np.ndarray
    mag_system: np.ndarray
    epoch: np.ndarray
    period: np.ndarray
    period_limit: np.ndarray
    period_flag: np.ndarray
    period_note: np.ndarray
    rise_eclipse_time: np.ndarray
    rise_eclipse_note: np.ndarray
    name: np.ndarray
    unit_vectors: np.ndarray
    kd_tree: Optional[object]


@dataclass
class Gcvs5State:
    match_idx: np.ndarray
    phase_offset: np.ndarray
    missing_phase: np.ndarray


def normalize_config(cfg):
    """Normalize a user config dict for GCVS5 variability.

    This sets defaults and harmonizes mag system filters. The mag-system filter
    is optional and simply excludes catalog rows whose GCVS `mag_system` does
    not match, which helps avoid mixing photometric bands.

    Args:
        cfg: `dict`, user-provided configuration or None.

    Returns:
        A `dict`, normalized configuration with defaults applied.
    """
    cfg = dict(cfg) if cfg is not None else {}
    cfg.setdefault("enabled", False)
    cfg.setdefault("path", DEFAULT_GCVS5_PATH)
    cfg.setdefault("match_radius_arcsec", 5.0)
    cfg.setdefault("template", "auto")
    cfg.setdefault("missing_period_mode", "constant")
    cfg.setdefault("missing_period_seed", None)

    mag_systems = cfg.get("mag_systems", cfg.get("mag_system"))
    if isinstance(mag_systems, str):
        mag_systems = [mag_systems]
    if mag_systems is not None:
        mag_systems = [str(v).strip().upper() for v in mag_systems if str(v).strip() != ""]
        if len(mag_systems) == 0:
            mag_systems = None
    cfg["mag_systems"] = mag_systems
    return cfg


@lru_cache(maxsize=4)
def _get_field_positions(path):
    try:
        with open(path, "r") as handle:
            line = handle.readline().rstrip("\n")
    except Exception:
        return _DEFAULT_SEP_POSITIONS

    pos = [i for i, ch in enumerate(line) if ch == "|"]
    if len(pos) != len(_DEFAULT_SEP_POSITIONS):
        return _DEFAULT_SEP_POSITIONS
    return tuple(pos)


def _split_fixed(line, positions):
    line = line.rstrip("\n")
    last = positions[-1]
    if len(line) < last:
        line = line.ljust(last)
    starts = [0] + [p + 1 for p in positions]
    ends = list(positions) + [len(line)]
    return [line[starts[i]:ends[i]] for i in range(len(starts))]


def _clean_token(token, allow_sign=True):
    allowed = "+-0123456789." if allow_sign else "0123456789."
    return "".join(ch for ch in token if ch in allowed)


def _parse_ra_dec(field):
    """Parse GCVS RA/Dec in the compact HHMMSS.SS +DDMMSS.S format."""
    if not field or field.strip() == "":
        return np.nan, np.nan
    parts = field.strip().split()
    if len(parts) < 2:
        return np.nan, np.nan
    ra_token = _clean_token(parts[0], allow_sign=False)
    dec_token = _clean_token(parts[1], allow_sign=True)
    if len(ra_token) < 6 or len(dec_token) < 7:
        return np.nan, np.nan

    try:
        rh = int(ra_token[0:2])
        rm = int(ra_token[2:4])
        rs = float(ra_token[4:])
        ra_deg = (rh + rm / 60.0 + rs / 3600.0) * 15.0
    except Exception:
        ra_deg = np.nan

    try:
        sign = -1.0 if dec_token.startswith("-") else 1.0
        dec_body = dec_token[1:] if dec_token[0] in "+-" else dec_token
        dd = int(dec_body[0:2])
        dm = int(dec_body[2:4])
        ds = float(dec_body[4:])
        dec_deg = sign * (dd + dm / 60.0 + ds / 3600.0)
    except Exception:
        dec_deg = np.nan

    return ra_deg, dec_deg


def _parse_mag_field(field):
    """Parse a magnitude field and extract flags.

    GCVS uses leading limits like '<' and '>' as well as '(' to mean the value
    is an amplitude rather than an absolute magnitude. A trailing ':' marks an
    uncertain value.
    """
    raw = field.strip() if field else ""
    if raw == "":
        return np.nan, "", False

    limit = ""
    for flag in ("<", ">", "("):
        if flag in raw:
            limit = flag
            break

    uncertain = ":" in raw
    raw = raw.replace("(", " ").replace(")", " ").replace("<", " ").replace(">", " ").replace(":", " ")
    match = re.search(r"[-+]?\d+(?:\.\d+)?", raw)
    if not match:
        return np.nan, limit, uncertain
    return float(match.group(0)), limit, uncertain


def _parse_epoch_field(field):
    """Parse epoch field and retain its uncertainty flag.

    Epoch values are stored as JD-2400000 in the GCVS table; conversion to JD
    happens later via `_epoch_to_jd`.
    """
    raw = field.strip() if field else ""
    if raw == "":
        return np.nan, ""
    flag = ""
    for f in (":", "+", "-"):
        if f in raw:
            flag = f
            break
    raw = raw.replace(":", " ").replace("+", " ").replace("-", " ")
    match = re.search(r"[-+]?\d+(?:\.\d+)?", raw)
    if not match:
        return np.nan, flag
    return float(match.group(0)), flag


def _parse_period_field(field):
    """Parse period field with limit/flag/note metadata.

    Period notes like MN/M2/FN indicate that the quoted period may be a multiple
    or fraction of the true period. These notes are stored; only M2 is applied
    as a 2x heuristic during variability modeling.
    """
    raw = field.strip() if field else ""
    if raw == "":
        return np.nan, "", "", ""

    period_limit = ""
    for flag in ("<", ">", "("):
        if flag in raw:
            period_limit = flag
            break

    period_flag = ":" if ":" in raw else ""
    note_match = re.search(r"(M2|MN|FN|/N)", raw)
    period_note = note_match.group(1) if note_match else ""

    raw = raw.replace("(", " ").replace(")", " ").replace("<", " ").replace(">", " ").replace(":", " ")
    raw = raw.replace("M2", " ").replace("MN", " ").replace("FN", " ").replace("/N", " ")
    match = re.search(r"[-+]?\d+(?:\.\d+)?", raw)
    if not match:
        return np.nan, period_limit, period_flag, period_note
    return float(match.group(0)), period_limit, period_flag, period_note


def _parse_float_field(field):
    raw = field.strip() if field else ""
    if raw == "":
        return np.nan, "", ""
    flag = ":" if ":" in raw else ""
    note = "*" if "*" in raw else ""
    raw = raw.replace(":", " ").replace("*", " ")
    match = re.search(r"[-+]?\d+(?:\.\d+)?", raw)
    if not match:
        return np.nan, flag, note
    return float(match.group(0)), flag, note


def _epoch_to_jd(epoch_val, name=None):
    """Convert GCVS epoch (JD-2400000) to full JD.

    GCVS notes an exception for WY Sge where the epoch is given as a full JD.
    """
    if epoch_val is None or not np.isfinite(epoch_val):
        return np.nan

    if name is not None:
        name_norm = " ".join(str(name).replace("*", " ").split()).upper()
        if name_norm == "WY SGE" and epoch_val > 200000.0:
            return epoch_val

    return epoch_val + 2400000.0 if epoch_val < 2400000.0 else epoch_val


def _split_var_type(var_type):
    vt = str(var_type or "").upper()
    return [tok for tok in re.split(r"[^A-Z0-9]+", vt) if tok]


def _is_eclipsing_type(var_type):
    tokens = _split_var_type(var_type)
    for tok in tokens:
        if tok in ("E", "EA", "EB", "EW"):
            return True
    return False


def _is_ellipsoidal_type(var_type):
    tokens = _split_var_type(var_type)
    return "ELL" in tokens


def _is_epoch_minimum(var_type):
    """Return True when GCVS epoch corresponds to minimum light.

    Per GCVS documentation, eclipsing and ellipsoidal binaries (E*, ELL) and
    RV Tau or RS CVn types use minimum-light epochs.
    """
    if not var_type:
        return False
    tokens = _split_var_type(var_type)
    if _is_eclipsing_type(var_type) or "ELL" in tokens:
        return True
    for tok in tokens:
        if tok.startswith("RV") or tok.startswith("RS"):
            return True
    return False


def _template_mag(phase, max_mag, min_mag, sec_min_mag, sec_min_mag_limit,
                  epoch_is_min, rise_eclipse_time, rise_eclipse_note,
                  var_type, template):
    """Generate a synthetic magnitude for a phase in [0, 1).

    Templates:
      - sin: symmetric sinusoid between max/min.
      - saw: asymmetric rise/decay using rise time (M-m) percent of period.
      - eclipse: box-shaped eclipse of duration D% centered at phase 0.
        If D is missing, a 10% duration is assumed. Secondary eclipses, when
        provided, are placed at phase 0.5 with the same duration.

    Auto selection is based on GCVS type and the presence of `rise_eclipse_time`:
      - eclipsing types -> eclipse (D if present, else 10% duration).
      - ellipsoidal types -> sinusoid.
      - intrinsic types with M-m -> sawtooth (epoch at min or max supported).
      - otherwise -> sinusoid.
    """
    amp = 0.5 * (min_mag - max_mag)
    mean = 0.5 * (min_mag + max_mag)

    def percent_or_default(value, default_frac):
        if value is None or not np.isfinite(value) or value <= 0.0:
            return default_frac
        return float(value) / 100.0

    def sinusoid():
        s = -1.0 if not epoch_is_min else 1.0
        return mean + s * amp * math.cos(2.0 * math.pi * phase)

    def sawtooth(rise_frac):
        rise_frac = float(np.clip(rise_frac, 1e-3, 0.999))
        if epoch_is_min:
            if phase < rise_frac:
                return min_mag + (max_mag - min_mag) * (phase / rise_frac)
            fall_frac = 1.0 - rise_frac
            return max_mag + (min_mag - max_mag) * ((phase - rise_frac) / fall_frac)
        fall_frac = 1.0 - rise_frac
        if phase < fall_frac:
            return max_mag + (min_mag - max_mag) * (phase / fall_frac)
        return min_mag + (max_mag - min_mag) * ((phase - fall_frac) / rise_frac)

    def _eclipse_at(center, ecl_frac, depth_mag, v_shape):
        half = 0.5 * ecl_frac
        dphi = abs((phase - center + 0.5) % 1.0 - 0.5)
        if dphi > half:
            return max_mag
        if v_shape:
            return max_mag + (depth_mag - max_mag) * (1.0 - (dphi / half))
        return depth_mag

    def eclipse_model(ecl_frac):
        ecl_frac = float(np.clip(ecl_frac, 1e-3, 0.999))
        v_shape = bool(rise_eclipse_note)
        primary = _eclipse_at(0.0, ecl_frac, min_mag, v_shape)

        sec_depth = np.nan
        if np.isfinite(sec_min_mag) and sec_min_mag_limit not in ("<", ">", "("):
            if sec_min_mag > max_mag and sec_min_mag < min_mag:
                sec_depth = sec_min_mag

        if np.isfinite(sec_depth):
            secondary = _eclipse_at(0.5, ecl_frac, sec_depth, v_shape)
            return max(primary, secondary)
        return primary

    is_eclipsing = _is_eclipsing_type(var_type)
    is_ellipsoidal = _is_ellipsoidal_type(var_type)

    if template == "sin":
        return sinusoid()
    if template == "saw":
        rf = percent_or_default(rise_eclipse_time, 0.5)
        return sawtooth(rf)
    if template == "eclipse":
        ef = percent_or_default(rise_eclipse_time, 0.1)
        return eclipse_model(ef)

    if is_eclipsing:
        ef = percent_or_default(rise_eclipse_time, 0.1)
        return eclipse_model(ef)
    if is_ellipsoidal:
        return sinusoid()
    if np.isfinite(rise_eclipse_time):
        return sawtooth(percent_or_default(rise_eclipse_time, 0.5))
    return sinusoid()


def get_lightcurve_type(var_type, rise_eclipse_time, template="auto"):
    """Return the chosen light-curve template name.

    If `template` is explicit, it is returned verbatim. For `auto`, this mirrors
    the template decision logic used in `_template_mag` so callers can explain
    why a shape was selected.

    Args:
        var_type: `string`, GCVS variability type code.
        rise_eclipse_time: `float`, rise time or eclipse duration (% of period).
        template: `string`, template selection (`auto`, `sin`, `saw`, `eclipse`).

    Returns:
        A `string`, the chosen light-curve template name.
    """
    if template in ("sin", "saw", "eclipse"):
        return template

    is_eclipsing = _is_eclipsing_type(var_type)
    is_ellipsoidal = _is_ellipsoidal_type(var_type)

    if is_eclipsing:
        return "eclipse"
    if is_ellipsoidal:
        return "sin"
    if np.isfinite(rise_eclipse_time):
        return "saw"
    return "sin"


def get_lightcurve_info(var_type, rise_eclipse_time, template, base_mag, max_mag, min_mag, max_flag, min_flag):
    """Return light-curve type plus whether a usable mag range exists.

    This uses `_derive_mag_range`, which interprets amplitude flags in GCVS and
    rejects limit-only magnitudes. When the range is invalid, lc_type is `none`.

    Args:
        var_type: `string`, GCVS variability type code.
        rise_eclipse_time: `float`, rise time or eclipse duration (% of period).
        template: `string`, template selection (`auto`, `sin`, `saw`, `eclipse`).
        base_mag: `float`, baseline magnitude used when only amplitudes are provided.
        max_mag: `float`, GCVS maximum magnitude value.
        min_mag: `float`, GCVS minimum magnitude value.
        max_flag: `string`, GCVS max magnitude limit or amplitude flag.
        min_flag: `string`, GCVS min magnitude limit or amplitude flag.

    Returns:
        A `dict`, with keys `lc_type`, `lc_status`, and `range_source`.
    """
    max_mag_use, min_mag_use, source = _derive_mag_range(
        base_mag,
        max_mag,
        min_mag,
        max_flag,
        min_flag,
    )
    if not np.isfinite(max_mag_use) or not np.isfinite(min_mag_use):
        return {
            "lc_type": "none",
            "lc_status": "invalid_mag_range",
            "range_source": source,
        }

    return {
        "lc_type": get_lightcurve_type(var_type, rise_eclipse_time, template),
        "lc_status": "ok",
        "range_source": source,
    }


def get_variable_flags(base_mag, catalog, state, config=None):
    """Return per-star variability markers for annotation.

    Args:
        base_mag: `array`, baseline magnitudes for each star.
        catalog: `Gcvs5Catalog`, catalog to match against.
        state: `Gcvs5State`, precomputed match indices.
        config: `dict` or `None`, normalized GCVS5 config.

    Returns:
        A `ndarray` of `object`, False for non-variable stars or the `lc_type`
        string (e.g., `eclipse`, `saw`, `sin`) when variability is defined.
    """
    base_mag = np.asarray(base_mag, dtype=float)
    out = np.full(base_mag.shape, False, dtype=object)
    if catalog is None or state is None:
        return out

    template = config.get("template", "auto") if config is not None else "auto"

    for i, idx in enumerate(state.match_idx):
        if idx < 0:
            continue
        info = get_lightcurve_info(
            catalog.var_type[idx],
            catalog.rise_eclipse_time[idx],
            template,
            base_mag[i],
            catalog.max_mag[idx],
            catalog.min_mag[idx],
            catalog.max_mag_limit[idx],
            catalog.min_mag_limit[idx],
        )
        if info["lc_status"] == "ok":
            out[i] = info["lc_type"]

    return out


def _apply_period_note(period, period_note):
    if not np.isfinite(period):
        return np.nan
    note = str(period_note).strip().upper()
    if note == "M2":
        return period * 2.0
    return period


def _derive_mag_range(base_mag, max_mag, min_mag, max_flag, min_flag):
    """Derive a [max, min] magnitude range from GCVS values.

    If GCVS marks a value as an amplitude (flag '('), a symmetric range around
    `base_mag` is created. If values are limits ('<' or '>') or missing, the
    range is invalid and variability is skipped.
    """
    if max_flag == "(" and np.isfinite(max_mag):
        amp = max_mag
        return base_mag - amp / 2.0, base_mag + amp / 2.0, "amplitude"
    if min_flag == "(" and np.isfinite(min_mag):
        amp = min_mag
        return base_mag - amp / 2.0, base_mag + amp / 2.0, "amplitude"

    if max_flag in ("<", ">") or min_flag in ("<", ">"):
        return np.nan, np.nan, "invalid"
    if not np.isfinite(max_mag) or not np.isfinite(min_mag):
        return np.nan, np.nan, "invalid"
    if min_mag <= max_mag:
        return np.nan, np.nan, "invalid"
    return max_mag, min_mag, "absolute"


def _stable_seed(seed, ra_deg, dec_deg, extra=0):
    base = 0 if seed is None else int(seed)
    key = f"{base}|{round(float(ra_deg), 6)}|{round(float(dec_deg), 6)}|{extra}"
    return zlib.crc32(key.encode("ascii")) & 0xFFFFFFFF


def _radec_to_unit(ra_deg, dec_deg):
    ra = np.deg2rad(ra_deg)
    dec = np.deg2rad(dec_deg)
    cosd = np.cos(dec)
    return np.column_stack([cosd * np.cos(ra), cosd * np.sin(ra), np.sin(dec)])


def load_gcvs5_catalog(path=DEFAULT_GCVS5_PATH, mag_systems=None):
    """Load and cache the GCVS5 catalog from the fixed-width text file.

    The file is parsed using fixed separators to avoid ambiguity from embedded
    '|' characters in some columns. Optionally filter by mag system (e.g. "V").
    A unit-vector KD-tree is built when SciPy is available to speed matching.

    Args:
        path: `string`, path to `gcvs5.txt`.
        mag_systems: `list` of `string` or `None`, filter by mag system codes.

    Returns:
        A `Gcvs5Catalog`, parsed catalog data and optional KD-tree.
    """
    mag_key = None
    if mag_systems is not None:
        mag_key = tuple(mag_systems)
    return _load_gcvs5_catalog(path, mag_key)


@lru_cache(maxsize=2)
def _load_gcvs5_catalog(path, mag_key):
    positions = _get_field_positions(path)
    mag_systems = mag_key

    rows = []
    with open(path, "r") as handle:
        for line in handle:
            fields = _split_fixed(line, positions)

            name = fields[1].strip()
            ra_deg, dec_deg = _parse_ra_dec(fields[2])
            if not np.isfinite(ra_deg) or not np.isfinite(dec_deg):
                continue

            var_type = fields[3].strip()
            max_mag, max_flag, _ = _parse_mag_field(fields[4])
            min_mag, min_flag, _ = _parse_mag_field(fields[5])
            sec_min_mag, sec_min_flag, _ = _parse_mag_field(fields[6])
            mag_system = fields[7].strip().upper()
            if mag_systems and mag_system not in mag_systems:
                continue

            epoch, _ = _parse_epoch_field(fields[8])
            period, period_limit, period_flag, period_note = _parse_period_field(fields[10])
            rise_eclipse_time, _, rise_eclipse_note = _parse_float_field(fields[11])

            rows.append(
                (
                    ra_deg, dec_deg, var_type, max_mag, max_flag, min_mag, min_flag,
                    sec_min_mag, sec_min_flag, mag_system, epoch, period, period_limit,
                    period_flag, period_note, rise_eclipse_time, rise_eclipse_note, name,
                )
            )

    if len(rows) == 0:
        empty = np.array([], dtype=float)
        return Gcvs5Catalog(
            ra_deg=empty,
            dec_deg=empty,
            var_type=np.array([], dtype=object),
            max_mag=empty,
            max_mag_limit=np.array([], dtype=object),
            min_mag=empty,
            min_mag_limit=np.array([], dtype=object),
            sec_min_mag=empty,
            sec_min_mag_limit=np.array([], dtype=object),
            mag_system=np.array([], dtype=object),
            epoch=empty,
            period=empty,
            period_limit=np.array([], dtype=object),
            period_flag=np.array([], dtype=object),
            period_note=np.array([], dtype=object),
            rise_eclipse_time=empty,
            rise_eclipse_note=np.array([], dtype=object),
            name=np.array([], dtype=object),
            unit_vectors=np.empty((0, 3), dtype=float),
            kd_tree=None,
        )

    arr = list(zip(*rows))
    ra_deg = np.array(arr[0], dtype=float)
    dec_deg = np.array(arr[1], dtype=float)
    unit_vectors = _radec_to_unit(ra_deg, dec_deg)
    tree = cKDTree(unit_vectors) if len(unit_vectors) > 0 else None

    return Gcvs5Catalog(
        ra_deg=ra_deg,
        dec_deg=dec_deg,
        var_type=np.array(arr[2], dtype=object),
        max_mag=np.array(arr[3], dtype=float),
        max_mag_limit=np.array(arr[4], dtype=object),
        min_mag=np.array(arr[5], dtype=float),
        min_mag_limit=np.array(arr[6], dtype=object),
        sec_min_mag=np.array(arr[7], dtype=float),
        sec_min_mag_limit=np.array(arr[8], dtype=object),
        mag_system=np.array(arr[9], dtype=object),
        epoch=np.array(arr[10], dtype=float),
        period=np.array(arr[11], dtype=float),
        period_limit=np.array(arr[12], dtype=object),
        period_flag=np.array(arr[13], dtype=object),
        period_note=np.array(arr[14], dtype=object),
        rise_eclipse_time=np.array(arr[15], dtype=float),
        rise_eclipse_note=np.array(arr[16], dtype=object),
        name=np.array(arr[17], dtype=object),
        unit_vectors=unit_vectors,
        kd_tree=tree,
    )


def match_gcvs5(ra_deg, dec_deg, catalog, radius_arcsec):
    """Match input RA/Dec to nearest GCVS5 rows within a radius.

    Uses a KD-tree in unit-vector space for speed; falls back to brute-force if
    SciPy is unavailable.

    Args:
        ra_deg: `array`, right ascension in degrees.
        dec_deg: `array`, declination in degrees.
        catalog: `Gcvs5Catalog`, catalog to match against.
        radius_arcsec: `float`, match radius in arcseconds.

    Returns:
        A `ndarray`, index of matched catalog rows (or -1 when no match).
    """
    ra_deg = np.asarray(ra_deg, dtype=float)
    dec_deg = np.asarray(dec_deg, dtype=float)
    if catalog is None or len(catalog.ra_deg) == 0:
        return np.full(ra_deg.shape, -1, dtype=int)

    if catalog.kd_tree is None:
        # Brute-force fallback; small arrays only.
        matches = np.full(ra_deg.shape, -1, dtype=int)
        for i, (ra, dec) in enumerate(zip(ra_deg, dec_deg)):
            v = _radec_to_unit(np.array([ra]), np.array([dec]))[0]
            dots = np.dot(catalog.unit_vectors, v)
            dots = np.clip(dots, -1.0, 1.0)
            ang = np.arccos(dots)
            j = int(np.argmin(ang))
            if ang[j] <= np.deg2rad(radius_arcsec / 3600.0):
                matches[i] = j
        return matches

    v = _radec_to_unit(ra_deg, dec_deg)
    ang = np.deg2rad(radius_arcsec / 3600.0)
    max_dist = 2.0 * np.sin(ang / 2.0)
    dist, idx = catalog.kd_tree.query(v, distance_upper_bound=max_dist)
    idx = np.asarray(idx, dtype=int)
    idx[~np.isfinite(dist)] = -1
    idx[idx >= len(catalog.ra_deg)] = -1
    return idx


def prepare_gcvs5_state(ra_deg, dec_deg, base_mag, catalog, config):
    """Precompute per-star match indices and phase offsets.

    Algorithm choices:
      - Stars with valid periods but missing epochs get a fixed phase offset
        seeded by (RA, Dec) to keep them deterministic across frames.
      - Stars missing valid periods use `missing_period_mode` ("constant" or
        "random") to select phases.
      - Period notes are only applied for M2 (2x); MN/FN are retained as flags.

    Args:
        ra_deg: `array`, right ascension in degrees.
        dec_deg: `array`, declination in degrees.
        base_mag: `array`, baseline magnitudes (e.g., SSTR7 mv).
        catalog: `Gcvs5Catalog`, catalog to match against.
        config: `dict`, normalized GCVS5 config.

    Returns:
        A `Gcvs5State`, prepared match indices and phase offsets.
    """
    if catalog is None:
        return None

    base_mag = np.asarray(base_mag, dtype=float)
    match_idx = match_gcvs5(ra_deg, dec_deg, catalog, config["match_radius_arcsec"])
    phase_offset = np.zeros_like(match_idx, dtype=float)
    missing_phase = np.full_like(match_idx, np.nan, dtype=float)

    rng_seed = config.get("missing_period_seed", None)
    missing_mode = config.get("missing_period_mode", "constant")

    match_count = int(np.sum(match_idx >= 0))
    valid_count = 0
    log_details = logger.isEnabledFor(logging.DEBUG)

    for i, idx in enumerate(match_idx):
        if idx < 0:
            continue

        max_mag, min_mag, source = _derive_mag_range(
            base_mag[i],
            catalog.max_mag[idx],
            catalog.min_mag[idx],
            catalog.max_mag_limit[idx],
            catalog.min_mag_limit[idx],
        )
        if np.isfinite(max_mag) and np.isfinite(min_mag):
            valid_count += 1

        period = catalog.period[idx]
        period_note = catalog.period_note[idx]
        epoch = catalog.epoch[idx]
        period_limit = catalog.period_limit[idx]

        period = _apply_period_note(period, period_note)
        period_valid = np.isfinite(period) and period > 0.0 and period_limit not in ("<", ">", "(")
        epoch_valid = np.isfinite(epoch)

        if log_details:
            lc_info = get_lightcurve_info(
                catalog.var_type[idx],
                catalog.rise_eclipse_time[idx],
                config.get("template", "auto"),
                base_mag[i],
                catalog.max_mag[idx],
                catalog.min_mag[idx],
                catalog.max_mag_limit[idx],
                catalog.min_mag_limit[idx],
            )
            logger.debug(
                "GCVS5 match sstrc7[%d] ra=%.6f dec=%.6f base_mag=%.3f | "
                "gcvs5[%d] name='%s' type='%s' mag_system='%s' max=%.3f min=%.3f "
                "period=%s epoch=%s lc_type=%s lc_status=%s source=%s",
                i,
                ra_deg[i],
                dec_deg[i],
                base_mag[i],
                idx,
                str(catalog.name[idx]).strip(),
                str(catalog.var_type[idx]).strip(),
                str(catalog.mag_system[idx]).strip(),
                catalog.max_mag[idx],
                catalog.min_mag[idx],
                "nan" if not np.isfinite(period) else f"{period:.6f}",
                "nan" if not np.isfinite(epoch) else f"{epoch:.3f}",
                lc_info["lc_type"],
                lc_info["lc_status"],
                source,
            )

        if period_valid and not epoch_valid:
            seed = _stable_seed(rng_seed, ra_deg[i], dec_deg[i], extra=1)
            rng = np.random.RandomState(seed)
            phase_offset[i] = rng.uniform(0.0, 1.0)

        if (not period_valid) and missing_mode == "constant":
            seed = _stable_seed(rng_seed, ra_deg[i], dec_deg[i], extra=2)
            rng = np.random.RandomState(seed)
            missing_phase[i] = rng.uniform(0.0, 1.0)

    if match_count > 0:
        logger.debug(
            "GCVS5: matched %d stars; %d with usable magnitude ranges.",
            match_count,
            valid_count,
        )
    else:
        logger.debug("GCVS5: matched 0 stars.")

    return Gcvs5State(match_idx=match_idx, phase_offset=phase_offset, missing_phase=missing_phase)


def apply_gcvs5_variability(base_mag, ra_deg, dec_deg, t_jd, catalog, state, config, frame_index=0):
    """Apply GCVS-derived variability to base magnitudes.

    Key choices:
      - The GCVS epoch is interpreted as minimum light for eclipsing/ellipsoidal,
        RV Tau, and RS CVn types, otherwise maximum light.
      - GCVS magnitude ranges override the SSTR7 magnitudes when available.
      - Eclipse durations use the GCVS rise/eclipse percentage (D). If missing,
        a 10% duration is assumed, and any secondary eclipse is placed at phase
        0.5 with the same duration.
      - Period limits ('<', '>', '(') are treated as invalid for periodic phase.
      - Period notes are only applied for M2 (2x); MN/FN are retained as flags.

    Args:
        base_mag: `array`, baseline magnitudes (e.g., SSTR7 mv).
        ra_deg: `array`, right ascension in degrees.
        dec_deg: `array`, declination in degrees.
        t_jd: `float`, observation time in Julian Date (UTC scale).
        catalog: `Gcvs5Catalog`, catalog to match against.
        state: `Gcvs5State`, precomputed match indices and phase offsets.
        config: `dict`, normalized GCVS5 config.
        frame_index: `int`, frame index used for deterministic random phases.

    Returns:
        A `ndarray`, updated magnitudes with variability applied.
    """
    if catalog is None or state is None:
        return base_mag

    base_mag = np.asarray(base_mag, dtype=float)
    out = np.array(base_mag, copy=True)
    template = config.get("template", "auto")
    missing_mode = config.get("missing_period_mode", "constant")
    missing_seed = config.get("missing_period_seed", None)
    log_details = logger.isEnabledFor(logging.DEBUG)

    for i, idx in enumerate(state.match_idx):
        if idx < 0:
            continue

        max_mag, min_mag, _ = _derive_mag_range(
            base_mag[i],
            catalog.max_mag[idx],
            catalog.min_mag[idx],
            catalog.max_mag_limit[idx],
            catalog.min_mag_limit[idx],
        )
        if not np.isfinite(max_mag) or not np.isfinite(min_mag):
            continue

        period = catalog.period[idx]
        period_note = catalog.period_note[idx]
        period_limit = catalog.period_limit[idx]
        period = _apply_period_note(period, period_note)
        period_valid = np.isfinite(period) and period > 0.0 and period_limit not in ("<", ">", "(")

        epoch = catalog.epoch[idx]
        epoch_valid = np.isfinite(epoch)

        if period_valid:
            if epoch_valid:
                epoch_jd = _epoch_to_jd(epoch, catalog.name[idx])
                phase = ((t_jd - epoch_jd) / period) % 1.0 if np.isfinite(epoch_jd) else 0.0
            else:
                phase = ((t_jd / period) + state.phase_offset[i]) % 1.0
        else:
            if missing_mode == "constant":
                if not np.isfinite(state.missing_phase[i]):
                    seed = _stable_seed(missing_seed, ra_deg[i], dec_deg[i], extra=2)
                    rng = np.random.RandomState(seed)
                    state.missing_phase[i] = rng.uniform(0.0, 1.0)
                phase = state.missing_phase[i]
            elif missing_mode == "random":
                seed = _stable_seed(missing_seed, ra_deg[i], dec_deg[i], extra=1000 + frame_index)
                rng = np.random.RandomState(seed)
                phase = rng.uniform(0.0, 1.0)
            else:
                continue

        epoch_is_min = _is_epoch_minimum(catalog.var_type[idx])
        m_gcvs = _template_mag(
            phase,
            max_mag,
            min_mag,
            catalog.sec_min_mag[idx],
            catalog.sec_min_mag_limit[idx],
            epoch_is_min,
            catalog.rise_eclipse_time[idx],
            catalog.rise_eclipse_note[idx],
            catalog.var_type[idx],
            template,
        )

        out[i] = m_gcvs
        if log_details:
            lc_type = get_lightcurve_type(
                catalog.var_type[idx],
                catalog.rise_eclipse_time[idx],
                template,
            )
            logger.debug(
                "GCVS5 frame=%d t_jd=%.6f sstrc7[%d] base_mag=%.3f mag=%.3f "
                "phase=%.6f lc_type=%s name='%s'",
                frame_index,
                t_jd,
                i,
                base_mag[i],
                m_gcvs,
                phase,
                lc_type,
                str(catalog.name[idx]).strip(),
            )

    return out
