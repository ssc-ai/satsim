"""Passive RF document loading and semantic validation."""

from __future__ import annotations

from itertools import combinations
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

from satsim import time
from satsim.geometry.factory import (
    create_observer_from_config,
    create_target_from_config,
    target_id_from_config,
)
from satsim.util.validation import (
    finite_number,
    integer,
    nonnegative_number,
    positive_integer,
    positive_number,
)

from .types import (
    EstimatorParams,
    ObservationRequest,
    PassiveRFRunConfig,
    ReceiverParams,
    TargetRFParams,
)


def _field_of_view(
    config: Mapping[str, Any],
    index: int,
) -> Tuple[
    Optional[Tuple[float, float]],
    Optional[Tuple[float, float]],
]:
    node = config.get('field_of_view', {})
    context = 'passive_rf[{}].field_of_view'.format(index)
    if not isinstance(node, dict):
        raise ValueError('{} must be an object.'.format(context))

    def limits(axis: str):
        value = node.get(axis)
        if value is None:
            return None
        if not isinstance(value, (list, tuple)) or len(value) != 2:
            raise ValueError(
                '{}.{} must contain [minimum, maximum].'.format(context, axis)
            )
        minimum = finite_number('{}.{}[0]'.format(context, axis), value[0])
        maximum = finite_number('{}.{}[1]'.format(context, axis), value[1])
        if minimum > maximum:
            raise ValueError(
                '{}.{} minimum must not exceed maximum.'.format(context, axis)
            )
        return minimum, maximum

    return limits('azimuth'), limits('elevation')


def _sites_and_sensor_configs(
    ssp: Mapping[str, Any],
) -> Tuple[Tuple[Mapping[str, Any], Mapping[str, Any]], ...]:
    geometry = ssp.get('geometry')
    if not isinstance(geometry, dict):
        raise ValueError('Passive RF requires a geometry object.')
    sites = geometry.get('site')
    if not isinstance(sites, list) or len(sites) < 2:
        raise ValueError(
            'Passive RF requires geometry.site to be an array of at least two sites.'
        )
    if not all(isinstance(site, dict) for site in sites):
        raise ValueError('Every geometry.site entry must be an object.')

    raw_config = ssp.get('passive_rf')
    if isinstance(raw_config, dict):
        configs = [raw_config] * len(sites)
    elif isinstance(raw_config, list):
        if len(raw_config) != len(sites):
            raise ValueError(
                'passive_rf array length must match geometry.site array length.'
            )
        if not all(isinstance(config, dict) for config in raw_config):
            raise ValueError('Every passive_rf array entry must be an object.')
        configs = raw_config
    else:
        raise ValueError('Configuration requires a passive_rf object or array.')
    return tuple(zip(sites, configs))


def _validate_ground_site(site: Mapping[str, Any], context: str) -> None:
    if 'tle' in site or 'tle1' in site or 'tle2' in site:
        raise ValueError('{} must be a ground site.'.format(context))
    if 'lat' not in site or 'lon' not in site:
        raise ValueError('{} requires lat and lon.'.format(context))
    for key, limit in (('lat', 90.0), ('lon', 180.0)):
        value = site[key]
        if isinstance(value, str):
            continue
        value = finite_number('{}.{}'.format(context, key), value)
        if value < -limit or value > limit:
            raise ValueError(
                '{}.{} must be in [{}, {}].'.format(context, key, -limit, limit)
            )


def load_receivers(
    ssp: Mapping[str, Any],
) -> Tuple[Dict[str, ReceiverParams], Dict[str, Any]]:
    """Build receiver parameters and standard SatSim observer objects."""
    receivers: Dict[str, ReceiverParams] = {}
    observers: Dict[str, Any] = {}
    for index, (site, config) in enumerate(_sites_and_sensor_configs(ssp)):
        context = 'geometry.site[{}]'.format(index)
        name = site.get('name')
        if not isinstance(name, str) or not name.strip():
            raise ValueError('{}.name must be a non-empty string.'.format(context))
        name = name.strip()
        if name in receivers:
            raise ValueError("Duplicate geometry.site name: '{}'".format(name))
        _validate_ground_site(site, context)
        finite_number('{}.alt'.format(context), site.get('alt', 0.0))

        az_limits, el_limits = _field_of_view(config, index)
        receivers[name] = ReceiverParams(
            name=name,
            antenna_gain_dbi=finite_number(
                'passive_rf[{}].antenna_gain'.format(index),
                config.get('antenna_gain', 30.0),
            ),
            noise_figure_db=finite_number(
                'passive_rf[{}].noise_figure'.format(index),
                config.get('noise_figure', 2.0),
            ),
            system_losses_db=nonnegative_number(
                'passive_rf[{}].system_losses'.format(index),
                config.get('system_losses', 1.0),
            ),
            antenna_noise_temperature_k=positive_number(
                'passive_rf[{}].antenna_noise_temperature'.format(index),
                config.get('antenna_noise_temperature', 290.0),
            ),
            az_limits=az_limits,
            el_limits=el_limits,
        )
        try:
            observers[name] = create_observer_from_config(site)
        except (TypeError, ValueError) as exc:
            raise ValueError('{} is invalid: {}'.format(context, exc)) from exc
    return receivers, observers


def load_targets(
    ssp: Mapping[str, Any],
    default_time: Sequence[Any],
) -> Tuple[Dict[int, TargetRFParams], Dict[int, Any]]:
    """Load RF properties and propagated objects from the standard target list."""
    obs = ssp.get('geometry', {}).get('obs')
    if not isinstance(obs, dict) or obs.get('mode') != 'list':
        raise ValueError("Passive RF requires geometry.obs.mode == 'list'.")
    entries = obs.get('list')
    if isinstance(entries, dict):
        entries = [entries]
    if not isinstance(entries, list) or not entries:
        raise ValueError('geometry.obs.list must contain at least one target.')

    targets: Dict[int, TargetRFParams] = {}
    target_objects: Dict[int, Any] = {}
    required_rf_fields = {'frequency', 'bandwidth', 'eirp'}
    recognized_rf_fields = required_rf_fields | {'band'}
    for index, record in enumerate(entries):
        context = 'geometry.obs.list[{}]'.format(index)
        if not isinstance(record, dict):
            raise ValueError('{} must be an object.'.format(context))
        present = recognized_rf_fields & set(record)
        if not present:
            raise ValueError(
                '{} requires frequency, bandwidth, and eirp for passive RF.'.format(
                    context
                )
            )
        missing = required_rf_fields - set(record)
        if missing:
            raise ValueError(
                '{} RF target is missing: {}.'.format(
                    context,
                    ', '.join(sorted(missing)),
                )
            )

        try:
            target_id = positive_integer(
                '{}.id'.format(context),
                target_id_from_config(record),
            )
        except (TypeError, ValueError) as exc:
            raise ValueError('{} has no valid target id: {}'.format(context, exc)) from exc
        if target_id in targets:
            raise ValueError('Duplicate passive RF target id: {}'.format(target_id))

        band = record.get('band')
        if band is not None and (not isinstance(band, str) or not band):
            raise ValueError('{}.band must be a non-empty string.'.format(context))
        name = record.get('name')
        if name is not None and (not isinstance(name, str) or not name):
            raise ValueError('{}.name must be a non-empty string.'.format(context))

        try:
            target_object = create_target_from_config(record, default_time=default_time)
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError('{} is invalid: {}'.format(context, exc)) from exc
        if target_object is None:
            raise ValueError(
                '{} must use a ranging-capable TLE, twobody, or ephemeris mode.'.format(
                    context
                )
            )

        targets[target_id] = TargetRFParams(
            target_id=target_id,
            frequency_hz=positive_number(
                '{}.frequency'.format(context), record.get('frequency')
            ),
            bandwidth_hz=positive_number(
                '{}.bandwidth'.format(context), record.get('bandwidth')
            ),
            eirp_dbw=finite_number('{}.eirp'.format(context), record.get('eirp')),
            name=name,
            band=band,
        )
        target_objects[target_id] = target_object
    if not targets:
        raise ValueError(
            'geometry.obs.list contains no targets with frequency, bandwidth, and eirp.'
        )
    return targets, target_objects


def _uncertainty(
    estimator: Mapping[str, Any],
    context: str,
) -> Tuple[float, float, float, float]:
    node = estimator.get('uncertainty', {})
    if not isinstance(node, dict):
        raise ValueError('{}.uncertainty must be an object.'.format(context))
    tdoa = node.get('tdoa', {})
    fdoa = node.get('fdoa', {})
    if not isinstance(tdoa, dict) or not isinstance(fdoa, dict):
        raise ValueError('{}.uncertainty tdoa/fdoa must be objects.'.format(context))
    return (
        nonnegative_number('{}.tdoa.floor'.format(context), tdoa.get('floor', 0.0)),
        nonnegative_number('{}.fdoa.floor'.format(context), fdoa.get('floor', 0.0)),
        nonnegative_number('{}.tdoa.scale'.format(context), tdoa.get('scale', 1.0)),
        nonnegative_number('{}.fdoa.scale'.format(context), fdoa.get('scale', 1.0)),
    )


def _estimator_controls(config: Mapping[str, Any], index: int) -> Tuple[Any, ...]:
    estimator = config.get('estimator', {})
    if not isinstance(estimator, dict):
        raise ValueError('passive_rf[{}].estimator must be an object.'.format(index))
    rms_model = estimator.get('rms_bandwidth', 'rect')
    if rms_model not in ('rect', 'gauss'):
        raise ValueError("rms_bandwidth must be 'rect' or 'gauss'.")
    return (
        positive_number('coherent_time', estimator.get('coherent_time', 0.005)),
        rms_model,
        nonnegative_number('caf_loss', estimator.get('caf_loss', 3.0)),
    )


def _timing_controls(config: Mapping[str, Any], index: int) -> Tuple[Any, ...]:
    time_config = config.get('time', {})
    if not isinstance(time_config, dict):
        raise ValueError('passive_rf[{}].time must be an object.'.format(index))
    return (
        positive_number('time.dwell', time_config.get('dwell', 1.0)),
        nonnegative_number('time.gap', time_config.get('gap', 0.0)),
        positive_integer('num_frames', config.get('num_frames', 1)),
        integer('seed', config.get('seed', 42)),
    )


def load_estimator(
    ssp: Mapping[str, Any],
    receiver_ids: Sequence[str],
) -> Tuple[EstimatorParams, Tuple[float, float, int, int]]:
    pairs = _sites_and_sensor_configs(ssp)
    first_estimator = _estimator_controls(pairs[0][1], 0)
    first_timing = _timing_controls(pairs[0][1], 0)
    mappings = ({}, {}, {}, {})
    for index, ((_, config), receiver_id) in enumerate(zip(pairs, receiver_ids)):
        if _estimator_controls(config, index) != first_estimator:
            raise ValueError(
                'All passive_rf array entries must use the same estimator controls.'
            )
        if _timing_controls(config, index) != first_timing:
            raise ValueError(
                'All passive_rf array entries must use the same timing and run controls.'
            )
        estimator = config.get('estimator', {})
        for mapping, value in zip(
            mappings,
            _uncertainty(estimator, 'passive_rf[{}].estimator'.format(index)),
        ):
            mapping[receiver_id] = value

    coherent_time, rms_model, caf_loss = first_estimator
    dwell, _, _, _ = first_timing
    return EstimatorParams(
        integration_time_s=dwell,
        coherent_time_s=coherent_time,
        rms_bw_from_rect=rms_model == 'rect',
        caf_loss_db=caf_loss,
        tdoa_floor_by_sensor=mappings[0],
        fdoa_floor_by_sensor=mappings[1],
        tdoa_scale_by_sensor=mappings[2],
        fdoa_scale_by_sensor=mappings[3],
    ), first_timing


def _geometry_start(ssp: Mapping[str, Any]) -> Tuple[Any, ...]:
    value = ssp.get('geometry', {}).get('time')
    if not isinstance(value, list) or len(value) != 6:
        raise ValueError(
            'geometry.time must be [year, month, day, hour, minute, second].'
        )
    try:
        normalized = tuple(int(item) for item in value[:5]) + (
            finite_number('geometry.time[5]', value[5]),
        )
        time.utc_from_list(list(normalized))
        return normalized
    except (TypeError, ValueError) as exc:
        raise ValueError('geometry.time is invalid: {}'.format(value)) from exc


def load_run_config(ssp: Mapping[str, Any]) -> PassiveRFRunConfig:
    if positive_integer('sim.samples', ssp.get('sim', {}).get('samples', 1)) != 1:
        raise ValueError('Passive RF requires sim.samples == 1 in v0.26.0.')
    if not isinstance(ssp.get('geometry'), dict):
        raise ValueError('Passive RF requires a geometry object.')
    start = _geometry_start(ssp)
    receivers, observers = load_receivers(ssp)
    targets, target_objects = load_targets(ssp, start)
    receiver_ids = tuple(receivers)
    estimator, timing = load_estimator(ssp, receiver_ids)
    dwell, gap, num_frames, seed = timing
    frame_times = tuple(
        time.utc_from_list(
            list(start),
            delta_sec=frame * (dwell + gap) + 0.5 * dwell,
        )
        for frame in range(num_frames)
    )

    observations = []
    source_index = 0
    for frame_time in frame_times:
        for sensor1, sensor2 in combinations(receiver_ids, 2):
            for target_id in targets:
                observations.append(ObservationRequest(
                    target_id=target_id,
                    sensor1=sensor1,
                    sensor2=sensor2,
                    time=frame_time,
                    source_index=source_index,
                ))
                source_index += 1

    return PassiveRFRunConfig(
        receivers=receivers,
        observers=observers,
        targets=targets,
        target_objects=target_objects,
        estimator=estimator,
        observations=tuple(observations),
        frame_times=frame_times,
        seed=seed,
    )


__all__ = [
    'load_estimator',
    'load_receivers',
    'load_run_config',
    'load_targets',
]
