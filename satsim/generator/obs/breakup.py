from __future__ import division, print_function, absolute_import

import numpy as np
import math

from skyfield.api import EarthSatellite
from skyfield.constants import AU_KM, DAY_S

from satsim import time
from satsim.image.fpa import mv_to_pe, pe_to_mv


def _rotation_from_vectors(v, local=np.array([[1,0,0], [0,1,0], [0,0,1]])):
    z_world = np.array(v)
    y_world = np.array([0,1,0])
    x_world = np.cross(z_world, y_world)
    y_world = np.cross(x_world, z_world)
    world = np.array([x_world, y_world, z_world])
    M = np.linalg.solve(local,world).T
    return M


def _generate_random_mv_or_size(mv_or_size, scale, n, sigma=1.0, brightness_model='mv'):
    # samples = np.abs(np.random.normal(0, sigma, n))
    samples = np.abs(np.random.lognormal(0, sigma, n))
    samples = samples / np.sum(samples)  # normalize

    if brightness_model == 'mv':
        ZP = 1.0
        pe_target = mv_to_pe(ZP, mv_or_size)
        pe_part = samples * pe_target
        return pe_to_mv(ZP, pe_target * scale[0]), pe_to_mv(ZP, pe_part * scale[1])
    else:
        return mv_or_size * scale[0], samples * mv_or_size * scale[1]


def collision_from_tle(tle, collision_time, radius=37.5, K=0.5,
                       attack_angle='random', attack_velocity=None, attack_velocity_scale=1.0,
                       n=[100,100], target_mv_scale=[0.5, 2.0], rpo_mv_scale=[0.5, 2.0],
                       target_mv=12.0, rpo_mv=12.0, offset=[0.0, 0.0], collision_time_offset=0.0, variable_brightness=True,
                       fragment_angle='random', scale_fragment_velocity=False, brightness_model='mv'):
    """Generates a two object collision configuration from the input `tle`.

    Args:
        tle: `array`, an array containing the SGP4 two line element set.
        collision_time: `array`, UCT time as year, month, day, hour, minute, seconds.
        radius: `float`, one sigma breakup cone radius in degrees. default=37.5
        K: `array or `float`, one sigma velocity transfer multiplier from colliding object. default=0.5
        attack_angle: `string`, `random` or `retrograde`. default='random'
        attack_velocity: `float`, the relative velocity magnitude of colliding object in km/s, if `None` then make equal to target velocity magnitude. default=None
        attack_velocity_scale: `float`, scale `attack_velocity` by this number. default=1.0
        n: `array`, number of particles to generate from target and colliding object. default=[100,100]
        target_mv_scale: `array`, brightness of target and total brightness of generated particles after collision. scale based on original target brightness. default=[0.5,2.0]
        rpo_mv_scale: `array`, brightness of colliding target and total brightness of generated particles after collision. scale based on original colliding target brightness. default=[0.5,2.0]
        target_mv: `float`, brightness of target before collision in visual magnitude. default=12
        rpo_mv: `float`, brightness of colliding object before collision in visual magnitude. default=12
        offset: `array`, row column offset of target position on fpa in normalized coordinates. default=[0,0]
        collision_time_offset: `float`, number of seconds to offset collision time from `collision_time`. default=0
        variable_brightness: `boolean`, if True randomly assign sine variable brightness between 0 to 1 hz. default=True
        fragment_angle: `string`, specified the fragment angle sampling. `random` or `linspace`. default=`random`
        scale_fragment_velocity: `boolean`, if True scale the fragment velocity by cosine of the exit velocity. default=`false`
        brightness_model: `string`, the model to use for brightness calculations. valid options `mv`, `lambertian_sphere` default=`mv`

    Example usage in SatSim configuration::

        "obs": {
            "$generator": {
                "module": "satsim.generator.obs.breakup",
                "function": "collision_from_tle",
                "kwargs": {
                    "n": [
                        237,
                        122
                    ],
                    "tle": {
                        "ref": "geometry.site.track.tle"
                    },
                    "collision_time": {
                        "ref": "geometry.time"
                    },
                    "collision_time_offset": 1.1863922763597272,
                    "K": 0.2717057721240569,
                    "attack_angle": "random",
                    "attack_velocity_scale": 0.2050451649939875,
                    "radius": 24.32114494260032,
                    "target_mv": 8.455948844613438,
                    "rpo_mv": 8.507130983802519,
                    "target_mv_scale": [
                        0.09979647598905288,
                        1.1819183509832407
                    ],
                    "rpo_mv_scale": [
                        0.26064801046837316,
                        1.2893265228436015
                    ],
                    "offset": [
                        0.1736509022094605,
                        0.22670133968196546
                    ],
                    "variable_brightness": true
                }
            }
        }
    """

    if not isinstance(n, (list, np.ndarray)):
        n = [n, n]

    if not isinstance(K, (list, np.ndarray)):
        K = [K, K]

    sat = EarthSatellite(tle[0], tle[1])
    t = time.utc_from_list(collision_time, collision_time_offset)
    collision_time = time.to_utc_list(t)

    # brightness calculations
    target_mv_after, target_particles_mv = _generate_random_mv_or_size(target_mv, target_mv_scale, n[0], np.random.uniform(0.0,1.0), brightness_model=brightness_model)

    if rpo_mv is not None:
        rpo_mv_after, rpo_particles_mv = _generate_random_mv_or_size(rpo_mv, rpo_mv_scale, n[1], np.random.uniform(0.0,1.0), brightness_model=brightness_model)
    else:
        rpo_mv_after, rpo_particles_mv = (None, [])

    # get state vector at time t
    position, velocity0, _, _ = sat._at(t)
    position *= AU_KM
    velocity0 *= AU_KM
    velocity0 /= DAY_S
    velocity0_mag = np.linalg.norm(velocity0)
    velocity0_norm = velocity0 / velocity0_mag

    # generate attack velocity vector
    # note that attack velocity is relative to target
    # for example, a retrograde collision should have 2x velocity magnitude
    if attack_angle == 'retrograde':
        velocity1 = -velocity0_norm
        attack_velocity = 2 * velocity0_mag
        attack_velocity_scale = 1.0
    else:
        velocity1 = np.random.normal(size=3)
        velocity1 /= np.linalg.norm(velocity1)

    if attack_velocity is None:
        attack_velocity = velocity0_mag

    velocity1 *= (attack_velocity * attack_velocity_scale)
    velocity1 += velocity0

    # delta of target and attack velocity
    dv0 = velocity0 - velocity1
    dv0_mag = np.linalg.norm(dv0)

    dv1 = velocity1 - velocity0
    dv1_mag = np.linalg.norm(dv1)

    ndiv2 = n[0]
    n2 = n
    n = n2[0] + n2[1]
    theta = np.random.uniform(0, math.pi * 2, n)

    # geneate random cone particles
    if fragment_angle == 'linspace':
        phi0 = np.linspace(0, np.radians(radius), n2[0])
        phi1 = np.linspace(0, np.radians(radius), n2[1])
        phi = np.append(phi0, phi1)
    else:
        phi = np.random.normal(0, np.radians(radius), n)

    r = np.sin(phi)
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    z = np.cos(r)
    vv = np.stack((x,y,z)).T

    # scale fragment velocity
    if scale_fragment_velocity:
        vv = vv * np.cos(phi)[:, np.newaxis]

    # define and solve for local to world transform
    # scale and rotate for target0
    M0 = _rotation_from_vectors(dv1 / dv1_mag)
    vv[0:ndiv2,:] *= np.random.normal(0, dv1_mag * K[0], n)[0:ndiv2, np.newaxis]
    vv[0:ndiv2,2] = np.abs(vv[0:ndiv2,2])
    vv[0:ndiv2,:] = M0.dot(vv[0:ndiv2,:].T).T + velocity0

    # scale and rotate for target1
    M1 = _rotation_from_vectors(dv0 / dv0_mag)
    vv[ndiv2:,:] *= np.random.normal(0, dv0_mag * K[1], n)[ndiv2:, np.newaxis]
    vv[ndiv2:,2] = np.abs(vv[ndiv2:,2])
    vv[ndiv2:,:] = M1.dot(vv[ndiv2:,:].T).T + velocity1

    obs = {
        "mode": "list",
        "list": [{
            "mode": "twobody",
            "position": list(position),
            "velocity": list(v),
            "epoch": collision_time,
            "offset": offset,
            "events": {
                "create": collision_time
            }
        } for v, m in zip(vv, np.concatenate([target_particles_mv, rpo_particles_mv]))]
    }

    target_obs = None
    if target_mv is not None:
        target_obs = {
            "mode": "twobody",
            "position": list(position),
            "velocity": list(velocity0),
            "epoch": collision_time,
            "offset": offset,
            "events": {
                "update": [
                    {
                        "time": collision_time,
                        "values": {
                        }
                    }
                ]
            }
        }
        obs['list'].append(target_obs)

    rpo_obs = None
    if rpo_mv is not None:
        rpo_obs = {
            "mode": "twobody",
            "position": list(position),
            "velocity": list(velocity1),
            "epoch": collision_time,
            "offset": offset,
            "events": {
                "update": [
                    {
                        "time": collision_time,
                        "values": {
                        }
                    }
                ]
            }
        }
        obs['list'].append(rpo_obs)

    if variable_brightness is True:
        def mv_func(mv_in):
            return {
                "$pipeline": [
                    {
                        "module": "satsim.pipeline",
                        "function": "constant",
                        "kwargs": {
                            "value": mv_in
                        }
                    },
                    {
                        "module": "satsim.pipeline",
                        "function": "sin_add",
                        "kwargs": {
                            "freq": np.random.normal(0, 1.0),
                            "mag_scale": np.random.uniform(0, 7.0)
                        }
                    }
                ]
            }
    else:
        def mv_func(mv_in):
            return mv_in

    if brightness_model == 'mv':

        for i, m in enumerate(np.concatenate([target_particles_mv, rpo_particles_mv])):
            obs['list'][i]['mv'] = mv_func(m)

        if target_obs is not None:
            target_obs['mv'] = mv_func(target_mv_after)
            target_obs['events']['update'][-1]['values']['mv'] = mv_func(target_mv_after)
        if rpo_obs is not None:
            rpo_obs['mv'] = mv_func(rpo_mv_after)
            rpo_obs['events']['update'][-1]['values']['mv'] = mv_func(rpo_mv_after)

    else:
        albedo = 0.3  # TODO: make this a parameter
        for i, m in enumerate(np.concatenate([target_particles_mv, rpo_particles_mv])):
            obs['list'][i]['model'] = {
                'mode': brightness_model,
                'albedo': albedo,
                'size': m
            }

        if target_obs is not None:
            target_obs['model'] = {
                'mode': brightness_model,
                'albedo': albedo,
                'size': target_mv
            }
            target_obs['events']['update'][-1]['values']['model'] = {
                'size': target_mv_after
            }

        if rpo_obs is not None:
            rpo_obs['model'] = {
                'mode': brightness_model,
                'albedo': albedo,
                'size': rpo_mv
            }
            rpo_obs['events']['update'][-1]['values']['model'] = {
                'size': rpo_mv_after
            }

    return obs


def breakup_from_tle(tle, breakup_time, radius=37.5, breakup_velocity=108,
                     n=100, target_mv_scale=[0.5, 2.0], target_mv=12.0,
                     offset=[0.0, 0.0], breakup_time_offset=0.0,
                     variable_brightness=True, brightness_model='mv'):
    """Generates a breakup configuration from the input `tle`.

    Args:
        tle: `array`, an array containing the SGP4 two line element set.
        breakup_time: `array`, UCT time as year, month, day, hour, minute, seconds.
        radius: `float`, one sigma breakup cone radius in degrees. default=37.5
        breakup_velocity: `float`, the relative velocity of generated particles in km/s. default=108
        n: `array`, number of particles to generate. default=100
        target_mv_scale: `array`, brightness of target and total brightness of generated particles after breakup. scale based on original target brightness. default=[0.5,2.0]
        offset: `array`, row column offset of target position on fpa in normalized coordinates. default=12
        breakup_time_offset: `float`, number of seconds to offset breakup time from `breakup`. default=[0,0]
        variable_brightness: `boolean`, if True randomly assign sine variable brightness between 0 to 1 hz. default=True
        brightness_model: `string`, the model to use for brightness calculations. valid options `mv`, `lambertian_sphere` default=`mv`

    Example usage in SatSim configuration::

        "obs": {
            "generator": {
                "module": "satsim.generator.obs.breakup",
                "function": "breakup_from_tle",
                "kwargs": {
                    "n": 39,
                    "tle": {
                        "ref": "geometry.site.track.tle"
                    },
                    "breakup_time": {
                        "ref": "geometry.time"
                    },
                    "breakup_time_offset": 2.0256606954447602,
                    "breakup_velocity": 0.01,
                    "radius": 35.38582838190615,
                    "target_mv": 7.906961060952362,
                    "target_mv_scale": [
                        0.8199894271217592,
                        0.7148000784383131
                    ],
                    "offset": [
                        -0.2808046122089142,
                        0.05683173727181447
                    ],
                    "variable_brightness": false
                }
            }
        }
    """
    obs = collision_from_tle(tle, breakup_time, radius=radius, K=1.0,
                             attack_angle='random', attack_velocity=breakup_velocity, attack_velocity_scale=1.0,
                             n=n, target_mv_scale=target_mv_scale, rpo_mv_scale=None, target_mv=target_mv, rpo_mv=None,
                             offset=offset, collision_time_offset=breakup_time_offset, variable_brightness=variable_brightness,
                             scale_fragment_velocity=False, brightness_model=brightness_model)

    return obs
