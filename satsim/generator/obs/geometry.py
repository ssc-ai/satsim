from __future__ import division, print_function, absolute_import

import numpy as np
import math


def circle(n=10, origin=[0.5,0.5], velocity=0.01, mv=1.0, t=0.0):
    """Generates objects that emanate from a point at equal angles.

    Args:
        n: `int`, number of objects
        origin: `array`, row and column coordinate in normalized image space
        velocity: `float`, velocity in pixel per second
        mv: `float`, brightness of each object
        t: `t`, seconds elapsed from epoch

    Example usage in SatSim configuration::

        "obs": {
            "generator": {
                "module": "satsim.generator.obs.geometry",
                "function": "circle",
                "kwargs": {
                    "n": 100,
                    "t": 0,
                    "velocity": 1.0
                    "origin": [0.5, 0.5],
                    "mv": 15.0
                }
            }
        }
    """
    r = np.linspace(0, math.pi * 2, n, endpoint=False)
    vxx = np.array([math.cos(x) * velocity for x in r])
    vyy = np.array([math.sin(x) * velocity for x in r])

    xx = vxx * t + origin[0]
    yy = vyy * t + origin[1]

    obs = {
        "mode": "list",
        "list": [{
            "mode": "line",
            "origin": [px, py],
            "velocity": [vx, vy],
            "mv": mv
        } for vx,vy,px,py in zip(vxx, vyy, xx, yy)]
    }

    return obs


def cone(n=10, origin=[0.5,0.5], velocity=[3.0, 5.0], direction=[0.0, 15.0], mv=[10.0, 16.0], t=0.0, target_mv=15.0, rpo_mv=15.0, collision_time=30.0):
    """Generates objects that emanate from a point in a random normal distribution about angle `direction[0]`.

    Args:
        n: `int`, number of objects
        origin: `array`, row and column coordinate in normalized image space
        velocity: `array`, mean velocity and sigma expressed as `velocity[0] / velocity[1]` of
            normal distribution in pixel per second
        direction: `array`, mean direction and sigma of normal distribution in degrees
        mv: `array`, min and max visual magnitude of a uniform distribution
        t: `t`, seconds elapsed from epoch

    Example usage in SatSim configuration::

        "obs": {
            "generator": {
                "module": "satsim.generator.obs.geometry",
                "function": "cone",
                "kwargs": {
                    "n": { "sample": "random.randint", "low": 500, "high": 1000 },
                    "t": 0,
                    "origin": [
                        { "sample": "random.uniform", "low": 0.2, "high": 0.8 },
                        { "sample": "random.uniform", "low": 0.2, "high": 0.8 }],
                    "direction": [
                        { "sample": "random.uniform", "low": 0, "high": 360 },
                        { "sample": "random.uniform", "low": 10, "high": 30 }],
                    "velocity": [
                        { "sample": "random.uniform", "low": 1.0, "high": 5.0},
                        { "sample": "random.uniform", "low": 3.0, "high": 5.0 }],
                    "mv": [15.0, 17.0]
                }
            }
        }
    """
    d = direction[0] * math.pi / 180.0
    r = direction[1] * math.pi / 180.0

    rr = np.random.normal(d, r, n)
    vv = np.abs(np.random.normal(velocity[0], velocity[0] / velocity[1], n))
    vxx = np.array([math.cos(r) * v for r, v in zip(rr, vv)])
    vyy = np.array([math.sin(r) * v for r, v in zip(rr, vv)])

    xx = vxx * t + origin[1]
    yy = vyy * t + origin[0]

    obs = {
        "mode": "list",
        "list": [{
            "mode": "line",
            "origin": [py, px],
            "velocity": [vy, vx],
            "mv": np.random.uniform(mv[0], mv[1]),
            "epoch": -collision_time,
            "events": {
                "create": collision_time
            }
        } for vx,vy,px,py in zip(vxx, vyy, xx, yy)]
    }

    if target_mv is not None:
        obs['list'].append({
            "mode": "line",
            "origin": origin,
            "velocity": [0, 0],
            "mv": target_mv
        })

    if rpo_mv is not None:
        obs['list'].append({
            "mode": "line-polar",
            "origin": origin,
            "epoch": -collision_time,
            "velocity": [direction[0], velocity[0] + 2 * velocity[0] / velocity[1]],
            "mv": rpo_mv
        })

    return obs


def sphere(n=10, origin=[0.5,0.5], velocity=[0.01, 5.0], mv=[10.0, 16.0], t=0.0):
    """Generates objects that emanate from a point at random angles.

    Args:
        n: `int`, number of objects
        origin: `array`, row and column coordinate in normalized image space
        velocity: `array`, mean velocity and sigma of a normal distribution in pixel per second
        mv: `array`, min and max visual magnitude of a uniform distribution
        t: `t`, seconds elapsed from epoch

    Example usage in SatSim configuration::

        "obs": {
            "$generator": {
                "module": "satsim.generator.obs.geometry",
                "function": "sphere",
                "kwargs": {
                    "n": { "sample": "random.randint", "low": 500, "high": 1000 },
                    "t": 0,
                    "origin": [
                        { "sample": "random.uniform", "low": 0.2, "high": 0.8 },
                        { "sample": "random.uniform", "low": 0.2, "high": 0.8 }],
                    "velocity": [
                        { "sample": "random.uniform", "low": 0.0, "high": 1.0},
                        { "sample": "random.uniform", "low": 3.0, "high": 10.0 }],
                    "mv": [15.0, 17.0]
                }
            }
        }
    """
    rr = np.random.uniform(0, math.pi * 2, n)
    vv = np.random.uniform(velocity[0], velocity[1], n)
    vxx = np.array([math.cos(r) * v for r, v in zip(rr, vv)])
    vyy = np.array([math.sin(r) * v for r, v in zip(rr, vv)])

    xx = vxx * t + origin[0]
    yy = vyy * t + origin[1]

    obs = {
        "mode": "list",
        "list": [{
            "mode": "line",
            "origin": [px, py],
            "velocity": [vx, vy],
            "mv": np.random.uniform(mv[0], mv[1])
        } for vx,vy,px,py in zip(vxx, vyy, xx, yy)]
    }

    return obs
