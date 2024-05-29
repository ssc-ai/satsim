"""Tests for `satsim.generator`."""

import numpy as np
from satsim.generator.obs import geometry, io, cso, breakup, rpo


def test_cone():

    c = geometry.cone(n=50)
    output = c['list']

    for i in range(50):
        assert(output[i]['mode'] == 'line')
        assert('origin' in output[i])
        assert('velocity' in output[i])
        assert('mv' in output[i])


def test_circle():

    c = geometry.circle(n=50)
    output = c['list']

    for i in range(50):
        assert(output[i]['mode'] == 'line')
        assert('origin' in output[i])
        assert('velocity' in output[i])
        assert('mv' in output[i])


def test_sphere():

    c = geometry.sphere(n=50)
    output = c['list']

    for i in range(50):
        assert(output[i]['mode'] == 'line')
        assert('origin' in output[i])
        assert('velocity' in output[i])
        assert('mv' in output[i])


def test_tle_file():

    c = io.tle_file('./tests/geo.txt', lines=3)
    output = c['list']

    assert(len(output) == 513)

    for i in range(513):
        assert(output[i]['mode'] == 'tle')
        assert('tle1' in output[i])
        assert('tle2' in output[i])
        assert('mv' in output[i])


def test_cso():

    output = cso.one()

    assert(len(output) == 2)

    for i in range(2):
        assert(output[i]['mode'] == 'line')
        assert('origin' in output[i])
        assert('velocity' in output[i])
        assert('mv' in output[i])

    assert(output[1]['origin'][0] == 0.5)
    assert(output[1]['origin'][1] == 0.51)

    output = cso.one(velocity=[0,1])

    assert(len(output) == 2)

    for i in range(2):
        assert(output[i]['mode'] == 'line')
        assert('origin' in output[i])
        assert('velocity' in output[i])
        assert('mv' in output[i])

    assert(output[1]['origin'][0] == 0.49)
    assert(output[1]['origin'][1] == 0.5)


def test_breakup():

    n = 50
    tle = [
        "1 36411U 10008A   15115.45079343  .00000069  00000-0  00000+0 0  9992",
        "2 36411 000.0719 125.6855 0001927 217.7585 256.6121 01.00266852 18866"
    ]
    collision_time = [2015, 4, 24, 9, 7, 30.128]
    breakup_velocity = 0.05
    radius = 45.0

    output = breakup.breakup_from_tle(tle, collision_time, breakup_velocity, radius, n)

    assert(len(output['list']) == n + 1)

    target_size = 12.0
    target_scale = [0.5, 2.0]
    output = breakup.breakup_from_tle(tle, collision_time, breakup_velocity, radius, n, target_scale, target_size, brightness_model='lambertian_sphere')
    assert(len(output['list']) == n + 1)

    total_size = 0.0
    for item in output['list']:
        model = item.get('model')
        size = model.get('size')
        total_size += size

    expected_size = target_size + target_size * target_scale[1]

    np.testing.assert_almost_equal(total_size, expected_size)
    np.testing.assert_almost_equal(output['list'][-1]['events']['update'][0]['values']['model']['size'], target_size * target_scale[0])


def test_collision():

    n = 50
    tle = [
        "1 36411U 10008A   15115.45079343  .00000069  00000-0  00000+0 0  9992",
        "2 36411 000.0719 125.6855 0001927 217.7585 256.6121 01.00266852 18866"
    ]
    collision_time = [2015, 4, 24, 9, 7, 30.128]
    K = 1.0
    attach_angle = 'random'
    attack_velocity = None
    attack_velocity_scale = 1.0
    radius = 45

    output = breakup.collision_from_tle(tle, collision_time, radius, K, attach_angle, attack_velocity, attack_velocity_scale, n)
    assert(len(output['list']) == n * 2 + 2)

    attach_angle = 'retrograde'
    attack_velocity = 1.0
    output = breakup.collision_from_tle(tle, collision_time, radius, K, attach_angle, attack_velocity, attack_velocity_scale, n)
    assert(len(output['list']) == n * 2 + 2)

    output = breakup.collision_from_tle(tle, collision_time, radius, K, attach_angle, attack_velocity, attack_velocity_scale,
                                        [10, 20], fragment_angle='linspace', scale_fragment_velocity=True, variable_brightness=False)
    assert(len(output['list']) == 10 + 20 + 2)

    target_size = 12.0
    target_scale = [0.5, 2.0]
    rpo_size = 6.0
    rpo_scale = [0.5, 2.0]
    output = breakup.collision_from_tle(tle, collision_time, radius, K, attach_angle, attack_velocity, attack_velocity_scale,
                                        [10, 20], target_scale, rpo_scale, target_size, rpo_size,
                                        fragment_angle='linspace', scale_fragment_velocity=True, variable_brightness=False, brightness_model='lambertian_sphere')
    assert(len(output['list']) == 10 + 20 + 2)

    total_size = 0.0
    for item in output['list']:
        model = item.get('model')
        size = model.get('size')
        total_size += size

    expected_size = target_size + target_size * target_scale[1] + rpo_size + rpo_size * rpo_scale[1]

    np.testing.assert_almost_equal(total_size, expected_size)
    np.testing.assert_almost_equal(output['list'][-2]['events']['update'][0]['values']['model']['size'], target_size * target_scale[0])
    np.testing.assert_almost_equal(output['list'][-1]['events']['update'][0]['values']['model']['size'], rpo_size * rpo_scale[0])


def test_rpo():

    tle = [
        "1 36411U 10008A   15115.45079343  .00000069  00000-0  00000+0 0  9992",
        "2 36411 000.0719 125.6855 0001927 217.7585 256.6121 01.00266852 18866"
    ]
    epoch = [2015, 4, 24, 9, 7, 30.128]

    output = rpo.rpo_from_tle(tle, epoch, delta_distance=5, delta_position_direction='random',
                              delta_velocity=0, delta_velocity_direction='random',
                              target_mv=12.0, rpo_mv=12.0, offset=[0.0, 0.0])
    assert(len(output['list']) == 2)

    output = rpo.rpo_from_tle(tle, epoch, delta_distance=5, delta_position_direction=np.array([0,0,1]),
                              delta_velocity=0, delta_velocity_direction=np.array([0,0,0.1]),
                              target_mv=12.0, rpo_mv=12.0, offset=[0.0, 0.0])
    assert(len(output['list']) == 2)
