import numpy as np
from satsim.geometry.astrometric import angle_between, distance_between
from satsim.geometry.astrometric import load_sun


def lambertian_sphere_to_mv(albedo, distance, radius, phase_angle):
    """Applies lambertian sphere approximation to convert target brightness
    to visual magnitudes based on sun brightness of -26.74.

    Args:
        albedo: `float`, The ratio of reflected light to incident light
            off of the object's surface
        distance: `float`, distance to object in meters
        radius: `float`, radius of sphere in meters
        phase_angle: `float`, the angle between observer, object, and sun in degrees

    Returns:
        A `float`, calculated visual magnitude
    """
    phase_angle = np.deg2rad(phase_angle)

    mv_sun = -26.74

    # Lambertian sphere approximation.
    phase_factor = np.sin(phase_angle) + (np.pi - phase_angle) * np.cos(phase_angle)
    intensity = phase_factor * (2 * albedo * (radius * radius)) / (3 * np.pi * (distance * distance))

    # Convert intensities to magnitudes
    mvVector = mv_sun - 2.5 * np.log10(intensity)

    return mvVector


def model_to_mv(observer, target, model, time):
    """Calculates the phase angle between the observer, target, and sun.

    Args:
        observer: `object`, observer as a Skyfield object
        target: `object`, target as a Skyfield object
        time: `object`, Skyfield time

    Returns:
        A `float`, the phase angle in degrees
    """
    if model['mode'] == 'lambertian_sphere':
        return lambertian_sphere_model_to_mv(observer, target, model, time)
    else:
        return None


def lambertian_sphere_model_to_mv(observer, target, model, time):
    """Calculates the phase angle between the observer, target, and sun.

    Args:
        observer: `object`, observer as a Skyfield object
        target: `object`, target as a Skyfield object
        sun: `object`, Skyfield sun object
        time: `object`, Skyfield time

    Returns:
        A `float`, the phase angle in degrees
    """
    sun = load_sun()
    distance = model['distance'] * 1000.0 if 'distance' in model else distance_between(target, observer, time) * 1000.0
    phase_angle = model['phase_angle'] if 'phase_angle' in model else angle_between(target, observer, sun, time)
    diameter = model['diameter'] if 'diameter' in model else model['size']

    return lambertian_sphere_to_mv(model['albedo'], distance, diameter * 0.5, phase_angle)
