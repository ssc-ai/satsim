"""Coordinate conversions between oversampled and detector pixel grids.

Pixel positions use zero-based array coordinates, where integer values denote
pixel centers.  Differences and extents (for example velocities and padding)
must be scaled separately and do not use the half-pixel terms below.
"""


def oversampled_to_detector(position, osf):
    """Convert an oversampled pixel-center position to detector coordinates."""
    return (position + 0.5) / osf - 0.5


def detector_to_oversampled(position, osf):
    """Convert a detector pixel-center position to oversampled coordinates."""
    return (position + 0.5) * osf - 0.5


def normalized_to_detector(position, size):
    """Convert a normalized image-edge coordinate to detector coordinates."""
    return position * size - 0.5


def detector_to_normalized(position, size):
    """Convert a detector coordinate to normalized image-edge coordinates."""
    return (position + 0.5) / size


def normalized_to_oversampled(position, size):
    """Convert a normalized image-edge coordinate to an oversampled position."""
    return position * size - 0.5


def delta_oversampled_to_detector(delta, osf):
    """Scale a vector or extent from the oversampled grid to the detector."""
    return delta / osf


def delta_detector_to_oversampled(delta, osf):
    """Scale a vector or extent from the detector to the oversampled grid."""
    return delta * osf
