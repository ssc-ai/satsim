"""Cloud preset constants and public config surface."""

CLOUD_TYPE_NAMES = ('patchy', 'cellular', 'veil', 'sheet', 'fog')
CUSTOM_CLOUD_TYPE = 'custom'
DEFAULT_CLOUD_RANGE_M = 3000.0
SEED_MAX = 2 ** 32

SOURCE_BRIGHTENING = {
    'artificial': {'gain': 6.0, 'scale_height_m': 4000.0},
    'lunar': {'gain': 0.35, 'scale_height_m': 8000.0},
    'solar': {'gain': 0.25, 'scale_height_m': 12000.0},
}
DEFAULT_SOURCE_GAINS = {
    name: params['gain']
    for name, params in SOURCE_BRIGHTENING.items()
}

LUNAR_DIRECT_BRIGHTENING = {
    'gain': 8.0,
    'scale_height_m': 12000.0,
}

SOLAR_DIRECT_BRIGHTENING = {
    'gain': 5.0,
    'scale_height_m': 12000.0,
    'twilight_min_sun_el_deg': -18.0,
}

PUBLIC_LAYER_FIELDS = {
    'type',
    'name',
    'enabled',
    'seed',
    'coverage',
    'texture',
    'optical',
    'geometry',
    'motion',
    'illumination',
    'feature_scales_m',
    'density_edge_width',
    'density_floor',
    'brightness',
    'range',
    'altitude',
    'wind_speed',
    'wind_direction',
    'texture_contrast',
    'locality_degree',
    'tau_min',
    'tau_max',
    'tau_gamma',
}

GROUP_FIELDS = {
    'texture': {
        'feature_scales_m': 'feature_scales_m',
        'scales_m': 'feature_scales_m',
        'density_edge_width': 'density_edge_width',
        'edge_width': 'density_edge_width',
        'density_floor': 'density_floor',
        'floor': 'density_floor',
        'texture_contrast': 'texture_contrast',
        'contrast': 'texture_contrast',
        'locality_degree': 'locality_degree',
    },
    'optical': {
        'tau_min': 'tau_min',
        'tau_max': 'tau_max',
        'tau_gamma': 'tau_gamma',
    },
    'geometry': {
        'range': 'range',
        'range_km': 'range',
        'altitude': 'altitude',
        'altitude_km': 'altitude',
    },
    'motion': {
        'wind_speed': 'wind_speed',
        'speed_m_per_s': 'wind_speed',
        'wind_direction': 'wind_direction',
        'direction_deg': 'wind_direction',
    },
    'illumination': {
        'brightness': 'brightness',
        'brightness_mag_arcsec2': 'brightness',
        'source_gains': 'source_gains',
    },
}

INTERNAL_DEFAULTS = {
    'coverage_mode': 'density',
    'mask_threshold': 0.02,
    'min_feature_scale_px': 4.0,
    'amplitude_decay': 0.7,
}

CUSTOM_DEFAULTS = dict(INTERNAL_DEFAULTS, **{
    'seed_offset': 0,
    'coverage': 0.5,
    'feature_scales_m': (40.0, 80.0, 160.0, 320.0, 640.0),
    'density_edge_width': 0.12,
    'density_floor': None,
    'brightness': None,
    'range': DEFAULT_CLOUD_RANGE_M / 1000.0,
    'altitude': None,
    'wind_speed': 0.0,
    'wind_direction': 0.0,
    'texture_contrast': 1.0,
    'locality_degree': 1,
    'tau_min': 0.02,
    'tau_max': 3.5,
    'tau_gamma': 1.15,
})

PRESETS = {
    'patchy': dict(INTERNAL_DEFAULTS, **{
        'seed_offset': 101,
        'coverage': 0.45,
        'feature_scales_m': (
            2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0, 256.0,
            512.0, 1024.0, 2048.0, 4096.0, 8192.0, 16384.0,
        ),
        'min_feature_scale_px': 12.0,
        'density_edge_width': 0.08,
        'density_floor': None,
        'mask_threshold': 0.03,
        'amplitude_decay': 0.65,
        'texture_contrast': 1.0,
        'locality_degree': 2,
        'tau_min': 0.2,
        'tau_max': 9.0,
        'tau_gamma': 0.55,
    }),
    'cellular': dict(INTERNAL_DEFAULTS, **{
        'seed_offset': 202,
        'coverage': 0.55,
        'feature_scales_m': (960.0, 1920.0, 3840.0, 7680.0, 15360.0),
        'density_edge_width': 0.16,
        'density_floor': None,
        'mask_threshold': 0.02,
        'amplitude_decay': 0.85,
        'texture_contrast': 0.9,
        'locality_degree': 1,
        'tau_min': 0.02,
        'tau_max': 3.5,
        'tau_gamma': 1.15,
    }),
    'veil': dict(INTERNAL_DEFAULTS, **{
        'seed_offset': 303,
        'coverage': 0.40,
        'feature_scales_m': (640.0, 1280.0, 2560.0, 5120.0, 10240.0, 20480.0),
        'density_edge_width': 0.42,
        'density_floor': 0.06,
        'mask_threshold': 0.02,
        'amplitude_decay': 1.1,
        'texture_contrast': 0.6,
        'locality_degree': 1,
        'tau_min': 0.01,
        'tau_max': 0.9,
        'tau_gamma': 1.15,
    }),
    'sheet': dict(INTERNAL_DEFAULTS, **{
        'seed_offset': 404,
        'coverage': 0.90,
        'feature_scales_m': (2048.0, 4096.0, 8192.0, 16384.0, 32768.0),
        'min_feature_scale_px': 8.0,
        'density_edge_width': 0.34,
        'density_floor': 0.4,
        'mask_threshold': 0.02,
        'amplitude_decay': 1.3,
        'texture_contrast': 0.52,
        'locality_degree': 1,
        'tau_min': 0.5,
        'tau_max': 10.0,
        'tau_gamma': 1.15,
    }),
    'fog': dict(INTERNAL_DEFAULTS, **{
        'seed_offset': 505,
        'coverage': 0.95,
        'feature_scales_m': (8192.0, 16384.0, 32768.0, 65536.0),
        'density_edge_width': 0.45,
        'density_floor': 0.45,
        'mask_threshold': 0.01,
        'amplitude_decay': 1.6,
        'texture_contrast': 0.28,
        'locality_degree': 1,
        'tau_min': 0.05,
        'tau_max': 1.5,
        'tau_gamma': 1.15,
    }),
}
