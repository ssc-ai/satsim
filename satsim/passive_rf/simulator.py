"""SatSim passive RF analytical simulator orchestration."""

from __future__ import annotations

import os
from datetime import datetime
from typing import Any, Dict

from satsim.config import load_json, load_yaml, save_debug, save_json, transform
from satsim.io.analytical import format_ob_time, save as save_observations

from .config import load_run_config
from .model import simulate_run_config


def _run_directory(output_dir: str) -> str:
    timestamp = datetime.now().isoformat().replace(':', '-')
    run_dir = os.path.join(output_dir, timestamp)
    os.makedirs(run_dir, exist_ok=False)
    return run_dir


def simulate(ssp: Dict[str, Any], output_dir: str = './') -> str:
    """Validate, simulate, and write passive RF analytical observations."""
    config = load_run_config(ssp)

    # Compute before creating output so configuration/propagation failures do
    # not leave a partial run directory.
    records = simulate_run_config(config)

    run_dir = _run_directory(output_dir)
    records_by_time = {}
    for record in records:
        records_by_time.setdefault(record['obTime'], []).append(record)
    for frame_index, when in enumerate(config.frame_times):
        save_observations(
            run_dir,
            frame_index,
            records_by_time.get(format_ob_time(when), []),
        )
    save_json(os.path.join(run_dir, 'config.json'), ssp)
    return run_dir


def simulate_from_file(config_file: str, output_dir: str = './') -> str:
    """Load, transform, and run a passive RF JSON or YAML configuration."""
    config_file_lower = config_file.lower()
    if config_file_lower.endswith('.json'):
        ssp = load_json(config_file)
    elif config_file_lower.endswith(('.yml', '.yaml')):
        ssp = load_yaml(config_file)
    else:
        raise ValueError('Config file must be JSON or YAML.')
    input_dir = os.path.dirname(os.path.abspath(config_file))
    transformed, stages = transform(ssp, input_dir, with_debug=True)
    run_dir = simulate(transformed, output_dir)
    save_debug(stages, run_dir)
    return run_dir


__all__ = ['simulate', 'simulate_from_file']
