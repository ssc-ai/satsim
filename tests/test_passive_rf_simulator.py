import json
from datetime import datetime
from pathlib import Path

from click import ClickException
from click.testing import CliRunner
import pytest
import yaml

from satsim import cli
from satsim.passive_rf.simulator import simulate_from_file


CONFIG_PATH = Path(__file__).parent / 'config_passive_rf.json'


def _observation_files(run_dir):
    directory = Path(run_dir) / 'AnalyticalObservations'
    return sorted(directory.glob('*.json'))


def test_simulate_from_file_writes_one_file_per_frame_and_debug_config(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    run_dir = simulate_from_file(str(CONFIG_PATH.resolve()), str(tmp_path / 'out'))
    assert '+' not in Path(run_dir).name
    files = _observation_files(run_dir)
    assert len(files) == 2
    records = []
    for path in files:
        records.extend(json.loads(path.read_text(encoding='utf-8')))
    assert len(records) == 6
    times = [
        datetime.fromisoformat(record['obTime'].replace('Z', '+00:00'))
        for record in records
    ]
    assert times == sorted(times)
    assert {record['type'] for record in records} == {'PASSIVE_RF'}
    assert all(record['obTime'].endswith('000000Z') for record in records)
    assert all(record['snr'] > 0.0 for record in records)
    assert all('snrRawDb' in record for record in records)
    assert all('snrProcessedDb' in record for record in records)
    assert all('createdAt' not in record for record in records)
    assert all('dataMode' not in record for record in records)
    assert all('seed' not in record for record in records)
    assert all(record['createdBy'] == 'satsim' for record in records)
    assert list(Path(run_dir).glob('config_pass_*.json'))
    assert (Path(run_dir) / 'config.json').is_file()


def test_cli_dispatches_passive_rf(tmp_path):
    runner = CliRunner()
    result = runner.invoke(cli.main, [
        '--debug', 'INFO',
        'run',
        '--output_dir', str(tmp_path),
        str(CONFIG_PATH.resolve()),
    ])
    assert result.exit_code == 0, result.output
    run_dirs = [path for path in tmp_path.iterdir() if path.is_dir()]
    assert len(run_dirs) == 1
    assert _observation_files(run_dirs[0])


def test_cli_accepts_yaml_extension_for_analytical_modes(tmp_path):
    config_path = tmp_path / 'config.yaml'
    config_path.write_text(
        yaml.safe_dump(json.loads(CONFIG_PATH.read_text(encoding='utf-8'))),
        encoding='utf-8',
    )
    runner = CliRunner()
    result = runner.invoke(cli.main, [
        'run',
        '--output_dir', str(tmp_path / 'out'),
        str(config_path),
    ])
    assert result.exit_code == 0, result.output


@pytest.mark.parametrize('config', [
    {},
    {'fpa': {}, 'radar': {}},
    {'radar': {}, 'passive_rf': {}},
])
def test_cli_requires_exactly_one_simulation_mode(config):
    with pytest.raises(ClickException, match='exactly one'):
        cli._simulation_mode(config)


def test_invalid_target_fails_before_run_directory(tmp_path):
    config = json.loads(CONFIG_PATH.read_text(encoding='utf-8'))
    config['geometry']['obs']['list'][0].pop('tle')

    case = tmp_path / 'case'
    case.mkdir()
    (case / 'config.json').write_text(json.dumps(config), encoding='utf-8')
    output = tmp_path / 'out'

    with pytest.raises(ValueError, match='TLE target requires'):
        simulate_from_file(str(case / 'config.json'), str(output))
    assert not output.exists() or not list(output.iterdir())
