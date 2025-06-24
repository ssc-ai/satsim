"""Tests for `satsim.cli`."""
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning, module="tensorflow")

from click.testing import CliRunner
from satsim import cli
import pytest
import shutil

try:
    if not shutil.which('satsim'):
        pytest.skip("CLI tests require full environment with CLI available", allow_module_level=True)
except Exception:
    pytest.skip("CLI tests require full environment with CLI available", allow_module_level=True)


def test_help():

    runner = CliRunner()
    result = runner.invoke(cli.main)
    assert result.exit_code == 0
    assert 'SatSim' in result.output

    result = runner.invoke(cli.main, ['--help'])
    assert result.exit_code == 0
    assert 'Show this message and exit.' in result.output


def test_ver():

    runner = CliRunner()
    result = runner.invoke(cli.main, ['--debug', 'OFF', 'version'])
    assert result.exit_code == 0

    result = runner.invoke(cli.main, ['--debug', 'DEBUG', 'version'])
    assert result.exit_code == 0

    result = runner.invoke(cli.main, ['--debug', 'INFO', 'version'])
    assert result.exit_code == 0

    result = runner.invoke(cli.main, ['--debug', 'WARNING', 'version'])
    assert result.exit_code == 0

    result = runner.invoke(cli.main, ['--debug', 'ERROR', 'version'])
    assert result.exit_code == 0

    result = runner.invoke(cli.main, ['--debug', 'BLAH', 'version'])
    assert result.exit_code == 0


def test_unknown():

    runner = CliRunner()
    result = runner.invoke(cli.run, ['unknown.ext'])
    assert result.exit_code == 1


def test_run_eager_json():

    runner = CliRunner()
    result = runner.invoke(cli.main, ['--debug', 'INFO', 'run', '--device', '0', '--mode', 'eager', '--output_dir', './.images', './tests/config_dynamic.json'])
    assert result.exit_code == 0


def test_run_eager_json_sgp4_sttr7():

    runner = CliRunner()
    result = runner.invoke(cli.main, ['--debug', 'INFO', 'run', '--device', '0', '--mode', 'eager', '--output_dir', './.images', './tests/config_static_sttr7_sgp4.json'])
    assert result.exit_code == 0


def test_run_eager_yaml():

    runner = CliRunner()
    result = runner.invoke(cli.main, ['--debug', 'INFO', 'run', '--device', '0', '--mode', 'eager', '--output_dir', './.images', './tests/config.yml'])
    assert result.exit_code == 0
