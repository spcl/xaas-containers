import pytest
import yaml
from typing import Dict, Tuple

from xaas.config import Config


@pytest.fixture
def resources() -> Tuple[Config, Dict]:
    with open(Config.DEFAULT_CONFIGURATION, "r") as f:
        return Config(), yaml.safe_load(f)


def test_file_not_found(resources: Tuple[Config, Dict]):
    with pytest.raises(FileNotFoundError):
        Config().initialize("nonexistent_file.yaml")


def test_singleton_pattern(resources: Tuple[Config, Dict]):
    config = Config()

    assert resources[0] == config


def test_load_config(resources: Tuple[Config, Dict]):
    config = Config()
    config.initialize(Config.DEFAULT_CONFIGURATION)

    # check also singleton property
    assert resources[0].docker_repository == resources[1]["docker_repository"]
    assert resources[0].ir_type == resources[1]["ir_type"]
    assert resources[0].parallelism_level == resources[1]["parallelism_level"]


def test_initialization_once(resources: Tuple[Config, Dict]):
    with pytest.raises(RuntimeError):
        resources[0].initialize("config.yaml")
