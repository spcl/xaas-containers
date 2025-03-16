import pytest
import yaml
from typing import Dict, Tuple

from xaas.config import XaaSConfig


@pytest.fixture
def resources() -> Tuple[XaaSConfig, Dict]:
    with open(XaaSConfig.DEFAULT_CONFIGURATION, "r") as f:
        return XaaSConfig(), yaml.safe_load(f)


def test_file_not_found(resources: Tuple[XaaSConfig, Dict]):
    with pytest.raises(FileNotFoundError):
        XaaSConfig().initialize("nonexistent_file.yaml")


def test_singleton_pattern(resources: Tuple[XaaSConfig, Dict]):
    config = XaaSConfig()

    assert resources[0] == config


def test_load_config(resources: Tuple[XaaSConfig, Dict]):
    config = XaaSConfig()
    config.initialize(XaaSConfig.DEFAULT_CONFIGURATION)

    # check also singleton property
    assert resources[0].docker_repository == resources[1]["docker_repository"]
    assert resources[0].ir_type.value == resources[1]["ir_type"]
    assert resources[0].parallelism_level == resources[1]["parallelism_level"]


def test_initialization_once(resources: Tuple[XaaSConfig, Dict]):
    with pytest.raises(RuntimeError):
        resources[0].initialize("config.yaml")
