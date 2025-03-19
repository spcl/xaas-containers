import os
from pathlib import Path

import pytest

from xaas.config import BuildSystem
from xaas.config import FeatureType
from xaas.config import RunConfig


def test_file_not_found() -> None:
    with pytest.raises(FileNotFoundError):
        RunConfig.load("nonexistent_file.yaml")


def test_load_lulesh() -> None:

    cfg = RunConfig.load(os.path.join(Path(__file__).parent.parent, "configs", "lulesh.yml"))

    assert cfg.working_directory == "lulesh-builds"
    assert cfg.source_directory == "LULESH"
    assert cfg.project_name == "LULESH"
    assert cfg.build_system == BuildSystem.CMAKE
    assert cfg.features[FeatureType.MPI] == "WITH_MPI"
    assert cfg.features[FeatureType.OPENMP] == "WITH_OPENMP"
