from __future__ import annotations

from typing import Optional
import os
from pathlib import Path
import yaml


class Config:
    _instance: Optional[Config] = None

    DEFAULT_CONFIGURATION = os.path.join(Path(__file__).parent, "config", "system.yaml")

    @property
    def initialized(self) -> bool:
        return self._initialized

    def __new__(cls) -> Config:
        if cls._instance is None:
            cls._instance = super(Config, cls).__new__(cls)
            cls._instance._initialized = False

        return cls._instance

    def __init__(self):
        self._initialized: bool
        self.docker_repository: str
        self.ir_type: str
        self.parallelism_level: int

    def initialize(self, config_path: str):
        if self._initialized:
            raise RuntimeError("Configuration already initialized")

        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Configuration file not found: {config_path}")

        with open(config_path, "r") as f:
            config_data = yaml.safe_load(f)

        self.docker_repository = config_data["docker_repository"]
        self.ir_type = config_data["ir_type"]
        self.parallelism_level = config_data["parallelism_level"]

        self._initialized = True
