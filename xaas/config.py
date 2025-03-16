from __future__ import annotations

from typing import Optional
from enum import Enum
import os
from pathlib import Path
import yaml


class IRType(Enum):
    LLVM_IR = "llvm-ir"


class XaaSConfig:
    _instance: Optional[XaaSConfig] = None

    DEFAULT_CONFIGURATION = os.path.join(Path(__file__).parent, "config", "system.yaml")

    @property
    def initialized(self) -> bool:
        return self._initialized

    def __new__(cls) -> XaaSConfig:
        if cls._instance is None:
            cls._instance = super(XaaSConfig, cls).__new__(cls)
            cls._instance._initialized = False

        return cls._instance

    def __init__(self):
        self._initialized: bool
        self.docker_repository: str
        self.ir_type: IRType
        self.parallelism_level: int

    def initialize(self, config_path: str):
        if self._initialized:
            raise RuntimeError("Configuration already initialized")

        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Configuration file not found: {config_path}")

        with open(config_path, "r") as f:
            config_data = yaml.safe_load(f)

        self.docker_repository = config_data["docker_repository"]
        self.parallelism_level = config_data["parallelism_level"]

        match config_data["ir_type"]:
            case IRType.LLVM_IR.value:
                self.ir_type = IRType.LLVM_IR
            case _:
                raise ValueError(f"Unsupported IR type: {config_data['ir_type']}")

        self._initialized = True
