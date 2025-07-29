import logging
from typing import Literal

import torch
from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict
from rich.logging import RichHandler

CUDA_AVAILABLE = torch.cuda.is_available()


class Settings(BaseSettings):
    """Settings used to configure the CLI and application behavior.

    The default values of these settings are designed to work in a Docker container
    environment, following the interface of the DEEP-PSMA Grand Challenge.
    To override these settings, use a `.env` file to override them or set environment variables.
    """

    model_config = SettingsConfigDict(case_sensitive=True, env_file=".env", extra="allow")

    # Inference configuration
    INPUT_FORMAT: Literal["gc", "csv"] = "gc"
    INPUT_CSV: str = ""
    INPUT_DIR: str = "/input"
    OUTPUT_DIR: str = "/output"
    WEIGHTS_DIR: str = "/opt/ml/model"
    DEVICE: str = "auto"
    MIXED_PRECISION: bool = False
    POSTPROCESS_FDG_BASED_ON_PSMA_CLASSES: bool = False
    USE_TTA: bool = False

    # Logging configuration
    LOG_FORMAT: str = "%(message)s"
    LOG_LEVEL: str = "INFO"

    @field_validator("DEVICE")
    def validate_device(cls, value: str) -> str:
        value = value.lower()
        if value == "auto":
            return "cuda" if CUDA_AVAILABLE else "cpu"
        return "cpu" if not CUDA_AVAILABLE else value


settings = Settings()

logging.basicConfig(
    level=settings.LOG_LEVEL,
    format=settings.LOG_FORMAT,
    datefmt="[%X]",
    handlers=[RichHandler()],
)
