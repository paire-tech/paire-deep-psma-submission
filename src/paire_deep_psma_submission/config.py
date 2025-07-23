import logging
from typing import Literal

import torch
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
    INPUT_DIR: str = "/input"
    OUTPUT_DIR: str = "/output"

    # Logging configuration
    LOG_FORMAT: str = "%(message)s"
    LOG_LEVEL: str = "INFO"


settings = Settings()

logging.basicConfig(
    level=settings.LOG_LEVEL,
    format=settings.LOG_FORMAT,
    datefmt="[%X]",
    handlers=[RichHandler()],
)
