"""Configuration management for the Equix application."""

from __future__ import annotations

from pathlib import Path
from typing import Any
from typing import Literal

import dotenv
from pydantic import Field
from pydantic import field_validator
from pydantic_settings import BaseSettings
from pydantic_settings import SettingsConfigDict

# Load environment variables
dotenv.load_dotenv(override=True)


class ModelConfig(BaseSettings):
    """Configuration for ML models."""

    model_config = SettingsConfigDict(env_prefix="MODEL_")

    # Model identifiers
    baseline_model: str = Field(
        default="meta-llama/Llama-3.2-11B-Vision",
        description="Baseline vision-language model identifier",
    )
    layout_detection_model: str = Field(
        default="cmarkea/detr-layout-detection",
        description="Layout detection model identifier",
    )
    # deepseek-ai/deepseek-vl2-tiny will also be tested
    picture_extraction_model: str = Field(
        default="mPLUG/mPLUG-Owl3-2B-241014",
        description="Picture information extraction model identifier",
    )
    # Qwen/Qwen2.5-1.5B / deepseek-ai/deepseek-llm-7b-chat /
    # deepseek-ai/DeepSeek-V2-Lite-Chat will also be tested
    llm_model: str = Field(
        default="meta-llama/Llama-3.2-1B",
        description="Language model identifier",
    )
    # deepset/sentence_bert is also suitable for this task
    embedding_model: str = Field(
        default="sentence-transformers/all-MiniLM-L6-v2",
        description="Text embedding model identifier",
    )

    # Model settings
    device: str = Field(
        default="auto",
        description="Device to run models on (auto, cpu, cuda, cuda:0, etc.)",
    )
    detection_threshold: float = Field(
        default=0.3,
        ge=0.0,
        le=1.0,
        description="Detection confidence threshold",
    )
    max_tokens: int = Field(
        default=200,
        ge=1,
        description="Maximum tokens for text generation",
    )
    batch_size: int = Field(
        default=1,
        ge=1,
        description="Batch size for model inference",
    )

    @field_validator("device")
    @classmethod
    def validate_device(cls, v: str) -> str:
        """Validate device configuration."""
        if v == "auto":
            return v

        valid_devices = ["cpu"]
        if v.startswith("cuda"):
            valid_devices.extend(["cuda", "cuda:0", "cuda:1", "cuda:2", "cuda:3"])

        if v not in valid_devices and not v.startswith("cuda:"):
            raise ValueError(f"Invalid device: {v}")

        return v


class DataConfig(BaseSettings):
    """Configuration for datasets and data processing."""

    model_config = SettingsConfigDict(env_prefix="DATA_")

    # Dataset identifiers
    chart_qa_dataset: str = Field(
        default="ahmed-masry/ChartQA",
        description="Chart QA dataset identifier",
    )
    table_extraction_dataset: str = Field(
        default="bsmock/pubtables-1m",
        description="Table extraction dataset identifier",
    )
    document_qa_generator: str = Field(
        default="ds4sd/DocLayNet",
        description="Document QA generator dataset identifier",
    )
    papers_dataset: str = Field(
        default="Cornell-University/arxiv",
        description="Papers dataset identifier",
    )

    # Processing settings
    max_file_size_mb: int = Field(
        default=50,
        ge=1,
        description="Maximum file size in MB",
    )
    supported_formats: list[str] = Field(
        default=[".pdf"],
        description="Supported file formats",
    )
    pdf_dpi: int = Field(
        default=200,
        ge=72,
        le=600,
        description="DPI for PDF to image conversion",
    )


class ServiceConfig(BaseSettings):
    """Configuration for services."""

    model_config = SettingsConfigDict(env_prefix="SERVICE_")

    # Service settings
    host: str = Field(default="0.0.0.0", description="Service host")
    port: int = Field(default=5123, ge=1, le=65535, description="Service port")
    workers: int = Field(default=1, ge=1, description="Number of workers")

    # CORS settings
    cors_origins: list[str] = Field(
        default=["http://localhost:1420", "*"],
        description="CORS allowed origins",
    )
    cors_methods: list[str] = Field(
        default=["GET", "POST", "PUT", "DELETE"],
        description="CORS allowed methods",
    )
    cors_headers: list[str] = Field(
        default=["*"],
        description="CORS allowed headers",
    )

    # Request/Response settings
    max_request_size: int = Field(
        default=50 * 1024 * 1024,  # 50MB
        ge=1024,
        description="Maximum request size in bytes",
    )
    request_timeout: int = Field(
        default=300,
        ge=1,
        description="Request timeout in seconds",
    )


class CacheConfig(BaseSettings):
    """Configuration for caching."""

    model_config = SettingsConfigDict(env_prefix="CACHE_")

    # Cache settings
    enabled: bool = Field(default=True, description="Enable caching")
    ttl: int = Field(default=3600, ge=0, description="Default TTL in seconds")
    max_size: int = Field(default=1000, ge=1, description="Maximum cache size")

    # Redis settings (optional)
    redis_url: str | None = Field(default=None, description="Redis URL")
    redis_password: str | None = Field(default=None, description="Redis password")
    redis_db: int = Field(default=0, ge=0, description="Redis database number")


class LoggingConfig(BaseSettings):
    """Configuration for logging."""

    model_config = SettingsConfigDict(env_prefix="LOG_")

    # Logging settings
    level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = Field(
        default="INFO",
        description="Logging level",
    )
    format: str = Field(
        default="{time:YYYY-MM-DD HH:mm:ss} | {level} | {name}:{function}:{line} | {message}",
        description="Log format",
    )
    rotation: str = Field(
        default="1 day",
        description="Log rotation schedule",
    )
    retention: str = Field(
        default="30 days",
        description="Log retention period",
    )

    # File settings
    log_dir: Path = Field(
        default=Path("logs"),
        description="Log directory",
    )
    log_file: str = Field(
        default="equix.log",
        description="Log file name",
    )


class SecurityConfig(BaseSettings):
    """Configuration for security."""

    model_config = SettingsConfigDict(env_prefix="SECURITY_")

    # API Security
    api_key_required: bool = Field(default=False, description="Require API key")
    api_key: str | None = Field(default=None, description="API key")

    # Rate limiting
    rate_limit_enabled: bool = Field(default=True, description="Enable rate limiting")
    rate_limit_requests: int = Field(
        default=100,
        ge=1,
        description="Requests per minute",
    )

    # File upload security
    scan_uploads: bool = Field(default=True, description="Scan uploaded files")
    allowed_extensions: list[str] = Field(
        default=[".pdf", ".png", ".jpg", ".jpeg"],
        description="Allowed file extensions",
    )


class Config(BaseSettings):
    """Main configuration class."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # Environment settings
    environment: Literal["development", "staging", "production"] = Field(
        default="development",
        description="Application environment",
    )
    debug: bool = Field(default=False, description="Enable debug mode")

    # Directory settings
    project_dir: Path = Field(
        default=Path(__file__).resolve().parents[1],
        description="Project root directory",
    )
    data_dir: Path = Field(
        default=Path(__file__).resolve().parents[1] / "data",
        description="Data directory",
    )
    model_dir: Path = Field(
        default=Path(__file__).resolve().parents[1] / "models",
        description="Models directory",
    )
    cache_dir: Path = Field(
        default=Path(__file__).resolve().parents[1] / "cache",
        description="Cache directory",
    )

    # External API settings
    huggingface_token: str | None = Field(
        default=None,
        description="Hugging Face API token",
    )

    # Random seed for reproducibility
    random_seed: int = Field(default=42, ge=0, description="Random seed")

    # Sub-configurations
    models: ModelConfig = Field(default_factory=ModelConfig)
    data: DataConfig = Field(default_factory=DataConfig)
    service: ServiceConfig = Field(default_factory=ServiceConfig)
    cache: CacheConfig = Field(default_factory=CacheConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    security: SecurityConfig = Field(default_factory=SecurityConfig)

    def __init__(self, **kwargs: Any) -> None:
        """Initialize configuration with validation."""
        super().__init__(**kwargs)
        self._validate_directories()
        self._validate_environment()

    def _validate_directories(self) -> None:
        """Validate that required directories exist or can be created."""
        directories = [
            self.data_dir,
            self.model_dir,
            self.cache_dir,
            self.logging.log_dir,
        ]

        for directory in directories:
            try:
                directory.mkdir(parents=True, exist_ok=True)
            except Exception as e:
                from src.core.exceptions import ConfigurationError

                raise ConfigurationError(
                    f"Cannot create directory {directory}: {e}",
                    "DIRECTORY_CREATION_ERROR",
                ) from e

    def _validate_environment(self) -> None:
        """Validate environment-specific settings."""
        if self.environment == "production":
            if self.debug:
                from src.core.exceptions import ConfigurationError

                raise ConfigurationError(
                    "Debug mode should not be enabled in production",
                    "INVALID_PRODUCTION_CONFIG",
                )

            if not self.security.api_key_required:
                # Warning: consider requiring API key in production
                pass

    @property
    def is_production(self) -> bool:
        """Check if running in production environment."""
        return self.environment == "production"

    @property
    def is_development(self) -> bool:
        """Check if running in development environment."""
        return self.environment == "development"


# Global configuration instance
try:
    config = Config()
except Exception as e:
    raise ConfigurationError(
        f"Failed to load configuration: {e}",
        "CONFIG_LOAD_ERROR",
    ) from e
