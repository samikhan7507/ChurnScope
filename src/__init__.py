"""ChurnScope source package: configuration management and structured logging."""

from __future__ import annotations

import copy
import logging
import logging.config
import logging.handlers
import os
from pathlib import Path
from typing import Any

import yaml

try:
    from dotenv import load_dotenv

    load_dotenv(override=False)
except ImportError:
    pass

_LOGGING_INITIALIZED = False


def _parse_bool(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _apply_env_overrides(cfg: dict[str, Any]) -> dict[str, Any]:
    """Overlay runtime environment variables (local .env or Render) onto YAML config."""
    api_cfg = cfg.setdefault("api", {})
    log_cfg = cfg.setdefault("logging", {})
    model_cfg = cfg.setdefault("model", {})
    threshold_cfg = cfg.setdefault("threshold", {})

    if log_level := os.getenv("LOG_LEVEL"):
        log_cfg["level"] = log_level.strip().upper()

    if model_version := os.getenv("MODEL_VERSION"):
        model_cfg["version"] = model_version.strip()
        api_cfg["model_version"] = model_version.strip()

    if model_algorithm := os.getenv("MODEL_ALGORITHM"):
        model_cfg["algorithm"] = model_algorithm.strip().lower()

    if default_threshold := os.getenv("DEFAULT_THRESHOLD"):
        try:
            threshold_cfg["default"] = float(default_threshold)
        except ValueError:
            pass

    if threshold_strategy := os.getenv("THRESHOLD_STRATEGY"):
        threshold_cfg["strategy"] = threshold_strategy.strip().lower()

    if api_host := os.getenv("API_HOST"):
        api_cfg["host"] = api_host.strip()

    port_env = os.getenv("PORT") or os.getenv("API_PORT")
    if port_env:
        try:
            api_cfg["port"] = int(port_env)
        except ValueError:
            pass

    if api_workers := os.getenv("API_WORKERS"):
        try:
            api_cfg["workers"] = int(api_workers)
        except ValueError:
            pass

    if (api_reload := os.getenv("API_RELOAD")) is not None and api_reload != "":
        api_cfg["reload"] = _parse_bool(api_reload)

    if max_batch := os.getenv("MAX_BATCH_SIZE"):
        try:
            api_cfg["max_batch_size"] = int(max_batch)
        except ValueError:
            pass

    if allowed_origins := os.getenv("ALLOWED_ORIGINS"):
        origins = [o.strip() for o in allowed_origins.split(",") if o.strip()]
        if origins:
            api_cfg["cors_origins"] = origins

    if rate_limit := os.getenv("RATE_LIMIT_PER_MINUTE"):
        try:
            api_cfg["rate_limit_per_minute"] = int(rate_limit)
        except ValueError:
            pass

    if (enable_metrics := os.getenv("ENABLE_METRICS")) is not None and enable_metrics != "":
        api_cfg["enable_metrics"] = _parse_bool(enable_metrics)

    if (enable_docs := os.getenv("ENABLE_DOCS")) is not None and enable_docs != "":
        if not _parse_bool(enable_docs):
            api_cfg["docs_url"] = None
            api_cfg["redoc_url"] = None

    if (auto_train := os.getenv("AUTO_TRAIN_ON_STARTUP")) is not None and auto_train != "":
        api_cfg["auto_train_on_startup"] = _parse_bool(auto_train)

    if auto_samples := os.getenv("AUTO_TRAIN_SAMPLES"):
        try:
            api_cfg["auto_train_samples"] = int(auto_samples)
        except ValueError:
            pass

    return cfg


def load_config(config_path: str | None = None, apply_env: bool = True) -> dict[str, Any]:
    """Load YAML configuration and optionally apply environment variable overrides."""
    resolved = config_path or os.getenv("CONFIG_PATH", "config/config.yaml")
    path = Path(resolved)
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {resolved}")
    with path.open(encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}
    cfg = copy.deepcopy(raw)
    return _apply_env_overrides(cfg) if apply_env else cfg


def setup_logging(logging_config_path: str | None = None, level: str | None = None) -> None:
    """Configure application logging from logging.yaml with safe read-only fallback."""
    global _LOGGING_INITIALIZED
    effective_level = (level or os.getenv("LOG_LEVEL", "INFO")).upper()
    resolved_path = Path(
        logging_config_path or os.getenv("LOGGING_CONFIG_PATH", "config/logging.yaml")
    )

    if resolved_path.exists():
        try:
            with resolved_path.open(encoding="utf-8") as fh:
                log_cfg = yaml.safe_load(fh)
            for handler in log_cfg.get("handlers", {}).values():
                filename = handler.get("filename")
                if filename:
                    Path(filename).parent.mkdir(parents=True, exist_ok=True)
            log_cfg.setdefault("root", {})["level"] = effective_level
            if "console" in log_cfg.get("handlers", {}):
                log_cfg["handlers"]["console"]["level"] = effective_level
            logging.config.dictConfig(log_cfg)
            _LOGGING_INITIALIZED = True
            return
        except OSError:
            # Read-only container filesystem or non-writable volume mount: fall back to stdout.
            pass
        except Exception:
            pass

    logging.basicConfig(
        level=getattr(logging, effective_level, logging.INFO),
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    _LOGGING_INITIALIZED = True


def get_logger(name: str, config: dict[str, Any] | None = None) -> logging.Logger:
    """Return a configured module logger, initializing logging once on first use."""
    global _LOGGING_INITIALIZED
    if not _LOGGING_INITIALIZED:
        cfg_level = None
        log_cfg_path = None
        if config:
            cfg_level = config.get("logging", {}).get("level")
            log_cfg_path = config.get("paths", {}).get("logging_config")
        setup_logging(logging_config_path=log_cfg_path, level=cfg_level)

    logger = logging.getLogger(name)
    if config and "logging" in config:
        lvl_name = str(config["logging"].get("level", "INFO")).upper()
        logger.setLevel(getattr(logging, lvl_name, logging.INFO))
    return logger
