from __future__ import annotations

import importlib
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent

ROOT_MODULES = [
    "run_store",
]

SRC_MODULES = [
    "adr",
    "adr_refiner",
    "lda_go",
    "edges",
    "embedding",
    "fa2_layout",
    "k_estimator_gmm",
    "k_estimator_hdbscan",
    "naming",
    "nova",
    "adept",
    "render",
]


@pytest.mark.parametrize("name", ROOT_MODULES + SRC_MODULES)
def test_module_imports(name: str) -> None:
    importlib.import_module(name)


def test_utils_logger_is_importable_and_has_notice() -> None:
    from utils.logger import get_logger

    logger = get_logger("tests")
    assert hasattr(logger, "notice"), "NOTICE level missing — see utils/logger.py"


def test_shipped_config_parses() -> None:
    config_dir = REPO_ROOT / "config"
    cfg = yaml.safe_load((config_dir / "config.yaml").read_text(encoding="utf-8"))
    advanced = yaml.safe_load((config_dir / "advanced.yaml").read_text(encoding="utf-8"))

    assert cfg["run_mode"] in {"full", "clustering_only"}
    assert cfg["content_type"] in {"social_media", "document"}
    for slot in ("heavy", "light"):
        assert slot in cfg["models"], f"models.{slot} missing from config.yaml"
        assert not cfg["models"][slot].get("api_key"), (
            f"models.{slot}.api_key must stay blank — keys belong in .env"
        )

    assert advanced["clustering"]["k_estimation_method"] in {"gmm_bic", "hdbscan"}


def test_env_example_exists_and_env_is_ignored() -> None:
    assert (REPO_ROOT / ".env.example").is_file(), ".env.example is referenced by the README"

    gitignore = (REPO_ROOT / ".gitignore").read_text(encoding="utf-8")
    ignored = {line.strip().lstrip("/") for line in gitignore.splitlines()}
    assert ".env" in ignored, ".env must be listed in .gitignore"
