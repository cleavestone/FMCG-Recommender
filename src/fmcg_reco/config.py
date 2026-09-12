"""Project-wide paths and constants shared across notebooks, library code, and tests."""
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"

# Time-based split boundaries (week_no 1-102), fixed by notebook 02's analysis.
TRAIN_END_WEEK = 88
VAL_END_WEEK = 95
TEST_END_WEEK = 102

# Candidate universe size (top-N most-purchased products in train), fixed by
# notebook 02: this caps the reachable recall ceiling at ~58.5% on validation.
N_CANDIDATES = 5000


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    jwt_secret: str = "dev-secret-change-me-in-production-32bytes+"
    jwt_algorithm: str = "HS256"
    access_token_expire_minutes: int = 30
    refresh_token_expire_minutes: int = 60 * 24 * 7
    admin_api_key: str = "dev-admin-key-change-me"

    database_url: str = f"sqlite:///{ROOT / 'auth.db'}"
    hybrid_artifact_dir: Path = ROOT / "artifacts" / "latest" / "hybrid"
    affinity_artifact_dir: Path = ROOT / "artifacts" / "latest" / "affinity"


settings = Settings()
