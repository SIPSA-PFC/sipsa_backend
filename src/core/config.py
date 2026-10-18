"""
core/config.py
--------------
Centralised settings loaded from environment variables.
Copy .env.example to .env and fill in your values.
"""

from pathlib import Path

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT = Path(__file__).resolve().parent.parent.parent

# Example values that must never be used as the real signing key.
_PLACEHOLDER_SECRETS = {
    "change-me-in-production",
    "change-this-to-a-random-string-in-production",
}


class Settings(BaseSettings):
    # ── Database ──────────────────────────────────────────────────────────────
    database_url: str = "postgresql+asyncpg://user:password@localhost:5432/dengue"

    # ── Model weights ─────────────────────────────────────────────────────────
    model_weights_path: str = str(ROOT / "data" / "models" / "eval_results.npz")

    # ── Algorithm parameters (must match training in ml/train.py) ─────────────
    n_back_samples:    list[int] = [1, 3, 4, 5, 6]
    train_split:       float     = 0.8
    min_history_weeks: int       = 6       # ValidationService: minimum weeks required

    # ── Auth (RN07) ───────────────────────────────────────────────────────────
    # Required, no default: tokens signed with a known key can be forged.
    secret_key:                    str
    algorithm:                     str = "HS256"
    access_token_expire_minutes:   int = 60

    # ── CORS: origins allowed to call the API from a browser ──────────────────
    # In .env as a JSON list, e.g.  CORS_ORIGINS=["https://sipsa.example.org"]
    cors_origins: list[str] = ["http://localhost:3000"]

    # .env is read from the repo root, so scripts and Alembic find it from
    # any working directory.
    model_config = SettingsConfigDict(env_file=ROOT / ".env", env_file_encoding="utf-8")

    @field_validator("secret_key")
    @classmethod
    def _secret_key_is_strong(cls, value: str) -> str:
        if value in _PLACEHOLDER_SECRETS or len(value) < 32:
            raise ValueError(
                "SECRET_KEY must be a random string of at least 32 characters. "
                'Generate one with:  python -c "import secrets; print(secrets.token_urlsafe(48))"'
            )
        return value


settings = Settings()