import os
from pathlib import Path

from dotenv import load_dotenv
from pydantic import BaseModel

ENV_PATH = Path(__file__).resolve().parents[1] / ".env"
load_dotenv(ENV_PATH)


def _int_env(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return int(raw)
    except ValueError:
        return default


class Settings(BaseModel):
    database_url: str = os.getenv("DATABASE_URL", "sqlite:///./sign_app.db")
    jwt_secret: str = os.getenv("JWT_SECRET", "change-me-in-production")
    jwt_algorithm: str = os.getenv("JWT_ALGORITHM", "HS256")
    # Default 24 hours; JWT `exp` is set from this value.
    access_token_expire_minutes: int = _int_env("ACCESS_TOKEN_EXPIRE_MINUTES", 24 * 60)


settings = Settings()
