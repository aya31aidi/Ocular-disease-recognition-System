import logging
import os
import secrets
import tempfile
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

# Vercel and AWS Lambda mount the code read-only; only the temp folder is writable,
# and it is wiped whenever the function instance is recycled.
IS_SERVERLESS = bool(os.environ.get("VERCEL") or os.environ.get("AWS_LAMBDA_FUNCTION_NAME"))

log = logging.getLogger(__name__)


def _load_secret_key(instance_dir: Path) -> str:
    """Prefer SECRET_KEY from the environment, otherwise persist a random one
    in the instance folder so sessions survive restarts during development."""
    env_key = os.environ.get("SECRET_KEY")
    if env_key:
        return env_key
    if IS_SERVERLESS:
        log.warning("SECRET_KEY is not set; sessions will reset whenever the function restarts")
    key_file = instance_dir / ".secret_key"
    if key_file.exists():
        return key_file.read_text().strip()
    key = secrets.token_hex(32)
    key_file.write_text(key)
    return key


def _default_model_path() -> Path:
    # The original project kept the weights next to app.py, so both spots are checked.
    for candidate in (BASE_DIR / "models" / "ocular_disease_model.h5",
                      BASE_DIR / "ocular_disease_model.h5"):
        if candidate.exists():
            return candidate
    return BASE_DIR / "models" / "ocular_disease_model.h5"


def _database_url(instance_dir: Path) -> str:
    # Vercel's Postgres integrations (Neon, Supabase) expose POSTGRES_URL or DATABASE_URL.
    url = os.environ.get("DATABASE_URL") or os.environ.get("POSTGRES_URL")
    if not url:
        return f"sqlite:///{instance_dir / 'ocular.db'}"
    for prefix in ("postgres://", "postgresql://"):
        if url.startswith(prefix):
            return "postgresql+psycopg://" + url[len(prefix):]
    return url


class Config:
    INSTANCE_DIR = Path(os.environ.get(
        "ODR_INSTANCE_DIR",
        Path(tempfile.gettempdir()) / "ocular" if IS_SERVERLESS else BASE_DIR / "instance",
    ))

    SQLALCHEMY_DATABASE_URI = _database_url(INSTANCE_DIR)
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    # Serverless functions freeze between requests, so pooled connections can go stale.
    SQLALCHEMY_ENGINE_OPTIONS = {"pool_pre_ping": True, "pool_recycle": 280}

    # Only used to read images saved by older versions; new analyses live in the database.
    UPLOAD_DIR = INSTANCE_DIR / "uploads"
    MAX_CONTENT_LENGTH = 10 * 1024 * 1024  # 10 MB per upload
    ALLOWED_IMAGE_FORMATS = {"JPEG", "PNG", "BMP", "TIFF", "WEBP"}

    MODEL_PATH = Path(os.environ.get("ODR_MODEL_PATH", _default_model_path()))
    # "auto" falls back to demo predictions when the weights are missing,
    # "on" forces demo mode, "off" refuses to start without a model.
    DEMO_MODE = os.environ.get("ODR_DEMO_MODE", "auto").lower()
    SERVERLESS = IS_SERVERLESS

    WTF_CSRF_TIME_LIMIT = None
    SESSION_COOKIE_HTTPONLY = True
    SESSION_COOKIE_SAMESITE = "Lax"
    SESSION_COOKIE_SECURE = IS_SERVERLESS
    REMEMBER_COOKIE_SECURE = IS_SERVERLESS

    @classmethod
    def init_instance(cls, app_config: dict) -> None:
        instance_dir = Path(app_config["INSTANCE_DIR"])
        instance_dir.mkdir(parents=True, exist_ok=True)
        Path(app_config["UPLOAD_DIR"]).mkdir(parents=True, exist_ok=True)
        if not app_config.get("SECRET_KEY"):
            app_config["SECRET_KEY"] = _load_secret_key(instance_dir)


class TestConfig(Config):
    TESTING = True
    WTF_CSRF_ENABLED = False
    DEMO_MODE = "on"
    SESSION_COOKIE_SECURE = False
    REMEMBER_COOKIE_SECURE = False
