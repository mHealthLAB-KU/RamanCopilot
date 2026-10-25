"""Agent package. Loads the .env file from the project root."""


from pathlib import Path

try:
    from dotenv import load_dotenv

    _PROJECT_ROOT = Path(__file__).resolve().parents[3]
    _ENV_PATH = _PROJECT_ROOT / ".env"

    load_dotenv(dotenv_path=_ENV_PATH, override=False)
except Exception:
    pass
