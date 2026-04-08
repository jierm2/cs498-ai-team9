"""Shared google-genai client (Gemini API or Vertex AI).

Vertex AI (recommended for GCP): set in .env or the environment
  GOOGLE_GENAI_USE_VERTEXAI=true
  GOOGLE_CLOUD_PROJECT=<project-id>   # optional if using a service-account JSON (project_id in file wins)
  GOOGLE_CLOUD_LOCATION=global   # optional; default is global

  GOOGLE_APPLICATION_CREDENTIALS=<path-to-service-account.json>

If your shell still exports GOOGLE_CLOUD_PROJECT from another gcloud project, python-dotenv will not
override it by default — we therefore use the JSON key's project_id for Vertex when a key file is used.

Gemini Developer API: unset GOOGLE_GENAI_USE_VERTEXAI (or false) and set GEMINI_API_KEY
(see google-genai / AI Studio docs).
"""

from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path

from dotenv import load_dotenv
from google import genai

_REPO_ROOT = Path(__file__).resolve().parents[2]


def _resolve_credentials_env() -> None:
    """If GOOGLE_APPLICATION_CREDENTIALS is relative, resolve it from repo root."""
    raw = os.getenv("GOOGLE_APPLICATION_CREDENTIALS", "").strip()
    if not raw:
        return
    p = Path(raw)
    if p.is_absolute():
        return
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = str((_REPO_ROOT / p).resolve())


load_dotenv()
_resolve_credentials_env()

logger = logging.getLogger(__name__)

_client: RetryClient | None = None

MAX_RETRIES = 5
BASE_DELAY = 2  # seconds


def _use_vertex_ai() -> bool:
    return os.getenv("GOOGLE_GENAI_USE_VERTEXAI", "").lower() in ("1", "true", "yes")


def _vertex_service_account_credentials():
    """Load JSON key explicitly so Application Default Credentials (e.g. gcloud user login) cannot override."""
    path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS", "").strip()
    if not path:
        return None
    fp = Path(path)
    if not fp.is_file():
        logger.warning("GOOGLE_APPLICATION_CREDENTIALS is set but file not found: %s", fp)
        return None
    from google.oauth2 import service_account

    return service_account.Credentials.from_service_account_file(
        str(fp),
        scopes=["https://www.googleapis.com/auth/cloud-platform"],
    )


def _service_account_project_id(creds_path: Path) -> str | None:
    try:
        with open(creds_path, encoding="utf-8") as f:
            return json.load(f).get("project_id")
    except (OSError, json.JSONDecodeError, TypeError):
        return None


def _create_base_client() -> genai.Client:
    if _use_vertex_ai():
        env_project = os.getenv("GOOGLE_CLOUD_PROJECT", "").strip()
        location = (os.getenv("GOOGLE_CLOUD_LOCATION") or "global").strip() or "global"
        creds = _vertex_service_account_credentials()
        creds_path = Path(os.getenv("GOOGLE_APPLICATION_CREDENTIALS", "").strip())
        sa_project = _service_account_project_id(creds_path) if creds_path.is_file() else None

        # Service-account JSON is authoritative: URL project must match the key's project_id.
        # Stale shell exports (e.g. another gcloud default) often cause 403 on the wrong project.
        if creds is not None and sa_project:
            project = sa_project
            if env_project and env_project != sa_project:
                logger.warning(
                    "Ignoring GOOGLE_CLOUD_PROJECT=%r; using service account project_id=%r "
                    "(unset wrong GOOGLE_CLOUD_PROJECT in your shell if this surprises you).",
                    env_project,
                    sa_project,
                )
        else:
            project = env_project
            if not project:
                raise ValueError(
                    "Vertex AI is enabled (GOOGLE_GENAI_USE_VERTEXAI=true) but GOOGLE_CLOUD_PROJECT is empty "
                    "and no usable GOOGLE_APPLICATION_CREDENTIALS JSON with project_id was found."
                )

        os.environ["GOOGLE_CLOUD_PROJECT"] = project

        kwargs: dict = {"vertexai": True, "project": project, "location": location}
        if creds is not None:
            kwargs["credentials"] = creds
            logger.info(
                "Vertex AI using explicit service account file (not ADC): %s",
                creds_path.name,
            )
        else:
            logger.info(
                "Vertex AI without GOOGLE_APPLICATION_CREDENTIALS file — using Application Default Credentials"
            )

        logger.info("Vertex AI backend project=%s location=%s", project, location)
        return genai.Client(**kwargs)
    return genai.Client()


class RetryClient:
    """Wraps genai.Client to add automatic retry with exponential backoff on 429/503."""

    def __init__(self, client: genai.Client):
        self._client = client
        self.models = RetryModels(client.models)


class RetryModels:
    def __init__(self, models):
        self._models = models

    def generate_content(self, **kwargs):
        for attempt in range(MAX_RETRIES + 1):
            try:
                return self._models.generate_content(**kwargs)
            except Exception as e:
                error_str = str(e).lower()
                retryable = any(
                    code in error_str
                    for code in ["429", "503", "resource exhausted", "unavailable", "rate limit"]
                )

                if not retryable or attempt == MAX_RETRIES:
                    raise

                delay = BASE_DELAY * (2**attempt)
                logger.warning(
                    "API error (attempt %s/%s): %s. Retrying in %ss...",
                    attempt + 1,
                    MAX_RETRIES + 1,
                    e,
                    delay,
                )
                time.sleep(delay)


def get_client() -> RetryClient:
    global _client
    if _client is None:
        _client = RetryClient(_create_base_client())
    return _client
