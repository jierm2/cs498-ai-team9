import os
import sys
from pathlib import Path

from dotenv import load_dotenv

_root = Path(__file__).resolve().parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

load_dotenv()

# Service account JSON (set explicitly, or default to repo-root key file if present).
_default_key = _root / "noble-operation-492621-c6-8f8ec6acb12e.json"
if not os.environ.get("GOOGLE_APPLICATION_CREDENTIALS"):
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = str(_default_key)

os.environ.setdefault("GOOGLE_GENAI_USE_VERTEXAI", "true")
os.environ.setdefault("GOOGLE_CLOUD_PROJECT", "noble-operation-492621-c6")
os.environ.setdefault("GOOGLE_CLOUD_LOCATION", "global")

from src.utils.gemini_client import get_client

client = get_client()

response = client.models.generate_content(
    model="gemini-3-flash-preview",
    contents="Hello!",
)
print(response.text)
