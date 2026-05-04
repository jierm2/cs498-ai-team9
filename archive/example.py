import sys
from pathlib import Path

from dotenv import load_dotenv

_root = Path(__file__).resolve().parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

load_dotenv()

from src.utils.gemini_client import get_client

client = get_client()

response = client.models.generate_content(
    model="gemini-3.1-flash-lite-preview",
    contents=[{"role": "user", "parts": [{"text": "What's 2+2?"}]}],
)
print(response.text)
