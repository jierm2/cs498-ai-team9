from dotenv import load_dotenv
from google import genai
import os

load_dotenv()

_client = genai.Client()

def get_client():
    return _client
