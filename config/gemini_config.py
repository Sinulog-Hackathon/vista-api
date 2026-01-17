from dotenv import load_dotenv
from google import genai
import os

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Global Gemini client instance
_client = None

# Initialize and return a singleton Gemini client instance
def get_gemini_client() -> genai.Client:
    global _client

    if _client is not None:
        return _client

    if not GEMINI_API_KEY:
        raise ValueError("GEMINI_API_KEY is missing.")

    # In the new SDK, it is simply genai.Client
    _client = genai.Client(api_key=GEMINI_API_KEY)
    return _client