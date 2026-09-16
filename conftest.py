"""Global pytest configuration — stubs required env vars before any import."""
import os

os.environ.setdefault("OPENAI_API_KEY", "sk-test-key")
