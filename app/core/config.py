"""Application settings, loaded from environment variables (and an optional .env file).

Every tunable value lives here, so the rest of the code never reads os.environ directly.
"""

from functools import lru_cache

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    app_name: str = "Atlas AI Service"
    environment: str = "development"
    log_level: str = "INFO"

    # LLM provider. Any OpenAI-compatible API works: Groq in production,
    # Ollama locally (http://localhost:11434/v1), OpenAI, ...
    llm_base_url: str = "https://api.groq.com/openai/v1"
    llm_api_key: SecretStr = SecretStr("")
    llm_model: str = "llama-3.1-8b-instant"
    llm_temperature: float = Field(default=0.3, ge=0.0, le=2.0)
    llm_max_tokens: int = Field(default=2500, gt=0)
    llm_timeout_seconds: float = Field(default=30.0, gt=0)
    # Transport-level retries (timeouts, 429, 5xx), handled by the OpenAI SDK with backoff.
    llm_max_retries: int = Field(default=2, ge=0)

    # Output-level attempts: how many times we ask the model to fix an invalid answer.
    generation_max_attempts: int = Field(default=3, ge=1)
    # Return a clearly-flagged template itinerary when the LLM cannot produce a valid one.
    enable_fallback: bool = True

    # Shared secret for service-to-service calls (header X-API-Key). Empty = disabled.
    # No CORS: this service is called only by the backend, never by the browser.
    service_api_key: SecretStr = SecretStr("")


@lru_cache
def get_settings() -> Settings:
    return Settings()
