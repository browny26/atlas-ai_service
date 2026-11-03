# 🤖 Atlas AI Service

![Python](https://img.shields.io/badge/Python-3.12-blue?logo=python)
![FastAPI](https://img.shields.io/badge/FastAPI-009688?logo=fastapi&logoColor=white)
![Pydantic](https://img.shields.io/badge/Pydantic-v2-e92063?logo=pydantic)
![Groq](https://img.shields.io/badge/LLM-Groq%20%7C%20Ollama-orange)

The AI microservice of **Atlas**, a travel planning platform. It turns a trip request
(destination, days, budget, interests) into a **validated, structured itinerary** using an LLM.

The Spring Boot backend owns users, data and security; this service owns everything about AI:
prompts, model calls, output validation and cost calculation.

```mermaid
graph LR
  FE[React frontend] --> BE[Spring Boot backend]
  BE -->|POST /v1/itineraries<br/>X-API-Key| AI[Atlas AI Service<br/>FastAPI]
  AI -->|OpenAI-compatible API| LLM[(Groq / Ollama)]
  BE --> DB[(PostgreSQL)]
```

## How a request is processed

```mermaid
sequenceDiagram
  participant BE as Backend
  participant API as FastAPI route
  participant S as ItineraryService
  participant LLM as LLMClient
  BE->>API: ItineraryRequest (validated by Pydantic)
  API->>S: generate(request)
  loop up to GENERATION_MAX_ATTEMPTS
    S->>LLM: messages (JSON mode)
    LLM-->>S: raw JSON
    S->>S: validate against LLMItinerary
    alt invalid
      S->>S: add validation errors to the conversation ("self-repair")
    end
  end
  S->>S: compute total cost and budget status in Python
  S-->>BE: ItineraryResponse + metadata (source, attempts, tokens, latency)
```

## Design decisions

| Decision | Why |
|---|---|
| **One Pydantic model as the output contract** (`LLMItinerary`) | Its JSON Schema goes into the prompt, and the same model validates the answer, so prompt and parser cannot drift apart. |
| **JSON mode + Pydantic validation + self-repair** | JSON mode guarantees valid syntax. Pydantic checks the structure (times `HH:MM`, costs ≥ 0, days 1..N). If validation fails, the errors are sent back to the model so it can fix its answer. No regex parsing. |
| **The LLM proposes, Python calculates** | Costs are numbers (`cost_eur`), not strings like `"€20-40"`. The total and `budget_status` are computed in code, because LLMs are unreliable at arithmetic. |
| **`LLMClient` Protocol + dependency injection** | Business logic depends on an interface. Tests inject a `FakeLLM` (no network, no tokens spent). Groq, Ollama or OpenAI can be swapped through configuration, because they all expose an OpenAI-compatible API. |
| **Two retry levels** | The transport (timeouts, 429, 5xx) is retried with backoff by the SDK. Invalid output is retried by the service. |
| **Explicit fallback** | If the LLM fails, a template itinerary is returned with `metadata.source = "fallback"`, so the caller is never misled. It can be disabled (`ENABLE_FALLBACK=false` → HTTP 502). |
| **User input sent as data** | Trip fields are serialized as JSON in the user message, never concatenated into the instructions, and their length is limited (basic prompt-injection hygiene). |
| **Observability** | Every response reports model, prompt version, attempts, token usage and latency. These are also logged. |

## Project structure

```
app/
├── main.py                  # app factory (create_app) + lifespan (shared LLM client)
├── core/
│   ├── config.py            # typed settings from env vars (pydantic-settings)
│   └── logging.py
├── api/
│   ├── deps.py              # dependency injection + API key auth
│   ├── errors.py            # domain exceptions → HTTP responses
│   └── routes/              # thin HTTP layer: health, itineraries
├── schemas/itinerary.py     # request / LLM output / response models
├── services/
│   ├── itinerary_service.py # use case: prompt → LLM → validate → repair
│   ├── budget.py            # deterministic cost calculations
│   └── fallback.py
├── llm/
│   ├── base.py              # LLMClient Protocol, LLMResult, LLMError
│   └── openai_compatible.py # Groq / Ollama / OpenAI implementation
└── prompts/itinerary.py     # versioned prompts, separate from code
tests/                       # unit + API tests with a fake LLM
scripts/evaluate.py          # evaluation of the prompt against the real model
```

Dependencies flow in one direction: `api → services → llm / prompts / schemas`.
Nothing in `services` knows about HTTP or about a specific LLM vendor.

## Getting started

Requirements: Python 3.12+ and [uv](https://docs.astral.sh/uv/).

```bash
cp .env.example .env        # add your Groq key, or point LLM_BASE_URL to Ollama
uv sync
uv run uvicorn app.main:app --reload --port 8000
```

Interactive docs: http://localhost:8000/docs

With Docker: `docker compose up --build`.

**Free local mode with Ollama:** `ollama pull mistral:7b`, then set
`LLM_BASE_URL=http://localhost:11434/v1` and `LLM_MODEL=mistral:7b`.

## API

`POST /v1/itineraries` (header `X-API-Key` when `SERVICE_API_KEY` is set)

```json
{ "destination": "Rome, Italy", "days": 2, "budget_eur": 400, "interests": ["history", "food"] }
```

Response (shortened):

```json
{
  "destination": "Rome, Italy",
  "total_days": 2,
  "budget_eur": 400.0,
  "nights": 1,
  "total_estimated_cost_eur": 312.0,
  "budget_status": "within_budget",
  "days": [
    {
      "day": 1,
      "morning":   { "activity": "Colosseum and Roman Forum", "time": "09:00", "cost_eur": 18 },
      "afternoon": { "activity": "Pantheon and Piazza Navona", "time": "14:00", "cost_eur": 0 },
      "evening":   { "activity": "Dinner in Trastevere", "time": "20:00", "cost_eur": 35 }
    }
  ],
  "accommodation": { "name": "…", "type": "B&B", "details": "…", "cost_per_night_eur": 110 },
  "tips": ["Book the Colosseum online to skip the queue"],
  "metadata": {
    "source": "llm", "model": "llama-3.1-8b-instant", "prompt_version": "itinerary-v2",
    "attempts": 1, "latency_ms": 1840,
    "usage": { "prompt_tokens": 1210, "completion_tokens": 690, "total_tokens": 1900 }
  }
}
```

| Status | Meaning |
|---|---|
| 200 | Itinerary generated (check `metadata.source`) |
| 401 | Missing or wrong `X-API-Key` |
| 422 | Invalid request (e.g. `days` outside 1–14) |
| 502 | The LLM failed and fallback is disabled |

`GET /health` is a liveness probe and never calls the LLM.

## Quality

```bash
uv run pytest              # tests, no network needed
uv run ruff check .        # lint
uv run ruff format .       # formatting
uv run mypy app tests      # strict type checking
```

GitHub Actions runs all of these, plus a Docker build, on every push and pull request.

### Evaluating the prompt

Unit tests prove that the code is correct. They cannot prove that the *model* behaves well.
`scripts/evaluate.py` runs a set of sample trips against the real configured model and
reports the valid-output rate, how many answers were right at the first try, the
within-budget rate, latency and tokens. Use it to compare prompts or models before deploying:

```bash
uv run python -m scripts.evaluate
```

## Configuration

All settings are environment variables (see [`.env.example`](.env.example)):

| Variable | Default | Description |
|---|---|---|
| `LLM_BASE_URL` | Groq | Any OpenAI-compatible endpoint |
| `LLM_API_KEY` | – | Provider API key |
| `LLM_MODEL` | `llama-3.1-8b-instant` | Model name |
| `LLM_TIMEOUT_SECONDS` / `LLM_MAX_RETRIES` | `30` / `2` | Transport resilience |
| `GENERATION_MAX_ATTEMPTS` | `3` | Self-repair attempts on invalid output |
| `ENABLE_FALLBACK` | `true` | Return a flagged template instead of 502 |
| `SERVICE_API_KEY` | – | Shared secret with the backend (empty = auth disabled) |

## Deployment

`render.yaml` deploys the service on Render as a Docker web service. Set `LLM_API_KEY`
and `SERVICE_API_KEY` in the dashboard, then configure the same `SERVICE_API_KEY` and the
service URL in the backend.

## Possible next steps

- Streaming responses (SSE) to show the itinerary while it is generated
- Response caching for identical requests
- Native structured outputs (`json_schema`) on models that support it
- Grounding activities with real data (e.g. the Amadeus API the backend already uses)

## History

v1 was a single-file Flask prototype calling Mistral 7B through Ollama, with regex-based
JSON repair. v2 is a rewrite with a layered architecture, typed contracts, tests and CI.
