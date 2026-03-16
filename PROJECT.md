# Silo

**A unified CLI and server for local AI models on Apple Silicon.**

Silo is a single entrypoint to download, convert, serve, and orchestrate models — LLMs, speech-to-text, and text-to-speech — behind an OpenAI-compatible API. Supports MLX and llama.cpp backends.

---

## Problem

The MLX ecosystem is growing fast. There are separate tools for each domain:

- **mlx-lm** — text generation, conversion, quantization
- **mlx-audio** — speech-to-text, text-to-speech
- **mlx-whisper** — audio transcription

Each has its own CLI, its own conversion scripts, its own serving logic. Working across them means juggling multiple tools, remembering different flags, and writing glue code. There is no unified way to:

- Download and serve a model from HuggingFace in one command
- Serve STT, TTS, and LLM models behind a single OpenAI-compatible API
- Chain workflows (e.g., transcribe audio → summarize text → generate speech)
- Manage local model storage, conversion, and lifecycle

### Existing Tools

- [mlx-openai-server](https://pypi.org/project/mlx-openai-server/) — OpenAI-compatible server for MLX LLMs, but text-only, no STT/TTS, no process management
- [vllm-mlx](https://pypi.org/project/vllm-mlx/) — vLLM backend for MLX, focused on high-throughput LLM serving

Silo differs by treating **STT and TTS as first-class modalities**, providing **process orchestration** (up/down/ps), **multi-backend support** (MLX + llama.cpp), and a **unified model lifecycle** (download → convert → serve → manage).

## Solution

Silo provides:

1. **Unified CLI** — One command to download, convert, serve, and manage models
2. **OpenAI-Compatible Server** — Standalone server exposing standard endpoints for chat, STT, and TTS
3. **Automatic Model Acquisition** — Downloads models via `huggingface-cli`, auto-detects if conversion is needed
4. **Process Orchestration** — Docker Compose-style management of model instances (`up`, `down`, `ps`)
5. **Workflow Engine** — Prefect-powered pipelines across modalities (optional, future)

---

## Architecture

```
┌──────────────────────────────────────────────────────┐
│                     Silo CLI                      │
│  silo serve | convert | run | up | down | ps      │
└───────────────────────┬──────────────────────────────┘
                        │
            ┌───────────┴───────────┐
            │     Core Engine        │
            │  ┌────────────────┐   │
            │  │ Model Registry  │   │
            │  │ Process Manager │   │
            │  │ Config Loader   │   │
            │  │ HF Downloader   │   │
            │  └────────────────┘   │
            └──┬────────────────┬───┘
               │                │
      ┌────────▼────────┐      │
      │  OpenAI-Compat   │      │
      │  HTTP Server     │      │
      │  (per model)     │      │
      └────────┬────────┘      │
               │                │
      ┌────────▼────────────────▼──────────────┐
      │       Inference Backends (wrappers)     │
      │  ┌──────────┬──────────┬─────────────┐ │
      │  │  mlx-lm  │mlx-audio │ llama.cpp   │ │
      │  │  (chat)  │(STT/TTS) │ (GGUF chat) │ │
      │  └──────────┴──────────┴─────────────┘ │
      └────────────────┬───────────────────────┘
                       │
              ┌────────▼────────┐
              │  HuggingFace Hub │
              │  (huggingface-cli)│
              └─────────────────┘
```

### Key Design Decisions

- **Standalone server** — Silo is its own OpenAI-compatible server. It does NOT embed LiteLLM. In the broader system architecture, LiteLLM (or any OpenAI-compatible router) sits upstream and routes to Silo as a backend.
- **One model per process** — Each `silo serve` spawns a single-model server on its own port. Simple, isolated, independently restartable.
- **Wrapper, not reimplementation** — Silo wraps `mlx-lm`, `mlx-audio`, `llama-cpp-python`, etc. It does not reimplement conversion or inference logic.
- **Multi-backend** — MLX for STT/TTS and LLMs, llama.cpp for GGUF models. Auto-detects backend from model format, or user specifies explicitly. Same API either way.
- **Runs natively on macOS** — MLX requires direct access to Apple Silicon's Metal GPU and unified memory. Cannot run in Docker.
- **STT and TTS are first-class** — Not afterthoughts. Equal priority with LLM serving.
- **No auth, no HTTPS** — Silo is designed for local/LAN use. Authentication and TLS termination should be handled upstream (LiteLLM, nginx, Caddy, etc.). Auth may be added in a future version.

---

## Backend Interface

Backends are separated by modality. Each protocol defines only the methods relevant to that modality, so backends don't need to stub unrelated methods.

```python
class BaseBackend(Protocol):
    """Shared lifecycle methods for all backends."""

    def load(self, model_path: str, config: dict) -> None:
        """Load a model into memory. Called on startup or first request (lazy load)."""
        ...

    def unload(self) -> None:
        """Unload model from memory. Called on shutdown."""
        ...

    def health(self) -> dict:
        """Return health status: {"status": "ok", "model": "...", "memory_mb": ...}"""
        ...


class ChatBackend(BaseBackend, Protocol):
    """Backend for LLM chat completions."""

    def chat(
        self,
        messages: list[dict],
        stream: bool = False,
        **kwargs,
    ) -> Iterator[dict] | dict:
        """OpenAI-compatible chat completion."""
        ...


class SttBackend(BaseBackend, Protocol):
    """Backend for speech-to-text."""

    def transcribe(
        self,
        audio: bytes,
        language: str | None = None,
        response_format: str = "json",
    ) -> dict:
        """OpenAI-compatible audio transcription."""
        ...


class TtsBackend(BaseBackend, Protocol):
    """Backend for text-to-speech."""

    def speak(
        self,
        text: str,
        voice: str = "default",
        response_format: str = "wav",
        speed: float = 1.0,
        stream: bool = False,
    ) -> bytes | Iterator[bytes]:
        """OpenAI-compatible text-to-speech."""
        ...
```

Concrete implementations:
- **MlxLmBackend** → `ChatBackend`
- **MlxAudioSttBackend** → `SttBackend`
- **MlxAudioTtsBackend** → `TtsBackend`
- **LlamaCppBackend** → `ChatBackend`

The server layer inspects which protocol a backend implements and registers only the relevant API routes.

---

## CLI Interface

### Serve

Serve a model behind an OpenAI-compatible API. Auto-downloads via `huggingface-cli` if not cached locally. Auto-detects if the model is already MLX format or needs conversion.

```bash
# Serve a pre-converted MLX model (downloads automatically)
silo serve mlx-community/Llama-3.2-1B-4bit --port 8800

# Serve a Whisper STT model
silo serve mlx-community/whisper-large-v3-turbo --port 8801

# Serve a TTS model
silo serve mlx-community/kokoro-tts --port 8802

# Expose on local network
silo serve mlx-community/whisper-large-v3-turbo --host 0.0.0.0 --port 8801

# Serve a non-MLX model (auto-downloads, auto-converts)
silo serve meta-llama/Llama-3.2-1B --quantize q4 --output ~/models/llama-3.2-q4 --port 8800

# Serve and persist to config.yaml (so `silo up` includes it next time)
silo serve mlx-community/Llama-3.2-1B-4bit --port 8800 --save
```

### Process Management (Docker Compose-style)

Manage multiple model instances from a single config file.

```bash
# Bring up all models from config
silo up

# Bring up a specific model
silo up whisper

# Stop all models (graceful: drains in-flight requests, SIGTERM with 30s timeout)
silo down

# Stop a specific model
silo down tts

# Restart a model
silo restart whisper

# Show status of all running models
silo ps

# View logs
silo logs whisper
silo logs whisper --follow
silo logs --all --follow

# Generate a starter config.yaml interactively
silo init

# Diagnose common issues
silo doctor
```

### Convert

Convert non-MLX models to MLX format. Wraps `mlx-lm convert`, `mlx-audio`, etc.

```bash
# Convert with quantization, specify output directory
silo convert meta-llama/Llama-3.2-1B --quantize q4 --output ~/models/llama-3.2-q4

# Auto-detect modality
silo convert openai/whisper-large-v3 --output ~/models/whisper-large-v3

# Explicit modality
silo convert openai/whisper-large-v3 --type audio --output ~/models/whisper-v3

# Batch convert from a manifest
silo convert --from models.yaml
```

### Run (one-shot inference)

```bash
# Text generation
silo run mlx-community/Llama-3.2-1B-4bit "Explain quantum computing"

# Audio transcription
silo run mlx-community/whisper-large-v3-turbo recording.mp3

# Text-to-speech
silo run mlx-community/kokoro-tts "Hello world" --output hello.wav
```

### Models

```bash
# List locally cached models
silo models list

# Show model info
silo models info mlx-community/Llama-3.2-1B-4bit

# Remove from registry (does NOT delete from HF cache — other tools may use it)
silo models rm mlx-community/Llama-3.2-1B-4bit

# Remove from registry AND delete from HF cache
silo models rm mlx-community/Llama-3.2-1B-4bit --purge

# Search HuggingFace for MLX models
silo models search "whisper" --type audio
silo models search "llama" --mlx-only
```

### Flows (requires `silo[flows]`)

```bash
# Run a predefined flow
silo flow run transcribe-summarize --input meeting.mp3

# List available flows
silo flow list

# Create a flow from YAML
silo flow create my-flow.yaml

# Schedule a flow
silo flow schedule transcribe-summarize --cron "0 2 * * *" --input-dir ./recordings/

# View flow status
silo flow status

# Open Prefect dashboard
silo flow dashboard
```

### Doctor

Diagnose common setup issues:

```bash
silo doctor
```

Checks:
- Python version (3.12+)
- MLX installed and Metal available
- Apple Silicon detected
- `huggingface-cli` installed and authenticated
- `ffmpeg` installed (required for audio format conversion)
- Available unified memory
- Optional dependencies installed (llama.cpp, Prefect, Textual)
- Registry integrity (`~/.silo/registry.json`)

---

## API Endpoints

Each model instance serves the appropriate OpenAI-compatible endpoints based on its modality.

### LLM Models

```
POST /v1/chat/completions       — text generation (streaming via SSE)
GET  /v1/models                 — list served model
GET  /health                    — health check
GET  /metrics                   — Prometheus-compatible metrics
```

### STT Models (Speech-to-Text)

```
POST /v1/audio/transcriptions   — audio transcription (OpenAI Whisper-compatible)
GET  /v1/models                 — list served model
GET  /health                    — health check
GET  /metrics                   — Prometheus-compatible metrics
```

### TTS Models (Text-to-Speech)

```
POST /v1/audio/speech           — text-to-speech (streaming audio output)
GET  /v1/models                 — list served model
GET  /health                    — health check
GET  /metrics                   — Prometheus-compatible metrics
```

Supported audio formats: `wav` (native), `mp3`, `opus`, `aac`, `flac`, `pcm` (via `ffmpeg`)

### Model Name Resolution

When a client sends a request, the `model` field in the request body maps to the `name` field in `config.yaml`:

```yaml
# config.yaml
models:
  - name: llama        # ← clients use {"model": "llama"} in requests
    repo: mlx-community/Llama-3.2-1B-4bit
    port: 8800
```

For `silo serve` without config, the model name defaults to the repo ID (e.g., `mlx-community/Llama-3.2-1B-4bit`).

### Error Responses

All errors follow the OpenAI error format:

```json
{
  "error": {
    "message": "Model not loaded: whisper-large-v3-turbo",
    "type": "model_not_found",
    "code": 404
  }
}
```

Error types: `model_not_found`, `model_loading`, `invalid_request`, `backend_error`, `out_of_memory`, `timeout`

### OpenAI Compatibility Scope

**Supported:**
- `POST /v1/chat/completions` — messages, temperature, top_p, max_tokens, stream, stop
- `POST /v1/audio/transcriptions` — file, language, response_format (json, text, verbose_json)
- `POST /v1/audio/speech` — input, voice, response_format, speed
- `GET /v1/models` — list available models

**Not supported (out of scope):**
- Function calling / tool use
- Logprobs
- Embeddings (`/v1/embeddings`)
- Image generation (`/v1/images/generations`)
- Fine-tuning API
- Batch API
- Assistants API

**Deferred to v2.0:**
- Vision input in `/v1/chat/completions` (image_url in messages)

### Metrics

Each model instance exposes `/metrics` in Prometheus format:

```
silo_requests_total{model="llama", method="POST", endpoint="/v1/chat/completions", status="200"} 142
silo_request_duration_seconds{model="llama", quantile="0.95"} 1.23
silo_tokens_per_second{model="llama"} 45.2
silo_model_memory_bytes{model="llama"} 1073741824
silo_model_loaded{model="llama"} 1
```

---

## Configuration

### config.yaml

Persistent configuration for managing multiple models. Validated with Pydantic models at load time — invalid config fails fast with clear error messages.

**Location:** `~/.silo/config.yaml` (canonical). Can be overridden with `--config <path>` on any command.

```yaml
# ~/.silo/config.yaml
models:
  - name: llama
    repo: mlx-community/Llama-3.2-1B-4bit
    host: 0.0.0.0
    port: 8800
    warmup: true              # preload model on `silo up` (default: false = lazy load)
    restart: on-failure       # always | on-failure | never (default: on-failure)
    timeout: 120              # request timeout in seconds (default: 120)

  - name: whisper
    repo: mlx-community/whisper-large-v3-turbo
    host: 0.0.0.0
    port: 8801
    warmup: true
    restart: always
    timeout: 60

  - name: tts
    repo: mlx-community/kokoro-tts
    host: 0.0.0.0
    port: 8802
    restart: always
    timeout: 30

  # Non-MLX model — will be auto-converted on first `silo up`
  - name: custom-llama
    repo: meta-llama/Llama-3.2-1B
    quantize: q4
    output: ~/models/llama-3.2-q4
    host: 127.0.0.1
    port: 8803

  # GGUF model via llama.cpp backend
  - name: qwen
    repo: bartowski/Qwen2.5-7B-Instruct-GGUF
    backend: llamacpp
    host: 0.0.0.0
    port: 8804
```

### Environment Variable Overrides

All config values can be overridden via environment variables for headless/scripted use:

```bash
SILO_HOST=0.0.0.0 SILO_PORT=9000 silo serve mlx-community/Llama-3.2-1B-4bit
```

Pattern: `SILO_<KEY>` for global settings. Per-model overrides use the config file.

### Global State

```
~/.silo/
  config.yaml           # canonical config location
  registry.json         # tracks all known models + metadata (points to HF cache paths)
  logs/                 # structured JSON log files per model instance
    llama.log
    whisper.log
    tts.log
  pids/                 # PID files for running model processes
```

Models are stored in the HuggingFace cache (`~/.cache/huggingface/hub/`). Silo does not duplicate model files — the registry tracks model metadata and resolves paths to the HF cache. Converted models (from `silo convert --output`) are stored at the user-specified location.

---

## Process Lifecycle

### Startup (`silo up`)

1. Parse and validate `config.yaml`
2. For each model:
   - Check if already downloaded; if not, run `huggingface-cli download`
   - Check if conversion needed; if so, run conversion
   - Spawn a server process on the configured port
   - If `warmup: true`, preload model into memory
   - If `warmup: false` (default), model loads on first request (lazy load)
3. Write PID file to track running processes
4. Begin health check polling

### Shutdown (`silo down`)

1. Send SIGTERM to all model processes
2. Each process drains in-flight requests (30s grace period)
3. If still running after grace period, send SIGKILL
4. Clean up PID files

### Crash Recovery

Controlled by `restart` policy per model:

- **`always`** — Restart immediately with exponential backoff (1s, 2s, 4s, ... max 60s)
- **`on-failure`** — Restart only on non-zero exit (default)
- **`never`** — Do not restart; mark as `exited` in `silo ps`

---

## Memory Management

Each model instance runs as a separate process. For v1, memory management is manual — use `silo ps` to see memory usage per model, and `silo down <model>` to free memory.

`silo ps` shows:

```
NAME      REPO                                    PORT   STATUS    MEMORY
llama     mlx-community/Llama-3.2-1B-4bit         8800   running   1.2 GB
whisper   mlx-community/whisper-large-v3-turbo     8801   running   0.8 GB
tts       mlx-community/kokoro-tts                 8802   idle      —
```

Automatic memory pressure monitoring and LRU-based unloading is planned for v0.8.

---

## Flows (Workflow Engine)

Flows chain operations across modalities. Defined in YAML. Requires the optional `silo[flows]` install for Prefect integration.

```yaml
name: transcribe-and-summarize
description: Transcribe audio and summarize the result

steps:
  - id: transcribe
    model: mlx-community/whisper-large-v3-turbo
    type: audio.transcribe
    input: $input

  - id: summarize
    model: mlx-community/Llama-3.2-1B-4bit
    type: text.generate
    input: |
      Summarize the following transcript concisely:
      {{ steps.transcribe.output }}

output: $steps.summarize.output
```

```yaml
name: batch-transcribe
description: Transcribe all audio files in a directory

schedule: "0 2 * * *"  # nightly at 2am

config:
  retries: 3
  retry_delay: 30
  concurrency: 4
  cache: true

steps:
  - id: discover
    type: fs.glob
    input: "{{ input_dir }}/**/*.mp3"

  - id: transcribe
    model: mlx-community/whisper-large-v3-turbo
    type: audio.transcribe
    input: $steps.discover.output
    map: true

  - id: save
    type: fs.write
    input:
      content: $steps.transcribe.output
      path: "{{ output_dir }}/{{ item.stem }}.txt"
    map: true

output: $steps.save.output
```

### Why Prefect

- **Observability** — Dashboard with logs, timing, and state for every flow run
- **Retries** — Failed steps retry automatically with backoff
- **Scheduling** — Run flows on cron
- **Caching** — Skip steps whose inputs haven't changed
- **Concurrency** — Parallel step execution where the DAG allows

---

## Upstream Integration

Silo is designed to sit behind a routing layer. Example LiteLLM config pointing at Silo instances:

```yaml
# litellm_config.yaml
model_list:
  - model_name: llama-3.2
    litellm_params:
      model: openai/llama
      api_base: http://mac-mini.local:8800/v1
      api_key: unused

  - model_name: whisper
    litellm_params:
      model: openai/whisper
      api_base: http://mac-mini.local:8801/v1
      api_key: unused

  - model_name: tts
    litellm_params:
      model: openai/tts
      api_base: http://mac-mini.local:8802/v1
      api_key: unused

  # Cloud fallback when local is down
  - model_name: llama-3.2
    litellm_params:
      model: groq/llama-3.2-1b
      api_key: os.environ/GROQ_API_KEY

router_settings:
  routing_strategy: latency-based
  fallbacks:
    - llama-3.2: [groq/llama-3.2-1b]
```

This keeps Silo focused on inference and model management, while LiteLLM handles routing, auth, rate limiting, cost tracking, and cloud fallback.

---

## Tech Stack

- **Language**: Python 3.12+
- **Package Manager**: uv
- **CLI Framework**: Typer
- **Server**: FastAPI (OpenAI-compatible HTTP server, auto-generated OpenAPI docs)
- **Orchestration**: Prefect (optional, via `silo[flows]`)
- **TUI**: Textual (optional, via `silo[tui]`)
- **ML Backends**: mlx-lm, mlx-audio, llama-cpp-python (wrappers, not reimplemented)
- **Model Download**: huggingface-cli
- **Audio Encoding**: ffmpeg (system dependency for non-WAV formats)
- **Config**: YAML (PyYAML / ruamel.yaml), validated with Pydantic
- **Logging**: Structured JSON (stdout + file, read by `silo logs`)
- **Metrics**: Prometheus-compatible `/metrics` endpoint
- **Testing**: pytest, pytest-asyncio

## Dependencies

### System

```
ffmpeg                 # required for mp3/opus/aac/flac audio encoding (WAV/PCM work without it)
```

### Core

```
mlx >= 0.31
mlx-lm >= 0.31
mlx-audio >= 0.4
typer >= 0.24
huggingface-hub >= 1.6
pyyaml >= 6.0
uvicorn >= 0.41
fastapi >= 0.135
pydantic >= 2.12
prometheus-client >= 0.24
```

### Optional (llamacpp)

```
llama-cpp-python >= 0.3
```

### Optional (flows)

```
prefect >= 3.6
```

### Optional (tui)

```
textual >= 5.0
```

### Dev

```
pytest >= 9.0
pytest-asyncio >= 1.3
httpx >= 0.28          # async test client
respx >= 0.22          # mock HTTP for HF API tests
```

---

## Testing Strategy

### Unit Tests

- **Backend wrappers** — Mock `mlx-lm`, `mlx-audio`, `llama-cpp-python` imports. Test that the Backend protocols are correctly implemented.
- **Config loader** — Test YAML parsing, validation, defaults, env var overrides.
- **Model registry** — Test add/remove/search/detect operations on `registry.json`.
- **Process manager** — Test spawn/kill/restart logic with mock subprocesses.

### Integration Tests

- **API endpoints** — Spin up a server with a mock backend, test OpenAI-compatible request/response format, streaming, error responses.
- **HuggingFace download** — Mock `huggingface-cli` subprocess, test auto-detect and caching logic.
- **`silo up/down`** — Test full lifecycle with mock config and mock backends.

### E2E Tests (requires real models — CI-optional)

- Download a small MLX model, serve it, send a real request, verify response.
- Transcribe a short audio file via STT endpoint.
- Generate audio via TTS endpoint, verify output format.

Target: **80%+ coverage** on unit + integration. E2E tests run in a separate CI stage with model caching.

---

## Milestones

### v0.1 — Foundation

- [ ] Project setup (uv, pyproject.toml, CLI skeleton with Typer)
- [ ] Backend protocol definitions (`ChatBackend`, `SttBackend`, `TtsBackend`)
- [ ] Model registry (`~/.silo/registry.json`)
- [ ] `silo models list | info | rm [--purge] | search`
- [ ] Auto-download via `huggingface-cli` (exact repo ID)
- [ ] Auto-detect MLX vs non-MLX models
- [ ] `silo convert` wrapping `mlx-lm` (with `--quantize` and `--output`)
- [ ] `silo run` for one-shot text generation
- [ ] `silo doctor` — environment diagnostics
- [ ] `silo init` — generate starter config.yaml

### v0.2 — LLM Serving

- [ ] `silo serve` for LLM models (OpenAI `/v1/chat/completions`)
- [ ] Streaming support (SSE)
- [ ] `--host` and `--port` flags, LAN exposure
- [ ] Environment variable overrides (`SILO_HOST`, `SILO_PORT`)
- [ ] `/health` endpoint
- [ ] OpenAI-compatible error responses
- [ ] Model name resolution (config `name` → request `model` field)

### v0.3 — STT & TTS Serving

- [ ] STT serving (`/v1/audio/transcriptions`) wrapping mlx-audio/whisper
- [ ] TTS serving (`/v1/audio/speech`) wrapping mlx-audio
- [ ] Streaming TTS (chunked audio output)
- [ ] Configurable request timeout per model
- [ ] Audio format support: `wav` native, `mp3`/`opus`/`aac`/`flac`/`pcm` via ffmpeg
- [ ] `ffmpeg` detection in `silo doctor`

### v0.4 — Process Management

- [ ] `config.yaml` for persistent multi-model configuration (canonical: `~/.silo/config.yaml`)
- [ ] Pydantic config validation with clear error messages
- [ ] `silo up | down | ps | restart`
- [ ] `silo logs <model> [--follow]`
- [ ] Per-model port assignment from config
- [ ] Auto-download and auto-convert on `silo up`
- [ ] Warm-up vs lazy load (configurable per model)
- [ ] Graceful shutdown (drain requests, SIGTERM, 30s grace, SIGKILL)
- [ ] Crash recovery with restart policy (`always | on-failure | never`)
- [ ] Structured JSON logging (stdout + `~/.silo/logs/<model>.log`)
- [ ] `--save` flag on `silo serve` to persist to config.yaml

### v0.5 — llama.cpp Backend (optional install)

- [ ] `silo[llamacpp]` optional dependency group (llama-cpp-python)
- [ ] llama.cpp `ChatBackend` wrapper (load GGUF, chat completions, streaming)
- [ ] Auto-detect backend from model format (MLX safetensors vs GGUF)
- [ ] Explicit `--backend llamacpp` flag and `backend:` config key
- [ ] `silo serve bartowski/Qwen2.5-7B-Instruct-GGUF --backend llamacpp`
- [ ] Same OpenAI-compatible API regardless of backend
- [ ] Mixed backends in `config.yaml` (e.g., mlx-audio for STT + llama.cpp for LLM)

### v0.6 — `/metrics` & Observability

- [ ] `/metrics` endpoint (Prometheus-compatible) per model instance
- [ ] Tokens/sec, request count, latency, memory usage metrics
- [ ] `silo run` for STT and TTS one-shot inference

### v0.7 — Terminal UI (optional install)

- [ ] `silo[tui]` optional dependency group (Textual)
- [ ] `silo ui` launches the TUI dashboard
- [ ] Live model status table (running, idle, loading, unloaded)
- [ ] Real-time request log (method, endpoint, status, latency)
- [ ] Memory gauge (unified memory usage, per-model breakdown)
- [ ] Keyboard shortcuts to manage models (up, down, restart, logs)
- [ ] Log viewer — drill into a model's structured logs
- [ ] Model quick-launch (search HF, download, serve from the TUI)

### v0.8 — Advanced Features

- [ ] Automatic memory pressure monitoring and LRU-based model unloading
- [ ] WebSocket endpoint for real-time streaming STT
- [ ] GGUF quantization support in `silo convert` (wrapping llama.cpp tooling)
- [ ] Batch convert from manifest (`--from models.yaml`)
- [ ] Progress bars for downloads and conversions
- [ ] Shell completions

### v0.9 — Prefect Flows (optional install)

- [ ] `silo[flows]` optional dependency group
- [ ] YAML flow parser → Prefect flow/task compilation
- [ ] Flow execution with retries and caching
- [ ] Parallel step execution (map over inputs)
- [ ] `flow schedule` for cron-based runs
- [ ] `flow dashboard` to launch Prefect UI
- [ ] Built-in flow templates (transcribe-summarize, batch-transcribe)

### v2.0 — Vision Language Models

- [ ] VLM support via mlx-vlm (image description, visual Q&A)
- [ ] VLM serving via `/v1/chat/completions` with image input

---

## Name

**Silo** — your models, locally contained.
