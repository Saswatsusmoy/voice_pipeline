# Voice Pipeline API

Production-ready ASR (Automatic Speech Recognition) API supporting **22 Indic languages** and **English + 30+ other languages**.

| Backend | Languages | Decode modes |
|---|---|---|
| [ai4bharat/indic-conformer-600m-multilingual](https://huggingface.co/ai4bharat/indic-conformer-600m-multilingual) | Hindi, Bengali, Tamil, Telugu, … (22 total) | `ctc` (fast), `rnnt` (accurate) |
| OpenAI Whisper (`base`) | English, French, German, … (30+) | beam-search, temperature fallback |

---

## Quick start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Copy and edit config
cp .env.example .env

# 3. Run
python run.py
```

API docs at **http://localhost:8000/docs**

---

## Docker

```bash
# CPU
docker compose up --build

# GPU (requires nvidia-container-toolkit)
docker compose --profile gpu up --build
```

---

## API Endpoints

### `POST /v1/transcribe`

Upload an audio file for transcription.

**Form fields**

| Field | Type | Required | Default | Description |
|---|---|---|---|---|
| `file` | file | Yes | – | Audio file (WAV, FLAC, MP3, OGG, OPUS, WebM, M4A, AAC) |
| `language` | string | No | `en` | Language code. Indic: `hi`, `bn`, `ta` … – Other: `en`, `fr`, `de` … |
| `decode_mode` | string | No | `ctc` | `ctc` or `rnnt` (Indic only) |
| `task` | string | No | `transcribe` | `transcribe` or `translate` (Whisper only) |
| `word_timestamps` | bool | No | `false` | Return per-word timestamps (Whisper only) |
| `initial_prompt` | string | No | – | Context hint for Whisper |
| `beam_size` | int | No | `5` | Whisper beam width (1–10) |

**Example – Hindi (Indic)**
```bash
curl -X POST http://localhost:8000/v1/transcribe \
  -F "file=@audio.wav" \
  -F "language=hi" \
  -F "decode_mode=rnnt"
```

**Example – English (Whisper) with word timestamps**
```bash
curl -X POST http://localhost:8000/v1/transcribe \
  -F "file=@audio.flac" \
  -F "language=en" \
  -F "word_timestamps=true"
```

**Response**
```json
{
  "request_id": "a1b2c3d4e5f6",
  "text": "नमस्ते, आप कैसे हैं?",
  "language": "hi",
  "language_name": "Hindi",
  "model": "indic-conformer-600m",
  "decode_mode": "ctc",
  "duration_seconds": 2.5,
  "processing_time_ms": 312.4,
  "segments": null,
  "word_timestamps": null
}
```

---

### `GET /v1/health`

Full health check with model status, GPU info, and request statistics.

### `GET /v1/ready`

Kubernetes readiness probe – returns `200` once both models are loaded.

### `GET /v1/live`

Kubernetes liveness probe – always returns `200`.

### `GET /v1/languages`

List all supported language codes.

### `GET /v1/languages/{code}`

Details for a single language code.

---

## Configuration

All settings are read from environment variables (or `.env`).

| Variable | Default | Description |
|---|---|---|
| `INDIC_MODEL_ID` | `ai4bharat/indic-conformer-600m-multilingual` | HuggingFace model ID |
| `WHISPER_MODEL_SIZE` | `base` | `tiny` / `base` / `small` / `medium` / `large` |
| `DEVICE` | `auto` | `auto` / `cpu` / `cuda` / `cuda:0` |
| `MODEL_CACHE_DIR` | `./model_cache` | Where downloaded models are stored |
| `WARMUP_ON_STARTUP` | `true` | JIT warm-up on first request |
| `MAX_AUDIO_DURATION_SECONDS` | `300` | Reject audio longer than this |
| `MAX_UPLOAD_SIZE_MB` | `50` | Reject uploads larger than this |
| `MAX_CONCURRENT_REQUESTS` | `4` | Semaphore cap for GPU inference |
| `REQUEST_TIMEOUT_SECONDS` | `120` | Per-request inference timeout |
| `LOG_LEVEL` | `INFO` | `DEBUG` / `INFO` / `WARNING` / `ERROR` |
| `ENVIRONMENT` | `production` | `development` outputs plain text logs; others use JSON |

---

## Project structure

```
voice_pipeline/
├── app/
│   ├── main.py                  # FastAPI app + lifespan
│   ├── core/
│   │   ├── config.py            # Settings (pydantic-settings)
│   │   └── logging.py           # Structured JSON logging
│   ├── models/
│   │   ├── indic_model.py       # Indic Conformer wrapper
│   │   ├── whisper_model.py     # Whisper wrapper
│   │   └── model_manager.py     # Lifecycle + semaphore
│   ├── schemas/
│   │   └── response.py          # Pydantic response schemas
│   ├── services/
│   │   └── transcription.py     # Route audio → correct model
│   ├── utils/
│   │   └── audio_utils.py       # Load, validate, resample
│   └── api/v1/
│       ├── router.py
│       └── endpoints/
│           ├── transcribe.py
│           ├── health.py
│           └── languages.py
├── run.py                       # Uvicorn entrypoint
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
└── .env.example
```

---

## Supported Indic languages

`as` · `bn` · `brx` · `doi` · `gu` · `hi` · `kn` · `kok` · `ks` · `mai` · `ml` · `mni` · `mr` · `ne` · `or` · `pa` · `sa` · `sat` · `sd` · `ta` · `te` · `ur`
