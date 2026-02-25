# Voice Pipeline — Test UI

A self-contained, zero-dependency HTML/JS interface for testing the API.

## How to open

**Option A – direct file (simplest):**
```bash
# Most browsers allow localhost fetch from file:// out of the box
open ui/index.html          # macOS
xdg-open ui/index.html      # Linux
```

**Option B – tiny HTTP server (if you see CORS or fetch errors):**
```bash
python3 -m http.server 3000 --directory ui/
# then visit http://localhost:3000
```

## What it does

| Feature | Detail |
|---|---|
| Health status | Live poll every 15 s — shows both models, GPU, uptime |
| Language selector | Populated from `GET /v1/languages`; auto-groups Indic vs Whisper |
| Drag & drop upload | Click or drop any audio file; inline preview player |
| Indic options | Decode mode — CTC / RNNT |
| Whisper options | Task, beam size, word timestamps |
| Result view | Full transcript, meta chips, segment timeline with hover tooltips |
| History | Last 20 results (click any row to re-display) |
| Copy button | One-click clipboard copy |
| Keyboard shortcut | `Ctrl+Enter` / `Cmd+Enter` to submit |

## Removing this folder

This folder contains only static files that call the API over HTTP.
**Deleting `ui/` has zero effect on the API.**
There are no imports, no shared code, and no references from the API codebase.

## Changing the API URL

The base URL defaults to `http://localhost:8000`.
Change it in the input bar at the top — no rebuild needed.
