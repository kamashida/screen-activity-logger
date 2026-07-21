# screen-activity-logger

Turn screen recordings (MP4) into **structured, timestamped work logs — local by default**. Three contexts are extracted and merged: ① on-screen text (OCR), ② what the user is doing described by a VLM, and ③ speech (ASR). Gemini and Anthropic are available only with explicit external-send approval.

> Local VLM runs do not send data to cloud APIs. Cloud VLM runs send frames externally, so check permission, retention, and contract requirements before using confidential or client recordings.

*日本語版: [README.md](./README.md)*

## Example output

```markdown
## 00:05:12〜00:12:30 (7m18s) — Excel — quote_2026Q2.xlsx (Sheet1)   ← heading with dwell time
👁 Checking the unit-price totals in column D       ← focus (VLM inference)
🗣️ "This unit price changed from last month"        ← speech (kotoba-whisper)
Editing a unit-price cell                            ← action (Qwen3-VL)
- `quote_2026Q2.xlsx - Excel`                       ← primary evidence (raw OCR, always kept)
```

## Why

- Re-watching screen recordings to write up "what was I doing" is heavy manual work.
- OCR alone misses *what is being done*; audio alone misses *what happened on screen*.
- Market research conclusion: no existing tool understands recorded MP4s locally with both "eyes (OCR+VLM) and ears (ASR)" (see docs/research/round4).

## Stack

| Layer | Choice | Notes |
|----|------|------|
| Frame extraction | ffmpeg (uniform fps + scene detection, long edge 1024px) | |
| OCR | PaddleOCR PP-OCRv6 tiny/small/medium (Japanese, CPU) | diff-based skipping; keyframes-only in meeting mode |
| VLM | Qwen3-VL 8B via Ollama or vllm-mlx (**~40x faster on Apple Silicon**, #8) | structured 5-field output (app/resource/location/focus/action), timeout + retry + fallback |
| ASR | kotoba-whisper v2.0 (macOS: whisper.cpp Metal / Windows & Linux: faster-whisper, auto-selected) | Japanese-specialized; mlx-whisper deprecated after 2-hour tests showed content collapse (#22); silent videos auto-skip |
| VLM gate | OCR-token Jaccard (meeting mode) | suppresses wasted VLM calls on speaker-view switches |
| Output | JSONL (machine, keeps raw evidence) + Markdown (human) | includes per-entry dwell time and per-app time summary |

## Quick start (macOS)

```bash
uv venv -p 3.12 .venv
uv sync --locked --extra dev --extra asr
ollama pull qwen3-vl:8b

.venv/bin/python -m screen_activity_logger.cli recording.mp4 -o out/
# meeting recordings:
.venv/bin/python -m screen_activity_logger.cli meeting.mp4 --mode meeting -o out/
```

## Quick start (Windows)

```powershell
winget install Gyan.FFmpeg
# Install Ollama for Windows: https://ollama.com/download/windows
ollama pull qwen3-vl:8b

uv venv -p 3.12 .venv
uv sync --locked --extra dev --extra asr-faster
uv run python -m screen_activity_logger.cli recording.mp4 -o out/
```

### Cloud VLM (explicit opt-in)

Cloud providers require an explicit model, an API-key environment variable, HTTPS, and
`--allow-external-vlm`. The key value is never passed as a CLI argument.

Use `--no-vlm` explicitly to run an OCR/ASR baseline without contacting any VLM.
This validates extraction and output wiring, but is not a replacement for VLM scene understanding.

```powershell
$env:GEMINI_API_KEY = "your-key"
uv run screen-activity-logger recording.mp4 --no-asr `
  --vlm-provider gemini --model <gemini-model> `
  --vlm-api-key-env GEMINI_API_KEY --allow-external-vlm -o out/

$env:ANTHROPIC_API_KEY = "your-key"
uv run screen-activity-logger recording.mp4 --no-asr `
  --vlm-provider anthropic --model <claude-model> `
  --vlm-api-key-env ANTHROPIC_API_KEY --allow-external-vlm -o out/
```

On macOS install whisper.cpp for the fast ASR path (`brew install whisper-cpp` plus a kotoba-whisper
GGUF at `~/.cache/screen-activity-logger/kotoba-whisper-v2.0-q5_0.bin`; without it the tool falls back
to faster-whisper — slower but correct). On Windows/Linux, ASR uses faster-whisper (CTranslate2) with automatic CUDA detection; CPU int8 works with zero extra setup. For NVIDIA GPUs add `pip install nvidia-cublas-cu12 nvidia-cudnn-cu12`.


## Use from Claude Code (skill)

```bash
ln -s "$(pwd)/skills/screen-activity-logger" ~/.claude/skills/screen-activity-logger
```

Then ask Claude Code: "turn this recording into a work log" — it handles prerequisite checks, mode selection, execution, and result summaries.

## Modes

| | screencast (default) | meeting |
|---|---|---|
| For | tutorials, demos, work sessions | Meet/Zoom recordings |
| OCR | all frames (diff-skip) | keyframes only |
| VLM | every keyframe (never drop a step) | only when on-screen text changes |

Batch multiple videos in one command (two-phase: all ASR first, then per-video OCR/VLM.
faster/mlx load the model once; whisper.cpp spawns a lightweight process per video.
Silent videos are skipped per-video):

```bash
.venv/bin/python -m screen_activity_logger.cli mtg1.mp4 mtg2.mp4 --mode meeting -o out/
```

## VLM backend: vllm-mlx (recommended on Apple Silicon, ~40x faster)

```bash
pip install vllm-mlx   # separate venv recommended
vllm-mlx serve mlx-community/Qwen3-VL-8B-Instruct-4bit --port 8991
.venv/bin/screen-activity-logger meeting.mp4 --mode meeting --vlm-backend vllm-mlx -o out/
```

Measured on M4 Air 24GB: a 5-minute meeting clip completes in **~63 seconds** (vs 15-80+ minutes via Ollama), and a **2-hour recording completes in 31 minutes with no late-half quality degradation** (verified end-to-end, Issue #22).

## Key options

```
--mode screencast|meeting     processing profile (default: screencast)
--format worklog|manual       output a step-structured visual manual (manual.md) instead of
                              the timeline log — for screencasts; no extra AI calls (default: worklog)
--save-frames                 save per-entry keyframe images under frames/ and embed them in
                              the Markdown (off by default: raw screen images may be sensitive)
--asr-backend auto|cpp|faster|mlx ASR backend (auto: Apple Silicon→cpp via whisper.cpp,
                              falls back to faster when whisper.cpp is missing; other OSes→faster.
                              mlx is deprecated — its output collapses on long recordings, Issue #22)
--no-asr                      disable speech recognition
--speech-summary              add a one-sentence Japanese gist (🧭) per entry, generated by
                              the same VLM backend/model with text-only input (zero extra
                              memory, ~+8% runtime; default off)
--ocr-tier tiny|small|medium  OCR model size (default: small)
--vlm-timeout 300             per-attempt VLM timeout seconds (auto-retry once on timeout, with per-call telemetry)
--vlm-skip-threshold 0.85     VLM gate Jaccard threshold (meeting mode)
--keep-fillers                keep filler-only speech lines ("はい", "えーと"; cut by default)
--no-asr-filter               disable the ASR hallucination filter (debug; also disables filler cut)
```

## Design principles

- **Two-layer separation**: OCR reads text (facts), the VLM explains activity (interpretation). Facts override guesses; raw OCR lines are always preserved in JSONL so any interpretation can be re-derived later.
- **Ports & adapters**: the domain layer has zero OS/backend branches. Swapping macOS whisper.cpp for Windows faster-whisper changes exactly one adapter.
- **Paid-for computation gets reused**: collapsed duplicate entries become dwell-time tracking; per-app time summaries come for free.

## Requirements

- Python 3.12, ffmpeg, and Ollama ≥ 0.30 (qwen3-vl:8b) or vllm-mlx for local VLM. Gemini/Anthropic require an API key and explicit external-send approval.
- macOS (Apple Silicon), Windows, or Linux
- **~16GB RAM recommended** (the 4-bit 8B VLM uses 6–8GB during inference; developed on 24GB)

### First-run model downloads (~8GB total, one-time, free)

For the local provider, inference runs locally, so models must be fetched once (offline afterwards, no API fees):

| Model | Role | Size | How |
|---|---|---|---|
| Qwen3-VL 8B (VLM/LMM) | screen understanding | **~6GB** | `ollama pull qwen3-vl:8b`, or auto on first vllm-mlx start |
| PaddleOCR (Japanese) | on-screen text | a few hundred MB | auto on first run |
| kotoba-whisper | speech recognition | 0.5–1.5GB | auto (faster-whisper) / manual GGUF for whisper.cpp |

For low-spec machines, **Qwen3-VL 4B (~3GB, 1.7x faster, comparable quality on meeting frames)** works:
`--model mlx-community/Qwen3-VL-4B-Instruct-4bit` (vllm-mlx). See docs/research/issue2_tuning_results.md.

## Privacy

Screen recordings may contain passwords and personal data. Local-provider recordings/outputs are excluded from the repository via `.gitignore`. Cloud-provider runs transmit frames to the selected provider; verify permission and retention settings, and consider encrypting output storage.

## License

[Apache-2.0](./LICENSE). Check individual model licenses separately (Qwen: Apache-2.0, kotoba-whisper: Apache-2.0, Ruri v3: Apache-2.0).
