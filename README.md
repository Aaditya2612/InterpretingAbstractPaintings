---
title: Chroma — Abstract Painting Interpreter
emoji: 🎨
colorFrom: indigo
colorTo: purple
sdk: docker
app_port: 8000
pinned: false
---

# 🎨 Chroma — Abstract Painting Interpreter

Upload an abstract painting and let a machine-vision studio analyse its colour,
texture and composition, then write you a thoughtful interpretation of what it
sees.

Chroma combines three vision models with an LLM critic:

| Stage | Model | What it produces |
| --- | --- | --- |
| Visual features | OpenCV (classical CV) | Colour diversity, colourfulness, symmetry, edge chaos, texture, dominant palette |
| Concept tagging | CLIP (`clip-vit-base-patch32`) | Ranked artistic concepts with confidence |
| Description | derived from CLIP + features | A short, on-theme summary of what the painting is |
| Attention map | ResNet-50 + Grad-CAM | A heatmap of where the model "looks" |
| Interpretation | Groq LLaMA (or local fallback) | A 5–7 sentence art-critic write-up |

## ✨ Why it's fast

- **Models load once** at startup (warmed in the background) and are reused.
- **Parallel inference** — concept tagging and the heatmap run concurrently.
- **Result caching** by image hash — repeat uploads and the sample gallery
  resolve instantly (both server- and client-side).
- **Aggressive downscaling** of large uploads before inference.
- **No caption model** — the description is derived from CLIP concepts, cutting
  ~1 GB of weights and the slowest inference step.
- **CPU-friendly** — works without a GPU, with configurable thread count.

## 🚀 Run it locally

Requirements: **Python 3.10+**.

```bash
# 1. Create and activate a virtual environment (optional but recommended)
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS / Linux:
source .venv/bin/activate

# 2. Install CPU torch (skip this line if you have a CUDA GPU)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# 3. Install the rest
pip install -r requirements.txt

# 4. (Optional) enable Groq-powered interpretation
#    Copy .env.example to .env and set GROQ_API_KEY

# 5. Run
uvicorn app:app --reload
```

Open <http://localhost:8000>.

> The first request downloads ~700 MB of model weights (cached to `.cache/`).
> With `WARMUP=1` (default) this happens automatically in the background.

### Without an API key

The app is fully functional **without** `GROQ_API_KEY` — it falls back to a
deterministic local interpreter built from the detected features, description
and concepts. Add a key to upgrade to a large-language-model critic.

## 🐳 Run with Docker

```bash
docker build -t chroma .
docker run -p 8000:8000 -e GROQ_API_KEY=your_key_here chroma
```

## ☁️ Deploy

See **[DEPLOY.md](./DEPLOY.md)** for step-by-step guides.

TL;DR — Chroma is a single Dockerised FastAPI app and needs ~1 GB RAM:

- **Hugging Face Spaces (free, 16 GB RAM)** — the easiest $0 option and a great
  fit for these ML models. The `README.md` front-matter makes it a one-click
  Docker Space.
- **Render** — use the *Standard* plan (2 GB); the free/Starter tiers are only
  512 MB and will OOM.
- **Any VPS** — `docker build -t chroma . && docker run -d -p 8000:8000 chroma`.

The frontend is static and the API is stateless (apart from an in-memory result
cache), so it scales horizontally with a load balancer if needed.

## ⚙️ Configuration

All settings are read from environment variables (see `.env.example`):

| Variable | Default | Purpose |
| --- | --- | --- |
| `GROQ_API_KEY` | *(empty)* | Enables LLM interpretation |
| `GROQ_MODEL` | `llama-3.3-70b-versatile` | Groq model |
| `CLIP_MODEL` | `openai/clip-vit-base-patch32` | Concept tagger |
| `MAX_INPUT_DIM` | `384` | Downscale limit for inputs |
| `TORCH_THREADS` | `4` | Threads per inference process |
| `WARMUP` | `1` | Pre-load models on startup |
| `PORT` | `8000` | Server port |

## 📁 Project structure

```
app.py               FastAPI server
core/                Analysis engine
  config.py            settings
  features.py          OpenCV visual metrics
  tagger.py            CLIP concept tagging
  describer.py         concept-derived description
  heatmap.py           ResNet-50 Grad-CAM
  interpreter.py       Groq + local fallback
  pipeline.py          orchestration + caching
static/              Frontend (HTML / CSS / JS, no build step)
  samples/             example paintings
requirements.txt
Dockerfile
.env.example
```
