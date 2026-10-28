---
title: Misty Edges — Interpreting Abstract Paintings
emoji: 🎨
colorFrom: indigo
colorTo: purple
sdk: docker
app_port: 8000
pinned: false
---

# 🎨 Misty Edges — Interpreting Abstract Paintings

Upload an abstract painting and a machine-vision studio will analyse what it
resembles, the themes and emotions it carries, and then write you a thoughtful
interpretation of what it sees.

Misty Edges runs three recognisers plus an interpreter:

| Stage | Model | What it produces |
| --- | --- | --- |
| Visual features | OpenCV (classical CV) | Colour diversity, colourfulness, symmetry, edge chaos, texture, dominant palette |
| Subject / Theme / Emotion recognition | CLIP (`clip-vit-base-patch32`) | What the forms resemble, the underlying themes, and the emotional tone |
| Artistic concepts | CLIP (`clip-vit-base-patch32`) | Ranked art-style descriptors with confidence |
| Attention map | ResNet-50 + Grad-CAM | A heatmap of where the model "looks" |
| Interpretation | Groq LLaMA (or local fallback) | A 5–7 sentence, original art-critic write-up |

## ✨ Why it's fast

- **Models load once** at startup (warmed in the background) and are reused.
- **Parallel inference** — recognition and the heatmap run concurrently.
- **Result caching** by image hash — repeat uploads and the sample gallery
  resolve instantly (both server- and client-side).
- **Aggressive downscaling** of large uploads before inference.
- **One recognition pass** — subjects, themes, emotions and concepts are scored
  by a single CLIP forward pass.
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
varied local interpreter built from the detected subject, themes, emotions and
features. Add a key to upgrade to a large-language-model critic.

## 🐳 Run with Docker

```bash
docker build -t misty-edges .
docker run -p 8000:8000 -e GROQ_API_KEY=your_key_here misty-edges
```

## ☁️ Deploy

See **[DEPLOY.md](./DEPLOY.md)** for step-by-step guides.

TL;DR — Misty Edges is a single Dockerised FastAPI app and needs ~1 GB RAM:

- **Oracle Cloud "Always Free" (free forever, India regions)** — the best free
  option: a 2-core / 12 GB ARM VM at no cost. See DEPLOY.md for step-by-step.
- **Google Cloud Run (free tier, Mumbai region)** — serverless; free within the
  monthly allowance.
- **Hugging Face Spaces** — note: the free CPU tier was recently removed for
  new accounts (Docker spaces are now paid).
- **Render** — use the *Standard* plan (2 GB); the free/Starter tiers are only
  512 MB and will OOM.

The frontend is static and the API is stateless (apart from an in-memory result
cache), so it scales horizontally with a load balancer if needed.

## ⚙️ Configuration

All settings are read from environment variables (see `.env.example`):

| Variable | Default | Purpose |
| --- | --- | --- |
| `GROQ_API_KEY` | *(empty)* | Enables LLM interpretation |
| `GROQ_MODEL` | `llama-3.3-70b-versatile` | Groq model |
| `CLIP_MODEL` | `openai/clip-vit-base-patch32` | Recognition model |
| `MAX_INPUT_DIM` | `384` | Downscale limit for inputs |
| `TORCH_THREADS` | `4` | Threads per inference process |
| `WARMUP` | `1` | Pre-load models on startup |
| `PORT` | `8000` | Server port |

## 📁 Project structure

```
app.py               FastAPI server
core/                Analysis engine
  config.py            settings
  features.py          OpenCV visual metrics + palette
  recognizer.py        CLIP subject / theme / emotion / concept recognition
  describer.py         short "what it sees" description
  heatmap.py           ResNet-50 Grad-CAM
  interpreter.py       Groq + varied local fallback
  pipeline.py          orchestration + caching
static/              Frontend (HTML / CSS / JS, no build step)
  samples/             example paintings
requirements.txt
Dockerfile
.env.example
```
