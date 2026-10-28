# ☁️ Deploying Chroma

Chroma is a single Dockerised FastAPI app with a static frontend, so it runs
anywhere that hosts Docker. It needs **~1 GB of RAM** (it loads CLIP +
ResNet-50 in memory) and will download ~700 MB of model weights once (baked in
at build time by the Dockerfile).

---

## Option A — Render (simple, from ~$7–25/mo)

Render's free and Starter tiers are **512 MB RAM**, which is too small for the
models — use the **Standard** plan (2 GB, $25/mo), or expect out-of-memory
crashes on the smaller tiers.

1. Push this repo to GitHub (already done).
2. Go to <https://render.com> → **New → Blueprint**.
3. Connect the repo — Render detects `render.yaml` and creates the service.
   - Or manually: **New → Web Service**, pick the repo, **Docker** runtime,
     **Standard** plan, health-check path `/health`.
4. (Optional) set `GROQ_API_KEY` in **Environment** to enable LLM
   interpretation. Without it the local interpreter is used.
5. Click **Deploy**. The first build takes a few minutes (it pre-downloads the
   models). Your app is then live at `https://<name>.onrender.com`.

> Free-tier note: Render's free web services sleep after 15 minutes of
> inactivity and wake on the next request (a ~30–60 s cold start).

---

## Option B — Hugging Face Spaces (free, 16 GB RAM — best for ML)

HF's free **CPU Basic** hardware gives 2 vCPU / **16 GB RAM** at no cost, which
is a much better fit for these models.

1. Go to <https://huggingface.co/new-space>.
2. Set **SDK = Docker**, and choose **CPU Basic** hardware (free).
3. The `README.md` already carries the required front-matter (`sdk: docker`,
   `app_port: 8000`), so the Space auto-detects the Dockerfile and port.
4. (Optional) add `GROQ_API_KEY` as a Space **Secret**.
5. Click **Create Space** — HF builds the image and serves it at
   `https://huggingface.co/spaces/<you>/<space-name>`.

> Spaces sleep after 48 h without visitors; the first visitor wakes them.

---

## Option C — Any VPS / self-host

```bash
docker build -t chroma .
docker run -d -p 8000:8000 --name chroma chroma
# put it behind nginx/caddy with TLS, e.g.:
# caddy reverse-proxy --from chroma.example.com --to localhost:8000
```

---

## Option D — Oracle Cloud "Always Free" (free forever, works in India) ★

Oracle's Always Free tier gives a **2-core ARM / 12 GB RAM** VM at no cost,
forever, and has **India** data centres (Mumbai / Hyderabad). 12 GB is plenty
for the models. You only need a card for identity verification — it is **never
charged** on Always Free.

1. Sign up at <https://www.oracle.com/cloud/free/> — pick a **Home region** of
   **India West (Mumbai)** or **India South (Hyderabad)**.
2. **Compute → Instances → Create instance**:
   - Image: **Ubuntu 22.04 / 24.04**
   - Shape: **Ampere A1 (ARM)**, **2 OCPU / 12 GB RAM** (Always Free)
   - Generate or paste an **SSH key** so you can log in.
3. Open a port: **Networking → Virtual Cloud Networks → Security List → Add
   Ingress Rule** for TCP **8000** (source `0.0.0.0/0`).
4. SSH in and run:

```bash
curl -fsSL https://get.docker.com | sh
git clone https://github.com/Aaditya2612/InterpretingAbstractPaintings.git
cd InterpretingAbstractPaintings
docker compose up -d --build   # first build downloads the ~700 MB of models
```

5. Open `http://<your-vm-public-ip>:8000` — it's live.

The Dockerfile already installs ARM-compatible torch, so it runs unchanged on
the Ampere (ARM) shape. For HTTPS + a domain (optional but nice), use a free
Cloudflare Tunnel: `cloudflared tunnel --url http://localhost:8000`.

---

## Option E — Google Cloud Run (free tier, India region)

Cloud Run has a **Mumbai region** (`asia-south1`) and a monthly free allowance
(180k vCPU-seconds + 360k GiB-seconds ≈ ~50 h/mo on a 2 GB instance). It needs
a billing account (credit card) but stays free for light use.

```bash
gcloud auth login
gcloud config set run/region asia-south1
gcloud builds submit --tag gcr.io/$PROJECT/chroma .
gcloud run deploy chroma --image gcr.io/$PROJECT/chroma --memory 2Gi --cpu 1 \
  --port 8000 --allow-unauthenticated
```

---

## Verify it's healthy

Hit `GET /health` — you should see:

```json
{"status": "ok", "models": {"clip": true, "resnet50": true, "groq_configured": false}}
```

`groq_configured` turns `true` once `GROQ_API_KEY` is set.
