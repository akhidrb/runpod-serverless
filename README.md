# FLUX.1-dev on RunPod Serverless

Deploys [black-forest-labs/FLUX.1-dev](https://huggingface.co/black-forest-labs/FLUX.1-dev)
(text-to-image) as a RunPod Serverless GPU endpoint, with a Gradio app for
testing it. Weights are served from a RunPod network volume rather than
baked into the Docker image, so the image stays small and you don't
re-download ~24GB on every build.

## Repo layout

- `endpoint/` — the serverless worker: `handler.py`, `Dockerfile`, `requirements.txt`.
  This is the Docker build context; it never contains model weights.
- `scripts/download_weights.py` — one-off script to populate the network
  volume with FLUX.1-dev weights. Run on a temporary RunPod Pod, not the
  serverless endpoint.
- `test/test_input.json` — sample payload for local handler testing.
- `gradio_app/` — a small web app that calls your deployed endpoint's REST
  API and renders the generated image.

## 0. Prerequisites (do these first — both are external dependencies with unknown turnaround)

1. Request access to the gated model at
   https://huggingface.co/black-forest-labs/FLUX.1-dev and accept its
   license. Approval time varies.
2. Create a RunPod account, then email your account email to
   `hailong.yang@runpod.io` (per the case study) to receive free credits.

## 1. Create and populate the network volume

1. In the RunPod console, create a **Network Volume** large enough for the
   weights (40GB+ recommended). Note which **datacenter** it's created in —
   a volume is pinned to one datacenter, and your serverless endpoint will
   later be constrained to that same datacenter.
2. Launch a temporary **Pod** in that same datacenter with the volume
   attached (Pods mount volumes at `/workspace` by default).
3. On the Pod:
   ```bash
   pip install huggingface_hub
   export HF_TOKEN=hf_...          # token for the account with FLUX.1-dev access
   python scripts/download_weights.py --dest /workspace/flux1-dev
   ```
4. Verify: the script prints file count + total size, or you can
   `ls /workspace/flux1-dev` and confirm `model_index.json` is present.
5. (Optional but recommended) While the Pod is up and has a GPU, do a local
   handler smoke test before ever building Docker — see step 3.
6. Terminate the Pod once done (you're billed per-minute for it; the volume
   persists independently).

## 2. Build and push the Docker image

```bash
cd endpoint
docker build -t <your-dockerhub-user>/flux-handler:latest .
docker push <your-dockerhub-user>/flux-handler:latest
```

The `Dockerfile` pins a `runpod/pytorch` base tag — check
https://hub.docker.com/r/runpod/pytorch/tags for the current tag with
Python 3.10+/CUDA 12.1+ and update it if the pinned one is no longer
available.

No `HF_TOKEN` or weights go into this image — only needed by step 1's Pod.

## 3. (Optional) Local handler test before deploying

On the Pod from step 1 (has GPU + volume mounted at `/workspace`):

```bash
export MODEL_DIR=/workspace/flux1-dev
pip install -r endpoint/requirements.txt
cd endpoint
python handler.py --test_input "$(cat ../test/test_input.json)"
```

This exercises input validation, model loading, and output encoding
without needing a deployed endpoint yet.

## 4. Create the serverless endpoint

In the RunPod console, create a new Serverless Endpoint:

- **Container image**: `<your-dockerhub-user>/flux-handler:latest`
- **GPU**: 40–48GB tier (A100 40/80GB or L40S) recommended — FLUX.1-dev's
  weights alone are ~24GB, so a 24GB card is tight. If you must use a 24GB
  card, set env var `USE_CPU_OFFLOAD=true` on the endpoint (trades latency
  for VRAM headroom via `enable_model_cpu_offload()`).
- **Workers**: min 0 (scale-to-zero), max 1–2.
- **Idle timeout**: short (e.g. 5–60s) — fine given min-workers=0.
- **Network volume**: attach the volume from step 1 under
  Advanced → Network Volumes. It mounts at `/runpod-volume` on serverless
  workers (note: different from the Pod's `/workspace` mount used during
  population), which is `handler.py`'s default `MODEL_DIR`.
- Endpoint's datacenter must match the volume's datacenter.

Smoke-test it from the console's built-in request tester, or:

```bash
curl -X POST https://api.runpod.ai/v2/<endpoint_id>/run \
  -H "Authorization: Bearer $RUNPOD_API_KEY" \
  -H "Content-Type: application/json" \
  -d @test/test_input.json
# then poll:
curl https://api.runpod.ai/v2/<endpoint_id>/status/<job_id> \
  -H "Authorization: Bearer $RUNPOD_API_KEY"
```

A `COMPLETED` status with `output.image_base64` confirms the endpoint works
end to end before involving the Gradio app.

## 5. Run the Gradio test app

```bash
cd gradio_app
pip install -r requirements.txt
cp ../.env.example ../.env   # fill in RUNPOD_API_KEY and RUNPOD_ENDPOINT_ID
python app.py
```

Open the printed local URL, enter a prompt, click Generate. The status box
shows the job id and polling state — cold starts (min-workers=0) can take
30s–2min+ for the first request after idle, so do a warm-up request before
recording your demo.

## Notes / known limitations

- FLUX.1-dev is guidance-distilled: `negative_prompt` and `guidance_scale`
  behave differently than in SD/SDXL and have limited effect.
- Output is returned as base64-encoded PNG in the job response. For larger
  images or production use, uploading to object storage and returning a URL
  would scale better — out of scope for this exercise.
- Image build/push time is non-trivial (diffusers/transformers/torch layers
  are large) even without baked-in weights — budget accordingly.
