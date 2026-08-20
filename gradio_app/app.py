import base64
import io
import os
import time

import gradio as gr
import requests
from dotenv import load_dotenv
from PIL import Image

load_dotenv()

API_KEY = os.environ.get("RUNPOD_API_KEY")
ENDPOINT_ID = os.environ.get("RUNPOD_ENDPOINT_ID")

if not API_KEY or not ENDPOINT_ID:
    raise RuntimeError(
        "RUNPOD_API_KEY and RUNPOD_ENDPOINT_ID must be set (see .env.example). "
        "Copy .env.example to .env and fill them in before launching this app."
    )

BASE_URL = f"https://api.runpod.ai/v2/{ENDPOINT_ID}"
HEADERS = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}

POLL_INTERVAL_SECONDS = 2
OVERALL_TIMEOUT_SECONDS = 300


def generate(prompt, width, height, steps, guidance_scale, seed):
    if not prompt or not prompt.strip():
        return None, "Error: prompt is required."

    payload = {
        "input": {
            "prompt": prompt,
            "width": int(width),
            "height": int(height),
            "num_inference_steps": int(steps),
            "guidance_scale": float(guidance_scale),
        }
    }
    if seed is not None and str(seed).strip() != "":
        payload["input"]["seed"] = int(seed)

    try:
        resp = requests.post(f"{BASE_URL}/run", headers=HEADERS, json=payload, timeout=30)
    except requests.RequestException as e:
        return None, f"Error contacting RunPod API: {e}"

    if resp.status_code == 401:
        return None, "Error: 401 Unauthorized — check RUNPOD_API_KEY."
    if resp.status_code == 404:
        return None, "Error: 404 Not Found — check RUNPOD_ENDPOINT_ID."
    if resp.status_code >= 400:
        return None, f"Error: RunPod API returned {resp.status_code}: {resp.text}"

    job = resp.json()
    job_id = job.get("id")
    if not job_id:
        return None, f"Error: no job id in response: {job}"

    status = job.get("status", "IN_QUEUE")
    start = time.time()

    while status not in ("COMPLETED", "FAILED", "CANCELLED", "TIMED_OUT"):
        if time.time() - start > OVERALL_TIMEOUT_SECONDS:
            return None, f"Timed out waiting for job {job_id} (last status: {status})."
        time.sleep(POLL_INTERVAL_SECONDS)
        try:
            poll_resp = requests.get(f"{BASE_URL}/status/{job_id}", headers=HEADERS, timeout=30)
        except requests.RequestException as e:
            return None, f"Error polling job {job_id}: {e}"
        poll_resp.raise_for_status()
        job = poll_resp.json()
        status = job.get("status", status)

    if status != "COMPLETED":
        error = (job.get("output") or {}).get("error") or job.get("error") or status
        return None, f"Job {job_id} did not complete: {error}"

    output = job.get("output") or {}
    if "error" in output:
        return None, f"Handler returned an error: {output['error']}"

    image_b64 = output.get("image_base64")
    if not image_b64:
        return None, f"Error: no image_base64 in output: {output}"

    image = Image.open(io.BytesIO(base64.b64decode(image_b64)))
    used_seed = output.get("seed")
    return image, f"Job {job_id} completed. seed={used_seed}"


with gr.Blocks(title="FLUX.1-dev on RunPod Serverless") as demo:
    gr.Markdown("# FLUX.1-dev on RunPod Serverless\nText prompt in, image out — calls your deployed RunPod endpoint.")
    with gr.Row():
        with gr.Column():
            prompt = gr.Textbox(label="Prompt", lines=3, placeholder="a photo of an astronaut riding a horse")
            with gr.Row():
                width = gr.Slider(512, 1536, value=1024, step=64, label="Width")
                height = gr.Slider(512, 1536, value=1024, step=64, label="Height")
            with gr.Row():
                steps = gr.Slider(1, 100, value=28, step=1, label="Steps")
                guidance_scale = gr.Slider(0, 10, value=3.5, step=0.1, label="Guidance scale")
            seed = gr.Number(label="Seed (optional)", precision=0)
            generate_btn = gr.Button("Generate", variant="primary")
        with gr.Column():
            image_output = gr.Image(label="Result")
            status_output = gr.Textbox(label="Status", interactive=False)

    generate_btn.click(
        fn=generate,
        inputs=[prompt, width, height, steps, guidance_scale, seed],
        outputs=[image_output, status_output],
    )

if __name__ == "__main__":
    demo.launch()
