import base64
import io
import logging
import os
import traceback

import runpod
import torch
from diffusers import FluxPipeline

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("handler")

MODEL_DIR = os.environ.get("MODEL_DIR", "/runpod-volume/flux1-dev")
USE_CPU_OFFLOAD = os.environ.get("USE_CPU_OFFLOAD", "false").lower() == "true"

MAX_DIM = 1536
MIN_DIM = 256
MAX_STEPS = 100

if not os.path.isfile(os.path.join(MODEL_DIR, "model_index.json")):
    raise RuntimeError(
        f"FLUX.1-dev weights not found at MODEL_DIR={MODEL_DIR!r} "
        "(missing model_index.json). Populate the network volume first "
        "with scripts/download_weights.py before deploying this endpoint."
    )

log.info("Loading FluxPipeline from %s ...", MODEL_DIR)
pipe = FluxPipeline.from_pretrained(MODEL_DIR, torch_dtype=torch.bfloat16)
if USE_CPU_OFFLOAD:
    pipe.enable_model_cpu_offload()
else:
    pipe = pipe.to("cuda")
log.info("Model loaded, worker ready.")


def _clamp(value, lo, hi):
    return max(lo, min(hi, value))


def _round_to_multiple(value, multiple=16):
    return int(round(value / multiple)) * multiple


def validate_input(job_input):
    if not isinstance(job_input, dict):
        raise ValueError("input must be a JSON object")

    prompt = job_input.get("prompt")
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("input.prompt is required and must be a non-empty string")

    negative_prompt = job_input.get("negative_prompt") or None

    width = _round_to_multiple(_clamp(int(job_input.get("width", 1024)), MIN_DIM, MAX_DIM))
    height = _round_to_multiple(_clamp(int(job_input.get("height", 1024)), MIN_DIM, MAX_DIM))

    num_inference_steps = int(_clamp(int(job_input.get("num_inference_steps", 28)), 1, MAX_STEPS))
    guidance_scale = float(job_input.get("guidance_scale", 3.5))

    seed = job_input.get("seed")
    if seed is not None:
        seed = int(seed)

    return {
        "prompt": prompt,
        "negative_prompt": negative_prompt,
        "width": width,
        "height": height,
        "num_inference_steps": num_inference_steps,
        "guidance_scale": guidance_scale,
        "seed": seed,
    }


def handler(job):
    try:
        params = validate_input(job.get("input", {}))
    except (ValueError, TypeError) as e:
        return {"error": f"invalid input: {e}"}

    seed = params["seed"]
    if seed is None:
        seed = torch.seed() % (2**32 - 1)
    generator = torch.Generator(device="cuda").manual_seed(seed)

    try:
        result = pipe(
            prompt=params["prompt"],
            negative_prompt=params["negative_prompt"],
            width=params["width"],
            height=params["height"],
            num_inference_steps=params["num_inference_steps"],
            guidance_scale=params["guidance_scale"],
            generator=generator,
        )
        image = result.images[0]
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        return {
            "error": (
                "CUDA out of memory. Try a smaller width/height or fewer "
                "num_inference_steps."
            )
        }
    except Exception as e:
        log.error("Inference failed: %s\n%s", e, traceback.format_exc())
        return {"error": str(e)}

    buf = io.BytesIO()
    image.save(buf, format="PNG")
    image_base64 = base64.b64encode(buf.getvalue()).decode("utf-8")

    return {"image_base64": image_base64, "seed": seed}


runpod.serverless.start({"handler": handler})
