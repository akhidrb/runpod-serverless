"""
Standalone verification: loads the base FluxPipeline plus a trained LoRA and
generates one test image, so you can eyeball whether training worked.
Independent of endpoint/handler.py -- doesn't touch the serverless code.

Usage:
    python verify_lora.py --lora-dir /workspace/loras/person \
        --prompt "a photo of sks person wearing a red jacket, studio lighting" \
        --out test.png
"""

import argparse
import os

import torch
from diffusers import FluxPipeline


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base-model-dir",
        default=os.environ.get("BASE_MODEL_DIR", "/workspace/flux1-dev"),
    )
    parser.add_argument("--lora-dir", required=True, help="Output dir from train_lora.sh")
    parser.add_argument(
        "--prompt", required=True, help="Should include the trained instance token, e.g. 'sks person'"
    )
    parser.add_argument("--out", default="verify_lora_output.png")
    parser.add_argument("--num-inference-steps", type=int, default=28)
    parser.add_argument("--guidance-scale", type=float, default=3.5)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    print(f"Loading base pipeline from {args.base_model_dir} ...")
    pipe = FluxPipeline.from_pretrained(args.base_model_dir, torch_dtype=torch.bfloat16).to("cuda")

    print(f"Loading LoRA weights from {args.lora_dir} ...")
    pipe.load_lora_weights(args.lora_dir)

    generator = None
    if args.seed is not None:
        generator = torch.Generator(device="cuda").manual_seed(args.seed)

    image = pipe(
        prompt=args.prompt,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
        generator=generator,
    ).images[0]

    image.save(args.out)
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
