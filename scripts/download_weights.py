"""
One-off script to populate a RunPod network volume with FLUX.1-dev weights.

Run this on a RunPod Pod (not the serverless endpoint) with the network
volume attached. Pods mount network volumes at /workspace by default, so
the default DEST below assumes that layout.

Usage (on the Pod):
    export HF_TOKEN=hf_...
    python download_weights.py [--dest /workspace/flux1-dev]

Requires accepting the FLUX.1-dev license on huggingface.co with the
account tied to HF_TOKEN before this will succeed.
"""

import argparse
import os
import sys

from huggingface_hub import snapshot_download

REPO_ID = "black-forest-labs/FLUX.1-dev"
DEFAULT_DEST = "/workspace/flux1-dev"

# The repo also ships redundant single-file checkpoints (flux1-dev.safetensors,
# ae.safetensors) for non-diffusers workflows like ComfyUI. FluxPipeline.from_pretrained
# only reads the component subfolders + model_index.json, so skip these to avoid
# downloading ~24GB of weights we'd never use.
IGNORE_PATTERNS = ["flux1-dev.safetensors", "ae.safetensors"]


def already_populated(dest: str) -> bool:
    return os.path.isfile(os.path.join(dest, "model_index.json"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dest", default=DEFAULT_DEST, help="Destination directory on the volume")
    args = parser.parse_args()

    token = os.environ.get("HF_TOKEN")
    if not token:
        print("ERROR: HF_TOKEN environment variable is not set.", file=sys.stderr)
        sys.exit(1)

    if already_populated(args.dest):
        print(f"{args.dest} already contains model_index.json, skipping download.")
        return

    print(f"Downloading {REPO_ID} into {args.dest} ...")
    snapshot_download(
        repo_id=REPO_ID,
        local_dir=args.dest,
        token=token,
        ignore_patterns=IGNORE_PATTERNS,
    )

    total_size = 0
    file_count = 0
    for root, _dirs, files in os.walk(args.dest):
        for f in files:
            file_count += 1
            total_size += os.path.getsize(os.path.join(root, f))

    print(f"Done. {file_count} files, {total_size / (1024**3):.2f} GiB at {args.dest}")


if __name__ == "__main__":
    main()
