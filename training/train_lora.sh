#!/usr/bin/env bash
set -euo pipefail

# DreamBooth LoRA fine-tuning of FLUX.1-dev on a custom subject/character.
# Run on a RunPod Pod (40GB+ VRAM recommended) with the network volume
# mounted at /workspace. See training/README.md for the full walkthrough.

# Pin a diffusers release tag rather than tracking main, so a training run is
# reproducible. Check https://github.com/huggingface/diffusers/releases for
# the current tag and bump this deliberately -- don't blindly track main.
DIFFUSERS_REF="${DIFFUSERS_REF:-v0.31.0}"
BASE_MODEL_DIR="${BASE_MODEL_DIR:-/workspace/flux1-dev}"
WORK_DIR="${WORK_DIR:-/workspace/diffusers-src}"

INSTANCE_DATA_DIR=""
CLASS_NAME=""
OUTPUT_DIR=""
RANK=16
RESOLUTION=1024
MAX_TRAIN_STEPS=500

usage() {
  cat <<EOF
Usage: $0 --instance-data-dir DIR --class-name NAME [options]

Required:
  --instance-data-dir DIR   Output of prepare_dataset.py
  --class-name NAME         e.g. "person", "dog" -- must match what you used with prepare_dataset.py

Options:
  --output-dir DIR          Default: /workspace/loras/<class-name>
  --rank N                  LoRA rank (default: $RANK)
  --resolution N            Training resolution (default: $RESOLUTION)
  --max-train-steps N       Default: $MAX_TRAIN_STEPS (use a low value like 20 for a smoke test first)
EOF
  exit 1
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --instance-data-dir) INSTANCE_DATA_DIR="$2"; shift 2 ;;
    --class-name) CLASS_NAME="$2"; shift 2 ;;
    --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
    --rank) RANK="$2"; shift 2 ;;
    --resolution) RESOLUTION="$2"; shift 2 ;;
    --max-train-steps) MAX_TRAIN_STEPS="$2"; shift 2 ;;
    -h|--help) usage ;;
    *) echo "Unknown argument: $1" >&2; usage ;;
  esac
done

if [[ -z "$INSTANCE_DATA_DIR" || -z "$CLASS_NAME" ]]; then
  usage
fi

if [[ -z "$OUTPUT_DIR" ]]; then
  OUTPUT_DIR="/workspace/loras/${CLASS_NAME}"
fi

INSTANCE_PROMPT="a photo of sks ${CLASS_NAME}"

if [[ ! -f "${BASE_MODEL_DIR}/model_index.json" ]]; then
  echo "ERROR: FLUX.1-dev weights not found at BASE_MODEL_DIR=${BASE_MODEL_DIR}" >&2
  echo "Run scripts/download_weights.py first (see top-level README.md)." >&2
  exit 1
fi

if [[ ! -d "${WORK_DIR}" ]]; then
  echo "Cloning diffusers @ ${DIFFUSERS_REF} into ${WORK_DIR} ..."
  git clone --depth 1 --branch "${DIFFUSERS_REF}" https://github.com/huggingface/diffusers.git "${WORK_DIR}"
else
  echo "Reusing existing checkout at ${WORK_DIR} (delete it to re-clone a different DIFFUSERS_REF)."
fi

TRAIN_SCRIPT="${WORK_DIR}/examples/dreambooth/train_dreambooth_lora_flux.py"
if [[ ! -f "$TRAIN_SCRIPT" ]]; then
  echo "ERROR: ${TRAIN_SCRIPT} not found -- DIFFUSERS_REF=${DIFFUSERS_REF} may not have this example script." >&2
  exit 1
fi

mkdir -p "$OUTPUT_DIR"

echo "Instance prompt: \"${INSTANCE_PROMPT}\""
echo "Output dir: ${OUTPUT_DIR}"
echo "Max train steps: ${MAX_TRAIN_STEPS}"

accelerate launch "$TRAIN_SCRIPT" \
  --pretrained_model_name_or_path="${BASE_MODEL_DIR}" \
  --instance_data_dir="${INSTANCE_DATA_DIR}" \
  --instance_prompt="${INSTANCE_PROMPT}" \
  --output_dir="${OUTPUT_DIR}" \
  --mixed_precision="bf16" \
  --resolution="${RESOLUTION}" \
  --train_batch_size=1 \
  --gradient_accumulation_steps=4 \
  --gradient_checkpointing \
  --use_8bit_adam \
  --cache_latents \
  --rank="${RANK}" \
  --guidance_scale=3.5 \
  --optimizer="AdamW" \
  --learning_rate=1e-4 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --max_train_steps="${MAX_TRAIN_STEPS}" \
  --seed=0
