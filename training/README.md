# DreamBooth LoRA Fine-Tuning for FLUX.1-dev

Fine-tune FLUX.1-dev on a custom subject/character (a specific person, pet,
or object) using your own photos, producing a LoRA adapter you can later
apply on top of the base model. This is a **standalone training flow** —
it runs on a RunPod Pod, not the serverless endpoint, and isn't wired into
`endpoint/handler.py`. Combining the two is a future step.

## Before you start

- **VRAM is tighter than inference.** A published benchmark of LoRA training
  even at minimal settings (rank 4, bf16, gradient checkpointing, 8-bit Adam)
  hit ~26GB peak on a 24GB card — it didn't fit. Budget **40–48GB+ VRAM**
  (A100 40/80GB, L40S, RTX A6000) for the Pod, same tier as inference but
  with less margin.
- **Photo variety matters more than count.** 15–30 photos with different
  angles, lighting, expressions, and backgrounds beats more photos that all
  look alike — the model will overfit to whatever's constant across your set
  (a pose, a background, an outfit) if you don't vary it.
- You need the base FLUX.1-dev weights already on the network volume — see
  the top-level `README.md`, section 1, if you haven't run
  `scripts/download_weights.py` yet.

## 1. Prepare your dataset

Gather 15–30 photos of the subject and run:

```bash
pip install -r training/requirements.txt
python training/prepare_dataset.py \
  --input-dir /path/to/raw_photos \
  --output-dir /workspace/data/sks_person \
  --class-name person
```

This validates the photos (min 5, hard-fails below that), strips EXIF
rotation, flags unusual aspect ratios / near-duplicates / low-resolution
images, and resize+center-crops everything to 1024x1024 PNGs. Read the
printed summary and `manifest.json` in the output dir, and fix or remove
any flagged photos before training — a few minutes here saves a wasted
training run.

`--class-name` should be a simple noun for the subject (`person`, `dog`,
a character name). It drives the instance prompt used everywhere else:
`"a photo of sks <class-name>"` — `sks` is the classic DreamBooth filler
token, chosen because it's rare enough not to collide with concepts FLUX
already knows.

## 2. Smoke test before a full run

On the Pod (GPU attached, network volume mounted at `/workspace`):

```bash
bash training/train_lora.sh \
  --instance-data-dir /workspace/data/sks_person \
  --class-name person \
  --max-train-steps 20
```

This clones a pinned `diffusers` tag (see `DIFFUSERS_REF` in
`train_lora.sh`), then runs the official
`examples/dreambooth/train_dreambooth_lora_flux.py` script via
`accelerate launch` with recommended defaults (bf16, gradient checkpointing,
8-bit Adam, cached latents, rank 16, resolution 1024). Confirm it gets
through 20 steps without CUDA OOM or missing-dependency errors, and note
the per-step time to estimate how long a full run will take.

## 3. Full training run

```bash
bash training/train_lora.sh \
  --instance-data-dir /workspace/data/sks_person \
  --class-name person \
  --max-train-steps 500
```

Start around 500 steps (the diffusers example's own default) and adjust
based on dataset size and how the result looks. Output LoRA weights land
in `/workspace/loras/person/` by default — on the same network volume as
the base weights, so they persist after the Pod is terminated. Pass
`--output-dir` to change this.

## 4. Verify the result

```bash
python training/verify_lora.py \
  --lora-dir /workspace/loras/person \
  --prompt "a photo of sks person wearing a red jacket, studio lighting" \
  --out test.png
```

Generate a few images across different prompts and check for:
- **Recognizable likeness** of the subject.
- **Overfitting**: if every output looks like one specific training photo
  regardless of prompt, or the model ignores the rest of the prompt, you
  likely trained too long or on too few/too-similar images. Try a lower
  `--max-train-steps` or a more varied dataset.

## Notes

- `DIFFUSERS_REF` in `train_lora.sh` pins a specific diffusers release tag
  rather than tracking `main`, so results are reproducible. Bump it
  deliberately — check https://github.com/huggingface/diffusers/releases
  and re-read `examples/dreambooth/README_flux.md` at the new tag before
  changing it, since upstream args can change between releases.
- Prior preservation (`--with_prior_preservation` / `--class_prompt` /
  `--class_data_dir` in the upstream script) isn't enabled by default here.
  It helps prevent the model from "forgetting" the general class concept
  (e.g. still being able to generate generic people, not just your subject)
  at the cost of extra setup and training time — worth adding if you notice
  that regression.
- Training time is on the order of 20–40 minutes for a few hundred steps,
  depending on GPU and settings — extrapolate from your smoke test's
  per-step time before committing to a long unattended run.
