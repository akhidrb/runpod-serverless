"""
Validates and normalizes a folder of raw subject photos into a training-ready
dataset for training/train_lora.sh (DreamBooth LoRA fine-tuning of FLUX.1-dev).

Usage:
    python prepare_dataset.py --input-dir raw_photos/ --output-dir data/sks_person \
        --class-name person

Produces a flat folder of resize+center-cropped PNGs (exactly what
--instance_data_dir expects) plus a manifest.json log of what happened to
each source file. No captioning is done: DreamBooth uses a single fixed
instance prompt ("a photo of sks <class-name>") applied to every image,
which train_lora.sh derives from the same --class-name value.
"""

import argparse
import hashlib
import json
import os
import sys

from PIL import Image, ImageOps

SUPPORTED_EXTS = {".jpg", ".jpeg", ".png", ".webp"}

MIN_IMAGES_HARD_FAIL = 5
MIN_IMAGES_WARN = 15
MAX_IMAGES_WARN = 50

INSTANCE_TOKEN = "sks"


def ahash(image: Image.Image, hash_size: int = 8) -> int:
    """Simple average hash for near-duplicate detection (no extra deps)."""
    small = image.convert("L").resize((hash_size, hash_size), Image.LANCZOS)
    pixels = list(small.getdata())
    avg = sum(pixels) / len(pixels)
    bits = 0
    for p in pixels:
        bits = (bits << 1) | (1 if p >= avg else 0)
    return bits


def hamming(a: int, b: int) -> int:
    return bin(a ^ b).count("1")


def resize_and_center_crop(image: Image.Image, target: int) -> Image.Image:
    w, h = image.size
    scale = target / min(w, h)
    new_w, new_h = round(w * scale), round(h * scale)
    image = image.resize((new_w, new_h), Image.LANCZOS)
    left = (new_w - target) // 2
    top = (new_h - target) // 2
    return image.crop((left, top, left + target, top + target))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", required=True, help="Folder of raw subject photos")
    parser.add_argument("--output-dir", required=True, help="Where normalized training images go")
    parser.add_argument("--class-name", required=True, help='e.g. "person", "dog", a character name')
    parser.add_argument("--resolution", type=int, default=1024, help="Target square resolution")
    parser.add_argument(
        "--dup-threshold",
        type=int,
        default=4,
        help="Hamming distance below which two images are flagged as near-duplicates",
    )
    args = parser.parse_args()

    if not os.path.isdir(args.input_dir):
        print(f"ERROR: input dir {args.input_dir!r} does not exist.", file=sys.stderr)
        sys.exit(1)

    os.makedirs(args.output_dir, exist_ok=True)

    candidates = sorted(
        f for f in os.listdir(args.input_dir)
        if os.path.splitext(f)[1].lower() in SUPPORTED_EXTS
    )

    manifest = []
    seen_md5 = {}
    seen_ahash = []
    processed = 0

    for fname in candidates:
        src_path = os.path.join(args.input_dir, fname)
        entry = {"source": fname, "status": "ok", "warnings": []}

        try:
            raw_bytes = open(src_path, "rb").read()
            md5 = hashlib.md5(raw_bytes).hexdigest()
            if md5 in seen_md5:
                entry["status"] = "skipped"
                entry["warnings"].append(f"exact duplicate of {seen_md5[md5]}")
                manifest.append(entry)
                continue
            seen_md5[md5] = fname

            image = Image.open(src_path)
            image = ImageOps.exif_transpose(image)  # strip EXIF rotation
            image = image.convert("RGB")
        except Exception as e:
            entry["status"] = "skipped"
            entry["warnings"].append(f"failed to open/decode: {e}")
            manifest.append(entry)
            continue

        w, h = image.size
        aspect = max(w, h) / min(w, h)
        if aspect > 1.6:
            entry["warnings"].append(
                f"unusual aspect ratio ({w}x{h}) — center-crop may cut off part of the subject"
            )
        if min(w, h) < args.resolution:
            entry["warnings"].append(
                f"source resolution ({w}x{h}) is below target {args.resolution}; will be upsampled"
            )

        h_val = ahash(image)
        for other_fname, other_hash in seen_ahash:
            if hamming(h_val, other_hash) <= args.dup_threshold:
                entry["warnings"].append(f"looks like a near-duplicate of {other_fname}")
                break
        seen_ahash.append((fname, h_val))

        cropped = resize_and_center_crop(image, args.resolution)
        out_name = f"{processed:03d}.png"
        cropped.save(os.path.join(args.output_dir, out_name), format="PNG")
        entry["output"] = out_name
        processed += 1
        manifest.append(entry)

    manifest_path = os.path.join(args.output_dir, "manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    warned = sum(1 for e in manifest if e.get("warnings") and e["status"] == "ok")
    skipped = sum(1 for e in manifest if e["status"] == "skipped")

    print(f"Input images found: {len(candidates)}")
    print(f"Processed: {processed}  Skipped: {skipped}  With warnings: {warned}")
    print(f"Output dir: {args.output_dir}")
    print(f"Manifest: {manifest_path}")

    if processed < MIN_IMAGES_HARD_FAIL:
        print(
            f"ERROR: only {processed} usable images (< {MIN_IMAGES_HARD_FAIL} minimum). "
            "Add more photos before training.",
            file=sys.stderr,
        )
        sys.exit(1)
    if processed < MIN_IMAGES_WARN:
        print(
            f"WARNING: {processed} images is on the low side; 15-30 varied photos "
            "(angles, lighting, backgrounds) generally work better than a few."
        )
    if processed > MAX_IMAGES_WARN:
        print(
            f"WARNING: {processed} images is more than usually needed; expect longer "
            "training with diminishing returns."
        )
    if warned:
        print(f"Review the {warned} warning(s) in {manifest_path} before training.")

    print(
        f'\nUse instance prompt "a photo of {INSTANCE_TOKEN} {args.class_name}" '
        f"and --class-name {args.class_name} with train_lora.sh."
    )


if __name__ == "__main__":
    main()
