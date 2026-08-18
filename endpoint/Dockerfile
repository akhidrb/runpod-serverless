# Pick a current runpod/pytorch tag from https://hub.docker.com/r/runpod/pytorch/tags
# with Python 3.10+ and CUDA 12.1+ (torch is preinstalled in these images).
# Fallback if you'd rather not depend on RunPod's image: start from
# nvidia/cuda:12.1.1-devel-ubuntu22.04 and `pip install torch` with a matching
# CUDA wheel yourself.
FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY handler.py .

CMD ["python3", "-u", "handler.py"]
