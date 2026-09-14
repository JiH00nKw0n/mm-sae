# Build target cpu for local smoke tests; gpu for the full NVIDIA server run.
FROM python:3.11.11-slim-bookworm AS cpu
ENV PYTHONUNBUFFERED=1 HF_HOME=/workspace/cache/huggingface MPLCONFIGDIR=/workspace/cache/matplotlib TOKENIZERS_PARALLELISM=false
WORKDIR /workspace
RUN pip install --no-cache-dir torch==2.6.0 --index-url https://download.pytorch.org/whl/cpu
COPY requirements-runtime.lock ./
RUN pip install --no-cache-dir --timeout 120 --retries 5 -r requirements-runtime.lock
COPY pyproject.toml ./
COPY src ./src
COPY experiments ./experiments
COPY configs ./configs
RUN pip install --no-cache-dir --no-deps . && pip check
ENTRYPOINT ["python", "-m", "mm_sae"]
CMD ["--config", "/workspace/configs/smoke-real.yaml"]

FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime AS gpu
ENV PYTHONUNBUFFERED=1 HF_HOME=/workspace/cache/huggingface MPLCONFIGDIR=/workspace/cache/matplotlib TOKENIZERS_PARALLELISM=false
WORKDIR /workspace
COPY requirements-runtime.lock ./
RUN pip install --no-cache-dir --timeout 120 --retries 5 -r requirements-runtime.lock
COPY pyproject.toml ./
COPY src ./src
COPY experiments ./experiments
COPY configs ./configs
RUN pip install --no-cache-dir --no-deps . && pip check
ENTRYPOINT ["python", "-m", "mm_sae"]
CMD ["--config", "/workspace/configs/coco.yaml"]
