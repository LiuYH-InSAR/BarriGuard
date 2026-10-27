# Preserve BarriGuard's modern FFmpeg decoder for these HEVC cameras.
FROM debian:bookworm-slim AS ffmpeg
ARG FFMPEG_URL=https://github.com/BtbN/FFmpeg-Builds/releases/download/latest/ffmpeg-master-latest-linux64-gpl.tar.xz
RUN apt-get update \
 && apt-get install -y --no-install-recommends ca-certificates curl xz-utils \
 && rm -rf /var/lib/apt/lists/* \
 && curl -fsSL "$FFMPEG_URL" -o /tmp/ff.tar.xz \
 && tar -xf /tmp/ff.tar.xz -C /tmp \
 && cp /tmp/ffmpeg-*/bin/ffmpeg /opt/ffmpeg \
 && /opt/ffmpeg -version | head -1

FROM python:3.12-slim-bookworm
ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu
ARG TORCH_VERSION=2.14.0
ARG TORCHVISION_VERSION=0.29.0
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/app/deploy/app \
    BARRIGUARD_CONFIG=/app/deploy/config.yaml \
    YOLO_CONFIG_DIR=/app/output/model_settings \
    MPLCONFIGDIR=/app/output/model_settings/matplotlib \
    TZ=Asia/Shanghai
WORKDIR /app

# Keep the validated opencv-python build, including its shared-library needs.
RUN apt-get update \
 && apt-get install -y --no-install-recommends libglib2.0-0 libgl1 libgomp1 \
 && rm -rf /var/lib/apt/lists/*
COPY deploy/requirements-server.txt /tmp/requirements-server.txt
RUN pip install --index-url "$TORCH_INDEX_URL" \
      "torch==$TORCH_VERSION" "torchvision==$TORCHVISION_VERSION" \
 && pip install -r /tmp/requirements-server.txt
COPY --from=ffmpeg /opt/ffmpeg /usr/local/bin/ffmpeg

# Build context is the repository root. Preserve deploy/app so PROJECT=/app.
# .dockerignore excludes all archives, training assets and generated results.
COPY . /app/
RUN mkdir -p /app/data /app/output/water_yolo /app/output/model_settings
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/cameras', timeout=4)"
CMD ["python", "-B", "-m", "server"]
