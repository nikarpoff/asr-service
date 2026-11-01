# BUILD
FROM python:3.12-slim-bookworm AS builder

# Install uv from the uv image; the version matches the one that wrote uv.lock
COPY --from=ghcr.io/astral-sh/uv:0.12.19 /uv /uvx /bin/

ENV UV_PYTHON_DOWNLOADS=never

# Go to app directory
WORKDIR /app

# Extras to install: the gpu profile and the backends. EXTRAS="" leaves the
# fake backend only
ARG EXTRAS="gpu"

# Install uv dependencies. One lock serves both profiles: the conflicting cpu and
# gpu extras are resolved side by side in it, and --extra picks which one lands.
COPY pyproject.toml uv.lock .python-version ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv venv /app/.venv --python python3.12 && uv sync --frozen --no-install-project --no-dev --prerelease=allow \
      $(for extra in $EXTRAS; do printf -- '--extra %s ' "$extra"; done)

# Copy code
COPY . /app

# RUNTIME
FROM nvidia/cuda:13.2.1-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PATH="/app/.venv/bin:$PATH"

# Install Python 3.12 and runtime-dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    software-properties-common \
    gnupg \
    && add-apt-repository ppa:deadsnakes/ppa -y \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
       python3.12 \
       libssl3 \
       libffi8 \
       libbz2-1.0 \
       zlib1g \
       libreadline8 \
       libsqlite3-0 \
       liblzma5 \
       libncursesw6 \
       ca-certificates \
       curl \
       libgomp1 \
       cuda-compat-13-2 \
    && ln -sf /usr/bin/python3.12 /usr/local/bin/python3.12 \
    && ln -sf /usr/bin/python3.12 /usr/local/bin/python \
    && ln -sf /app/.venv/bin/pip /usr/local/bin/pip \
    && apt-get purge -y --auto-remove software-properties-common gnupg \
    && rm -rf /var/lib/apt/lists/*

# Redefine path to compat-libs to load forward-compatible versions of libcuda.so and etc.
ENV LD_LIBRARY_PATH="/usr/local/cuda/compat:${LD_LIBRARY_PATH}"

WORKDIR /app

# Remove cache
COPY --from=builder /app /app
RUN set -eux; \
    find /app/.venv -type d \( -name "__pycache__" -o -name "tests" -o -name "test" \) -exec rm -rf {} +; \
    find /app/.venv -type f -name "*.py[co]" -delete; \
    rm -rf /app/.venv/lib/python3.12/site-packages/pip* /app/.venv/lib/python3.12/site-packages/wheel*

# The commit goes to /v1/info; it is the last layer, so a new commit rebuilds nothing else
ARG GIT_SHA=unknown
ENV GIT_SHA=${GIT_SHA}

EXPOSE 8000
ENTRYPOINT ["python", "-m", "uvicorn", "app.main:app"]
CMD ["--host", "0.0.0.0", "--port", "8000", "--loop", "uvloop", "--http", "httptools"]
