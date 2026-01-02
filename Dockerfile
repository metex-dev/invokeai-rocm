### Frontend Build
# ------------------------------------------
FROM docker.io/node:22-slim AS web-builder
ENV PNPM_HOME="/pnpm"
ENV PATH="$PNPM_HOME:$PATH"
RUN corepack use pnpm@8.x
RUN corepack enable

# Install necessary dependencies
RUN apt update && apt install -y --no-install-recommends ca-certificates git

WORKDIR /build
RUN git clone https://github.com/invoke-ai/InvokeAI.git /tmp/invokeai
RUN cp -r /tmp/invokeai/invokeai/frontend/web/* ./
RUN rm -rf /tmp/invokeai

RUN --mount=type=cache,target=/pnpm/store \
    pnpm install --frozen-lockfile
RUN npx vite build



### App Configuration
# ------------------------------------------
# Use the official ROCm PyTorch image. This image includes a pre-built,
# ROCm-enabled version of PyTorch, which is critical for stability.
FROM rocm/pytorch:rocm7.1.1_ubuntu24.04_py3.12_pytorch_release_2.9.1 AS app

# --- ROCm Runtime Architecture ---
# Ensures the runtime knows exactly which hardware to target, 
# preventing "Unsupported GPU" errors on newer APUs/GPUs.
ENV HSA_OVERRIDE_GFX_VERSION=11.5.1
ENV PYTORCH_ROCM_ARCH=gfx1151

# --- Device Visibility ---
# Forces PyTorch to ignore CUDA paths and use HIP exclusively.
ENV HIP_VISIBLE_DEVICES=0
ENV CUDA_VISIBLE_DEVICES=""

# --- Memory Management ---
ENV PYTORCH_ALLOC_CONF=max_split_size_mb:256
ENV PYTORCH_HIP_ALLOC_CONF=max_split_size_mb:256
ENV PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:256

# --- Speed & Kernel Optimizations ---
# Enables modern matrix multiplication libraries (significant boost for RDNA3)
ENV ROCBLAS_USE_HIPBLASLT=1
# Enables PyTorch to auto-tune kernels for our specific chip (new in ROCm 7.x)
# ! Unfortunately it causes infinite hangs on gfx1151 VAE decoding. So it is disabled for now.
ENV PYTORCH_TUNABLEOP_ENABLED=0
# Fixes ROCm 7 VAE speed issues (MIOpen has performance bugs on ROCm 7)
ENV MIOPEN_FIND_MODE=FAST
# Includes script in pythonstartup that disables cudnn to effectively disable MIOPEN and fix the performance issues during VAE decoding.
COPY ./scripts/disable_cudnn.py /app/scripts/disable_cudnn.py
ENV PYTHONSTARTUP=/app/fixes/disable_cudnn.py

# --- Stability & Workarounds ---
# Fixes a known issue where ROCm might crash if it detects a slight version
# mismatch between the host driver and the container libraries.
ENV ROCM_SOFT_SKIP_GPU_CHECK=1
# Disables the 'Triton' cache lock which often causes "File exists" errors
# during high-concurrency node loading in ComfyUI.
#ENV TRITON_CACHE_MANAGER=1
# Disable FRAGMENT_ALLOCATOR to prevent "Page Faults" on Unified Memory architectures
ENV HSA_DISABLE_FRAGMENT_ALLOCATOR=1

# --- Python Behavior ---
# Ensures logs appear in real-time in Docker logs
ENV PYTHONUNBUFFERED=1
# Prevents Python from creating annoying __pycache__ directories in your mapped volumes
ENV PYTHONDONTWRITEBYTECODE=1

# --- InvokeAI Configuration ---
# General
ENV INVOKEAI_DIR=/app/invokeai
ENV INVOKEAI_ROOT=${INVOKEAI_DIR}
ENV INVOKEAI_HOST=0.0.0.0
ENV INVOKEAI_PORT=9090
# Directory Paths
ENV INVOKEAI_OUTPUTS_DIR=/app/invokeai/outputs
ENV INVOKEAI_MODELS_DIR=/app/invokeai/models
ENV INVOKEAI_PROFILES_DIR=/app/invokeai/profiles
# Memory Management
ENV INVOKEAI_DEVICE_WORKING_MEM_GB=4
# Optimizations / Fixes
ENV INVOKEAI_FORCE_TILED_DECODE=false
ENV INVOKEAI_TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1
ENV INVOKEAI_ENABLE_PARTIAL_LOADING=false
ENV INVOKEAI_KEEP_RAM_COPY_OF_WEIGHTS=false
ENV INVOKEAI_PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:256



# --- LOCK BASE DEPENDENCIES ---
# Instead of an extra index URL, we freeze the current critical packages into a constraint file.
# This strictly forbids pip from updating them, relying entirely on the local version.
RUN pip freeze | grep -E '^(torch|torchvision|torchaudio|triton)' > /etc/base_constraints.txt
ENV PIP_CONSTRAINT=/etc/base_constraints.txt

# Update system dependencies
RUN apt update && apt install -y --no-install-recommends git curl wget libglib2.0-0 gosu libgl1 libglx-mesa0 build-essential libopencv-dev libstdc++-10-dev
RUN rm -rf /var/lib/apt/lists/*



### InvokeAI Installation
# ------------------------------------------
# Clone InvokeAI repository
RUN mkdir -p ${INVOKEAI_DIR}
RUN git clone https://github.com/invoke-ai/InvokeAI.git ${INVOKEAI_DIR}
WORKDIR /app/invokeai

# Adjusts pyproject.toml to use ROCm 7 compatible versions
RUN sed -i \
    -e 's/"torch~=2.7.0",/"torch>=2.9.0",/' \
    -e 's/"torch==2.7.1+rocm6.3",/"torch>=2.9.1",/' \
    -e 's/"torchvision==0.22.1+rocm6.3",/"torchvision",/' \
    pyproject.toml

# Install dependencies
RUN ulimit -n 30000
RUN pip install -e .

COPY --from=web-builder /build/dist ${INVOKEAI_DIR}/invokeai/frontend/web/dist

### Finish Image Build
# ------------------------------------------
# make ~/.local/bin available on the PATH so scripts like tqdm, torchrun, etc. are found
ENV PATH=/home/appuser/.local/bin:$PATH
# Clean up pip cache to reduce image size
RUN pip cache purge

# Prepare docker-entrypoint.sh
RUN chmod +x ${INVOKEAI_DIR}/docker/docker-entrypoint.sh
# Set User in entrypoint.sh to root
RUN sed -i 's/USER=ubuntu/USER=root/' ${INVOKEAI_DIR}/docker/docker-entrypoint.sh
RUN sed -i 's/USER_ID=${CONTAINER_UID:-1000}/USER_ID=${CONTAINER_UID:-0}/' ${INVOKEAI_DIR}/docker/docker-entrypoint.sh

EXPOSE ${INVOKEAI_PORT}

ENTRYPOINT ["/app/invokeai/docker/docker-entrypoint.sh"]
CMD ["invokeai-web"]
