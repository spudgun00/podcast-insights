# Stage 0 – pull the CUDA 12.4 + cuDNN9 runtime image for AMD64
FROM --platform=linux/amd64 nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04 AS cudnn

# Stage 1 – your existing image, also forced to AMD64
FROM --platform=linux/amd64 594331569440.dkr.ecr.eu-west-2.amazonaws.com/podcast-ingestor:latest

# ─── 1. Copy cuDNN shared libraries ─────────────────────────────────────────
# Copy into /usr/lib/x86_64-linux-gnu (already on the default search path)
COPY --from=cudnn /usr/lib/x86_64-linux-gnu/libcudnn* /usr/lib/x86_64-linux-gnu/

# (If you prefer the /usr/local/cuda layout, just make the dir and add to path)
# RUN mkdir -p /usr/local/cuda/lib64 && \
#     cp /usr/lib/x86_64-linux-gnu/libcudnn* /usr/local/cuda/lib64/ && \
#     echo "export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH" >> /etc/profile.d/cudnn.sh

###############################################################################
# 2. OS packages (unchanged)
###############################################################################
RUN apt-get update \
 && apt-get install -y --no-install-recommends ffmpeg wget \
 && rm -rf /var/lib/apt/lists/*

###############################################################################
# 3. Project code (unchanged)
###############################################################################
WORKDIR /app
COPY . /app

###############################################################################
# 4. PyTorch 2.4.1 + cu124 (unchanged)
###############################################################################
RUN pip uninstall -y torch torchvision torchaudio && \
    pip install --no-cache-dir \
      torch==2.4.1+cu124 \
      torchvision==0.19.1+cu124 \
      torchaudio==2.4.1+cu124 \
      --extra-index-url https://download.pytorch.org/whl/cu124

###############################################################################
# 5. CTranslate2 + faster-whisper (unchanged)
###############################################################################
RUN pip install --no-cache-dir \
      "ctranslate2[gpu]==4.6.0" \
      faster-whisper==0.10.0

###############################################################################
# 6. Pre-download Whisper “medium” weights (unchanged)
###############################################################################
RUN python - <<'PY'
import whisperx
_ = whisperx.load_model("medium", device="cpu", compute_type="default")
PY

###############################################################################
# 7. Build-time sanity check (unchanged)
###############################################################################
RUN python - <<'PY'
import torch, ctranslate2, whisperx, ctypes, glob, os
print("✅ Torch       :", torch.__version__, "CUDA:", torch.version.cuda)
print("✅ CTranslate2 :", ctranslate2.__version__)
print("✅ WhisperX    :", "weights present" if glob.glob('/root/.cache/whisper/*') else 'missing')
print("✅ cuDNN libs  :", [os.path.basename(x) for x in glob.glob('/usr/lib/x86_64-linux-gnu/libcudnn*so*')][:2])
PY
