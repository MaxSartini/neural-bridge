#!/usr/bin/env bash
set -euo pipefail

echo "============================================================"
echo "Neural Bridge Neuro-Prior / TRIBE v2 Apple Silicon Setup"
echo "============================================================"

if [ ! -d "backend" ] || [ ! -d "frontend" ]; then
  echo "Run this script from the Neural Bridge-main project root."
  exit 1
fi

echo "=== 1. macOS tools ==="
xcode-select -p >/dev/null 2>&1 || xcode-select --install || true

if command -v brew >/dev/null 2>&1; then
  brew install git git-lfs ffmpeg cmake pkg-config rustup-init || true
  git lfs install || true
else
  echo "Homebrew not found. Install Homebrew if git-lfs, ffmpeg, or cmake fail."
fi

echo "=== 2. Python virtual environment ==="
cd backend
if [ ! -d ".venv" ]; then
  python3 -m venv .venv
fi
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel

echo "=== 3. Install existing backend requirements ==="
if [ -f "requirements.txt" ]; then
  pip install -r requirements.txt
fi
if [ -f "pyproject.toml" ]; then
  pip install -e . || true
fi
cd ..

echo "=== 4. Install ML/neuro dependencies ==="
source backend/.venv/bin/activate
pip install --upgrade \
  torch torchvision torchaudio transformers accelerate safetensors sentencepiece \
  huggingface_hub "huggingface_hub[hf_xet]" hf-transfer numpy scipy pandas librosa \
  soundfile ffmpeg-python whisperx ctranslate2 mlx-whisper einops tqdm matplotlib nilearn nibabel \
  mlx mlx-lm || true

pip install \
  neuralset==0.0.2 neuraltrain==0.0.2 x_transformers==1.27.20 exca==0.5.20 \
  "moviepy>=2.2.1" gtts langdetect spacy soundfile Levenshtein julius nilearn nibabel || true

# Keep the app's pinned OASIS stack stable after WhisperX/pyannote dependency resolution.
pip install numpy==2.2.6 pandas==2.2.2 || true

EXTERNAL_ROOT="${NEURAL_BRIDGE_EXTERNAL_ROOT:-$(pwd)/external_assets}"

echo "=== 5. Clone TRIBE v2 Apple Silicon branch ==="
mkdir -p external_models
if [ ! -d "external_models/tribev2-apple-silicon" ]; then
  git clone https://github.com/alangnt/tribev2.git external_models/tribev2-apple-silicon
fi
cd external_models/tribev2-apple-silicon
git fetch origin || true
git checkout feature/apple-silicon-support || true
git pull origin feature/apple-silicon-support || true
pip install --no-deps -e . || true
cd ../..

echo "=== 6. Prepare external model directories ==="
mkdir -p "${EXTERNAL_ROOT}/models/tribe" \
  "${EXTERNAL_ROOT}/models/tribe-mlx" \
  "${EXTERNAL_ROOT}/models/cortical-upstream" \
  "${EXTERNAL_ROOT}/models/upstream-encoders-mlx" \
  "${EXTERNAL_ROOT}/cache" \
  "${EXTERNAL_ROOT}/tmp"
export HF_HUB_ENABLE_HF_TRANSFER=1
export HF_XET_HIGH_PERFORMANCE=1
export TRIBE_TEXT_EVENTS_DIRECT="${TRIBE_TEXT_EVENTS_DIRECT:-true}"
export TRIBE_WHISPERX_MODEL="${TRIBE_WHISPERX_MODEL:-small}"
export TRIBE_WHISPERX_BATCH_SIZE="${TRIBE_WHISPERX_BATCH_SIZE:-4}"

echo "=== 7. Hugging Face login check ==="
echo "If this fails or you lack access to gated models, run: huggingface-cli login"
huggingface-cli whoami || true

echo "=== 8. Download official TRIBE v2 checkpoint assets to external root ==="
huggingface-cli download facebook/tribev2 --local-dir "${EXTERNAL_ROOT}/models/tribe/facebook-tribev2" --local-dir-use-symlinks False || true

if [ "${DOWNLOAD_TRIBE_OPTIONAL:-false}" = "true" ]; then
  echo "=== 9. Download TRIBE-MLX assets ==="
  huggingface-cli download zimengxiong/tribev2-mlx --local-dir "${EXTERNAL_ROOT}/models/tribe-mlx/zimengxiong-tribev2-mlx" --local-dir-use-symlinks False || true

  echo "=== 10. Download upstream feature extractors if access permits ==="
  huggingface-cli download meta-llama/Llama-3.2-3B --local-dir "${EXTERNAL_ROOT}/models/upstream-encoders/meta-llama-Llama-3.2-3B" --local-dir-use-symlinks False || true
  huggingface-cli download facebook/w2v-bert-2.0 --local-dir "${EXTERNAL_ROOT}/models/upstream-encoders/facebook-w2v-bert-2.0" --local-dir-use-symlinks False || true
  huggingface-cli download facebook/vjepa2-vitg-fpc64-256 --local-dir "${EXTERNAL_ROOT}/models/cortical-upstream/facebook-vjepa2-vitg-fpc64-256" --local-dir-use-symlinks False || true
else
  echo "=== 9. Optional TRIBE downloads skipped ==="
  echo "Set DOWNLOAD_TRIBE_OPTIONAL=true to fetch TRIBE-MLX and upstream encoder assets."
fi

echo "=== 11. Frontend dependencies ==="
cd frontend
npm install
cd ..

echo "============================================================"
echo "Setup complete."
echo "============================================================"
echo "LM Studio must be available at http://localhost:1234/v1 with Qwen loaded."
echo "TRIBE failures are non-fatal unless NEURO_PRIOR_STRICT=true."
echo "Apple Silicon defaults: MLX handles transcription/TRIBE head; supported upstream extractors use bounded MPS."
echo "Use exact chunked attention and selective hidden-state capture for the official 64-frame V-JEPA2 contract."
echo "Text TRIBE feature extraction requires Hugging Face access to meta-llama/Llama-3.2-3B unless using proxy fallback."
