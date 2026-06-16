#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-${ROOT_DIR}/backend/.venv/bin/python}"

# momentfm 0.1.4 pins old NumPy/Transformers/Hugging Face versions, but its
# current code is compatible with the project's newer tested stack.
"${PYTHON_BIN}" -m pip install "netneurotools>=0.3.0"
"${PYTHON_BIN}" -m pip install --no-deps "momentfm==0.1.4"

"${PYTHON_BIN}" - <<'PY'
import numpy
import torch
import transformers
from momentfm import MOMENTPipeline

print({
    "moment_pipeline": MOMENTPipeline.__name__,
    "torch": torch.__version__,
    "transformers": transformers.__version__,
    "numpy": numpy.__version__,
    "mps_available": torch.backends.mps.is_available(),
})
PY
