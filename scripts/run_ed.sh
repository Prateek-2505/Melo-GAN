#!/usr/bin/env bash
# scripts/run_ed.sh
# Run Emotion Discriminator training, evaluation, and save logs for visualization.
# Usage: ./scripts/run_ed.sh [path-to-config] [device]
#
# Example:
#   ./scripts/run_ed.sh config/ed_config.yaml cuda

set -euo pipefail

CFG=${1:-"config/ed_config.yaml"}
DEVICE=${2:-""}   # optionally set CUDA device e.g. "cuda:0" or leave empty to use config device

# read some config keys with python to set directories (safe fallback if not present)
LOG_DIR=$(python - <<PY
import yaml,sys,os
c=yaml.safe_load(open("$CFG"))
print(c.get("log_dir","data/experiments/ed"))
PY
)

CKPT_DIR=$(python - <<PY
import yaml,sys,os
c=yaml.safe_load(open("$CFG"))
print(c.get("checkpoint_dir","data/models/ed"))
PY
)

mkdir -p "$LOG_DIR"
mkdir -p "$CKPT_DIR"

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOGFILE="${LOG_DIR}/train_ed_${TIMESTAMP}.log"

echo "=== MELO-GAN Phase1A: ED Training Run ==="
echo "Config: $CFG"
echo "Device override: ${DEVICE:-<none>}"
echo "Logfile: $LOGFILE"
echo

# Run training and tee output to logfile (both stdout and stderr)
if [ -n "$DEVICE" ]; then
  python -u -m src.emotion_discriminator.train_ed --config "$CFG" --device "$DEVICE" 2>&1 | tee "$LOGFILE"
else
  python -u -m src.emotion_discriminator.train_ed --config "$CFG" 2>&1 | tee "$LOGFILE"
fi

# Look up the best checkpoint filename (train_ed saved ed_best.pth)
BEST_CKPT="${CKPT_DIR}/ed_best.pth"
if [ ! -f "$BEST_CKPT" ]; then
  echo "Warning: best checkpoint not found at $BEST_CKPT. Searching for the most recent epoch checkpoint..."
  LATEST=$(ls -1t "${CKPT_DIR}"/ed_epoch*.pth 2>/dev/null | head -n1 || true)
  if [ -n "$LATEST" ]; then
    BEST_CKPT="$LATEST"
    echo "Using $BEST_CKPT"
  else
    echo "No checkpoint found — aborting evaluation."
    exit 0
  fi
fi

# Run evaluation on test split
echo
echo "=== Running evaluation on test split ==="
python -u -m src.emotion_discriminator.evaluate_ed --config "$CFG" --ckpt "$BEST_CKPT" --split test --out_dir "$LOG_DIR"

echo
echo "=== Done. Training log and evaluation artifacts are in: $LOG_DIR ==="
echo "To visualize training curves, run:"
echo "  python -m src.emotion_discriminator.visualize_training --logfile \"$LOGFILE\" --out_dir \"$LOG_DIR\""
