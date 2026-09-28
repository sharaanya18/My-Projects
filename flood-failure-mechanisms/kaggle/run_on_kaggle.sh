#!/usr/bin/env bash
# Run solution.py on a Kaggle GPU and download the log and submission.
# Needs Kaggle credentials the CLI can read: KAGGLE_USERNAME + KAGGLE_KEY, or an
# API token (KAGGLE_API_TOKEN or ~/.kaggle/access_token).
# Usage: bash run_on_kaggle.sh <path/to/dataset/public> [NvidiaTeslaT4]
set -euo pipefail
KAGGLE_USERNAME="${KAGGLE_USERNAME:-$(kaggle config view | sed -n 's/^- username: //p')}"
: "${KAGGLE_USERNAME:?could not determine Kaggle username; check credentials}"
DATA_DIR="$(cd "$1" && pwd)"; ACC="${2:-NvidiaTeslaT4}"
HERE="$(cd "$(dirname "$0")" && pwd)"
DS_SLUG="flood-failure-eris-data"; K_SLUG="flood-failure-eris-solution"
WORK="$(mktemp -d)"

# 1) Dataset: upload once as a private dataset; skip if it already exists.
if ! kaggle datasets status "$KAGGLE_USERNAME/$DS_SLUG" >/dev/null 2>&1; then
  mkdir -p "$WORK/ds" && cp "$DATA_DIR"/*.csv "$WORK/ds/"
  (cd "$DATA_DIR" && zip -qr "$WORK/ds/images.zip" images)
  cat > "$WORK/ds/dataset-metadata.json" <<JSON
{"title": "$DS_SLUG", "id": "$KAGGLE_USERNAME/$DS_SLUG", "licenses": [{"name": "other"}]}
JSON
  kaggle datasets create -p "$WORK/ds"
  echo "Waiting for dataset processing..."
  until kaggle datasets status "$KAGGLE_USERNAME/$DS_SLUG" 2>/dev/null | grep -qi ready; do sleep 20; done
fi

# 2) Kernel: prelude + solution.py as one private GPU script with internet on (timm weights).
mkdir -p "$WORK/k"
cat "$HERE/kaggle_prelude.py" "$HERE/../solution.py" > "$WORK/k/run.py"
cat > "$WORK/k/kernel-metadata.json" <<JSON
{"id": "$KAGGLE_USERNAME/$K_SLUG", "title": "$K_SLUG", "code_file": "run.py",
 "language": "python", "kernel_type": "script", "is_private": true,
 "enable_gpu": true, "enable_internet": true,
 "dataset_sources": ["$KAGGLE_USERNAME/$DS_SLUG"], "competition_sources": [], "kernel_sources": []}
JSON
kaggle kernels push -p "$WORK/k" --accelerator "$ACC"

# 3) Poll, then fetch outputs (log + submission.csv) into ./kaggle_output.
echo "Polling kernel status..."
while true; do
  S="$(kaggle kernels status "$KAGGLE_USERNAME/$K_SLUG" 2>&1 || true)"; echo "$(date +%T) $S"
  echo "$S" | grep -qiE 'complete|error|cancel' && break; sleep 60
done
mkdir -p "$HERE/kaggle_output"
kaggle kernels output "$KAGGLE_USERNAME/$K_SLUG" -p "$HERE/kaggle_output"
ls -la "$HERE/kaggle_output"
