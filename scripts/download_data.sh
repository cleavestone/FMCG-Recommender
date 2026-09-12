#!/usr/bin/env bash
# Download the Dunnhumby "The Complete Journey" dataset from Kaggle.
# Requires: ~/.kaggle/kaggle.json with a valid API token. Uses the kaggle CLI
# from the uv-managed environment.
set -euo pipefail

SLUG="frtgnn/dunnhumby-the-complete-journey"
DEST="data/raw"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

mkdir -p "$DEST"

if [ -f "$DEST/transaction_data.csv" ]; then
  echo "Data already present in $DEST — skipping download."
  exit 0
fi

echo "Downloading $SLUG -> $DEST"
uv run --no-sync kaggle datasets download -d "$SLUG" -p "$DEST" --unzip

echo "Files:"
ls -lh "$DEST"
