#!/usr/bin/env bash
set -euo pipefail

mkdir -p datasets
cd datasets

declare -A URLS=(
  ["ca-GrQc.txt.gz"]="https://snap.stanford.edu/data/ca-GrQc.txt.gz"
  ["facebook_combined.txt.gz"]="https://snap.stanford.edu/data/facebook_combined.txt.gz"
  ["ca-AstroPh.txt.gz"]="https://snap.stanford.edu/data/ca-AstroPh.txt.gz"
  ["email-Enron.txt.gz"]="https://snap.stanford.edu/data/email-Enron.txt.gz"
  ["p2p-Gnutella30.txt.gz"]="https://snap.stanford.edu/data/p2p-Gnutella30.txt.gz"
  ["soc-Slashdot0811.txt.gz"]="https://snap.stanford.edu/data/soc-Slashdot0811.txt.gz"
  ["amazon0601.txt.gz"]="https://snap.stanford.edu/data/amazon0601.txt.gz"
  ["roadNet-CA.txt.gz"]="https://snap.stanford.edu/data/roadNet-CA.txt.gz"
)

for file in "${!URLS[@]}"; do
  txt="${file%.gz}"
  if [ ! -f "$txt" ]; then
    echo "Downloading $file..."
    curl -fL --retry 3 -o "$file" "${URLS[$file]}"
    gunzip -f "$file"
  else
    echo "Already exists: $txt"
  fi
done

echo ""
echo "Verified uncompressed files in datasets/:"
ls -lh
