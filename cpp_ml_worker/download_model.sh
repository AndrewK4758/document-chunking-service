#!/bin/bash
set -e

echo "Downloading GLINER PII model (ONNX) and tokenizer to /home/ak/projects/ml..."

mkdir -p /home/ak/projects/ml
cd /home/ak/projects/ml

curl -L -o model.onnx "https://huggingface.co/onnx-community/gliner_multi_pii-v1/resolve/main/onnx/model.onnx"
curl -L -o tokenizer.json "https://huggingface.co/onnx-community/gliner_multi_pii-v1/resolve/main/tokenizer.json"

echo "Download complete! model.onnx and tokenizer.json are now in /home/ak/projects/ml."
