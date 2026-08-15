#!/bin/bash
set -e

echo "Downloading GLINER PII model (ONNX) and tokenizer..."

cd "$(dirname "$0")"

# Download the ONNX model
curl -L -o model.onnx "https://huggingface.co/onnx-community/gliner_multi_pii-v1/resolve/main/onnx/model.onnx"

# Download the tokenizer
curl -L -o tokenizer.json "https://huggingface.co/onnx-community/gliner_multi_pii-v1/resolve/main/tokenizer.json"

echo "Download complete! model.onnx and tokenizer.json are now in the cpp_ml_worker directory."
echo "Remember to rebuild the docker container so they are copied into the image:"
echo "docker-compose build cpp-ml-worker"
