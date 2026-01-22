#!/bin/bash

# ==========================================
# AES Training Execution Script
# ==========================================
# Usage: ./run_training.sh SJunha/aes-dataset
# ==========================================

HF_DATASET_ID=$1

if [ -z "$HF_DATASET_ID" ]; then
  echo "Error: Please provide your Hugging Face Dataset ID."
  echo "Usage: ./run_training.sh <HF_DATASET_ID>"
  exit 1
fi

# Activate Virtual Environment if exists
if [ -d "venv" ]; then
    echo "Activating virtual environment..."
    source venv/bin/activate
else
    echo "Warning: Virtual environment 'venv' not found."
    echo "Make sure you have run ./setup_env.sh or installed dependencies."
fi

# Run Training
echo "=========================================="
echo " Starting LoRA Fine-tuning..."
echo " Dataset: $HF_DATASET_ID"
echo "=========================================="

export PYTHONPATH=$PYTHONPATH:$(pwd)
python src/train/train_lora.py --dataset_name "$HF_DATASET_ID"

echo "=========================================="
echo " Training Complete!"
echo " Check 'aes_finetuned/' directory for results."
echo "=========================================="
