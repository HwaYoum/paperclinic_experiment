#!/bin/bash

# ==========================================
# AES Training Execution Script
# ==========================================
# Usage: ./run_training.sh [MODE]
# Modes: zero_shot, original, augmented, full
# Default: full
# ==========================================

MODE=${1:-full}
HF_DATASET_ID="SJunha/aes-dataset"

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
echo " Experiment Mode: $MODE"
echo " Dataset: $HF_DATASET_ID"
echo "=========================================="

export PYTHONPATH=$PYTHONPATH:$(pwd)
python src/train/train_lora.py --dataset_name "$HF_DATASET_ID" --experiment_mode "$MODE"

echo "=========================================="
echo " Process Complete!"
echo " Check 'aes_finetuned_$MODE/' directory for results."
echo "=========================================="
