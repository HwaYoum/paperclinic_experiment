#!/bin/bash

# ==========================================
# AES Fine-tuning Server Script
# ==========================================
# Usage: ./run_server.sh SJunha/aes-dataset
# ==========================================

HF_DATASET_ID=$1

if [ -z "$HF_DATASET_ID" ]; then
  echo "Error: Please provide your Hugging Face Dataset ID."
  echo "Usage: ./run_server.sh <HF_DATASET_ID>"
  exit 1
fi

echo "=========================================="
echo " Starting Setup for Dataset: $HF_DATASET_ID"
echo "=========================================="

# 1. System Update & Dependencies (Optional, requires sudo)
# echo "Checking system dependencies..."
# sudo apt-get update && sudo apt-get install -y python3-pip python3-venv git

# 2. Virtual Environment Setup
if [ ! -d "venv" ]; then
    echo "Creating virtual environment 'venv'..."
    python3 -m venv venv
fi

echo "Activating virtual environment..."
source venv/bin/activate

# 3. Install Python Dependencies
echo "Installing Python dependencies from requirements.txt..."
pip install --upgrade pip
# Force reinstall torch to match CUDA if needed, usually standard pip works for modern torch
pip install -r requirements.txt

# 4. Login to Hugging Face (Interactive)
# if ! python3 -c "import huggingface_hub; print(huggingface_hub.get_token())" &> /dev/null; then
#     echo "=========================================="
#     echo " Hugging Face Login Required"
#     echo " Please paste your HF Write Token below."
#     echo "=========================================="
#     huggingface-cli login
# fi

# 5. Run Training
echo "=========================================="
echo " Starting LoRA Fine-tuning..."
echo " This may take a while. Logs will be shown below."
echo "=========================================="

# Run the training script with the HF dataset argument
python train_lora.py --dataset_name "$HF_DATASET_ID"

echo "=========================================="
echo " Training Complete!"
echo " Check 'aes_finetuned/' directory for results."
echo "=========================================="
