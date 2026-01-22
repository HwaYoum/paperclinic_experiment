#!/bin/bash

# ==========================================
# AES Environment Setup Script
# ==========================================
# Usage: ./setup_env.sh
# ==========================================

echo "=========================================="
echo " Setting up Environment..."
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
pip install -r requirements.txt

echo "=========================================="
echo " Environment Setup Complete!"
echo " Please activate venv before running training:"
echo " source venv/bin/activate"
echo "=========================================="
