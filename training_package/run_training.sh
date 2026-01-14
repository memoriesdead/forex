#!/bin/bash
# Vast.ai H100 Training Script
# Upload this package and run: ./run_training.sh

pip install -r requirements.txt
python train_models.py

echo "Training complete! Download hft_models.tar.gz"
