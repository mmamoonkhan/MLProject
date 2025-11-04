#!/bin/bash
set -e
echo "=== Step 1: Preprocessing data ==="
python preprocess_data.py
echo "=== Step 2: Training policy ==="
python train_policy.py
echo "=== Step 3: Evaluating policy ==="
python evaluate_policy.py
