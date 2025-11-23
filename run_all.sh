#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"
python -m src.preprocess_data --config configs/config.yaml
python -m src.train_policy --config configs/config.yaml
python -m src.evaluate_policy --config configs/config.yaml