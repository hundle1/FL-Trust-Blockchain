#!/bin/bash
# ============================================================
# Reproduce Paper Results
# Uses fixed seeds and paper hyperparameters
# ============================================================

set -e

echo "============================================================"
echo " Reproducing Paper Results"
echo " Seeds: torch=42, numpy=42"
echo "============================================================"

mkdir -p results/figures

# Run with paper config (full rounds)
echo ""
echo "[1/3] Main comparison experiment (100 rounds)..."
python experiments/exp_static_attack.py

echo ""
echo "[2/3] Adaptive attacker experiment (80 rounds)..."
python experiments/exp_adaptive_attack.py

echo ""
echo "[3/3] Ablation study (80 rounds)..."
python experiments/exp_ablation.py

echo ""
echo "Figures saved to ./results/figures/"
echo "Done."