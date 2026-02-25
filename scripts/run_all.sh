#!/bin/bash
# ============================================================
# Run All Experiments
# Trust-Aware Federated Learning with Blockchain Audit
# ============================================================

set -e  # Exit on first error

echo "============================================================"
echo " Trust-Aware FL + Blockchain: Full Experiment Suite"
echo "============================================================"

# Create directories
mkdir -p results/figures
mkdir -p data/mnist

echo ""
echo "[1/5] Exp 1: Static Attack Baseline (FedAvg vs Krum vs TrustAware)"
python experiments/exp_static_attack.py

echo ""
echo "[2/5] Exp 2: Adaptive Attack vs Trust-Aware Defense"
python experiments/exp_adaptive_attack.py

echo ""
echo "[3/5] Exp 3: Alpha & Tau Sensitivity Analysis"
python experiments/exp_alpha_sensitivity.py

echo ""
echo "[4/5] Exp 4: Ablation Study (Component Contribution)"
python experiments/exp_ablation.py

echo ""
echo "[5/5] Exp 5: Blockchain Overhead Measurement"
python experiments/exp_overhead.py

echo ""
echo "============================================================"
echo " ALL EXPERIMENTS COMPLETE"
echo " Results saved to: ./results/figures/"
echo "============================================================"