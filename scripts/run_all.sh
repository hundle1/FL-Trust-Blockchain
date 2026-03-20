#!/bin/bash
# ================================================================
# Run All Experiments — Trust-Aware FL (Q1 Paper)
# ================================================================
# Thứ tự chạy:
#   1. exp_oldmethod.py   — Baseline comparison (tất cả defenses × attacks)
#   2. exp_newmethod.py   — Trust-Aware full analysis (tất cả attacks)
#   3. exp_ablation.py    — Ablation study (component contribution)
#   4. exp_alpha_sensitivity.py — Sensitivity analysis α × τ
# ================================================================

set -e

echo "================================================================"
echo " Trust-Aware FL — Full Experiment Suite (Q1 Paper)"
echo " Reference: coordinate-wise median (realistic)"
echo " Attacks: Static, Delayed, Adaptive, Intermittent,"
echo "          Norm-Tuned, Label Flip, Gaussian Noise"
echo "================================================================"

mkdir -p results/figures

echo ""
echo "[1/4] Baseline Comparison (FedAvg vs Krum vs TrimmedMean vs TrustAware)"
echo "       All 8 attack types — heatmap + ASR + radar chart"
python experiments/exp_oldmethod.py

echo ""
echo "[2/4] Trust-Aware Full Analysis"
echo "       Trust separation, detection rate, statistical significance"
python experiments/exp_newmethod.py all --ref median

echo ""
echo "[3/4] Ablation Study (FedAvg vs Clip-only vs Trust-only vs Full)"
python experiments/exp_ablation.py

echo ""
echo "[4/4] Sensitivity Analysis (α and τ)"
python experiments/exp_alpha_sensitivity.py

echo ""
echo "================================================================"
echo " ALL EXPERIMENTS COMPLETE"
echo " Results: ./results/figures/"
echo ""
echo " PRIMARY figures (use in paper):"
echo "   oldmethod_heatmap.png        — Defense × Attack comparison"
echo "   oldmethod_asr.png            — Attack Success Rate"
echo "   oldmethod_radar.png          — Robustness radar"
echo "   newmethod_trust_separation.png — Trust evolution ± 95% CI"
echo "   newmethod_detection.png        — Detection + FP rate"
echo "   newmethod_statistical.png      — Cohen's d + p-values"
echo "   newmethod_summary.png          — Summary heatmap + stats table"
echo "   ablation_heatmap.png           — Component contribution"
echo "   ablation_contribution.png      — Marginal gains"
echo "   sensitivity_alpha_tau.png      — α and τ sensitivity"
echo "   sensitivity_joint_adaptive.png — Joint α×τ heatmap"
echo "================================================================"