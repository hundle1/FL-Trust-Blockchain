# Trust-Aware Federated Learning with Blockchain Audit Trail

A research implementation of Trust-Aware Federated Learning (FL) with blockchain-based audit logging and support for various gradient poisoning attacks.

## Overview

This project investigates how a trust-based defense mechanism can protect FL systems from poisoning attacks, with blockchain providing an immutable audit trail.

**Key components:**
- **Trust Scoring** — EMA-based trust scores updated via gradient cosine similarity (`α`, `τ` parameters)
- **Trust-Aware Aggregation** — Clients below trust threshold `τ` are filtered out; remaining clients are weighted by trust
- **Blockchain Audit** — Mock blockchain (low-overhead simulation) logs all client updates and aggregation events
- **Attack Suite** — Static, delayed, intermittent, adaptive, and norm-tuned poisoning attacks

## Project Structure

```
fl_project/
├── fl_core/
│   ├── client.py               # FLClient: local training, parameter management
│   ├── server.py               # FLServer: client selection, aggregation coordination
│   ├── trainer.py              # FLTrainer: high-level orchestration
│   └── aggregation/
│       ├── trust_aware.py      # TrustAwareAggregator, FedAvg, Krum, TrimmedMean
│       ├── fedavg.py           # Standalone FedAvg
│       ├── krum.py             # Standalone Krum
│       └── trimmed_mean.py     # Standalone TrimmedMean
├── trust/
│   ├── trust_score.py          # TrustScoreManager (core trust mechanism)
│   ├── trust_decay.py          # Decay strategies: exponential, threshold, adaptive
│   ├── behavior_metrics.py     # Cosine similarity, norm ratio, anomaly scores
│   └── history_buffer.py       # Per-client history management
├── attacks/
│   ├── adaptive_controller.py  # Unified attack controller (static/delayed/intermittent/adaptive)
│   ├── delayed_poisoning.py    # Delayed poisoning attack
│   ├── intermittent_poisoning.py  # Intermittent attack with patterns
│   └── norm_tuned_attack.py    # Norm-matched stealthy attack
├── blockchain/
│   ├── ledger.py               # MockBlockchain (low-overhead simulation)
│   ├── mock_chain.py           # High-level chain API for FL logging
│   ├── audit_logger.py         # Hash-chained audit logger
│   └── smart_contract.py       # Automated trust-threshold actions
├── models/
│   ├── cnn_mnist.py            # CNN for MNIST (primary model)
│   ├── cnn_cifar.py            # VGG-style CNN for CIFAR-10
│   └── resnet_femnist.py       # Lightweight ResNet for FEMNIST
├── evaluation/
│   ├── metrics.py              # ASR, convergence, trust, defense metrics
│   ├── attack_success_rate.py  # ASR computation and plotting
│   ├── convergence.py          # Convergence speed and stability analysis
│   └── trust_evolution.py      # Trust score evolution visualization
├── experiments/
│   ├── exp_static_attack.py    # Exp 1: FedAvg vs Krum vs TrustAware vs Static attack
│   ├── exp_adaptive_attack.py  # Exp 2: Static vs Delayed vs Adaptive attack
│   ├── exp_alpha_sensitivity.py # Exp 3: α and τ hyperparameter sensitivity
│   ├── exp_ablation.py         # Exp 4: Component contribution ablation
│   └── exp_overhead.py         # Exp 5: Blockchain overhead measurement
├── config/
│   ├── fl_config.yaml          # FL training hyperparameters
│   ├── trust_config.yaml       # Trust mechanism configuration
│   ├── attack_config.yaml      # Attack parameters
│   └── blockchain_config.yaml  # Blockchain simulation settings
├── scripts/
│   ├── run_all.sh              # Run all 5 experiments sequentially
│   └── reproduce_paper.sh      # Reproduce main paper results (3 core exps)
├── requirements.txt
└── README.md
```

## Setup

```bash
pip install -r requirements.txt
```

**Requirements:** Python 3.8+, PyTorch ≥ 1.13, torchvision, numpy, matplotlib, seaborn, tqdm, pyyaml

## Quick Start

```bash
# Run all experiments
bash scripts/run_all.sh

# Or run individual experiments
python experiments/exp_static_attack.py    # ~5-10 min
python experiments/exp_adaptive_attack.py  # ~10 min
python experiments/exp_alpha_sensitivity.py  # ~30 min (grid sweep)
python experiments/exp_ablation.py          # ~15 min
python experiments/exp_overhead.py          # ~5 min
```

Results are saved to `results/figures/`.

## Trust Mechanism

Trust scores are updated each round using EMA over gradient cosine similarity:

```
T_i(t+1) = α · T_i(t) + (1 − α) · S_i(t)
```

Where:
- `α` (alpha) — memory factor (0.9 default). Higher = slower trust change
- `S_i(t)` — gradient cosine similarity with reference gradient, mapped to [0, 1]
- `τ` (tau) — threshold below which clients are excluded from aggregation (0.3 default)

## Experiments

| # | Script | Description | Key Result |
|---|--------|-------------|------------|
| 1 | `exp_static_attack.py` | FedAvg vs Krum vs TrustAware under static attack | TrustAware outperforms all |
| 2 | `exp_adaptive_attack.py` | Static vs delayed vs adaptive attackers | Trust-Aware remains robust |
| 3 | `exp_alpha_sensitivity.py` | α ∈ [0.5, 0.99] and τ ∈ [0.1, 0.6] sweep | α=0.9, τ=0.3 optimal |
| 4 | `exp_ablation.py` | Component contribution (no trust / +trust / +decay / +blockchain) | Each component helps |
| 5 | `exp_overhead.py` | Blockchain time and storage overhead | <5% time overhead |

## Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `alpha` (α) | 0.9 | EMA decay factor — trust memory |
| `tau` (τ) | 0.3 | Trust threshold for client filtering |
| `attack_rate` | 0.2 | Fraction of malicious clients (20%) |
| `poisoning_scale` | 5.0 | Gradient flip multiplier |
| `num_clients` | 100 | Total FL clients |
| `clients_per_round` | 10 | Clients selected each round |
| `local_epochs` | 5 | Local training epochs per round |
| `block_time` | 0.05s | Simulated blockchain block creation time |

## Attack Types

- **Static** — Attacks every round
- **Delayed** — Behaves honestly for D rounds to build trust, then attacks
- **Intermittent** — Attacks with probability p (random/periodic/burst patterns)
- **Adaptive** — Monitors own trust estimate; attacks only when trust > threshold
- **Norm-Tuned** — Scales attack gradient to match benign gradient norms (stealthy)

## Blockchain Overhead

The blockchain uses a mock simulation with `block_time=0.05s` (configurable). For 50-round experiments with 50 clients, the typical overhead is:
- **Time:** < 5% additional wall-clock time
- **Storage:** ~5–15 KB total chain size

## Citation / Reference

If you use this code, please cite the accompanying paper.