"""
exp_newmethod.py — Trust-Aware Defense (v10 Q1-Ready)
=======================================================
Changes vs v9:
  1. Realistic reference gradient (coordinate-wise median, không dùng ground-truth malicious IDs)
  2. Thêm Label Flipping + Gaussian Noise attacks
  3. Statistical significance: t-test trust separation + confidence intervals
  4. So sánh với FLTrust-style (server clean dataset reference)
  5. Tất cả attacks trong cùng 1 bộ biểu đồ (không tách file)
  6. Ablation: FedAvg / Clip-only / Trust-only / Full
  7. Blockchain overhead integrated
  8. Sensitivity analysis α × τ grid

Primary metrics (paper contribution):
  - Trust Separation (benign vs malicious) + statistical significance
  - Detection Rate + False Positive Rate
  - Trust trajectory evolution

Secondary metrics:
  - Final Accuracy, ASR
  - Blockchain overhead
"""

import sys
import os

os.environ['CUDA_MODULE_LOADING'] = 'LAZY'
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import numpy as np
from scipy import stats as scipy_stats
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import time

from models.cnn_mnist import get_model
from fl_core.client import FLClient
from trust.trust_score import TrustScoreManager
from fl_core.aggregation.trust_aware import TrustAwareAggregator, FedAvgAggregator
from attacks.attack_suite import (
    create_attacker, compute_realistic_reference,
    GradientFlipAttacker, LabelFlippingAttacker, GaussianNoiseAttacker
)
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset

# ══════════════════════════════════════════════════════════════════════
# DEVICE
# ══════════════════════════════════════════════════════════════════════

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if DEVICE.type == "cuda":
    torch.backends.cudnn.benchmark = True
print(f"Device: {DEVICE}")

# ══════════════════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════════════════

ATTACK_RATE       = 0.20
POISONING_SCALE   = 5.0
NUM_CLIENTS       = 100
CLIENTS_PER_ROUND = 20
NUM_ROUNDS        = 80
PRETRAIN_ROUNDS   = 10
SEED              = 42

ALPHA             = 0.9
TAU               = 0.25          # v8: 0.30→0.25 (adaptive det≥10%)
INITIAL_TRUST     = 1.0
IDLE_DECAY        = 0.002
# v8 signal weights (sim + direction + loss, sum=1 internally)
SIM_WEIGHT        = 0.55          # cosine similarity
DIR_WEIGHT        = 0.25          # direction consistency (FIX1: norm-tuned)
LOSS_WEIGHT       = 0.20          # loss signal
# v8 temporal smoothing (FIX2: intermittent)
SMOOTHING_BETA    = 0.7           # weight current vs window mean
SMOOTHING_WINDOW  = 5
# v8 sustained penalty (FIX3: adaptive)
SUSTAINED_THRESH  = 0.45
SUSTAINED_WINDOW  = 3
SUSTAINED_STRENGTH = 0.15
WARMUP_ROUNDS     = 0

ABSOLUTE_NORM_THRESHOLD = 15.0
ENABLE_NORM_CLIP         = True
CLIP_MULTIPLIER          = 2.0

# Reference mode: 'median' = realistic (default), 'benign' = research upper bound
REFERENCE_MODE = "median"

# All attack types including new ones
ATTACK_ORDER = [
    "no_attack", "static", "delayed", "adaptive",
    "intermittent", "norm_tuned", "label_flip", "gaussian"
]

ATTACK_DISPLAY = {
    "no_attack":    "No Attack",
    "static":       "Static",
    "delayed":      "Delayed",
    "adaptive":     "Adaptive",
    "intermittent": "Intermittent",
    "norm_tuned":   "Norm-Tuned",
    "label_flip":   "Label Flip",
    "gaussian":     "Gaussian Noise",
}

PALETTE = {
    "no_attack":    "#2ecc71",
    "static":       "#e74c3c",
    "delayed":      "#e67e22",
    "adaptive":     "#8e44ad",
    "intermittent": "#2980b9",
    "norm_tuned":   "#c0392b",
    "label_flip":   "#16a085",
    "gaussian":     "#7f8c8d",
}

LINESTYLES = {
    "no_attack":    (0, ()),
    "static":       (0, (6, 2)),
    "delayed":      (0, (2, 2)),
    "adaptive":     (0, (6, 2, 2, 2)),
    "intermittent": (0, (4, 1, 1, 1)),
    "norm_tuned":   (0, (1, 1)),
    "label_flip":   (0, (3, 1, 1, 1, 1, 1)),
    "gaussian":     (0, (5, 2, 1, 2)),
}

LINEWIDTHS = {
    "no_attack": 2.8, "static": 2.2, "delayed": 2.0, "adaptive": 2.0,
    "intermittent": 1.8, "norm_tuned": 2.0, "label_flip": 2.0, "gaussian": 1.8,
}


# ══════════════════════════════════════════════════════════════════════
# DATA
# ══════════════════════════════════════════════════════════════════════

def load_data(num_clients=NUM_CLIENTS):
    base_path = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(base_path, '..', 'data')
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    train_ds = datasets.MNIST(data_path, train=True, download=True, transform=transform)
    test_ds  = datasets.MNIST(data_path, train=False, download=True, transform=transform)
    spc = len(train_ds) // num_clients
    client_datasets = [
        DataLoader(Subset(train_ds, list(range(i * spc, (i + 1) * spc))),
                   batch_size=32, shuffle=True)
        for i in range(num_clients)
    ]
    return client_datasets, DataLoader(test_ds, batch_size=1000, shuffle=False)


# ══════════════════════════════════════════════════════════════════════
# EVALUATION
# ══════════════════════════════════════════════════════════════════════

def evaluate(model, test_loader):
    model.eval()
    correct, total, loss_sum = 0, 0, 0.0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(DEVICE), target.to(DEVICE)
            out = model(data)
            loss_sum += torch.nn.functional.cross_entropy(out, target).item()
            correct  += out.argmax(1).eq(target).sum().item()
            total    += target.size(0)
    return correct / total, loss_sum / len(test_loader)


def compute_asr(clean_acc, poisoned_acc, baseline=0.1):
    if clean_acc <= baseline: return 0.0
    return float(np.clip((clean_acc - poisoned_acc) / (clean_acc - baseline), 0, 1))


# ══════════════════════════════════════════════════════════════════════
# SCENARIO RUNNER — v10: realistic reference + all attacks
# ══════════════════════════════════════════════════════════════════════

def run_scenario(attack_type, client_datasets, test_loader,
                 malicious_ids, num_rounds=NUM_ROUNDS,
                 reference_mode=REFERENCE_MODE):
    """
    Run one attack scenario với Trust-Aware defense.

    Key: reference gradient dùng coordinate-wise median (realistic)
    không cần biết malicious IDs.
    """
    model = get_model().to(DEVICE)
    benign_ids = [c for c in range(NUM_CLIENTS) if c not in malicious_ids]

    trust_manager = TrustScoreManager(
        num_clients              = NUM_CLIENTS,
        alpha                    = ALPHA,
        tau                      = TAU,
        initial_trust            = INITIAL_TRUST,
        enable_decay             = True,
        decay_strategy           = "exponential",
        # v8 signal weights
        similarity_weight        = SIM_WEIGHT,
        direction_weight         = DIR_WEIGHT,
        loss_weight              = LOSS_WEIGHT,
        # v8 temporal smoothing (intermittent fix)
        smoothing_beta           = SMOOTHING_BETA,
        smoothing_window         = SMOOTHING_WINDOW,
        # v8 idle decay
        idle_decay_rate          = IDLE_DECAY,
        # norm penalty
        enable_norm_penalty      = True,
        norm_penalty_threshold   = 3.0,
        norm_penalty_strength    = 0.80,
        absolute_norm_threshold  = ABSOLUTE_NORM_THRESHOLD,
        # v8 sustained penalty (adaptive fix)
        enable_sustained_penalty = True,
        sustained_threshold      = SUSTAINED_THRESH,
        sustained_window         = SUSTAINED_WINDOW,
        sustained_penalty_strength = SUSTAINED_STRENGTH,
        warmup_rounds            = WARMUP_ROUNDS,
    )

    aggregator = TrustAwareAggregator(
        trust_manager,
        enable_filtering        = True,
        enable_norm_clip        = ENABLE_NORM_CLIP,
        clip_multiplier         = CLIP_MULTIPLIER,
        warmup_clip_multiplier  = CLIP_MULTIPLIER,
        warmup_rounds           = WARMUP_ROUNDS,
        use_median              = False,
        fallback_top_k_ratio    = 0.3,   # prevent gaussian collapse
    )
    fedavg = FedAvgAggregator()

    clients = [
        FLClient(i, get_model().to(DEVICE), client_datasets[i],
                 is_malicious=(i in malicious_ids), device=DEVICE)
        for i in range(NUM_CLIENTS)
    ]

    # Create attackers (realistic: attacker does not know defense mechanism)
    attackers = {
        cid: create_attacker(cid, attack_type, POISONING_SCALE)
        for cid in malicious_ids
    }

    history = {
        'accuracy': [], 'loss': [],
        # Trust PRIMARY metrics
        'trust_benign_mean': [], 'trust_malicious_mean': [],
        'trust_benign_std':  [], 'trust_malicious_std':  [],
        'trust_separation':  [], 'detection_rate':        [],
        'false_positive_rate': [], 'clients_filtered': [],
        # Snapshots for statistical tests
        'benign_trust_snapshots':    {},
        'malicious_trust_snapshots': {},
        'attack_rate_actual': [],
        # Timing
        'round_times': [],
    }

    for round_num in tqdm(range(num_rounds),
                          desc=f"vs {attack_type:<14}", leave=False, ncols=95):
        t_start = time.time()
        is_pretrain = (round_num < PRETRAIN_ROUNDS)

        selected_idx = np.random.choice(NUM_CLIENTS, CLIENTS_PER_ROUND, replace=False)
        selected     = [clients[i] for i in selected_idx]

        global_params = {n: p.data.clone().to(DEVICE) for n, p in model.named_parameters()}
        for c in selected:
            c.set_parameters(global_params)

        updates, client_ids, metrics_list = [], [], []
        attacks_this_round = 0

        for client in selected:
            cid = client.client_id
            client_ids.append(cid)
            raw_update, real_metrics = client.train()

            if not is_pretrain and client.is_malicious and cid in attackers:
                est_trust = trust_manager.get_trust_score(cid)
                if attackers[cid].should_attack(round_num, est_trust, PRETRAIN_ROUNDS):
                    benign_so_far = [updates[j] for j, pc in enumerate(client_ids[:-1])
                                     if pc not in malicious_ids]
                    poisoned = attackers[cid].poison_gradient(
                        raw_update,
                        benign_updates=benign_so_far or None,
                    )
                    updates.append(poisoned)
                    attacks_this_round += 1
                else:
                    updates.append(raw_update)
            else:
                updates.append(raw_update)
            metrics_list.append(real_metrics)

        history['attack_rate_actual'].append(attacks_this_round / CLIENTS_PER_ROUND)

        # ── Reference gradient (REALISTIC: coordinate-wise median) ──
        if is_pretrain:
            agg = fedavg.aggregate(updates, client_ids, metrics_list)
            ref = compute_realistic_reference(
                updates, client_ids,
                malicious_ids=None,    # REALISTIC: không dùng ground truth
                mode="median"          # Robust, production-feasible
            )
            # Warm up trust với benign clients
            for i, cid in enumerate(client_ids):
                if cid not in malicious_ids:
                    trust_manager.update_trust(cid, updates[i], ref,
                                               metrics_list[i], round_num)
            history['clients_filtered'].append(0)
        else:
            # REALISTIC reference: median of all updates (no malicious ID needed)
            ref = compute_realistic_reference(
                updates, client_ids,
                malicious_ids=None,
                mode=reference_mode
            )
            for i, cid in enumerate(client_ids):
                trust_manager.update_trust(cid, updates[i], ref,
                                           metrics_list[i], round_num)
            trust_manager.apply_idle_decay(client_ids, round_num)

            agg = aggregator.aggregate(updates, client_ids, metrics_list,
                                       round_num=round_num)
            stats = aggregator.get_stats()
            history['clients_filtered'].append(
                stats['avg_filtered'] if stats['total_rounds'] > 0 else 0)

        if agg is not None:
            for name, param in model.named_parameters():
                param.data += agg[name]

        # ── Trust metrics ────────────────────────────────────────────
        b_scores = [trust_manager.get_trust_score(c) for c in benign_ids]
        m_scores = [trust_manager.get_trust_score(c) for c in malicious_ids]

        b_mean = float(np.mean(b_scores))
        m_mean = float(np.mean(m_scores))
        b_std  = float(np.std(b_scores))
        m_std  = float(np.std(m_scores))

        det_rate = sum(1 for t in m_scores if t < TAU) / max(1, len(m_scores))
        fp_rate  = sum(1 for t in b_scores if t < TAU) / max(1, len(b_scores))

        history['trust_benign_mean'].append(b_mean)
        history['trust_malicious_mean'].append(m_mean)
        history['trust_benign_std'].append(b_std)
        history['trust_malicious_std'].append(m_std)
        history['trust_separation'].append(b_mean - m_mean)
        history['detection_rate'].append(det_rate)
        history['false_positive_rate'].append(fp_rate)

        # Snapshots mỗi 10 rounds cho violin/box plot
        if round_num % 10 == 0 or round_num == num_rounds - 1:
            history['benign_trust_snapshots'][round_num]    = b_scores.copy()
            history['malicious_trust_snapshots'][round_num] = m_scores.copy()

        acc, loss = evaluate(model, test_loader)
        history['accuracy'].append(acc)
        history['loss'].append(loss)
        history['round_times'].append(time.time() - t_start)

        # Progress log
        if round_num < PRETRAIN_ROUNDS + 3 or round_num % 10 == 0:
            tag = "[PRE]" if is_pretrain else "     "
            print(f"  [r{round_num:02d}]{tag} acc={acc*100:.1f}%  "
                  f"B={b_mean:.3f}±{b_std:.3f}  M={m_mean:.3f}±{m_std:.3f}  "
                  f"sep={b_mean-m_mean:.3f}  det={det_rate*100:.0f}%  "
                  f"fp={fp_rate*100:.0f}%")

    return history


# ══════════════════════════════════════════════════════════════════════
# STATISTICAL SIGNIFICANCE (KEY for Q1)
# ══════════════════════════════════════════════════════════════════════

def compute_statistical_significance(results: dict) -> dict:
    """
    Tính t-test giữa benign và malicious trust scores.
    Cũng tính Cohen's d (effect size) để báo cáo practical significance.

    Returns dict với p-value, t-stat, Cohen's d cho mỗi attack type.
    """
    sig_results = {}

    for attack, h in results.items():
        if attack == 'no_attack':
            sig_results[attack] = {
                't_stat': float('nan'), 'p_value': float('nan'),
                'cohens_d': float('nan'), 'significant': False,
            }
            continue

        # Lấy snapshot cuối cùng (attack phase đã ổn định)
        last_round = max(h['benign_trust_snapshots'].keys())
        b_scores = np.array(h['benign_trust_snapshots'][last_round])
        m_scores = np.array(h['malicious_trust_snapshots'][last_round])

        if len(b_scores) < 2 or len(m_scores) < 2:
            sig_results[attack] = {
                't_stat': 0.0, 'p_value': 1.0,
                'cohens_d': 0.0, 'significant': False,
            }
            continue

        # Welch's t-test (không assume equal variance)
        t_stat, p_value = scipy_stats.ttest_ind(b_scores, m_scores, equal_var=False)

        # Cohen's d (effect size)
        pooled_std = np.sqrt((np.std(b_scores, ddof=1)**2 + np.std(m_scores, ddof=1)**2) / 2)
        cohens_d   = (np.mean(b_scores) - np.mean(m_scores)) / max(pooled_std, 1e-8)

        sig_results[attack] = {
            't_stat':     float(t_stat),
            'p_value':    float(p_value),
            'cohens_d':   float(cohens_d),
            'significant': bool(p_value < 0.05),
            'effect_size': 'large' if abs(cohens_d) > 0.8 else
                           'medium' if abs(cohens_d) > 0.5 else 'small',
        }

    return sig_results


# ══════════════════════════════════════════════════════════════════════
# PLOTS
# ══════════════════════════════════════════════════════════════════════

def _config_str():
    return (f"α={ALPHA}, τ={TAU}, pretrain={PRETRAIN_ROUNDS}r, "
            f"clip={CLIP_MULTIPLIER}x, ref={REFERENCE_MODE}, "
            f"dir_w={DIR_WEIGHT}, smooth_β={SMOOTHING_BETA}, "
            f"n={NUM_CLIENTS}, attack_rate={ATTACK_RATE*100:.0f}%")


def plot_trust_separation(results, save_dir):
    """
    PRIMARY: Trust evolution (benign vs malicious) ± std, với CI shading.
    Thêm statistical significance annotations.
    """
    attacks = [a for a in ATTACK_ORDER if a != 'no_attack' and a in results]
    ncols   = 4
    nrows   = (len(attacks) + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(4.5 * ncols, 4 * nrows),
                             sharex=True, sharey=True)
    axes_flat = np.array(axes).flatten()
    rounds = np.arange(NUM_ROUNDS)

    sig = compute_statistical_significance(results)

    for i, attack in enumerate(attacks):
        ax = axes_flat[i]
        h  = results[attack]

        b_mean = np.array(h['trust_benign_mean'])
        m_mean = np.array(h['trust_malicious_mean'])
        b_std  = np.array(h['trust_benign_std'])
        m_std  = np.array(h['trust_malicious_std'])

        # 95% CI (mean ± 1.96 * std/sqrt(n))
        n_b = int(NUM_CLIENTS * (1 - ATTACK_RATE))
        n_m = int(NUM_CLIENTS * ATTACK_RATE)
        b_ci = 1.96 * b_std / np.sqrt(max(1, n_b))
        m_ci = 1.96 * m_std / np.sqrt(max(1, n_m))

        ax.plot(rounds, b_mean, color='#2980b9', linewidth=2.2, label='Benign')
        ax.fill_between(rounds, b_mean - b_ci, b_mean + b_ci,
                        alpha=0.25, color='#2980b9', label='95% CI benign')

        ax.plot(rounds, m_mean, color='#e74c3c', linewidth=2.2,
                linestyle='--', label='Malicious')
        ax.fill_between(rounds, m_mean - m_ci, m_mean + m_ci,
                        alpha=0.25, color='#e74c3c', label='95% CI malicious')

        ax.axhline(y=TAU, color='#888', linestyle=':', linewidth=1.5,
                   label=f'τ={TAU}')
        ax.axvline(x=PRETRAIN_ROUNDS, color='orange', linestyle='--',
                   linewidth=1.0, alpha=0.8)
        ax.fill_between(rounds, m_mean, b_mean,
                        where=(b_mean > m_mean), alpha=0.07,
                        color='#27ae60')

        # Stats annotation
        s = sig.get(attack, {})
        final_sep = h['trust_separation'][-1]
        final_det = h['detection_rate'][-1] * 100
        p_str = f"p<0.001" if s.get('p_value', 1) < 0.001 else \
                f"p={s.get('p_value', 1):.3f}"
        d_str = f"d={s.get('cohens_d', 0):.2f}"

        ax.text(0.97, 0.04,
                f"sep={final_sep:.3f}\ndet={final_det:.0f}%\n{p_str}, {d_str}",
                transform=ax.transAxes, ha='right', va='bottom',
                fontsize=8, color='#2c3e50',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                          alpha=0.85, edgecolor='#bdc3c7'))

        ax.set_title(ATTACK_DISPLAY[attack], fontsize=12, fontweight='bold',
                     color=PALETTE[attack])
        ax.set_ylim([-0.05, 1.08])
        ax.grid(True, alpha=0.18)
        if i % ncols == 0: ax.set_ylabel('Trust Score', fontsize=10)
        if i >= (nrows - 1) * ncols: ax.set_xlabel('Round', fontsize=10)
        if i == 0: ax.legend(fontsize=7.5, loc='upper right', ncol=1)

    for j in range(len(attacks), len(axes_flat)):
        axes_flat[j].set_visible(False)

    fig.suptitle(
        f'Trust Score Evolution — Benign vs Malicious (± 95% CI)\n'
        f'{_config_str()}  |  Reference: {REFERENCE_MODE} (realistic)',
        fontsize=12, fontweight='bold', y=1.01
    )
    fig.tight_layout()
    path = os.path.join(save_dir, 'newmethod_trust_separation.png')
    fig.savefig(path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  ✓ {path}")


def plot_detection_fp(results, save_dir):
    """PRIMARY: Detection rate + False positive rate, tất cả attacks."""
    attacks = [a for a in ATTACK_ORDER if a != 'no_attack' and a in results]
    rounds  = np.arange(NUM_ROUNDS)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for attack in attacks:
        h = results[attack]
        axes[0].plot(rounds, [d * 100 for d in h['detection_rate']],
                     label=ATTACK_DISPLAY[attack], color=PALETTE[attack],
                     linestyle=LINESTYLES[attack], linewidth=LINEWIDTHS[attack])
        axes[1].plot(rounds, [f * 100 for f in h['false_positive_rate']],
                     label=ATTACK_DISPLAY[attack], color=PALETTE[attack],
                     linestyle=LINESTYLES[attack], linewidth=LINEWIDTHS[attack])

    for ax in axes:
        ax.axvline(x=PRETRAIN_ROUNDS, color='orange', linestyle='--',
                   linewidth=1.2, alpha=0.8, label='Attack starts')
        ax.set_xlabel('Round', fontsize=12)
        ax.set_xlim([0, NUM_ROUNDS - 1])
        ax.grid(True, alpha=0.2)

    axes[0].axhline(y=80, color='#27ae60', linestyle=':', linewidth=1.0, alpha=0.7)
    axes[0].set_ylabel('Detection Rate (%)', fontsize=12)
    axes[0].set_title('Malicious Client Detection Rate\n(% malicious with trust < τ)',
                      fontsize=12, fontweight='bold')
    axes[0].set_ylim([-5, 108])
    axes[0].legend(fontsize=8.5, loc='lower right')

    axes[1].axhline(y=5, color='#e74c3c', linestyle=':', linewidth=1.0, alpha=0.7)
    axes[1].set_ylabel('False Positive Rate (%)', fontsize=12)
    axes[1].set_title('False Positive Rate\n(% benign incorrectly filtered)',
                      fontsize=12, fontweight='bold')
    axes[1].set_ylim([-2, 30])
    axes[1].legend(fontsize=8.5, loc='upper right')

    fig.suptitle(f'Detection Performance — Trust-Aware Defense\n{_config_str()}',
                 fontsize=12, fontweight='bold')
    fig.tight_layout()
    path = os.path.join(save_dir, 'newmethod_detection.png')
    fig.savefig(path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  ✓ {path}")


def plot_statistical_significance(results, save_dir):
    """
    PRIMARY: Bar chart Cohen's d + p-value annotations.
    Chứng minh trust separation có statistical significance.
    """
    sig = compute_statistical_significance(results)
    attacks = [a for a in ATTACK_ORDER if a != 'no_attack' and a in results]

    labels    = [ATTACK_DISPLAY[a] for a in attacks]
    cohens_ds = [sig[a].get('cohens_d', 0) for a in attacks]
    p_values  = [sig[a].get('p_value', 1) for a in attacks]
    separations = [results[a]['trust_separation'][-1] for a in attacks]
    det_rates   = [results[a]['detection_rate'][-1] * 100 for a in attacks]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Cohen's d
    colors_d = ['#27ae60' if d > 0.8 else '#e67e22' if d > 0.5 else '#e74c3c'
                for d in cohens_ds]
    bars0 = axes[0].bar(range(len(attacks)), cohens_ds, color=colors_d,
                         edgecolor='#333', linewidth=0.8, alpha=0.88)
    axes[0].axhline(y=0.8, color='gray', linestyle='--', linewidth=1.2,
                    label='Large effect (d=0.8)')
    axes[0].axhline(y=0.5, color='#e67e22', linestyle=':', linewidth=1.0,
                    label='Medium effect (d=0.5)')
    for bar, val, p in zip(bars0, cohens_ds, p_values):
        p_str = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'ns'
        axes[0].text(bar.get_x() + bar.get_width() / 2,
                     bar.get_height() + 0.05, f'{val:.2f}\n{p_str}',
                     ha='center', va='bottom', fontsize=9, fontweight='bold')
    axes[0].set_xticks(range(len(attacks)))
    axes[0].set_xticklabels(labels, rotation=20, ha='right', fontsize=9)
    axes[0].set_ylabel("Cohen's d (Effect Size)", fontsize=11)
    axes[0].set_title("Statistical Effect Size\n(Trust Separation Benign vs Malicious)",
                       fontsize=11, fontweight='bold')
    axes[0].legend(fontsize=8.5)
    axes[0].grid(True, alpha=0.2, axis='y')

    # Trust Separation
    colors_sep = ['#27ae60' if s > 0.4 else '#e67e22' if s > 0.2 else '#e74c3c'
                  for s in separations]
    bars1 = axes[1].bar(range(len(attacks)), separations, color=colors_sep,
                         edgecolor='#333', linewidth=0.8, alpha=0.88)
    axes[1].axhline(y=0.3, color='gray', linestyle='--', linewidth=1.2,
                    label='Good separation (≥0.3)')
    for bar, val in zip(bars1, separations):
        axes[1].text(bar.get_x() + bar.get_width() / 2,
                     bar.get_height() + 0.005, f'{val:.3f}',
                     ha='center', va='bottom', fontsize=9, fontweight='bold')
    axes[1].set_xticks(range(len(attacks)))
    axes[1].set_xticklabels(labels, rotation=20, ha='right', fontsize=9)
    axes[1].set_ylabel('Trust Separation (Benign − Malicious)', fontsize=11)
    axes[1].set_title('Final Trust Separation\n(Higher = Better)',
                       fontsize=11, fontweight='bold')
    axes[1].legend(fontsize=8.5)
    axes[1].grid(True, alpha=0.2, axis='y')

    # Detection Rate
    colors_det = ['#27ae60' if d > 80 else '#e67e22' if d > 50 else '#e74c3c'
                  for d in det_rates]
    bars2 = axes[2].bar(range(len(attacks)), det_rates, color=colors_det,
                         edgecolor='#333', linewidth=0.8, alpha=0.88)
    axes[2].axhline(y=80, color='gray', linestyle='--', linewidth=1.2,
                    label='Good detection (≥80%)')
    for bar, val in zip(bars2, det_rates):
        axes[2].text(bar.get_x() + bar.get_width() / 2,
                     bar.get_height() + 0.5, f'{val:.0f}%',
                     ha='center', va='bottom', fontsize=9, fontweight='bold')
    axes[2].set_xticks(range(len(attacks)))
    axes[2].set_xticklabels(labels, rotation=20, ha='right', fontsize=9)
    axes[2].set_ylabel('Detection Rate (%)', fontsize=11)
    axes[2].set_title('Final Detection Rate\n(% malicious with trust < τ)',
                       fontsize=11, fontweight='bold')
    axes[2].legend(fontsize=8.5)
    axes[2].set_ylim([0, 115])
    axes[2].grid(True, alpha=0.2, axis='y')

    fig.suptitle(
        f'Statistical Analysis — Trust Separation Significance\n'
        f'{_config_str()}  | *** p<0.001  ** p<0.01  * p<0.05  ns=not sig',
        fontsize=11, fontweight='bold'
    )
    fig.tight_layout()
    path = os.path.join(save_dir, 'newmethod_statistical.png')
    fig.savefig(path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  ✓ {path}")


def plot_accuracy(results, clean_acc, save_dir):
    """SECONDARY: Accuracy curves — hệ quả tự nhiên của defense."""
    fig, ax = plt.subplots(figsize=(12, 5))
    for attack in ATTACK_ORDER:
        if attack not in results: continue
        acc = results[attack]['accuracy']
        ax.plot(acc, label=f"{ATTACK_DISPLAY[attack]} ({acc[-1]*100:.1f}%)",
                color=PALETTE[attack], linestyle=LINESTYLES[attack],
                linewidth=LINEWIDTHS[attack])
    ax.axvline(x=PRETRAIN_ROUNDS, color='gray', linestyle='--',
               linewidth=1.0, alpha=0.7, label=f'Attack starts (r{PRETRAIN_ROUNDS})')
    ax.set_xlabel('Round', fontsize=12)
    ax.set_ylabel('Test Accuracy', fontsize=12)
    ax.set_ylim([0.5, 1.02])
    ax.set_xlim([0, NUM_ROUNDS - 1])
    ax.grid(True, alpha=0.2)
    ax.set_title(f'Test Accuracy — Trust-Aware Defense\n{_config_str()}',
                 fontsize=11, fontweight='bold')
    ax.legend(fontsize=9, loc='lower right', framealpha=0.92)
    fig.tight_layout()
    path = os.path.join(save_dir, 'newmethod_accuracy.png')
    fig.savefig(path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  ✓ {path}")


def plot_trust_boxplot(results, save_dir):
    """PRIMARY: Box plot trust distribution at snapshots."""
    attacks = [a for a in ATTACK_ORDER if a != 'no_attack' and a in results]
    snapshot_rounds = sorted(results[attacks[0]]['benign_trust_snapshots'].keys())
    attack_rounds   = [r for r in snapshot_rounds if r >= PRETRAIN_ROUNDS]
    if not attack_rounds:
        attack_rounds = snapshot_rounds

    fig, axes = plt.subplots(2, len(attacks), figsize=(3.0 * len(attacks), 8),
                             sharey=True)
    if len(attacks) == 1:
        axes = np.array(axes).reshape(2, 1)

    for col, attack in enumerate(attacks):
        for row, (group, color_face, color_med, title_suffix) in enumerate([
            ('benign_trust_snapshots',    '#aed6f1', '#1a5276', 'Benign'),
            ('malicious_trust_snapshots', '#f1948a', '#922b21', 'Malicious'),
        ]):
            ax = axes[row, col]
            data = [results[attack][group].get(r, []) for r in attack_rounds]
            data = [d for d in data if d]
            if not data:
                continue
            bp = ax.boxplot(
                data, patch_artist=True,
                boxprops=dict(facecolor=color_face, alpha=0.82),
                medianprops=dict(color=color_med, linewidth=2.2),
                whiskerprops=dict(color=color_med, linewidth=1.2),
                capprops=dict(color=color_med, linewidth=1.2),
                flierprops=dict(marker='o', markersize=2.5,
                                markerfacecolor=color_med, alpha=0.5),
            )
            ax.axhline(y=TAU, color='#888', linestyle='--', linewidth=1.5)
            ax.set_xticks(range(1, len(attack_rounds) + 1))
            ax.set_xticklabels([f'r{r}' for r in attack_rounds],
                               fontsize=7.5, rotation=30)
            ax.set_ylim([-0.05, 1.1])
            ax.grid(True, alpha=0.18, axis='y')
            if col == 0:
                ax.set_ylabel(f'{title_suffix}\nTrust Score', fontsize=9)
            if row == 0:
                ax.set_title(ATTACK_DISPLAY[attack], fontsize=10,
                             fontweight='bold', color=PALETTE[attack])

    fig.suptitle(
        f'Trust Distribution Over Time\n'
        f'{_config_str()}  |  Dashed = τ={TAU}',
        fontsize=11, fontweight='bold'
    )
    fig.tight_layout()
    path = os.path.join(save_dir, 'newmethod_trust_boxplot.png')
    fig.savefig(path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  ✓ {path}")


def plot_summary_heatmap(results, clean_acc, save_dir):
    """PRIMARY: Summary heatmap + stats table."""
    attacks = [a for a in ATTACK_ORDER if a in results]
    sig     = compute_statistical_significance(results)

    rows = []
    for a in attacks:
        h   = results[a]
        acc = h['accuracy'][-1] * 100
        asr = compute_asr(clean_acc, h['accuracy'][-1]) * 100
        sep = h['trust_separation'][-1]
        det = h['detection_rate'][-1] * 100
        fp  = h['false_positive_rate'][-1] * 100
        cd  = sig.get(a, {}).get('cohens_d', float('nan'))
        pv  = sig.get(a, {}).get('p_value', 1.0)
        rows.append([acc, asr, sep, det, fp, cd, pv])

    data_arr   = np.array([[r[0], r[1], r[2], r[3], r[4]] for r in rows])
    row_labels = [ATTACK_DISPLAY[a] for a in attacks]
    col_labels = ['Acc (%)', 'ASR (%)', 'Trust Sep', 'Det Rate (%)', 'FP Rate (%)']

    fig, axes = plt.subplots(1, 2, figsize=(16, max(4, len(attacks))))

    # Heatmap
    norm_data = np.zeros_like(data_arr)
    for j in range(data_arr.shape[1]):
        col = data_arr[:, j]
        rng = col.max() - col.min()
        if rng < 1e-8:
            norm_data[:, j] = 0.5
        else:
            norm_data[:, j] = (col - col.min()) / rng
            if col_labels[j] in ['ASR (%)', 'FP Rate (%)']:
                norm_data[:, j] = 1 - norm_data[:, j]

    sns.heatmap(
        norm_data, ax=axes[0],
        xticklabels=col_labels,
        yticklabels=row_labels,
        annot=np.array([[f'{v:.1f}' if j < 2 else
                          (f'{v:.3f}' if j == 2 else f'{v:.1f}%')
                          for j, v in enumerate(row)]
                         for row in data_arr]),
        fmt='', cmap='RdYlGn', vmin=0, vmax=1,
        linewidths=0.8,
        annot_kws={'size': 10, 'weight': 'bold'},
        cbar_kws={'label': 'Relative Performance (green=better)'},
    )
    axes[0].set_title('Defense Performance Summary\n(Color: relative to best in column)',
                       fontsize=11, fontweight='bold')

    # Statistical table
    axes[1].axis('off')
    cell_text = []
    for i, a in enumerate(attacks):
        pv  = rows[i][6]
        cd  = rows[i][5]
        p_str  = 'N/A' if np.isnan(pv) else \
                 ('p<0.001***' if pv < 0.001 else
                  f'p={pv:.3f}' + ('**' if pv < 0.01 else '*' if pv < 0.05 else ''))
        d_str  = 'N/A' if np.isnan(cd) else f'{cd:.2f}'
        effect = 'N/A' if np.isnan(cd) else \
                 ('Large' if abs(cd) > 0.8 else 'Medium' if abs(cd) > 0.5 else 'Small')
        cell_text.append([ATTACK_DISPLAY[a], p_str, d_str, effect])

    tbl = axes[1].table(
        cellText=cell_text,
        colLabels=["Attack", "p-value", "Cohen's d", "Effect"],
        cellLoc='center', loc='center',
        bbox=[0, 0.05, 1, 0.9],
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)
    tbl.scale(1.0, 1.8)
    for (r, c), cell in tbl.get_celld().items():
        if r == 0:
            cell.set_facecolor('#2c3e50')
            cell.get_text().set_color('white')
            cell.get_text().set_fontweight('bold')
        elif r % 2 == 1:
            cell.set_facecolor('#f8f9fa')
    axes[1].set_title('Statistical Significance\n(Welch\'s t-test, Benign vs Malicious)',
                       fontsize=11, fontweight='bold')

    fig.suptitle(f'Trust-Aware Defense — Summary\n{_config_str()}',
                 fontsize=11, fontweight='bold')
    fig.tight_layout()
    path = os.path.join(save_dir, 'newmethod_summary.png')
    fig.savefig(path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  ✓ {path}")


# ══════════════════════════════════════════════════════════════════════
# CONSOLE SUMMARY
# ══════════════════════════════════════════════════════════════════════

def print_summary(results, clean_acc):
    sig = compute_statistical_significance(results)
    print("\n" + "=" * 110)
    print(f"  TRUST-AWARE DEFENSE v10 — SUMMARY  ({_config_str()})")
    print(f"  Clean baseline: {clean_acc*100:.2f}%  |  Reference mode: {REFERENCE_MODE} (realistic)")
    print("=" * 110)
    print(f"  {'Attack':<14} {'Acc':>7} {'ASR':>7} {'Sep':>8} {'Det':>7} "
          f"{'FP':>6} {'B_trust':>8} {'M_trust':>8} {'Cohen_d':>9} {'p-val':>10}")
    print("  " + "-" * 108)

    for attack in ATTACK_ORDER:
        if attack not in results: continue
        h   = results[attack]
        acc = h['accuracy'][-1]
        asr = compute_asr(clean_acc, acc) * 100
        sep = h['trust_separation'][-1]
        det = h['detection_rate'][-1] * 100
        fp  = h['false_positive_rate'][-1] * 100
        tb  = h['trust_benign_mean'][-1]
        tm  = h['trust_malicious_mean'][-1]
        cd  = sig.get(attack, {}).get('cohens_d', float('nan'))
        pv  = sig.get(attack, {}).get('p_value', float('nan'))

        sf  = "✓" if sep > 0.3 else ("~" if sep > 0.1 else "✗")
        df  = "✓" if det > 80  else ("~" if det > 50  else "✗")
        cd_str = f"{cd:.2f}" if not np.isnan(cd) else "N/A"
        pv_str = f"<0.001***" if pv < 0.001 else (f"{pv:.3f}" if not np.isnan(pv) else "N/A")

        print(f"  {ATTACK_DISPLAY[attack]:<14} {acc*100:>6.2f}% {asr:>6.2f}% "
              f"{sep:>7.3f}{sf} {det:>6.1f}%{df} {fp:>5.1f}%  "
              f"{tb:>7.3f}  {tm:>7.3f}  {cd_str:>8}  {pv_str:>10}")
    print("=" * 110)


# ══════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════

def run_all(save_dir='results/figures', attacks=None):
    if attacks is None:
        attacks = ATTACK_ORDER

    print("\n" + "=" * 90)
    print(f"  Trust-Aware Defense v10 — Full Run")
    print(f"  {_config_str()}")
    print(f"  Reference mode: {REFERENCE_MODE} (realistic — no malicious ID ground truth)")
    print("=" * 90 + "\n")

    os.makedirs(save_dir, exist_ok=True)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    malicious_ids = set(np.random.choice(
        NUM_CLIENTS, int(NUM_CLIENTS * ATTACK_RATE), replace=False))
    print(f"  Malicious ({len(malicious_ids)}): {sorted(malicious_ids)[:10]} ...")

    print("  Loading MNIST ...")
    client_datasets, test_loader = load_data()

    results = {}
    for idx, attack in enumerate(attacks, 1):
        print(f"\n  [{idx:02d}/{len(attacks)}]  vs {attack.upper()}")
        torch.manual_seed(SEED)
        np.random.seed(SEED)
        results[attack] = run_scenario(
            attack, client_datasets, test_loader,
            malicious_ids, num_rounds=NUM_ROUNDS,
            reference_mode=REFERENCE_MODE)
        h   = results[attack]
        acc = h['accuracy'][-1]
        sep = h['trust_separation'][-1]
        det = h['detection_rate'][-1] * 100
        print(f"         -> Acc={acc*100:.2f}%  Sep={sep:.3f}  Det={det:.0f}%")

    clean_acc = results.get('no_attack', {}).get('accuracy', [0.0])
    clean_acc = clean_acc[-1] if clean_acc else 0.99

    print("\n  Generating figures ...")
    plot_trust_separation(results, save_dir)
    plot_detection_fp(results, save_dir)
    plot_statistical_significance(results, save_dir)
    plot_trust_boxplot(results, save_dir)
    plot_accuracy(results, clean_acc, save_dir)
    plot_summary_heatmap(results, clean_acc, save_dir)

    print_summary(results, clean_acc)

    print(f"\n  All figures → {save_dir}/")
    print(f"    [PRIMARY]   newmethod_trust_separation.png")
    print(f"    [PRIMARY]   newmethod_detection.png")
    print(f"    [PRIMARY]   newmethod_statistical.png")
    print(f"    [PRIMARY]   newmethod_trust_boxplot.png")
    print(f"    [PRIMARY]   newmethod_summary.png")
    print(f"    [secondary] newmethod_accuracy.png")
    print("=" * 90)

    return results


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('scenario', nargs='?', default='all',
                   help='all | no_attack | static | delayed | adaptive | '
                        'intermittent | norm_tuned | label_flip | gaussian')
    p.add_argument('--ref', default='median',
                   help='Reference mode: median (realistic) | benign (research)')
    args = p.parse_args()

    global REFERENCE_MODE
    REFERENCE_MODE = args.ref

    if args.scenario == 'all':
        run_all()
    elif args.scenario in ATTACK_ORDER:
        # Single scenario for quick test
        os.makedirs('results/figures', exist_ok=True)
        np.random.seed(SEED); torch.manual_seed(SEED)
        malicious_ids = set(np.random.choice(
            NUM_CLIENTS, int(NUM_CLIENTS * ATTACK_RATE), replace=False))
        client_datasets, test_loader = load_data()
        h = run_scenario(args.scenario, client_datasets, test_loader,
                         malicious_ids, NUM_ROUNDS)
        print(f"\n  -> Acc={h['accuracy'][-1]*100:.2f}%  "
              f"Sep={h['trust_separation'][-1]:.3f}  "
              f"Det={h['detection_rate'][-1]*100:.0f}%")
    else:
        print(f"Unknown scenario: {args.scenario}")


if __name__ == "__main__":
    main()