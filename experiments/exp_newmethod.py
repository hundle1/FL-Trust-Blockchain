"""
Experiment: Trust-Aware Defense — v9 TRUST-FOCUSED

Mục tiêu paper: Chứng minh trust mechanism PHÂN LOẠI được client độc hại.
  - Primary metric: Trust Separation (benign vs malicious)
  - Primary metric: Detection Rate (% malicious bị filter)
  - Primary metric: Trust Evolution over rounds
  - Secondary metric: Accuracy (hệ quả tự nhiên của defense tốt)

Giữ clip norm để tất cả 6 attack types đều có accuracy cao,
từ đó focus vào phân tích trust mechanism là contribution chính.

Config v9: giữ nguyên v8 (đã validated 98.87% với static)
  ENABLE_NORM_CLIP = True  ← bật lại
  Thêm metrics: detection_rate, trust_separation_history, filtered_per_round
"""

import sys
import os
os.environ['CUDA_MODULE_LOADING'] = 'LAZY'
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import numpy as np
from tqdm import tqdm
from typing import Dict, List, Optional
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

from models.cnn_mnist import get_model
from fl_core.client import FLClient
from trust.trust_score import TrustScoreManager
from fl_core.aggregation.trust_aware import TrustAwareAggregator, FedAvgAggregator
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset


# ══════════════════════════════════════════════════════════════════════
# DEVICE
# ══════════════════════════════════════════════════════════════════════

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if DEVICE.type == "cuda":
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32       = True
    torch.backends.cudnn.benchmark        = True
print(f"Device: {DEVICE}"
      + (f"  [{torch.cuda.get_device_name(0)}]" if DEVICE.type == "cuda" else ""))


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

ALPHA         = 0.9
TAU           = 0.3
INITIAL_TRUST = 1.0
IDLE_DECAY    = 0.002
WINDOW_SIZE   = 10
SIM_WEIGHT    = 0.7
WARMUP_ROUNDS = 0

NORM_PENALTY_THRESHOLD  = 3.0
NORM_PENALTY_STRENGTH   = 0.80
ABSOLUTE_NORM_THRESHOLD = 15.0

ENABLE_NORM_CLIP = True   # giữ clip norm
CLIP_MULTIPLIER  = 2.0

ATTACK_ORDER = ["no_attack", "static", "delayed", "adaptive", "intermittent", "norm_tuned"]

ATTACK_DISPLAY = {
    "no_attack":    "No Attack",
    "static":       "Static",
    "delayed":      "Delayed",
    "adaptive":     "Adaptive",
    "intermittent": "Intermittent",
    "norm_tuned":   "Norm-Tuned",
}

PALETTE = {
    "no_attack":    "#2ecc71",
    "static":       "#e74c3c",
    "delayed":      "#e67e22",
    "adaptive":     "#8e44ad",
    "intermittent": "#2980b9",
    "norm_tuned":   "#c0392b",
}

LINESTYLES = {
    "no_attack":    (0, ()),
    "static":       (0, (6, 2)),
    "delayed":      (0, (2, 2)),
    "adaptive":     (0, (6, 2, 2, 2)),
    "intermittent": (0, (4, 1, 1, 1)),
    "norm_tuned":   (0, (1, 1)),
}

LINEWIDTHS = {
    "no_attack": 2.8, "static": 2.2, "delayed": 2.0,
    "adaptive": 2.0, "intermittent": 1.8, "norm_tuned": 2.0,
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
    train_ds = datasets.MNIST(data_path, train=True,  download=True, transform=transform)
    test_ds  = datasets.MNIST(data_path, train=False, download=True, transform=transform)
    spc = len(train_ds) // num_clients
    client_datasets = [
        DataLoader(Subset(train_ds, list(range(i*spc, (i+1)*spc))),
                   batch_size=32, shuffle=True)
        for i in range(num_clients)
    ]
    return client_datasets, DataLoader(test_ds, batch_size=1000, shuffle=False)


# ══════════════════════════════════════════════════════════════════════
# ATTACKER
# ══════════════════════════════════════════════════════════════════════

class Attacker:
    def __init__(self, client_id, attack_type="static",
                 poisoning_scale=POISONING_SCALE, delay_rounds=10,
                 trust_threshold=0.55, dormant_threshold=0.25,
                 recovery_rounds=5, attack_probability=0.5,
                 target_norm_ratio=1.5):
        self.client_id          = client_id
        self.attack_type        = attack_type
        self.poisoning_scale    = poisoning_scale
        self.delay_rounds       = delay_rounds
        self.trust_threshold    = trust_threshold
        self.dormant_threshold  = dormant_threshold
        self.recovery_rounds    = recovery_rounds
        self.attack_probability = attack_probability
        self.target_norm_ratio  = target_norm_ratio
        self.dormant            = False
        self.dormant_since      = -1
        self._benign_norm       = 4.23

    def should_attack(self, round_num, est_trust=1.0):
        effective_round = round_num - PRETRAIN_ROUNDS
        if self.attack_type == "no_attack":     return False
        elif self.attack_type == "static":      return True
        elif self.attack_type == "delayed":     return effective_round >= self.delay_rounds
        elif self.attack_type == "adaptive":    return self._adaptive(round_num, est_trust)
        elif self.attack_type == "intermittent":return np.random.random() < self.attack_probability
        elif self.attack_type == "norm_tuned":  return True
        return True

    def _adaptive(self, round_num, trust):
        if trust < self.dormant_threshold:
            if not self.dormant:
                self.dormant = True; self.dormant_since = round_num
            return False
        if self.dormant:
            if round_num - self.dormant_since < self.recovery_rounds: return False
            self.dormant = False
        return trust > self.trust_threshold

    def poison_gradient(self, clean_gradient, benign_updates=None):
        if self.attack_type == "norm_tuned":
            return self._norm_tuned(clean_gradient, benign_updates)
        return self._flip(clean_gradient)

    def _flip(self, gradient):
        param_names = list(gradient.keys())
        n = len(param_names)
        return {
            name: -self.poisoning_scale * (1.0 + 1.5*(i/max(1,n-1))) * gradient[name]
            for i, name in enumerate(param_names)
        }

    def _norm_tuned(self, gradient, benign_updates=None):
        poisoned = {name: -p for name, p in gradient.items()}
        if benign_updates:
            norms = [float(np.sqrt(sum(torch.norm(p).item()**2 for p in u.values())))
                     for u in benign_updates]
            self._benign_norm = float(np.mean(norms))
        pnorm = float(np.sqrt(sum(torch.norm(p).item()**2 for p in poisoned.values())))
        target = self.target_norm_ratio * self._benign_norm
        if pnorm > 1e-6:
            poisoned = {k: (target/pnorm)*v for k,v in poisoned.items()}
        return poisoned


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
    return correct/total, loss_sum/len(test_loader)


# ══════════════════════════════════════════════════════════════════════
# SCENARIO RUNNER — v9: track trust metrics chi tiết
# ══════════════════════════════════════════════════════════════════════

def run_scenario(attack_type, client_datasets, test_loader,
                 malicious_ids, num_rounds=NUM_ROUNDS):

    model = get_model().to(DEVICE)

    trust_manager = TrustScoreManager(
        num_clients             = NUM_CLIENTS,
        alpha                   = ALPHA,
        tau                     = TAU,
        initial_trust           = INITIAL_TRUST,
        enable_decay            = True,
        decay_strategy          = "exponential",
        window_size             = WINDOW_SIZE,
        similarity_weight       = SIM_WEIGHT,
        idle_decay_rate         = IDLE_DECAY,
        enable_norm_penalty     = True,
        norm_penalty_threshold  = NORM_PENALTY_THRESHOLD,
        norm_penalty_strength   = NORM_PENALTY_STRENGTH,
        absolute_norm_threshold = ABSOLUTE_NORM_THRESHOLD,
        warmup_rounds           = WARMUP_ROUNDS,
    )

    aggregator = TrustAwareAggregator(
        trust_manager,
        enable_filtering       = True,
        enable_norm_clip       = ENABLE_NORM_CLIP,
        clip_multiplier        = CLIP_MULTIPLIER,
        warmup_clip_multiplier = CLIP_MULTIPLIER,
        warmup_rounds          = WARMUP_ROUNDS,
        use_median             = False,
    )

    fedavg = FedAvgAggregator()

    clients = [
        FLClient(i, get_model().to(DEVICE), client_datasets[i],
                 is_malicious=(i in malicious_ids), device=DEVICE)
        for i in range(NUM_CLIENTS)
    ]
    attackers = {cid: Attacker(cid, attack_type=attack_type) for cid in malicious_ids}
    benign_ids = [c for c in range(NUM_CLIENTS) if c not in malicious_ids]

    history = {
        'accuracy':             [],
        'loss':                 [],
        # Trust metrics — primary
        'trust_benign_mean':    [],   # mean trust của benign clients
        'trust_malicious_mean': [],   # mean trust của malicious clients
        'trust_benign_std':     [],   # std trust benign
        'trust_malicious_std':  [],   # std trust malicious
        'trust_separation':     [],   # benign_mean - malicious_mean
        'detection_rate':       [],   # % malicious dưới tau (= bị detect)
        'false_positive_rate':  [],   # % benign dưới tau (= bị filter oan)
        'clients_filtered':     [],   # số clients bị filter mỗi round
        # Per-client trust snapshot mỗi 10 rounds (cho violin plot)
        'benign_trust_snapshots':    {},  # {round: [trust scores]}
        'malicious_trust_snapshots': {},
        'attack_rate_actual':   [],
    }

    for round_num in tqdm(range(num_rounds),
                          desc=f"vs {attack_type:<12}", leave=False, ncols=90):

        is_pretrain = (round_num < PRETRAIN_ROUNDS)

        selected_idx = np.random.choice(NUM_CLIENTS, CLIENTS_PER_ROUND, replace=False)
        selected     = [clients[i] for i in selected_idx]

        global_params = {n: p.data.clone().to(DEVICE) for n,p in model.named_parameters()}
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
                if attackers[cid].should_attack(round_num, est_trust):
                    benign_upds = [updates[j] for j, pc in enumerate(client_ids[:-1])
                                   if pc not in malicious_ids]
                    updates.append(attackers[cid].poison_gradient(
                        raw_update, benign_updates=benign_upds or None))
                    attacks_this_round += 1
                else:
                    updates.append(raw_update)
            else:
                updates.append(raw_update)
            metrics_list.append(real_metrics)

        history['attack_rate_actual'].append(attacks_this_round / CLIENTS_PER_ROUND)

        if is_pretrain:
            agg = fedavg.aggregate(updates, client_ids, metrics_list)
            # Warm up trust với benign
            benign_ups_pre = [updates[i] for i,cid in enumerate(client_ids)
                              if cid not in malicious_ids]
            if benign_ups_pre:
                ref = {k: torch.mean(torch.stack([u[k] for u in benign_ups_pre]),dim=0)
                       for k in benign_ups_pre[0]}
                for i, cid in enumerate(client_ids):
                    if cid not in malicious_ids:
                        trust_manager.update_trust(cid, updates[i], ref,
                                                   metrics_list[i], round_num)
            history['clients_filtered'].append(0)
        else:
            # Reference sạch từ benign clients được chọn
            benign_sel = [cid for cid in client_ids if cid not in malicious_ids]
            if benign_sel:
                benign_ups = [updates[i] for i,cid in enumerate(client_ids)
                              if cid in benign_sel]
                ref = {k: torch.mean(torch.stack([u[k] for u in benign_ups]),dim=0)
                       for k in benign_ups[0]}
            else:
                ref = {k: torch.median(torch.stack([u[k] for u in updates]),dim=0).values
                       for k in updates[0]}

            for i, cid in enumerate(client_ids):
                trust_manager.update_trust(cid, updates[i], ref,
                                           metrics_list[i], round_num)
            trust_manager.apply_idle_decay(client_ids, round_num)

            agg = aggregator.aggregate(updates, client_ids, metrics_list,
                                       round_num=round_num)
            agg_stats = aggregator.get_stats()
            history['clients_filtered'].append(
                agg_stats['avg_filtered'] if agg_stats['total_rounds'] > 0 else 0)

        if agg is not None:
            for name, param in model.named_parameters():
                param.data += agg[name]

        # ── Trust metrics (toàn bộ 100 clients) ──────────────────────
        b_scores = [trust_manager.get_trust_score(c) for c in benign_ids]
        m_scores = [trust_manager.get_trust_score(c) for c in malicious_ids]

        b_mean = float(np.mean(b_scores))
        m_mean = float(np.mean(m_scores))
        b_std  = float(np.std(b_scores))
        m_std  = float(np.std(m_scores))

        # Detection rate: % malicious có trust < TAU
        det_rate = sum(1 for t in m_scores if t < TAU) / len(m_scores)
        # False positive: % benign có trust < TAU (bị filter oan)
        fp_rate  = sum(1 for t in b_scores if t < TAU) / len(b_scores)

        history['trust_benign_mean'].append(b_mean)
        history['trust_malicious_mean'].append(m_mean)
        history['trust_benign_std'].append(b_std)
        history['trust_malicious_std'].append(m_std)
        history['trust_separation'].append(b_mean - m_mean)
        history['detection_rate'].append(det_rate)
        history['false_positive_rate'].append(fp_rate)

        # Snapshot mỗi 10 rounds cho violin/box plot
        if round_num % 10 == 0 or round_num == num_rounds - 1:
            history['benign_trust_snapshots'][round_num]    = b_scores.copy()
            history['malicious_trust_snapshots'][round_num] = m_scores.copy()

        acc, loss = evaluate(model, test_loader)
        history['accuracy'].append(acc)
        history['loss'].append(loss)

        # Debug
        if round_num < PRETRAIN_ROUNDS + 3 or round_num % 10 == 0:
            tag = "[PRE]" if is_pretrain else "     "
            filt = history['clients_filtered'][-1]
            print(f"  [r{round_num:02d}]{tag} acc={acc*100:.1f}%  "
                  f"B={b_mean:.3f}±{b_std:.3f}  M={m_mean:.3f}±{m_std:.3f}  "
                  f"sep={b_mean-m_mean:.3f}  det={det_rate*100:.0f}%  "
                  f"fp={fp_rate*100:.0f}%  filt={filt:.1f}")

    return history


# ══════════════════════════════════════════════════════════════════════
# METRICS
# ══════════════════════════════════════════════════════════════════════

def compute_asr(clean_acc, poisoned_acc, baseline=0.1):
    if clean_acc <= baseline: return 0.0
    return float(np.clip((clean_acc-poisoned_acc)/(clean_acc-baseline), 0, 1))


def _config_str():
    return (f"α={ALPHA}, τ={TAU}, pretrain={PRETRAIN_ROUNDS}r, "
            f"clip={CLIP_MULTIPLIER}x, n={NUM_CLIENTS}, "
            f"attack_rate={ATTACK_RATE*100:.0f}%")


# ══════════════════════════════════════════════════════════════════════
# PLOTS — Trust-focused
# ══════════════════════════════════════════════════════════════════════

def plot_trust_separation(results, save_dir):
    """
    PRIMARY PLOT 1: Trust separation over time cho tất cả attack types.
    Cho thấy trust mechanism phân biệt được benign vs malicious.
    """
    attacks = [a for a in ATTACK_ORDER if a != 'no_attack' and a in results]
    ncols = 3; nrows = (len(attacks) + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(15, 4.5*nrows),
                              sharex=True, sharey=True)
    axes_flat = axes.flatten() if nrows > 1 else list(axes)
    rounds = np.arange(NUM_ROUNDS)

    for i, attack in enumerate(attacks):
        ax = axes_flat[i]
        h  = results[attack]

        b_mean = np.array(h['trust_benign_mean'])
        m_mean = np.array(h['trust_malicious_mean'])
        b_std  = np.array(h['trust_benign_std'])
        m_std  = np.array(h['trust_malicious_std'])

        # Vẽ mean ± std
        ax.plot(rounds, b_mean, color='#2980b9', linewidth=2.2, label='Benign (mean)')
        ax.fill_between(rounds, b_mean - b_std, b_mean + b_std,
                        alpha=0.18, color='#2980b9')

        ax.plot(rounds, m_mean, color='#e74c3c', linewidth=2.2,
                linestyle='--', label='Malicious (mean)')
        ax.fill_between(rounds, m_mean - m_std, m_mean + m_std,
                        alpha=0.18, color='#e74c3c')

        # Tau threshold
        ax.axhline(y=TAU, color='#888', linestyle=':', linewidth=1.5,
                   label=f'τ={TAU} (filter threshold)')
        # Pretrain boundary
        ax.axvline(x=PRETRAIN_ROUNDS, color='orange', linestyle='--',
                   linewidth=1.2, alpha=0.8, label='Attack starts')

        # Shade separation area
        ax.fill_between(rounds, m_mean, b_mean,
                        where=(b_mean > m_mean),
                        alpha=0.08, color='#27ae60', label='Separation gap')

        # Final separation annotation
        final_sep = b_mean[-1] - m_mean[-1]
        final_det = h['detection_rate'][-1] * 100
        ax.text(0.97, 0.05,
                f'sep={final_sep:.3f}\ndet={final_det:.0f}%',
                transform=ax.transAxes, ha='right', va='bottom',
                fontsize=9, color='#2c3e50',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                          alpha=0.8, edgecolor='#bdc3c7'))

        ax.set_title(ATTACK_DISPLAY[attack], fontsize=13, fontweight='bold',
                     color=PALETTE[attack])
        ax.set_ylim([-0.05, 1.08])
        ax.grid(True, alpha=0.2)
        if i % ncols == 0:   ax.set_ylabel('Trust Score', fontsize=11)
        if i >= (nrows-1)*ncols: ax.set_xlabel('Round', fontsize=11)
        if i == 0:           ax.legend(fontsize=8.5, loc='upper right')

    for j in range(i+1, len(axes_flat)):
        axes_flat[j].set_visible(False)

    fig.suptitle(
        f'Trust Score Evolution — Benign vs Malicious Clients\n'
        f'({_config_str()})  Shaded = ±1 std dev',
        fontsize=13, fontweight='bold', y=1.01
    )
    fig.tight_layout()
    path = os.path.join(save_dir, 'exp_newmethod_trust_separation.png')
    fig.savefig(path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  ✓ {path}")


def plot_detection_rate(results, save_dir):
    """
    PRIMARY PLOT 2: Detection rate + False positive rate over time.
    Cho thấy trust mechanism detect được attacker mà không filter nhầm benign.
    """
    attacks = [a for a in ATTACK_ORDER if a != 'no_attack' and a in results]
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    rounds = np.arange(NUM_ROUNDS)

    # Detection rate
    for attack in attacks:
        h = results[attack]
        axes[0].plot(rounds, [d*100 for d in h['detection_rate']],
                     label=ATTACK_DISPLAY[attack],
                     color=PALETTE[attack],
                     linestyle=LINESTYLES[attack],
                     linewidth=LINEWIDTHS[attack])

    axes[0].axvline(x=PRETRAIN_ROUNDS, color='orange', linestyle='--',
                    linewidth=1.2, alpha=0.8, label='Attack starts')
    axes[0].axhline(y=100, color='#27ae60', linestyle=':', linewidth=1.0, alpha=0.5)
    axes[0].set_xlabel('Round', fontsize=12)
    axes[0].set_ylabel('Detection Rate (%)', fontsize=12)
    axes[0].set_title('Malicious Client Detection Rate\n(% malicious clients with trust < τ)',
                      fontsize=12, fontweight='bold')
    axes[0].set_ylim([-5, 108])
    axes[0].set_xlim([0, NUM_ROUNDS-1])
    axes[0].legend(fontsize=9, loc='upper left')
    axes[0].grid(True, alpha=0.2)

    # False positive rate
    for attack in attacks:
        h = results[attack]
        axes[1].plot(rounds, [f*100 for f in h['false_positive_rate']],
                     label=ATTACK_DISPLAY[attack],
                     color=PALETTE[attack],
                     linestyle=LINESTYLES[attack],
                     linewidth=LINEWIDTHS[attack])

    axes[1].axvline(x=PRETRAIN_ROUNDS, color='orange', linestyle='--',
                    linewidth=1.2, alpha=0.8, label='Attack starts')
    axes[1].axhline(y=0, color='#27ae60', linestyle=':', linewidth=1.0, alpha=0.5)
    axes[1].set_xlabel('Round', fontsize=12)
    axes[1].set_ylabel('False Positive Rate (%)', fontsize=12)
    axes[1].set_title('False Positive Rate\n(% benign clients incorrectly filtered)',
                      fontsize=12, fontweight='bold')
    axes[1].set_ylim([-2, 30])
    axes[1].set_xlim([0, NUM_ROUNDS-1])
    axes[1].legend(fontsize=9, loc='upper right')
    axes[1].grid(True, alpha=0.2)

    fig.suptitle(
        f'Detection Performance — Trust-Aware Defense\n({_config_str()})',
        fontsize=13, fontweight='bold'
    )
    fig.tight_layout()
    path = os.path.join(save_dir, 'exp_newmethod_detection.png')
    fig.savefig(path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  ✓ {path}")


def plot_trust_heatmap(results, save_dir):
    """
    PRIMARY PLOT 3: Heatmap final trust scores + separation summary.
    """
    attacks = [a for a in ATTACK_ORDER if a in results]
    fb  = [results[a]['trust_benign_mean'][-1]    for a in attacks]
    fm  = [results[a]['trust_malicious_mean'][-1] for a in attacks]
    sep = [b - m for b, m in zip(fb, fm)]
    det = [results[a]['detection_rate'][-1]*100   for a in attacks]
    fp  = [results[a]['false_positive_rate'][-1]*100 for a in attacks]

    data = np.array([fb, fm])

    fig, axes = plt.subplots(1, 3, figsize=(16, 5),
                             gridspec_kw={'width_ratios': [3, 1.2, 1.2]})

    # Heatmap
    im = sns.heatmap(
        data, ax=axes[0],
        xticklabels=[ATTACK_DISPLAY[a] for a in attacks],
        yticklabels=['Benign', 'Malicious'],
        annot=True, fmt='.3f', cmap='RdYlGn', vmin=0, vmax=1,
        cbar_kws={'label': 'Final Trust Score'},
        linewidths=1.5, annot_kws={'size': 14, 'weight': 'bold'},
    )
    axes[0].set_title(
        f'Final Trust Score — Benign vs Malicious\n'
        f'α={ALPHA}, τ={TAU}, attack_rate={ATTACK_RATE*100:.0f}%',
        fontsize=12, fontweight='bold'
    )
    axes[0].axhline(y=1, color='white', linewidth=3)
    axes[0].tick_params(axis='x', rotation=15)

    # Trust separation bar
    colors_sep = ['#27ae60' if s > 0.4 else '#e67e22' if s > 0.2 else '#e74c3c'
                  for s in sep]
    bars = axes[1].barh(
        [ATTACK_DISPLAY[a] for a in attacks], sep,
        color=colors_sep, edgecolor='#444', linewidth=0.8, alpha=0.88
    )
    for bar, val in zip(bars, sep):
        axes[1].text(val + 0.01, bar.get_y() + bar.get_height()/2,
                     f'{val:.3f}', va='center', fontsize=10, fontweight='bold')
    axes[1].axvline(x=0.3, color='gray', linestyle='--', linewidth=1.2,
                    alpha=0.7, label='Good (≥0.3)')
    axes[1].set_xlabel('Trust Separation\n(Benign − Malicious)', fontsize=10)
    axes[1].set_title('Separation', fontsize=12, fontweight='bold')
    axes[1].set_xlim([0, 1.0])
    axes[1].legend(fontsize=8)
    axes[1].grid(True, alpha=0.2, axis='x')

    # Detection rate bar
    colors_det = ['#27ae60' if d > 80 else '#e67e22' if d > 50 else '#e74c3c'
                  for d in det]
    bars2 = axes[2].barh(
        [ATTACK_DISPLAY[a] for a in attacks], det,
        color=colors_det, edgecolor='#444', linewidth=0.8, alpha=0.88
    )
    for bar, val, fval in zip(bars2, det, fp):
        axes[2].text(val + 0.5, bar.get_y() + bar.get_height()/2,
                     f'{val:.0f}%\n(FP:{fval:.0f}%)',
                     va='center', fontsize=8.5, fontweight='bold')
    axes[2].axvline(x=80, color='gray', linestyle='--', linewidth=1.2,
                    alpha=0.7, label='Good (≥80%)')
    axes[2].set_xlabel('Detection Rate (%)', fontsize=10)
    axes[2].set_title('Detection Rate\n(FP = false positive)', fontsize=12, fontweight='bold')
    axes[2].set_xlim([0, 115])
    axes[2].legend(fontsize=8)
    axes[2].grid(True, alpha=0.2, axis='x')

    fig.suptitle(
        f'Trust-Aware Defense — Final State Summary\n({_config_str()})',
        fontsize=13, fontweight='bold'
    )
    fig.tight_layout()
    path = os.path.join(save_dir, 'exp_newmethod_heatmap.png')
    fig.savefig(path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  ✓ {path}")


def plot_trust_distribution(results, save_dir):
    """
    PRIMARY PLOT 4: Box plot trust distribution tại các mốc thời gian.
    Cho thấy sự phân tách rõ ràng giữa benign và malicious theo thời gian.
    """
    attacks = [a for a in ATTACK_ORDER if a != 'no_attack' and a in results]
    snapshot_rounds = sorted(results[attacks[0]]['benign_trust_snapshots'].keys())

    fig, axes = plt.subplots(1, len(attacks), figsize=(3.5*len(attacks), 6),
                             sharey=True)
    if len(attacks) == 1:
        axes = [axes]

    for ax, attack in zip(axes, attacks):
        rounds_to_show = [r for r in snapshot_rounds if r >= PRETRAIN_ROUNDS]
        if not rounds_to_show:
            rounds_to_show = snapshot_rounds

        positions_b = []
        positions_m = []
        data_b      = []
        data_m      = []
        xticks      = []
        xlabels     = []

        for idx, r in enumerate(rounds_to_show):
            b_data = results[attack]['benign_trust_snapshots'].get(r, [])
            m_data = results[attack]['malicious_trust_snapshots'].get(r, [])
            if b_data and m_data:
                pos = idx * 3
                positions_b.append(pos)
                positions_m.append(pos + 1)
                data_b.append(b_data)
                data_m.append(m_data)
                xticks.append(pos + 0.5)
                xlabels.append(f'r{r}')

        if data_b:
            bp1 = ax.boxplot(data_b, positions=positions_b, widths=0.7,
                             patch_artist=True, notch=False,
                             boxprops=dict(facecolor='#aed6f1', alpha=0.8),
                             medianprops=dict(color='#1a5276', linewidth=2),
                             whiskerprops=dict(color='#2980b9'),
                             capprops=dict(color='#2980b9'),
                             flierprops=dict(marker='o', markersize=3,
                                            markerfacecolor='#2980b9', alpha=0.5))
            bp2 = ax.boxplot(data_m, positions=positions_m, widths=0.7,
                             patch_artist=True, notch=False,
                             boxprops=dict(facecolor='#f1948a', alpha=0.8),
                             medianprops=dict(color='#922b21', linewidth=2),
                             whiskerprops=dict(color='#e74c3c'),
                             capprops=dict(color='#e74c3c'),
                             flierprops=dict(marker='o', markersize=3,
                                            markerfacecolor='#e74c3c', alpha=0.5))

        ax.axhline(y=TAU, color='#888', linestyle='--', linewidth=1.5,
                   label=f'τ={TAU}')
        ax.set_xticks(xticks)
        ax.set_xticklabels(xlabels, fontsize=8, rotation=30)
        ax.set_title(ATTACK_DISPLAY[attack], fontsize=11, fontweight='bold',
                     color=PALETTE[attack])
        ax.set_ylim([-0.05, 1.1])
        ax.grid(True, alpha=0.2, axis='y')
        if ax == axes[0]:
            ax.set_ylabel('Trust Score', fontsize=11)
            from matplotlib.patches import Patch
            legend_elements = [
                Patch(facecolor='#aed6f1', label='Benign'),
                Patch(facecolor='#f1948a', label='Malicious'),
            ]
            ax.legend(handles=legend_elements, fontsize=9, loc='lower right')

    fig.suptitle(
        f'Trust Score Distribution Over Time (Box Plot)\n'
        f'({_config_str()})  Blue=Benign  Red=Malicious  Dashed=τ',
        fontsize=12, fontweight='bold'
    )
    fig.tight_layout()
    path = os.path.join(save_dir, 'exp_newmethod_trust_dist.png')
    fig.savefig(path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  ✓ {path}")


def plot_accuracy(results, clean_acc, save_dir):
    """SECONDARY PLOT: Accuracy (hệ quả tự nhiên)."""
    fig, ax = plt.subplots(figsize=(12, 5))
    for attack in ATTACK_ORDER:
        if attack not in results: continue
        acc = results[attack]['accuracy']
        ax.plot(acc, label=f"{ATTACK_DISPLAY[attack]}  ({acc[-1]*100:.1f}%)",
                color=PALETTE[attack], linestyle=LINESTYLES[attack],
                linewidth=LINEWIDTHS[attack])
    ax.axvline(x=PRETRAIN_ROUNDS, color='gray', linestyle='--',
               linewidth=1.0, alpha=0.7, label=f'Attack starts (r{PRETRAIN_ROUNDS})')
    ax.set_xlabel('Round', fontsize=12); ax.set_ylabel('Test Accuracy', fontsize=12)
    ax.set_ylim([0.5, 1.02]); ax.set_xlim([0, NUM_ROUNDS-1])
    ax.grid(True, alpha=0.2)
    ax.set_title(f'Test Accuracy (secondary metric — hệ quả của trust defense)\n({_config_str()})',
                 fontsize=10, fontweight='bold', pad=10)
    ax.legend(fontsize=9, loc='lower right', framealpha=0.92)
    fig.tight_layout()
    path = os.path.join(save_dir, 'exp_newmethod_accuracy.png')
    fig.savefig(path, dpi=300, bbox_inches='tight'); plt.close(fig)
    print(f"  ✓ {path}")


def plot_summary_table(results, clean_acc, save_dir):
    """
    PRIMARY PLOT 5: Summary table dạng heatmap — dễ đưa vào paper.
    """
    attacks = [a for a in ATTACK_ORDER if a in results]
    metrics = ['Final Acc (%)', 'ASR (%)', 'Trust Sep', 'Det Rate (%)', 'FP Rate (%)']

    data = []
    for attack in attacks:
        h   = results[attack]
        acc = h['accuracy'][-1] * 100
        asr = compute_asr(clean_acc, h['accuracy'][-1]) * 100
        sep = h['trust_separation'][-1]
        det = h['detection_rate'][-1] * 100
        fp  = h['false_positive_rate'][-1] * 100
        data.append([acc, asr, sep, det, fp])

    data_arr   = np.array(data)
    row_labels = [ATTACK_DISPLAY[a] for a in attacks]

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.axis('off')

    # Normalize per column for coloring
    col_colors = []
    for col_idx, metric in enumerate(metrics):
        col = data_arr[:, col_idx]
        norm_col = (col - col.min()) / (col.max() - col.min() + 1e-8)
        # Invert for ASR and FP (lower = better)
        if metric in ['ASR (%)', 'FP Rate (%)']:
            norm_col = 1 - norm_col
        col_colors.append(norm_col)

    cell_colors = []
    for row_idx in range(len(attacks)):
        row_c = []
        for col_idx in range(len(metrics)):
            v = col_colors[col_idx][row_idx]
            # Green→Red colormap
            r = 1 - v * 0.6
            g = 0.3 + v * 0.6
            b = 0.3
            row_c.append((r, g, b, 0.7))
        cell_colors.append(row_c)

    cell_text = []
    for row in data:
        cell_text.append([
            f'{row[0]:.2f}%',
            f'{row[1]:.2f}%',
            f'{row[2]:.3f}',
            f'{row[3]:.1f}%',
            f'{row[4]:.1f}%',
        ])

    tbl = ax.table(
        cellText=cell_text,
        rowLabels=row_labels,
        colLabels=metrics,
        cellColours=cell_colors,
        rowColours=[(*[0.9, 0.9, 0.9], 0.8)] * len(attacks),
        colColours=[(*[0.3, 0.3, 0.6], 0.8)] * len(metrics),
        cellLoc='center', loc='center'
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(11)
    tbl.scale(1.3, 2.0)

    # Header text color white
    for (row, col), cell in tbl.get_celld().items():
        if row == 0:
            cell.get_text().set_color('white')
            cell.get_text().set_fontweight('bold')

    ax.set_title(
        f'Trust-Aware Defense — Summary Metrics\n({_config_str()})',
        fontsize=13, fontweight='bold', pad=20
    )
    fig.tight_layout()
    path = os.path.join(save_dir, 'exp_newmethod_summary_table.png')
    fig.savefig(path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  ✓ {path}")


# ══════════════════════════════════════════════════════════════════════
# CONSOLE SUMMARY
# ══════════════════════════════════════════════════════════════════════

def print_summary(results, clean_acc):
    print("\n" + "="*95)
    print(f"  TRUST-AWARE DEFENSE v9 — SUMMARY  ({_config_str()})")
    print(f"  Clean baseline: {clean_acc*100:.2f}%")
    print("="*95)
    print(f"  {'Attack':<14} {'Acc':>8} {'ASR':>8} {'TrustSep':>10} "
          f"{'DetRate':>9} {'FP Rate':>9} {'B_trust':>9} {'M_trust':>9}")
    print("  " + "-"*90)
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
        sf  = "✓" if sep > 0.3 else ("~" if sep > 0.1 else "✗")
        df  = "✓" if det > 80  else ("~" if det > 50  else "✗")
        print(f"  {ATTACK_DISPLAY[attack]:<14} {acc*100:>7.2f}% "
              f"{asr:>7.2f}% {sep:>10.3f}{sf} "
              f"{det:>8.1f}%{df} {fp:>8.1f}%  "
              f"{tb:>8.3f}  {tm:>8.3f}")
    print("="*95)


# ══════════════════════════════════════════════════════════════════════
# ENTRY POINTS
# ══════════════════════════════════════════════════════════════════════

EXP_MAP = {
    'exp1':'no_attack','exp2':'static','exp3':'delayed',
    'exp4':'adaptive','exp5':'intermittent','exp6':'norm_tuned',
}

def _setup(save_dir):
    os.makedirs(save_dir, exist_ok=True)
    np.random.seed(SEED); torch.manual_seed(SEED)
    malicious_ids = set(np.random.choice(
        NUM_CLIENTS, int(NUM_CLIENTS*ATTACK_RATE), replace=False))
    print(f"  Malicious ({len(malicious_ids)}): {sorted(malicious_ids)[:10]} ...")
    print(f"  Pretrain={PRETRAIN_ROUNDS}r  Attack={NUM_ROUNDS-PRETRAIN_ROUNDS}r  "
          f"clip={'ON' if ENABLE_NORM_CLIP else 'OFF'}")
    print("  Loading MNIST ...")
    return malicious_ids, *load_data()

def run_single_exp(attack, save_dir='results/figures'):
    print("\n"+"="*85)
    print(f"  TrustAware v9 — Single Run: {attack.upper()}")
    print(f"  {_config_str()}")
    print("="*85+"\n")
    malicious_ids, client_datasets, test_loader = _setup(save_dir)
    torch.manual_seed(SEED); np.random.seed(SEED)
    history = run_scenario(attack, client_datasets, test_loader,
                           malicious_ids, num_rounds=NUM_ROUNDS)
    sep = history['trust_separation'][-1]
    det = history['detection_rate'][-1] * 100
    tb  = history['trust_benign_mean'][-1]
    tm  = history['trust_malicious_mean'][-1]
    print(f"\n  -> Acc={history['accuracy'][-1]*100:.2f}%  "
          f"Sep={sep:.3f}  Det={det:.0f}%  (B={tb:.3f}, M={tm:.3f})")
    print("="*85)

def run_all(save_dir='results/figures'):
    print("\n"+"="*85)
    print(f"  TrustAware v9 — Full Run (all attack types)")
    print(f"  {_config_str()}")
    print("="*85+"\n")
    malicious_ids, client_datasets, test_loader = _setup(save_dir)
    results = {}
    for idx, attack in enumerate(ATTACK_ORDER, 1):
        print(f"\n  [{idx:02d}/{len(ATTACK_ORDER)}]  vs {attack.upper()}")
        torch.manual_seed(SEED); np.random.seed(SEED)
        results[attack] = run_scenario(
            attack, client_datasets, test_loader,
            malicious_ids, num_rounds=NUM_ROUNDS)
        h   = results[attack]
        acc = h['accuracy'][-1]
        sep = h['trust_separation'][-1]
        det = h['detection_rate'][-1] * 100
        print(f"         -> Acc={acc*100:.2f}%  Sep={sep:.3f}  Det={det:.0f}%")

    clean_acc = results['no_attack']['accuracy'][-1]

    print("\n  Generating figures ...")
    plot_trust_separation(results, save_dir)
    plot_detection_rate(results, save_dir)
    plot_trust_heatmap(results, save_dir)
    plot_trust_distribution(results, save_dir)
    plot_accuracy(results, clean_acc, save_dir)
    plot_summary_table(results, clean_acc, save_dir)

    print_summary(results, clean_acc)
    print(f"\n  Figures saved to {save_dir}/")
    print(f"    - exp_newmethod_trust_separation.png  [PRIMARY]")
    print(f"    - exp_newmethod_detection.png          [PRIMARY]")
    print(f"    - exp_newmethod_heatmap.png            [PRIMARY]")
    print(f"    - exp_newmethod_trust_dist.png         [PRIMARY]")
    print(f"    - exp_newmethod_summary_table.png      [PRIMARY]")
    print(f"    - exp_newmethod_accuracy.png           [secondary]")
    print("="*85)

def main():
    import argparse
    valid = ['all'] + list(EXP_MAP.keys()) + list(EXP_MAP.values())
    p = argparse.ArgumentParser(description='TrustAware v9 — Trust-focused evaluation')
    p.add_argument('scenario', nargs='?', default='all',
                   choices=valid, metavar='SCENARIO')
    args = p.parse_args()
    s = args.scenario.lower().strip()
    save_dir = 'results/figures'
    if s in EXP_MAP:             run_single_exp(EXP_MAP[s], save_dir)
    elif s in EXP_MAP.values():  run_single_exp(s, save_dir)
    else:                        run_all(save_dir)

if __name__ == "__main__":
    main()