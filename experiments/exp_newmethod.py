"""
Experiment: Trust-Aware Defense vs Attackers — v8 PRETRAIN

Insight từ tất cả các version trước:
  - v5: 29% vì model frozen từ round 1 (no trusted clients → skip update mọi round)
  - v6/v7: 11% vì model collapse round 1 rồi stuck

Giải pháp: PRE-TRAIN model N rounds sạch TRƯỚC khi bật attack + defense.
  - Pretrain 10 rounds: model đạt ~90%+ accuracy
  - Sau đó bật attack + trust defense bình thường
  - Defense chỉ cần giữ model không bị kéo xuống, không cần recover từ đầu

Config: quay về gốc đơn giản, đã được validate.
  ALPHA=0.9, TAU=0.3, INITIAL_TRUST=1.0 (config gốc hoạt động tốt)
  Thêm PRETRAIN_ROUNDS=10
  Norm clip: CLIP_MULTIPLIER=2.0 (4.23*2=8.46, attacker 40→8.46 clipped)
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
# CONFIG — v8: pretrain + đơn giản hóa
# ══════════════════════════════════════════════════════════════════════

ATTACK_RATE       = 0.20
POISONING_SCALE   = 5.0
NUM_CLIENTS       = 100
CLIENTS_PER_ROUND = 20
NUM_ROUNDS        = 80     # tổng số rounds (bao gồm pretrain)
PRETRAIN_ROUNDS   = 10     # rounds train sạch trước khi bật attack
SEED              = 42

# Trust config — quay về gần gốc, đã hoạt động
ALPHA         = 0.9    # memory cao → trust thay đổi chậm → ổn định
TAU           = 0.3    # threshold thấp → benign dễ pass
INITIAL_TRUST = 1.0    # start với max trust
IDLE_DECAY    = 0.002
WINDOW_SIZE   = 10
SIM_WEIGHT    = 0.7
WARMUP_ROUNDS = 0      # không cần warmup vì đã pretrain

# Norm penalty — calibrated từ diagnostic
NORM_PENALTY_THRESHOLD  = 3.0   # bắt đầu penalty ở 3x benign median
NORM_PENALTY_STRENGTH   = 0.80
ABSOLUTE_NORM_THRESHOLD = 15.0  # benign~4.2, attacker~40

# Norm clip
ENABLE_NORM_CLIP       = True
CLIP_MULTIPLIER        = 2.0    # 4.23*2=8.46 → attacker 40→8.46

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

def load_data(num_clients: int = NUM_CLIENTS):
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
        DataLoader(Subset(train_ds, list(range(i * spc, (i + 1) * spc))),
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

    def should_attack(self, round_num: int, est_trust: float = 1.0) -> bool:
        # round_num tính từ 0 nhưng delay tính từ sau pretrain
        effective_round = round_num - PRETRAIN_ROUNDS
        if self.attack_type == "no_attack":      return False
        elif self.attack_type == "static":        return True
        elif self.attack_type == "delayed":       return effective_round >= self.delay_rounds
        elif self.attack_type == "adaptive":      return self._adaptive(round_num, est_trust)
        elif self.attack_type == "intermittent":  return np.random.random() < self.attack_probability
        elif self.attack_type == "norm_tuned":    return True
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


def _norm(update):
    return float(np.sqrt(sum(torch.norm(v).item()**2 for v in update.values())))


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
# SCENARIO RUNNER — v8: pretrain + defense
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

    history = {
        'accuracy': [], 'loss': [],
        'trust_benign': [], 'trust_malicious': [],
        'attack_rate_actual': [], 'clients_filtered': [],
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

            # Chỉ attack sau pretrain
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
            # Pretrain: FedAvg thuần, không trust, không attack
            agg = fedavg.aggregate(updates, client_ids, metrics_list)
            # Update trust với benign updates để warm up norm_history
            benign_ups = [updates[i] for i,cid in enumerate(client_ids)
                          if cid not in malicious_ids]
            if benign_ups:
                ref = {k: torch.mean(torch.stack([u[k] for u in benign_ups]),dim=0)
                       for k in benign_ups[0]}
                for i, cid in enumerate(client_ids):
                    if cid not in malicious_ids:
                        trust_manager.update_trust(cid, updates[i], ref,
                                                   metrics_list[i], round_num)
            history['clients_filtered'].append(0)
        else:
            # Attack phase: trust-aware aggregation
            # Reference: mean of benign (known in research mode)
            benign_sel = [cid for cid in client_ids if cid not in malicious_ids]
            if benign_sel:
                benign_ups = [updates[i] for i,cid in enumerate(client_ids)
                              if cid in benign_sel]
                ref = {k: torch.mean(torch.stack([u[k] for u in benign_ups]),dim=0)
                       for k in benign_ups[0]}
            else:
                # Fallback: median of all
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

        # Trust stats
        b_trust = [trust_manager.get_trust_score(c)
                   for c in range(NUM_CLIENTS) if c not in malicious_ids]
        m_trust = [trust_manager.get_trust_score(c) for c in malicious_ids]
        history['trust_benign'].append(float(np.mean(b_trust)))
        history['trust_malicious'].append(float(np.mean(m_trust)))

        acc, loss = evaluate(model, test_loader)
        history['accuracy'].append(acc)
        history['loss'].append(loss)

        # Debug
        if round_num < PRETRAIN_ROUNDS + 3 or round_num % 10 == 0:
            tb   = history['trust_benign'][-1]
            tm   = history['trust_malicious'][-1]
            filt = history['clients_filtered'][-1]
            tag  = "[PRE]" if is_pretrain else "     "
            stats = trust_manager.get_statistics()
            print(f"  [r{round_num:02d}]{tag} acc={acc*100:.1f}%  "
                  f"B={tb:.3f}  M={tm:.3f}  sep={tb-tm:.3f}  "
                  f"filt={filt:.1f}  trusted={stats['num_trusted']}/{NUM_CLIENTS}  "
                  f"norm_hist={len(trust_manager._norm_history)}")

    return history


# ══════════════════════════════════════════════════════════════════════
# METRICS + PLOTS
# ══════════════════════════════════════════════════════════════════════

def compute_asr(clean_acc, poisoned_acc, baseline=0.1):
    if clean_acc <= baseline: return 0.0
    return float(np.clip((clean_acc-poisoned_acc)/(clean_acc-baseline), 0, 1))


def _config_str():
    return (f"alpha={ALPHA}, tau={TAU}, pretrain={PRETRAIN_ROUNDS}r, "
            f"clip={CLIP_MULTIPLIER}x, abs={ABSOLUTE_NORM_THRESHOLD}, "
            f"pen={NORM_PENALTY_THRESHOLD}x/{NORM_PENALTY_STRENGTH}")


def plot_accuracy(results, clean_acc, save_dir):
    fig, ax = plt.subplots(figsize=(12, 6))
    for attack in ATTACK_ORDER:
        if attack not in results: continue
        acc = results[attack]['accuracy']
        ax.plot(acc, label=f"{ATTACK_DISPLAY[attack]}  ({acc[-1]*100:.1f}%)",
                color=PALETTE[attack], linestyle=LINESTYLES[attack],
                linewidth=LINEWIDTHS[attack])
    ax.axvline(x=PRETRAIN_ROUNDS, color='gray', linestyle='--',
               linewidth=1.0, alpha=0.7, label=f'attack starts (r{PRETRAIN_ROUNDS})')
    ax.axhline(y=TAU, color='gray', linestyle=':', linewidth=1.0, alpha=0.4)
    ax.set_xlabel('Round', fontsize=13); ax.set_ylabel('Test Accuracy', fontsize=13)
    ax.set_ylim([0, 1.08]); ax.set_xlim([0, NUM_ROUNDS - 1])
    ax.grid(True, alpha=0.2)
    ax.set_title(f"TrustAware v8 — Accuracy\n({_config_str()})",
                 fontsize=9, fontweight='bold', pad=10)
    ax.legend(fontsize=9, loc='lower right', framealpha=0.92)
    fig.tight_layout()
    path = os.path.join(save_dir, 'exp_newmethod_accuracy.png')
    fig.savefig(path, dpi=300, bbox_inches='tight'); plt.close(fig)
    print(f"  ✓ {path}")


def plot_asr(results, clean_acc, save_dir):
    attacks  = [a for a in ATTACK_ORDER if a != 'no_attack' and a in results]
    asrs     = [compute_asr(clean_acc, results[a]['accuracy'][-1])*100 for a in attacks]
    fin_accs = [results[a]['accuracy'][-1]*100 for a in attacks]
    colors   = [PALETTE[a] for a in attacks]
    x, w     = np.arange(len(attacks)), 0.38

    fig, ax = plt.subplots(figsize=(11, 6))
    bars1 = ax.bar(x-w/2, fin_accs, w, label='Final Accuracy (%)',
                   color=colors, alpha=0.85, edgecolor='#333', linewidth=0.8)
    bars2 = ax.bar(x+w/2, asrs, w, label='ASR (%)',
                   color=colors, alpha=0.45, edgecolor='#333', linewidth=0.8, hatch='//')
    for bar, val in zip(bars1, fin_accs):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.8,
                f'{val:.1f}%', ha='center', va='bottom', fontsize=9, fontweight='bold')
    for bar, val in zip(bars2, asrs):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.8,
                f'{val:.1f}%', ha='center', va='bottom',
                fontsize=9, fontweight='bold', color='#c0392b')
    ax.axhline(y=clean_acc*100, color='#2ecc71', linestyle='--',
               linewidth=1.5, alpha=0.7, label=f'No Attack ({clean_acc*100:.1f}%)')
    ax.set_xticks(x); ax.set_xticklabels([ATTACK_DISPLAY[a] for a in attacks], fontsize=12)
    ax.set_ylabel('Percentage (%)', fontsize=12); ax.set_ylim([0, 115])
    ax.grid(True, alpha=0.25, axis='y')
    ax.set_title(f"TrustAware v8 — ASR & Final Accuracy\n{_config_str()}",
                 fontsize=10, fontweight='bold', pad=10)
    ax.legend(fontsize=10, loc='upper right')
    fig.tight_layout()
    path = os.path.join(save_dir, 'exp_newmethod_asr.png')
    fig.savefig(path, dpi=300, bbox_inches='tight'); plt.close(fig)
    print(f"  ✓ {path}")


def plot_heatmap(results, save_dir):
    attacks = [a for a in ATTACK_ORDER if a in results]
    fb = [results[a]['trust_benign'][-1]    for a in attacks]
    fm = [results[a]['trust_malicious'][-1] for a in attacks]
    sep = [b-m for b,m in zip(fb,fm)]
    data = np.array([fb, fm])

    fig, axes = plt.subplots(1, 2, figsize=(14,4), gridspec_kw={'width_ratios':[3,1]})
    sns.heatmap(data, ax=axes[0],
                xticklabels=[ATTACK_DISPLAY[a] for a in attacks],
                yticklabels=['Benign','Malicious'],
                annot=True, fmt='.3f', cmap='RdYlGn', vmin=0, vmax=1,
                cbar_kws={'label':'Final Trust Score'},
                linewidths=1.0, annot_kws={'size':13,'weight':'bold'})
    axes[0].set_title(f"Final Trust\nalpha={ALPHA}, tau={TAU}, pretrain={PRETRAIN_ROUNDS}r",
                      fontsize=11, fontweight='bold')
    axes[0].axhline(y=1, color='white', linewidth=2)
    colors_sep = ['#27ae60' if s>0.3 else '#e67e22' if s>0.1 else '#e74c3c' for s in sep]
    bars = axes[1].barh([ATTACK_DISPLAY[a] for a in attacks], sep,
                        color=colors_sep, edgecolor='#333', linewidth=0.7, alpha=0.85)
    for bar, val in zip(bars, sep):
        axes[1].text(val+0.01, bar.get_y()+bar.get_height()/2,
                     f'{val:.3f}', va='center', fontsize=9, fontweight='bold')
    axes[1].axvline(x=0.3, color='gray', linestyle='--', linewidth=1.0, alpha=0.6)
    axes[1].set_xlabel('Trust Separation', fontsize=10)
    axes[1].set_title('Separation', fontsize=11, fontweight='bold')
    axes[1].set_xlim([0,1.0]); axes[1].grid(True, alpha=0.2, axis='x')
    fig.tight_layout()
    path = os.path.join(save_dir, 'exp_newmethod_heatmap.png')
    fig.savefig(path, dpi=300, bbox_inches='tight'); plt.close(fig)
    print(f"  ✓ {path}")


def plot_trust_evolution(results, save_dir):
    attacks_to_plot = [a for a in ATTACK_ORDER if a != 'no_attack' and a in results]
    ncols = 3; nrows = (len(attacks_to_plot)+ncols-1)//ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(14,4*nrows), sharex=True, sharey=True)
    axes_flat = axes.flatten() if nrows > 1 else list(axes)
    rounds = np.arange(NUM_ROUNDS)
    for i, attack in enumerate(attacks_to_plot):
        ax = axes_flat[i]
        tb = results[attack]['trust_benign']
        tm = results[attack]['trust_malicious']
        ax.plot(rounds, tb, color='#2980b9', linewidth=2.0, label='Benign avg')
        ax.plot(rounds, tm, color='#e74c3c', linewidth=2.0, linestyle='--', label='Malicious avg')
        ax.fill_between(rounds, tm, tb, alpha=0.12, color='#27ae60')
        ax.axhline(y=TAU, color='gray', linestyle=':', linewidth=1.2, alpha=0.7, label=f'tau={TAU}')
        ax.axvline(x=PRETRAIN_ROUNDS, color='orange', linestyle='--',
                   linewidth=1.0, alpha=0.8, label='attack start')
        ax.set_title(ATTACK_DISPLAY[attack], fontsize=12, fontweight='bold', color=PALETTE[attack])
        ax.set_ylim([0,1.05]); ax.grid(True, alpha=0.2)
        if i % ncols == 0: ax.set_ylabel('Trust Score', fontsize=10)
        if i >= (nrows-1)*ncols: ax.set_xlabel('Round', fontsize=10)
        if i == 0: ax.legend(fontsize=8, loc='center right')
    for j in range(i+1, len(axes_flat)):
        axes_flat[j].set_visible(False)
    fig.suptitle(f"Trust Evolution — v8 (pretrain={PRETRAIN_ROUNDS}r)\n"
                 f"alpha={ALPHA}, tau={TAU}, abs_thresh={ABSOLUTE_NORM_THRESHOLD}",
                 fontsize=11, fontweight='bold', y=1.01)
    fig.tight_layout()
    path = os.path.join(save_dir, 'exp_newmethod_trust.png')
    fig.savefig(path, dpi=300, bbox_inches='tight'); plt.close(fig)
    print(f"  ✓ {path}")


def print_summary(results, clean_acc):
    print("\n" + "="*85)
    print(f"  TRUST-AWARE DEFENSE v8 — SUMMARY")
    print(f"  {_config_str()}")
    print(f"  Clean baseline: {clean_acc*100:.2f}%")
    print("="*85)
    print(f"  {'Attack':<16} {'Final Acc':>11} {'ASR':>9} {'TrustSep':>10} {'B_trust':>9} {'M_trust':>10}")
    print("  "+"-"*75)
    for attack in ATTACK_ORDER:
        if attack not in results: continue
        acc = results[attack]['accuracy'][-1]
        asr = compute_asr(clean_acc, acc)*100
        tb  = results[attack]['trust_benign'][-1]
        tm  = results[attack]['trust_malicious'][-1]
        sep = tb-tm
        sf  = "✓" if sep>0.3 else ("~" if sep>0.1 else "✗")
        af  = "✓" if asr<20 else ("~" if asr<50 else "✗")
        print(f"  {ATTACK_DISPLAY[attack]:<16} {acc*100:>10.2f}% "
              f"{asr:>8.2f}% {sep:>10.3f}{sf} {tb:>8.3f}  {tm:>9.3f}  {af}")
    print("="*85)


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
    malicious_ids = set(np.random.choice(NUM_CLIENTS, int(NUM_CLIENTS*ATTACK_RATE), replace=False))
    print(f"  Malicious ({len(malicious_ids)}): {sorted(malicious_ids)[:10]} ...")
    print(f"  Pretrain: {PRETRAIN_ROUNDS} rounds (no attack)")
    print(f"  Attack:   rounds {PRETRAIN_ROUNDS}–{NUM_ROUNDS-1}")
    print("  Loading MNIST ...")
    return malicious_ids, *load_data()

def run_single_exp(attack, save_dir='results/figures'):
    print("\n"+"="*85)
    print(f"  TrustAware v8 — Single Run: {attack.upper()}")
    print(f"  {_config_str()}")
    print("="*85+"\n")
    malicious_ids, client_datasets, test_loader = _setup(save_dir)
    torch.manual_seed(SEED); np.random.seed(SEED)
    history = run_scenario(attack, client_datasets, test_loader,
                           malicious_ids, num_rounds=NUM_ROUNDS)
    tb=history['trust_benign'][-1]; tm=history['trust_malicious'][-1]
    print(f"\n  -> Accuracy={history['accuracy'][-1]*100:.2f}%  "
          f"TrustSep={tb-tm:.3f}  (B={tb:.3f}, M={tm:.3f})")
    print("="*85)

def run_all(save_dir='results/figures'):
    print("\n"+"="*85)
    print(f"  TrustAware v8 DEFENSE — Full Run")
    print(f"  {_config_str()}")
    print("="*85+"\n")
    malicious_ids, client_datasets, test_loader = _setup(save_dir)
    results = {}
    for idx, attack in enumerate(ATTACK_ORDER, 1):
        print(f"\n  [{idx:02d}/{len(ATTACK_ORDER)}]  vs {attack.upper()}")
        torch.manual_seed(SEED); np.random.seed(SEED)
        results[attack] = run_scenario(attack, client_datasets, test_loader,
                                       malicious_ids, num_rounds=NUM_ROUNDS)
        acc=results[attack]['accuracy'][-1]
        tb=results[attack]['trust_benign'][-1]; tm=results[attack]['trust_malicious'][-1]
        print(f"         -> Acc={acc*100:.2f}%  Sep={tb-tm:.3f} (B={tb:.3f} M={tm:.3f})")
    clean_acc = results['no_attack']['accuracy'][-1]
    print("\n  Generating figures ...")
    plot_accuracy(results, clean_acc, save_dir)
    plot_asr(results, clean_acc, save_dir)
    plot_heatmap(results, save_dir)
    plot_trust_evolution(results, save_dir)
    print_summary(results, clean_acc)
    print("="*85)

def main():
    import argparse
    valid = ['all']+list(EXP_MAP.keys())+list(EXP_MAP.values())
    p = argparse.ArgumentParser()
    p.add_argument('scenario', nargs='?', default='all', choices=valid, metavar='SCENARIO')
    args = p.parse_args()
    s = args.scenario.lower().strip()
    save_dir = 'results/figures'
    if s in EXP_MAP:            run_single_exp(EXP_MAP[s], save_dir)
    elif s in EXP_MAP.values(): run_single_exp(s, save_dir)
    else:                       run_all(save_dir)

if __name__ == "__main__":
    main()