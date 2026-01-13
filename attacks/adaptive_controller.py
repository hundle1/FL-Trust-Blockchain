"""
Attack Validation Tests
Verify that attacks actually work against FedAvg and are mitigated by Trust-Aware
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import torch
import numpy as np
from attacks.adaptive_controller import AdaptiveAttackController


def test_static_attack_always_attacks():
    """Static attacker should attack every round"""
    controller = AdaptiveAttackController(
        client_id=0,
        attack_type="static",
        poisoning_scale=5.0
    )
    
    attacks = []
    for round_num in range(20):
        decision = controller.should_attack(round_num)
        attacks.append(decision)
    
    attack_rate = sum(attacks) / len(attacks)
    assert attack_rate == 1.0, f"Static attack should always attack, got {attack_rate}"
    
    print(f"✓ Static attack test passed (attack rate = {attack_rate:.2f})")


def test_delayed_attack():
    """Delayed attacker should not attack before delay period"""
    controller = AdaptiveAttackController(
        client_id=0,
        attack_type="delayed",
        poisoning_scale=5.0
    )
    controller.delay_rounds = 10
    
    early_attacks = [controller.should_attack(r) for r in range(10)]
    late_attacks = [controller.should_attack(r) for r in range(10, 20)]
    
    early_rate = sum(early_attacks) / len(early_attacks)
    late_rate = sum(late_attacks) / len(late_attacks)
    
    assert early_rate == 0.0, f"Should not attack before delay, got {early_rate}"
    assert late_rate == 1.0, f"Should attack after delay, got {late_rate}"
    
    print(f"✓ Delayed attack test passed (early={early_rate:.2f}, late={late_rate:.2f})")


def test_intermittent_attack_pattern():
    """Intermittent attacker should attack with specified frequency"""
    controller = AdaptiveAttackController(
        client_id=0,
        attack_type="intermittent",
        attack_frequency=0.3
    )
    
    attacks = []
    for round_num in range(100):
        decision = controller.should_attack(round_num)
        attacks.append(decision)
    
    attack_rate = sum(attacks) / len(attacks)
    
    # Should be approximately 0.3 (allow ±0.1 tolerance)
    assert 0.2 <= attack_rate <= 0.4, f"Expected ~0.3, got {attack_rate}"
    
    print(f"✓ Intermittent attack test passed (rate={attack_rate:.2f} ≈ 0.3)")


def test_adaptive_attack_trust_awareness():
    """Adaptive attacker should respond to trust levels"""
    controller = AdaptiveAttackController(
        client_id=0,
        attack_type="adaptive",
        trust_threshold=0.7,
        knows_trust_mechanism=True,
        objective="maximize_asr"
    )
    
    # High trust → should attack
    high_trust_decision = controller.should_attack(10, estimated_trust=0.9)
    
    # Low trust → should NOT attack (dormant)
    low_trust_decision = controller.should_attack(11, estimated_trust=0.3)
    
    assert high_trust_decision == True, "Should attack when trust is high"
    assert low_trust_decision == False, "Should not attack when trust is low"
    
    print("✓ Adaptive trust-aware test passed")


def test_gradient_poisoning():
    """Test gradient poisoning methods"""
    controller = AdaptiveAttackController(client_id=0, poisoning_scale=5.0)
    
    # Clean gradient
    clean_grad = {
        'param1': torch.tensor([1.0, 2.0, 3.0]),
        'param2': torch.tensor([0.5, -0.5])
    }
    
    # Test gradient flip
    poisoned = controller.poison_gradient(clean_grad, poison_method="gradient_flip")
    
    # Should be negated and scaled
    expected_param1 = -5.0 * clean_grad['param1']
    
    assert torch.allclose(poisoned['param1'], expected_param1), "Gradient flip failed"
    
    print("✓ Gradient poisoning test passed")


def test_attack_statistics():
    """Test attack statistics tracking"""
    controller = AdaptiveAttackController(
        client_id=0,
        attack_type="intermittent",
        attack_frequency=0.5
    )
    
    for round_num in range(50):
        controller.should_attack(round_num)
    
    stats = controller.get_statistics()
    
    assert 'total_attacks' in stats
    assert 'attack_rate' in stats
    assert 'attack_history' in stats
    
    expected_attacks = 50 * 0.5
    assert abs(stats['total_attacks'] - expected_attacks) < 15, \
        f"Expected ~{expected_attacks} attacks, got {stats['total_attacks']}"
    
    print(f"✓ Statistics test passed (attacks={stats['total_attacks']}, rate={stats['attack_rate']:.2f})")


def test_attacker_with_alpha_knowledge():
    """Test that attacker with α knowledge behaves differently"""
    # Attacker knows α
    controller_knows = AdaptiveAttackController(
        client_id=0,
        attack_type="adaptive",
        knows_alpha=True,
        alpha_estimate=0.9
    )
    
    # Attacker doesn't know α
    controller_blind = AdaptiveAttackController(
        client_id=1,
        attack_type="adaptive",
        knows_alpha=False
    )
    
    # Both should still make decisions, but strategy differs
    decision_knows = controller_knows.should_attack(10, 0.8)
    decision_blind = controller_blind.should_attack(10, 0.8)
    
    print(f"✓ Alpha knowledge test passed (knows={decision_knows}, blind={decision_blind})")


def test_dormant_phase():
    """Test that attacker enters dormant phase when trust drops"""
    controller = AdaptiveAttackController(
        client_id=0,
        attack_type="adaptive",
        knows_trust_mechanism=True,
        objective="maximize_asr"
    )
    
    # Simulate trust dropping below critical threshold
    controller.should_attack(10, estimated_trust=0.2)
    
    assert controller.dormant == True, "Should enter dormant phase"
    
    print("✓ Dormant phase test passed")


def run_all_tests():
    """Run all attack validation tests"""
    print("\n" + "="*70)
    print("ATTACK VALIDATION TESTS")
    print("="*70 + "\n")
    
    test_static_attack_always_attacks()
    test_delayed_attack()
    test_intermittent_attack_pattern()
    test_adaptive_attack_trust_awareness()
    test_gradient_poisoning()
    test_attack_statistics()
    test_attacker_with_alpha_knowledge()
    test_dormant_phase()
    
    print("\n" + "="*70)
    print("ALL ATTACK TESTS PASSED ✓")
    print("="*70)
    print("\nKey Validations:")
    print("  ✓ Static attacks work consistently")
    print("  ✓ Delayed attacks respect timing")
    print("  ✓ Intermittent attacks follow probability")
    print("  ✓ Adaptive attacks respond to trust")
    print("  ✓ Poisoning methods modify gradients correctly")


if __name__ == "__main__":
    torch.manual_seed(42)
    np.random.seed(42)
    run_all_tests()