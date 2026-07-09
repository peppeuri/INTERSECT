"""
INTERSECT — Autonomous AI Trading System
ensemble_voting.py: Sharpe-weighted ensemble voting between agents
"""

import numpy as np
from typing import Dict, Tuple


class EnsembleVoting:
    def __init__(
        self,
        min_sharpe_threshold: float = 0.2,
        boost_sharpe_threshold: float = 1.0,
        boost_size_factor: float = 1.25,
    ):
        self.min_sharpe_threshold = min_sharpe_threshold
        self.boost_sharpe_threshold = boost_sharpe_threshold
        self.boost_size_factor = boost_size_factor
        self.agent_weights = {}
        self.agent_sharpes = {}
        self.last_decision = None
        self.no_trade_count = 0

    def update_weights(self, agent_metrics: Dict[str, Dict]):
        for name, metrics in agent_metrics.items():
            self.agent_sharpes[name] = max(metrics.get('rolling_sharpe', 0), 0)
        total = sum(self.agent_sharpes.values())
        if total > 0:
            for name in self.agent_sharpes:
                self.agent_weights[name] = self.agent_sharpes[name] / total
        else:
            for name in self.agent_sharpes:
                self.agent_weights[name] = 1.0 / len(self.agent_sharpes)

    def vote(self, agent_predictions: Dict[str, Tuple[int, np.ndarray, float]]) -> Tuple[int, np.ndarray, float, Dict]:
        if not agent_predictions:
            return 0, np.ones(7) / 7, 0.0, {'action': 'no_agents', 'size_factor': 1.0}

        self.update_weights({n: {'rolling_sharpe': self.agent_sharpes.get(n, 0)} for n in agent_predictions})

        best_sharpe = max(self.agent_sharpes.values()) if self.agent_sharpes else 0

        if best_sharpe < self.min_sharpe_threshold:
            self.no_trade_count += 1
            return 0, np.ones(7) / 7, 0.0, {
                'action': 'no_trade',
                'reason': f'Best Sharpe {best_sharpe:.4f} < {self.min_sharpe_threshold}',
                'size_factor': 0.0, 'no_trade_streak': self.no_trade_count
            }
        self.no_trade_count = 0

        size_factor = self.boost_size_factor if best_sharpe > self.boost_sharpe_threshold else 1.0

        weighted = np.zeros(7)
        total_w = 0.0
        for agent_name, (action, probs, value) in agent_predictions.items():
            w = self.agent_weights.get(agent_name, 0)
            weighted += w * probs
            total_w += w

        weighted = weighted / total_w if total_w > 0 else np.ones(7) / 7
        final_action = int(np.argmax(weighted))

        info = {
            'action': 'trade', 'final_action': final_action,
            'confidence': float(weighted[final_action]), 'size_factor': size_factor,
            'weights': self.agent_weights.copy(), 'sharpes': self.agent_sharpes.copy(),
            'best_sharpe': best_sharpe,
        }
        self.last_decision = info
        return final_action, weighted, info['confidence'], info

    def get_decision_info(self) -> Dict:
        return self.last_decision or {}

    def reset(self):
        self.agent_weights = {}
        self.agent_sharpes = {}
        self.last_decision = None
        self.no_trade_count = 0
