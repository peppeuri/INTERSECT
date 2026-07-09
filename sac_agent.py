"""
INTERSECT — Autonomous AI Trading System
sac_agent.py: SAC agent with twin critics, auto entropy tuning, ensemble backup
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
import numpy as np
from collections import deque
import os
import time
import copy
from typing import Dict, List, Tuple, Optional


class SACActor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
        )
        self.logits = nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        features = self.net(x)
        logits = self.logits(features)
        return logits, features


class SACTwinCritic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim):
        super().__init__()
        self.q1 = nn.Sequential(
            nn.Linear(state_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )
        self.q2 = nn.Sequential(
            nn.Linear(state_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )

    def forward(self, x):
        return self.q1(x), self.q2(x)


class SACAgent:
    def __init__(
        self,
        state_dim: int,
        action_dim: int = 7,
        hidden_dim: int = 256,
        lr: float = 3e-4,
        gamma: float = 0.99,
        tau: float = 0.005,
        alpha: float = 0.2,
        auto_entropy_tuning: bool = True,
        target_entropy: Optional[float] = None,
        device: str = 'cpu',
        buffer_size: int = 5000,
        batch_size: int = 64,
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.tau = tau
        self.alpha = alpha
        self.auto_entropy_tuning = auto_entropy_tuning
        self.device = torch.device(device)
        self.buffer_size = buffer_size
        self.batch_size = batch_size

        self.target_entropy = target_entropy or -np.log(action_dim)

        self.actor = SACActor(state_dim, action_dim, hidden_dim).to(self.device)
        self.critic = SACTwinCritic(state_dim, action_dim, hidden_dim).to(self.device)
        self.critic_target = copy.deepcopy(self.critic).to(self.device)

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr)

        if auto_entropy_tuning:
            self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
            self.alpha_optimizer = optim.Adam([self.log_alpha], lr=lr)
        else:
            self.log_alpha = torch.log(torch.tensor(alpha, device=self.device))

        self.replay_buffer = deque(maxlen=buffer_size)
        self.rolling_sharpe = deque(maxlen=30)
        self.daily_returns = deque(maxlen=30)
        self.last_equity = 1000.0

        os.makedirs('checkpoints', exist_ok=True)
        os.makedirs('logs', exist_ok=True)

    def predict(self, state: np.ndarray, action_mask: Optional[np.ndarray] = None,
                deterministic: bool = False) -> Tuple[int, np.ndarray, float]:
        self.actor.eval()
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            logits, _ = self.actor(state_tensor)

            if action_mask is not None:
                mask_tensor = torch.FloatTensor(action_mask).to(self.device)
                logits = logits + (mask_tensor - 1) * 1e8

            probs = F.softmax(logits, dim=-1)

            if deterministic:
                action = torch.argmax(probs, dim=-1).item()
            else:
                dist = Categorical(probs)
                action = dist.sample().item()

            return action, probs.cpu().numpy()[0], 0.0

    def store_transition(self, state, action, reward, next_state, done):
        self.replay_buffer.append({
            'state': state, 'action': action, 'reward': reward,
            'next_state': next_state, 'done': done
        })

    def update_rolling_metrics(self, equity: float):
        if self.last_equity > 0:
            ret = (equity - self.last_equity) / self.last_equity
            self.daily_returns.append(ret)
        self.last_equity = equity
        if len(self.daily_returns) >= 2:
            returns = np.array(self.daily_returns)
            sharpe = np.mean(returns) / (np.std(returns) + 1e-8) * np.sqrt(252 * 96)
            self.rolling_sharpe.append(sharpe)

    def train(self):
        if len(self.replay_buffer) < self.batch_size:
            return

        self.actor.train()
        self.critic.train()

        indices = np.random.choice(len(self.replay_buffer), self.batch_size, replace=False)
        batch = [self.replay_buffer[i] for i in indices]

        states = torch.FloatTensor(np.array([b['state'] for b in batch])).to(self.device)
        actions = torch.LongTensor(np.array([b['action'] for b in batch])).to(self.device)
        rewards = torch.FloatTensor(np.array([b['reward'] for b in batch])).to(self.device)
        next_states = torch.FloatTensor(np.array([b['next_state'] for b in batch])).to(self.device)
        dones = torch.FloatTensor(np.array([b['done'] for b in batch])).to(self.device)

        with torch.no_grad():
            next_logits, _ = self.actor(next_states)
            next_probs = F.softmax(next_logits, dim=-1)
            next_log_probs = F.log_softmax(next_logits, dim=-1)
            next_q1, next_q2 = self.critic_target(next_states)
            next_q = torch.min(next_q1, next_q2)
            next_v = (next_probs * (next_q - self.alpha * next_log_probs)).sum(dim=1)
            target_q = rewards + (1 - dones) * self.gamma * next_v

        q1, q2 = self.critic(states)
        q1 = q1.gather(1, actions.unsqueeze(1)).squeeze()
        q2 = q2.gather(1, actions.unsqueeze(1)).squeeze()

        critic_loss = F.mse_loss(q1, target_q) + F.mse_loss(q2, target_q)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        logits, _ = self.actor(states)
        probs = F.softmax(logits, dim=-1)
        log_probs = F.log_softmax(logits, dim=-1)

        q1, q2 = self.critic(states)
        q = torch.min(q1, q2)

        actor_loss = (probs * (self.alpha * log_probs - q)).sum(dim=1).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        if self.auto_entropy_tuning:
            entropy = -(probs * log_probs).sum(dim=1).mean()
            alpha_loss = -(self.log_alpha * (entropy - self.target_entropy).detach()).mean()
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
            self.alpha = self.log_alpha.exp().item()

        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    def save_checkpoint(self, path: str):
        torch.save({
            'actor': self.actor.state_dict(),
            'critic': self.critic.state_dict(),
            'critic_target': self.critic_target.state_dict(),
            'actor_optimizer': self.actor_optimizer.state_dict(),
            'critic_optimizer': self.critic_optimizer.state_dict(),
            'log_alpha': self.log_alpha,
            'alpha': self.alpha,
            'rolling_sharpe': list(self.rolling_sharpe),
        }, path)

    def load_checkpoint(self, path: str):
        checkpoint = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(checkpoint['actor'])
        self.critic.load_state_dict(checkpoint['critic'])
        self.critic_target.load_state_dict(checkpoint['critic_target'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer'])
        self.log_alpha = checkpoint['log_alpha']
        self.alpha = checkpoint['alpha']
        self.rolling_sharpe = deque(checkpoint.get('rolling_sharpe', []), maxlen=30)

    def get_metrics(self) -> Dict:
        return {
            'rolling_sharpe': np.mean(list(self.rolling_sharpe)) if self.rolling_sharpe else 0,
            'alpha': self.alpha,
            'buffer_size': len(self.replay_buffer),
        }
