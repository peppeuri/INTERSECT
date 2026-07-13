"""
INTERSECT — Autonomous AI Trading System
ppo_agent.py: PPO agent with LSTM, GAE, online learning, self-healing
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
from typing import Dict, List, Tuple, Optional


class PPONetwork(nn.Module):
    def __init__(self, state_dim, action_dim, lstm_hidden, mlp_hidden, dropout):
        super().__init__()
        self.lstm = nn.LSTM(state_dim, lstm_hidden, 2, batch_first=True, dropout=dropout)
        layers = []
        in_dim = lstm_hidden
        for h in mlp_hidden:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            in_dim = h
        self.mlp = nn.Sequential(*layers)
        self.actor = nn.Linear(in_dim, action_dim)
        self.critic = nn.Linear(in_dim, 1)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
            nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.LSTM):
            for name, param in module.named_parameters():
                if 'weight' in name:
                    nn.init.orthogonal_(param, gain=1.0)
                elif 'bias' in name:
                    nn.init.constant_(param, 0)

    def forward(self, x, hidden=None):
        if x.dim() == 2:
            x = x.unsqueeze(1)
        lstm_out, hidden = self.lstm(x, hidden)
        lstm_out = lstm_out[:, -1, :]
        features = self.mlp(lstm_out)
        logits = self.actor(features)
        value = self.critic(features)
        return logits, value, hidden


class PPOAgent:
    def __init__(
        self,
        state_dim: int,
        action_dim: int = 7,
        lstm_hidden: int = 128,
        mlp_hidden: List[int] = None,
        lr: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_eps: float = 0.2,
        entropy_coef: float = 0.01,
        value_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        dropout: float = 0.2,
        device: str = 'cpu',
        buffer_size: int = 5000,
        batch_size: int = 64,
        n_epochs: int = 10,
    ):
        mlp_hidden = mlp_hidden or [128, 64]
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.lr = lr
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_eps = clip_eps
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.max_grad_norm = max_grad_norm
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.device = torch.device(device)
        self.buffer_size = buffer_size

        self.policy_net = PPONetwork(state_dim, action_dim, lstm_hidden, mlp_hidden, dropout).to(self.device)
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)

        self.replay_buffer = deque(maxlen=buffer_size)
        self.recent_buffer = deque(maxlen=64)
        self.lstm_hidden = None
        self.lstm_cell = None
        self.rolling_sharpe = deque(maxlen=30)
        self.daily_returns = deque(maxlen=30)
        self.last_equity = 1000.0
        self.exploration_mode = False
        self.exploration_steps = 0
        self.total_steps = 0
        self._base_entropy_coef = entropy_coef

        self.checkpoint_dir = 'checkpoints'
        self.log_dir = 'logs'
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)

    def predict(self, state: np.ndarray, action_mask: Optional[np.ndarray] = None,
                deterministic: bool = False) -> Tuple[int, np.ndarray, float, float]:
        self.policy_net.eval()
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            hidden = (self.lstm_hidden, self.lstm_cell) if self.lstm_hidden is not None else None
            logits, value, (self.lstm_hidden, self.lstm_cell) = self.policy_net(
                state_tensor, hidden
            )

            if action_mask is not None:
                mask_tensor = torch.FloatTensor(action_mask).to(self.device)
                logits = logits + (mask_tensor - 1) * 1e8

            probs = F.softmax(logits, dim=-1)
            dist = Categorical(probs)

            if deterministic:
                action = torch.argmax(probs, dim=-1).item()
            else:
                action = dist.sample().item()

            log_prob = dist.log_prob(torch.tensor(action, device=self.device)).item()

            return action, probs.cpu().numpy()[0], value.item(), log_prob

    def store_transition(self, state, action, reward, next_state, done, log_prob, value):
        entry = {'state': state, 'action': action, 'reward': reward,
                 'next_state': next_state, 'done': done,
                 'log_prob': log_prob, 'value': value}
        self.replay_buffer.append(entry)
        self.recent_buffer.append(entry)
        self.total_steps += 1

    def update_rolling_metrics(self, equity: float):
        if self.last_equity > 0:
            ret = (equity - self.last_equity) / self.last_equity
            self.daily_returns.append(ret)
        self.last_equity = equity

        if len(self.daily_returns) >= 2:
            returns = np.array(self.daily_returns)
            sharpe = np.mean(returns) / (np.std(returns) + 1e-8) * np.sqrt(252 * 96)
            self.rolling_sharpe.append(sharpe)

            if len(self.rolling_sharpe) >= 30:
                avg_sharpe = np.mean(list(self.rolling_sharpe))
                if avg_sharpe < 0 and not self.exploration_mode:
                    self._activate_exploration_mode()

    def _activate_exploration_mode(self):
        self.exploration_mode = True
        self.exploration_steps = 0
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.lr * 3
        self.entropy_coef = self._base_entropy_coef * 2
        self.replay_buffer = deque(list(self.replay_buffer)[-100:], maxlen=self.buffer_size)
        self._log('EXPLORATION MODE ACTIVATED')

    def _deactivate_exploration_mode(self):
        if self.exploration_mode:
            self.exploration_mode = False
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = self.lr
            self.entropy_coef = self._base_entropy_coef
            self._log('EXPLORATION MODE DEACTIVATED')

    def train_online(self):
        if len(self.recent_buffer) < 16:
            return
        self.policy_net.train()
        self._ppo_update(list(self.recent_buffer))

    def train_batch(self):
        if len(self.replay_buffer) < self.batch_size:
            return
        self.policy_net.train()
        self._ppo_update(list(self.replay_buffer))

    def _ppo_update(self, batch: List[Dict]):
        states = torch.tensor(np.array([b['state'] for b in batch]), dtype=torch.float32, device=self.device)
        actions = torch.tensor(np.array([b['action'] for b in batch]), dtype=torch.long, device=self.device)
        old_log_probs = torch.tensor([b['log_prob'] for b in batch], dtype=torch.float32, device=self.device)
        rewards = torch.tensor(np.array([b['reward'] for b in batch]), dtype=torch.float32, device=self.device)
        next_states = torch.tensor(np.array([b['next_state'] for b in batch]), dtype=torch.float32, device=self.device)
        dones = torch.tensor(np.array([b['done'] for b in batch]), dtype=torch.float32, device=self.device)

        n = len(batch)

        with torch.no_grad():
            self.policy_net.eval()
            values = []
            next_vals = []
            h, c = None, None
            for i in range(n):
                if i > 0 and batch[i - 1]['done']:
                    h, c = None, None
                _, v, (h, c) = self.policy_net(states[i].unsqueeze(0), (h, c))
                _, nv, _ = self.policy_net(next_states[i].unsqueeze(0), (h, c))
                values.append(v)
                next_vals.append(nv)
            values = torch.cat(values).squeeze()
            next_values = torch.cat(next_vals).squeeze()
            advantages = self._compute_gae(rewards, values, next_values, dones)
            returns = advantages + values
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        self.policy_net.train()
        for _ in range(self.n_epochs):
            idx = torch.randperm(n)
            for start in range(0, n, self.batch_size):
                end = start + self.batch_size
                mb_idx = idx[start:end]
                mb_states = states[mb_idx]
                mb_actions = actions[mb_idx]
                mb_old_log_probs = old_log_probs[mb_idx]
                mb_advantages = advantages[mb_idx]
                mb_returns = returns[mb_idx]

                logits, mb_values, _ = self.policy_net(mb_states)
                mb_values = mb_values.squeeze()
                probs = F.softmax(logits, dim=-1)
                dist = Categorical(probs)
                new_log_probs = dist.log_prob(mb_actions)
                entropy = dist.entropy().mean()

                ratio = (new_log_probs - mb_old_log_probs).exp()
                surr1 = ratio * mb_advantages
                surr2 = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * mb_advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                value_loss = F.mse_loss(mb_values, mb_returns)
                entropy_loss = -entropy

                loss = policy_loss + self.value_coef * value_loss + self.entropy_coef * entropy_loss

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy_net.parameters(), self.max_grad_norm)
                self.optimizer.step()

        if self.exploration_mode:
            self.exploration_steps += 1
            if self.exploration_steps >= 100:
                self._deactivate_exploration_mode()

    def _compute_gae(self, rewards, values, next_values, dones):
        advantages = torch.zeros_like(rewards)
        gae = 0
        for t in reversed(range(len(rewards))):
            delta = rewards[t] + self.gamma * next_values[t] * (1 - dones[t]) - values[t]
            gae = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * gae
            advantages[t] = gae
        return advantages

    def reset_lstm(self):
        self.lstm_hidden = None
        self.lstm_cell = None

    def save_checkpoint(self, path: str):
        torch.save({
            'policy_net': self.policy_net.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'lr': self.lr,
            'entropy_coef': self.entropy_coef,
            'base_entropy_coef': self._base_entropy_coef,
            'total_steps': self.total_steps,
            'rolling_sharpe': list(self.rolling_sharpe),
            'exploration_mode': self.exploration_mode,
            'architecture': {
                'state_dim': self.state_dim,
                'action_dim': self.action_dim,
                'lstm_hidden': self.policy_net.lstm.hidden_size,
                'mlp_hidden': [m.out_features for m in self.policy_net.mlp if isinstance(m, nn.Linear)],
                'dropout': getattr(self.policy_net.lstm, 'dropout', 0.2),
            },
        }, path)
        self._log(f'Checkpoint saved: {path}')

    def load_checkpoint(self, path: str, strict: bool = True):
        if not os.path.exists(path):
            self._log(f'Checkpoint not found: {path}')
            return
        try:
            checkpoint = torch.load(path, map_location=self.device)
            if strict:
                self.policy_net.load_state_dict(checkpoint['policy_net'])
            else:
                own_state = self.policy_net.state_dict()
                for name, param in checkpoint['policy_net'].items():
                    if name in own_state and param.shape == own_state[name].shape:
                        own_state[name].copy_(param)
                self._log(f'Loaded {sum(1 for n in checkpoint["policy_net"] if n in own_state and checkpoint["policy_net"][n].shape == own_state[n].shape)}/{len(checkpoint["policy_net"])} layers from checkpoint')
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.lr = checkpoint.get('lr', self.lr)
            self._base_entropy_coef = self.entropy_coef = checkpoint.get('entropy_coef', self.entropy_coef)
            self.total_steps = checkpoint.get('total_steps', 0)
            self.rolling_sharpe = deque(checkpoint.get('rolling_sharpe', []), maxlen=30)
            self.exploration_mode = checkpoint.get('exploration_mode', False)
            self._log(f'Checkpoint loaded: {path}')
        except (KeyError, RuntimeError, ValueError) as e:
            self._log(f'Failed to load checkpoint: {e}')

    def get_metrics(self) -> Dict:
        return {
            'total_steps': self.total_steps,
            'rolling_sharpe': np.mean(list(self.rolling_sharpe)) if self.rolling_sharpe else 0,
            'entropy_coef': self.entropy_coef,
            'lr': self.optimizer.param_groups[0]['lr'],
            'exploration_mode': self.exploration_mode,
            'buffer_size': len(self.replay_buffer),
        }

    def _log(self, msg: str):
        timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
        log_entry = f'{timestamp} | PPO | {msg}\n'
        print(log_entry.strip())
        with open(os.path.join(self.log_dir, 'ppo_agent.log'), 'a') as f:
            f.write(log_entry)
