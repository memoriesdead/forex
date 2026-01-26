"""
Deep Q-Network (DQN) for Adaptive Position Sizing
==================================================
Source: Stable-Baselines3 (12.5k+ stars)
FinRL framework concepts (13.7k+ stars)

Key Innovation:
- Learn optimal position sizes from state (features, portfolio, risk)
- Adapt to market regimes automatically
- Balance risk/reward through experience replay
- Handle transaction costs in reward function

For HFT Forex:
- State: price features, current position, volatility regime
- Actions: position sizes (0%, 25%, 50%, 75%, 100%)
- Reward: risk-adjusted returns (Sharpe-like)
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from collections import deque
import random
import logging

logger = logging.getLogger(__name__)

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torch.optim as optim
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    logger.warning("PyTorch not available for DQN")


@dataclass
class DQNConfig:
    """DQN configuration."""
    state_dim: int = 20  # Input state dimension
    action_dim: int = 5  # Number of discrete actions (position sizes)
    hidden_dim: int = 128  # Hidden layer size
    learning_rate: float = 1e-4
    gamma: float = 0.99  # Discount factor
    epsilon_start: float = 1.0
    epsilon_end: float = 0.01
    epsilon_decay: float = 0.995
    buffer_size: int = 100000
    batch_size: int = 64
    target_update: int = 100  # Update target network every N steps
    tau: float = 0.005  # Soft update coefficient


class ReplayBuffer:
    """Experience replay buffer for DQN."""

    def __init__(self, capacity: int):
        self.buffer = deque(maxlen=capacity)

    def push(self, state: np.ndarray, action: int, reward: float,
             next_state: np.ndarray, done: bool):
        """Add experience to buffer."""
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int) -> Tuple:
        """Sample batch of experiences."""
        batch = random.sample(self.buffer, batch_size)

        states = np.array([e[0] for e in batch])
        actions = np.array([e[1] for e in batch])
        rewards = np.array([e[2] for e in batch])
        next_states = np.array([e[3] for e in batch])
        dones = np.array([e[4] for e in batch])

        return states, actions, rewards, next_states, dones

    def __len__(self) -> int:
        return len(self.buffer)


class DuelingDQN(nn.Module):
    """
    Dueling DQN Architecture.

    Separates value and advantage streams for better learning.
    Q(s,a) = V(s) + A(s,a) - mean(A(s,a))
    """

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 128):
        super().__init__()

        # Shared feature layers
        self.feature = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )

        # Value stream
        self.value_stream = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )

        # Advantage stream
        self.advantage_stream = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, action_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.feature(x)
        value = self.value_stream(features)
        advantage = self.advantage_stream(features)

        # Combine: Q = V + A - mean(A)
        q_values = value + advantage - advantage.mean(dim=1, keepdim=True)
        return q_values


class NoisyLinear(nn.Module):
    """
    Noisy Linear layer for exploration.

    Adds learnable noise to weights for exploration instead of epsilon-greedy.
    """

    def __init__(self, in_features: int, out_features: int, sigma_init: float = 0.5):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.sigma_init = sigma_init

        # Learnable parameters
        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))
        self.bias_mu = nn.Parameter(torch.empty(out_features))
        self.bias_sigma = nn.Parameter(torch.empty(out_features))

        # Factorized noise
        self.register_buffer('weight_epsilon', torch.empty(out_features, in_features))
        self.register_buffer('bias_epsilon', torch.empty(out_features))

        self.reset_parameters()
        self.reset_noise()

    def reset_parameters(self):
        mu_range = 1 / np.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.sigma_init / np.sqrt(self.in_features))
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(self.sigma_init / np.sqrt(self.out_features))

    def reset_noise(self):
        epsilon_in = self._scale_noise(self.in_features)
        epsilon_out = self._scale_noise(self.out_features)
        self.weight_epsilon.copy_(epsilon_out.outer(epsilon_in))
        self.bias_epsilon.copy_(epsilon_out)

    def _scale_noise(self, size: int) -> torch.Tensor:
        x = torch.randn(size)
        return x.sign() * x.abs().sqrt()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            weight = self.weight_mu + self.weight_sigma * self.weight_epsilon
            bias = self.bias_mu + self.bias_sigma * self.bias_epsilon
        else:
            weight = self.weight_mu
            bias = self.bias_mu
        return F.linear(x, weight, bias)


class NoisyDuelingDQN(nn.Module):
    """Dueling DQN with Noisy layers for exploration."""

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 128):
        super().__init__()

        self.feature = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )

        self.value_stream = nn.Sequential(
            NoisyLinear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            NoisyLinear(hidden_dim // 2, 1)
        )

        self.advantage_stream = nn.Sequential(
            NoisyLinear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            NoisyLinear(hidden_dim // 2, action_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.feature(x)
        value = self.value_stream(features)
        advantage = self.advantage_stream(features)
        return value + advantage - advantage.mean(dim=1, keepdim=True)

    def reset_noise(self):
        for module in self.modules():
            if isinstance(module, NoisyLinear):
                module.reset_noise()


class PrioritizedReplayBuffer:
    """
    Prioritized Experience Replay.

    Samples important transitions more frequently based on TD error.
    """

    def __init__(self, capacity: int, alpha: float = 0.6, beta: float = 0.4):
        self.capacity = capacity
        self.alpha = alpha  # Priority exponent
        self.beta = beta  # Importance sampling exponent
        self.beta_increment = 0.001

        self.buffer = []
        self.priorities = np.zeros(capacity, dtype=np.float32)
        self.position = 0
        self.max_priority = 1.0

    def push(self, state: np.ndarray, action: int, reward: float,
             next_state: np.ndarray, done: bool):
        """Add experience with max priority."""
        experience = (state, action, reward, next_state, done)

        if len(self.buffer) < self.capacity:
            self.buffer.append(experience)
        else:
            self.buffer[self.position] = experience

        self.priorities[self.position] = self.max_priority
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size: int) -> Tuple:
        """Sample batch with priorities."""
        if len(self.buffer) == self.capacity:
            priorities = self.priorities
        else:
            priorities = self.priorities[:len(self.buffer)]

        # Calculate sampling probabilities
        probs = priorities ** self.alpha
        probs /= probs.sum()

        # Sample indices
        indices = np.random.choice(len(self.buffer), batch_size, p=probs)

        # Calculate importance sampling weights
        total = len(self.buffer)
        weights = (total * probs[indices]) ** (-self.beta)
        weights /= weights.max()

        # Anneal beta
        self.beta = min(1.0, self.beta + self.beta_increment)

        # Gather experiences
        batch = [self.buffer[i] for i in indices]
        states = np.array([e[0] for e in batch])
        actions = np.array([e[1] for e in batch])
        rewards = np.array([e[2] for e in batch])
        next_states = np.array([e[3] for e in batch])
        dones = np.array([e[4] for e in batch])

        return states, actions, rewards, next_states, dones, indices, weights

    def update_priorities(self, indices: np.ndarray, priorities: np.ndarray):
        """Update priorities based on TD errors."""
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = priority + 1e-6  # Small constant to avoid zero
            self.max_priority = max(self.max_priority, priority)

    def __len__(self) -> int:
        return len(self.buffer)


class DQNPositionSizer:
    """
    DQN-based Position Sizing for Forex Trading.

    State:
    - Price features (returns, volatility, momentum)
    - Current position
    - Account metrics (drawdown, recent PnL)
    - Risk indicators

    Actions (discrete position sizes):
    - 0: 0% (no position)
    - 1: 25% of max position
    - 2: 50% of max position
    - 3: 75% of max position
    - 4: 100% of max position

    Reward:
    - Risk-adjusted return (Sharpe-like)
    - Penalty for drawdown
    - Penalty for transaction costs
    """

    # Action to position size mapping
    POSITION_SIZES = [0.0, 0.25, 0.50, 0.75, 1.0]

    def __init__(self, config: DQNConfig = None, use_prioritized: bool = True,
                 use_noisy: bool = True, use_double: bool = True):
        self.config = config or DQNConfig()
        self.use_prioritized = use_prioritized
        self.use_double = use_double

        if not HAS_TORCH:
            logger.warning("PyTorch not available")
            return

        # Networks
        if use_noisy:
            self.policy_net = NoisyDuelingDQN(
                self.config.state_dim,
                self.config.action_dim,
                self.config.hidden_dim
            )
            self.target_net = NoisyDuelingDQN(
                self.config.state_dim,
                self.config.action_dim,
                self.config.hidden_dim
            )
        else:
            self.policy_net = DuelingDQN(
                self.config.state_dim,
                self.config.action_dim,
                self.config.hidden_dim
            )
            self.target_net = DuelingDQN(
                self.config.state_dim,
                self.config.action_dim,
                self.config.hidden_dim
            )

        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        # Optimizer
        self.optimizer = optim.Adam(
            self.policy_net.parameters(),
            lr=self.config.learning_rate
        )

        # Replay buffer
        if use_prioritized:
            self.buffer = PrioritizedReplayBuffer(self.config.buffer_size)
        else:
            self.buffer = ReplayBuffer(self.config.buffer_size)

        # Exploration
        self.epsilon = self.config.epsilon_start
        self.steps = 0

        self.is_fitted = False

    def prepare_state(self, df: pd.DataFrame, position: float = 0.0,
                     account_value: float = 10000.0, peak_value: float = 10000.0,
                     recent_pnl: float = 0.0) -> np.ndarray:
        """
        Prepare state vector from market data and account state.

        Returns 20-dimensional state:
        - 10 price features
        - 5 volatility/risk features
        - 5 account/position features
        """
        mid = df['mid'].values if 'mid' in df.columns else (df['bid'] + df['ask']).values / 2

        # Price features (10)
        price_features = [
            pd.Series(mid).pct_change().iloc[-1] * 10000,  # Last return (bps)
            pd.Series(mid).pct_change(5).iloc[-1] * 10000,  # 5-tick return
            pd.Series(mid).pct_change(20).iloc[-1] * 10000,  # 20-tick return
            pd.Series(mid).pct_change().rolling(10).mean().iloc[-1] * 10000,  # Momentum
            pd.Series(mid).pct_change().rolling(20).mean().iloc[-1] * 10000,  # Longer momentum
            ((mid[-1] - pd.Series(mid).rolling(20).mean().iloc[-1]) /
             (pd.Series(mid).rolling(20).std().iloc[-1] + 1e-10)),  # Z-score
            ((mid[-1] - pd.Series(mid).rolling(50).mean().iloc[-1]) /
             (pd.Series(mid).rolling(50).std().iloc[-1] + 1e-10)),  # Longer z-score
            pd.Series(mid).pct_change().diff().iloc[-1] * 10000,  # Acceleration
            mid[-1] / mid[-20] - 1,  # Trend strength
            (mid[-1] - min(mid[-20:])) / (max(mid[-20:]) - min(mid[-20:]) + 1e-10)  # Range position
        ]

        # Volatility features (5)
        vol_features = [
            pd.Series(mid).pct_change().rolling(10).std().iloc[-1] * 10000,  # Short vol
            pd.Series(mid).pct_change().rolling(20).std().iloc[-1] * 10000,  # Medium vol
            pd.Series(mid).pct_change().rolling(50).std().iloc[-1] * 10000,  # Long vol
            pd.Series(mid).pct_change().rolling(10).std().iloc[-1] /
            (pd.Series(mid).pct_change().rolling(50).std().iloc[-1] + 1e-10),  # Vol ratio
            (max(mid[-20:]) - min(mid[-20:])) / mid[-1] * 10000  # Recent range
        ]

        # Account features (5)
        drawdown = (peak_value - account_value) / peak_value
        account_features = [
            position,  # Current position (-1 to 1)
            drawdown,  # Current drawdown (0 to 1)
            recent_pnl / account_value,  # Recent PnL as fraction
            account_value / peak_value,  # Account health (0 to 1)
            abs(position)  # Position magnitude
        ]

        state = np.array(price_features + vol_features + account_features, dtype=np.float32)
        state = np.nan_to_num(state, nan=0.0, posinf=0.0, neginf=0.0)

        return state

    def select_action(self, state: np.ndarray, training: bool = True) -> int:
        """Select action using epsilon-greedy or noisy exploration."""
        if not HAS_TORCH:
            return 2  # Default to 50%

        if training and not isinstance(self.policy_net, NoisyDuelingDQN):
            # Epsilon-greedy for non-noisy networks
            if random.random() < self.epsilon:
                return random.randrange(self.config.action_dim)

        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            q_values = self.policy_net(state_tensor)
            return q_values.argmax().item()

    def get_position_size(self, state: np.ndarray, direction: int = 1) -> float:
        """
        Get optimal position size for given state.

        Args:
            state: State vector
            direction: 1 for long, -1 for short

        Returns:
            Position size as fraction (0.0 to 1.0)
        """
        action = self.select_action(state, training=False)
        return self.POSITION_SIZES[action] * direction

    def train_step(self) -> Optional[float]:
        """Perform one training step."""
        if not HAS_TORCH:
            return None

        if len(self.buffer) < self.config.batch_size:
            return None

        # Sample batch
        if self.use_prioritized:
            states, actions, rewards, next_states, dones, indices, weights = \
                self.buffer.sample(self.config.batch_size)
            weights = torch.FloatTensor(weights)
        else:
            states, actions, rewards, next_states, dones = \
                self.buffer.sample(self.config.batch_size)
            weights = torch.ones(self.config.batch_size)

        # Convert to tensors
        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions).unsqueeze(1)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(next_states)
        dones = torch.FloatTensor(dones)

        # Current Q values
        current_q = self.policy_net(states).gather(1, actions).squeeze()

        # Target Q values
        with torch.no_grad():
            if self.use_double:
                # Double DQN: select action with policy, evaluate with target
                next_actions = self.policy_net(next_states).argmax(1).unsqueeze(1)
                next_q = self.target_net(next_states).gather(1, next_actions).squeeze()
            else:
                next_q = self.target_net(next_states).max(1)[0]

            target_q = rewards + (1 - dones) * self.config.gamma * next_q

        # Compute loss
        td_errors = current_q - target_q
        loss = (weights * td_errors.pow(2)).mean()

        # Update priorities if using prioritized replay
        if self.use_prioritized:
            self.buffer.update_priorities(indices, td_errors.abs().detach().numpy())

        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 10)
        self.optimizer.step()

        # Update target network (soft update)
        if self.steps % self.config.target_update == 0:
            for target_param, policy_param in zip(
                self.target_net.parameters(), self.policy_net.parameters()
            ):
                target_param.data.copy_(
                    self.config.tau * policy_param.data +
                    (1 - self.config.tau) * target_param.data
                )

        # Update epsilon
        self.epsilon = max(
            self.config.epsilon_end,
            self.epsilon * self.config.epsilon_decay
        )

        # Reset noise for noisy networks
        if isinstance(self.policy_net, NoisyDuelingDQN):
            self.policy_net.reset_noise()
            self.target_net.reset_noise()

        self.steps += 1
        return loss.item()

    def train(self, df: pd.DataFrame, epochs: int = 100,
             transaction_cost: float = 0.0001) -> Dict[str, List[float]]:
        """
        Train DQN on historical data.

        Simulates trading and learns optimal position sizes.
        """
        if not HAS_TORCH:
            logger.warning("PyTorch not available")
            return {}

        mid = df['mid'].values if 'mid' in df.columns else (df['bid'] + df['ask']).values / 2
        returns = pd.Series(mid).pct_change().fillna(0).values

        history = {'rewards': [], 'losses': [], 'positions': []}

        for epoch in range(epochs):
            # Initialize episode
            account_value = 10000.0
            peak_value = 10000.0
            position = 0.0
            recent_pnl = 0.0
            episode_reward = 0.0

            for t in range(100, len(df) - 1):
                # Get state
                state = self.prepare_state(
                    df.iloc[t-100:t],
                    position=position,
                    account_value=account_value,
                    peak_value=peak_value,
                    recent_pnl=recent_pnl
                )

                # Select action
                action = self.select_action(state, training=True)
                new_position = self.POSITION_SIZES[action]

                # Calculate reward
                ret = returns[t + 1]
                pnl = new_position * ret * account_value
                cost = abs(new_position - position) * transaction_cost * account_value

                account_value += pnl - cost
                peak_value = max(peak_value, account_value)
                recent_pnl = pnl - cost

                # Risk-adjusted reward
                drawdown = (peak_value - account_value) / peak_value
                reward = (pnl - cost) / 100  # Scale reward
                reward -= drawdown * 10  # Penalty for drawdown

                episode_reward += reward

                # Get next state
                if t + 1 < len(df) - 1:
                    next_state = self.prepare_state(
                        df.iloc[t-99:t+1],
                        position=new_position,
                        account_value=account_value,
                        peak_value=peak_value,
                        recent_pnl=recent_pnl
                    )
                    done = False
                else:
                    next_state = state
                    done = True

                # Store experience
                self.buffer.push(state, action, reward, next_state, done)

                # Train
                loss = self.train_step()
                if loss is not None:
                    history['losses'].append(loss)

                position = new_position
                history['positions'].append(position)

            history['rewards'].append(episode_reward)

            if epoch % 10 == 0:
                avg_reward = np.mean(history['rewards'][-10:]) if history['rewards'] else 0
                logger.info(f"Epoch {epoch}, Avg Reward: {avg_reward:.4f}, Epsilon: {self.epsilon:.3f}")

        self.is_fitted = True
        logger.info("DQN Position Sizer trained")
        return history

    def predict(self, df: pd.DataFrame, position: float = 0.0,
               account_value: float = 10000.0, peak_value: float = 10000.0,
               direction: int = 1) -> Dict[str, Any]:
        """
        Predict optimal position size.

        Returns:
            Dict with position_size, action, q_values
        """
        if not HAS_TORCH or not self.is_fitted:
            return {
                'position_size': 0.5,
                'action': 2,
                'q_values': [0.0] * 5,
                'confidence': 0.0
            }

        state = self.prepare_state(df, position, account_value, peak_value)

        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            q_values = self.policy_net(state_tensor)[0].numpy()

        action = int(np.argmax(q_values))
        position_size = self.POSITION_SIZES[action] * direction

        # Confidence based on Q-value gap
        sorted_q = np.sort(q_values)[::-1]
        confidence = (sorted_q[0] - sorted_q[1]) / (abs(sorted_q[0]) + 1e-10)

        return {
            'position_size': position_size,
            'action': action,
            'q_values': q_values.tolist(),
            'confidence': min(1.0, max(0.0, confidence))
        }


class PPOPositionSizer:
    """
    PPO-based Position Sizing (alternative to DQN).

    Uses Proximal Policy Optimization for continuous action space.
    More stable training than DQN for some environments.
    """

    def __init__(self, state_dim: int = 20, hidden_dim: int = 128,
                 lr: float = 3e-4, gamma: float = 0.99,
                 clip_epsilon: float = 0.2, epochs_per_update: int = 10):
        self.state_dim = state_dim
        self.hidden_dim = hidden_dim
        self.gamma = gamma
        self.clip_epsilon = clip_epsilon
        self.epochs_per_update = epochs_per_update

        if not HAS_TORCH:
            return

        # Actor network (policy)
        self.actor = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 2)  # Mean and log_std
        )

        # Critic network (value)
        self.critic = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )

        self.optimizer = optim.Adam(
            list(self.actor.parameters()) + list(self.critic.parameters()),
            lr=lr
        )

        self.is_fitted = False

    def get_action(self, state: np.ndarray, training: bool = True) -> Tuple[float, float]:
        """Get action from policy."""
        if not HAS_TORCH:
            return 0.5, 0.0

        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        output = self.actor(state_tensor)
        mean = torch.sigmoid(output[0, 0])  # Position size in [0, 1]
        log_std = output[0, 1].clamp(-2, 0)  # Bounded log std

        if training:
            std = torch.exp(log_std)
            action = torch.normal(mean, std)
            action = action.clamp(0, 1)
        else:
            action = mean

        return action.item(), log_std.item()

    def get_value(self, state: np.ndarray) -> float:
        """Get value estimate."""
        if not HAS_TORCH:
            return 0.0

        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        return self.critic(state_tensor).item()

    def update(self, states: np.ndarray, actions: np.ndarray,
               old_log_probs: np.ndarray, returns: np.ndarray,
               advantages: np.ndarray) -> float:
        """Update policy using PPO objective."""
        if not HAS_TORCH:
            return 0.0

        states = torch.FloatTensor(states)
        actions = torch.FloatTensor(actions)
        old_log_probs = torch.FloatTensor(old_log_probs)
        returns = torch.FloatTensor(returns)
        advantages = torch.FloatTensor(advantages)

        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        total_loss = 0
        for _ in range(self.epochs_per_update):
            # Get current policy distribution
            output = self.actor(states)
            means = torch.sigmoid(output[:, 0])
            log_stds = output[:, 1].clamp(-2, 0)
            stds = torch.exp(log_stds)

            # Calculate log probs
            dist = torch.distributions.Normal(means, stds)
            log_probs = dist.log_prob(actions)

            # PPO clipped objective
            ratio = torch.exp(log_probs - old_log_probs)
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()

            # Value loss
            values = self.critic(states).squeeze()
            value_loss = F.mse_loss(values, returns)

            # Entropy bonus
            entropy = dist.entropy().mean()

            # Total loss
            loss = policy_loss + 0.5 * value_loss - 0.01 * entropy

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(self.actor.parameters()) + list(self.critic.parameters()), 0.5
            )
            self.optimizer.step()

            total_loss += loss.item()

        return total_loss / self.epochs_per_update


def create_position_sizer(method: str = 'dqn', **kwargs) -> Any:
    """
    Factory function for position sizers.

    Args:
        method: 'dqn' or 'ppo'
        **kwargs: Configuration parameters

    Returns:
        Position sizer instance
    """
    if method == 'dqn':
        config = DQNConfig(**{k: v for k, v in kwargs.items() if hasattr(DQNConfig, k)})
        return DQNPositionSizer(
            config,
            use_prioritized=kwargs.get('use_prioritized', True),
            use_noisy=kwargs.get('use_noisy', True),
            use_double=kwargs.get('use_double', True)
        )
    elif method == 'ppo':
        return PPOPositionSizer(
            state_dim=kwargs.get('state_dim', 20),
            hidden_dim=kwargs.get('hidden_dim', 128),
            lr=kwargs.get('lr', 3e-4)
        )
    else:
        raise ValueError(f"Unknown method: {method}. Use 'dqn' or 'ppo'.")
