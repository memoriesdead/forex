"""
Gold Standard RL Agents for Forex Trading

Academic Citations:
══════════════════════════════════════════════════════════════════════════════

PPO - Proximal Policy Optimization:
    Schulman et al. (2017)
    "Proximal Policy Optimization Algorithms"
    arXiv:1707.06347
    Best for: Stability, no divergence, sample efficiency

SAC - Soft Actor-Critic:
    Haarnoja et al. (2018)
    "Soft Actor-Critic: Off-Policy Maximum Entropy Deep RL"
    ICML 2018
    Best for: Aggressive trading, exploration-exploitation balance

A2C - Advantage Actor-Critic:
    Mnih et al. (2016)
    "Asynchronous Methods for Deep Reinforcement Learning"
    ICML 2016
    Best for: Cumulative returns, parallel training

TD3 - Twin Delayed DDPG:
    Fujimoto et al. (2018)
    "Addressing Function Approximation Error in Actor-Critic Methods"
    ICML 2018
    Best for: Continuous actions, reduced overestimation

DQN - Deep Q-Network:
    Mnih et al. (2015)
    "Human-level control through deep reinforcement learning"
    Nature 518, 529-533
    Best for: Discrete actions, experience replay

DDPG - Deep Deterministic Policy Gradient:
    Lillicrap et al. (2016)
    "Continuous control with deep reinforcement learning"
    ICLR 2016
    Best for: Continuous control, off-policy learning

Implementation Based On:
    - Stable-Baselines3: Raffin et al. (2021), JMLR
    - FinRL: Liu et al. (2020), NeurIPS Deep RL Workshop
    - ElegantRL: Liu et al. (2021), ICAIF

══════════════════════════════════════════════════════════════════════════════
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Tuple, Any, Union
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
import gymnasium as gym


@dataclass
class AgentConfig:
    """Configuration for RL Agents.

    Based on Stable-Baselines3 and FinRL configurations.
    """
    # Network architecture
    hidden_dims: List[int] = field(default_factory=lambda: [256, 256])
    activation: str = 'relu'

    # Training
    learning_rate: float = 3e-4
    gamma: float = 0.99  # Discount factor
    batch_size: int = 64
    buffer_size: int = 100000

    # PPO specific (Schulman et al. 2017)
    ppo_clip: float = 0.2
    ppo_epochs: int = 10
    gae_lambda: float = 0.95
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    max_grad_norm: float = 0.5

    # SAC specific (Haarnoja et al. 2018)
    tau: float = 0.005  # Soft update coefficient
    alpha: float = 0.2  # Entropy coefficient
    auto_alpha: bool = True  # Automatic entropy tuning

    # TD3 specific (Fujimoto et al. 2018)
    policy_noise: float = 0.2
    noise_clip: float = 0.5
    policy_freq: int = 2

    # DQN specific (Mnih et al. 2015)
    epsilon_start: float = 1.0
    epsilon_end: float = 0.01
    epsilon_decay: float = 0.995
    target_update_freq: int = 100

    # Device
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'


class MLP(nn.Module):
    """Multi-Layer Perceptron for actor/critic networks."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dims: List[int] = [256, 256],
        activation: str = 'relu',
        output_activation: Optional[str] = None,
    ):
        super().__init__()

        activations = {
            'relu': nn.ReLU,
            'tanh': nn.Tanh,
            'leaky_relu': nn.LeakyReLU,
            'elu': nn.ELU,
        }

        act_fn = activations.get(activation, nn.ReLU)

        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(act_fn())
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, output_dim))

        if output_activation:
            layers.append(activations.get(output_activation, nn.Identity)())

        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class GaussianActor(nn.Module):
    """
    Gaussian Policy Network for continuous actions.

    Used by: PPO, SAC, A2C
    Reference: Schulman et al. (2017), Haarnoja et al. (2018)
    """

    LOG_STD_MIN = -20
    LOG_STD_MAX = 2

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dims: List[int] = [256, 256],
        activation: str = 'relu',
    ):
        super().__init__()

        self.backbone = MLP(state_dim, hidden_dims[-1], hidden_dims[:-1], activation)
        self.mean_head = nn.Linear(hidden_dims[-1], action_dim)
        self.log_std_head = nn.Linear(hidden_dims[-1], action_dim)

    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass returning mean and log_std."""
        features = self.backbone(state)
        mean = self.mean_head(features)
        log_std = self.log_std_head(features)
        log_std = torch.clamp(log_std, self.LOG_STD_MIN, self.LOG_STD_MAX)
        return mean, log_std

    def sample(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample action and compute log probability."""
        mean, log_std = self.forward(state)
        std = log_std.exp()
        dist = Normal(mean, std)
        action = dist.rsample()  # Reparameterization trick
        log_prob = dist.log_prob(action).sum(-1, keepdim=True)

        # Squash to [-1, 1] (for SAC)
        action_tanh = torch.tanh(action)
        log_prob -= torch.log(1 - action_tanh.pow(2) + 1e-6).sum(-1, keepdim=True)

        return action_tanh, log_prob

    def get_action(self, state: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        """Get action for execution."""
        mean, log_std = self.forward(state)
        if deterministic:
            return torch.tanh(mean)
        std = log_std.exp()
        dist = Normal(mean, std)
        action = dist.sample()
        return torch.tanh(action)


class DeterministicActor(nn.Module):
    """
    Deterministic Policy Network for continuous actions.

    Used by: DDPG, TD3
    Reference: Lillicrap et al. (2016), Fujimoto et al. (2018)
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dims: List[int] = [256, 256],
        activation: str = 'relu',
        max_action: float = 1.0,
    ):
        super().__init__()
        self.net = MLP(state_dim, action_dim, hidden_dims, activation)
        self.max_action = max_action

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.max_action * torch.tanh(self.net(state))


class Critic(nn.Module):
    """
    Value/Q-function Network.

    Used by: All agents
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int = 0,
        hidden_dims: List[int] = [256, 256],
        activation: str = 'relu',
    ):
        super().__init__()
        input_dim = state_dim + action_dim
        self.net = MLP(input_dim, 1, hidden_dims, activation)

    def forward(self, state: torch.Tensor, action: Optional[torch.Tensor] = None) -> torch.Tensor:
        if action is not None:
            x = torch.cat([state, action], dim=-1)
        else:
            x = state
        return self.net(x)


class TwinCritic(nn.Module):
    """
    Twin Q-Networks for TD3/SAC.

    Reference: Fujimoto et al. (2018) - "Clipped Double Q-Learning"
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dims: List[int] = [256, 256],
        activation: str = 'relu',
    ):
        super().__init__()
        self.q1 = Critic(state_dim, action_dim, hidden_dims, activation)
        self.q2 = Critic(state_dim, action_dim, hidden_dims, activation)

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.q1(state, action), self.q2(state, action)

    def q1_forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return self.q1(state, action)


class ReplayBuffer:
    """
    Experience Replay Buffer.

    Reference: Mnih et al. (2015) - "Experience Replay"
    """

    def __init__(self, state_dim: int, action_dim: int, max_size: int = 100000):
        self.max_size = max_size
        self.ptr = 0
        self.size = 0

        self.states = np.zeros((max_size, state_dim), dtype=np.float32)
        self.actions = np.zeros((max_size, action_dim), dtype=np.float32)
        self.rewards = np.zeros((max_size, 1), dtype=np.float32)
        self.next_states = np.zeros((max_size, state_dim), dtype=np.float32)
        self.dones = np.zeros((max_size, 1), dtype=np.float32)

    def add(self, state, action, reward, next_state, done):
        self.states[self.ptr] = state
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.next_states[self.ptr] = next_state
        self.dones[self.ptr] = done

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size: int, device: str = 'cpu') -> Dict[str, torch.Tensor]:
        idx = np.random.randint(0, self.size, size=batch_size)
        return {
            'states': torch.FloatTensor(self.states[idx]).to(device),
            'actions': torch.FloatTensor(self.actions[idx]).to(device),
            'rewards': torch.FloatTensor(self.rewards[idx]).to(device),
            'next_states': torch.FloatTensor(self.next_states[idx]).to(device),
            'dones': torch.FloatTensor(self.dones[idx]).to(device),
        }


class BaseRLAgent(ABC):
    """
    Abstract Base Class for RL Agents.

    Follows Stable-Baselines3 interface pattern.
    """

    def __init__(
        self,
        env: gym.Env,
        config: Optional[AgentConfig] = None,
    ):
        self.env = env
        self.config = config or AgentConfig()
        self.device = torch.device(self.config.device)

        # Get dimensions from environment
        self.state_dim = env.observation_space.shape[0]
        if isinstance(env.action_space, gym.spaces.Box):
            self.action_dim = env.action_space.shape[0]
            self.discrete = False
        else:
            self.action_dim = env.action_space.n
            self.discrete = True

        self.total_steps = 0

    @abstractmethod
    def select_action(self, state: np.ndarray, deterministic: bool = False) -> np.ndarray:
        """Select action given state."""
        pass

    @abstractmethod
    def train(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Train on a batch of data."""
        pass

    def learn(self, total_timesteps: int, callback=None) -> 'BaseRLAgent':
        """Main training loop."""
        state, _ = self.env.reset()
        episode_reward = 0
        episode_length = 0

        for step in range(total_timesteps):
            action = self.select_action(state)

            next_state, reward, terminated, truncated, info = self.env.step(action)
            done = terminated or truncated

            self.buffer.add(state, action, reward, next_state, float(done))

            state = next_state
            episode_reward += reward
            episode_length += 1
            self.total_steps += 1

            if done:
                state, _ = self.env.reset()
                if callback:
                    callback({
                        'episode_reward': episode_reward,
                        'episode_length': episode_length,
                        'total_steps': self.total_steps,
                    })
                episode_reward = 0
                episode_length = 0

            if self.buffer.size >= self.config.batch_size:
                batch = self.buffer.sample(self.config.batch_size, self.device)
                self.train(batch)

        return self

    def predict(self, state: np.ndarray, deterministic: bool = True) -> np.ndarray:
        """Predict action for given state."""
        return self.select_action(state, deterministic=deterministic)

    def save(self, path: str):
        """Save model to path."""
        torch.save(self.state_dict(), path)

    def load(self, path: str):
        """Load model from path."""
        self.load_state_dict(torch.load(path, map_location=self.device))

    @abstractmethod
    def state_dict(self) -> Dict[str, Any]:
        """Get state dict for saving."""
        pass

    @abstractmethod
    def load_state_dict(self, state_dict: Dict[str, Any]):
        """Load state dict."""
        pass


class PPOAgent(BaseRLAgent):
    """
    Proximal Policy Optimization Agent.

    Reference:
        Schulman et al. (2017)
        "Proximal Policy Optimization Algorithms"
        arXiv:1707.06347

    Key innovations:
    - Clipped surrogate objective prevents large policy updates
    - GAE (Generalized Advantage Estimation) for variance reduction
    - Multiple epochs of minibatch updates

    PPO is the MOST STABLE algorithm for trading:
    - No divergence from clipped objective
    - Sample efficient on-policy learning
    - Works well with high-dimensional state spaces
    """

    def __init__(
        self,
        env: gym.Env,
        config: Optional[AgentConfig] = None,
    ):
        super().__init__(env, config)

        # Networks
        self.actor = GaussianActor(
            self.state_dim,
            self.action_dim,
            self.config.hidden_dims,
            self.config.activation,
        ).to(self.device)

        self.critic = Critic(
            self.state_dim,
            0,  # No action input for value function
            self.config.hidden_dims,
            self.config.activation,
        ).to(self.device)

        # Optimizers
        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(),
            lr=self.config.learning_rate
        )
        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(),
            lr=self.config.learning_rate
        )

        # Rollout buffer
        self.buffer = RolloutBuffer(
            self.state_dim,
            self.action_dim,
            self.config.buffer_size,
        )

    def select_action(self, state: np.ndarray, deterministic: bool = False) -> np.ndarray:
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            action = self.actor.get_action(state_tensor, deterministic)
            return action.cpu().numpy()[0]

    def compute_gae(
        self,
        rewards: torch.Tensor,
        values: torch.Tensor,
        dones: torch.Tensor,
        next_value: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute Generalized Advantage Estimation.

        Reference: Schulman et al. (2016)
        "High-Dimensional Continuous Control Using GAE"
        """
        advantages = torch.zeros_like(rewards)
        last_gae = 0

        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_val = next_value
            else:
                next_val = values[t + 1]

            delta = rewards[t] + self.config.gamma * next_val * (1 - dones[t]) - values[t]
            advantages[t] = last_gae = delta + self.config.gamma * self.config.gae_lambda * (1 - dones[t]) * last_gae

        returns = advantages + values
        return advantages, returns

    def train(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Train PPO agent on rollout data."""
        states = batch['states']
        actions = batch['actions']
        old_log_probs = batch['log_probs']
        advantages = batch['advantages']
        returns = batch['returns']

        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Multiple epochs of updates
        total_loss = 0.0
        total_pg_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0

        for _ in range(self.config.ppo_epochs):
            # Get current policy distribution
            mean, log_std = self.actor(states)
            std = log_std.exp()
            dist = Normal(mean, std)

            new_log_probs = dist.log_prob(actions).sum(-1, keepdim=True)
            entropy = dist.entropy().sum(-1).mean()

            # PPO clipped objective (Schulman et al. 2017)
            ratio = (new_log_probs - old_log_probs).exp()
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.config.ppo_clip, 1 + self.config.ppo_clip) * advantages
            pg_loss = -torch.min(surr1, surr2).mean()

            # Value loss
            values = self.critic(states)
            value_loss = F.mse_loss(values, returns)

            # Total loss
            loss = pg_loss + self.config.value_coef * value_loss - self.config.entropy_coef * entropy

            # Update networks
            self.actor_optimizer.zero_grad()
            self.critic_optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.actor.parameters(), self.config.max_grad_norm)
            nn.utils.clip_grad_norm_(self.critic.parameters(), self.config.max_grad_norm)
            self.actor_optimizer.step()
            self.critic_optimizer.step()

            total_loss += loss.item()
            total_pg_loss += pg_loss.item()
            total_value_loss += value_loss.item()
            total_entropy += entropy.item()

        n_epochs = self.config.ppo_epochs
        return {
            'loss': total_loss / n_epochs,
            'pg_loss': total_pg_loss / n_epochs,
            'value_loss': total_value_loss / n_epochs,
            'entropy': total_entropy / n_epochs,
        }

    def state_dict(self) -> Dict[str, Any]:
        return {
            'actor': self.actor.state_dict(),
            'critic': self.critic.state_dict(),
            'actor_optimizer': self.actor_optimizer.state_dict(),
            'critic_optimizer': self.critic_optimizer.state_dict(),
        }

    def load_state_dict(self, state_dict: Dict[str, Any]):
        self.actor.load_state_dict(state_dict['actor'])
        self.critic.load_state_dict(state_dict['critic'])
        self.actor_optimizer.load_state_dict(state_dict['actor_optimizer'])
        self.critic_optimizer.load_state_dict(state_dict['critic_optimizer'])


class RolloutBuffer:
    """Buffer for on-policy algorithms (PPO, A2C)."""

    def __init__(self, state_dim: int, action_dim: int, max_size: int = 2048):
        self.max_size = max_size
        self.ptr = 0
        self.size = 0

        self.states = np.zeros((max_size, state_dim), dtype=np.float32)
        self.actions = np.zeros((max_size, action_dim), dtype=np.float32)
        self.rewards = np.zeros((max_size, 1), dtype=np.float32)
        self.log_probs = np.zeros((max_size, 1), dtype=np.float32)
        self.dones = np.zeros((max_size, 1), dtype=np.float32)
        self.values = np.zeros((max_size, 1), dtype=np.float32)

    def add(self, state, action, reward, log_prob, done, value):
        self.states[self.ptr] = state
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.log_probs[self.ptr] = log_prob
        self.dones[self.ptr] = done
        self.values[self.ptr] = value

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def get(self, device: str = 'cpu') -> Dict[str, torch.Tensor]:
        return {
            'states': torch.FloatTensor(self.states[:self.size]).to(device),
            'actions': torch.FloatTensor(self.actions[:self.size]).to(device),
            'rewards': torch.FloatTensor(self.rewards[:self.size]).to(device),
            'log_probs': torch.FloatTensor(self.log_probs[:self.size]).to(device),
            'dones': torch.FloatTensor(self.dones[:self.size]).to(device),
            'values': torch.FloatTensor(self.values[:self.size]).to(device),
        }

    def clear(self):
        self.ptr = 0
        self.size = 0


class SACAgent(BaseRLAgent):
    """
    Soft Actor-Critic Agent.

    Reference:
        Haarnoja et al. (2018)
        "Soft Actor-Critic: Off-Policy Maximum Entropy Deep RL"
        ICML 2018

    Key innovations:
    - Maximum entropy framework for exploration
    - Automatic temperature tuning
    - Twin Q-networks to reduce overestimation

    SAC is BEST for AGGRESSIVE trading:
    - Entropy bonus encourages exploration
    - Off-policy = sample efficient
    - Works well with continuous actions
    """

    def __init__(
        self,
        env: gym.Env,
        config: Optional[AgentConfig] = None,
    ):
        super().__init__(env, config)

        # Networks
        self.actor = GaussianActor(
            self.state_dim,
            self.action_dim,
            self.config.hidden_dims,
            self.config.activation,
        ).to(self.device)

        self.critic = TwinCritic(
            self.state_dim,
            self.action_dim,
            self.config.hidden_dims,
            self.config.activation,
        ).to(self.device)

        self.critic_target = TwinCritic(
            self.state_dim,
            self.action_dim,
            self.config.hidden_dims,
            self.config.activation,
        ).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        # Automatic entropy tuning (Haarnoja et al. 2018)
        if self.config.auto_alpha:
            self.target_entropy = -self.action_dim
            self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
            self.alpha = self.log_alpha.exp()
            self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=self.config.learning_rate)
        else:
            self.alpha = self.config.alpha

        # Optimizers
        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(),
            lr=self.config.learning_rate
        )
        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(),
            lr=self.config.learning_rate
        )

        # Replay buffer
        self.buffer = ReplayBuffer(
            self.state_dim,
            self.action_dim,
            self.config.buffer_size,
        )

    def select_action(self, state: np.ndarray, deterministic: bool = False) -> np.ndarray:
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            if deterministic:
                action = self.actor.get_action(state_tensor, deterministic=True)
            else:
                action, _ = self.actor.sample(state_tensor)
            return action.cpu().numpy()[0]

    def train(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Train SAC agent."""
        states = batch['states']
        actions = batch['actions']
        rewards = batch['rewards']
        next_states = batch['next_states']
        dones = batch['dones']

        # Update critic
        with torch.no_grad():
            next_actions, next_log_probs = self.actor.sample(next_states)
            q1_next, q2_next = self.critic_target(next_states, next_actions)
            q_next = torch.min(q1_next, q2_next) - self.alpha * next_log_probs
            q_target = rewards + self.config.gamma * (1 - dones) * q_next

        q1, q2 = self.critic(states, actions)
        critic_loss = F.mse_loss(q1, q_target) + F.mse_loss(q2, q_target)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Update actor
        new_actions, log_probs = self.actor.sample(states)
        q1_new, q2_new = self.critic(states, new_actions)
        q_new = torch.min(q1_new, q2_new)
        actor_loss = (self.alpha * log_probs - q_new).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Update alpha (automatic entropy tuning)
        alpha_loss = 0.0
        if self.config.auto_alpha:
            alpha_loss = -(self.log_alpha * (log_probs + self.target_entropy).detach()).mean()
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
            self.alpha = self.log_alpha.exp()

        # Soft update target networks
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(self.config.tau * param.data + (1 - self.config.tau) * target_param.data)

        return {
            'critic_loss': critic_loss.item(),
            'actor_loss': actor_loss.item(),
            'alpha_loss': alpha_loss.item() if self.config.auto_alpha else 0.0,
            'alpha': self.alpha.item() if isinstance(self.alpha, torch.Tensor) else self.alpha,
        }

    def state_dict(self) -> Dict[str, Any]:
        state = {
            'actor': self.actor.state_dict(),
            'critic': self.critic.state_dict(),
            'critic_target': self.critic_target.state_dict(),
            'actor_optimizer': self.actor_optimizer.state_dict(),
            'critic_optimizer': self.critic_optimizer.state_dict(),
        }
        if self.config.auto_alpha:
            state['log_alpha'] = self.log_alpha
            state['alpha_optimizer'] = self.alpha_optimizer.state_dict()
        return state

    def load_state_dict(self, state_dict: Dict[str, Any]):
        self.actor.load_state_dict(state_dict['actor'])
        self.critic.load_state_dict(state_dict['critic'])
        self.critic_target.load_state_dict(state_dict['critic_target'])
        self.actor_optimizer.load_state_dict(state_dict['actor_optimizer'])
        self.critic_optimizer.load_state_dict(state_dict['critic_optimizer'])
        if self.config.auto_alpha and 'log_alpha' in state_dict:
            self.log_alpha = state_dict['log_alpha']
            self.alpha = self.log_alpha.exp()
            self.alpha_optimizer.load_state_dict(state_dict['alpha_optimizer'])


class A2CAgent(BaseRLAgent):
    """
    Advantage Actor-Critic Agent.

    Reference:
        Mnih et al. (2016)
        "Asynchronous Methods for Deep Reinforcement Learning"
        ICML 2016

    A2C (synchronous version of A3C) is BEST for CUMULATIVE RETURNS:
    - Actor-critic reduces variance
    - Fast convergence on trading tasks
    - Good for parallel training
    """

    def __init__(
        self,
        env: gym.Env,
        config: Optional[AgentConfig] = None,
    ):
        super().__init__(env, config)

        # Shared backbone for efficiency
        self.actor = GaussianActor(
            self.state_dim,
            self.action_dim,
            self.config.hidden_dims,
            self.config.activation,
        ).to(self.device)

        self.critic = Critic(
            self.state_dim,
            0,
            self.config.hidden_dims,
            self.config.activation,
        ).to(self.device)

        # Optimizers
        self.optimizer = torch.optim.Adam(
            list(self.actor.parameters()) + list(self.critic.parameters()),
            lr=self.config.learning_rate
        )

        # Rollout buffer
        self.buffer = RolloutBuffer(
            self.state_dim,
            self.action_dim,
            self.config.buffer_size,
        )

    def select_action(self, state: np.ndarray, deterministic: bool = False) -> np.ndarray:
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            action = self.actor.get_action(state_tensor, deterministic)
            return action.cpu().numpy()[0]

    def train(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Train A2C agent."""
        states = batch['states']
        actions = batch['actions']
        returns = batch['returns']
        advantages = batch['advantages']

        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Policy loss
        mean, log_std = self.actor(states)
        std = log_std.exp()
        dist = Normal(mean, std)
        log_probs = dist.log_prob(actions).sum(-1, keepdim=True)
        entropy = dist.entropy().sum(-1).mean()

        pg_loss = -(log_probs * advantages).mean()

        # Value loss
        values = self.critic(states)
        value_loss = F.mse_loss(values, returns)

        # Total loss
        loss = pg_loss + self.config.value_coef * value_loss - self.config.entropy_coef * entropy

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(
            list(self.actor.parameters()) + list(self.critic.parameters()),
            self.config.max_grad_norm
        )
        self.optimizer.step()

        return {
            'loss': loss.item(),
            'pg_loss': pg_loss.item(),
            'value_loss': value_loss.item(),
            'entropy': entropy.item(),
        }

    def state_dict(self) -> Dict[str, Any]:
        return {
            'actor': self.actor.state_dict(),
            'critic': self.critic.state_dict(),
            'optimizer': self.optimizer.state_dict(),
        }

    def load_state_dict(self, state_dict: Dict[str, Any]):
        self.actor.load_state_dict(state_dict['actor'])
        self.critic.load_state_dict(state_dict['critic'])
        self.optimizer.load_state_dict(state_dict['optimizer'])


class TD3Agent(BaseRLAgent):
    """
    Twin Delayed DDPG Agent.

    Reference:
        Fujimoto et al. (2018)
        "Addressing Function Approximation Error in Actor-Critic Methods"
        ICML 2018

    Key innovations:
    - Twin Q-networks to reduce overestimation
    - Delayed policy updates
    - Target policy smoothing

    TD3 is BEST for CONTINUOUS ACTIONS:
    - Deterministic policy for precise control
    - Clipped double Q-learning
    - Works well with large action spaces
    """

    def __init__(
        self,
        env: gym.Env,
        config: Optional[AgentConfig] = None,
    ):
        super().__init__(env, config)

        # Networks
        self.actor = DeterministicActor(
            self.state_dim,
            self.action_dim,
            self.config.hidden_dims,
            self.config.activation,
        ).to(self.device)

        self.actor_target = DeterministicActor(
            self.state_dim,
            self.action_dim,
            self.config.hidden_dims,
            self.config.activation,
        ).to(self.device)
        self.actor_target.load_state_dict(self.actor.state_dict())

        self.critic = TwinCritic(
            self.state_dim,
            self.action_dim,
            self.config.hidden_dims,
            self.config.activation,
        ).to(self.device)

        self.critic_target = TwinCritic(
            self.state_dim,
            self.action_dim,
            self.config.hidden_dims,
            self.config.activation,
        ).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        # Optimizers
        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(),
            lr=self.config.learning_rate
        )
        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(),
            lr=self.config.learning_rate
        )

        # Replay buffer
        self.buffer = ReplayBuffer(
            self.state_dim,
            self.action_dim,
            self.config.buffer_size,
        )

        self.update_count = 0

    def select_action(self, state: np.ndarray, deterministic: bool = False) -> np.ndarray:
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            action = self.actor(state_tensor).cpu().numpy()[0]

            if not deterministic:
                noise = np.random.normal(0, self.config.policy_noise, size=action.shape)
                action = np.clip(action + noise, -1.0, 1.0)

            return action

    def train(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Train TD3 agent."""
        states = batch['states']
        actions = batch['actions']
        rewards = batch['rewards']
        next_states = batch['next_states']
        dones = batch['dones']

        self.update_count += 1

        # Target policy smoothing (Fujimoto et al. 2018)
        with torch.no_grad():
            noise = (torch.randn_like(actions) * self.config.policy_noise).clamp(
                -self.config.noise_clip, self.config.noise_clip
            )
            next_actions = (self.actor_target(next_states) + noise).clamp(-1.0, 1.0)

            q1_next, q2_next = self.critic_target(next_states, next_actions)
            q_next = torch.min(q1_next, q2_next)
            q_target = rewards + self.config.gamma * (1 - dones) * q_next

        # Update critic
        q1, q2 = self.critic(states, actions)
        critic_loss = F.mse_loss(q1, q_target) + F.mse_loss(q2, q_target)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Delayed policy updates (Fujimoto et al. 2018)
        actor_loss = 0.0
        if self.update_count % self.config.policy_freq == 0:
            actor_loss = -self.critic.q1_forward(states, self.actor(states)).mean()

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # Soft update target networks
            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(self.config.tau * param.data + (1 - self.config.tau) * target_param.data)
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(self.config.tau * param.data + (1 - self.config.tau) * target_param.data)

        return {
            'critic_loss': critic_loss.item(),
            'actor_loss': actor_loss.item() if isinstance(actor_loss, torch.Tensor) else 0.0,
        }

    def state_dict(self) -> Dict[str, Any]:
        return {
            'actor': self.actor.state_dict(),
            'actor_target': self.actor_target.state_dict(),
            'critic': self.critic.state_dict(),
            'critic_target': self.critic_target.state_dict(),
            'actor_optimizer': self.actor_optimizer.state_dict(),
            'critic_optimizer': self.critic_optimizer.state_dict(),
        }

    def load_state_dict(self, state_dict: Dict[str, Any]):
        self.actor.load_state_dict(state_dict['actor'])
        self.actor_target.load_state_dict(state_dict['actor_target'])
        self.critic.load_state_dict(state_dict['critic'])
        self.critic_target.load_state_dict(state_dict['critic_target'])
        self.actor_optimizer.load_state_dict(state_dict['actor_optimizer'])
        self.critic_optimizer.load_state_dict(state_dict['critic_optimizer'])


class DQNAgent(BaseRLAgent):
    """
    Deep Q-Network Agent.

    Reference:
        Mnih et al. (2015)
        "Human-level control through deep reinforcement learning"
        Nature 518, 529-533

    DQN is BEST for DISCRETE ACTIONS:
    - Q-learning with neural network
    - Experience replay for stability
    - Target network for convergence
    """

    def __init__(
        self,
        env: gym.Env,
        config: Optional[AgentConfig] = None,
    ):
        super().__init__(env, config)
        assert self.discrete, "DQN requires discrete action space"

        # Q-network
        self.q_network = MLP(
            self.state_dim,
            self.action_dim,
            self.config.hidden_dims,
            self.config.activation,
        ).to(self.device)

        self.q_target = MLP(
            self.state_dim,
            self.action_dim,
            self.config.hidden_dims,
            self.config.activation,
        ).to(self.device)
        self.q_target.load_state_dict(self.q_network.state_dict())

        # Optimizer
        self.optimizer = torch.optim.Adam(
            self.q_network.parameters(),
            lr=self.config.learning_rate
        )

        # Replay buffer
        self.buffer = ReplayBuffer(
            self.state_dim,
            1,  # Discrete action stored as single int
            self.config.buffer_size,
        )

        self.epsilon = self.config.epsilon_start
        self.update_count = 0

    def select_action(self, state: np.ndarray, deterministic: bool = False) -> np.ndarray:
        if not deterministic and np.random.random() < self.epsilon:
            return np.array([np.random.randint(self.action_dim)])

        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.q_network(state_tensor)
            action = q_values.argmax(dim=-1).cpu().numpy()
            return action

    def train(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Train DQN agent."""
        states = batch['states']
        actions = batch['actions'].long()
        rewards = batch['rewards']
        next_states = batch['next_states']
        dones = batch['dones']

        self.update_count += 1

        # Current Q values
        q_values = self.q_network(states)
        q_values = q_values.gather(1, actions)

        # Target Q values
        with torch.no_grad():
            q_next = self.q_target(next_states).max(dim=-1, keepdim=True)[0]
            q_target = rewards + self.config.gamma * (1 - dones) * q_next

        # Loss
        loss = F.mse_loss(q_values, q_target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Update target network
        if self.update_count % self.config.target_update_freq == 0:
            self.q_target.load_state_dict(self.q_network.state_dict())

        # Decay epsilon
        self.epsilon = max(self.config.epsilon_end, self.epsilon * self.config.epsilon_decay)

        return {
            'loss': loss.item(),
            'epsilon': self.epsilon,
        }

    def state_dict(self) -> Dict[str, Any]:
        return {
            'q_network': self.q_network.state_dict(),
            'q_target': self.q_target.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
        }

    def load_state_dict(self, state_dict: Dict[str, Any]):
        self.q_network.load_state_dict(state_dict['q_network'])
        self.q_target.load_state_dict(state_dict['q_target'])
        self.optimizer.load_state_dict(state_dict['optimizer'])
        self.epsilon = state_dict.get('epsilon', self.config.epsilon_start)


class DDPGAgent(BaseRLAgent):
    """
    Deep Deterministic Policy Gradient Agent.

    Reference:
        Lillicrap et al. (2016)
        "Continuous control with deep reinforcement learning"
        ICLR 2016

    DDPG is the foundation for TD3:
    - Deterministic policy gradient
    - Off-policy actor-critic
    - Good for continuous control
    """

    def __init__(
        self,
        env: gym.Env,
        config: Optional[AgentConfig] = None,
    ):
        super().__init__(env, config)

        # Networks
        self.actor = DeterministicActor(
            self.state_dim,
            self.action_dim,
            self.config.hidden_dims,
            self.config.activation,
        ).to(self.device)

        self.actor_target = DeterministicActor(
            self.state_dim,
            self.action_dim,
            self.config.hidden_dims,
            self.config.activation,
        ).to(self.device)
        self.actor_target.load_state_dict(self.actor.state_dict())

        self.critic = Critic(
            self.state_dim,
            self.action_dim,
            self.config.hidden_dims,
            self.config.activation,
        ).to(self.device)

        self.critic_target = Critic(
            self.state_dim,
            self.action_dim,
            self.config.hidden_dims,
            self.config.activation,
        ).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        # Optimizers
        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(),
            lr=self.config.learning_rate
        )
        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(),
            lr=self.config.learning_rate
        )

        # Replay buffer
        self.buffer = ReplayBuffer(
            self.state_dim,
            self.action_dim,
            self.config.buffer_size,
        )

    def select_action(self, state: np.ndarray, deterministic: bool = False) -> np.ndarray:
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            action = self.actor(state_tensor).cpu().numpy()[0]

            if not deterministic:
                noise = np.random.normal(0, self.config.policy_noise, size=action.shape)
                action = np.clip(action + noise, -1.0, 1.0)

            return action

    def train(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Train DDPG agent."""
        states = batch['states']
        actions = batch['actions']
        rewards = batch['rewards']
        next_states = batch['next_states']
        dones = batch['dones']

        # Update critic
        with torch.no_grad():
            next_actions = self.actor_target(next_states)
            q_next = self.critic_target(next_states, next_actions)
            q_target = rewards + self.config.gamma * (1 - dones) * q_next

        q = self.critic(states, actions)
        critic_loss = F.mse_loss(q, q_target)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Update actor
        actor_loss = -self.critic(states, self.actor(states)).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Soft update target networks
        for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
            target_param.data.copy_(self.config.tau * param.data + (1 - self.config.tau) * target_param.data)
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(self.config.tau * param.data + (1 - self.config.tau) * target_param.data)

        return {
            'critic_loss': critic_loss.item(),
            'actor_loss': actor_loss.item(),
        }

    def state_dict(self) -> Dict[str, Any]:
        return {
            'actor': self.actor.state_dict(),
            'actor_target': self.actor_target.state_dict(),
            'critic': self.critic.state_dict(),
            'critic_target': self.critic_target.state_dict(),
            'actor_optimizer': self.actor_optimizer.state_dict(),
            'critic_optimizer': self.critic_optimizer.state_dict(),
        }

    def load_state_dict(self, state_dict: Dict[str, Any]):
        self.actor.load_state_dict(state_dict['actor'])
        self.actor_target.load_state_dict(state_dict['actor_target'])
        self.critic.load_state_dict(state_dict['critic'])
        self.critic_target.load_state_dict(state_dict['critic_target'])
        self.actor_optimizer.load_state_dict(state_dict['actor_optimizer'])
        self.critic_optimizer.load_state_dict(state_dict['critic_optimizer'])


def create_agent(
    agent_type: str,
    env: gym.Env,
    config: Optional[AgentConfig] = None,
) -> BaseRLAgent:
    """
    Factory function to create RL agents.

    Args:
        agent_type: One of 'ppo', 'sac', 'a2c', 'td3', 'dqn', 'ddpg'
        env: Gym environment
        config: Agent configuration

    Returns:
        RL agent instance
    """
    agents = {
        'ppo': PPOAgent,
        'sac': SACAgent,
        'a2c': A2CAgent,
        'td3': TD3Agent,
        'dqn': DQNAgent,
        'ddpg': DDPGAgent,
    }

    agent_type = agent_type.lower()
    if agent_type not in agents:
        raise ValueError(f"Unknown agent type: {agent_type}. Available: {list(agents.keys())}")

    return agents[agent_type](env, config)
