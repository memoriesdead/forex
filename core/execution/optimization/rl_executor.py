"""
DDPG Adaptive Execution Agent
==============================
Deep Deterministic Policy Gradient agent for adaptive execution.

Learns optimal execution decisions from experience:
- When to use market vs limit orders
- Optimal limit price offset
- Trade rate adjustment
- Aggressiveness based on market conditions

State Space:
- Remaining quantity (normalized)
- Time remaining (normalized)
- Current spread (bps)
- Recent volatility
- Session liquidity factor
- Order flow imbalance

Action Space (continuous):
- Trade rate adjustment (0.5 to 2.0)
- Use limit order probability (0 to 1)
- Limit price offset (0 to 3 bps)
- Aggressiveness (0 to 1)

Reward:
- Negative execution cost
- Penalty for unfilled orders
- Bonus for beating benchmark (TWAP)
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Tuple, List, Dict
from collections import deque
import logging
import pickle
from pathlib import Path

logger = logging.getLogger(__name__)

# Try to import torch for neural networks
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    import torch.nn.functional as F
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    logger.warning("PyTorch not available, RL executor will use rule-based fallback")


@dataclass
class ExecutionState:
    """State representation for RL agent."""
    remaining_qty_pct: float      # 0-1, portion remaining
    time_remaining_pct: float     # 0-1, portion of horizon remaining
    spread_bps: float             # Current spread in bps
    volatility: float             # Recent volatility
    session_liquidity: float      # Session liquidity factor (0.2-1.8)
    order_flow: float             # -1 to 1, recent order flow imbalance
    fill_rate: float              # Recent fill rate for limit orders
    slippage_so_far_bps: float    # Execution cost so far

    def to_tensor(self) -> 'torch.Tensor':
        """Convert to PyTorch tensor."""
        if not HAS_TORCH:
            raise RuntimeError("PyTorch not available")
        return torch.FloatTensor([
            self.remaining_qty_pct,
            self.time_remaining_pct,
            self.spread_bps / 5.0,  # Normalize spread
            self.volatility * 10000,  # Normalize vol
            self.session_liquidity,
            self.order_flow,
            self.fill_rate,
            self.slippage_so_far_bps / 10.0  # Normalize slippage
        ])

    def to_array(self) -> np.ndarray:
        """Convert to numpy array."""
        return np.array([
            self.remaining_qty_pct,
            self.time_remaining_pct,
            self.spread_bps / 5.0,
            self.volatility * 10000,
            self.session_liquidity,
            self.order_flow,
            self.fill_rate,
            self.slippage_so_far_bps / 10.0
        ])


@dataclass
class ExecutionAction:
    """Action output from RL agent."""
    trade_rate: float        # Multiplier on base trade rate (0.5-2.0)
    use_limit: bool          # Whether to use limit order
    limit_offset_bps: float  # Limit price offset in bps
    aggressiveness: float    # Overall aggressiveness (0-1)

    @classmethod
    def from_array(cls, arr: np.ndarray) -> 'ExecutionAction':
        """Create from raw action array."""
        return cls(
            trade_rate=np.clip(arr[0] * 0.75 + 1.25, 0.5, 2.0),  # Map to [0.5, 2.0]
            use_limit=arr[1] > 0.5,
            limit_offset_bps=np.clip(arr[2] * 3.0, 0.0, 3.0),  # Map to [0, 3]
            aggressiveness=np.clip(arr[3], 0.0, 1.0)
        )


@dataclass
class Experience:
    """Experience tuple for replay buffer."""
    state: ExecutionState
    action: np.ndarray
    reward: float
    next_state: ExecutionState
    done: bool


if HAS_TORCH:
    class Actor(nn.Module):
        """Actor network for DDPG."""

        def __init__(self, state_dim: int = 8, action_dim: int = 4, hidden_dim: int = 128):
            super().__init__()
            self.fc1 = nn.Linear(state_dim, hidden_dim)
            self.fc2 = nn.Linear(hidden_dim, hidden_dim)
            self.fc3 = nn.Linear(hidden_dim, action_dim)

            # Initialize weights
            nn.init.xavier_uniform_(self.fc1.weight)
            nn.init.xavier_uniform_(self.fc2.weight)
            nn.init.xavier_uniform_(self.fc3.weight)

        def forward(self, state: torch.Tensor) -> torch.Tensor:
            x = F.relu(self.fc1(state))
            x = F.relu(self.fc2(x))
            x = torch.tanh(self.fc3(x))  # Output in [-1, 1]
            return x

    class Critic(nn.Module):
        """Critic network for DDPG."""

        def __init__(self, state_dim: int = 8, action_dim: int = 4, hidden_dim: int = 128):
            super().__init__()
            self.fc1 = nn.Linear(state_dim + action_dim, hidden_dim)
            self.fc2 = nn.Linear(hidden_dim, hidden_dim)
            self.fc3 = nn.Linear(hidden_dim, 1)

            nn.init.xavier_uniform_(self.fc1.weight)
            nn.init.xavier_uniform_(self.fc2.weight)
            nn.init.xavier_uniform_(self.fc3.weight)

        def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
            x = torch.cat([state, action], dim=-1)
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            x = self.fc3(x)
            return x


class ReplayBuffer:
    """Experience replay buffer."""

    def __init__(self, capacity: int = 100000):
        self.buffer = deque(maxlen=capacity)

    def push(self, experience: Experience):
        """Add experience to buffer."""
        self.buffer.append(experience)

    def sample(self, batch_size: int) -> List[Experience]:
        """Sample batch of experiences."""
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        return [self.buffer[i] for i in indices]

    def __len__(self) -> int:
        return len(self.buffer)


class DDPGExecutor:
    """
    DDPG-based adaptive execution agent.

    Learns optimal execution strategy from experience.
    Falls back to rule-based strategy if PyTorch unavailable.
    """

    def __init__(self,
                 state_dim: int = 8,
                 action_dim: int = 4,
                 hidden_dim: int = 128,
                 lr_actor: float = 1e-4,
                 lr_critic: float = 1e-3,
                 gamma: float = 0.99,
                 tau: float = 0.005,
                 exploration_noise: float = 0.1):
        """
        Initialize DDPG agent.

        Args:
            state_dim: Dimension of state space
            action_dim: Dimension of action space
            hidden_dim: Hidden layer size
            lr_actor: Actor learning rate
            lr_critic: Critic learning rate
            gamma: Discount factor
            tau: Soft update coefficient
            exploration_noise: Noise for exploration
        """
        self.gamma = gamma
        self.tau = tau
        self.exploration_noise = exploration_noise
        self.use_torch = HAS_TORCH

        if HAS_TORCH:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            # Actor networks
            self.actor = Actor(state_dim, action_dim, hidden_dim).to(self.device)
            self.actor_target = Actor(state_dim, action_dim, hidden_dim).to(self.device)
            self.actor_target.load_state_dict(self.actor.state_dict())

            # Critic networks
            self.critic = Critic(state_dim, action_dim, hidden_dim).to(self.device)
            self.critic_target = Critic(state_dim, action_dim, hidden_dim).to(self.device)
            self.critic_target.load_state_dict(self.critic.state_dict())

            # Optimizers
            self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr_actor)
            self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr_critic)

        # Replay buffer
        self.replay_buffer = ReplayBuffer()

        # Training stats
        self.train_steps = 0
        self.episodes = 0

    def select_action(self,
                     state: ExecutionState,
                     explore: bool = True) -> ExecutionAction:
        """
        Select action given current state.

        Args:
            state: Current execution state
            explore: Whether to add exploration noise

        Returns:
            ExecutionAction with decisions
        """
        if self.use_torch:
            state_tensor = state.to_tensor().unsqueeze(0).to(self.device)

            with torch.no_grad():
                action = self.actor(state_tensor).cpu().numpy()[0]

            # Add exploration noise
            if explore:
                noise = np.random.normal(0, self.exploration_noise, size=action.shape)
                action = np.clip(action + noise, -1, 1)

            return ExecutionAction.from_array(action)
        else:
            # Rule-based fallback
            return self._rule_based_action(state)

    def _rule_based_action(self, state: ExecutionState) -> ExecutionAction:
        """Rule-based action selection (fallback when no PyTorch)."""
        # More aggressive when time running out
        urgency = 1 - state.time_remaining_pct

        # Use limit when spread is tight and we have time
        use_limit = (state.spread_bps < 1.5 and
                    state.time_remaining_pct > 0.3 and
                    state.remaining_qty_pct > 0.2)

        # Trade rate based on urgency and liquidity
        trade_rate = 1.0 + urgency * 0.5 * state.session_liquidity

        # Limit offset based on spread
        limit_offset = state.spread_bps * 0.3 if use_limit else 0

        # Aggressiveness from multiple factors
        aggressiveness = 0.3 + 0.4 * urgency + 0.3 * (1 - state.remaining_qty_pct)

        return ExecutionAction(
            trade_rate=np.clip(trade_rate, 0.5, 2.0),
            use_limit=use_limit,
            limit_offset_bps=limit_offset,
            aggressiveness=np.clip(aggressiveness, 0, 1)
        )

    def store_experience(self, experience: Experience):
        """Store experience in replay buffer."""
        self.replay_buffer.push(experience)

    def train(self, batch_size: int = 64) -> Dict[str, float]:
        """
        Train the agent on a batch of experiences.

        Returns:
            Dictionary with training metrics
        """
        if not self.use_torch:
            return {'status': 'no_torch'}

        if len(self.replay_buffer) < batch_size:
            return {'status': 'insufficient_data'}

        # Sample batch
        batch = self.replay_buffer.sample(batch_size)

        # Convert to tensors
        states = torch.stack([e.state.to_tensor() for e in batch]).to(self.device)
        actions = torch.FloatTensor([e.action for e in batch]).to(self.device)
        rewards = torch.FloatTensor([e.reward for e in batch]).unsqueeze(1).to(self.device)
        next_states = torch.stack([e.next_state.to_tensor() for e in batch]).to(self.device)
        dones = torch.FloatTensor([float(e.done) for e in batch]).unsqueeze(1).to(self.device)

        # Update critic
        with torch.no_grad():
            next_actions = self.actor_target(next_states)
            target_q = rewards + (1 - dones) * self.gamma * self.critic_target(next_states, next_actions)

        current_q = self.critic(states, actions)
        critic_loss = F.mse_loss(current_q, target_q)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Update actor
        actor_loss = -self.critic(states, self.actor(states)).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Soft update target networks
        self._soft_update(self.actor, self.actor_target)
        self._soft_update(self.critic, self.critic_target)

        self.train_steps += 1

        return {
            'critic_loss': critic_loss.item(),
            'actor_loss': actor_loss.item(),
            'train_steps': self.train_steps
        }

    def _soft_update(self, source: 'nn.Module', target: 'nn.Module'):
        """Soft update target network."""
        for target_param, source_param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(
                self.tau * source_param.data + (1 - self.tau) * target_param.data
            )

    def save(self, path: str):
        """Save model to file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        if self.use_torch:
            torch.save({
                'actor': self.actor.state_dict(),
                'actor_target': self.actor_target.state_dict(),
                'critic': self.critic.state_dict(),
                'critic_target': self.critic_target.state_dict(),
                'actor_optimizer': self.actor_optimizer.state_dict(),
                'critic_optimizer': self.critic_optimizer.state_dict(),
                'train_steps': self.train_steps,
                'episodes': self.episodes
            }, path)
        else:
            # Save empty marker
            with open(path, 'wb') as f:
                pickle.dump({'use_torch': False}, f)

    def load(self, path: str):
        """Load model from file."""
        path = Path(path)
        if not path.exists():
            logger.warning(f"Model file not found: {path}")
            return

        if self.use_torch:
            checkpoint = torch.load(path, map_location=self.device)
            self.actor.load_state_dict(checkpoint['actor'])
            self.actor_target.load_state_dict(checkpoint['actor_target'])
            self.critic.load_state_dict(checkpoint['critic'])
            self.critic_target.load_state_dict(checkpoint['critic_target'])
            self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer'])
            self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer'])
            self.train_steps = checkpoint.get('train_steps', 0)
            self.episodes = checkpoint.get('episodes', 0)


class RLExecutionOptimizer:
    """
    High-level RL-based execution optimizer.

    Wraps DDPG agent with execution-specific logic.
    """

    def __init__(self, model_path: Optional[str] = None):
        self.agent = DDPGExecutor()

        if model_path and Path(model_path).exists():
            self.agent.load(model_path)

        # Execution tracking
        self.current_executions: Dict[str, dict] = {}

    def start_execution(self,
                       order_id: str,
                       total_quantity: float,
                       horizon_seconds: float,
                       symbol: str) -> ExecutionAction:
        """
        Start tracking an execution and get initial action.

        Returns initial action recommendation.
        """
        initial_state = ExecutionState(
            remaining_qty_pct=1.0,
            time_remaining_pct=1.0,
            spread_bps=1.0,  # Will be updated
            volatility=0.0001,
            session_liquidity=1.0,
            order_flow=0.0,
            fill_rate=0.8,
            slippage_so_far_bps=0.0
        )

        self.current_executions[order_id] = {
            'total_qty': total_quantity,
            'executed_qty': 0.0,
            'horizon': horizon_seconds,
            'elapsed': 0.0,
            'symbol': symbol,
            'last_state': initial_state,
            'cumulative_cost': 0.0,
            'experiences': []
        }

        return self.agent.select_action(initial_state, explore=True)

    def update_and_decide(self,
                         order_id: str,
                         executed_qty: float,
                         elapsed_seconds: float,
                         current_spread_bps: float,
                         current_volatility: float,
                         session_liquidity: float,
                         order_flow: float,
                         execution_cost_bps: float) -> ExecutionAction:
        """
        Update state and get next action.

        Args:
            order_id: Order identifier
            executed_qty: Quantity executed so far
            elapsed_seconds: Time elapsed
            current_spread_bps: Current spread
            current_volatility: Current volatility
            session_liquidity: Session liquidity factor
            order_flow: Order flow imbalance
            execution_cost_bps: Execution cost so far

        Returns:
            Next action recommendation
        """
        if order_id not in self.current_executions:
            # Return default action
            return ExecutionAction(
                trade_rate=1.0,
                use_limit=False,
                limit_offset_bps=0.0,
                aggressiveness=0.5
            )

        exec_state = self.current_executions[order_id]

        # Calculate new state
        remaining_qty = exec_state['total_qty'] - executed_qty
        remaining_time = exec_state['horizon'] - elapsed_seconds

        new_state = ExecutionState(
            remaining_qty_pct=remaining_qty / exec_state['total_qty'],
            time_remaining_pct=max(0, remaining_time / exec_state['horizon']),
            spread_bps=current_spread_bps,
            volatility=current_volatility,
            session_liquidity=session_liquidity,
            order_flow=order_flow,
            fill_rate=executed_qty / max(1, exec_state['elapsed']) * 60,  # Fills per minute
            slippage_so_far_bps=execution_cost_bps
        )

        # Calculate reward (negative cost is good)
        reward = -execution_cost_bps

        # Store experience
        if exec_state['last_state'] is not None:
            experience = Experience(
                state=exec_state['last_state'],
                action=np.zeros(4),  # Will be filled from last action
                reward=reward,
                next_state=new_state,
                done=remaining_qty <= 0 or remaining_time <= 0
            )
            self.agent.store_experience(experience)

        # Update tracking
        exec_state['executed_qty'] = executed_qty
        exec_state['elapsed'] = elapsed_seconds
        exec_state['last_state'] = new_state
        exec_state['cumulative_cost'] = execution_cost_bps

        # Get next action
        is_done = new_state.remaining_qty_pct <= 0.01 or new_state.time_remaining_pct <= 0.01

        if is_done:
            # Cleanup
            del self.current_executions[order_id]
            return ExecutionAction(
                trade_rate=2.0,  # Execute remaining fast
                use_limit=False,
                limit_offset_bps=0.0,
                aggressiveness=1.0
            )

        return self.agent.select_action(new_state, explore=True)

    def train_from_buffer(self, batch_size: int = 64) -> Dict[str, float]:
        """Train agent from replay buffer."""
        return self.agent.train(batch_size)

    def save_model(self, path: str):
        """Save model."""
        self.agent.save(path)


def get_rl_executor(model_path: Optional[str] = None) -> RLExecutionOptimizer:
    """Factory function to get RL executor."""
    return RLExecutionOptimizer(model_path)


if __name__ == '__main__':
    print("DDPG Execution Agent Test")
    print("=" * 70)
    print(f"PyTorch available: {HAS_TORCH}")

    optimizer = RLExecutionOptimizer()

    # Simulate an execution
    print("\nSimulating 1M EURUSD execution over 5 minutes")
    print("-" * 70)

    order_id = "TEST_001"
    total_qty = 1_000_000
    horizon = 300

    # Start execution
    action = optimizer.start_execution(
        order_id=order_id,
        total_quantity=total_qty,
        horizon_seconds=horizon,
        symbol='EURUSD'
    )

    print(f"Initial action: rate={action.trade_rate:.2f}, "
          f"limit={action.use_limit}, offset={action.limit_offset_bps:.1f}bps, "
          f"aggr={action.aggressiveness:.2f}")

    # Simulate updates
    for step in range(10):
        elapsed = (step + 1) * 30  # 30 second intervals
        executed = (step + 1) * 100000  # 100k per step

        # Simulate varying conditions
        spread = 0.8 + 0.2 * np.sin(step)
        vol = 0.0001 * (1 + 0.3 * np.random.randn())
        liquidity = 1.2 - 0.4 * (step / 10)
        flow = 0.2 * np.random.randn()
        cost = 0.5 + 0.1 * step

        action = optimizer.update_and_decide(
            order_id=order_id,
            executed_qty=executed,
            elapsed_seconds=elapsed,
            current_spread_bps=spread,
            current_volatility=vol,
            session_liquidity=liquidity,
            order_flow=flow,
            execution_cost_bps=cost
        )

        print(f"Step {step+1}: exec={executed/1e6:.1f}M, "
              f"rate={action.trade_rate:.2f}, limit={action.use_limit}, "
              f"aggr={action.aggressiveness:.2f}")

    # Try training
    print("\n" + "=" * 70)
    print("Training test")
    print("-" * 70)

    # Add some dummy experiences
    for _ in range(100):
        state = ExecutionState(
            remaining_qty_pct=np.random.random(),
            time_remaining_pct=np.random.random(),
            spread_bps=np.random.uniform(0.5, 2.0),
            volatility=np.random.uniform(0.00005, 0.0002),
            session_liquidity=np.random.uniform(0.2, 1.8),
            order_flow=np.random.uniform(-1, 1),
            fill_rate=np.random.uniform(0, 1),
            slippage_so_far_bps=np.random.uniform(0, 5)
        )
        next_state = ExecutionState(
            remaining_qty_pct=state.remaining_qty_pct - 0.1,
            time_remaining_pct=state.time_remaining_pct - 0.1,
            spread_bps=state.spread_bps,
            volatility=state.volatility,
            session_liquidity=state.session_liquidity,
            order_flow=state.order_flow,
            fill_rate=state.fill_rate,
            slippage_so_far_bps=state.slippage_so_far_bps + 0.5
        )

        exp = Experience(
            state=state,
            action=np.random.randn(4) * 0.5,
            reward=-np.random.uniform(0, 2),
            next_state=next_state,
            done=np.random.random() < 0.1
        )
        optimizer.agent.store_experience(exp)

    # Train
    metrics = optimizer.train_from_buffer(batch_size=32)
    print(f"Training metrics: {metrics}")
