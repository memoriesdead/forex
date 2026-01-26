#!/usr/bin/env python3
"""
Train RL Position Sizer - Risk-Sensitive Reinforcement Learning
================================================================
Trains CVaR-PPO agents for position sizing with:
- Tail risk control (CVaR)
- Drawdown constraints
- Kelly-optimal sizing

Uses the RL agents from core/rl/:
- CVaRPPO: CVaR-constrained PPO
- DrawdownConstrainedPPO: Max drawdown limits
- TailSafePPO: Combined tail risk protection

Usage:
    # Train RL sizer for all pairs
    python scripts/train_rl_sizer.py --all

    # Train for specific pairs
    python scripts/train_rl_sizer.py --pairs EURUSD,GBPUSD

    # Use specific agent
    python scripts/train_rl_sizer.py --all --agent cvar_ppo

    # Evaluate trained agents
    python scripts/train_rl_sizer.py --evaluate
"""

import argparse
import json
import sys
import gc
import time
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional, Tuple
import logging

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import torch
import gymnasium as gym
from gymnasium import spaces

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')
logger = logging.getLogger(__name__)


class ForexTradingEnv(gym.Env):
    """
    Forex trading environment for RL position sizing.

    State:
        - ML model prediction probability
        - Recent returns
        - Current position
        - Portfolio value
        - Drawdown

    Action:
        - Position size (-1 to 1) or (0, 0.25, 0.5, 0.75, 1.0)

    Reward:
        - Risk-adjusted PnL
        - CVaR penalty for tail losses
        - Drawdown penalty
    """

    metadata = {'render_modes': ['human']}

    def __init__(
        self,
        returns: np.ndarray,
        predictions: np.ndarray,
        initial_capital: float = 100000,
        max_position: float = 1.0,
        transaction_cost: float = 0.0001,  # 1 pip
        max_drawdown: float = 0.05,
        cvar_alpha: float = 0.05,
    ):
        """
        Initialize trading environment.

        Args:
            returns: Array of historical returns
            predictions: Array of ML model predictions (0-1)
            initial_capital: Starting capital
            max_position: Maximum position size (fraction of capital)
            transaction_cost: Cost per trade (fraction)
            max_drawdown: Maximum allowed drawdown
            cvar_alpha: CVaR percentile (e.g., 0.05 for 5% worst cases)
        """
        super().__init__()

        self.returns = returns
        self.predictions = predictions
        self.initial_capital = initial_capital
        self.max_position = max_position
        self.transaction_cost = transaction_cost
        self.max_drawdown = max_drawdown
        self.cvar_alpha = cvar_alpha

        # State space: [prediction, return_5, return_20, position, drawdown, volatility]
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(6,), dtype=np.float32
        )

        # Action space: position size from -1 to 1
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(1,), dtype=np.float32
        )

        # State
        self.current_step = 0
        self.position = 0.0
        self.portfolio_value = initial_capital
        self.max_portfolio_value = initial_capital
        self.pnl_history = []

    def reset(self, seed=None, options=None):
        """Reset environment to initial state."""
        super().reset(seed=seed)

        self.current_step = 50  # Start after warmup period
        self.position = 0.0
        self.portfolio_value = self.initial_capital
        self.max_portfolio_value = self.initial_capital
        self.pnl_history = []

        return self._get_observation(), {}

    def _get_observation(self) -> np.ndarray:
        """Get current observation."""
        # ML prediction
        pred = self.predictions[self.current_step]

        # Recent returns
        ret_5 = np.sum(self.returns[self.current_step-5:self.current_step])
        ret_20 = np.sum(self.returns[self.current_step-20:self.current_step])

        # Current position
        pos = self.position

        # Drawdown
        drawdown = (self.max_portfolio_value - self.portfolio_value) / self.max_portfolio_value

        # Volatility
        vol = np.std(self.returns[self.current_step-20:self.current_step])

        return np.array([pred, ret_5, ret_20, pos, drawdown, vol], dtype=np.float32)

    def step(self, action):
        """Execute one step in the environment."""
        # Get target position
        target_position = float(action[0]) * self.max_position

        # Transaction cost for changing position
        position_change = abs(target_position - self.position)
        cost = position_change * self.transaction_cost * self.portfolio_value

        # Update position
        old_position = self.position
        self.position = target_position

        # Get return for this step
        ret = self.returns[self.current_step]

        # Calculate PnL
        pnl = self.position * ret * self.portfolio_value - cost
        self.portfolio_value += pnl
        self.pnl_history.append(pnl)

        # Update max portfolio value
        self.max_portfolio_value = max(self.max_portfolio_value, self.portfolio_value)

        # Calculate drawdown
        drawdown = (self.max_portfolio_value - self.portfolio_value) / self.max_portfolio_value

        # Calculate reward
        reward = self._calculate_reward(pnl, drawdown)

        # Check if done
        self.current_step += 1
        done = (
            self.current_step >= len(self.returns) - 1 or
            self.portfolio_value <= 0 or
            drawdown >= self.max_drawdown
        )

        return self._get_observation(), reward, done, False, {
            'portfolio_value': self.portfolio_value,
            'drawdown': drawdown,
            'position': self.position,
        }

    def _calculate_reward(self, pnl: float, drawdown: float) -> float:
        """
        Calculate risk-adjusted reward.

        Combines:
        - PnL (normalized by initial capital)
        - CVaR penalty (for tail losses)
        - Drawdown penalty
        """
        # Base reward: PnL
        reward = pnl / self.initial_capital

        # CVaR penalty: penalize being in worst alpha% of outcomes
        if len(self.pnl_history) >= 20:
            sorted_pnl = np.sort(self.pnl_history)
            var_idx = int(self.cvar_alpha * len(sorted_pnl))
            cvar = np.mean(sorted_pnl[:max(1, var_idx)])
            if cvar < 0:
                reward += 0.1 * cvar / self.initial_capital  # Penalty

        # Drawdown penalty
        if drawdown > self.max_drawdown * 0.5:  # Warning at 50% of max
            reward -= 0.5 * (drawdown - self.max_drawdown * 0.5)

        if drawdown >= self.max_drawdown:
            reward -= 10.0  # Large penalty for hitting max drawdown

        return float(reward)


class RLPositionSizerTrainer:
    """
    Trainer for RL position sizing agents.

    Trains CVaR-PPO, DrawdownPPO, or TailSafePPO agents
    to learn optimal position sizing.
    """

    def __init__(
        self,
        training_dir: Optional[Path] = None,
        output_dir: Optional[Path] = None,
        agent_type: str = 'cvar_ppo',
        total_timesteps: int = 100000,
        learning_rate: float = 3e-4,
    ):
        """
        Initialize trainer.

        Args:
            training_dir: Directory with training data
            output_dir: Directory to save trained agents
            agent_type: Type of RL agent ('cvar_ppo', 'drawdown_ppo', 'tail_safe')
            total_timesteps: Total training timesteps
            learning_rate: Learning rate
        """
        self.training_dir = training_dir or PROJECT_ROOT / 'training_package'
        self.output_dir = output_dir or PROJECT_ROOT / 'models' / 'production' / 'rl'
        self.agent_type = agent_type
        self.total_timesteps = total_timesteps
        self.learning_rate = learning_rate

        self.output_dir.mkdir(parents=True, exist_ok=True)

    def get_available_pairs(self) -> List[str]:
        """Get pairs with training data."""
        pairs = []
        for d in self.training_dir.iterdir():
            if d.is_dir() and (d / 'train.parquet').exists():
                if d.name not in ['features_cache', '__pycache__']:
                    pairs.append(d.name)
        return sorted(pairs)

    def load_data_for_pair(self, pair: str) -> Tuple[np.ndarray, np.ndarray]:
        """Load returns and predictions for a pair."""
        pair_dir = self.training_dir / pair

        # Load training data
        train = pd.read_parquet(pair_dir / 'train.parquet')

        # Get returns
        if 'close' not in train.columns and 'mid' in train.columns:
            train['close'] = train['mid']

        returns = train['close'].pct_change().fillna(0).values

        # Get predictions (or generate random for training)
        # In production, this would come from the ML model
        # For training, we use random predictions to learn position sizing
        predictions = np.random.rand(len(returns))

        return returns, predictions

    def train_pair(self, pair: str) -> Dict:
        """
        Train RL agent for a single pair.

        Args:
            pair: Currency pair symbol

        Returns:
            Training results dict
        """
        logger.info(f"Training {self.agent_type} for {pair}...")

        # Load data
        returns, predictions = self.load_data_for_pair(pair)

        if len(returns) < 1000:
            logger.warning(f"  Too few samples for {pair}")
            return {'success': False, 'error': 'Too few samples'}

        # Create environment
        env = ForexTradingEnv(returns, predictions)

        # Create agent
        from core.rl.risk_sensitive import (
            CVaRPPO,
            DrawdownConstrainedPPO,
            TailSafePPO,
            RiskSensitiveConfig,
        )

        config = RiskSensitiveConfig(
            learning_rate=self.learning_rate,
            hidden_dims=[256, 128, 64],
            gamma=0.99,
            ppo_epochs=10,
            ppo_clip=0.2,
        )

        if self.agent_type == 'cvar_ppo':
            agent = CVaRPPO(env, config)
        elif self.agent_type == 'drawdown_ppo':
            agent = DrawdownConstrainedPPO(env, config)
        elif self.agent_type == 'tail_safe':
            agent = TailSafePPO(env, config)
        else:
            raise ValueError(f"Unknown agent type: {self.agent_type}")

        # Training loop
        episode_rewards = []
        episode_lengths = []

        state, _ = env.reset()
        episode_reward = 0
        episode_length = 0

        for step in range(self.total_timesteps):
            # Select action
            action = agent.select_action(state)

            # Step environment
            next_state, reward, done, truncated, info = env.step(action)

            episode_reward += reward
            episode_length += 1

            state = next_state

            if done or truncated:
                episode_rewards.append(episode_reward)
                episode_lengths.append(episode_length)

                state, _ = env.reset()
                episode_reward = 0
                episode_length = 0

            # Log progress
            if (step + 1) % 10000 == 0:
                if episode_rewards:
                    avg_reward = np.mean(episode_rewards[-10:])
                    logger.info(f"  Step {step+1}: Avg reward = {avg_reward:.4f}")

        # Save agent
        agent_path = self.output_dir / f"{pair}_{self.agent_type}.pt"
        torch.save(agent.state_dict(), agent_path)

        results = {
            'success': True,
            'pair': pair,
            'agent_type': self.agent_type,
            'total_episodes': len(episode_rewards),
            'avg_reward': float(np.mean(episode_rewards)) if episode_rewards else 0,
            'max_reward': float(np.max(episode_rewards)) if episode_rewards else 0,
            'saved_to': str(agent_path),
        }

        logger.info(f"  Completed: avg_reward = {results['avg_reward']:.4f}")

        return results

    def train_all(self, pairs: Optional[List[str]] = None) -> Dict:
        """Train RL agents for all pairs."""
        if pairs is None:
            pairs = self.get_available_pairs()

        results = {
            'success': [],
            'failed': [],
            'metrics': {},
        }

        for i, pair in enumerate(pairs):
            logger.info(f"\n[{i+1}/{len(pairs)}] {pair}")

            try:
                pair_results = self.train_pair(pair)

                if pair_results['success']:
                    results['success'].append(pair)
                    results['metrics'][pair] = pair_results
                else:
                    results['failed'].append(pair)

            except Exception as e:
                logger.error(f"  Failed: {e}")
                results['failed'].append(pair)

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # Save summary
        summary_path = self.output_dir / 'rl_training_summary.json'
        with open(summary_path, 'w') as f:
            json.dump(results, f, indent=2)

        logger.info(f"\nRL Training Complete")
        logger.info(f"  Success: {len(results['success'])}")
        logger.info(f"  Failed: {len(results['failed'])}")
        logger.info(f"  Results saved to: {summary_path}")

        return results


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Train RL Position Sizer'
    )

    parser.add_argument('--all', action='store_true', help='Train for all pairs')
    parser.add_argument('--pairs', type=str, help='Comma-separated pairs')
    parser.add_argument('--agent', type=str, default='cvar_ppo',
                       choices=['cvar_ppo', 'drawdown_ppo', 'tail_safe'],
                       help='RL agent type')
    parser.add_argument('--timesteps', type=int, default=100000,
                       help='Total training timesteps')
    parser.add_argument('--lr', type=float, default=3e-4,
                       help='Learning rate')
    parser.add_argument('--evaluate', action='store_true',
                       help='Evaluate trained agents')

    args = parser.parse_args()

    trainer = RLPositionSizerTrainer(
        agent_type=args.agent,
        total_timesteps=args.timesteps,
        learning_rate=args.lr,
    )

    if args.all:
        trainer.train_all()

    elif args.pairs:
        pairs = [p.strip() for p in args.pairs.split(',')]
        trainer.train_all(pairs)

    elif args.evaluate:
        # TODO: Implement evaluation
        logger.info("Evaluation not yet implemented")

    else:
        parser.print_help()


if __name__ == '__main__':
    main()
