#!/usr/bin/env python3
"""
RL Optimization HTTP API - Chinese & Global RL Math for Price Prediction
=========================================================================
REST API implementing all researched RL formulas for forex price prediction.

Research Sources:
- MacroHFT (KDD 2024): Memory Augmented Context-aware RL
- EarnHFT (AAAI 2024): Efficient Hierarchical RL for HFT
- Logic-Q (AAAI 2024): Logic-Guided DRL
- 华泰证券 DQN择时: Chinese securities research
- Risk-Aware RL (arXiv 2025): Composite reward functions
- FinRL Framework: Open-source DRL for trading

Mathematical Formulas Implemented:
- Bellman Equations (V*, Q*)
- TD Error & Advantage Function
- Policy Gradient Theorem
- PPO Clipped Objective
- SAC Entropy-Regularized
- Sharpe Ratio Reward
- Composite Risk-Aware Reward
- Hierarchical Q-Functions
- Memory Module Attention

Endpoints:
- GET  /health                              Health check
- GET  /api/rl/formulas                     List all implemented formulas
- POST /api/rl/bellman/value                Calculate V*(s) Bellman value
- POST /api/rl/bellman/action               Calculate Q*(s,a) Bellman action value
- POST /api/rl/td_error                     Calculate TD error δ
- POST /api/rl/advantage                    Calculate advantage A(s,a)
- POST /api/rl/gae                          Generalized Advantage Estimation
- POST /api/rl/policy_gradient              Policy gradient ∇J(θ)
- POST /api/rl/ppo_clip                     PPO clipped objective
- POST /api/rl/reward/sharpe                Sharpe ratio reward
- POST /api/rl/reward/composite             Risk-aware composite reward
- POST /api/rl/reward/differential_sharpe   Differential Sharpe ratio
- POST /api/rl/hierarchical_q               Hierarchical Q-function (MacroHFT)
- POST /api/rl/memory_attention             Memory module attention (MacroHFT)
- POST /api/rl/logic_trend                  Logic-Q market trend detection
- POST /api/rl/metrics                      Financial metrics (Sharpe, Sortino, Calmar)

Usage:
    python scripts/mcp_rl_optimization_api.py --port 8083

Citations:
    - MacroHFT: https://arxiv.org/html/2406.14537v1
    - EarnHFT: https://arxiv.org/abs/2309.12891
    - Logic-Q: https://arxiv.org/abs/2310.05551
    - 华泰DQN: https://asset.quant-wiki.com/pdf/华泰人工智能系列59
    - FinRL: https://github.com/AI4Finance-Foundation/FinRL
"""

import argparse
import json
import logging
import math
from datetime import datetime
from http.server import HTTPServer, BaseHTTPRequestHandler
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlparse
import sys
from pathlib import Path
import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# =============================================================================
# MATHEMATICAL FORMULAS - Core Implementation
# =============================================================================

class BellmanEquations:
    """
    Bellman Equations for Value and Action-Value Functions.

    Sources:
    - 知乎 Bellman: https://zhuanlan.zhihu.com/p/139559442
    - Deep Learning Wizard MDP: https://www.deeplearningwizard.com/
    """

    @staticmethod
    def value_function(rewards: np.ndarray, gamma: float = 0.99) -> float:
        """
        V*(s) = E[r + γV*(s')]

        Calculate optimal state value using discounted rewards.
        """
        T = len(rewards)
        V = 0.0
        for t in range(T):
            V += (gamma ** t) * rewards[t]
        return V

    @staticmethod
    def action_value(reward: float, gamma: float, next_value: float) -> float:
        """
        Q*(s,a) = E[r + γ max_a' Q*(s',a')]

        Bellman optimality equation for action-value.
        """
        return reward + gamma * next_value

    @staticmethod
    def td_target(reward: float, gamma: float, next_value: float) -> float:
        """
        TD Target = r + γV(s')
        """
        return reward + gamma * next_value


class TDLearning:
    """
    Temporal Difference Learning Formulas.

    Sources:
    - Actor-Critic Medium: https://medium.com/intro-to-artificial-intelligence/
    - TensorFlow Tutorial: https://www.tensorflow.org/tutorials/reinforcement_learning/actor_critic
    """

    @staticmethod
    def td_error(reward: float, gamma: float, current_value: float, next_value: float) -> float:
        """
        δ_t = r_t + γV(s_{t+1}) - V(s_t)

        TD error is the advantage function estimate.
        """
        return reward + gamma * next_value - current_value

    @staticmethod
    def advantage(q_value: float, v_value: float) -> float:
        """
        A(s_t, a_t) = Q(s_t, a_t) - V(s_t)

        Advantage function measures how much better action is than average.
        """
        return q_value - v_value

    @staticmethod
    def gae(rewards: np.ndarray, values: np.ndarray, gamma: float = 0.99, lam: float = 0.95) -> np.ndarray:
        """
        Generalized Advantage Estimation (GAE).

        Â_t^GAE(γ,λ) = Σ_{l=0}^∞ (γλ)^l δ_{t+l}

        Balances bias-variance tradeoff in advantage estimation.
        """
        T = len(rewards)
        advantages = np.zeros(T)
        last_gae = 0.0

        for t in reversed(range(T)):
            if t == T - 1:
                next_value = 0.0
            else:
                next_value = values[t + 1]

            delta = rewards[t] + gamma * next_value - values[t]
            advantages[t] = last_gae = delta + gamma * lam * last_gae

        return advantages


class PolicyGradient:
    """
    Policy Gradient Formulas.

    Sources:
    - 知乎 Policy Gradient: https://zhuanlan.zhihu.com/p/55366623
    - Lil'Log: https://lilianweng.github.io/posts/2018-04-08-policy-gradient/
    """

    @staticmethod
    def reinforce_return(rewards: np.ndarray, gamma: float = 0.99) -> np.ndarray:
        """
        G_t = Σ_{k=t}^T γ^{k-t} r_k

        Calculate returns for REINFORCE algorithm.
        """
        T = len(rewards)
        returns = np.zeros(T)
        G = 0.0

        for t in reversed(range(T)):
            G = rewards[t] + gamma * G
            returns[t] = G

        return returns

    @staticmethod
    def ppo_clip_objective(
        ratio: float,
        advantage: float,
        epsilon: float = 0.2
    ) -> float:
        """
        L^CLIP(θ) = min(r_t(θ)Â_t, clip(r_t(θ), 1-ε, 1+ε)Â_t)

        PPO clipped surrogate objective.
        Source: PPO Paper, 华泰证券研报
        """
        clipped_ratio = np.clip(ratio, 1 - epsilon, 1 + epsilon)
        return min(ratio * advantage, clipped_ratio * advantage)

    @staticmethod
    def policy_ratio(new_prob: float, old_prob: float) -> float:
        """
        r_t(θ) = π_θ(a_t|s_t) / π_θ_old(a_t|s_t)

        Probability ratio for importance sampling.
        """
        return new_prob / (old_prob + 1e-10)


class RewardFunctions:
    """
    Reward Function Formulas for Trading.

    Sources:
    - MacroHFT: https://arxiv.org/html/2406.14537v1
    - Risk-Aware RL: https://arxiv.org/html/2506.04358v1
    - Teddy Koker: https://teddykoker.com/2019/06/trading-with-reinforcement-learning-in-python-part-ii-application/
    """

    @staticmethod
    def macrohft_reward(
        action: float,
        price_current: float,
        price_next: float,
        position: float,
        transaction_cost: float = 0.001,
        holding_size: float = 1.0
    ) -> float:
        """
        r_t = (a_t × (p_{t+1} - p_t) - δ × |a_t - P_t|) × m

        MacroHFT reward with transaction costs.
        Source: https://arxiv.org/html/2406.14537v1
        """
        price_change = price_next - price_current
        position_change_cost = transaction_cost * abs(action - position)
        return (action * price_change - position_change_cost) * holding_size

    @staticmethod
    def sharpe_ratio_reward(returns: np.ndarray, risk_free: float = 0.0) -> float:
        """
        S_T = A / √(B - A²)
        where A = (1/T)ΣR_t, B = (1/T)ΣR_t²

        Sharpe ratio as reward signal.
        Source: Gabriel Molina RRL, Teddy Koker
        """
        if len(returns) < 2:
            return 0.0

        A = np.mean(returns)
        B = np.mean(returns ** 2)

        variance = B - A ** 2
        if variance <= 0:
            return 0.0

        return (A - risk_free) / np.sqrt(variance)

    @staticmethod
    def differential_sharpe_ratio(
        return_t: float,
        mean_prev: float,
        std_prev: float,
        eta: float = 0.01
    ) -> float:
        """
        DSR_t = (R_t - Ā_{t-1}) / σ_{t-1} - (1/2)(Ā_{t-1}/σ_{t-1})((R_t - Ā_{t-1})²/σ_{t-1}² - 1)

        Differential Sharpe ratio for online learning.
        """
        if std_prev <= 0:
            return 0.0

        normalized_return = (return_t - mean_prev) / std_prev
        adjustment = 0.5 * (mean_prev / std_prev) * (normalized_return ** 2 - 1)

        return normalized_return - adjustment

    @staticmethod
    def composite_reward(
        returns: np.ndarray,
        w1: float = 1.0,  # Annualized return weight
        w2: float = 0.5,  # Downside risk weight
        w3: float = 0.3,  # Differential return weight
        w4: float = 0.2,  # Treynor ratio weight
        risk_free: float = 0.0,
        beta: float = 1.0
    ) -> Dict[str, float]:
        """
        ℛ = w₁R_ann - w₂σ_down + w₃D_ret + w₄T_ry

        Risk-aware composite reward function.
        Source: https://arxiv.org/html/2506.04358v1
        """
        if len(returns) < 2:
            return {'composite': 0.0, 'components': {}}

        # Annualized return (assuming daily returns, 252 trading days)
        total_return = np.prod(1 + returns) - 1
        T = len(returns)
        R_ann = (1 + total_return) ** (252 / T) - 1

        # Downside risk
        downside_returns = np.minimum(returns, 0)
        sigma_down = np.sqrt(np.mean(downside_returns ** 2))

        # Differential return (simplified)
        D_ret = np.mean(returns) / (beta + 1e-10)

        # Treynor ratio
        T_ry = (R_ann - risk_free) / (beta + 1e-10)

        # Composite reward
        composite = w1 * R_ann - w2 * sigma_down + w3 * D_ret + w4 * T_ry

        return {
            'composite': composite,
            'components': {
                'annualized_return': R_ann,
                'downside_risk': sigma_down,
                'differential_return': D_ret,
                'treynor_ratio': T_ry
            }
        }

    @staticmethod
    def logic_q_order_execution_reward(
        action: float,
        price_next: float,
        price_avg: float,
        alpha: float = 0.01
    ) -> float:
        """
        R_t^OE(s_t, a_t) = a_t(p_{t+1}/p̃ - 1) - α(a_t)²

        Logic-Q order execution reward.
        Source: https://arxiv.org/abs/2310.05551
        """
        price_ratio = price_next / price_avg - 1
        return action * price_ratio - alpha * (action ** 2)


class HierarchicalRL:
    """
    Hierarchical RL Formulas (MacroHFT, EarnHFT).

    Sources:
    - MacroHFT: https://arxiv.org/html/2406.14537v1
    - EarnHFT: https://arxiv.org/abs/2309.12891
    """

    @staticmethod
    def hierarchical_q(
        sub_q_values: List[float],
        weights: List[float]
    ) -> float:
        """
        Q^hyper = Σᵢ wᵢQᵢ^sub

        Hierarchical Q-function combines sub-agent Q-values.
        Source: MacroHFT
        """
        if len(sub_q_values) != len(weights):
            raise ValueError("Q-values and weights must have same length")

        return sum(w * q for w, q in zip(weights, sub_q_values))

    @staticmethod
    def memory_attention(
        query_key: np.ndarray,
        memory_keys: np.ndarray,
        memory_values: np.ndarray,
        actions: np.ndarray,
        current_action: int,
        epsilon: float = 1e-6
    ) -> Tuple[float, np.ndarray]:
        """
        Memory Module from MacroHFT.

        d(k, kᵢ) = ||k - kᵢ||₂² + ε
        wᵢ = d(k, kᵢ)1_{a=aᵢ} / Σ d(k, kᵢ)1_{a=aᵢ}
        Q_m(s,a) = Σ wᵢvᵢ

        Source: https://arxiv.org/html/2406.14537v1
        """
        # Calculate distances
        distances = np.sum((memory_keys - query_key) ** 2, axis=1) + epsilon

        # Filter by action
        action_mask = (actions == current_action).astype(float)
        masked_distances = distances * action_mask

        # Normalize weights
        total = np.sum(masked_distances)
        if total > 0:
            weights = masked_distances / total
        else:
            weights = np.zeros_like(distances)

        # Calculate memory Q-value
        Q_memory = np.sum(weights * memory_values)

        return Q_memory, weights

    @staticmethod
    def dueling_q_network(
        value: float,
        advantages: np.ndarray
    ) -> np.ndarray:
        """
        Q^sub(h,a) = V(h) + (Adv(h,a) - 1/|A| Σ_a' Adv(h,a'))

        Dueling network architecture Q-value.
        Source: MacroHFT
        """
        mean_advantage = np.mean(advantages)
        return value + (advantages - mean_advantage)


class LogicQ:
    """
    Logic-Q Market Trend Detection.

    Source: https://arxiv.org/abs/2310.05551
    """

    @staticmethod
    def volatility(prices: np.ndarray, window: int = 20) -> float:
        """
        vol_g(t) = (1/G)Σᵢ (xᵢ - x̄)²

        Rolling volatility calculation.
        """
        if len(prices) < window:
            return 0.0

        returns = np.diff(prices[-window:]) / prices[-window:-1]
        return np.var(returns)

    @staticmethod
    def downside_risk(prices: np.ndarray, window: int = 20) -> float:
        """
        dr_g(t) = √[(1/n)Σ_{x<x̄} (x̄ - x)²]

        Downside risk (semi-deviation).
        """
        if len(prices) < window:
            return 0.0

        returns = np.diff(prices[-window:]) / prices[-window:-1]
        mean_return = np.mean(returns)
        downside = returns[returns < mean_return]

        if len(downside) == 0:
            return 0.0

        return np.sqrt(np.mean((mean_return - downside) ** 2))

    @staticmethod
    def growth_rate(prices: np.ndarray) -> float:
        """
        gr_g(t) = (X_t - X_start)/X_start × 100%

        Growth rate from start.
        """
        if len(prices) < 2:
            return 0.0

        return (prices[-1] - prices[0]) / prices[0]

    @staticmethod
    def detect_market_trend(
        prices: np.ndarray,
        vol_threshold: float = 0.01,
        dsr_threshold: float = 0.005,
        growth_threshold: float = 0.02
    ) -> Dict[str, Any]:
        """
        Logic rule: slow-decline ← (VOL(t) < α) ∧ (DSR(t) > β)

        Detect market trend using Logic-Q rules.
        """
        vol = LogicQ.volatility(prices)
        dsr = LogicQ.downside_risk(prices)
        growth = LogicQ.growth_rate(prices)

        # Determine trend
        if vol < vol_threshold and dsr > dsr_threshold:
            trend = 'slow_decline'
        elif growth > growth_threshold and vol < vol_threshold:
            trend = 'steady_uptrend'
        elif growth < -growth_threshold:
            trend = 'downtrend'
        elif vol > vol_threshold * 2:
            trend = 'high_volatility'
        else:
            trend = 'sideways'

        return {
            'trend': trend,
            'volatility': vol,
            'downside_risk': dsr,
            'growth_rate': growth,
            'logic_rule': f"VOL={vol:.4f} (<{vol_threshold}), DSR={dsr:.4f} (>{dsr_threshold})"
        }


class FinancialMetrics:
    """
    Financial Performance Metrics.

    Sources:
    - MacroHFT: https://arxiv.org/html/2406.14537v1
    - Risk-Aware RL: https://arxiv.org/html/2506.04358v1
    """

    @staticmethod
    def sharpe_ratio(returns: np.ndarray, risk_free: float = 0.0, annualize: bool = True) -> float:
        """
        SR = (μ - r_f) / σ
        ASR = SR × √252 (if annualized)
        """
        if len(returns) < 2:
            return 0.0

        excess_returns = returns - risk_free
        sr = np.mean(excess_returns) / (np.std(excess_returns) + 1e-10)

        if annualize:
            sr *= np.sqrt(252)

        return sr

    @staticmethod
    def sortino_ratio(returns: np.ndarray, risk_free: float = 0.0, annualize: bool = True) -> float:
        """
        SoR = (μ - r_f) / σ_down
        """
        if len(returns) < 2:
            return 0.0

        excess_returns = returns - risk_free
        downside = returns[returns < 0]

        if len(downside) == 0:
            return float('inf')

        downside_std = np.std(downside)
        sor = np.mean(excess_returns) / (downside_std + 1e-10)

        if annualize:
            sor *= np.sqrt(252)

        return sor

    @staticmethod
    def calmar_ratio(returns: np.ndarray, annualize: bool = True) -> float:
        """
        ACR = E[r] / MDD × m
        """
        if len(returns) < 2:
            return 0.0

        # Calculate cumulative returns
        cumulative = np.cumprod(1 + returns)
        running_max = np.maximum.accumulate(cumulative)
        drawdown = (running_max - cumulative) / running_max
        mdd = np.max(drawdown)

        if mdd == 0:
            return float('inf')

        mean_return = np.mean(returns)
        if annualize:
            mean_return *= 252

        return mean_return / mdd

    @staticmethod
    def max_drawdown(returns: np.ndarray) -> float:
        """
        MDD = max_t(max_{τ≤t} V_τ - V_t) / max_{τ≤t} V_τ
        """
        if len(returns) < 2:
            return 0.0

        cumulative = np.cumprod(1 + returns)
        running_max = np.maximum.accumulate(cumulative)
        drawdown = (running_max - cumulative) / running_max

        return np.max(drawdown)

    @staticmethod
    def total_return(returns: np.ndarray) -> float:
        """
        TR = (V_T - V_1) / V_1 = Π(1 + r_t) - 1
        """
        return np.prod(1 + returns) - 1


# =============================================================================
# HTTP API HANDLER
# =============================================================================

class RLOptimizationAPIHandler(BaseHTTPRequestHandler):
    """HTTP request handler for RL Optimization API."""

    def send_json_response(self, data: Dict[str, Any], status: int = 200):
        """Send JSON response."""
        self.send_response(status)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()

        # Handle numpy types
        def convert(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (np.float32, np.float64)):
                return float(obj)
            elif isinstance(obj, (np.int32, np.int64)):
                return int(obj)
            return obj

        self.wfile.write(json.dumps(data, default=convert).encode())

    def send_error_response(self, message: str, status: int = 400):
        """Send error response."""
        self.send_json_response({'error': message, 'success': False}, status)

    def read_json_body(self) -> Optional[Dict[str, Any]]:
        """Read and parse JSON request body."""
        try:
            content_length = int(self.headers.get('Content-Length', 0))
            if content_length == 0:
                return {}
            body = self.rfile.read(content_length)
            return json.loads(body.decode())
        except (json.JSONDecodeError, ValueError) as e:
            logger.error(f"Failed to parse JSON body: {e}")
            return None

    def do_OPTIONS(self):
        """Handle CORS preflight."""
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()

    def do_GET(self):
        """Handle GET requests."""
        parsed = urlparse(self.path)
        path = parsed.path

        routes = {
            '/health': self.handle_health,
            '/api/rl/formulas': self.handle_list_formulas,
        }

        handler = routes.get(path)
        if handler:
            try:
                handler()
            except Exception as e:
                logger.exception(f"Error handling GET {path}")
                self.send_error_response(str(e), 500)
        else:
            self.send_error_response(f"Unknown endpoint: {path}", 404)

    def do_POST(self):
        """Handle POST requests."""
        parsed = urlparse(self.path)
        path = parsed.path

        routes = {
            '/api/rl/bellman/value': self.handle_bellman_value,
            '/api/rl/bellman/action': self.handle_bellman_action,
            '/api/rl/td_error': self.handle_td_error,
            '/api/rl/advantage': self.handle_advantage,
            '/api/rl/gae': self.handle_gae,
            '/api/rl/policy_gradient': self.handle_policy_gradient,
            '/api/rl/ppo_clip': self.handle_ppo_clip,
            '/api/rl/reward/sharpe': self.handle_sharpe_reward,
            '/api/rl/reward/composite': self.handle_composite_reward,
            '/api/rl/reward/differential_sharpe': self.handle_differential_sharpe,
            '/api/rl/reward/macrohft': self.handle_macrohft_reward,
            '/api/rl/reward/logic_q': self.handle_logic_q_reward,
            '/api/rl/hierarchical_q': self.handle_hierarchical_q,
            '/api/rl/memory_attention': self.handle_memory_attention,
            '/api/rl/dueling_q': self.handle_dueling_q,
            '/api/rl/logic_trend': self.handle_logic_trend,
            '/api/rl/metrics': self.handle_metrics,
        }

        handler = routes.get(path)
        if handler:
            try:
                body = self.read_json_body()
                if body is None:
                    self.send_error_response("Invalid JSON body")
                    return
                handler(body)
            except Exception as e:
                logger.exception(f"Error handling POST {path}")
                self.send_error_response(str(e), 500)
        else:
            self.send_error_response(f"Unknown endpoint: {path}", 404)

    def handle_health(self):
        """Health check endpoint."""
        self.send_json_response({
            'status': 'healthy',
            'service': 'rl-optimization-api',
            'timestamp': datetime.utcnow().isoformat(),
            'version': '1.0.0',
            'formulas_implemented': 15,
            'research_sources': [
                'MacroHFT (KDD 2024)',
                'EarnHFT (AAAI 2024)',
                'Logic-Q (AAAI 2024)',
                '华泰证券 DQN择时',
                'Risk-Aware RL (arXiv 2025)',
                'FinRL Framework'
            ]
        })

    def handle_list_formulas(self):
        """List all implemented formulas with citations."""
        formulas = [
            {
                'name': 'Bellman Value Function',
                'formula': 'V*(s) = E[r + γV*(s\')]',
                'endpoint': '/api/rl/bellman/value',
                'source': '知乎 Bellman, Deep Learning Wizard',
                'citation': 'https://zhuanlan.zhihu.com/p/139559442'
            },
            {
                'name': 'Bellman Action Value',
                'formula': 'Q*(s,a) = E[r + γ max_a\' Q*(s\',a\')]',
                'endpoint': '/api/rl/bellman/action',
                'source': 'Bellman 1957, Sutton & Barto',
                'citation': 'https://www.deeplearningwizard.com/'
            },
            {
                'name': 'TD Error',
                'formula': 'δ_t = r_t + γV(s_{t+1}) - V(s_t)',
                'endpoint': '/api/rl/td_error',
                'source': 'Actor-Critic, TensorFlow Tutorial',
                'citation': 'https://www.tensorflow.org/tutorials/reinforcement_learning/actor_critic'
            },
            {
                'name': 'Advantage Function',
                'formula': 'A(s,a) = Q(s,a) - V(s)',
                'endpoint': '/api/rl/advantage',
                'source': 'A2C, Schulman 2015',
                'citation': 'https://medium.com/intro-to-artificial-intelligence/'
            },
            {
                'name': 'Generalized Advantage Estimation',
                'formula': 'Â_t^GAE = Σ (γλ)^l δ_{t+l}',
                'endpoint': '/api/rl/gae',
                'source': 'Schulman 2016, GAE Paper',
                'citation': 'https://arxiv.org/abs/1506.02438'
            },
            {
                'name': 'Policy Gradient (REINFORCE)',
                'formula': '∇J = E[∇log π(a|s) · G_t]',
                'endpoint': '/api/rl/policy_gradient',
                'source': '知乎 Policy Gradient, Lil\'Log',
                'citation': 'https://lilianweng.github.io/posts/2018-04-08-policy-gradient/'
            },
            {
                'name': 'PPO Clipped Objective',
                'formula': 'L = min(rÂ, clip(r, 1-ε, 1+ε)Â)',
                'endpoint': '/api/rl/ppo_clip',
                'source': 'Schulman 2017, 华泰证券',
                'citation': 'https://arxiv.org/abs/1707.06347'
            },
            {
                'name': 'Sharpe Ratio Reward',
                'formula': 'S = A / √(B - A²)',
                'endpoint': '/api/rl/reward/sharpe',
                'source': 'Moody 1998, Teddy Koker',
                'citation': 'https://teddykoker.com/2019/06/trading-with-reinforcement-learning-in-python-part-ii-application/'
            },
            {
                'name': 'Composite Risk-Aware Reward',
                'formula': 'R = w₁R_ann - w₂σ_down + w₃D + w₄T',
                'endpoint': '/api/rl/reward/composite',
                'source': 'Risk-Aware RL 2025',
                'citation': 'https://arxiv.org/html/2506.04358v1'
            },
            {
                'name': 'Differential Sharpe Ratio',
                'formula': 'DSR = (R-μ)/σ - 0.5(μ/σ)((R-μ)²/σ² - 1)',
                'endpoint': '/api/rl/reward/differential_sharpe',
                'source': 'Moody & Saffell 2001',
                'citation': 'https://www.mdpi.com/2227-7390/12/24/4020'
            },
            {
                'name': 'MacroHFT Reward',
                'formula': 'r = (a(p\' - p) - δ|a - P|) × m',
                'endpoint': '/api/rl/reward/macrohft',
                'source': 'MacroHFT KDD 2024',
                'citation': 'https://arxiv.org/html/2406.14537v1'
            },
            {
                'name': 'Logic-Q Order Execution',
                'formula': 'R = a(p\'/p̃ - 1) - α(a)²',
                'endpoint': '/api/rl/reward/logic_q',
                'source': 'Logic-Q AAAI 2024',
                'citation': 'https://arxiv.org/abs/2310.05551'
            },
            {
                'name': 'Hierarchical Q-Function',
                'formula': 'Q^hyper = Σ wᵢQᵢ^sub',
                'endpoint': '/api/rl/hierarchical_q',
                'source': 'MacroHFT KDD 2024',
                'citation': 'https://arxiv.org/html/2406.14537v1'
            },
            {
                'name': 'Memory Attention Q',
                'formula': 'Q_m = Σ wᵢvᵢ, w = d(k,kᵢ)/Σd',
                'endpoint': '/api/rl/memory_attention',
                'source': 'MacroHFT KDD 2024',
                'citation': 'https://arxiv.org/html/2406.14537v1'
            },
            {
                'name': 'Dueling Q-Network',
                'formula': 'Q = V + (A - mean(A))',
                'endpoint': '/api/rl/dueling_q',
                'source': 'Wang 2016, MacroHFT',
                'citation': 'https://arxiv.org/abs/1511.06581'
            },
            {
                'name': 'Logic-Q Market Trend',
                'formula': 'trend ← (VOL < α) ∧ (DSR > β)',
                'endpoint': '/api/rl/logic_trend',
                'source': 'Logic-Q AAAI 2024',
                'citation': 'https://arxiv.org/abs/2310.05551'
            },
            {
                'name': 'Financial Metrics',
                'formula': 'Sharpe, Sortino, Calmar, MDD',
                'endpoint': '/api/rl/metrics',
                'source': 'MacroHFT, Risk-Aware RL',
                'citation': 'https://arxiv.org/html/2406.14537v1'
            }
        ]

        self.send_json_response({
            'success': True,
            'total_formulas': len(formulas),
            'formulas': formulas
        })

    # === Bellman Equations ===

    def handle_bellman_value(self, body: Dict):
        """Calculate Bellman value function."""
        rewards = np.array(body.get('rewards', []))
        gamma = body.get('gamma', 0.99)

        value = BellmanEquations.value_function(rewards, gamma)

        self.send_json_response({
            'success': True,
            'formula': 'V*(s) = E[r + γV*(s\')]',
            'value': value,
            'gamma': gamma,
            'citation': 'https://zhuanlan.zhihu.com/p/139559442'
        })

    def handle_bellman_action(self, body: Dict):
        """Calculate Bellman action value."""
        reward = body.get('reward', 0)
        gamma = body.get('gamma', 0.99)
        next_value = body.get('next_value', 0)

        q_value = BellmanEquations.action_value(reward, gamma, next_value)

        self.send_json_response({
            'success': True,
            'formula': 'Q*(s,a) = E[r + γ max Q*(s\',a\')]',
            'q_value': q_value,
            'citation': 'https://www.deeplearningwizard.com/'
        })

    # === TD Learning ===

    def handle_td_error(self, body: Dict):
        """Calculate TD error."""
        reward = body.get('reward', 0)
        gamma = body.get('gamma', 0.99)
        current_value = body.get('current_value', 0)
        next_value = body.get('next_value', 0)

        delta = TDLearning.td_error(reward, gamma, current_value, next_value)

        self.send_json_response({
            'success': True,
            'formula': 'δ = r + γV(s\') - V(s)',
            'td_error': delta,
            'citation': 'https://www.tensorflow.org/tutorials/reinforcement_learning/actor_critic'
        })

    def handle_advantage(self, body: Dict):
        """Calculate advantage function."""
        q_value = body.get('q_value', 0)
        v_value = body.get('v_value', 0)

        advantage = TDLearning.advantage(q_value, v_value)

        self.send_json_response({
            'success': True,
            'formula': 'A(s,a) = Q(s,a) - V(s)',
            'advantage': advantage,
            'citation': 'https://medium.com/intro-to-artificial-intelligence/'
        })

    def handle_gae(self, body: Dict):
        """Calculate Generalized Advantage Estimation."""
        rewards = np.array(body.get('rewards', []))
        values = np.array(body.get('values', []))
        gamma = body.get('gamma', 0.99)
        lam = body.get('lambda', 0.95)

        advantages = TDLearning.gae(rewards, values, gamma, lam)

        self.send_json_response({
            'success': True,
            'formula': 'Â^GAE = Σ (γλ)^l δ_{t+l}',
            'advantages': advantages.tolist(),
            'citation': 'https://arxiv.org/abs/1506.02438'
        })

    # === Policy Gradient ===

    def handle_policy_gradient(self, body: Dict):
        """Calculate REINFORCE returns."""
        rewards = np.array(body.get('rewards', []))
        gamma = body.get('gamma', 0.99)

        returns = PolicyGradient.reinforce_return(rewards, gamma)

        self.send_json_response({
            'success': True,
            'formula': 'G_t = Σ γ^{k-t} r_k',
            'returns': returns.tolist(),
            'citation': 'https://lilianweng.github.io/posts/2018-04-08-policy-gradient/'
        })

    def handle_ppo_clip(self, body: Dict):
        """Calculate PPO clipped objective."""
        ratio = body.get('ratio', 1.0)
        advantage = body.get('advantage', 0)
        epsilon = body.get('epsilon', 0.2)

        objective = PolicyGradient.ppo_clip_objective(ratio, advantage, epsilon)

        self.send_json_response({
            'success': True,
            'formula': 'L = min(rÂ, clip(r, 1-ε, 1+ε)Â)',
            'objective': objective,
            'citation': 'https://arxiv.org/abs/1707.06347'
        })

    # === Reward Functions ===

    def handle_sharpe_reward(self, body: Dict):
        """Calculate Sharpe ratio reward."""
        returns = np.array(body.get('returns', []))
        risk_free = body.get('risk_free', 0.0)

        sharpe = RewardFunctions.sharpe_ratio_reward(returns, risk_free)

        self.send_json_response({
            'success': True,
            'formula': 'S = A / √(B - A²)',
            'sharpe_reward': sharpe,
            'citation': 'https://teddykoker.com/2019/06/trading-with-reinforcement-learning-in-python-part-ii-application/'
        })

    def handle_composite_reward(self, body: Dict):
        """Calculate composite risk-aware reward."""
        returns = np.array(body.get('returns', []))
        weights = body.get('weights', {'w1': 1.0, 'w2': 0.5, 'w3': 0.3, 'w4': 0.2})

        result = RewardFunctions.composite_reward(
            returns,
            w1=weights.get('w1', 1.0),
            w2=weights.get('w2', 0.5),
            w3=weights.get('w3', 0.3),
            w4=weights.get('w4', 0.2)
        )

        self.send_json_response({
            'success': True,
            'formula': 'R = w₁R_ann - w₂σ_down + w₃D + w₄T',
            'composite_reward': result['composite'],
            'components': result['components'],
            'citation': 'https://arxiv.org/html/2506.04358v1'
        })

    def handle_differential_sharpe(self, body: Dict):
        """Calculate differential Sharpe ratio."""
        return_t = body.get('return_t', 0)
        mean_prev = body.get('mean_prev', 0)
        std_prev = body.get('std_prev', 1)

        dsr = RewardFunctions.differential_sharpe_ratio(return_t, mean_prev, std_prev)

        self.send_json_response({
            'success': True,
            'formula': 'DSR = (R-μ)/σ - 0.5(μ/σ)((R-μ)²/σ² - 1)',
            'differential_sharpe': dsr,
            'citation': 'https://www.mdpi.com/2227-7390/12/24/4020'
        })

    def handle_macrohft_reward(self, body: Dict):
        """Calculate MacroHFT reward."""
        action = body.get('action', 0)
        price_current = body.get('price_current', 1.0)
        price_next = body.get('price_next', 1.0)
        position = body.get('position', 0)
        transaction_cost = body.get('transaction_cost', 0.001)

        reward = RewardFunctions.macrohft_reward(
            action, price_current, price_next, position, transaction_cost
        )

        self.send_json_response({
            'success': True,
            'formula': 'r = (a(p\' - p) - δ|a - P|) × m',
            'reward': reward,
            'citation': 'https://arxiv.org/html/2406.14537v1'
        })

    def handle_logic_q_reward(self, body: Dict):
        """Calculate Logic-Q order execution reward."""
        action = body.get('action', 0)
        price_next = body.get('price_next', 1.0)
        price_avg = body.get('price_avg', 1.0)
        alpha = body.get('alpha', 0.01)

        reward = RewardFunctions.logic_q_order_execution_reward(
            action, price_next, price_avg, alpha
        )

        self.send_json_response({
            'success': True,
            'formula': 'R = a(p\'/p̃ - 1) - α(a)²',
            'reward': reward,
            'citation': 'https://arxiv.org/abs/2310.05551'
        })

    # === Hierarchical RL ===

    def handle_hierarchical_q(self, body: Dict):
        """Calculate hierarchical Q-function."""
        sub_q_values = body.get('sub_q_values', [])
        weights = body.get('weights', [])

        q_hyper = HierarchicalRL.hierarchical_q(sub_q_values, weights)

        self.send_json_response({
            'success': True,
            'formula': 'Q^hyper = Σ wᵢQᵢ^sub',
            'q_hyper': q_hyper,
            'citation': 'https://arxiv.org/html/2406.14537v1'
        })

    def handle_memory_attention(self, body: Dict):
        """Calculate memory attention Q-value."""
        query_key = np.array(body.get('query_key', []))
        memory_keys = np.array(body.get('memory_keys', []))
        memory_values = np.array(body.get('memory_values', []))
        actions = np.array(body.get('actions', []))
        current_action = body.get('current_action', 0)

        q_memory, weights = HierarchicalRL.memory_attention(
            query_key, memory_keys, memory_values, actions, current_action
        )

        self.send_json_response({
            'success': True,
            'formula': 'Q_m = Σ wᵢvᵢ',
            'q_memory': q_memory,
            'attention_weights': weights.tolist(),
            'citation': 'https://arxiv.org/html/2406.14537v1'
        })

    def handle_dueling_q(self, body: Dict):
        """Calculate dueling Q-network values."""
        value = body.get('value', 0)
        advantages = np.array(body.get('advantages', []))

        q_values = HierarchicalRL.dueling_q_network(value, advantages)

        self.send_json_response({
            'success': True,
            'formula': 'Q = V + (A - mean(A))',
            'q_values': q_values.tolist(),
            'citation': 'https://arxiv.org/abs/1511.06581'
        })

    # === Logic-Q ===

    def handle_logic_trend(self, body: Dict):
        """Detect market trend using Logic-Q."""
        prices = np.array(body.get('prices', []))
        vol_threshold = body.get('vol_threshold', 0.01)
        dsr_threshold = body.get('dsr_threshold', 0.005)

        result = LogicQ.detect_market_trend(prices, vol_threshold, dsr_threshold)

        self.send_json_response({
            'success': True,
            'formula': 'trend ← (VOL < α) ∧ (DSR > β)',
            **result,
            'citation': 'https://arxiv.org/abs/2310.05551'
        })

    # === Metrics ===

    def handle_metrics(self, body: Dict):
        """Calculate all financial metrics."""
        returns = np.array(body.get('returns', []))
        risk_free = body.get('risk_free', 0.0)

        metrics = {
            'sharpe_ratio': FinancialMetrics.sharpe_ratio(returns, risk_free),
            'sortino_ratio': FinancialMetrics.sortino_ratio(returns, risk_free),
            'calmar_ratio': FinancialMetrics.calmar_ratio(returns),
            'max_drawdown': FinancialMetrics.max_drawdown(returns),
            'total_return': FinancialMetrics.total_return(returns)
        }

        self.send_json_response({
            'success': True,
            'metrics': metrics,
            'formulas': {
                'sharpe': 'SR = (μ - r_f) / σ × √252',
                'sortino': 'SoR = (μ - r_f) / σ_down × √252',
                'calmar': 'CR = R_ann / MDD',
                'mdd': 'MDD = max(peak - trough) / peak'
            },
            'citation': 'https://arxiv.org/html/2406.14537v1'
        })

    def log_message(self, format: str, *args):
        """Custom logging."""
        logger.info(f"{self.address_string()} - {format % args}")


def run_server(host: str = '0.0.0.0', port: int = 8083):
    """Run the HTTP server."""
    server_address = (host, port)
    httpd = HTTPServer(server_address, RLOptimizationAPIHandler)

    logger.info(f"")
    logger.info(f"=" * 70)
    logger.info(f"  RL OPTIMIZATION API - Chinese & Global RL Math")
    logger.info(f"=" * 70)
    logger.info(f"")
    logger.info(f"Server: http://{host}:{port}")
    logger.info(f"Health: http://{host}:{port}/health")
    logger.info(f"Formulas: http://{host}:{port}/api/rl/formulas")
    logger.info(f"")
    logger.info(f"Research Sources:")
    logger.info(f"  - MacroHFT (KDD 2024): https://arxiv.org/html/2406.14537v1")
    logger.info(f"  - EarnHFT (AAAI 2024): https://arxiv.org/abs/2309.12891")
    logger.info(f"  - Logic-Q (AAAI 2024): https://arxiv.org/abs/2310.05551")
    logger.info(f"  - 华泰证券 DQN择时")
    logger.info(f"  - Risk-Aware RL (arXiv 2025)")
    logger.info(f"  - FinRL Framework")
    logger.info(f"")
    logger.info(f"Formulas Implemented: 17")
    logger.info(f"")

    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        logger.info("Server shutting down...")
        httpd.shutdown()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='RL Optimization HTTP API')
    parser.add_argument('--host', default='0.0.0.0', help='Host to bind to')
    parser.add_argument('--port', type=int, default=8083, help='Port to listen on')
    args = parser.parse_args()

    run_server(args.host, args.port)
