"""
Pure Mathematical Reinforcement Learning Algorithms for Trading
===============================================================
Algorithmic RL methods WITHOUT neural networks - pure mathematical formulations.

This module implements the mathematical foundations of RL as used by:
- Chinese quants (强化学习量化交易)
- Worldwide institutional traders
- Academic research (Almgren-Chriss, Hamilton-Jacobi-Bellman)

═══════════════════════════════════════════════════════════════════════════════
ACADEMIC CITATIONS (Full Bibliographic References)
═══════════════════════════════════════════════════════════════════════════════

Q-LEARNING:
    Watkins, C.J.C.H. (1989). "Learning from Delayed Rewards."
    PhD Thesis, King's College, Cambridge University.
    - Original Q-learning algorithm with convergence proof

    Watkins, C.J.C.H. & Dayan, P. (1992). "Q-learning."
    Machine Learning, 8(3-4), 279-292. DOI: 10.1007/BF00992698
    - Formal convergence analysis

SARSA:
    Rummery, G.A. & Niranjan, M. (1994). "On-Line Q-Learning Using
    Connectionist Systems." Technical Report CUED/F-INFENG/TR 166,
    Cambridge University Engineering Department.
    - Original SARSA algorithm (State-Action-Reward-State-Action)

    Sutton, R.S. (1996). "Generalization in Reinforcement Learning:
    Successful Examples Using Sparse Coarse Coding."
    Advances in Neural Information Processing Systems (NIPS), 8, 1038-1044.

TD(λ) - TEMPORAL DIFFERENCE WITH ELIGIBILITY TRACES:
    Sutton, R.S. (1988). "Learning to Predict by the Methods of
    Temporal Differences." Machine Learning, 3(1), 9-44.
    DOI: 10.1007/BF00115009
    - Foundation of TD learning

    Tesauro, G. (1994). "TD-Gammon, a Self-Teaching Backgammon Program,
    Achieves Master-Level Play." Neural Computation, 6(2), 215-219.
    - Famous TD(λ) application achieving superhuman performance

DYNA-Q (MODEL-BASED RL):
    Sutton, R.S. (1991). "Dyna, an Integrated Architecture for Learning,
    Planning, and Reacting." ACM SIGART Bulletin, 2(4), 160-163.
    - Original Dyna architecture combining model-free and model-based RL

    Sutton, R.S. (1990). "Integrated Architectures for Learning, Planning,
    and Reacting Based on Approximating Dynamic Programming."
    Proceedings of the 7th International Conference on Machine Learning.

THOMPSON SAMPLING:
    Thompson, W.R. (1933). "On the Likelihood that One Unknown Probability
    Exceeds Another in View of the Evidence of Two Samples."
    Biometrika, 25(3-4), 285-294. DOI: 10.2307/2332286
    - Original Thompson Sampling algorithm (90+ years old, still SOTA)

    Agrawal, S. & Goyal, N. (2012). "Analysis of Thompson Sampling for
    the Multi-armed Bandit Problem."
    Proceedings of the 25th Conference on Learning Theory (COLT), 39.1-39.26.
    - Modern regret analysis proving near-optimality

    Russo, D.J. et al. (2018). "A Tutorial on Thompson Sampling."
    Foundations and Trends in Machine Learning, 11(1), 1-96.
    DOI: 10.1561/2200000070

ALMGREN-CHRISS OPTIMAL EXECUTION:
    Almgren, R. & Chriss, N. (2001). "Optimal Execution of Portfolio
    Transactions." Journal of Risk, 3(2), 5-39.
    - Seminal paper on optimal trade execution with market impact

    Almgren, R. (2003). "Optimal Execution with Nonlinear Impact Functions
    and Trading-Enhanced Risk." Applied Mathematical Finance, 10(1), 1-18.
    DOI: 10.1080/135048602100056

    Gatheral, J. & Schied, A. (2011). "Optimal Trade Execution under
    Geometric Brownian Motion in the Almgren and Chriss Framework."
    International Journal of Theoretical and Applied Finance, 14(3), 353-368.

    Ning, B., Ling, F.H.T. & Jaimungal, S. (2021). "Double Deep Q-Learning
    for Optimal Execution." Applied Mathematical Finance, 28(4), 361-380.
    DOI: 10.1080/1350486X.2022.2077783
    - RL extension achieving 10.3% improvement over base Almgren-Chriss

KELLY CRITERION:
    Kelly, J.L. (1956). "A New Interpretation of Information Rate."
    Bell System Technical Journal, 35(4), 917-926.
    DOI: 10.1002/j.1538-7305.1956.tb03809.x
    - Original optimal betting fraction

    Thorp, E.O. (2006). "The Kelly Criterion in Blackjack, Sports Betting,
    and the Stock Market." Handbook of Asset and Liability Management, 1, 385-428.
    - Kelly applications in finance

COMPREHENSIVE RL IN FINANCE:
    Hambly, B., Xu, R. & Yang, H. (2023). "Recent Advances in Reinforcement
    Learning in Finance." Mathematical Finance, 33(3), 437-503.
    DOI: 10.1111/mafi.12382
    - Comprehensive review of RL applications in quantitative finance

    Sutton, R.S. & Barto, A.G. (2018). "Reinforcement Learning:
    An Introduction" (2nd ed.). MIT Press. ISBN: 978-0262039246
    - Definitive RL textbook

CHINESE QUANT SOURCES (中文量化资源):
    Liu, X.Y. et al. (2022). "FinRL: Deep Reinforcement Learning Framework
    to Automate Trading in Quantitative Finance."
    ACM International Conference on AI in Finance (ICAIF).
    GitHub: https://github.com/AI4Finance-Foundation/FinRL

    BigQuant (2024). "强化学习在量化交易中的应用白皮书"
    (Whitepaper: Applications of RL in Quantitative Trading)

    ai_quant_trade - Chinese Quant RL Implementations
    GitHub: https://github.com/wizardforcel/ai_quant_trade

═══════════════════════════════════════════════════════════════════════════════
FEATURE SUMMARY (35 Total)
═══════════════════════════════════════════════════════════════════════════════

Q-Learning Signals (6 features):
    - RL_QLEARN_SIGNAL: Trading signal from Q-value softmax
    - RL_QLEARN_TD_ERROR: Temporal difference error
    - RL_QLEARN_Q_BUY: Q-value for buy action
    - RL_QLEARN_Q_SELL: Q-value for sell action
    - RL_QLEARN_Q_SPREAD: Spread between buy/sell Q-values
    - RL_QLEARN_CONFIDENCE: Max Q-value (action confidence)

SARSA Signals (5 features):
    - RL_SARSA_SIGNAL: On-policy trading signal
    - RL_SARSA_TD_ERROR: SARSA TD error (more conservative)
    - RL_SARSA_Q_DIV: Q-value divergence from Q-learning
    - RL_SARSA_SAFE: Safety signal (lower when risky)
    - RL_SARSA_CAUTION: Caution level for volatile states

TD(λ) Features (5 features):
    - RL_TD_VALUE: State value estimate
    - RL_TD_ERROR: TD error with traces
    - RL_TD_ELIGIBILITY: Eligibility trace strength
    - RL_TD_CREDIT: Temporal credit assignment
    - RL_TD_HORIZON: Effective horizon (λ-dependent)

Dyna-Q Features (5 features):
    - RL_DYNA_SIGNAL: Model-based trading signal
    - RL_DYNA_PLANNING: Planning value improvement
    - RL_DYNA_MODEL_SIZE: Transition model coverage
    - RL_DYNA_CONFIDENCE: Model confidence
    - RL_DYNA_NOVELTY: State novelty (inverse of visits)

Thompson Sampling (5 features):
    - RL_TS_SIGNAL: Posterior mean signal
    - RL_TS_UNCERTAINTY: Posterior uncertainty
    - RL_TS_SAMPLED: Sampled action value
    - RL_TS_EXPLORE: Exploration bonus
    - RL_TS_EXPLOIT: Exploitation score

Optimal Execution - Almgren-Chriss (5 features):
    - RL_EXEC_URGENCY: Execution urgency (κ parameter)
    - RL_EXEC_AGGRESSION: Aggressiveness level
    - RL_EXEC_VOL_ADJ: Volatility-adjusted execution
    - RL_EXEC_SPREAD_ADJ: Spread-adjusted execution
    - RL_EXEC_REGIME: Execution regime signal

Kelly-RL Integration (4 features):
    - RL_KELLY_FRAC: Kelly fraction from RL estimates
    - RL_KELLY_EDGE: Estimated edge from Q-values
    - RL_KELLY_CONFIDENCE: RL-weighted Kelly confidence
    - RL_KELLY_SIGNAL: Combined Kelly-RL signal

═══════════════════════════════════════════════════════════════════════════════
MATHEMATICAL NOTATION
═══════════════════════════════════════════════════════════════════════════════

- Q(s,a): Action-value function (expected return from state s taking action a)
- V(s): State-value function (expected return from state s)
- π(a|s): Policy (probability of action a in state s)
- α: Learning rate (step size, typically 0.01-0.3)
- γ: Discount factor (0 < γ < 1, typically 0.95-0.99)
- λ: Eligibility trace decay (0 = TD(0), 1 = Monte Carlo)
- δ: TD error = r + γV(s') - V(s)
- e(s): Eligibility trace for state s
- κ: Almgren-Chriss urgency parameter = √(λσ²/η)
- f*: Kelly optimal fraction = (p·b - q) / b where p=win prob, q=1-p, b=odds
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from collections import defaultdict
import warnings
from scipy import stats

warnings.filterwarnings('ignore')


@dataclass
class RLState:
    """Discretized market state for tabular RL."""
    trend: int      # -1: down, 0: neutral, 1: up
    volatility: int # 0: low, 1: medium, 2: high
    momentum: int   # -1: negative, 0: neutral, 1: positive

    def to_tuple(self) -> Tuple[int, int, int]:
        return (self.trend, self.volatility, self.momentum)


class QLearningTrader:
    """
    Tabular Q-Learning for trading decisions.

    Q-Learning Update Rule:
    Q(s,a) ← Q(s,a) + α[r + γ·max_a' Q(s',a') - Q(s,a)]

    This is off-policy: learns optimal Q* regardless of behavior policy.
    Aggressive in seeking maximum reward.

    References:
    - Watkins (1989): "Learning from Delayed Rewards"
    - FinRL: DQN variations for trading
    """

    def __init__(
        self,
        alpha: float = 0.1,      # Learning rate
        gamma: float = 0.95,     # Discount factor
        epsilon: float = 0.1,    # Exploration rate
        n_actions: int = 3       # Buy, Hold, Sell
    ):
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.n_actions = n_actions
        self.q_table = defaultdict(lambda: np.zeros(n_actions))
        self.action_map = {0: -1, 1: 0, 2: 1}  # Sell, Hold, Buy

    def discretize_state(self, returns: np.ndarray, volatility: float) -> RLState:
        """Convert continuous market data to discrete state."""
        # Trend: based on recent returns
        recent_return = returns[-5:].mean() if len(returns) >= 5 else 0
        if recent_return > 0.001:
            trend = 1
        elif recent_return < -0.001:
            trend = -1
        else:
            trend = 0

        # Volatility regime
        if volatility < 0.01:
            vol_state = 0
        elif volatility < 0.02:
            vol_state = 1
        else:
            vol_state = 2

        # Momentum
        if len(returns) >= 20:
            mom = returns[-5:].sum() - returns[-20:-5].sum()
            momentum = 1 if mom > 0.002 else (-1 if mom < -0.002 else 0)
        else:
            momentum = 0

        return RLState(trend=trend, volatility=vol_state, momentum=momentum)

    def get_q_value(self, state: RLState, action: int) -> float:
        """Get Q-value for state-action pair."""
        return self.q_table[state.to_tuple()][action]

    def get_best_action(self, state: RLState) -> int:
        """Select action with highest Q-value (exploitation)."""
        return np.argmax(self.q_table[state.to_tuple()])

    def get_action_signal(self, state: RLState) -> float:
        """Convert best action to trading signal [-1, 1]."""
        q_values = self.q_table[state.to_tuple()]
        # Softmax to get action probabilities
        exp_q = np.exp(q_values - np.max(q_values))
        probs = exp_q / exp_q.sum()
        # Weighted signal: -1 * P(sell) + 0 * P(hold) + 1 * P(buy)
        return probs[2] - probs[0]

    def update(self, state: RLState, action: int, reward: float,
               next_state: RLState) -> float:
        """
        Q-Learning update.

        Returns TD error for analysis.
        """
        current_q = self.q_table[state.to_tuple()][action]
        max_next_q = np.max(self.q_table[next_state.to_tuple()])

        # TD error
        td_error = reward + self.gamma * max_next_q - current_q

        # Update Q-value
        self.q_table[state.to_tuple()][action] += self.alpha * td_error

        return td_error


class SARSATrader:
    """
    SARSA (State-Action-Reward-State-Action) for safer trading.

    SARSA Update Rule:
    Q(s,a) ← Q(s,a) + α[r + γ·Q(s',a') - Q(s,a)]

    This is on-policy: learns Q for current policy, not optimal Q*.
    More conservative than Q-learning in volatile markets.

    References:
    - Rummery & Niranjan (1994): "On-Line Q-Learning"
    - MQL5 Article: SARSA for MetaTrader
    """

    def __init__(
        self,
        alpha: float = 0.1,
        gamma: float = 0.95,
        epsilon: float = 0.1,
        n_actions: int = 3
    ):
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.n_actions = n_actions
        self.q_table = defaultdict(lambda: np.zeros(n_actions))

    def epsilon_greedy_action(self, state: RLState) -> int:
        """Select action using epsilon-greedy policy."""
        if np.random.random() < self.epsilon:
            return np.random.randint(self.n_actions)
        return np.argmax(self.q_table[state.to_tuple()])

    def get_action_signal(self, state: RLState) -> float:
        """Convert policy to trading signal."""
        q_values = self.q_table[state.to_tuple()]
        exp_q = np.exp(q_values - np.max(q_values))
        probs = exp_q / exp_q.sum()
        return probs[2] - probs[0]

    def update(self, state: RLState, action: int, reward: float,
               next_state: RLState, next_action: int) -> float:
        """
        SARSA update (on-policy).

        Key difference from Q-learning: uses actual next action,
        not maximum over all possible actions.
        """
        current_q = self.q_table[state.to_tuple()][action]
        next_q = self.q_table[next_state.to_tuple()][next_action]

        td_error = reward + self.gamma * next_q - current_q
        self.q_table[state.to_tuple()][action] += self.alpha * td_error

        return td_error


class TDLambdaTrader:
    """
    TD(λ) with Eligibility Traces for trading.

    TD(λ) bridges TD(0) and Monte Carlo:
    - λ = 0: One-step TD (low variance, high bias)
    - λ = 1: Monte Carlo (low bias, high variance)

    Update with eligibility traces:
    e(s) ← γλe(s) + 1  (for visited state)
    V(s) ← V(s) + αδe(s)  (for all states)

    References:
    - Sutton (1988): "Learning to Predict by the Methods of TD"
    - Tesauro (1994): TD-Gammon
    """

    def __init__(
        self,
        alpha: float = 0.1,
        gamma: float = 0.95,
        lambda_: float = 0.8,   # Eligibility trace decay
    ):
        self.alpha = alpha
        self.gamma = gamma
        self.lambda_ = lambda_
        self.v_table = defaultdict(float)
        self.eligibility = defaultdict(float)

    def update(self, state: RLState, reward: float,
               next_state: RLState) -> Tuple[float, float]:
        """
        TD(λ) update with eligibility traces.

        Returns (TD error, eligibility of current state).
        """
        # TD error
        current_v = self.v_table[state.to_tuple()]
        next_v = self.v_table[next_state.to_tuple()]
        td_error = reward + self.gamma * next_v - current_v

        # Update eligibility trace for current state
        self.eligibility[state.to_tuple()] += 1

        # Update all states proportional to eligibility
        for s, e in list(self.eligibility.items()):
            self.v_table[s] += self.alpha * td_error * e
            # Decay eligibility
            self.eligibility[s] = self.gamma * self.lambda_ * e
            if self.eligibility[s] < 1e-6:
                del self.eligibility[s]

        return td_error, self.eligibility.get(state.to_tuple(), 0)

    def get_value_signal(self, state: RLState) -> float:
        """Get normalized value estimate."""
        return self.v_table[state.to_tuple()]


class DynaQTrader:
    """
    Dyna-Q: Model-Based RL for sample-efficient trading.

    Dyna-Q combines:
    1. Direct RL (learning from real experience)
    2. Model learning (building transition model)
    3. Planning (learning from simulated experience)

    For each real experience:
    - Update Q from real experience
    - Update model
    - Perform k planning steps with simulated experience

    References:
    - Sutton (1991): "Dyna, an Integrated Architecture for Learning"
    - ML4Trading course: Dyna for trading
    """

    def __init__(
        self,
        alpha: float = 0.1,
        gamma: float = 0.95,
        n_planning_steps: int = 10,
        n_actions: int = 3
    ):
        self.alpha = alpha
        self.gamma = gamma
        self.n_planning = n_planning_steps
        self.n_actions = n_actions
        self.q_table = defaultdict(lambda: np.zeros(n_actions))
        # Model: stores (next_state, reward) for each (state, action)
        self.model = {}
        self.visited_sa = []  # Track visited state-action pairs

    def update_real(self, state: RLState, action: int, reward: float,
                    next_state: RLState) -> float:
        """Update from real experience."""
        # Q-learning update
        current_q = self.q_table[state.to_tuple()][action]
        max_next_q = np.max(self.q_table[next_state.to_tuple()])
        td_error = reward + self.gamma * max_next_q - current_q
        self.q_table[state.to_tuple()][action] += self.alpha * td_error

        # Update model
        sa = (state.to_tuple(), action)
        self.model[sa] = (next_state.to_tuple(), reward)
        if sa not in self.visited_sa:
            self.visited_sa.append(sa)

        return td_error

    def planning_step(self) -> int:
        """
        Perform planning steps using model.

        Returns number of updates made.
        """
        updates = 0
        for _ in range(self.n_planning):
            if not self.visited_sa:
                break

            # Sample random previously visited state-action
            sa = self.visited_sa[np.random.randint(len(self.visited_sa))]
            state, action = sa

            # Get simulated experience from model
            if sa in self.model:
                next_state, reward = self.model[sa]

                # Q-learning update on simulated experience
                current_q = self.q_table[state][action]
                max_next_q = np.max(self.q_table[next_state])
                td_error = reward + self.gamma * max_next_q - current_q
                self.q_table[state][action] += self.alpha * td_error
                updates += 1

        return updates

    def get_action_signal(self, state: RLState) -> float:
        """Get trading signal from Q-values."""
        q_values = self.q_table[state.to_tuple()]
        exp_q = np.exp(q_values - np.max(q_values))
        probs = exp_q / exp_q.sum()
        return probs[2] - probs[0]


class ThompsonSamplingBandit:
    """
    Thompson Sampling for portfolio allocation.

    Multi-armed bandit approach where each "arm" is a trading action.
    Uses Bayesian posterior sampling to balance exploration/exploitation.

    For each action a, maintain Beta(α_a, β_a) prior:
    - α_a: pseudo-count of successes (profitable trades)
    - β_a: pseudo-count of failures (losing trades)

    Thompson Sampling Algorithm:
    1. Sample θ_a ~ Beta(α_a, β_a) for each action
    2. Select action with highest θ_a
    3. Update α_a or β_a based on outcome

    References:
    - Thompson (1933): "On the Likelihood that One Unknown Probability..."
    - Agrawal & Goyal (2012): "Analysis of Thompson Sampling"
    - Springer (2025): "Bandit Networks for Portfolio Optimization"
    """

    def __init__(self, n_actions: int = 3, prior_strength: float = 1.0):
        """
        Initialize with uniform priors.

        Args:
            n_actions: Number of trading actions (sell, hold, buy)
            prior_strength: Strength of prior (higher = more conservative)
        """
        self.n_actions = n_actions
        # Beta distribution parameters (alpha, beta) for each action
        # Start with uniform prior Beta(1, 1)
        self.alpha = np.ones(n_actions) * prior_strength
        self.beta = np.ones(n_actions) * prior_strength
        self.action_counts = np.zeros(n_actions)

    def sample_action(self) -> Tuple[int, np.ndarray]:
        """
        Thompson Sampling: sample from posterior and select best.

        Returns (selected action, sampled probabilities).
        """
        # Sample from Beta posterior for each action
        samples = np.array([
            np.random.beta(self.alpha[i], self.beta[i])
            for i in range(self.n_actions)
        ])
        return np.argmax(samples), samples

    def update(self, action: int, reward: float):
        """
        Update posterior based on observed reward.

        Args:
            action: Action taken
            reward: Binary reward (1 = success, 0 = failure)
                    or continuous reward (normalized to [0, 1])
        """
        # Clip reward to [0, 1]
        reward = np.clip(reward, 0, 1)

        # Update Beta parameters
        self.alpha[action] += reward
        self.beta[action] += (1 - reward)
        self.action_counts[action] += 1

    def get_action_signal(self) -> float:
        """
        Get trading signal based on expected values.

        Returns weighted signal in [-1, 1].
        """
        # Expected value of Beta distribution = α / (α + β)
        expected = self.alpha / (self.alpha + self.beta)
        # Signal: P(buy) - P(sell), normalized
        return expected[2] - expected[0]

    def get_uncertainty(self) -> np.ndarray:
        """Get uncertainty (variance) for each action."""
        # Variance of Beta = αβ / ((α+β)²(α+β+1))
        total = self.alpha + self.beta
        return (self.alpha * self.beta) / (total ** 2 * (total + 1))


class AlmgrenChrisExecutor:
    """
    Almgren-Chriss Optimal Execution Framework.

    Minimizes: E[Cost] + λ·Var[Cost]

    For linear temporary impact: g(v) = η·v
    For linear permanent impact: h(v) = γ·v

    Optimal trajectory (TWAP adjusted):
    n_k = (sinh(κ(T-t_k)) / sinh(κT)) · X_0

    where κ = sqrt(λσ²/η)

    References:
    - Almgren & Chriss (2001): "Optimal Execution"
    - Gatheral & Schied (2011): HJB solution
    - arXiv:1403.2229: RL extension achieving 10.3% improvement
    """

    def __init__(
        self,
        risk_aversion: float = 1e-6,  # λ
        temp_impact: float = 1e-4,     # η (temporary impact)
        perm_impact: float = 1e-5,     # γ (permanent impact)
    ):
        self.lambda_ = risk_aversion
        self.eta = temp_impact
        self.gamma = perm_impact

    def optimal_trajectory(
        self,
        total_shares: float,
        n_periods: int,
        volatility: float
    ) -> np.ndarray:
        """
        Calculate optimal execution trajectory.

        Args:
            total_shares: Total shares to execute
            n_periods: Number of trading periods
            volatility: Price volatility (σ)

        Returns:
            Array of shares to trade in each period
        """
        # Calculate κ
        kappa = np.sqrt(self.lambda_ * volatility ** 2 / self.eta)

        # Generate trajectory
        T = n_periods
        trajectory = np.zeros(n_periods)
        remaining = total_shares

        for k in range(n_periods):
            t_k = k
            if kappa * T < 1e-6:
                # Small κ: uniform (TWAP)
                trajectory[k] = total_shares / n_periods
            else:
                # Almgren-Chriss formula
                n_k = remaining * np.sinh(kappa * (T - t_k)) / np.sinh(kappa * (T - t_k + 1))
                trajectory[k] = n_k
                remaining -= n_k

        return trajectory

    def expected_cost(
        self,
        trajectory: np.ndarray,
        initial_price: float,
        volatility: float
    ) -> Tuple[float, float]:
        """
        Calculate expected execution cost and variance.

        Returns (expected_cost, variance).
        """
        n = len(trajectory)

        # Temporary impact cost: Σ η·v_k²
        temp_cost = self.eta * np.sum(trajectory ** 2)

        # Permanent impact cost: Σ γ·v_k·X_k
        remaining = np.cumsum(trajectory[::-1])[::-1]
        perm_cost = self.gamma * np.sum(trajectory * remaining)

        # Variance
        variance = volatility ** 2 * np.sum(remaining ** 2)

        expected = temp_cost + perm_cost
        return expected, variance

    def execution_urgency(self, volatility: float, spread: float) -> float:
        """
        Calculate execution urgency signal.

        High urgency = execute faster (market moving against us)
        Low urgency = execute slower (favorable conditions)
        """
        kappa = np.sqrt(self.lambda_ * volatility ** 2 / self.eta)

        # Urgency based on vol-adjusted impact
        base_urgency = kappa * volatility

        # Adjust for spread (wider spread = less urgency)
        spread_factor = 1 / (1 + spread / volatility)

        return base_urgency * spread_factor


class RLFeatureGenerator:
    """
    Generate RL-based trading features.

    Implements pure mathematical RL algorithms:
    - Q-Learning signals
    - SARSA signals
    - TD(λ) eligibility-weighted features
    - Dyna-Q model confidence
    - Thompson Sampling posterior signals
    - Almgren-Chriss execution features
    - Kelly-RL integrated sizing

    Total: 35 features
    """

    def __init__(
        self,
        alpha: float = 0.1,
        gamma: float = 0.95,
        lambda_: float = 0.8,
        risk_aversion: float = 1e-6,
    ):
        self.q_learner = QLearningTrader(alpha=alpha, gamma=gamma)
        self.sarsa = SARSATrader(alpha=alpha, gamma=gamma)
        self.td_lambda = TDLambdaTrader(alpha=alpha, gamma=gamma, lambda_=lambda_)
        self.dyna = DynaQTrader(alpha=alpha, gamma=gamma, n_planning_steps=10)
        self.thompson = ThompsonSamplingBandit(n_actions=3)
        self.executor = AlmgrenChrisExecutor(risk_aversion=risk_aversion)

    def _discretize_states(self, df: pd.DataFrame) -> List[RLState]:
        """Convert price series to discrete RL states."""
        returns = df['close'].pct_change().fillna(0)
        volatility = returns.rolling(20, min_periods=2).std().fillna(0.01)

        states = []
        for i in range(len(df)):
            if i < 20:
                ret_arr = returns.iloc[:i+1].values if i > 0 else np.array([0])
            else:
                ret_arr = returns.iloc[i-20:i+1].values

            state = self.q_learner.discretize_state(ret_arr, volatility.iloc[i])
            states.append(state)

        return states

    def _calculate_rewards(self, df: pd.DataFrame) -> np.ndarray:
        """Calculate trading rewards (forward returns)."""
        returns = df['close'].pct_change().shift(-1).fillna(0)
        return returns.values

    def _q_learning_features(self, df: pd.DataFrame, states: List[RLState],
                             rewards: np.ndarray) -> pd.DataFrame:
        """Generate Q-Learning based features."""
        features = pd.DataFrame(index=df.index)

        signals = []
        td_errors = []
        q_values_buy = []
        q_values_sell = []

        for i in range(len(states) - 1):
            state = states[i]
            next_state = states[i + 1]
            reward = rewards[i]

            # Get signal before update
            signal = self.q_learner.get_action_signal(state)
            signals.append(signal)

            # Get Q-values
            q_values_buy.append(self.q_learner.get_q_value(state, 2))
            q_values_sell.append(self.q_learner.get_q_value(state, 0))

            # Determine action (based on return sign)
            if reward > 0:
                action = 2  # Buy was correct
            elif reward < 0:
                action = 0  # Sell was correct
            else:
                action = 1  # Hold

            # Update Q-table
            td_error = self.q_learner.update(state, action, reward * 100, next_state)
            td_errors.append(td_error)

        # Pad last values
        signals.append(signals[-1] if signals else 0)
        td_errors.append(0)
        q_values_buy.append(q_values_buy[-1] if q_values_buy else 0)
        q_values_sell.append(q_values_sell[-1] if q_values_sell else 0)

        features['RL_QLEARN_SIGNAL'] = signals
        features['RL_QLEARN_TD_ERROR'] = td_errors
        features['RL_QLEARN_Q_BUY'] = q_values_buy
        features['RL_QLEARN_Q_SELL'] = q_values_sell
        features['RL_QLEARN_Q_SPREAD'] = np.array(q_values_buy) - np.array(q_values_sell)
        features['RL_QLEARN_CONFIDENCE'] = np.abs(np.array(q_values_buy) - np.array(q_values_sell))

        return features

    def _sarsa_features(self, df: pd.DataFrame, states: List[RLState],
                        rewards: np.ndarray) -> pd.DataFrame:
        """Generate SARSA-based features (safer than Q-learning)."""
        features = pd.DataFrame(index=df.index)

        signals = []
        td_errors = []

        prev_action = 1  # Start with hold

        for i in range(len(states) - 1):
            state = states[i]
            next_state = states[i + 1]
            reward = rewards[i]

            # Get signal
            signal = self.sarsa.get_action_signal(state)
            signals.append(signal)

            # Get action using policy
            action = self.sarsa.epsilon_greedy_action(state)
            next_action = self.sarsa.epsilon_greedy_action(next_state)

            # Update (SARSA uses actual next action, not max)
            td_error = self.sarsa.update(state, action, reward * 100,
                                         next_state, next_action)
            td_errors.append(td_error)

        signals.append(signals[-1] if signals else 0)
        td_errors.append(0)

        features['RL_SARSA_SIGNAL'] = signals
        features['RL_SARSA_TD_ERROR'] = td_errors
        # SARSA vs Q-learning divergence (indicates market danger)
        features['RL_SARSA_Q_DIV'] = (
            np.array(features['RL_SARSA_SIGNAL']) -
            np.array(self._q_learning_features(df, states, rewards)['RL_QLEARN_SIGNAL'])
        )
        features['RL_SARSA_SAFE'] = np.where(np.abs(features['RL_SARSA_Q_DIV']) > 0.2, 1, 0)
        features['RL_SARSA_CAUTION'] = features['RL_SARSA_Q_DIV'].rolling(10, min_periods=1).std()

        return features

    def _td_lambda_features(self, df: pd.DataFrame, states: List[RLState],
                            rewards: np.ndarray) -> pd.DataFrame:
        """Generate TD(λ) eligibility trace features."""
        features = pd.DataFrame(index=df.index)

        values = []
        td_errors = []
        eligibilities = []

        for i in range(len(states) - 1):
            state = states[i]
            next_state = states[i + 1]
            reward = rewards[i]

            # Get value estimate
            value = self.td_lambda.get_value_signal(state)
            values.append(value)

            # Update with eligibility traces
            td_error, eligibility = self.td_lambda.update(state, reward * 100, next_state)
            td_errors.append(td_error)
            eligibilities.append(eligibility)

        values.append(values[-1] if values else 0)
        td_errors.append(0)
        eligibilities.append(0)

        features['RL_TD_VALUE'] = values
        features['RL_TD_ERROR'] = td_errors
        features['RL_TD_ELIGIBILITY'] = eligibilities
        features['RL_TD_CREDIT'] = np.array(td_errors) * np.array(eligibilities)
        # Long-term vs short-term value (λ effect)
        features['RL_TD_HORIZON'] = pd.Series(values).diff(10).fillna(0).values

        return features

    def _dyna_features(self, df: pd.DataFrame, states: List[RLState],
                       rewards: np.ndarray) -> pd.DataFrame:
        """Generate Dyna-Q model-based features."""
        features = pd.DataFrame(index=df.index)

        signals = []
        planning_counts = []
        model_sizes = []

        for i in range(len(states) - 1):
            state = states[i]
            next_state = states[i + 1]
            reward = rewards[i]

            # Get signal
            signal = self.dyna.get_action_signal(state)
            signals.append(signal)

            # Determine action
            action = 2 if reward > 0 else (0 if reward < 0 else 1)

            # Real experience update
            self.dyna.update_real(state, action, reward * 100, next_state)

            # Planning (simulated experience)
            n_updates = self.dyna.planning_step()
            planning_counts.append(n_updates)
            model_sizes.append(len(self.dyna.model))

        signals.append(signals[-1] if signals else 0)
        planning_counts.append(0)
        model_sizes.append(model_sizes[-1] if model_sizes else 0)

        features['RL_DYNA_SIGNAL'] = signals
        features['RL_DYNA_PLANNING'] = planning_counts
        features['RL_DYNA_MODEL_SIZE'] = model_sizes
        # Model confidence: more experience = higher confidence
        features['RL_DYNA_CONFIDENCE'] = np.log1p(model_sizes)
        features['RL_DYNA_NOVELTY'] = 1 / (1 + np.array(model_sizes))

        return features

    def _thompson_features(self, df: pd.DataFrame, rewards: np.ndarray) -> pd.DataFrame:
        """Generate Thompson Sampling bandit features."""
        features = pd.DataFrame(index=df.index)

        signals = []
        uncertainties = []
        sampled_probs = []

        for i in range(len(rewards)):
            reward = rewards[i]

            # Get current signal
            signal = self.thompson.get_action_signal()
            signals.append(signal)

            # Get uncertainty
            unc = self.thompson.get_uncertainty()
            uncertainties.append(unc.mean())

            # Sample action
            _, probs = self.thompson.sample_action()
            sampled_probs.append(probs[2] - probs[0])  # Buy - Sell probability

            # Update based on return
            if reward > 0:
                self.thompson.update(2, 1)  # Buy success
            elif reward < 0:
                self.thompson.update(0, 1)  # Sell success
            else:
                self.thompson.update(1, 0.5)  # Hold neutral

        features['RL_TS_SIGNAL'] = signals
        features['RL_TS_UNCERTAINTY'] = uncertainties
        features['RL_TS_SAMPLED'] = sampled_probs
        features['RL_TS_EXPLORE'] = np.array(uncertainties) > np.median(uncertainties)
        features['RL_TS_EXPLOIT'] = 1 - features['RL_TS_EXPLORE']

        return features

    def _execution_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate Almgren-Chriss execution features."""
        features = pd.DataFrame(index=df.index)

        returns = df['close'].pct_change().fillna(0)
        volatility = returns.rolling(20, min_periods=2).std().fillna(0.01)
        spread = (df.get('high', df['close']) - df.get('low', df['close'])) / df['close']

        urgency = []
        trajectory_aggression = []

        for i in range(len(df)):
            vol = volatility.iloc[i]
            sprd = spread.iloc[i] if i < len(spread) else 0.001

            # Execution urgency
            urg = self.executor.execution_urgency(vol, sprd)
            urgency.append(urg)

            # Optimal trajectory aggression (first period as %)
            if vol > 0:
                traj = self.executor.optimal_trajectory(1.0, 10, vol)
                trajectory_aggression.append(traj[0])
            else:
                trajectory_aggression.append(0.1)

        features['RL_EXEC_URGENCY'] = urgency
        features['RL_EXEC_AGGRESSION'] = trajectory_aggression
        features['RL_EXEC_VOL_ADJ'] = np.array(urgency) * volatility.values
        features['RL_EXEC_SPREAD_ADJ'] = np.array(urgency) / (spread.values + 1e-6)
        # Execution regime
        features['RL_EXEC_REGIME'] = np.where(
            np.array(urgency) > np.median(urgency), 1, 0
        )

        return features

    def _kelly_rl_features(self, df: pd.DataFrame, rewards: np.ndarray) -> pd.DataFrame:
        """Generate Kelly Criterion integrated with RL signals."""
        features = pd.DataFrame(index=df.index)

        returns = df['close'].pct_change().fillna(0)

        # Rolling win probability
        win_prob = (returns > 0).rolling(60, min_periods=10).mean().fillna(0.5)

        # Rolling win/loss ratio
        wins = returns.where(returns > 0, np.nan)
        losses = returns.where(returns < 0, np.nan).abs()
        avg_win = wins.rolling(60, min_periods=10).mean().fillna(0.001)
        avg_loss = losses.rolling(60, min_periods=10).mean().fillna(0.001)
        wl_ratio = avg_win / (avg_loss + 1e-8)

        # Kelly fraction
        q = 1 - win_prob
        kelly = (win_prob * wl_ratio - q) / (wl_ratio + 1e-8)
        kelly = kelly.clip(-1, 1)

        features['RL_KELLY_FRAC'] = kelly
        features['RL_KELLY_EDGE'] = win_prob * avg_win - q * avg_loss
        features['RL_KELLY_CONFIDENCE'] = win_prob.rolling(20, min_periods=5).std()
        # Combine Kelly with RL signal
        features['RL_KELLY_SIGNAL'] = kelly * self.thompson.get_action_signal()

        return features

    def generate_all(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate all RL-based features.

        Args:
            df: DataFrame with OHLCV data

        Returns:
            DataFrame with 35 RL features
        """
        if 'close' not in df.columns:
            raise ValueError("Missing required column: 'close'")

        df = df.copy()
        if 'open' not in df.columns:
            df['open'] = df['close'].shift(1).fillna(df['close'])
        if 'high' not in df.columns:
            df['high'] = df['close']
        if 'low' not in df.columns:
            df['low'] = df['close']

        # Discretize states
        states = self._discretize_states(df)
        rewards = self._calculate_rewards(df)

        # Generate all feature groups
        q_features = self._q_learning_features(df, states, rewards)
        sarsa_features = self._sarsa_features(df, states, rewards)
        td_features = self._td_lambda_features(df, states, rewards)
        dyna_features = self._dyna_features(df, states, rewards)
        thompson_features = self._thompson_features(df, rewards)
        exec_features = self._execution_features(df)
        kelly_features = self._kelly_rl_features(df, rewards)

        # Combine
        features = pd.concat([
            q_features, sarsa_features, td_features, dyna_features,
            thompson_features, exec_features, kelly_features
        ], axis=1)

        # Clean up
        features = features.replace([np.inf, -np.inf], np.nan)
        features = features.fillna(0)

        return features

    def get_feature_names(self) -> List[str]:
        """Get list of all feature names."""
        return [
            # Q-Learning (6)
            'RL_QLEARN_SIGNAL', 'RL_QLEARN_TD_ERROR', 'RL_QLEARN_Q_BUY',
            'RL_QLEARN_Q_SELL', 'RL_QLEARN_Q_SPREAD', 'RL_QLEARN_CONFIDENCE',
            # SARSA (5)
            'RL_SARSA_SIGNAL', 'RL_SARSA_TD_ERROR', 'RL_SARSA_Q_DIV',
            'RL_SARSA_SAFE', 'RL_SARSA_CAUTION',
            # TD(λ) (5)
            'RL_TD_VALUE', 'RL_TD_ERROR', 'RL_TD_ELIGIBILITY',
            'RL_TD_CREDIT', 'RL_TD_HORIZON',
            # Dyna-Q (5)
            'RL_DYNA_SIGNAL', 'RL_DYNA_PLANNING', 'RL_DYNA_MODEL_SIZE',
            'RL_DYNA_CONFIDENCE', 'RL_DYNA_NOVELTY',
            # Thompson Sampling (5)
            'RL_TS_SIGNAL', 'RL_TS_UNCERTAINTY', 'RL_TS_SAMPLED',
            'RL_TS_EXPLORE', 'RL_TS_EXPLOIT',
            # Execution (5)
            'RL_EXEC_URGENCY', 'RL_EXEC_AGGRESSION', 'RL_EXEC_VOL_ADJ',
            'RL_EXEC_SPREAD_ADJ', 'RL_EXEC_REGIME',
            # Kelly-RL (4)
            'RL_KELLY_FRAC', 'RL_KELLY_EDGE', 'RL_KELLY_CONFIDENCE',
            'RL_KELLY_SIGNAL',
        ]


# Convenience function
def generate_rl_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate RL algorithm-based features.

    Args:
        df: OHLCV DataFrame

    Returns:
        DataFrame with 35 pure algorithmic RL features
    """
    generator = RLFeatureGenerator()
    return generator.generate_all(df)


def get_rl_citations() -> Dict[str, List[Dict[str, str]]]:
    """
    Get academic citations for all RL algorithms implemented.

    Returns:
        Dictionary mapping algorithm names to list of citation dictionaries.
        Each citation has: authors, year, title, journal/venue, doi (if available)
    """
    return {
        'q_learning': [
            {
                'authors': 'Watkins, C.J.C.H.',
                'year': '1989',
                'title': 'Learning from Delayed Rewards',
                'venue': 'PhD Thesis, King\'s College, Cambridge University',
                'doi': None,
                'note': 'Original Q-learning algorithm'
            },
            {
                'authors': 'Watkins, C.J.C.H. & Dayan, P.',
                'year': '1992',
                'title': 'Q-learning',
                'venue': 'Machine Learning, 8(3-4), 279-292',
                'doi': '10.1007/BF00992698',
                'note': 'Convergence proof'
            }
        ],
        'sarsa': [
            {
                'authors': 'Rummery, G.A. & Niranjan, M.',
                'year': '1994',
                'title': 'On-Line Q-Learning Using Connectionist Systems',
                'venue': 'Technical Report CUED/F-INFENG/TR 166, Cambridge University',
                'doi': None,
                'note': 'Original SARSA algorithm'
            }
        ],
        'td_lambda': [
            {
                'authors': 'Sutton, R.S.',
                'year': '1988',
                'title': 'Learning to Predict by the Methods of Temporal Differences',
                'venue': 'Machine Learning, 3(1), 9-44',
                'doi': '10.1007/BF00115009',
                'note': 'Foundation of TD learning'
            },
            {
                'authors': 'Tesauro, G.',
                'year': '1994',
                'title': 'TD-Gammon, a Self-Teaching Backgammon Program',
                'venue': 'Neural Computation, 6(2), 215-219',
                'doi': None,
                'note': 'Famous TD(λ) application'
            }
        ],
        'dyna_q': [
            {
                'authors': 'Sutton, R.S.',
                'year': '1991',
                'title': 'Dyna, an Integrated Architecture for Learning, Planning, and Reacting',
                'venue': 'ACM SIGART Bulletin, 2(4), 160-163',
                'doi': None,
                'note': 'Original Dyna architecture'
            }
        ],
        'thompson_sampling': [
            {
                'authors': 'Thompson, W.R.',
                'year': '1933',
                'title': 'On the Likelihood that One Unknown Probability Exceeds Another',
                'venue': 'Biometrika, 25(3-4), 285-294',
                'doi': '10.2307/2332286',
                'note': 'Original Thompson Sampling (90+ years old, still SOTA)'
            },
            {
                'authors': 'Agrawal, S. & Goyal, N.',
                'year': '2012',
                'title': 'Analysis of Thompson Sampling for the Multi-armed Bandit Problem',
                'venue': 'Conference on Learning Theory (COLT)',
                'doi': None,
                'note': 'Modern regret analysis'
            },
            {
                'authors': 'Russo, D.J. et al.',
                'year': '2018',
                'title': 'A Tutorial on Thompson Sampling',
                'venue': 'Foundations and Trends in Machine Learning, 11(1), 1-96',
                'doi': '10.1561/2200000070',
                'note': 'Comprehensive tutorial'
            }
        ],
        'almgren_chriss': [
            {
                'authors': 'Almgren, R. & Chriss, N.',
                'year': '2001',
                'title': 'Optimal Execution of Portfolio Transactions',
                'venue': 'Journal of Risk, 3(2), 5-39',
                'doi': None,
                'note': 'Seminal optimal execution paper'
            },
            {
                'authors': 'Almgren, R.',
                'year': '2003',
                'title': 'Optimal Execution with Nonlinear Impact Functions',
                'venue': 'Applied Mathematical Finance, 10(1), 1-18',
                'doi': '10.1080/135048602100056',
                'note': 'Nonlinear extension'
            },
            {
                'authors': 'Ning, B., Ling, F.H.T. & Jaimungal, S.',
                'year': '2021',
                'title': 'Double Deep Q-Learning for Optimal Execution',
                'venue': 'Applied Mathematical Finance, 28(4), 361-380',
                'doi': '10.1080/1350486X.2022.2077783',
                'note': 'RL extension with 10.3% improvement'
            }
        ],
        'kelly_criterion': [
            {
                'authors': 'Kelly, J.L.',
                'year': '1956',
                'title': 'A New Interpretation of Information Rate',
                'venue': 'Bell System Technical Journal, 35(4), 917-926',
                'doi': '10.1002/j.1538-7305.1956.tb03809.x',
                'note': 'Original Kelly criterion'
            },
            {
                'authors': 'Thorp, E.O.',
                'year': '2006',
                'title': 'The Kelly Criterion in Blackjack, Sports Betting, and the Stock Market',
                'venue': 'Handbook of Asset and Liability Management, 1, 385-428',
                'doi': None,
                'note': 'Kelly in finance'
            }
        ],
        'rl_in_finance': [
            {
                'authors': 'Hambly, B., Xu, R. & Yang, H.',
                'year': '2023',
                'title': 'Recent Advances in Reinforcement Learning in Finance',
                'venue': 'Mathematical Finance, 33(3), 437-503',
                'doi': '10.1111/mafi.12382',
                'note': 'Comprehensive RL finance review'
            },
            {
                'authors': 'Sutton, R.S. & Barto, A.G.',
                'year': '2018',
                'title': 'Reinforcement Learning: An Introduction (2nd ed.)',
                'venue': 'MIT Press, ISBN: 978-0262039246',
                'doi': None,
                'note': 'Definitive RL textbook'
            }
        ],
        'chinese_quant': [
            {
                'authors': 'Liu, X.Y. et al.',
                'year': '2022',
                'title': 'FinRL: Deep Reinforcement Learning Framework to Automate Trading',
                'venue': 'ACM International Conference on AI in Finance (ICAIF)',
                'doi': None,
                'note': 'GitHub: https://github.com/AI4Finance-Foundation/FinRL'
            }
        ]
    }


def print_citations():
    """Print all citations in academic format."""
    citations = get_rl_citations()

    print("=" * 80)
    print("REINFORCEMENT LEARNING ALGORITHMS - ACADEMIC CITATIONS")
    print("=" * 80)

    for algo, refs in citations.items():
        print(f"\n{algo.upper().replace('_', ' ')}:")
        print("-" * 40)
        for ref in refs:
            authors = ref['authors']
            year = ref['year']
            title = ref['title']
            venue = ref['venue']
            doi = ref.get('doi')
            note = ref.get('note', '')

            print(f"  {authors} ({year}).")
            print(f"  \"{title}\"")
            print(f"  {venue}")
            if doi:
                print(f"  DOI: {doi}")
            if note:
                print(f"  [{note}]")
            print()
