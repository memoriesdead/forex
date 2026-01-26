"""
Adverse Selection Modeling for HFT

Models the probability of being adversely selected (filled when price moves against you).
Critical for understanding TRUE edge after accounting for informed traders.

Academic Citations:
- Glosten & Milgrom (1985): "Bid, Ask and Transaction Prices in a Specialist Market"
  Journal of Financial Economics - Foundational adverse selection model

- Kyle (1985): "Continuous Auctions and Insider Trading"
  Econometrica - Information-based market microstructure

- Easley & O'Hara (1987): "Price, Trade Size, and Information in Securities Markets"
  Journal of Financial Economics - PIN model origins

- Oxford Man Institute (2024): "Adverse Selection and Market Making in HFT"
  Working Paper - Modern ML approach to adverse selection

Chinese Quant Application:
- 中信证券: 逆向选择成本分析 (adverse selection cost analysis)
- 海通证券: 订单毒性检测 (order toxicity detection)
- 华泰证券: 知情交易者识别 (informed trader identification)

The Core Problem:
    When your limit order gets filled, it's often because:
    1. An informed trader knows something you don't
    2. Price is about to move against you

    Fill probability given adverse move:
    P(fill | price drops) > P(fill | price rises)  for buy orders

    This is the "winner's curse" of market making.

Why This Matters for Certainty:
    Your 82% win rate might be CONDITIONAL on not being adversely selected.
    True edge = Win rate accounting for adverse selection bias.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Tuple, Callable
from scipy import stats
import warnings


@dataclass
class AdverseSelectionResult:
    """Result of adverse selection analysis."""

    # Core metrics
    probability_informed: float  # P(trade was from informed trader)
    adverse_selection_cost_bps: float  # Cost attributed to AS
    toxicity_score: float  # 0-1 score of order toxicity

    # Conditional probabilities
    p_fill_given_adverse: float  # P(filled | price moves against)
    p_fill_given_favorable: float  # P(filled | price moves in favor)

    # Information content
    expected_information_loss: float  # Expected loss from informed traders
    winner_curse_adjustment: float  # Adjustment to apply to win rate

    # Confidence
    confidence: float  # Confidence in the estimate


@dataclass
class FillAnalysis:
    """Analysis of fill behavior for adverse selection detection."""

    # Fill rates
    overall_fill_rate: float
    fill_rate_adverse_move: float  # Fill rate when price moved against
    fill_rate_favorable_move: float  # Fill rate when price moved favorably

    # Adverse selection ratio
    as_ratio: float  # fill_adverse / fill_favorable (>1 = AS present)

    # Post-fill performance
    avg_return_after_fill: float  # Average return after getting filled
    pct_adverse_fills: float  # % of fills that were adversely selected


class GlostenMilgromModel:
    """
    Glosten-Milgrom (1985) Adverse Selection Model.

    The spread exists because of adverse selection:
    - Market maker faces informed and uninformed traders
    - Must set bid/ask to break even on average
    - Informed traders cause losses, uninformed provide profits

    The Model:
        V = True value (unknown)
        V_H = High value, V_L = Low value
        μ = Probability trade is informed
        σ = Probability informed trader knows V_H

    Bid = E[V | sell order] = V_L + (1-μ)(V_H - V_L) / 2
    Ask = E[V | buy order] = V_L + (1+μ)(V_H - V_L) / 2

    Spread = Ask - Bid = μ(V_H - V_L)

    Reference:
        Glosten & Milgrom (1985), Proposition 1
    """

    def __init__(
        self,
        informed_prob: float = 0.1,  # μ: Probability of informed trader
        sigma: float = 0.5,  # Information precision
    ):
        """
        Initialize Glosten-Milgrom model.

        Args:
            informed_prob: Probability a trade is from informed trader
            sigma: Probability informed trader has correct signal
        """
        self.mu = informed_prob
        self.sigma = sigma

    def compute_spread(
        self,
        value_range: float,  # V_H - V_L
    ) -> Tuple[float, float]:
        """
        Compute equilibrium bid-ask spread.

        Args:
            value_range: Range of possible values

        Returns:
            (spread, adverse_selection_component)
        """
        spread = self.mu * value_range
        as_component = spread  # Entire spread is due to AS in this model

        return spread, as_component

    def adverse_selection_cost(
        self,
        fill_price: float,
        true_value: float,
        is_buy: bool,
    ) -> float:
        """
        Calculate adverse selection cost of a filled order.

        Args:
            fill_price: Price at which order filled
            true_value: True value at fill time
            is_buy: Whether this was a buy order

        Returns:
            Adverse selection cost (positive = loss)
        """
        if is_buy:
            # Buy: AS cost = fill_price - true_value
            return fill_price - true_value
        else:
            # Sell: AS cost = true_value - fill_price
            return true_value - fill_price


class PINModel:
    """
    Probability of Informed Trading (PIN) Model.

    Estimates the probability that a trade is information-driven.

    The Model:
        α = Probability of information event
        δ = Probability bad news given event
        μ = Informed arrival rate
        ε_b = Uninformed buy rate
        ε_s = Uninformed sell rate

    PIN = αμ / (αμ + ε_b + ε_s)

    Reference:
        Easley, Kiefer & O'Hara (1997): "One Day in the Life of a Very Common Stock"
    """

    def __init__(
        self,
        alpha: float = 0.3,  # Info event probability
        delta: float = 0.5,  # Bad news probability
        mu: float = 0.2,  # Informed rate
        epsilon_b: float = 0.4,  # Uninformed buy rate
        epsilon_s: float = 0.4,  # Uninformed sell rate
    ):
        """
        Initialize PIN model.

        Args:
            alpha: Probability of information event
            delta: Probability of bad news given event
            mu: Arrival rate of informed traders
            epsilon_b: Arrival rate of uninformed buyers
            epsilon_s: Arrival rate of uninformed sellers
        """
        self.alpha = alpha
        self.delta = delta
        self.mu = mu
        self.epsilon_b = epsilon_b
        self.epsilon_s = epsilon_s

    def compute_pin(self) -> float:
        """
        Compute Probability of Informed Trading.

        Returns:
            PIN value (0-1)
        """
        informed_rate = self.alpha * self.mu
        total_rate = informed_rate + self.epsilon_b + self.epsilon_s

        return informed_rate / total_rate if total_rate > 0 else 0.0

    def estimate_from_counts(
        self,
        buy_count: int,
        sell_count: int,
    ) -> float:
        """
        Estimate PIN from buy/sell counts.

        Simplified estimation (full MLE is more complex).

        Args:
            buy_count: Number of buyer-initiated trades
            sell_count: Number of seller-initiated trades

        Returns:
            Estimated PIN
        """
        total = buy_count + sell_count
        if total == 0:
            return 0.0

        # Order imbalance as proxy for informed trading
        imbalance = abs(buy_count - sell_count) / total

        # Simple approximation: high imbalance = more informed trading
        estimated_pin = min(imbalance * 2, 1.0)

        return estimated_pin


class MLAdverseSelectionDetector:
    """
    ML-based adverse selection detection (Oxford 2024 style).

    Uses features to predict probability of adverse selection.
    """

    def __init__(self):
        """Initialize ML adverse selection detector."""
        self._model = None
        self._fitted = False

        # Tracking
        self._fill_history: List[Dict] = []

    def _extract_features(
        self,
        spread_bps: float,
        book_imbalance: float,
        volume_ratio: float,
        recent_return: float,
        order_size_pct: float,
        time_of_day_frac: float,
    ) -> np.ndarray:
        """
        Extract features for AS prediction.

        Args:
            spread_bps: Current spread
            book_imbalance: (bid_vol - ask_vol) / total
            volume_ratio: Current vs average volume
            recent_return: Return over last N periods
            order_size_pct: Order size as % of book depth
            time_of_day_frac: Time as fraction of trading day

        Returns:
            Feature vector
        """
        return np.array([
            spread_bps,
            book_imbalance,
            volume_ratio,
            recent_return,
            order_size_pct,
            time_of_day_frac,
            spread_bps * abs(book_imbalance),  # Interaction
            volume_ratio * abs(recent_return),  # Interaction
        ])

    def predict_adverse_selection(
        self,
        spread_bps: float,
        book_imbalance: float,
        volume_ratio: float,
        recent_return: float,
        order_size_pct: float = 0.01,
        time_of_day_frac: float = 0.5,
    ) -> AdverseSelectionResult:
        """
        Predict adverse selection probability.

        Args:
            spread_bps: Current spread in basis points
            book_imbalance: Order book imbalance
            volume_ratio: Current volume vs average
            recent_return: Recent price return
            order_size_pct: Order size as % of depth
            time_of_day_frac: Time of day

        Returns:
            AdverseSelectionResult with predictions
        """
        # Heuristic model (replace with trained ML in production)

        # Spread widens when informed traders are active
        spread_signal = min(spread_bps / 5.0, 1.0)

        # High imbalance = directional pressure
        imbalance_signal = abs(book_imbalance)

        # High volume often accompanies informed trading
        volume_signal = min(volume_ratio / 3.0, 1.0) if volume_ratio > 1 else 0

        # Recent momentum = possible information
        momentum_signal = min(abs(recent_return) / 0.001, 1.0)

        # Combine signals
        toxicity = 0.25 * spread_signal + 0.30 * imbalance_signal + \
                   0.25 * volume_signal + 0.20 * momentum_signal

        # Probability informed
        p_informed = min(toxicity * 0.3, 0.5)  # Cap at 50%

        # Conditional fill probabilities
        # If adverse selection exists, fills more likely when price moves against
        p_fill_adverse = 0.7 + 0.2 * toxicity
        p_fill_favorable = 0.5 - 0.2 * toxicity

        # AS cost estimate
        as_cost = p_informed * spread_bps * 0.5

        # Winner's curse adjustment
        # If AS ratio is high, reduce effective win rate
        as_ratio = p_fill_adverse / max(p_fill_favorable, 0.1)
        winner_curse = 1 / (1 + 0.1 * (as_ratio - 1))

        return AdverseSelectionResult(
            probability_informed=p_informed,
            adverse_selection_cost_bps=as_cost,
            toxicity_score=toxicity,
            p_fill_given_adverse=p_fill_adverse,
            p_fill_given_favorable=p_fill_favorable,
            expected_information_loss=as_cost,
            winner_curse_adjustment=winner_curse,
            confidence=0.7,  # Heuristic model confidence
        )

    def record_fill(
        self,
        fill_price: float,
        mid_price_at_fill: float,
        mid_price_after: float,
        is_buy: bool,
    ) -> None:
        """
        Record a fill for adverse selection analysis.

        Args:
            fill_price: Price at which order filled
            mid_price_at_fill: Mid price when filled
            mid_price_after: Mid price after N periods
        """
        # Calculate realized adverse selection
        if is_buy:
            # Buy: adverse if price dropped after fill
            adverse = mid_price_after < mid_price_at_fill
        else:
            # Sell: adverse if price rose after fill
            adverse = mid_price_after > mid_price_at_fill

        self._fill_history.append({
            'fill_price': fill_price,
            'mid_at_fill': mid_price_at_fill,
            'mid_after': mid_price_after,
            'is_buy': is_buy,
            'adverse': adverse,
        })

    def analyze_fills(self) -> FillAnalysis:
        """
        Analyze fill history for adverse selection patterns.

        Returns:
            FillAnalysis with statistics
        """
        if len(self._fill_history) < 20:
            return FillAnalysis(
                overall_fill_rate=0.0,
                fill_rate_adverse_move=0.0,
                fill_rate_favorable_move=0.0,
                as_ratio=1.0,
                avg_return_after_fill=0.0,
                pct_adverse_fills=0.0,
            )

        # Calculate statistics
        adverse_count = sum(1 for f in self._fill_history if f['adverse'])
        total_count = len(self._fill_history)

        pct_adverse = adverse_count / total_count

        # Average return after fill
        returns = []
        for f in self._fill_history:
            ret = (f['mid_after'] - f['mid_at_fill']) / f['mid_at_fill']
            if not f['is_buy']:
                ret = -ret  # Flip sign for sells
            returns.append(ret)

        avg_return = np.mean(returns)

        # AS ratio (simplified)
        as_ratio = pct_adverse / max(1 - pct_adverse, 0.1)

        return FillAnalysis(
            overall_fill_rate=1.0,  # All recorded are fills
            fill_rate_adverse_move=pct_adverse,
            fill_rate_favorable_move=1 - pct_adverse,
            as_ratio=as_ratio,
            avg_return_after_fill=avg_return * 10000,  # bps
            pct_adverse_fills=pct_adverse,
        )


# =============================================================================
# HFT-Optimized Functions
# =============================================================================

def quick_toxicity_score(
    spread_bps: float,
    book_imbalance: float,
    volume_ratio: float,
) -> float:
    """
    Ultra-fast toxicity score for HFT.

    Higher score = more likely to be adversely selected.

    Args:
        spread_bps: Current spread
        book_imbalance: Order book imbalance (-1 to 1)
        volume_ratio: Current volume / average volume

    Returns:
        Toxicity score (0-1)
    """
    # Normalize inputs
    spread_norm = min(spread_bps / 5.0, 1.0)
    imbalance_norm = abs(book_imbalance)
    volume_norm = min(max(volume_ratio - 1, 0) / 2.0, 1.0)

    # Weighted combination
    toxicity = 0.35 * spread_norm + 0.40 * imbalance_norm + 0.25 * volume_norm

    return toxicity


def adjusted_win_rate(
    gross_win_rate: float,
    adverse_selection_prob: float,
    as_cost_bps: float,
    avg_profit_bps: float,
) -> float:
    """
    Adjust win rate for adverse selection.

    True win rate < Gross win rate when AS is present.

    Args:
        gross_win_rate: Observed win rate
        adverse_selection_prob: P(adversely selected)
        as_cost_bps: Adverse selection cost
        avg_profit_bps: Average profit per winning trade

    Returns:
        Adjusted win rate accounting for AS
    """
    # Effective win rate reduction from AS
    # When adversely selected, even "wins" may be smaller
    as_adjustment = adverse_selection_prob * as_cost_bps / avg_profit_bps

    adjusted = gross_win_rate * (1 - as_adjustment)

    return max(adjusted, 0.0)


def should_trade_given_toxicity(
    toxicity_score: float,
    signal_strength: float,
    max_toxicity: float = 0.7,
) -> bool:
    """
    Decide whether to trade based on toxicity level.

    Args:
        toxicity_score: Current toxicity (0-1)
        signal_strength: Strength of trading signal (0-1)
        max_toxicity: Maximum toxicity to tolerate

    Returns:
        True if should trade
    """
    if toxicity_score > max_toxicity:
        return False

    # Require stronger signals when toxicity is higher
    required_signal = 0.5 + 0.5 * toxicity_score

    return signal_strength >= required_signal


# =============================================================================
# Example Usage
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("ADVERSE SELECTION MODELING FOR HFT")
    print("=" * 70)
    print()
    print("Citations:")
    print("  - Glosten & Milgrom (1985): Foundational AS model")
    print("  - Kyle (1985): Information-based microstructure")
    print("  - Easley & O'Hara (1987): PIN model origins")
    print("  - Oxford Man Institute (2024): ML adverse selection")
    print()

    # Glosten-Milgrom
    print("GLOSTEN-MILGROM MODEL")
    print("-" * 50)

    gm = GlostenMilgromModel(informed_prob=0.15)
    spread, as_cost = gm.compute_spread(value_range=0.001)

    print(f"Informed trader probability: 15%")
    print(f"Equilibrium spread: {spread*10000:.2f} bps")
    print(f"Adverse selection component: {as_cost*10000:.2f} bps")
    print()

    # PIN Model
    print("PIN MODEL")
    print("-" * 50)

    pin = PINModel(alpha=0.3, mu=0.2, epsilon_b=0.4, epsilon_s=0.4)
    pin_value = pin.compute_pin()

    print(f"Probability of Informed Trading: {pin_value:.2%}")
    print()

    # Quick toxicity
    print("QUICK TOXICITY SCORES")
    print("-" * 50)

    scenarios = [
        (1.0, 0.1, 1.0, "Normal market"),
        (3.0, 0.4, 2.0, "Elevated toxicity"),
        (5.0, 0.7, 3.0, "High toxicity"),
    ]

    for spread, imbalance, volume, desc in scenarios:
        tox = quick_toxicity_score(spread, imbalance, volume)
        print(f"{desc}:")
        print(f"  Spread={spread}bps, Imbalance={imbalance:.1f}, Volume={volume}x")
        print(f"  Toxicity score: {tox:.2f}")
        should = should_trade_given_toxicity(tox, signal_strength=0.7)
        print(f"  Trade with 70% signal? {'YES' if should else 'NO'}")
        print()

    # Win rate adjustment
    print("WIN RATE ADJUSTMENT")
    print("-" * 50)

    gross_wr = 0.82
    as_prob = 0.15
    as_cost = 0.5  # bps
    avg_profit = 10  # bps

    adj_wr = adjusted_win_rate(gross_wr, as_prob, as_cost, avg_profit)

    print(f"Gross win rate: {gross_wr:.0%}")
    print(f"AS probability: {as_prob:.0%}")
    print(f"AS cost: {as_cost} bps")
    print(f"Adjusted win rate: {adj_wr:.1%}")
    print()

    print("=" * 70)
    print("KEY INSIGHT:")
    print("  Your 82% win rate is CONDITIONAL on market conditions.")
    print("  When adversely selected, you're filled because price")
    print("  is about to move against you. Account for this!")
    print()
    print("  True edge = Edge accounting for adverse selection bias")
    print("=" * 70)
