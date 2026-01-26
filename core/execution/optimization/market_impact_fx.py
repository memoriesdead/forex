"""
FX Market Impact Model
=======================
Dealer-based market impact model adapted for OTC FX markets.

Unlike exchange-traded assets, FX has:
- No central order book (OTC via dealers)
- Dealer spreads instead of bid-ask from book
- No public tape (volume is estimated)
- Session-dependent liquidity

Components:
1. Dealer Spread: Base cost of crossing the spread
2. Size Impact: Larger orders get worse pricing (sqrt model)
3. Information Leakage: Permanent price impact from signaling

References:
- Almgren & Chriss (2001): Market microstructure optimal execution
- Kyle (1985): Continuous Auctions and Insider Trading
- BIS Triennial Survey: FX market volume data
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple
from datetime import datetime, timezone

from .config import (
    FXSession, FXSessionConfig, ExecutionConfig, SymbolExecutionConfig,
    get_current_session, get_session_config, get_symbol_config
)


@dataclass
class MarketImpactEstimate:
    """Result of market impact estimation."""
    total_impact_bps: float      # Total expected impact in basis points
    dealer_spread_bps: float     # Spread component
    size_impact_bps: float       # Size-related temporary impact
    info_leakage_bps: float      # Permanent impact from signaling
    session_multiplier: float    # Session adjustment factor
    confidence: float            # Confidence in estimate (0-1)


class FXMarketImpactModel:
    """
    FX-specific market impact model.

    The total cost of execution is:
        Cost = dealer_spread/2 + η·√(size/ADV) + γ·(size/ADV)

    where:
        - dealer_spread: bid-ask spread (session-dependent)
        - η: temporary impact coefficient
        - γ: permanent impact coefficient
        - ADV: average daily volume
    """

    def __init__(self, config: Optional[ExecutionConfig] = None):
        self.config = config or ExecutionConfig()

        # Calibration factors (from empirical studies)
        self.base_temp_impact = 1.0    # bps per sqrt(participation)
        self.base_perm_impact = 0.3    # bps per participation rate
        self.dealer_count_factor = 0.8 # More dealers = lower impact

        # Session impact adjustments
        self.session_multipliers = {
            FXSession.TOKYO: 1.3,      # Lower liquidity, higher impact
            FXSession.LONDON: 0.85,    # High liquidity, lower impact
            FXSession.NEW_YORK: 0.9,
            FXSession.OVERLAP_LN: 0.7, # Best liquidity
            FXSession.OVERLAP_TL: 1.0,
            FXSession.OFF_HOURS: 1.8   # Wide spreads, high impact
        }

    def estimate_impact(self,
                       symbol: str,
                       quantity: float,
                       direction: int,
                       mid_price: float,
                       spread_bps: Optional[float] = None,
                       session: Optional[FXSession] = None,
                       volatility: Optional[float] = None) -> MarketImpactEstimate:
        """
        Estimate market impact for an FX order.

        Args:
            symbol: Currency pair (e.g., 'EURUSD')
            quantity: Order size in base currency units
            direction: 1 for buy, -1 for sell
            mid_price: Current mid price
            spread_bps: Current spread in basis points (optional)
            session: FX session (auto-detected if None)
            volatility: Current volatility (optional)

        Returns:
            MarketImpactEstimate with breakdown of costs
        """
        # Get configurations
        symbol_cfg = get_symbol_config(symbol)
        if session is None:
            session = get_current_session()
        session_cfg = get_session_config(session)

        # Use provided spread or default
        if spread_bps is None:
            spread_bps = symbol_cfg.avg_spread_bps * session_cfg.spread_multiplier

        # Calculate participation rate (size / estimated session volume)
        session_volume = self._estimate_session_volume(symbol_cfg, session_cfg)
        participation_rate = quantity / session_volume if session_volume > 0 else 0.01

        # 1. Dealer spread component (half spread for crossing)
        dealer_spread_cost = spread_bps / 2.0

        # 2. Temporary impact (square root model)
        # Impact increases with sqrt of participation
        temp_impact = self.base_temp_impact * np.sqrt(participation_rate) * 10000
        temp_impact *= session_cfg.spread_multiplier  # Adjust for session

        # 3. Permanent impact (linear in participation)
        # Information leakage to dealers
        perm_impact = self.base_perm_impact * participation_rate * 10000

        # Volatility adjustment (higher vol = more impact)
        if volatility is not None:
            vol_factor = 1.0 + (volatility - 0.0001) * 1000  # Normalize to ~1.0
            temp_impact *= vol_factor
            perm_impact *= vol_factor

        # Session multiplier
        session_mult = self.session_multipliers.get(session, 1.0)

        # Total impact
        total_impact = (dealer_spread_cost + temp_impact + perm_impact) * session_mult

        # Confidence based on data quality
        confidence = 0.8 if spread_bps == symbol_cfg.avg_spread_bps else 0.9

        return MarketImpactEstimate(
            total_impact_bps=total_impact,
            dealer_spread_bps=dealer_spread_cost * session_mult,
            size_impact_bps=temp_impact * session_mult,
            info_leakage_bps=perm_impact * session_mult,
            session_multiplier=session_mult,
            confidence=confidence
        )

    def _estimate_session_volume(self,
                                symbol_cfg: SymbolExecutionConfig,
                                session_cfg: FXSessionConfig) -> float:
        """Estimate volume available during current session."""
        daily_volume = symbol_cfg.daily_volume_estimate

        # Distribute across session based on volume weight
        session_hours = session_cfg.end_hour - session_cfg.start_hour
        if session_hours < 0:
            session_hours += 24

        # Session's share of daily volume
        session_volume = daily_volume * session_cfg.volume_weight

        # Adjust for liquidity characteristics
        session_volume *= session_cfg.liquidity_multiplier

        return session_volume

    def optimal_execution_rate(self,
                              symbol: str,
                              total_quantity: float,
                              horizon_seconds: float,
                              session: Optional[FXSession] = None) -> float:
        """
        Calculate optimal execution rate to minimize impact.

        Returns shares per second.
        """
        if session is None:
            session = get_current_session()

        session_cfg = get_session_config(session)
        symbol_cfg = get_symbol_config(symbol)

        # Estimate volume per second during session
        session_volume = self._estimate_session_volume(symbol_cfg, session_cfg)
        session_hours = session_cfg.end_hour - session_cfg.start_hour
        if session_hours <= 0:
            session_hours += 24
        volume_per_second = session_volume / (session_hours * 3600)

        # Max participation rate from config
        max_rate = volume_per_second * self.config.max_participation_rate

        # Target rate to complete within horizon
        target_rate = total_quantity / horizon_seconds

        # Use minimum of target and max allowed
        optimal_rate = min(target_rate, max_rate)

        return optimal_rate

    def estimate_execution_cost(self,
                               symbol: str,
                               quantity: float,
                               horizon_seconds: float,
                               num_slices: int,
                               session: Optional[FXSession] = None) -> float:
        """
        Estimate total execution cost for a sliced execution.

        Returns expected cost in basis points.
        """
        if session is None:
            session = get_current_session()

        slice_qty = quantity / num_slices
        total_cost = 0.0

        for i in range(num_slices):
            # Each slice has its own impact
            impact = self.estimate_impact(
                symbol=symbol,
                quantity=slice_qty,
                direction=1,
                mid_price=1.0,  # Normalized
                session=session
            )
            total_cost += impact.total_impact_bps

        # Average cost per unit
        return total_cost / num_slices


class TemporaryImpactModel:
    """
    Temporary (instantaneous) market impact model.

    Temporary impact affects only the current trade and
    decays quickly after execution.

    Model: η · σ · sign(x) · |x|^α
    where x = trade_rate / volume_rate
    """

    def __init__(self, eta: float = 0.1, alpha: float = 0.5):
        """
        Args:
            eta: Impact coefficient
            alpha: Power law exponent (0.5 = square root)
        """
        self.eta = eta
        self.alpha = alpha

    def impact(self,
              trade_rate: float,
              volume_rate: float,
              volatility: float,
              direction: int) -> float:
        """
        Calculate temporary impact in price units.

        Args:
            trade_rate: Our trading rate (units/second)
            volume_rate: Market volume rate (units/second)
            volatility: Price volatility
            direction: 1 for buy, -1 for sell

        Returns:
            Temporary impact in price units
        """
        if volume_rate <= 0:
            return 0.0

        participation = trade_rate / volume_rate
        impact = self.eta * volatility * direction * np.power(participation, self.alpha)

        return impact


class PermanentImpactModel:
    """
    Permanent market impact model.

    Permanent impact persists after the trade and
    represents information leakage to the market.

    Model: γ · x
    where x = total_traded / ADV
    """

    def __init__(self, gamma: float = 0.1):
        """
        Args:
            gamma: Permanent impact coefficient
        """
        self.gamma = gamma

    def impact(self,
              total_traded: float,
              adv: float,
              direction: int) -> float:
        """
        Calculate permanent impact in price units.

        Args:
            total_traded: Total quantity traded
            adv: Average daily volume
            direction: 1 for buy, -1 for sell

        Returns:
            Permanent impact in price units
        """
        if adv <= 0:
            return 0.0

        participation = total_traded / adv
        impact = self.gamma * direction * participation

        return impact


class DealerSpreadModel:
    """
    FX dealer spread model.

    In FX, the "spread" is not from an order book but from
    dealer quotes. This model estimates dealer behavior.
    """

    def __init__(self,
                 base_spread_bps: float = 1.0,
                 size_sensitivity: float = 0.1,
                 info_sensitivity: float = 0.2):
        """
        Args:
            base_spread_bps: Base spread in basis points
            size_sensitivity: How much spread widens with size
            info_sensitivity: How much spread widens with info asymmetry
        """
        self.base_spread_bps = base_spread_bps
        self.size_sensitivity = size_sensitivity
        self.info_sensitivity = info_sensitivity

    def estimate_spread(self,
                       quantity: float,
                       typical_size: float,
                       session: FXSession,
                       volatility: float = 0.0001,
                       recent_order_flow: float = 0.0) -> Tuple[float, float]:
        """
        Estimate dealer spread for a given order.

        Args:
            quantity: Order size
            typical_size: Typical order size in market
            session: Current FX session
            volatility: Current price volatility
            recent_order_flow: Recent net order flow (+ = buying pressure)

        Returns:
            Tuple of (bid_spread_bps, ask_spread_bps)
        """
        session_cfg = get_session_config(session)

        # Base spread adjusted for session
        spread = self.base_spread_bps * session_cfg.spread_multiplier

        # Size adjustment (larger orders get wider spreads)
        size_ratio = quantity / typical_size if typical_size > 0 else 1.0
        size_adj = self.size_sensitivity * np.log1p(size_ratio)

        # Volatility adjustment
        vol_adj = volatility * 10000 * 0.5  # Convert to bps

        # Order flow adjustment (asymmetric spread based on flow)
        flow_adj = self.info_sensitivity * recent_order_flow

        total_spread = spread + size_adj + vol_adj

        # Distribute spread (potentially asymmetric)
        bid_spread = total_spread / 2 + flow_adj / 2
        ask_spread = total_spread / 2 - flow_adj / 2

        return (max(0.1, bid_spread), max(0.1, ask_spread))


def get_market_impact_model(config: Optional[ExecutionConfig] = None) -> FXMarketImpactModel:
    """Factory function to get market impact model."""
    return FXMarketImpactModel(config)


if __name__ == '__main__':
    print("FX Market Impact Model Test")
    print("=" * 60)

    model = FXMarketImpactModel()

    # Test different order sizes
    test_sizes = [10_000, 100_000, 1_000_000, 10_000_000]

    print("\nEURUSD Market Impact by Size (London Session)")
    print("-" * 60)
    for size in test_sizes:
        impact = model.estimate_impact(
            symbol='EURUSD',
            quantity=size,
            direction=1,
            mid_price=1.0850,
            session=FXSession.LONDON
        )
        print(f"Size: ${size:>12,} | "
              f"Total: {impact.total_impact_bps:6.2f} bps | "
              f"Spread: {impact.dealer_spread_bps:4.2f} | "
              f"Size: {impact.size_impact_bps:4.2f} | "
              f"Info: {impact.info_leakage_bps:4.2f}")

    # Test different sessions
    print("\n" + "=" * 60)
    print("\nEURUSD 1M Impact by Session")
    print("-" * 60)
    for session in FXSession:
        impact = model.estimate_impact(
            symbol='EURUSD',
            quantity=1_000_000,
            direction=1,
            mid_price=1.0850,
            session=session
        )
        print(f"{session.value:12s} | "
              f"Total: {impact.total_impact_bps:6.2f} bps | "
              f"Multiplier: {impact.session_multiplier:.2f}x")

    # Test execution cost over time
    print("\n" + "=" * 60)
    print("\nExecution Cost: 10M EURUSD in London")
    print("-" * 60)
    horizons = [60, 300, 600, 1800, 3600]
    slices_map = {60: 2, 300: 10, 600: 20, 1800: 60, 3600: 120}

    for horizon in horizons:
        slices = slices_map[horizon]
        cost = model.estimate_execution_cost(
            symbol='EURUSD',
            quantity=10_000_000,
            horizon_seconds=horizon,
            num_slices=slices,
            session=FXSession.LONDON
        )
        print(f"Horizon: {horizon:4d}s ({slices:3d} slices) | "
              f"Cost: {cost:5.2f} bps")
