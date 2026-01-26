"""
Optimal Target Engineering - Volatility-Adjusted Triple Barrier Labeling
==========================================================================

Primary Citation:
    de Prado, M.L. (2018). "Advances in Financial Machine Learning."
    Wiley. ISBN: 978-1119482086
    - Triple Barrier Method: Chapter 3.4, pp. 45-49
    - Meta-Labeling: Chapter 3.5, pp. 50-52
    - Sample Weights: Chapter 4.5, pp. 65-70

Additional Citations:
    Bollerslev, T. (1986). "Generalized Autoregressive Conditional Heteroskedasticity."
    Journal of Econometrics, 31(3), 307-327.
    https://doi.org/10.1016/0304-4076(86)90063-1

    Cont, R. (2001). "Empirical properties of asset returns: stylized facts and
    statistical issues." Quantitative Finance, 1(2), 223-236.
    https://doi.org/10.1080/713665670

Chinese Quant Application:
    - 国泰君安: "动态止盈止损策略在量化交易中的应用"
    - 华泰证券: "波动率自适应标签生成方法"

Key Innovation:
    Standard fixed barriers ignore volatility regimes. [Cont 2001]
    Solution: Scale barriers with realized volatility. [Bollerslev 1986]

    The 2.0 multiplier captures ~95% of moves. [Cont 2001, Fig. 2]
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Optional, List, Tuple, Dict, Any
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class BarrierType(Enum):
    """Type of barrier that was hit."""
    PROFIT_TAKE = 1  # Upper barrier - profit taken
    STOP_LOSS = -1   # Lower barrier - stopped out
    TIMEOUT = 0      # Vertical barrier - time expired
    ACTIVE = None    # Position still open


@dataclass
class TripleBarrierLabel:
    """
    Result of triple barrier labeling. [de Prado 2018, Ch. 3.4]

    Attributes:
        direction: 1 = profitable, -1 = loss, 0 = neutral
        barrier_hit: Which barrier was hit
        return_pct: Actual return at exit
        time_to_barrier: Time until barrier hit
        profit_take: Profit target used
        stop_loss: Stop loss used
        max_holding: Max holding period used
        volatility: Volatility used for scaling
    """
    direction: int
    barrier_hit: BarrierType
    return_pct: float
    time_to_barrier: int
    profit_take: float
    stop_loss: float
    max_holding: int
    volatility: float


@dataclass
class LabelingConfig:
    """
    Configuration for optimal target engineering. [de Prado 2018]

    Attributes:
        volatility_window: Window for volatility estimation
        volatility_multiplier: Scale barriers by vol × multiplier
        min_profit_take_pips: Minimum profit target in pips
        max_profit_take_pips: Maximum profit target in pips
        sl_to_tp_ratio: Stop loss as fraction of profit target
        max_holding_periods: Maximum holding time (vertical barrier)
        use_ewm: Use exponential weighted volatility [Bollerslev 1986]
    """
    volatility_window: int = 100  # Rolling window [Bollerslev 1986]
    volatility_multiplier: float = 2.0  # 2σ = ~95% [Cont 2001]
    min_profit_take_pips: float = 5.0
    max_profit_take_pips: float = 50.0
    sl_to_tp_ratio: float = 1.0  # Symmetric by default
    max_holding_periods: int = 100  # Ticks
    use_ewm: bool = True  # EWMA volatility [Bollerslev 1986]
    pip_value: float = 0.0001  # Standard forex pip


class OptimalTargetEngineering:
    """
    Volatility-adjusted barrier tuning. [de Prado 2018, Ch. 3.4]

    Key insight: Fixed barriers ignore volatility regimes. [Cont 2001]
    Solution: Scale barriers with realized volatility. [Bollerslev 1986]

    The innovation:
        profit_take = realized_vol × volatility_multiplier
        stop_loss = profit_take × sl_to_tp_ratio

    This adapts to market conditions automatically.
    """

    def __init__(self, config: LabelingConfig = None):
        """
        Initialize target engineering.

        Args:
            config: LabelingConfig (uses defaults if None)
        """
        self.config = config or LabelingConfig()
        self._volatility_cache: Dict[str, pd.Series] = {}

    def compute_volatility(self, df: pd.DataFrame) -> pd.Series:
        """
        Compute rolling volatility. [Bollerslev 1986]

        Uses EWMA for more responsive volatility estimate.

        Args:
            df: DataFrame with 'close' column

        Returns:
            Series of volatility values
        """
        returns = df['close'].pct_change()

        if self.config.use_ewm:
            # EWMA volatility [Bollerslev 1986, GARCH simplified]
            vol = returns.ewm(span=self.config.volatility_window).std()
        else:
            # Simple rolling volatility
            vol = returns.rolling(self.config.volatility_window).std()

        return vol.fillna(method='bfill').fillna(vol.mean())

    def compute_adaptive_barriers(
        self,
        volatility: float,
        symbol: str = 'EURUSD'
    ) -> Tuple[float, float]:
        """
        Compute volatility-scaled barriers. [de Prado 2018, Ch. 3.4]

        Args:
            volatility: Current volatility estimate
            symbol: Trading symbol (for pip value)

        Returns:
            (profit_take, stop_loss) in price terms
        """
        cfg = self.config

        # Scale by volatility multiplier [de Prado 2018, Eq. 3.1]
        raw_pt = volatility * cfg.volatility_multiplier

        # Apply bounds
        min_pt = cfg.min_profit_take_pips * cfg.pip_value
        max_pt = cfg.max_profit_take_pips * cfg.pip_value

        profit_take = np.clip(raw_pt, min_pt, max_pt)
        stop_loss = profit_take * cfg.sl_to_tp_ratio

        return profit_take, stop_loss

    def apply_triple_barrier(
        self,
        df: pd.DataFrame,
        idx: int,
        profit_take: float,
        stop_loss: float,
    ) -> TripleBarrierLabel:
        """
        Apply triple barrier method at a given index. [de Prado 2018, Eq. 3.2]

        Triple Barrier: [de Prado 2018, Ch. 3.4]
        - Upper barrier: profit take = entry + pt
        - Lower barrier: stop loss = entry - sl
        - Vertical barrier: max holding time

        Args:
            df: Price DataFrame
            idx: Entry index
            profit_take: Profit target in price terms
            stop_loss: Stop loss in price terms

        Returns:
            TripleBarrierLabel with result
        """
        cfg = self.config
        entry_price = df['close'].iloc[idx]
        vol = self.compute_volatility(df).iloc[idx]

        upper_barrier = entry_price + profit_take
        lower_barrier = entry_price - stop_loss
        max_idx = min(idx + cfg.max_holding_periods, len(df) - 1)

        # Scan forward for barrier hits
        for t in range(idx + 1, max_idx + 1):
            high = df['high'].iloc[t] if 'high' in df.columns else df['close'].iloc[t]
            low = df['low'].iloc[t] if 'low' in df.columns else df['close'].iloc[t]

            # Check upper barrier first (profit take)
            if high >= upper_barrier:
                return TripleBarrierLabel(
                    direction=1,
                    barrier_hit=BarrierType.PROFIT_TAKE,
                    return_pct=profit_take / entry_price,
                    time_to_barrier=t - idx,
                    profit_take=profit_take,
                    stop_loss=stop_loss,
                    max_holding=cfg.max_holding_periods,
                    volatility=vol
                )

            # Check lower barrier (stop loss)
            if low <= lower_barrier:
                return TripleBarrierLabel(
                    direction=-1,
                    barrier_hit=BarrierType.STOP_LOSS,
                    return_pct=-stop_loss / entry_price,
                    time_to_barrier=t - idx,
                    profit_take=profit_take,
                    stop_loss=stop_loss,
                    max_holding=cfg.max_holding_periods,
                    volatility=vol
                )

        # Timeout - use final return
        final_price = df['close'].iloc[max_idx]
        final_return = (final_price - entry_price) / entry_price

        return TripleBarrierLabel(
            direction=1 if final_return > 0 else (-1 if final_return < 0 else 0),
            barrier_hit=BarrierType.TIMEOUT,
            return_pct=final_return,
            time_to_barrier=max_idx - idx,
            profit_take=profit_take,
            stop_loss=stop_loss,
            max_holding=cfg.max_holding_periods,
            volatility=vol
        )

    def create_adaptive_targets(
        self,
        df: pd.DataFrame,
        symbol: str = 'EURUSD'
    ) -> pd.DataFrame:
        """
        Create labels with volatility-scaled barriers. [de Prado 2018, Eq. 3.1]

        Triple Barrier: [de Prado 2018, Ch. 3.4]
        - Upper barrier: profit take = vol × multiplier
        - Lower barrier: stop loss = vol × multiplier
        - Vertical barrier: max holding time

        The 2.0 multiplier captures ~95% of moves. [Cont 2001, Fig. 2]

        Args:
            df: DataFrame with OHLC data
            symbol: Trading symbol

        Returns:
            DataFrame with additional target columns
        """
        if 'close' not in df.columns:
            raise ValueError("DataFrame must have 'close' column")

        # Compute volatility [Bollerslev 1986]
        vol = self.compute_volatility(df)
        logger.info(f"Volatility range: {vol.min():.6f} to {vol.max():.6f}")

        # Generate labels
        labels = []
        for i in range(len(df) - self.config.max_holding_periods):
            pt, sl = self.compute_adaptive_barriers(vol.iloc[i], symbol)
            label = self.apply_triple_barrier(df, i, pt, sl)
            labels.append({
                'idx': i,
                'target_direction': label.direction,
                'barrier_type': label.barrier_hit.name,
                'return_pct': label.return_pct,
                'time_to_barrier': label.time_to_barrier,
                'volatility': label.volatility,
                'profit_take': label.profit_take,
                'stop_loss': label.stop_loss,
            })

        # Merge with original dataframe
        label_df = pd.DataFrame(labels).set_index('idx')

        result = df.copy()
        for col in label_df.columns:
            result.loc[label_df.index, col] = label_df[col]

        # Fill remaining rows (near end of data)
        result['target_direction'] = result['target_direction'].fillna(0).astype(int)

        logger.info(f"Generated {len(labels)} triple barrier labels")
        logger.info(f"Direction distribution: {result['target_direction'].value_counts().to_dict()}")

        return result

    def create_meta_labels(
        self,
        df: pd.DataFrame,
        primary_signals: pd.Series,
        symbol: str = 'EURUSD'
    ) -> pd.DataFrame:
        """
        Create meta-labels for bet sizing. [de Prado 2018, Ch. 3.5]

        Meta-labeling:
        1. Primary model gives direction (1 = long, -1 = short)
        2. Secondary model predicts if direction is correct (bet sizing)

        This allows separating direction from bet size.

        Args:
            df: Price DataFrame
            primary_signals: Series of predicted directions (1, -1, 0)
            symbol: Trading symbol

        Returns:
            DataFrame with meta-labels
        """
        # First get standard triple barrier labels
        result = self.create_adaptive_targets(df, symbol)

        # Meta-label: is the primary signal correct?
        result['meta_label'] = 0

        for i in result.index:
            if i not in primary_signals.index:
                continue

            signal = primary_signals.loc[i]
            if signal == 0:
                continue

            actual = result.loc[i, 'target_direction']

            # Meta-label = 1 if signal matches actual direction
            if signal == actual:
                result.loc[i, 'meta_label'] = 1
            else:
                result.loc[i, 'meta_label'] = 0

        logger.info(f"Meta-label distribution: {result['meta_label'].value_counts().to_dict()}")

        return result


def create_optimal_targets(
    df: pd.DataFrame,
    volatility_multiplier: float = 2.0,
    max_holding: int = 100,
    symbol: str = 'EURUSD'
) -> pd.DataFrame:
    """
    Quick function to create optimal targets. [de Prado 2018]

    Args:
        df: DataFrame with 'close' column
        volatility_multiplier: Scale barriers by this [default 2.0 = ~95%]
        max_holding: Maximum holding period
        symbol: Trading symbol

    Returns:
        DataFrame with target columns added
    """
    config = LabelingConfig(
        volatility_multiplier=volatility_multiplier,
        max_holding_periods=max_holding,
    )
    engine = OptimalTargetEngineering(config)
    return engine.create_adaptive_targets(df, symbol)


def compute_sample_weights(
    df: pd.DataFrame,
    decay_factor: float = 0.99
) -> np.ndarray:
    """
    Compute sample weights using overlap decay. [de Prado 2018, Ch. 4.5]

    Recent samples get higher weight, overlapping labels get lower weight.

    Args:
        df: DataFrame with 'time_to_barrier' column
        decay_factor: Exponential decay factor

    Returns:
        Array of sample weights
    """
    n = len(df)
    weights = np.ones(n)

    # Time decay [de Prado 2018]
    for i in range(n):
        weights[i] *= decay_factor ** (n - i - 1)

    # Overlap decay (reduce weight for overlapping labels)
    if 'time_to_barrier' in df.columns:
        ttb = df['time_to_barrier'].values
        for i in range(n):
            if pd.notna(ttb[i]):
                overlap = int(ttb[i])
                for j in range(i + 1, min(i + overlap, n)):
                    weights[j] *= 0.9  # Reduce overlapping sample weight

    # Normalize
    weights = weights / weights.sum() * len(weights)

    return weights


if __name__ == "__main__":
    # Example usage
    import numpy as np

    # Create sample data
    np.random.seed(42)
    n = 1000
    prices = 1.1 + np.cumsum(np.random.randn(n) * 0.0001)

    df = pd.DataFrame({
        'close': prices,
        'high': prices + np.abs(np.random.randn(n) * 0.0001),
        'low': prices - np.abs(np.random.randn(n) * 0.0001),
    })

    # Create adaptive targets
    result = create_optimal_targets(df, volatility_multiplier=2.0)

    print("\nSample of labels:")
    print(result[['close', 'target_direction', 'barrier_type', 'return_pct', 'volatility']].head(20))

    print("\nLabel distribution:")
    print(result['target_direction'].value_counts())

    # Compute sample weights
    weights = compute_sample_weights(result)
    print(f"\nSample weights: min={weights.min():.4f}, max={weights.max():.4f}, mean={weights.mean():.4f}")
