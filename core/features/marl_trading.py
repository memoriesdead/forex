"""
Multi-Agent Reinforcement Learning (MARL) Trading Features
==========================================================
Feature extraction inspired by multi-agent market dynamics from Chinese research.

Citations:
[1] Sun, S. et al. (2023). "Multi-Agent Reinforcement Learning for Quantitative
    Trading: A Survey" arXiv:2302.13753.
    Comprehensive survey of MARL in trading.

[2] 国泰君安/Fudan University (2024). "Market as Multi-Agent System"
    Chinese institutional research on agent-based market modeling.

[3] Littman, M.L. (1994). "Markov Games as a Framework for Multi-Agent
    Reinforcement Learning" ICML.
    Foundation for multi-agent learning in games.

[4] Hu, J. & Wellman, M.P. (2003). "Nash Q-Learning for General-Sum
    Stochastic Games" JMLR.
    Nash equilibrium learning in multi-agent settings.

Key Innovation:
    - Market = multiple interacting agents with different strategies
    - Price is emergent from agent interactions
    - Features capture agent behavior signatures

Implementation:
    - Agent consensus signals
    - Market impact from agent actions
    - Equilibrium price estimation
    - Agent type detection

Total: 15 features

Usage:
    from core.features.marl_trading import MultiAgentRLFeatures
    marl = MultiAgentRLFeatures()
    features = marl.generate_all(df)
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Tuple
import warnings
from scipy import stats
from scipy.special import softmax

warnings.filterwarnings('ignore')


class AgentTypeDetector:
    """
    Detects signatures of different agent types in market data.

    Agent Types:
    - Momentum: Trend followers, buy high/sell low
    - Value: Mean-reversion, buy low/sell high
    - Noise: Random trading, no clear pattern
    - Informed: Anticipates moves, leads price

    Reference: Kyle (1985) on informed vs noise traders
    """

    def detect_momentum_signature(self, returns: pd.Series, window: int = 20) -> pd.Series:
        """Detect momentum agent activity (positive autocorrelation)."""
        autocorr = returns.rolling(window, min_periods=5).apply(
            lambda x: np.corrcoef(x[:-1], x[1:])[0, 1] if len(x) > 2 else 0,
            raw=True
        )
        return autocorr.clip(-1, 1)

    def detect_value_signature(self, returns: pd.Series, window: int = 20) -> pd.Series:
        """Detect value agent activity (negative autocorrelation)."""
        autocorr = self.detect_momentum_signature(returns, window)
        return -autocorr

    def detect_noise_signature(self, returns: pd.Series, window: int = 20) -> pd.Series:
        """Detect noise trader activity (high variance, low autocorrelation)."""
        vol = returns.rolling(window, min_periods=2).std()
        autocorr = self.detect_momentum_signature(returns, window)
        # High vol + low autocorr = noise
        noise_sig = vol / (vol.rolling(60, min_periods=10).mean() + 1e-10) * (1 - np.abs(autocorr))
        return noise_sig

    def detect_informed_signature(
        self,
        returns: pd.Series,
        volume: pd.Series,
        window: int = 20
    ) -> pd.Series:
        """
        Detect informed trader activity.

        Informed traders: large volume precedes price moves
        """
        # Volume leads price (informed trading signature)
        vol_lead = volume.shift(1).rolling(5, min_periods=1).mean()
        ret_follow = returns.abs().rolling(5, min_periods=1).mean()

        # Correlation between lagged volume and current returns
        informed_sig = vol_lead.rolling(window, min_periods=5).corr(ret_follow)
        return informed_sig.fillna(0)


class MultiAgentRLFeatures:
    """
    Multi-Agent Reinforcement Learning Feature Generator.

    Models market as multi-agent system where price emerges from
    interactions between different agent types.

    Citations:
    [1] Sun et al. (2023). "Multi-Agent Reinforcement Learning for
        Quantitative Trading" arXiv:2302.13753.
    [2] Littman (1994). "Markov Games as a Framework for Multi-Agent RL"
    [3] Hu & Wellman (2003). "Nash Q-Learning for General-Sum Games"

    Total: 15 features
    - Agent Detection: 4 features (momentum, value, noise, informed signatures)
    - Agent Consensus: 4 features (agreement, divergence, dominance, balance)
    - Market Impact: 4 features (price impact, volume impact, persistence, decay)
    - Equilibrium: 3 features (Nash estimate, deviation, convergence)
    """

    def __init__(
        self,
        detection_window: int = 20,
        consensus_window: int = 10,
        impact_window: int = 30
    ):
        """
        Initialize MARL feature generator.

        Args:
            detection_window: Window for agent type detection
            consensus_window: Window for consensus signals
            impact_window: Window for market impact estimation
        """
        self.detection_window = detection_window
        self.consensus_window = consensus_window
        self.impact_window = impact_window
        self.agent_detector = AgentTypeDetector()

    # =========================================================================
    # AGENT DETECTION FEATURES (4)
    # =========================================================================

    def _agent_detection_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Detect signatures of different agent types.

        Reference: Sun et al. (2023) MARL Survey
        """
        features = pd.DataFrame(index=df.index)
        close = df['close']
        returns = close.pct_change()
        volume = df.get('volume', pd.Series(1, index=df.index))

        # 1. Momentum agent signature
        features['MARL_agent_momentum'] = self.agent_detector.detect_momentum_signature(
            returns, self.detection_window
        )

        # 2. Value agent signature
        features['MARL_agent_value'] = self.agent_detector.detect_value_signature(
            returns, self.detection_window
        )

        # 3. Noise trader signature
        features['MARL_agent_noise'] = self.agent_detector.detect_noise_signature(
            returns, self.detection_window
        )

        # 4. Informed trader signature
        features['MARL_agent_informed'] = self.agent_detector.detect_informed_signature(
            returns, volume, self.detection_window
        )

        return features

    # =========================================================================
    # AGENT CONSENSUS FEATURES (4)
    # =========================================================================

    def _agent_consensus_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Agent consensus and disagreement features.

        When agents agree, trends are stronger.
        When agents disagree, volatility increases.

        Reference: Littman (1994) Markov Games
        """
        features = pd.DataFrame(index=df.index)
        close = df['close']
        returns = close.pct_change()
        volume = df.get('volume', pd.Series(1, index=df.index))

        # Get agent signatures
        mom_sig = self.agent_detector.detect_momentum_signature(returns, self.detection_window)
        val_sig = self.agent_detector.detect_value_signature(returns, self.detection_window)

        # 1. Agent agreement (both momentum and value aligned)
        # When both positive or both negative = agreement
        agreement = mom_sig * val_sig
        features['MARL_consensus'] = agreement.rolling(self.consensus_window, min_periods=2).mean()

        # 2. Agent divergence (momentum vs value disagreement)
        divergence = np.abs(mom_sig - val_sig)
        features['MARL_divergence'] = divergence.rolling(self.consensus_window, min_periods=2).mean()

        # 3. Dominant agent type
        # Which agent type is winning
        mom_strength = np.abs(mom_sig)
        val_strength = np.abs(val_sig)
        features['MARL_dominant'] = np.where(
            mom_strength > val_strength, 1,  # Momentum dominant
            np.where(val_strength > mom_strength, -1, 0)  # Value dominant / balanced
        )

        # 4. Market balance (how evenly matched agents are)
        total_strength = mom_strength + val_strength + 1e-10
        features['MARL_balance'] = 1 - np.abs(mom_strength - val_strength) / total_strength

        return features

    # =========================================================================
    # MARKET IMPACT FEATURES (4)
    # =========================================================================

    def _market_impact_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Market impact from agent actions.

        Models how agent trading affects prices.

        Reference: Almgren & Chriss (2001) Optimal Execution
        """
        features = pd.DataFrame(index=df.index)
        close = df['close']
        returns = close.pct_change()
        volume = df.get('volume', pd.Series(1, index=df.index))

        # 1. Price impact (return per unit volume)
        # Kyle's lambda proxy
        vol_normalized = volume / (volume.rolling(self.impact_window, min_periods=5).mean() + 1e-10)
        impact = returns.abs() / (vol_normalized + 0.1)
        features['MARL_price_impact'] = impact.rolling(self.impact_window, min_periods=2).mean()

        # 2. Volume impact (volume effect on future returns)
        vol_signal = (volume - volume.rolling(self.impact_window, min_periods=5).mean()) / \
                     (volume.rolling(self.impact_window, min_periods=5).std() + 1e-10)
        future_ret = returns.shift(-1).fillna(0)
        vol_impact = vol_signal.rolling(self.impact_window, min_periods=5).corr(future_ret)
        features['MARL_vol_impact'] = vol_impact.fillna(0)

        # 3. Impact persistence (how long price impact lasts)
        ret_autocorr_1 = returns.rolling(self.impact_window, min_periods=5).apply(
            lambda x: np.corrcoef(x[:-1], x[1:])[0, 1] if len(x) > 2 else 0,
            raw=True
        )
        ret_autocorr_5 = returns.rolling(self.impact_window, min_periods=10).apply(
            lambda x: np.corrcoef(x[:-5], x[5:])[0, 1] if len(x) > 6 else 0,
            raw=True
        )
        features['MARL_impact_persist'] = ret_autocorr_1 - ret_autocorr_5

        # 4. Impact decay rate
        # How quickly impact diminishes
        decay = -np.log(np.abs(ret_autocorr_5) + 0.01) / 5
        features['MARL_impact_decay'] = decay.clip(-1, 1)

        return features

    # =========================================================================
    # EQUILIBRIUM FEATURES (3)
    # =========================================================================

    def _equilibrium_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Nash equilibrium estimation features.

        In multi-agent setting, prices should converge to Nash equilibrium.

        Reference: Hu & Wellman (2003) Nash Q-Learning
        """
        features = pd.DataFrame(index=df.index)
        close = df['close']
        returns = close.pct_change()

        # 1. Nash equilibrium price estimate
        # Rolling fair value based on multiple signals
        ma_short = close.rolling(10, min_periods=1).mean()
        ma_long = close.rolling(50, min_periods=1).mean()

        # Equilibrium is weighted average of different timeframe views
        mom_fair = close + (close - ma_short)  # Momentum: extrapolate trend
        val_fair = ma_long  # Value: revert to mean

        # Get agent strengths
        mom_sig = np.abs(self.agent_detector.detect_momentum_signature(
            returns, self.detection_window))
        val_sig = np.abs(self.agent_detector.detect_value_signature(
            returns, self.detection_window))

        total = mom_sig + val_sig + 1e-10
        nash_price = (mom_sig * mom_fair + val_sig * val_fair) / total
        features['MARL_nash_price'] = (close - nash_price) / (close + 1e-10)

        # 2. Deviation from equilibrium
        deviation = np.abs(features['MARL_nash_price'])
        features['MARL_nash_deviation'] = deviation

        # 3. Convergence indicator
        # Price approaching equilibrium = low and decreasing deviation
        dev_change = deviation.diff(5)
        features['MARL_convergence'] = -dev_change.clip(-0.1, 0.1) / 0.1

        return features

    # =========================================================================
    # MAIN GENERATION METHOD
    # =========================================================================

    def generate_all(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate all MARL features.

        Args:
            df: DataFrame with OHLCV data

        Returns:
            DataFrame with 15 MARL features
        """
        # Validate input
        if 'close' not in df.columns:
            raise ValueError("Missing required column: 'close'")

        df = df.copy()
        if 'open' not in df.columns:
            df['open'] = df['close'].shift(1).fillna(df['close'])
        if 'high' not in df.columns:
            df['high'] = df['close']
        if 'low' not in df.columns:
            df['low'] = df['close']
        if 'volume' not in df.columns:
            df['volume'] = 1

        # Generate all feature groups
        agent_features = self._agent_detection_features(df)
        consensus_features = self._agent_consensus_features(df)
        impact_features = self._market_impact_features(df)
        equilibrium_features = self._equilibrium_features(df)

        # Combine
        result = pd.concat([
            agent_features, consensus_features, impact_features, equilibrium_features
        ], axis=1)

        # Clean up
        result = result.replace([np.inf, -np.inf], np.nan)
        result = result.fillna(0)

        return result

    def get_feature_names(self) -> List[str]:
        """Get list of all feature names."""
        return [
            # Agent Detection (4)
            'MARL_agent_momentum', 'MARL_agent_value',
            'MARL_agent_noise', 'MARL_agent_informed',
            # Agent Consensus (4)
            'MARL_consensus', 'MARL_divergence',
            'MARL_dominant', 'MARL_balance',
            # Market Impact (4)
            'MARL_price_impact', 'MARL_vol_impact',
            'MARL_impact_persist', 'MARL_impact_decay',
            # Equilibrium (3)
            'MARL_nash_price', 'MARL_nash_deviation', 'MARL_convergence'
        ]

    @staticmethod
    def get_citations() -> Dict[str, str]:
        """Get academic citations for MARL trading."""
        return {
            'MARL_Survey': """Sun, S. et al. (2023). "Multi-Agent Reinforcement Learning
                              for Quantitative Trading: A Survey" arXiv:2302.13753.
                              Comprehensive survey of MARL approaches in trading.""",
            'Markov_Games': """Littman, M.L. (1994). "Markov Games as a Framework for
                               Multi-Agent Reinforcement Learning" ICML.
                               Foundation for multi-agent RL in games.""",
            'Nash_Q': """Hu, J. & Wellman, M.P. (2003). "Nash Q-Learning for General-Sum
                         Stochastic Games" JMLR, 4, 1039-1069.
                         Nash equilibrium learning in multi-agent settings.""",
            'Kyle': """Kyle, A.S. (1985). "Continuous Auctions and Insider Trading"
                       Econometrica, 53(6), 1315-1335.
                       Foundation for informed vs noise trader distinction."""
        }


# Convenience function
def generate_marl_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate MARL trading features.

    Args:
        df: OHLCV DataFrame

    Returns:
        DataFrame with 15 MARL features
    """
    generator = MultiAgentRLFeatures()
    return generator.generate_all(df)
