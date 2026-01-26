"""
Institutional-Grade ML Predictor
================================
Uses ACTUAL methods from billion-dollar quants:
- Hidden Markov Models (Renaissance Technologies) - Regime detection
- Kalman Filters (Goldman Sachs) - Dynamic parameter estimation
- XGBoost/LightGBM/CatBoost ensemble (What actually wins)
- WorldQuant Alpha101 (101 Formulaic Alphas)
- Avellaneda-Stoikov (HFT Market Making)
- Kelly Criterion (Optimal Position Sizing)
- Triple Barrier Method (Lopez de Prado)

NOT using (overfits/doesn't work):
- Deep Learning / Neural Networks
- Transformers
- Deep RL
"""

import pickle
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Tuple, Optional, List

# Import gold standard quant formulas
try:
    from core.quant_formulas import (
        Alpha101Forex,
        AvellanedaStoikov,
        KellyCriterion,
        TripleBarrier,
        FractionalDifferentiation,
        AlmgrenChriss
    )
    QUANT_FORMULAS_AVAILABLE = True
except ImportError:
    QUANT_FORMULAS_AVAILABLE = False

logger = logging.getLogger(__name__)


class InstitutionalPredictor:
    """
    Institutional-grade predictor using proven quant methods.

    This class implements what billion-dollar quants ACTUALLY use:
    - HMM for regime detection (Renaissance)
    - Kalman filter for dynamic hedging (Goldman)
    - Gradient boosting ensemble (Industry standard)
    - WorldQuant Alpha101 (101 weak signals)
    - Avellaneda-Stoikov (Market making quotes)
    - Kelly Criterion (Position sizing)
    """

    def __init__(self, models_dir: Path):
        self.models_dir = models_dir

        # Institutional models
        self.hmm_models = {}      # Hidden Markov Models (regime detection)
        self.kalman_models = {}   # Kalman filters (dynamic parameters)
        self.ensemble_models = {} # XGBoost/LightGBM/CatBoost
        self.scalers = {}
        self.feature_cols = []

        # Current state
        self.current_regime = {}  # Per-pair regime state
        self.kalman_state = {}    # Per-pair Kalman state

        # Gold standard quant formulas
        if QUANT_FORMULAS_AVAILABLE:
            self.alpha101 = Alpha101Forex()
            self.avellaneda_stoikov = AvellanedaStoikov(gamma=0.1, sigma=0.02, k=1.5)
            self.kelly = KellyCriterion()
            self.triple_barrier = TripleBarrier()
            self.frac_diff = FractionalDifferentiation()
        else:
            self.alpha101 = None
            self.avellaneda_stoikov = None
            self.kelly = None
            self.triple_barrier = None
            self.frac_diff = None

        self.loaded = False

    def load_models(self) -> bool:
        """Load all institutional-grade models."""
        try:
            # Load HMM models (Renaissance method)
            hmm_path = self.models_dir / 'hmm_models.pkl'
            if hmm_path.exists():
                with open(hmm_path, 'rb') as f:
                    self.hmm_models = pickle.load(f)
                logger.info(f"Loaded HMM models for {len(self.hmm_models)} pairs")

            # Load Kalman models (Goldman method)
            kalman_path = self.models_dir / 'kalman_models.pkl'
            if kalman_path.exists():
                with open(kalman_path, 'rb') as f:
                    self.kalman_models = pickle.load(f)
                logger.info(f"Loaded Kalman models for {len(self.kalman_models)} pairs")

            # Load ensemble models (XGBoost/LightGBM/CatBoost)
            ensemble_path = self.models_dir / 'ensemble_models.pkl'
            if ensemble_path.exists():
                with open(ensemble_path, 'rb') as f:
                    self.ensemble_models = pickle.load(f)
                logger.info(f"Loaded ensemble models for {len(self.ensemble_models)} pairs")

            # Load scalers
            scalers_path = self.models_dir / 'scalers.pkl'
            if scalers_path.exists():
                with open(scalers_path, 'rb') as f:
                    self.scalers = pickle.load(f)
                logger.info(f"Loaded scalers for {len(self.scalers)} pairs")

            # Load feature columns
            features_path = self.models_dir / 'features.pkl'
            if features_path.exists():
                with open(features_path, 'rb') as f:
                    self.feature_cols = pickle.load(f)
                logger.info(f"Loaded {len(self.feature_cols)} feature columns")

            self.loaded = bool(self.ensemble_models)

            if not self.loaded:
                logger.warning("No ensemble models loaded - using fallback signals")

            return self.loaded

        except Exception as e:
            logger.error(f"Error loading models: {e}")
            return False

    def detect_regime(self, pair: str, returns: np.ndarray, volatility: np.ndarray) -> Dict:
        """
        Detect market regime using Hidden Markov Model.

        This is the Renaissance Technologies method:
        - Uses HMM to identify hidden market states
        - States typically represent: low vol, normal, high vol regimes
        - Adjust position sizing based on regime

        Returns:
            Dict with regime info: state (0,1,2), probabilities, recommended action
        """
        if pair not in self.hmm_models:
            return {'state': 1, 'confidence': 0.5, 'action': 'normal'}

        try:
            hmm = self.hmm_models[pair]

            # Prepare observation data
            X = np.column_stack([returns[-50:], volatility[-50:]])
            X = np.nan_to_num(X, nan=0.0)

            if len(X) < 10:
                return {'state': 1, 'confidence': 0.5, 'action': 'normal'}

            # Get current regime
            regime = hmm.predict(X)[-1]
            regime_probs = hmm.predict_proba(X)[-1]
            confidence = regime_probs[regime]

            # Store current regime
            self.current_regime[pair] = regime

            # Determine action based on regime
            # Regime 0: Low volatility - normal trading
            # Regime 1: Normal - full position size
            # Regime 2: High volatility - reduce position or wait
            actions = {
                0: 'aggressive',  # Low vol - can be more aggressive
                1: 'normal',      # Normal conditions
                2: 'conservative' # High vol - reduce risk
            }

            return {
                'state': int(regime),
                'confidence': float(confidence),
                'probabilities': regime_probs.tolist(),
                'action': actions.get(regime, 'normal')
            }

        except Exception as e:
            logger.warning(f"HMM regime detection failed for {pair}: {e}")
            return {'state': 1, 'confidence': 0.5, 'action': 'normal'}

    def get_kalman_estimate(self, pair: str, price: float) -> Dict:
        """
        Get Kalman filter estimate for trend/mean.

        This is the Goldman Sachs method for market making:
        - Kalman filter estimates true underlying value
        - Deviation from estimate suggests mean reversion opportunity
        - Updates dynamically with each new observation

        Returns:
            Dict with: estimated_mean, deviation, signal strength
        """
        if pair not in self.kalman_models:
            return {'mean': price, 'deviation': 0.0, 'signal': 0.0}

        try:
            kf = self.kalman_models[pair]

            # Initialize state if needed
            if pair not in self.kalman_state:
                self.kalman_state[pair] = {
                    'state_mean': np.array([price]),
                    'state_cov': np.array([[1.0]])
                }

            # Update with new observation
            state = self.kalman_state[pair]
            new_mean, new_cov = kf.filter_update(
                state['state_mean'],
                state['state_cov'],
                observation=price
            )

            # Store updated state
            self.kalman_state[pair] = {
                'state_mean': new_mean,
                'state_cov': new_cov
            }

            # Calculate deviation
            estimated_mean = float(new_mean[0])
            deviation = (price - estimated_mean) / estimated_mean if estimated_mean != 0 else 0

            # Signal: negative deviation = price below mean = potential long
            # Positive deviation = price above mean = potential short
            signal = -deviation * 100  # Scale for usability

            return {
                'mean': estimated_mean,
                'deviation': deviation,
                'signal': signal,
                'uncertainty': float(new_cov[0, 0])
            }

        except Exception as e:
            logger.warning(f"Kalman filter failed for {pair}: {e}")
            return {'mean': price, 'deviation': 0.0, 'signal': 0.0}

    def get_ensemble_prediction(self, pair: str, features: np.ndarray) -> Tuple[str, float]:
        """
        Get ensemble prediction from XGBoost/LightGBM/CatBoost.

        This is what actually wins in production:
        - Gradient boosting trees (NOT neural networks)
        - Ensemble of 3 models with weighted voting
        - Simple, interpretable, generalizes well

        Returns:
            Tuple of (signal: 'long'/'short'/'neutral', confidence: 0-1)
        """
        if pair not in self.ensemble_models:
            return 'neutral', 0.5

        try:
            model_data = self.ensemble_models[pair]
            models = model_data['models']
            weights = model_data['weights']

            # Scale features if scaler available
            if pair in self.scalers:
                features = self.scalers[pair].transform(features.reshape(1, -1))
            else:
                features = features.reshape(1, -1)

            # Get predictions from each model
            ensemble_prob = 0.0
            total_weight = sum(weights.values())

            predictions = {}
            for name, model in models.items():
                try:
                    pred_proba = model.predict_proba(features)[0]
                    prob_up = pred_proba[1] if len(pred_proba) > 1 else pred_proba[0]
                    predictions[name] = prob_up
                    ensemble_prob += prob_up * (weights[name] / total_weight)
                except Exception as e:
                    logger.warning(f"Model {name} failed for {pair}: {e}")

            # Determine signal
            # "We're right 50.75% of the time" - Renaissance
            if ensemble_prob > 0.52:  # Slight edge is enough
                signal = 'long'
                confidence = ensemble_prob
            elif ensemble_prob < 0.48:
                signal = 'short'
                confidence = 1 - ensemble_prob
            else:
                signal = 'neutral'
                confidence = 0.5

            return signal, confidence

        except Exception as e:
            logger.error(f"Ensemble prediction failed for {pair}: {e}")
            return 'neutral', 0.5

    def predict(self, pair: str, data: pd.DataFrame) -> Dict:
        """
        Full institutional-grade prediction combining all methods.

        Combines:
        1. HMM regime detection (position sizing adjustment)
        2. Kalman filter (mean reversion signal)
        3. Ensemble prediction (direction and confidence)

        Returns:
            Dict with signal, confidence, regime, and recommendations
        """
        if not self.loaded:
            return {
                'signal': 'neutral',
                'confidence': 0.0,
                'regime': 'unknown',
                'position_multiplier': 1.0,
                'notes': 'Models not loaded'
            }

        try:
            # Prepare data
            returns = data['returns'].values if 'returns' in data.columns else np.diff(data['mid'].values) / data['mid'].values[:-1]
            volatility = data['vol_20'].values if 'vol_20' in data.columns else pd.Series(returns).rolling(20).std().values
            current_price = data['mid'].iloc[-1] if 'mid' in data.columns else data.iloc[-1, 0]

            # 1. HMM Regime Detection (Renaissance method)
            regime_info = self.detect_regime(pair, returns, volatility)

            # 2. Kalman Filter (Goldman method)
            kalman_info = self.get_kalman_estimate(pair, current_price)

            # 3. Ensemble Prediction (Industry standard)
            features = self._extract_features(data)
            ensemble_signal, ensemble_confidence = self.get_ensemble_prediction(pair, features)

            # Combine signals
            # Adjust position size based on regime
            position_multipliers = {
                'aggressive': 1.5,
                'normal': 1.0,
                'conservative': 0.5
            }
            position_mult = position_multipliers.get(regime_info['action'], 1.0)

            # Combine Kalman and ensemble signals
            kalman_signal = 'long' if kalman_info['signal'] > 1 else ('short' if kalman_info['signal'] < -1 else 'neutral')

            # Final signal: require agreement or strong ensemble confidence
            if ensemble_signal == kalman_signal:
                final_signal = ensemble_signal
                final_confidence = (ensemble_confidence + abs(kalman_info['signal']) / 10) / 2
            elif ensemble_confidence > 0.58:  # Strong ensemble signal
                final_signal = ensemble_signal
                final_confidence = ensemble_confidence * 0.8  # Slight penalty for disagreement
            else:
                final_signal = 'neutral'
                final_confidence = 0.5

            # Reduce confidence in high volatility regime
            if regime_info['action'] == 'conservative':
                final_confidence *= 0.8

            return {
                'signal': final_signal,
                'confidence': min(final_confidence, 0.95),  # Cap confidence
                'regime': {
                    'state': regime_info['state'],
                    'action': regime_info['action'],
                    'confidence': regime_info['confidence']
                },
                'kalman': {
                    'mean': kalman_info['mean'],
                    'deviation': kalman_info['deviation'],
                    'signal': kalman_info['signal']
                },
                'ensemble': {
                    'signal': ensemble_signal,
                    'confidence': ensemble_confidence
                },
                'position_multiplier': position_mult,
                'notes': f"Regime: {regime_info['action']}, Kalman dev: {kalman_info['deviation']:.4f}"
            }

        except Exception as e:
            logger.error(f"Prediction failed for {pair}: {e}")
            return {
                'signal': 'neutral',
                'confidence': 0.0,
                'regime': 'error',
                'position_multiplier': 0.5,
                'notes': str(e)
            }

    def get_alpha101_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Generate WorldQuant Alpha101 signals.

        These are 101 formulaic alphas from WorldQuant research.
        Each alpha is a weak signal (51-55% accuracy) that contributes
        to ensemble predictions.

        Args:
            data: DataFrame with OHLCV or forex tick data

        Returns:
            DataFrame with alpha signals added
        """
        if self.alpha101 is None:
            logger.warning("Alpha101 not available")
            return data

        try:
            return self.alpha101.generate_all_alphas(data)
        except Exception as e:
            logger.warning(f"Alpha101 generation failed: {e}")
            return data

    def get_market_making_quotes(self, mid_price: float, inventory: int,
                                  time_remaining: float = 0.5) -> Dict:
        """
        Get Avellaneda-Stoikov optimal market making quotes.

        This is the gold standard for HFT market making.

        Args:
            mid_price: Current mid price
            inventory: Current position (positive = long)
            time_remaining: Fraction of trading period remaining

        Returns:
            Dict with bid, ask, and spread info
        """
        if self.avellaneda_stoikov is None:
            return {'bid': mid_price * 0.9999, 'ask': mid_price * 1.0001}

        bid, ask = self.avellaneda_stoikov.optimal_quotes(
            mid_price, inventory, time_remaining
        )
        reservation = self.avellaneda_stoikov.reservation_price(
            mid_price, inventory, time_remaining
        )

        return {
            'bid': bid,
            'ask': ask,
            'spread': ask - bid,
            'spread_pips': (ask - bid) * 10000,
            'reservation_price': reservation,
            'inventory_skew': self.avellaneda_stoikov.inventory_skew(inventory)
        }

    def calculate_position_size(self, account_value: float, win_prob: float,
                                 win_loss_ratio: float,
                                 kelly_fraction: float = 0.25) -> Dict:
        """
        Calculate optimal position size using Kelly Criterion.

        Uses fractional Kelly (default 25%) for safety:
        - 50% Kelly = 75% optimal growth, 25% variance
        - 25% Kelly = 43.75% optimal growth, 6.25% variance

        Args:
            account_value: Total account value
            win_prob: Probability of winning (from model confidence)
            win_loss_ratio: Average win / Average loss
            kelly_fraction: Fraction of full Kelly to use (0.25-0.5)

        Returns:
            Dict with position sizing info
        """
        if self.kelly is None:
            # Fallback to simple 1% risk
            return {
                'position_size': account_value * 0.01,
                'kelly_pct': 0.01,
                'method': 'fallback_1pct'
            }

        full_kelly = self.kelly.kelly_fraction(win_prob, win_loss_ratio)
        frac_kelly = self.kelly.fractional_kelly(win_prob, win_loss_ratio, kelly_fraction)
        position = self.kelly.position_size(
            account_value, win_prob, win_loss_ratio,
            fraction=kelly_fraction, max_position_pct=0.10  # Max 10% per trade
        )

        return {
            'position_size': position,
            'kelly_pct': frac_kelly,
            'full_kelly_pct': full_kelly,
            'fraction_used': kelly_fraction,
            'method': f'{int(kelly_fraction*100)}%_kelly'
        }

    def _extract_features(self, data: pd.DataFrame) -> np.ndarray:
        """Extract features for ensemble prediction."""
        # First, generate Alpha101 signals if available
        if self.alpha101 is not None:
            try:
                data = self.get_alpha101_signals(data)
            except Exception as e:
                logger.debug(f"Alpha101 feature extraction failed: {e}")

        if not self.feature_cols:
            # Use all numeric columns except excluded ones
            exclude = ['bid', 'ask', 'mid', 'spread', 'pair', 'timestamp', 'returns', 'target']
            cols = [c for c in data.columns if c not in exclude and data[c].dtype in [np.float64, np.int64, np.float32, np.int32]]
        else:
            cols = [c for c in self.feature_cols if c in data.columns]

        if not cols:
            return np.zeros(50)

        return data[cols].iloc[-1].values


class FallbackPredictor:
    """
    Simple fallback predictor when institutional models aren't available.
    Uses basic technical signals (still better than deep learning for forex).
    """

    def predict(self, pair: str, data: pd.DataFrame) -> Dict:
        """Generate prediction from simple signals."""
        try:
            mid = data['mid'].values if 'mid' in data.columns else data.iloc[:, 0].values

            # Simple moving average crossover
            sma_fast = pd.Series(mid).rolling(10).mean().iloc[-1]
            sma_slow = pd.Series(mid).rolling(50).mean().iloc[-1]

            # RSI
            returns = np.diff(mid) / mid[:-1]
            gains = np.where(returns > 0, returns, 0)
            losses = np.where(returns < 0, -returns, 0)
            avg_gain = np.mean(gains[-14:])
            avg_loss = np.mean(losses[-14:])
            rs = avg_gain / (avg_loss + 1e-10)
            rsi = 100 - (100 / (1 + rs))

            # Signal logic
            if sma_fast > sma_slow and rsi < 70:
                signal = 'long'
                confidence = 0.52 + (sma_fast - sma_slow) / sma_slow * 10
            elif sma_fast < sma_slow and rsi > 30:
                signal = 'short'
                confidence = 0.52 + (sma_slow - sma_fast) / sma_slow * 10
            else:
                signal = 'neutral'
                confidence = 0.5

            return {
                'signal': signal,
                'confidence': min(max(confidence, 0.5), 0.6),  # Cap at 60% for fallback
                'regime': {'state': 1, 'action': 'normal', 'confidence': 0.5},
                'position_multiplier': 1.0,
                'notes': 'Fallback predictor (simple signals)'
            }

        except Exception as e:
            return {
                'signal': 'neutral',
                'confidence': 0.0,
                'regime': {'state': 1, 'action': 'normal', 'confidence': 0.5},
                'position_multiplier': 0.5,
                'notes': f'Error: {e}'
            }


def create_predictor(models_dir: Path) -> InstitutionalPredictor:
    """Factory function to create the appropriate predictor."""
    predictor = InstitutionalPredictor(models_dir)

    if predictor.load_models():
        logger.info("Using institutional-grade predictor (HMM + Kalman + Ensemble)")
        return predictor
    else:
        logger.warning("Falling back to simple predictor")
        return FallbackPredictor()
