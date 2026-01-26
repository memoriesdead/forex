#!/usr/bin/env python3
"""
GRPO Training Data Preparation for forex-r1 v2
===============================================
Converts forex tick data + ML predictions to reasoning traces with rewards.

Chinese Quant Style (幻方量化, 九坤投资, 明汯投资):
- Uses actual trade outcomes as rewards
- Generates <think>...</think> reasoning traces
- Balanced sampling: correct/incorrect predictions
- Kelly-optimal position sizing in responses

Sources:
- DeepSeek-Math GRPO: https://arxiv.org/abs/2402.03300
- TradingAgents: https://github.com/TauricResearch/TradingAgents
"""

import os
import sys
import json
import random
import hashlib
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
import argparse

import numpy as np
import pandas as pd

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Constants
TRAINING_PACKAGE_DIR = Path(__file__).parent.parent / "training_package"
OUTPUT_DIR = Path(__file__).parent.parent / "vastai_forex_training" / "data"
FORMULA_FILE = Path(__file__).parent.parent / "training_data" / "all_formulas.json"


@dataclass
class GRPOSample:
    """Single GRPO training sample."""
    prompt: str
    completion: str
    reward: float
    symbol: str
    timestamp: Optional[str] = None
    ml_confidence: float = 0.0
    actual_direction: int = 0
    predicted_direction: int = 0


class ReasoningTraceGenerator:
    """Generates <think>...</think> reasoning traces for trading decisions."""

    # Key features to include in analysis
    KEY_FEATURES = [
        'ret_1', 'ret_5', 'ret_10', 'ret_20',
        'vol_20', 'vol_50',
        'momentum_10', 'momentum_20',
        'zscore_20', 'zscore_50',
        'range_position_20', 'range_position_50',
    ]

    # Reasoning templates (Chinese Quant style with formulas)
    REASONING_TEMPLATES = [
        # Template 1: Technical Analysis Focus
        """<think>
Analyzing {symbol} at {price:.5f}:

1. **Returns Analysis**:
   - 1-tick return: {ret_1:.2f} bps
   - 5-tick return: {ret_5:.2f} bps
   - 10-tick return: {ret_10:.2f} bps
   {return_conclusion}

2. **Volatility Assessment**:
   - Short-term vol (20): {vol_20:.4f}
   - Long-term vol (50): {vol_50:.4f}
   - Regime: {regime}

3. **ML Ensemble Signal**:
   - Direction: {direction_word}
   - Confidence: {confidence:.1%}
   - Kelly Criterion: f* = (p*b - q) / b = ({confidence:.2f}*1.5 - {q:.2f}) / 1.5 = {kelly:.3f}

4. **Risk Assessment**:
   - Position Size: {position_pct:.1f}% of capital
   - Max Risk: 2% per trade
   {risk_note}
</think>
{action} with {position_pct:.1f}% position, Kelly fraction {kelly:.2f}.""",

        # Template 2: Momentum Focus
        """<think>
{symbol} Trade Analysis:

**Momentum Signals**:
- Short momentum (10): {momentum_10:.2f} bps
- Medium momentum (20): {momentum_20:.2f} bps
- Trend alignment: {trend_alignment}

**Mean Reversion Check**:
- Z-score (20): {zscore_20:.2f}
- Z-score (50): {zscore_50:.2f}
- Reversion signal: {reversion_signal}

**ML Prediction**:
- {direction_word} with {confidence:.1%} confidence
- Using {kelly:.1%} Kelly sizing (conservative 50% Kelly)

**Formula**: Position = (Win% × PayoffRatio - Loss%) / PayoffRatio
= ({confidence:.2f} × 1.5 - {q:.2f}) / 1.5 = {kelly:.3f}
</think>
{action}. Position: {position_pct:.1f}%.""",

        # Template 3: Risk-First Analysis
        """<think>
Risk-First Analysis for {symbol}:

**Current State**:
- Price: {price:.5f}
- Volatility regime: {regime}
- Range position: {range_position:.1%}

**ML Signal Quality**:
- Confidence: {confidence:.1%} ({confidence_quality})
- Historical accuracy: ~63%
- Edge vs random: {edge:.1%}

**Kelly Position Sizing**:
f* = (2p - 1) / 1 for 1:1 payoff
f* = (2×{confidence:.2f} - 1) = {kelly_simple:.3f}
Using 50% Kelly: {kelly:.3f}

**Risk Limits**:
- Max position: 50% of account
- Stop loss: 2× ATR
- Daily limit: 100 trades
</think>
{action} with {kelly:.1%} Kelly ({position_pct:.1f}% position).""",

        # Template 4: Chinese Quant Style (Simple)
        """<think>
{symbol} | ML信号: {direction_word} | 置信度: {confidence:.1%}

收益分析:
- 短期动量: {ret_5:.2f} bps
- 中期动量: {ret_10:.2f} bps

波动率: {vol_20:.4f} ({regime})
Z分数: {zscore_20:.2f}

凯利公式: f* = {kelly:.3f}
仓位: {position_pct:.1f}%
</think>
{action_cn}: {position_pct:.1f}%仓位""",
    ]

    def __init__(self):
        self.template_idx = 0

    def generate_trace(self, row: pd.Series, symbol: str,
                       predicted_direction: int, ml_confidence: float,
                       actual_direction: int) -> GRPOSample:
        """Generate a reasoning trace for a single tick."""
        # Extract features with defaults
        features = {}
        for feat in self.KEY_FEATURES:
            features[feat] = row.get(feat, 0.0) if feat in row.index else 0.0

        # Get price
        price = row.get('close', row.get('price', row.get('mid', 1.0)))

        # Calculate Kelly sizing
        p = ml_confidence
        q = 1 - p
        kelly_full = (p * 1.5 - q) / 1.5 if ml_confidence > 0.5 else 0
        kelly = max(0, min(0.5, kelly_full * 0.5))  # 50% Kelly, capped
        kelly_simple = 2 * p - 1

        # Position size (Kelly as percentage)
        position_pct = kelly * 100

        # Direction words
        direction_word = "BUY" if predicted_direction == 1 else "SELL"
        action = "BUY" if predicted_direction == 1 else "SELL"
        action_cn = "买入" if predicted_direction == 1 else "卖出"

        # Regime detection
        vol_20 = features.get('vol_20', 0)
        vol_50 = features.get('vol_50', vol_20)
        if vol_50 > 0:
            vol_ratio = vol_20 / vol_50
            if vol_ratio < 0.8:
                regime = "low volatility"
            elif vol_ratio > 1.2:
                regime = "high volatility"
            else:
                regime = "normal"
        else:
            regime = "normal"

        # Trend analysis
        ret_5 = features.get('ret_5', 0)
        ret_10 = features.get('ret_10', 0)
        momentum_10 = features.get('momentum_10', ret_5)
        momentum_20 = features.get('momentum_20', ret_10)

        if momentum_10 > 0 and momentum_20 > 0:
            trend_alignment = "bullish alignment"
        elif momentum_10 < 0 and momentum_20 < 0:
            trend_alignment = "bearish alignment"
        else:
            trend_alignment = "mixed signals"

        # Return conclusion
        if ret_5 > 0 and ret_10 > 0:
            return_conclusion = "Strong positive momentum, favorable for longs"
        elif ret_5 < 0 and ret_10 < 0:
            return_conclusion = "Strong negative momentum, favorable for shorts"
        else:
            return_conclusion = "Mixed momentum, exercise caution"

        # Z-score analysis
        zscore_20 = features.get('zscore_20', 0)
        zscore_50 = features.get('zscore_50', 0)
        if zscore_20 > 2 or zscore_50 > 2:
            reversion_signal = "OVERBOUGHT - expect reversion"
        elif zscore_20 < -2 or zscore_50 < -2:
            reversion_signal = "OVERSOLD - expect reversion"
        else:
            reversion_signal = "neutral range"

        # Confidence quality
        if ml_confidence >= 0.7:
            confidence_quality = "high confidence"
        elif ml_confidence >= 0.6:
            confidence_quality = "moderate confidence"
        else:
            confidence_quality = "low confidence"

        # Range position
        range_position = features.get('range_position_20', 0.5)

        # Edge calculation
        edge = (ml_confidence - 0.5) * 100

        # Risk note
        if ml_confidence < 0.55:
            risk_note = "WARNING: Low confidence signal, reduce size"
        elif vol_ratio > 1.5 if vol_50 > 0 else False:
            risk_note = "WARNING: High volatility regime, tighten stops"
        else:
            risk_note = "Risk within acceptable parameters"

        # Select template (rotate through them)
        template_idx = random.randint(0, len(self.REASONING_TEMPLATES) - 1)
        template = self.REASONING_TEMPLATES[template_idx]

        # Format the reasoning trace
        try:
            completion = template.format(
                symbol=symbol,
                price=price,
                ret_1=features.get('ret_1', 0) * 100,
                ret_5=ret_5 * 100,
                ret_10=ret_10 * 100,
                ret_20=features.get('ret_20', 0) * 100,
                vol_20=vol_20,
                vol_50=vol_50,
                momentum_10=momentum_10 * 100,
                momentum_20=momentum_20 * 100,
                zscore_20=zscore_20,
                zscore_50=zscore_50,
                range_position=range_position,
                regime=regime,
                trend_alignment=trend_alignment,
                return_conclusion=return_conclusion,
                reversion_signal=reversion_signal,
                direction_word=direction_word,
                confidence=ml_confidence,
                confidence_quality=confidence_quality,
                q=q,
                kelly=kelly,
                kelly_simple=kelly_simple,
                kelly_full=kelly_full,
                position_pct=position_pct,
                action=action,
                action_cn=action_cn,
                edge=edge,
                risk_note=risk_note,
            )
        except KeyError as e:
            # Fallback to simple template
            completion = f"""<think>
{symbol} analysis: ML predicts {direction_word} with {ml_confidence:.1%} confidence.
Kelly sizing: {kelly:.3f}
</think>
{action} with {position_pct:.1f}% position."""

        # Create prompt
        feature_str = ", ".join([f"{k}={features.get(k, 0):.4f}" for k in ['ret_5', 'ret_10', 'vol_20', 'zscore_20'] if k in features])
        prompt = f"{symbol}: price={price:.5f}, ML_confidence={ml_confidence:.2%}, direction={direction_word}. Key features: {feature_str}. Analyze and recommend action."

        # Calculate reward based on actual outcome
        if predicted_direction == actual_direction:
            # Correct prediction
            base_reward = 1.0
            # Bonus for high confidence correct predictions
            confidence_bonus = 0.5 * (ml_confidence - 0.5) if ml_confidence > 0.5 else 0
            reward = base_reward + confidence_bonus
        else:
            # Wrong prediction
            base_reward = -0.5
            # Penalty for high confidence wrong predictions
            confidence_penalty = -0.5 * (ml_confidence - 0.5) if ml_confidence > 0.5 else 0
            reward = base_reward + confidence_penalty

        # Normalize reward to [-1, 1]
        reward = max(-1.0, min(1.0, reward))

        return GRPOSample(
            prompt=prompt,
            completion=completion,
            reward=reward,
            symbol=symbol,
            ml_confidence=ml_confidence,
            actual_direction=actual_direction,
            predicted_direction=predicted_direction,
        )


def process_symbol(symbol: str, max_samples: int = 20000) -> List[Dict]:
    """Process a single symbol's data into GRPO samples."""
    logger.info(f"Processing {symbol}...")

    train_file = TRAINING_PACKAGE_DIR / symbol / "train.parquet"
    if not train_file.exists():
        logger.warning(f"No training data for {symbol}")
        return []

    # Load data
    df = pd.read_parquet(train_file)
    logger.info(f"  Loaded {len(df)} samples for {symbol}")

    # Need target columns for direction
    direction_col = None
    for col in ['target_direction_10', 'target_direction_5', 'target_direction_1']:
        if col in df.columns:
            direction_col = col
            break

    if direction_col is None:
        logger.warning(f"  No direction target found for {symbol}")
        return []

    # Simulate ML predictions (using actual targets + noise for training data)
    # In production, these would be actual ML model predictions
    df['predicted_direction'] = df[direction_col].copy()

    # Add some noise to simulate imperfect predictions (~63% accuracy)
    np.random.seed(42)
    noise_mask = np.random.random(len(df)) < 0.37  # 37% error rate
    df.loc[noise_mask, 'predicted_direction'] = -df.loc[noise_mask, 'predicted_direction']

    # ML confidence (higher for correct predictions, simulate realistic distribution)
    base_confidence = np.random.beta(5, 3, len(df))  # Skewed towards higher values
    correct_mask = df['predicted_direction'] == df[direction_col]
    df['ml_confidence'] = 0.5 + base_confidence * 0.3
    df.loc[correct_mask, 'ml_confidence'] += 0.05  # Slightly higher for correct
    df['ml_confidence'] = df['ml_confidence'].clip(0.5, 0.95)

    # Sample strategy: balance correct and incorrect predictions
    correct_df = df[df['predicted_direction'] == df[direction_col]]
    incorrect_df = df[df['predicted_direction'] != df[direction_col]]

    n_correct = min(len(correct_df), int(max_samples * 0.65))  # 65% correct
    n_incorrect = min(len(incorrect_df), int(max_samples * 0.35))  # 35% incorrect

    sampled_correct = correct_df.sample(n=n_correct, random_state=42) if n_correct > 0 else pd.DataFrame()
    sampled_incorrect = incorrect_df.sample(n=n_incorrect, random_state=42) if n_incorrect > 0 else pd.DataFrame()

    sampled_df = pd.concat([sampled_correct, sampled_incorrect]).sample(frac=1, random_state=42)

    # Generate reasoning traces
    generator = ReasoningTraceGenerator()
    samples = []

    for idx, row in sampled_df.iterrows():
        try:
            sample = generator.generate_trace(
                row=row,
                symbol=symbol,
                predicted_direction=int(row['predicted_direction']),
                ml_confidence=float(row['ml_confidence']),
                actual_direction=int(row[direction_col]),
            )
            samples.append({
                'prompt': sample.prompt,
                'completion': sample.completion,
                'reward': sample.reward,
                'symbol': sample.symbol,
                'ml_confidence': sample.ml_confidence,
            })
        except Exception as e:
            logger.debug(f"  Error generating sample: {e}")
            continue

    logger.info(f"  Generated {len(samples)} samples for {symbol}")
    return samples


def load_formula_samples() -> List[Dict]:
    """Load existing formula samples for SFT base."""
    samples = []

    if FORMULA_FILE.exists():
        with open(FORMULA_FILE, encoding='utf-8') as f:
            formulas = json.load(f)

        for formula in formulas:
            # Convert to GRPO format
            prompt = formula.get('question', formula.get('instruction', ''))
            completion = formula.get('answer', formula.get('output', ''))

            if prompt and completion:
                samples.append({
                    'prompt': prompt,
                    'completion': f"<think>\n{completion}\n</think>",
                    'reward': 1.0,  # Formulas are always correct
                    'symbol': 'FORMULA',
                    'ml_confidence': 1.0,
                })

    # Also check for other training data files
    training_data_dir = Path(__file__).parent.parent / "training_data"
    for jsonl_file in training_data_dir.glob("*.jsonl"):
        with open(jsonl_file, encoding='utf-8') as f:
            for line in f:
                try:
                    d = json.loads(line.strip())
                    prompt = d.get('instruction', d.get('prompt', ''))
                    output = d.get('output', d.get('completion', ''))
                    if prompt and output:
                        samples.append({
                            'prompt': prompt,
                            'completion': output,
                            'reward': 1.0,
                            'symbol': 'KNOWLEDGE',
                            'ml_confidence': 1.0,
                        })
                except:
                    continue

    logger.info(f"Loaded {len(samples)} formula/knowledge samples")
    return samples


def main():
    parser = argparse.ArgumentParser(description='Prepare GRPO training data')
    parser.add_argument('--max-samples-per-symbol', type=int, default=15000,
                        help='Max samples per symbol (default: 15000)')
    parser.add_argument('--symbols', type=str, default=None,
                        help='Comma-separated list of symbols (default: all)')
    parser.add_argument('--output-dir', type=str, default=str(OUTPUT_DIR),
                        help='Output directory')
    parser.add_argument('--workers', type=int, default=4,
                        help='Number of parallel workers')
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Get list of symbols
    if args.symbols:
        symbols = [s.strip() for s in args.symbols.split(',')]
    else:
        symbols = [d.name for d in TRAINING_PACKAGE_DIR.iterdir()
                   if d.is_dir() and (d / "train.parquet").exists()]

    logger.info(f"Processing {len(symbols)} symbols...")

    # Process symbols in parallel
    all_samples = []

    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(process_symbol, symbol, args.max_samples_per_symbol): symbol
            for symbol in symbols
        }

        for future in as_completed(futures):
            symbol = futures[future]
            try:
                samples = future.result()
                all_samples.extend(samples)
            except Exception as e:
                logger.error(f"Error processing {symbol}: {e}")

    logger.info(f"Total trade samples: {len(all_samples)}")

    # Load formula samples
    formula_samples = load_formula_samples()

    # Combine all samples
    all_data = formula_samples + all_samples
    random.shuffle(all_data)

    logger.info(f"Total combined samples: {len(all_data)}")

    # Split into SFT and GRPO datasets
    # SFT: High reward samples (correct predictions + formulas)
    sft_samples = [s for s in all_data if s['reward'] > 0.5]
    # GRPO: All samples with rewards
    grpo_samples = all_data

    # Save SFT dataset
    sft_file = output_dir / "sft_training.jsonl"
    with open(sft_file, 'w', encoding='utf-8') as f:
        for sample in sft_samples:
            f.write(json.dumps({
                'instruction': sample['prompt'],
                'output': sample['completion'],
            }, ensure_ascii=False) + '\n')
    logger.info(f"Saved {len(sft_samples)} SFT samples to {sft_file}")

    # Save GRPO dataset
    grpo_file = output_dir / "grpo_training.jsonl"
    with open(grpo_file, 'w', encoding='utf-8') as f:
        for sample in grpo_samples:
            f.write(json.dumps({
                'prompt': sample['prompt'],
                'completion': sample['completion'],
                'reward': sample['reward'],
            }, ensure_ascii=False) + '\n')
    logger.info(f"Saved {len(grpo_samples)} GRPO samples to {grpo_file}")

    # Save summary statistics
    stats = {
        'total_samples': len(all_data),
        'sft_samples': len(sft_samples),
        'grpo_samples': len(grpo_samples),
        'formula_samples': len(formula_samples),
        'trade_samples': len(all_samples),
        'symbols_processed': len(symbols),
        'avg_reward': np.mean([s['reward'] for s in all_data]),
        'positive_reward_pct': len([s for s in all_data if s['reward'] > 0]) / len(all_data) * 100,
    }

    stats_file = output_dir / "training_stats.json"
    with open(stats_file, 'w') as f:
        json.dump(stats, f, indent=2)

    logger.info(f"\n{'='*60}")
    logger.info("GRPO Training Data Preparation Complete")
    logger.info(f"{'='*60}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Total samples: {stats['total_samples']:,}")
    logger.info(f"SFT samples: {stats['sft_samples']:,}")
    logger.info(f"GRPO samples: {stats['grpo_samples']:,}")
    logger.info(f"Avg reward: {stats['avg_reward']:.3f}")
    logger.info(f"Positive reward %: {stats['positive_reward_pct']:.1f}%")
    logger.info(f"{'='*60}")

    # Print files ready for upload
    print(f"\nFiles ready for Vast.ai upload:")
    for f in output_dir.glob("*"):
        size_mb = f.stat().st_size / (1024 * 1024)
        print(f"  {f.name}: {size_mb:.1f} MB")


if __name__ == '__main__':
    main()
