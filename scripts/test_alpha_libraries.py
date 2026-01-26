#!/usr/bin/env python3
"""
Test All Alpha Libraries
========================
Verifies all Chinese quant alpha libraries are working correctly.
"""

import sys
sys.path.insert(0, '.')

import numpy as np
import pandas as pd
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')


def create_test_data(n: int = 1000) -> pd.DataFrame:
    """Create synthetic OHLCV data for testing."""
    np.random.seed(42)

    # Generate random walk price series
    returns = np.random.randn(n) * 0.001 + 0.00001
    close = 1.0000 * np.cumprod(1 + returns)

    # Generate OHLCV
    df = pd.DataFrame({
        'open': close * (1 + np.random.randn(n) * 0.0001),
        'high': close * (1 + np.abs(np.random.randn(n) * 0.0003)),
        'low': close * (1 - np.abs(np.random.randn(n) * 0.0003)),
        'close': close,
        'volume': np.random.randint(100, 10000, n).astype(float)
    })

    # Ensure high >= close >= low
    df['high'] = df[['open', 'high', 'close']].max(axis=1)
    df['low'] = df[['open', 'low', 'close']].min(axis=1)

    return df


def test_alpha101():
    """Test Alpha101 (62 factors)."""
    print("\n" + "="*60)
    print("Testing Alpha101 (WorldQuant 62 factors)")
    print("="*60)

    try:
        from core.features.alpha101 import Alpha101Complete

        df = create_test_data(500)
        df['returns'] = df['close'].pct_change()
        df['vwap'] = (df['high'] + df['low'] + df['close']) / 3

        alpha101 = Alpha101Complete()
        result = alpha101.generate_all_alphas(df)

        alpha_cols = [c for c in result.columns if c.startswith('alpha')]
        print(f"  Features generated: {len(alpha_cols)}")
        print(f"  Sample features: {alpha_cols[:5]}")
        print(f"  NaN count: {result[alpha_cols].isna().sum().sum()}")
        print("  STATUS: PASS")
        return True, len(alpha_cols)
    except Exception as e:
        print(f"  ERROR: {e}")
        print("  STATUS: FAIL")
        return False, 0


def test_alpha158():
    """Test Alpha158 (158 factors)."""
    print("\n" + "="*60)
    print("Testing Alpha158 (Microsoft Qlib 158 factors)")
    print("="*60)

    try:
        from core.features.alpha158 import Alpha158, generate_alpha158

        df = create_test_data(500)

        alpha158 = Alpha158()
        result = alpha158.generate_all(df)

        print(f"  Features generated: {len(result.columns)}")
        print(f"  Sample features: {list(result.columns[:10])}")
        print(f"  NaN count: {result.isna().sum().sum()}")
        print("  STATUS: PASS")
        return True, len(result.columns)
    except Exception as e:
        print(f"  ERROR: {e}")
        import traceback
        traceback.print_exc()
        print("  STATUS: FAIL")
        return False, 0


def test_alpha360():
    """Test Alpha360 (360 factors)."""
    print("\n" + "="*60)
    print("Testing Alpha360 (Microsoft Qlib 360 factors)")
    print("="*60)

    try:
        from core.features.alpha360 import Alpha360, Alpha360Compact

        df = create_test_data(500)

        # Test full Alpha360
        alpha360 = Alpha360()
        result = alpha360.generate_all(df)

        print(f"  Full Alpha360 features: {len(result.columns)}")
        print(f"  Sample features: {list(result.columns[:6])}")

        # Test compact version
        compact = Alpha360Compact(lookback=20)
        result_compact = compact.generate_all(df)

        print(f"  Compact Alpha360 features: {len(result_compact.columns)}")
        print(f"  NaN count: {result.isna().sum().sum()}")
        print("  STATUS: PASS")
        return True, len(result.columns)
    except Exception as e:
        print(f"  ERROR: {e}")
        import traceback
        traceback.print_exc()
        print("  STATUS: FAIL")
        return False, 0


def test_barra_cne6():
    """Test Barra CNE6 (46 factors)."""
    print("\n" + "="*60)
    print("Testing Barra CNE6 (46 risk factors)")
    print("="*60)

    try:
        from core.features.barra_cne6 import BarraCNE6Forex, generate_barra_cne6

        df = create_test_data(500)

        barra = BarraCNE6Forex()
        result = barra.generate_all(df)

        print(f"  Features generated: {len(result.columns)}")
        print(f"  Sample features: {list(result.columns[:10])}")

        # Show categories
        categories = {}
        for col in result.columns:
            cat = barra.get_factor_category(col)
            categories[cat] = categories.get(cat, 0) + 1
        print(f"  Categories: {categories}")
        print(f"  NaN count: {result.isna().sum().sum()}")
        print("  STATUS: PASS")
        return True, len(result.columns)
    except Exception as e:
        print(f"  ERROR: {e}")
        import traceback
        traceback.print_exc()
        print("  STATUS: FAIL")
        return False, 0


def test_renaissance():
    """Test Renaissance signals (51 factors)."""
    print("\n" + "="*60)
    print("Testing Renaissance (51 weak signals)")
    print("="*60)

    try:
        from core.features.renaissance import RenaissanceSignalGenerator

        df = create_test_data(500)
        df['returns'] = df['close'].pct_change()
        df['price'] = df['close']  # Renaissance needs 'price' column

        renaissance = RenaissanceSignalGenerator()
        result = renaissance.generate_all_signals(df)

        signal_cols = [c for c in result.columns if c.startswith('signal_')]
        print(f"  Features generated: {len(signal_cols)}")
        print(f"  Sample features: {signal_cols[:5]}")
        print(f"  NaN count: {result[signal_cols].isna().sum().sum()}")
        print("  STATUS: PASS")
        return True, len(signal_cols)
    except Exception as e:
        print(f"  ERROR: {e}")
        print("  STATUS: FAIL")
        return False, 0


def test_us_academic():
    """Test US Academic factors (50 factors)."""
    print("\n" + "="*60)
    print("Testing US Academic Factors (50 peer-reviewed)")
    print("="*60)

    try:
        from core.features.us_academic_factors import USAcademicFactors, generate_us_academic_factors

        df = create_test_data(500)

        factors = USAcademicFactors()
        result = factors.generate_all(df)

        print(f"  Features generated: {len(result.columns)}")
        print(f"  Sample features: {list(result.columns[:10])}")

        # Show categories
        categories = {}
        for col in result.columns:
            cat = factors.get_factor_category(col)
            categories[cat] = categories.get(cat, 0) + 1
        print(f"  Categories: {categories}")
        print(f"  NaN count: {result.isna().sum().sum()}")
        print("  STATUS: PASS")
        return True, len(result.columns)
    except Exception as e:
        print(f"  ERROR: {e}")
        import traceback
        traceback.print_exc()
        print("  STATUS: FAIL")
        return False, 0


def test_mlfinlab():
    """Test MLFinLab features (Lopez de Prado)."""
    print("\n" + "="*60)
    print("Testing MLFinLab (Lopez de Prado - 17 factors)")
    print("="*60)

    try:
        from core.features.mlfinlab_features import MLFinLabFeatures, generate_mlfinlab_features

        df = create_test_data(500)

        generator = MLFinLabFeatures()
        result = generator.generate_all(df)

        print(f"  Features generated: {len(result.columns)}")
        print(f"  Sample features: {list(result.columns[:10])}")
        print(f"  NaN count: {result.isna().sum().sum()}")
        print("  STATUS: PASS")
        return True, len(result.columns)
    except Exception as e:
        print(f"  ERROR: {e}")
        import traceback
        traceback.print_exc()
        print("  STATUS: FAIL")
        return False, 0


def test_timeseries():
    """Test Time-Series-Library features."""
    print("\n" + "="*60)
    print("Testing Time-Series-Library (Tsinghua ML - 45 factors)")
    print("="*60)

    try:
        from core.features.timeseries_features import TimeSeriesLibraryFeatures, generate_timeseries_features

        df = create_test_data(500)

        generator = TimeSeriesLibraryFeatures()
        result = generator.generate_all(df)

        print(f"  Features generated: {len(result.columns)}")
        print(f"  Sample features: {list(result.columns[:10])}")
        print(f"  NaN count: {result.isna().sum().sum()}")
        print("  STATUS: PASS")
        return True, len(result.columns)
    except Exception as e:
        print(f"  ERROR: {e}")
        import traceback
        traceback.print_exc()
        print("  STATUS: FAIL")
        return False, 0


def test_with_real_data():
    """Test with real training data if available."""
    print("\n" + "="*60)
    print("Testing with REAL training data (EURUSD)")
    print("="*60)

    train_path = Path('training_package/EURUSD/train.parquet')
    if not train_path.exists():
        print("  Training data not found, skipping...")
        return

    try:
        # Load a sample
        df = pd.read_parquet(train_path).head(5000)

        # Prepare OHLCV
        df['mid'] = (df['bid'] + df['ask']) / 2
        df['open'] = df['mid']
        df['high'] = df['mid']
        df['low'] = df['mid']
        df['close'] = df['mid']
        df['volume'] = 1
        df['returns'] = df['close'].pct_change()
        df['vwap'] = df['mid']

        print(f"  Loaded {len(df)} rows")

        # Generate all features
        from core.features.alpha158 import Alpha158
        from core.features.alpha360 import Alpha360Compact
        from core.features.barra_cne6 import BarraCNE6Forex

        all_features = []

        # Alpha158
        print("  Generating Alpha158...", flush=True)
        a158 = Alpha158().generate_all(df)
        all_features.append(a158)

        # Alpha360 Compact (for speed)
        print("  Generating Alpha360 Compact...", flush=True)
        a360 = Alpha360Compact(lookback=20).generate_all(df)
        all_features.append(a360)

        # Barra
        print("  Generating Barra CNE6...", flush=True)
        barra = BarraCNE6Forex().generate_all(df)
        all_features.append(barra)

        # Combine
        combined = pd.concat(all_features, axis=1)
        combined = combined.fillna(0).replace([np.inf, -np.inf], 0)

        print(f"  Total features: {len(combined.columns)}")
        print(f"  Dataset shape: {combined.shape}")
        print(f"  Memory usage: {combined.memory_usage(deep=True).sum() / 1e6:.2f} MB")
        print("  STATUS: PASS")

    except Exception as e:
        print(f"  ERROR: {e}")
        import traceback
        traceback.print_exc()


def main():
    """Run all tests."""
    print("\n" + "="*60)
    print("ALPHA LIBRARY TEST SUITE")
    print("Chinese Quant Factor Libraries")
    print("="*60)

    results = {}

    # Run tests - Chinese Quant
    results['Alpha101'] = test_alpha101()
    results['Alpha158'] = test_alpha158()
    results['Alpha360'] = test_alpha360()
    results['BarraCNE6'] = test_barra_cne6()
    results['Renaissance'] = test_renaissance()

    # Run tests - US Academic
    results['USAcademic'] = test_us_academic()

    # Run tests - GitHub/Gitee Gold Standard
    results['MLFinLab'] = test_mlfinlab()
    results['TimeSeries'] = test_timeseries()

    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)

    total_features = 0
    all_pass = True

    print(f"{'Library':<15} {'Status':<10} {'Features':>10}")
    print("-"*40)

    for name, (passed, count) in results.items():
        status = "PASS" if passed else "FAIL"
        print(f"{name:<15} {status:<10} {count:>10}")
        total_features += count
        all_pass = all_pass and passed

    print("-"*40)
    print(f"{'TOTAL':<15} {'':10} {total_features:>10}")

    # Test with real data
    test_with_real_data()

    print("\n" + "="*60)
    if all_pass:
        print("ALL TESTS PASSED!")
        print(f"Total features available: {total_features}")
    else:
        print("SOME TESTS FAILED!")
    print("="*60)


if __name__ == '__main__':
    main()
