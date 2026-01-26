#!/usr/bin/env python3
"""
Training CLI
============
Train ML models for multi-symbol forex trading.

Usage:
    python scripts/train.py --tier majors
    python scripts/train.py --symbols EURUSD,GBPUSD,USDJPY
    python scripts/train.py --tier all --parallel 4
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import logging
import json
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import List, Dict, Any, Optional

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train ML models for forex trading"
    )
    parser.add_argument(
        '--tier',
        choices=['majors', 'crosses', 'exotics', 'all'],
        help='Symbol tier to train'
    )
    parser.add_argument(
        '--symbols',
        type=str,
        help='Comma-separated list of symbols'
    )
    parser.add_argument(
        '--parallel',
        type=int,
        default=1,
        help='Parallel training jobs (default: 1)'
    )
    parser.add_argument(
        '--target',
        type=str,
        default='target_direction_10',
        help='Target variable (default: target_direction_10)'
    )
    parser.add_argument(
        '--days',
        type=int,
        default=30,
        help='Days of data to use (default: 30)'
    )
    parser.add_argument(
        '--data-dir',
        type=str,
        default='training_package',
        help='Data directory (default: training_package)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='models/production',
        help='Output directory (default: models/production)'
    )
    parser.add_argument(
        '--list',
        action='store_true',
        help='List available symbols and exit'
    )
    return parser.parse_args()


def list_symbols():
    """List available symbols."""
    from core.symbol.registry import SymbolRegistry

    registry = SymbolRegistry.get()

    print("\n" + "=" * 60)
    print("AVAILABLE SYMBOLS")
    print("=" * 60)

    for tier in ['majors', 'crosses', 'exotics']:
        pairs = registry.get_enabled(tier=tier)
        print(f"\n{tier.upper()} ({len(pairs)} pairs):")
        for p in pairs:
            print(f"  {p.symbol}: {p.base_currency}/{p.quote_currency} (pip={p.pip_value})")

    print("\n" + "=" * 60 + "\n")


def train_symbol(
    symbol: str,
    data_dir: Path,
    output_dir: Path,
    target: str
) -> Dict[str, Any]:
    """
    Train models for a single symbol.

    Args:
        symbol: Trading symbol
        data_dir: Directory with training data
        output_dir: Directory for output models
        target: Target variable name

    Returns:
        Results dictionary
    """
    import numpy as np
    import pandas as pd
    import pickle

    result = {
        'symbol': symbol,
        'status': 'error',
        'accuracy': 0.0,
        'auc': 0.0,
        'models': {},
    }

    try:
        # Load data
        symbol_dir = data_dir / symbol
        if not symbol_dir.exists():
            result['error'] = f"No data directory: {symbol_dir}"
            return result

        train_path = symbol_dir / "train.parquet"
        val_path = symbol_dir / "val.parquet"
        test_path = symbol_dir / "test.parquet"

        if not all(p.exists() for p in [train_path, val_path, test_path]):
            result['error'] = "Missing train/val/test files"
            return result

        train = pd.read_parquet(train_path)
        val = pd.read_parquet(val_path)
        test = pd.read_parquet(test_path)

        # Load metadata
        meta_path = symbol_dir / "metadata.json"
        if meta_path.exists():
            with open(meta_path) as f:
                metadata = json.load(f)
            feature_names = metadata.get('feature_names', [])
        else:
            feature_names = [c for c in train.columns if not c.startswith('target_')]

        # Prepare data
        X_train = train[feature_names].values
        X_val = val[feature_names].values
        X_test = test[feature_names].values

        if target not in train.columns:
            result['error'] = f"Target {target} not in data"
            return result

        y_train = train[target].values
        y_val = val[target].values
        y_test = test[target].values

        models = {}

        # Train XGBoost
        try:
            import xgboost as xgb
            from core.ml.gpu_config import get_xgb_gpu_params

            params = get_xgb_gpu_params()
            params.update({
                'objective': 'binary:logistic',
                'eval_metric': 'auc',
            })

            dtrain = xgb.DMatrix(X_train, label=y_train)
            dval = xgb.DMatrix(X_val, label=y_val)

            xgb_model = xgb.train(
                params,
                dtrain,
                num_boost_round=500,
                evals=[(dval, 'val')],
                early_stopping_rounds=50,
                verbose_eval=False
            )
            models['xgboost'] = xgb_model
            logger.info(f"{symbol} XGBoost trained")
        except Exception as e:
            logger.warning(f"{symbol} XGBoost error: {e}")

        # Train LightGBM
        try:
            import lightgbm as lgb
            from core.ml.gpu_config import get_lgb_gpu_params

            params = get_lgb_gpu_params()
            params.update({
                'objective': 'binary',
                'metric': 'auc',
                'verbosity': -1,
            })

            train_data = lgb.Dataset(X_train, label=y_train)
            val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

            lgb_model = lgb.train(
                params,
                train_data,
                num_boost_round=500,
                valid_sets=[val_data],
                callbacks=[lgb.early_stopping(50, verbose=False)]
            )
            models['lightgbm'] = lgb_model
            logger.info(f"{symbol} LightGBM trained")
        except Exception as e:
            logger.warning(f"{symbol} LightGBM error: {e}")

        # Train CatBoost
        try:
            import catboost as cb
            from core.ml.gpu_config import get_catboost_gpu_params

            params = get_catboost_gpu_params()

            cb_model = cb.CatBoostClassifier(**params)
            cb_model.fit(
                X_train, y_train,
                eval_set=(X_val, y_val),
                early_stopping_rounds=50,
                verbose=False
            )
            models['catboost'] = cb_model
            logger.info(f"{symbol} CatBoost trained")
        except Exception as e:
            logger.warning(f"{symbol} CatBoost error: {e}")

        if not models:
            result['error'] = "No models trained successfully"
            return result

        # Evaluate on test set
        from sklearn.metrics import accuracy_score, roc_auc_score

        predictions = []
        probabilities = []

        for name, model in models.items():
            try:
                if name == 'xgboost':
                    dtest = xgb.DMatrix(X_test)
                    prob = model.predict(dtest)
                elif name == 'lightgbm':
                    prob = model.predict(X_test)
                elif name == 'catboost':
                    prob = model.predict_proba(X_test)[:, 1]
                else:
                    continue

                predictions.append((prob > 0.5).astype(int))
                probabilities.append(prob)
            except Exception:
                continue

        if not predictions:
            result['error'] = "No predictions generated"
            return result

        # Ensemble voting
        ensemble_pred = (np.mean(predictions, axis=0) > 0.5).astype(int)
        ensemble_prob = np.mean(probabilities, axis=0)

        accuracy = accuracy_score(y_test, ensemble_pred)
        auc = roc_auc_score(y_test, ensemble_prob)

        # Save models
        output_dir.mkdir(parents=True, exist_ok=True)
        model_path = output_dir / f"{symbol}_models.pkl"

        model_data = {
            target: {
                'xgboost': models.get('xgboost'),
                'lightgbm': models.get('lightgbm'),
                'catboost': models.get('catboost'),
                'features': feature_names,
            }
        }

        with open(model_path, 'wb') as f:
            pickle.dump(model_data, f)

        # Save results
        results_path = output_dir / f"{symbol}_results.json"
        with open(results_path, 'w') as f:
            json.dump({
                'symbol': symbol,
                'target': target,
                'accuracy': accuracy,
                'auc': auc,
                'n_train': len(X_train),
                'n_test': len(X_test),
                'n_features': len(feature_names),
                'trained_at': datetime.now().isoformat(),
            }, f, indent=2)

        result['status'] = 'success'
        result['accuracy'] = accuracy
        result['auc'] = auc
        result['models'] = list(models.keys())
        result['model_path'] = str(model_path)

        logger.info(f"{symbol}: Accuracy={accuracy:.2%}, AUC={auc:.4f}")
        return result

    except Exception as e:
        result['error'] = str(e)
        logger.error(f"{symbol} training error: {e}")
        return result


def main():
    args = parse_args()

    if args.list:
        list_symbols()
        return

    # Get symbols
    from core.symbol.registry import SymbolRegistry
    registry = SymbolRegistry.get()

    if args.symbols:
        symbols = [s.strip().upper() for s in args.symbols.split(',')]
    elif args.tier:
        if args.tier == 'all':
            symbols = [p.symbol for p in registry.get_enabled()]
        else:
            symbols = [p.symbol for p in registry.get_enabled(tier=args.tier)]
    else:
        print("Error: Must specify --tier or --symbols")
        print("Example: python scripts/train.py --tier majors")
        sys.exit(1)

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)

    print("\n" + "=" * 60)
    print("MODEL TRAINING")
    print("=" * 60)
    print(f"  Symbols: {len(symbols)}")
    print(f"  Target: {args.target}")
    print(f"  Parallel: {args.parallel}")
    print(f"  Data: {data_dir}")
    print(f"  Output: {output_dir}")
    print("=" * 60 + "\n")

    results = []
    start_time = datetime.now()

    if args.parallel > 1:
        # Parallel training
        with ProcessPoolExecutor(max_workers=args.parallel) as executor:
            futures = {
                executor.submit(
                    train_symbol, s, data_dir, output_dir, args.target
                ): s for s in symbols
            }

            for future in as_completed(futures):
                symbol = futures[future]
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    logger.error(f"{symbol} failed: {e}")
                    results.append({
                        'symbol': symbol,
                        'status': 'error',
                        'error': str(e)
                    })
    else:
        # Sequential training
        for symbol in symbols:
            result = train_symbol(symbol, data_dir, output_dir, args.target)
            results.append(result)

    elapsed = (datetime.now() - start_time).total_seconds()

    # Summary
    print("\n" + "=" * 60)
    print("TRAINING SUMMARY")
    print("=" * 60)

    successful = [r for r in results if r['status'] == 'success']
    failed = [r for r in results if r['status'] == 'error']

    print(f"\n  Total: {len(results)}")
    print(f"  Success: {len(successful)}")
    print(f"  Failed: {len(failed)}")
    print(f"  Time: {elapsed:.1f}s")

    if successful:
        avg_acc = sum(r['accuracy'] for r in successful) / len(successful)
        avg_auc = sum(r['auc'] for r in successful) / len(successful)
        print(f"\n  Average Accuracy: {avg_acc:.2%}")
        print(f"  Average AUC: {avg_auc:.4f}")

        print(f"\n  Results by symbol:")
        for r in sorted(successful, key=lambda x: x['accuracy'], reverse=True):
            print(f"    {r['symbol']}: {r['accuracy']:.2%} ({r['auc']:.4f})")

    if failed:
        print(f"\n  Failed symbols:")
        for r in failed:
            print(f"    {r['symbol']}: {r.get('error', 'Unknown error')}")

    print("\n" + "=" * 60 + "\n")


if __name__ == '__main__':
    main()
