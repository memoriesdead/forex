"""
ParallelTrainer - Pipeline Parallel Training for RTX 5080 + Ryzen 9 9900X
=========================================================================
Maximizes GPU utilization by running feature generation (CPU) in parallel
with model training (GPU).

Architecture:
    Stage 1: Feature Generation (CPU - 4 workers)
    ProcessPoolExecutor generates mega features for 4 pairs simultaneously

    Stage 2: GPU Training (3 concurrent jobs)
    ThreadPoolExecutor trains XGB+LGB+CB for 3 pairs on GPU

Expected speedup: ~3x (10 hours -> 3.5 hours for 52 pairs)

Usage:
    from core.ml.parallel_trainer import ParallelTrainer

    trainer = ParallelTrainer()
    results = trainer.train_all(pairs=['EURUSD', 'GBPUSD', ...])
"""

import numpy as np
import pandas as pd
from pathlib import Path
import json
import pickle
from datetime import datetime
import sys
import time
import gc
import threading
import queue
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from typing import List, Dict, Tuple, Optional
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def generate_features_for_pair(args: Tuple) -> Tuple[str, Optional[str], Optional[str]]:
    """
    CPU worker: Generate mega features for one pair.

    This function runs in a separate process to parallelize CPU-bound feature generation.

    Args:
        args: Tuple of (pair, training_dir, output_dir, max_samples)

    Returns:
        Tuple of (pair, features_path, error) - error is None on success
    """
    pair, training_dir, features_cache_dir, max_samples = args

    try:
        import pandas as pd
        import numpy as np
        import gc
        from pathlib import Path
        import sys

        # Add project root to path (required for subprocess)
        project_root = Path(training_dir).parent
        if str(project_root) not in sys.path:
            sys.path.insert(0, str(project_root))

        pair_dir = Path(training_dir) / pair
        features_path = Path(features_cache_dir) / f"{pair}_features.pkl"

        # Skip if already cached
        if features_path.exists():
            print(f"[CPU] {pair}: Using cached features")
            return (pair, str(features_path), None)

        print(f"[CPU] {pair}: Loading data...")

        # Load data
        train = pd.read_parquet(pair_dir / 'train.parquet')
        val = pd.read_parquet(pair_dir / 'val.parquet')
        test = pd.read_parquet(pair_dir / 'test.parquet')

        # Ensure close column
        for df in [train, val, test]:
            if 'close' not in df.columns and 'mid' in df.columns:
                df['close'] = df['mid']

        print(f"[CPU] {pair}: Generating mega features...")

        # Import here to avoid subprocess issues
        from core.features.mega_generator import MegaFeatureGenerator

        generator = MegaFeatureGenerator(verbose=False)

        train_features = generator.generate_all(train)
        val_features = generator.generate_all(val)
        test_features = generator.generate_all(test)

        # Filter to numeric only
        numeric_cols = train_features.select_dtypes(include=[np.number]).columns.tolist()
        train_features = train_features[numeric_cols]
        val_features = val_features[numeric_cols]
        test_features = test_features[numeric_cols]

        feature_names = numeric_cols
        print(f"[CPU] {pair}: Generated {len(feature_names)} features")

        # Get target columns
        target_cols = [c for c in train.columns if c.startswith('target_direction')]

        # Subsample if needed
        X_train = train_features.values
        X_val = val_features.values
        X_test = test_features.values

        y_train = {t: train[t].values for t in target_cols}
        y_val = {t: val[t].values for t in target_cols}
        y_test = {t: test[t].values for t in target_cols}

        # Subsample training data if too large
        if len(X_train) > max_samples:
            idx = np.random.choice(len(X_train), max_samples, replace=False)
            X_train = X_train[idx]
            y_train = {t: y_train[t][idx] for t in target_cols}

        # Save features to cache
        features_data = {
            'X_train': X_train,
            'X_val': X_val,
            'X_test': X_test,
            'y_train': y_train,
            'y_val': y_val,
            'y_test': y_test,
            'feature_names': feature_names,
            'target_cols': target_cols,
        }

        Path(features_cache_dir).mkdir(parents=True, exist_ok=True)
        with open(features_path, 'wb') as f:
            pickle.dump(features_data, f)

        print(f"[CPU] {pair}: Features cached to {features_path}")

        # Clean up memory
        del train, val, test, train_features, val_features, test_features
        gc.collect()

        return (pair, str(features_path), None)

    except Exception as e:
        import traceback
        error_msg = f"{e}\n{traceback.format_exc()}"
        print(f"[CPU] {pair}: ERROR - {e}")
        return (pair, None, error_msg)


class ParallelTrainer:
    """
    Pipeline parallel trainer for maximum GPU + CPU utilization.

    Architecture:
        CPU: 4 workers generate features in parallel
        GPU: 3 workers train models concurrently

    Memory budget:
        - Per pair GPU: ~5GB (XGB + LGB + CB)
        - Total GPU: 15GB / 16GB VRAM
    """

    def __init__(
        self,
        cpu_workers: int = 4,
        gpu_workers: int = 3,
        max_samples: int = 50000,
        training_dir: Optional[Path] = None,
        output_dir: Optional[Path] = None,
        features_cache_dir: Optional[Path] = None,
    ):
        """
        Initialize ParallelTrainer.

        Args:
            cpu_workers: Number of CPU workers for feature generation (default: 4)
            gpu_workers: Number of concurrent GPU training jobs (default: 3)
            max_samples: Max training samples per pair (default: 50000)
            training_dir: Path to training_package directory
            output_dir: Path to models/production directory
            features_cache_dir: Path to cache generated features (default: training_package/features_cache)
        """
        self.cpu_workers = cpu_workers
        self.gpu_workers = gpu_workers
        self.max_samples = max_samples

        # Paths
        project_dir = Path(__file__).parent.parent.parent
        self.training_dir = training_dir or project_dir / 'training_package'
        self.output_dir = output_dir or project_dir / 'models' / 'production'
        self.features_cache_dir = features_cache_dir or self.training_dir / 'features_cache'

        # Create directories
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.features_cache_dir.mkdir(parents=True, exist_ok=True)

        # Training queue and results
        self.feature_queue = queue.Queue()
        self.results = {}
        self.lock = threading.Lock()

        # GPU lock - XGBoost and CatBoost can't handle concurrent GPU access safely
        # LightGBM seems more robust with concurrent access
        self.gpu_lock = threading.Lock()

        # Progress tracking
        self.pairs_total = 0
        self.pairs_features_done = 0
        self.pairs_trained = 0
        self.start_time = None

    def _train_gpu_worker(self, pair: str, features_path: str) -> Tuple[str, Optional[Dict]]:
        """
        GPU worker: Train XGB+LGB+CB for one pair.

        This function runs in a thread pool to allow concurrent GPU training.
        The GIL is released during GPU operations, so 3 jobs can run concurrently.

        Args:
            pair: Currency pair symbol
            features_path: Path to cached features pickle

        Returns:
            Tuple of (pair, results_dict) - results_dict is None on failure
        """
        try:
            import xgboost as xgb
            import lightgbm as lgb
            import catboost as cb
            from sklearn.metrics import accuracy_score, roc_auc_score, f1_score

            print(f"[GPU] {pair}: Loading features...")

            # Load cached features
            with open(features_path, 'rb') as f:
                data = pickle.load(f)

            X_train = data['X_train']
            X_val = data['X_val']
            X_test = data['X_test']
            y_train = data['y_train']
            y_val = data['y_val']
            y_test = data['y_test']
            feature_names = data['feature_names']
            target_cols = data['target_cols']

            print(f"[GPU] {pair}: Training on {len(X_train)} samples, {len(feature_names)} features")

            results = {}
            models = {}

            # GPU params optimized for concurrent training (reduced depth to save VRAM)
            xgb_params = {
                'tree_method': 'hist',
                'device': 'cuda',
                'max_depth': 8,
                'max_bin': 128,
                'learning_rate': 0.05,
                'subsample': 0.7,
                'colsample_bytree': 0.6,
                'objective': 'binary:logistic',
                'eval_metric': 'auc',
                'nthread': 4,  # Reduce CPU threads per job
            }

            lgb_params = {
                'device': 'gpu',
                'gpu_platform_id': 0,
                'gpu_device_id': 0,
                'max_depth': 8,
                'num_leaves': 127,
                'feature_fraction': 0.6,
                'bagging_fraction': 0.7,
                'bagging_freq': 5,
                'verbose': -1,
                'objective': 'binary',
                'metric': 'auc',
                'n_jobs': 4,  # Reduce CPU threads per job
            }

            cb_params = {
                'task_type': 'GPU',
                'devices': '0',
                'depth': 6,
                'iterations': 300,
                'early_stopping_rounds': 20,
                'verbose': False,
                'loss_function': 'Logloss',
                'thread_count': 4,  # Reduce CPU threads per job
            }

            # GPU lock - serialize all GPU training to prevent memory conflicts
            # Feature generation is still parallel, but GPU training is sequential
            # This is safer and prevents OOM errors on 16GB VRAM
            with self.gpu_lock:
                for target in target_cols[:3]:  # Top 3 horizons
                    print(f"[GPU] {pair}: Training target {target}...")

                    y_tr = y_train[target]
                    y_v = y_val[target]
                    y_te = y_test[target]

                    # XGBoost
                    dtrain = xgb.DMatrix(X_train, label=y_tr)
                    dval = xgb.DMatrix(X_val, label=y_v)

                    xgb_model = xgb.train(
                        xgb_params,
                        dtrain,
                        num_boost_round=500,
                        evals=[(dval, 'val')],
                        early_stopping_rounds=30,
                        verbose_eval=False
                    )

                    # LightGBM
                    train_data = lgb.Dataset(X_train, label=y_tr)
                    val_data = lgb.Dataset(X_val, label=y_v, reference=train_data)

                    lgb_model = lgb.train(
                        lgb_params,
                        train_data,
                        num_boost_round=500,
                        valid_sets=[val_data],
                        callbacks=[lgb.early_stopping(30, verbose=False)]
                    )

                    # CatBoost
                    cb_model = cb.CatBoostClassifier(**cb_params)
                    cb_model.fit(X_train, y_tr, eval_set=(X_val, y_v))

                    # Ensemble predictions
                    dtest = xgb.DMatrix(X_test)
                    xgb_pred = xgb_model.predict(dtest)
                    lgb_pred = lgb_model.predict(X_test)
                    cb_pred = cb_model.predict_proba(X_test)[:, 1]

                    ensemble_pred = (xgb_pred + lgb_pred + cb_pred) / 3
                    ensemble_class = (ensemble_pred > 0.5).astype(int)

                    # Metrics
                    acc = accuracy_score(y_te, ensemble_class)
                    auc = roc_auc_score(y_te, ensemble_pred)
                    f1 = f1_score(y_te, ensemble_class)

                    print(f"[GPU] {pair} {target}: ACC={acc:.4f}, AUC={auc:.4f}, F1={f1:.4f}")

                    results[target] = {
                        'accuracy': float(acc),
                        'auc': float(auc),
                        'f1': float(f1)
                    }

                    models[target] = {
                        'xgb': xgb_model,
                        'lgb': lgb_model,
                        'cb': cb_model,
                    }

                    # Clean up intermediate data
                    del dtrain, dval, dtest

                # Explicit GPU memory cleanup after all targets trained
                gc.collect()
                try:
                    import torch
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                except:
                    pass

            # Save models
            model_path = self.output_dir / f"{pair}_models.pkl"
            with open(model_path, 'wb') as f:
                pickle.dump(models, f)

            # Save results
            results['_meta'] = {
                'feature_mode': 'mega',
                'feature_count': len(feature_names),
                'timestamp': datetime.now().isoformat()
            }

            results_path = self.output_dir / f"{pair}_results.json"
            with open(results_path, 'w') as f:
                json.dump(results, f, indent=2)

            print(f"[GPU] {pair}: Saved to {model_path}")

            # Clean up
            del X_train, X_val, X_test, data
            gc.collect()

            return (pair, results)

        except Exception as e:
            import traceback
            print(f"[GPU] {pair}: ERROR - {e}")
            traceback.print_exc()
            return (pair, None)

    def _get_already_trained(self) -> set:
        """Get set of pairs that already have results."""
        trained = set()
        for f in self.output_dir.glob('*_results.json'):
            pair = f.stem.replace('_results', '')
            trained.add(pair)
        return trained

    def _get_pairs_with_data(self) -> List[str]:
        """Get list of pairs that have training data."""
        pairs = []
        for d in self.training_dir.iterdir():
            if d.is_dir() and (d / 'train.parquet').exists():
                if d.name != 'features_cache':
                    pairs.append(d.name)
        return sorted(pairs)

    def train_all(
        self,
        pairs: Optional[List[str]] = None,
        skip_trained: bool = True,
        monitor: bool = True,
        dry_run: bool = False,
    ) -> Dict[str, Dict]:
        """
        Train all pairs with pipeline parallelism.

        Args:
            pairs: List of pairs to train (None = all with data)
            skip_trained: Skip pairs that already have results
            monitor: Print progress updates
            dry_run: Only print what would be done, don't train

        Returns:
            Dict mapping pair -> results
        """
        self.start_time = time.time()

        # Get pairs to process
        if pairs is None:
            pairs = self._get_pairs_with_data()

        if skip_trained:
            already_trained = self._get_already_trained()
            pairs = [p for p in pairs if p not in already_trained]

        self.pairs_total = len(pairs)

        if self.pairs_total == 0:
            print("No pairs to train!")
            return {}

        print(f"\n{'='*60}")
        print(f"PARALLEL TRAINING: {self.pairs_total} pairs")
        print(f"CPU workers: {self.cpu_workers}")
        print(f"GPU workers: {self.gpu_workers}")
        print(f"Max samples: {self.max_samples}")
        print(f"{'='*60}\n")

        if dry_run:
            print("DRY RUN - would train:")
            for i, p in enumerate(pairs):
                print(f"  [{i+1}] {p}")
            return {}

        # Stage 1: Feature generation (CPU, parallel processes)
        print(f"\n{'='*40}")
        print("STAGE 1: Feature Generation (CPU)")
        print(f"{'='*40}")

        feature_args = [
            (p, str(self.training_dir), str(self.features_cache_dir), self.max_samples)
            for p in pairs
        ]

        feature_results = []
        with ProcessPoolExecutor(max_workers=self.cpu_workers) as cpu_pool:
            futures = {cpu_pool.submit(generate_features_for_pair, args): args[0] for args in feature_args}

            for future in as_completed(futures):
                pair = futures[future]
                try:
                    result = future.result()
                    feature_results.append(result)
                    self.pairs_features_done += 1

                    if monitor:
                        elapsed = time.time() - self.start_time
                        rate = self.pairs_features_done / (elapsed / 60)
                        print(f"[Progress] Features: {self.pairs_features_done}/{self.pairs_total} "
                              f"({rate:.1f} pairs/min)")

                except Exception as e:
                    print(f"[ERROR] Feature generation failed for {pair}: {e}")
                    feature_results.append((pair, None, str(e)))

        # Filter successful feature generations
        ready_for_training = [(p, fp) for p, fp, err in feature_results if fp is not None]
        failed = [(p, err) for p, fp, err in feature_results if fp is None]

        if failed:
            print(f"\n[WARNING] {len(failed)} pairs failed feature generation:")
            for p, err in failed:
                print(f"  - {p}: {err[:100]}...")

        print(f"\n[Complete] Features ready: {len(ready_for_training)}/{self.pairs_total}")

        # Stage 2: GPU training (concurrent threads)
        print(f"\n{'='*40}")
        print("STAGE 2: GPU Training")
        print(f"{'='*40}")

        with ThreadPoolExecutor(max_workers=self.gpu_workers) as gpu_pool:
            futures = {
                gpu_pool.submit(self._train_gpu_worker, pair, features_path): pair
                for pair, features_path in ready_for_training
            }

            for future in as_completed(futures):
                pair = futures[future]
                try:
                    _, result = future.result()
                    if result:
                        self.results[pair] = result
                    self.pairs_trained += 1

                    if monitor:
                        elapsed = time.time() - self.start_time
                        rate = self.pairs_trained / (elapsed / 60)
                        remaining = (self.pairs_total - self.pairs_trained) / max(rate, 0.1)
                        print(f"[Progress] Trained: {self.pairs_trained}/{len(ready_for_training)} "
                              f"({rate:.1f} pairs/min, ~{remaining:.0f}min remaining)")

                except Exception as e:
                    print(f"[ERROR] Training failed for {pair}: {e}")

        # Summary
        total_time = time.time() - self.start_time
        print(f"\n{'='*60}")
        print("TRAINING COMPLETE")
        print(f"{'='*60}")
        print(f"Total time: {total_time/60:.1f} minutes")
        print(f"Pairs trained: {len(self.results)}/{self.pairs_total}")
        print(f"Average: {total_time/60/max(len(self.results), 1):.1f} min/pair")

        if self.results:
            print("\nResults Summary:")
            for pair, res in self.results.items():
                for target, metrics in res.items():
                    if not target.startswith('_'):
                        print(f"  {pair} {target}: ACC={metrics['accuracy']:.4f}, AUC={metrics['auc']:.4f}")

        # Save summary
        summary = {
            'timestamp': datetime.now().isoformat(),
            'total_time_minutes': total_time / 60,
            'pairs_trained': list(self.results.keys()),
            'results': self.results,
            'config': {
                'cpu_workers': self.cpu_workers,
                'gpu_workers': self.gpu_workers,
                'max_samples': self.max_samples,
            }
        }

        summary_path = self.output_dir / 'parallel_training_summary.json'
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)

        print(f"\nSummary saved to {summary_path}")

        return self.results

    def clear_feature_cache(self):
        """Clear the feature cache to free disk space."""
        import shutil
        if self.features_cache_dir.exists():
            shutil.rmtree(self.features_cache_dir)
            self.features_cache_dir.mkdir()
            print(f"Cleared feature cache: {self.features_cache_dir}")


# Export for module usage
__all__ = ['ParallelTrainer', 'generate_features_for_pair']
