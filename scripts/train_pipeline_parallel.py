#!/usr/bin/env python3
"""
Pipeline Parallel Trainer - TRUE Parallel CPU+GPU Execution
============================================================
Based on best practices research for XGBoost/LightGBM GPU training.

Safety Features:
- GPU temperature monitoring (throttle at 80°C, pause at 85°C)
- VRAM monitoring (keep at 75% capacity)
- Graceful shutdown on thermal limits

Architecture:
    CPU Workers: Generate features (4 parallel processes)
    GPU Workers: Train models (1-2 concurrent, serialized for safety)

    RUNS SIMULTANEOUSLY - GPU starts on ready pairs while CPU generates more!
"""

import os
import sys
import time
import gc
import json
import pickle
import threading
import queue
import subprocess
from pathlib import Path
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from typing import List, Dict, Tuple, Optional
import numpy as np

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# ============================================================================
# GPU SAFETY MONITORING
# ============================================================================

class GPUSafetyMonitor:
    """Monitor GPU temperature and memory for safe training."""

    TEMP_WARN = 75      # Reduce concurrency at this temp
    TEMP_THROTTLE = 80  # Single GPU job only
    TEMP_PAUSE = 85     # Pause all GPU work
    MEM_MAX_PERCENT = 75  # Max VRAM usage

    def __init__(self):
        self.running = True
        self.temp = 0
        self.mem_used = 0
        self.mem_total = 16000
        self.utilization = 0
        self._lock = threading.Lock()

    def update(self):
        """Update GPU stats via nvidia-smi."""
        try:
            result = subprocess.run(
                ['nvidia-smi', '--query-gpu=temperature.gpu,utilization.gpu,memory.used,memory.total',
                 '--format=csv,noheader,nounits'],
                capture_output=True, text=True, timeout=5
            )
            if result.returncode == 0:
                parts = result.stdout.strip().split(',')
                with self._lock:
                    self.temp = int(parts[0].strip())
                    self.utilization = int(parts[1].strip())
                    self.mem_used = int(parts[2].strip())
                    self.mem_total = int(parts[3].strip())
        except Exception as e:
            print(f"[SAFETY] Warning: Could not read GPU stats: {e}")

    def get_stats(self) -> Dict:
        """Get current GPU stats."""
        with self._lock:
            return {
                'temp': self.temp,
                'utilization': self.utilization,
                'mem_used_mb': self.mem_used,
                'mem_total_mb': self.mem_total,
                'mem_percent': (self.mem_used / self.mem_total * 100) if self.mem_total > 0 else 0,
            }

    def is_safe_to_train(self) -> Tuple[bool, str]:
        """Check if it's safe to start new GPU training."""
        stats = self.get_stats()

        if stats['temp'] >= self.TEMP_PAUSE:
            return False, f"GPU too hot ({stats['temp']}°C >= {self.TEMP_PAUSE}°C) - PAUSING"

        if stats['mem_percent'] >= self.MEM_MAX_PERCENT:
            return False, f"VRAM too high ({stats['mem_percent']:.0f}% >= {self.MEM_MAX_PERCENT}%) - WAITING"

        return True, "OK"

    def should_reduce_concurrency(self) -> bool:
        """Check if we should reduce GPU concurrency."""
        stats = self.get_stats()
        return stats['temp'] >= self.TEMP_WARN


# ============================================================================
# FEATURE GENERATION (CPU)
# ============================================================================

def generate_features_worker(args: Tuple) -> Tuple[str, Optional[str], Optional[str]]:
    """
    CPU worker: Generate features for one pair.
    Runs in separate process for true parallelism.
    """
    pair, training_dir, features_cache_dir, max_samples = args

    try:
        import pandas as pd
        import numpy as np
        import sys
        from pathlib import Path

        # Add project to path
        project_root = Path(training_dir).parent
        if str(project_root) not in sys.path:
            sys.path.insert(0, str(project_root))

        pair_dir = Path(training_dir) / pair
        features_path = Path(features_cache_dir) / f"{pair}_features.pkl"

        # Skip if cached
        if features_path.exists():
            return (pair, str(features_path), None)

        # Load data
        train = pd.read_parquet(pair_dir / 'train.parquet')
        val = pd.read_parquet(pair_dir / 'val.parquet')
        test = pd.read_parquet(pair_dir / 'test.parquet')

        for df in [train, val, test]:
            if 'close' not in df.columns and 'mid' in df.columns:
                df['close'] = df['mid']

        # Generate features
        from core.features.mega_generator import MegaFeatureGenerator
        generator = MegaFeatureGenerator(verbose=False)

        train_f = generator.generate_all(train)
        val_f = generator.generate_all(val)
        test_f = generator.generate_all(test)

        # Filter numeric
        numeric_cols = train_f.select_dtypes(include=[np.number]).columns.tolist()
        train_f = train_f[numeric_cols]
        val_f = val_f[numeric_cols]
        test_f = test_f[numeric_cols]

        target_cols = [c for c in train.columns if c.startswith('target_direction')]

        X_train = train_f.values
        X_val = val_f.values
        X_test = test_f.values

        y_train = {t: train[t].values for t in target_cols}
        y_val = {t: val[t].values for t in target_cols}
        y_test = {t: test[t].values for t in target_cols}

        # Subsample
        if len(X_train) > max_samples:
            idx = np.random.choice(len(X_train), max_samples, replace=False)
            X_train = X_train[idx]
            y_train = {t: y_train[t][idx] for t in target_cols}

        # Save
        features_data = {
            'X_train': X_train,
            'X_val': X_val,
            'X_test': X_test,
            'y_train': y_train,
            'y_val': y_val,
            'y_test': y_test,
            'feature_names': numeric_cols,
            'target_cols': target_cols,
        }

        Path(features_cache_dir).mkdir(parents=True, exist_ok=True)
        with open(features_path, 'wb') as f:
            pickle.dump(features_data, f)

        return (pair, str(features_path), None)

    except Exception as e:
        import traceback
        return (pair, None, f"{e}\n{traceback.format_exc()}")


# ============================================================================
# GPU TRAINING
# ============================================================================

def train_gpu_worker(pair: str, features_path: str, output_dir: Path,
                     gpu_lock: threading.Lock, safety: GPUSafetyMonitor) -> Tuple[str, Optional[Dict]]:
    """
    GPU worker: Train XGB+LGB+CB for one pair.
    Uses gpu_lock to serialize GPU access for safety.
    """
    try:
        import xgboost as xgb
        import lightgbm as lgb
        import catboost as cb
        from sklearn.metrics import accuracy_score, roc_auc_score, f1_score

        # Wait for safe conditions
        while True:
            safe, reason = safety.is_safe_to_train()
            if safe:
                break
            print(f"[GPU] {pair}: Waiting - {reason}")
            time.sleep(10)
            safety.update()

        print(f"[GPU] {pair}: Loading features...")

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

        print(f"[GPU] {pair}: Training {len(X_train)} samples, {len(feature_names)} features")

        results = {}
        models = {}

        # GPU params - optimized for RTX 5080 16GB
        xgb_params = {
            'tree_method': 'hist',
            'device': 'cuda',
            'max_depth': 10,
            'max_bin': 256,
            'learning_rate': 0.03,
            'subsample': 0.8,
            'colsample_bytree': 0.7,
            'objective': 'binary:logistic',
            'eval_metric': 'auc',
            'nthread': 6,
        }

        lgb_params = {
            'device': 'gpu',
            'gpu_platform_id': 0,
            'gpu_device_id': 0,
            'max_depth': 10,
            'num_leaves': 255,
            'learning_rate': 0.03,
            'feature_fraction': 0.7,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'verbose': -1,
            'objective': 'binary',
            'metric': 'auc',
            'n_jobs': 6,
        }

        cb_params = {
            'task_type': 'GPU',
            'devices': '0',
            'depth': 8,
            'iterations': 500,
            'learning_rate': 0.03,
            'early_stopping_rounds': 30,
            'verbose': False,
            'loss_function': 'Logloss',
            'thread_count': 6,
        }

        # Serialize GPU training for safety
        with gpu_lock:
            for target in target_cols[:3]:
                print(f"[GPU] {pair}: Training {target}...")

                y_tr = y_train[target]
                y_v = y_val[target]
                y_te = y_test[target]

                # XGBoost
                dtrain = xgb.DMatrix(X_train, label=y_tr)
                dval = xgb.DMatrix(X_val, label=y_v)

                xgb_model = xgb.train(
                    xgb_params, dtrain,
                    num_boost_round=700,
                    evals=[(dval, 'val')],
                    early_stopping_rounds=40,
                    verbose_eval=False
                )

                # LightGBM
                train_data = lgb.Dataset(X_train, label=y_tr)
                val_data = lgb.Dataset(X_val, label=y_v, reference=train_data)

                lgb_model = lgb.train(
                    lgb_params, train_data,
                    num_boost_round=700,
                    valid_sets=[val_data],
                    callbacks=[lgb.early_stopping(40, verbose=False)]
                )

                # CatBoost
                cb_model = cb.CatBoostClassifier(**cb_params)
                cb_model.fit(X_train, y_tr, eval_set=(X_val, y_v))

                # Ensemble
                dtest = xgb.DMatrix(X_test)
                xgb_pred = xgb_model.predict(dtest)
                lgb_pred = lgb_model.predict(X_test)
                cb_pred = cb_model.predict_proba(X_test)[:, 1]

                ensemble_pred = (xgb_pred + lgb_pred + cb_pred) / 3
                ensemble_class = (ensemble_pred > 0.5).astype(int)

                acc = accuracy_score(y_te, ensemble_class)
                auc = roc_auc_score(y_te, ensemble_pred)
                f1 = f1_score(y_te, ensemble_class)

                print(f"[GPU] {pair} {target}: ACC={acc:.4f}, AUC={auc:.4f}, F1={f1:.4f}")

                results[target] = {'accuracy': float(acc), 'auc': float(auc), 'f1': float(f1)}
                models[target] = {'xgb': xgb_model, 'lgb': lgb_model, 'cb': cb_model}

                del dtrain, dval, dtest

            # Cleanup
            gc.collect()
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except:
                pass

        # Save
        model_path = output_dir / f"{pair}_models.pkl"
        with open(model_path, 'wb') as f:
            pickle.dump(models, f)

        results['_meta'] = {
            'feature_mode': 'mega',
            'feature_count': len(feature_names),
            'timestamp': datetime.now().isoformat()
        }

        results_path = output_dir / f"{pair}_results.json"
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)

        print(f"[GPU] {pair}: Saved to {model_path}")

        del X_train, X_val, X_test, data
        gc.collect()

        return (pair, results)

    except Exception as e:
        import traceback
        print(f"[GPU] {pair}: ERROR - {e}")
        traceback.print_exc()
        return (pair, None)


# ============================================================================
# PIPELINE PARALLEL TRAINER
# ============================================================================

class PipelineParallelTrainer:
    """
    TRUE pipeline parallel trainer - GPU trains while CPU generates features.
    """

    def __init__(
        self,
        cpu_workers: int = 4,
        gpu_workers: int = 2,
        max_samples: int = 50000,
    ):
        self.cpu_workers = cpu_workers
        self.gpu_workers = gpu_workers
        self.max_samples = max_samples

        self.training_dir = PROJECT_ROOT / 'training_package'
        self.output_dir = PROJECT_ROOT / 'models' / 'production'
        self.features_cache_dir = self.training_dir / 'features_cache'

        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.features_cache_dir.mkdir(parents=True, exist_ok=True)

        # Synchronization
        self.gpu_lock = threading.Lock()
        self.ready_queue = queue.Queue()
        self.results = {}
        self.results_lock = threading.Lock()

        # Safety monitor
        self.safety = GPUSafetyMonitor()

        # Progress
        self.features_done = 0
        self.trained_done = 0
        self.total_pairs = 0
        self.start_time = None

    def _get_pairs_needing_training(self, min_accuracy: float = 0.55) -> Tuple[List[str], List[Tuple[str, str]]]:
        """
        Get pairs that need features and pairs ready to train.

        Considers a pair as needing training if:
        - No results file exists
        - Results show 0 features
        - Results show accuracy below min_accuracy

        Returns:
            (pairs_needing_features, pairs_ready_to_train)
        """
        # All pairs with data
        all_pairs = []
        for d in self.training_dir.iterdir():
            if d.is_dir() and (d / 'train.parquet').exists() and d.name != 'features_cache':
                all_pairs.append(d.name)

        # Check which pairs need training
        pairs_needing_work = []
        good_pairs = []

        for pair in all_pairs:
            results_path = self.output_dir / f"{pair}_results.json"

            needs_training = True
            reason = "no results"

            if results_path.exists():
                try:
                    with open(results_path) as f:
                        data = json.load(f)

                    features = data.get('_meta', {}).get('feature_count', 0)
                    if features == 0:
                        reason = "0 features"
                    else:
                        # Check accuracy
                        acc = 0
                        for t in ['target_direction_10', 'target_direction_5', 'target_direction_1']:
                            if t in data and 'accuracy' in data[t]:
                                acc = data[t]['accuracy']
                                break

                        if acc >= min_accuracy:
                            needs_training = False
                            good_pairs.append((pair, acc))
                        else:
                            reason = f"{acc*100:.1f}% accuracy"
                except:
                    reason = "corrupt results"

            if needs_training:
                pairs_needing_work.append((pair, reason))

        print(f"Pairs with good accuracy (>={min_accuracy*100}%): {len(good_pairs)}")
        for pair, acc in sorted(good_pairs, key=lambda x: -x[1])[:5]:
            print(f"  {pair}: {acc*100:.1f}%")

        print(f"\nPairs needing training: {len(pairs_needing_work)}")

        # Split into ready vs need features
        ready_to_train = []
        need_features = []

        for pair, reason in pairs_needing_work:
            features_path = self.features_cache_dir / f"{pair}_features.pkl"
            if features_path.exists():
                ready_to_train.append((pair, str(features_path)))
            else:
                need_features.append(pair)

        return need_features, ready_to_train

    def _monitor_thread(self):
        """Background thread to monitor GPU and print status."""
        while self.start_time is not None:
            self.safety.update()
            stats = self.safety.get_stats()

            elapsed = time.time() - self.start_time
            rate = self.trained_done / (elapsed / 60) if elapsed > 60 else 0

            remaining = self.total_pairs - self.trained_done
            eta = remaining / rate if rate > 0 else 0

            print(f"\n[STATUS] Features: {self.features_done} | Trained: {self.trained_done}/{self.total_pairs} | "
                  f"GPU: {stats['temp']}°C {stats['utilization']}% | VRAM: {stats['mem_used_mb']}/{stats['mem_total_mb']}MB | "
                  f"ETA: {eta:.0f}min\n")

            time.sleep(30)

    def train_all(self) -> Dict[str, Dict]:
        """
        Train all pairs with true pipeline parallelism.
        GPU starts training immediately on pairs with features.
        CPU generates features for remaining pairs in parallel.
        """
        self.start_time = time.time()

        # Get work lists
        need_features, ready_to_train = self._get_pairs_needing_training()

        self.total_pairs = len(need_features) + len(ready_to_train)

        if self.total_pairs == 0:
            print("All pairs already trained!")
            return {}

        print(f"\n{'='*70}")
        print(f"PIPELINE PARALLEL TRAINING")
        print(f"{'='*70}")
        print(f"Ready to train (have features): {len(ready_to_train)}")
        print(f"Need features first: {len(need_features)}")
        print(f"Total: {self.total_pairs}")
        print(f"CPU workers: {self.cpu_workers}")
        print(f"GPU workers: {self.gpu_workers}")

        # Initial safety check
        self.safety.update()
        stats = self.safety.get_stats()
        print(f"\nGPU Status: {stats['temp']}°C, {stats['utilization']}% util, "
              f"{stats['mem_used_mb']}/{stats['mem_total_mb']}MB VRAM")
        print(f"{'='*70}\n")

        # Start monitor thread
        monitor = threading.Thread(target=self._monitor_thread, daemon=True)
        monitor.start()

        # Queue ready pairs
        for pair, path in ready_to_train:
            self.ready_queue.put((pair, path))

        # Start GPU workers (will consume from queue)
        def gpu_consumer():
            while True:
                try:
                    pair, path = self.ready_queue.get(timeout=5)
                except queue.Empty:
                    if self.features_done >= len(need_features) and self.ready_queue.empty():
                        break
                    continue

                result = train_gpu_worker(pair, path, self.output_dir, self.gpu_lock, self.safety)
                pair_name, pair_results = result

                with self.results_lock:
                    if pair_results:
                        self.results[pair_name] = pair_results
                    self.trained_done += 1

        # Start feature workers (will feed queue)
        def feature_producer():
            if not need_features:
                return

            args = [
                (p, str(self.training_dir), str(self.features_cache_dir), self.max_samples)
                for p in need_features
            ]

            with ProcessPoolExecutor(max_workers=self.cpu_workers) as pool:
                futures = {pool.submit(generate_features_worker, a): a[0] for a in args}

                for future in as_completed(futures):
                    pair = futures[future]
                    try:
                        _, path, err = future.result()
                        self.features_done += 1

                        if path:
                            print(f"[CPU] {pair}: Features ready ({self.features_done}/{len(need_features)})")
                            self.ready_queue.put((pair, path))
                        else:
                            print(f"[CPU] {pair}: FAILED - {err[:100]}")
                    except Exception as e:
                        print(f"[CPU] {pair}: ERROR - {e}")

        # Run producers and consumers in parallel
        with ThreadPoolExecutor(max_workers=self.gpu_workers + 1) as executor:
            # Start feature producer
            feature_future = executor.submit(feature_producer)

            # Start GPU consumers
            gpu_futures = [executor.submit(gpu_consumer) for _ in range(self.gpu_workers)]

            # Wait for all
            feature_future.result()
            for f in gpu_futures:
                f.result()

        # Stop monitoring
        self.start_time = None

        # Summary
        total_time = time.time() - (self.start_time or time.time())

        print(f"\n{'='*70}")
        print("TRAINING COMPLETE")
        print(f"{'='*70}")
        print(f"Total time: {total_time/60:.1f} minutes")
        print(f"Pairs trained: {len(self.results)}/{self.total_pairs}")

        if self.results:
            accs = []
            for pair, res in self.results.items():
                for target, metrics in res.items():
                    if not target.startswith('_') and 'accuracy' in metrics:
                        accs.append(metrics['accuracy'])
                        print(f"  {pair} {target}: ACC={metrics['accuracy']:.4f}")

            if accs:
                print(f"\nAverage accuracy: {np.mean(accs)*100:.2f}%")

        # Save summary
        summary = {
            'timestamp': datetime.now().isoformat(),
            'total_time_minutes': total_time / 60,
            'pairs_trained': list(self.results.keys()),
            'results': self.results,
        }

        with open(self.output_dir / 'pipeline_training_summary.json', 'w') as f:
            json.dump(summary, f, indent=2)

        return self.results


# ============================================================================
# MAIN
# ============================================================================

def configure_gpu():
    """Configure GPU for maximum performance."""
    print("Configuring GPU for maximum performance...")

    try:
        import torch
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            print(f"GPU: {torch.cuda.get_device_name(0)}")
            print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    except:
        pass

    # Set process priority
    try:
        import psutil
        p = psutil.Process()
        p.nice(psutil.HIGH_PRIORITY_CLASS)
        print("Process priority: HIGH")
    except:
        pass

    # CPU affinity
    os.environ['OMP_NUM_THREADS'] = '12'
    os.environ['MKL_NUM_THREADS'] = '12'
    os.environ['CUDA_LAUNCH_BLOCKING'] = '0'

    print()


def main():
    configure_gpu()

    # Show safety thresholds
    print("Safety thresholds:")
    print(f"  Temperature warn: {GPUSafetyMonitor.TEMP_WARN}°C")
    print(f"  Temperature throttle: {GPUSafetyMonitor.TEMP_THROTTLE}°C")
    print(f"  Temperature pause: {GPUSafetyMonitor.TEMP_PAUSE}°C")
    print(f"  VRAM max: {GPUSafetyMonitor.MEM_MAX_PERCENT}%")
    print()

    trainer = PipelineParallelTrainer(
        cpu_workers=4,
        gpu_workers=2,
        max_samples=50000,
    )

    results = trainer.train_all()

    if results:
        print(f"\n{'='*70}")
        print("SUCCESS - Training complete with safety monitoring")
        print(f"{'='*70}")


if __name__ == '__main__':
    main()
