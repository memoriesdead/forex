"""
GPU Configuration for Maximum Performance on RTX 5080

System: AMD Ryzen 9 9900X (12c/24t) + RTX 5080 16GB + 32GB DDR5-4800
Target: 80-95% GPU utilization, minimize training time
"""

import os
import sys

# ============================================================================
# CUDA / GPU Environment Variables (set before any imports)
# ============================================================================

# XGBoost GPU optimization
os.environ['XGBOOST_GPU_ID'] = '0'

# LightGBM GPU
os.environ['LIGHTGBM_GPU_ID'] = '0'

# CUDA optimization - async operations
os.environ['CUDA_LAUNCH_BLOCKING'] = '0'  # Async GPU ops (faster)
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'  # Consistent GPU ordering

# PyTorch memory optimization
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True,garbage_collection_threshold:0.8'

# Disable debug modes for production speed
os.environ['CUDA_DEBUG'] = '0'
os.environ['TORCH_SHOW_CPP_STACKTRACES'] = '0'

# ============================================================================
# CPU Environment Variables (Ryzen 9 9900X optimization)
# ============================================================================

# OpenMP thread settings - use physical cores only (12 cores)
os.environ['OMP_NUM_THREADS'] = '12'  # Physical cores only
os.environ['OMP_SCHEDULE'] = 'dynamic'
os.environ['OMP_PROC_BIND'] = 'close'  # Bind threads to cores
os.environ['OMP_WAIT_POLICY'] = 'active'  # Keep threads active

# MKL (Intel Math Kernel Library) - often used by NumPy
os.environ['MKL_NUM_THREADS'] = '12'
os.environ['MKL_DYNAMIC'] = 'FALSE'

# OpenBLAS (alternative to MKL)
os.environ['OPENBLAS_NUM_THREADS'] = '12'

# NumExpr
os.environ['NUMEXPR_NUM_THREADS'] = '12'
os.environ['NUMEXPR_MAX_THREADS'] = '12'

# Avoid thread oversubscription
os.environ['VECLIB_MAXIMUM_THREADS'] = '12'

# ============================================================================
# XGBoost / LightGBM specific
# ============================================================================
os.environ['NCCL_DEBUG'] = 'WARN'
os.environ['NCCL_IB_DISABLE'] = '1'

def configure_gpu():
    """Configure GPU for maximum ML performance."""
    import torch

    if torch.cuda.is_available():
        # Enable cuDNN auto-tuner for best convolution algorithms
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.enabled = True

        # Enable TF32 for faster matrix ops on Ampere+ GPUs
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

        # Set default device
        torch.cuda.set_device(0)

        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        print(f"cuDNN benchmark: {torch.backends.cudnn.benchmark}")
        print(f"TF32 enabled: {torch.backends.cuda.matmul.allow_tf32}")

        return True
    return False


def get_xgb_gpu_params(aggressive: bool = True):
    """XGBoost params optimized for RTX 5080 16GB.

    Note: XGBoost 2.0+ changed GPU usage:
    - tree_method: 'hist' (not 'gpu_hist')
    - device: 'cuda' enables GPU

    Args:
        aggressive: If True, use max settings for speed (default)
                   If False, use conservative settings for stability
    """
    if aggressive:
        return {
            'tree_method': 'hist',
            'device': 'cuda',
            'max_depth': 14,            # Deeper for 16GB VRAM
            'max_bin': 512,             # More bins = better GPU util
            'learning_rate': 0.03,      # Lower LR, more trees = better
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'colsample_bylevel': 0.8,
            'min_child_weight': 3,
            'gamma': 0.1,
            'reg_alpha': 0.1,
            'reg_lambda': 1.0,
            'n_jobs': 12,               # Physical cores only
            'n_estimators': 2000,       # More trees
            'early_stopping_rounds': 100,
        }
    else:
        return {
            'tree_method': 'hist',
            'device': 'cuda',
            'max_depth': 12,
            'max_bin': 256,
            'learning_rate': 0.05,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'n_jobs': 12,
        }


def get_lgb_gpu_params(aggressive: bool = True):
    """LightGBM params optimized for RTX 5080 16GB.

    Args:
        aggressive: If True, use max settings for speed (default)
    """
    if aggressive:
        return {
            'device': 'gpu',
            'gpu_platform_id': 0,
            'gpu_device_id': 0,
            'max_depth': 14,
            'learning_rate': 0.03,
            'num_leaves': 2047,         # 2^11 - 1 (limited by LightGBM)
            'min_data_in_leaf': 20,
            'histogram_pool_size': 2048,  # More GPU memory for histograms
            'gpu_use_dp': False,        # FP32 for speed
            'feature_fraction': 0.8,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'lambda_l1': 0.1,
            'lambda_l2': 1.0,
            'n_estimators': 2000,
            'n_jobs': 12,               # Physical cores only
            'verbose': -1,
            'early_stopping_round': 100,
        }
    else:
        return {
            'device': 'gpu',
            'gpu_platform_id': 0,
            'gpu_device_id': 0,
            'max_depth': 12,
            'learning_rate': 0.05,
            'num_leaves': 511,
            'histogram_pool_size': 1024,
            'gpu_use_dp': False,
            'n_jobs': 12,
            'verbose': -1,
        }


def get_catboost_gpu_params(aggressive: bool = True):
    """CatBoost params optimized for RTX 5080 16GB.

    Args:
        aggressive: If True, use max settings for speed (default)
    """
    if aggressive:
        return {
            'task_type': 'GPU',
            'devices': '0',
            'depth': 12,                # Deeper for 16GB VRAM
            'border_count': 64,         # More bins
            'learning_rate': 0.03,
            'iterations': 2000,
            'boosting_type': 'Plain',   # 2-3x faster than Ordered
            'l2_leaf_reg': 3.0,
            'random_strength': 1.0,
            'bagging_temperature': 0.5,
            'grow_policy': 'SymmetricTree',
            'thread_count': 12,         # Physical cores only
            'early_stopping_rounds': 100,
            'verbose': 100,
        }
    else:
        return {
            'task_type': 'GPU',
            'devices': '0',
            'depth': 10,
            'border_count': 32,
            'learning_rate': 0.05,
            'iterations': 1000,
            'boosting_type': 'Plain',
            'thread_count': 12,
        }


# ============================================================================
# CONCURRENT TRAINING PARAMS (for 3 GPU jobs simultaneously)
# ============================================================================
# These params are optimized for running 3 training jobs on RTX 5080
# Memory budget: ~5GB per job = 15GB total (leaves 1GB headroom)

def get_xgb_concurrent_params():
    """XGBoost params for concurrent training (3 jobs on RTX 5080).

    Reduced depth and bins to fit 3 models in 16GB VRAM.
    """
    return {
        'tree_method': 'hist',
        'device': 'cuda',
        'max_depth': 8,             # Reduced from 14
        'max_bin': 128,             # Reduced from 512
        'learning_rate': 0.05,
        'subsample': 0.7,
        'colsample_bytree': 0.6,
        'objective': 'binary:logistic',
        'eval_metric': 'auc',
        'nthread': 4,               # 12 cores / 3 jobs
    }


def get_lgb_concurrent_params():
    """LightGBM params for concurrent training (3 jobs on RTX 5080).

    Reduced depth and leaves to fit 3 models in 16GB VRAM.
    """
    return {
        'device': 'gpu',
        'gpu_platform_id': 0,
        'gpu_device_id': 0,
        'max_depth': 8,             # Reduced from 14
        'num_leaves': 127,          # Reduced from 2047
        'feature_fraction': 0.6,
        'bagging_fraction': 0.7,
        'bagging_freq': 5,
        'verbose': -1,
        'objective': 'binary',
        'metric': 'auc',
        'n_jobs': 4,                # 12 cores / 3 jobs
        'histogram_pool_size': 512, # Reduced from 2048
    }


def get_catboost_concurrent_params():
    """CatBoost params for concurrent training (3 jobs on RTX 5080).

    Reduced depth and iterations to fit 3 models in 16GB VRAM.
    """
    return {
        'task_type': 'GPU',
        'devices': '0',
        'depth': 6,                 # Reduced from 12
        'border_count': 32,         # Reduced from 64
        'iterations': 300,          # Reduced from 2000
        'early_stopping_rounds': 20,
        'verbose': False,
        'loss_function': 'Logloss',
        'thread_count': 4,          # 12 cores / 3 jobs
    }


def get_concurrent_training_config():
    """Get full config for parallel training.

    Returns dict with recommended settings for parallel training.
    """
    return {
        'cpu_workers': 4,           # Feature generation processes
        'gpu_workers': 3,           # Concurrent GPU training jobs
        'max_samples': 50000,       # Max samples per pair
        'vram_per_job_gb': 5,       # ~5GB VRAM per model set
        'total_vram_gb': 16,        # RTX 5080
        'xgb': get_xgb_concurrent_params(),
        'lgb': get_lgb_concurrent_params(),
        'cb': get_catboost_concurrent_params(),
    }


def set_high_priority():
    """Set current process to high priority for ML training."""
    import psutil
    try:
        p = psutil.Process()
        p.nice(psutil.HIGH_PRIORITY_CLASS)  # Windows high priority
        print(f"Process priority set to HIGH")
        return True
    except Exception as e:
        print(f"Could not set high priority: {e}")
        return False


def set_cpu_affinity(physical_cores_only: bool = True):
    """Set CPU affinity to physical cores only (avoids hyperthreading overhead).

    Args:
        physical_cores_only: If True, use only cores 0-11 (physical)
                            If False, use all 24 logical processors
    """
    import psutil
    try:
        p = psutil.Process()
        if physical_cores_only:
            # Use first 12 cores (physical cores on Ryzen 9 9900X)
            p.cpu_affinity(list(range(12)))
            print(f"CPU affinity set to physical cores 0-11")
        else:
            p.cpu_affinity(list(range(24)))
            print(f"CPU affinity set to all 24 logical processors")
        return True
    except Exception as e:
        print(f"Could not set CPU affinity: {e}")
        return False


def optimize_for_training():
    """One-call function to optimize everything for ML training."""
    print("=" * 60)
    print("OPTIMIZING FOR ML TRAINING")
    print("=" * 60)

    # Configure GPU
    gpu_ok = configure_gpu()

    # Set process priority
    priority_ok = set_high_priority()

    # Set CPU affinity to physical cores
    affinity_ok = set_cpu_affinity(physical_cores_only=True)

    # Verify settings
    print("\n--- Environment Variables ---")
    print(f"OMP_NUM_THREADS: {os.environ.get('OMP_NUM_THREADS', 'not set')}")
    print(f"MKL_NUM_THREADS: {os.environ.get('MKL_NUM_THREADS', 'not set')}")
    print(f"CUDA_LAUNCH_BLOCKING: {os.environ.get('CUDA_LAUNCH_BLOCKING', 'not set')}")

    print("\n--- Status ---")
    print(f"GPU configured: {gpu_ok}")
    print(f"High priority: {priority_ok}")
    print(f"CPU affinity: {affinity_ok}")
    print("=" * 60)

    return gpu_ok and priority_ok and affinity_ok


# ============================================================================
# NEURAL NETWORK PARAMS (TabNet, CNN, MLP for RTX 5080 16GB)
# ============================================================================

def get_tabnet_gpu_params(aggressive: bool = True):
    """TabNet params optimized for RTX 5080 16GB.

    TabNet uses attention mechanism for feature selection.
    Good for interpretability + competitive with GBMs.

    Args:
        aggressive: If True, use larger architecture
    """
    if aggressive:
        return {
            'n_d': 64,                  # Width of decision step
            'n_a': 64,                  # Width of attention step
            'n_steps': 5,               # Number of decision steps
            'gamma': 1.5,               # Feature reusage coefficient
            'n_independent': 2,         # Independent GLU layers
            'n_shared': 2,              # Shared GLU layers
            'momentum': 0.02,
            'lambda_sparse': 1e-4,      # Sparsity regularization
            'mask_type': 'sparsemax',   # or 'entmax'
            'optimizer_params': {
                'lr': 2e-2,
                'weight_decay': 1e-5,
            },
            'scheduler_params': {
                'step_size': 10,
                'gamma': 0.9,
            },
            'device_name': 'cuda',
            'batch_size': 1024,         # Large batch for GPU
            'virtual_batch_size': 256,  # For ghost batch norm
            'max_epochs': 100,
            'patience': 15,
        }
    else:
        return {
            'n_d': 32,
            'n_a': 32,
            'n_steps': 3,
            'gamma': 1.3,
            'n_independent': 1,
            'n_shared': 1,
            'momentum': 0.02,
            'device_name': 'cuda',
            'batch_size': 512,
            'virtual_batch_size': 128,
            'max_epochs': 50,
            'patience': 10,
        }


def get_cnn_gpu_params(aggressive: bool = True):
    """1D-CNN params optimized for RTX 5080 16GB.

    Treats features as 1D signal, captures local patterns.
    Good for detecting feature interactions.

    Args:
        aggressive: If True, use deeper architecture
    """
    if aggressive:
        return {
            'conv_layers': [
                {'out_channels': 64, 'kernel_size': 3, 'stride': 1, 'padding': 1},
                {'out_channels': 128, 'kernel_size': 3, 'stride': 1, 'padding': 1},
                {'out_channels': 256, 'kernel_size': 3, 'stride': 1, 'padding': 1},
            ],
            'fc_layers': [512, 256, 128],
            'dropout': 0.3,
            'batch_norm': True,
            'activation': 'relu',
            'pool_size': 2,
            'optimizer': 'adamw',
            'learning_rate': 1e-3,
            'weight_decay': 1e-4,
            'batch_size': 512,
            'max_epochs': 100,
            'patience': 15,
            'device': 'cuda',
            'num_workers': 4,
            'pin_memory': True,
            'cudnn_benchmark': True,
        }
    else:
        return {
            'conv_layers': [
                {'out_channels': 32, 'kernel_size': 3, 'stride': 1, 'padding': 1},
                {'out_channels': 64, 'kernel_size': 3, 'stride': 1, 'padding': 1},
            ],
            'fc_layers': [256, 128],
            'dropout': 0.2,
            'batch_norm': True,
            'activation': 'relu',
            'learning_rate': 1e-3,
            'batch_size': 256,
            'max_epochs': 50,
            'device': 'cuda',
        }


def get_mlp_gpu_params(aggressive: bool = True):
    """MLP (Multi-Layer Perceptron) params for RTX 5080 16GB.

    Deep fully-connected network with batch normalization.
    Good baseline for tabular data.

    Args:
        aggressive: If True, use deeper/wider architecture
    """
    if aggressive:
        return {
            'hidden_layers': [512, 256, 128, 64],
            'dropout': 0.3,
            'batch_norm': True,
            'activation': 'relu',
            'use_residual': True,       # Skip connections
            'optimizer': 'adamw',
            'learning_rate': 1e-3,
            'weight_decay': 1e-4,
            'scheduler': 'cosine',
            'warmup_epochs': 5,
            'batch_size': 1024,
            'max_epochs': 100,
            'patience': 15,
            'device': 'cuda',
            'num_workers': 4,
            'pin_memory': True,
            'mixed_precision': True,    # FP16 for speed
        }
    else:
        return {
            'hidden_layers': [256, 128, 64],
            'dropout': 0.2,
            'batch_norm': True,
            'activation': 'relu',
            'learning_rate': 1e-3,
            'batch_size': 512,
            'max_epochs': 50,
            'device': 'cuda',
        }


def get_transformer_gpu_params(aggressive: bool = True):
    """Transformer params for time series (iTransformer style).

    For sequence modeling of OHLCV data.
    Best for capturing temporal dependencies.

    Args:
        aggressive: If True, use larger model
    """
    if aggressive:
        return {
            'd_model': 256,             # Embedding dimension
            'n_heads': 8,               # Attention heads
            'n_layers': 4,              # Encoder layers
            'd_ff': 1024,               # FFN hidden dim
            'dropout': 0.1,
            'seq_len': 96,              # Input sequence length
            'pred_len': 1,              # Prediction length
            'attention_type': 'full',   # or 'sparse', 'linear'
            'optimizer': 'adamw',
            'learning_rate': 1e-4,
            'weight_decay': 1e-2,
            'scheduler': 'cosine',
            'warmup_steps': 1000,
            'batch_size': 64,           # Smaller for transformers
            'max_epochs': 100,
            'patience': 15,
            'device': 'cuda',
            'mixed_precision': True,
            'gradient_checkpointing': False,
        }
    else:
        return {
            'd_model': 128,
            'n_heads': 4,
            'n_layers': 2,
            'd_ff': 512,
            'dropout': 0.1,
            'seq_len': 48,
            'learning_rate': 1e-4,
            'batch_size': 32,
            'max_epochs': 50,
            'device': 'cuda',
        }


def get_neural_concurrent_params():
    """Neural network params for concurrent training with GBMs.

    Reduced model sizes to fit alongside XGB/LGB/CB training.
    Memory budget: ~2GB for neural nets (vs 5GB each for GBMs).
    """
    return {
        'tabnet': {
            'n_d': 32,
            'n_a': 32,
            'n_steps': 3,
            'batch_size': 512,
            'device_name': 'cuda',
        },
        'cnn': {
            'conv_layers': [
                {'out_channels': 32, 'kernel_size': 3, 'stride': 1, 'padding': 1},
            ],
            'fc_layers': [128],
            'batch_size': 256,
            'device': 'cuda',
        },
        'mlp': {
            'hidden_layers': [128, 64],
            'batch_size': 512,
            'device': 'cuda',
        },
    }


def get_stacking_training_config():
    """Get full config for 6-model stacking ensemble.

    Memory budget for RTX 5080 16GB:
    - XGBoost: ~3GB
    - LightGBM: ~2GB
    - CatBoost: ~3GB
    - TabNet: ~1.5GB
    - CNN: ~0.5GB
    - MLP: ~0.5GB
    - Total: ~10.5GB (leaves 5.5GB headroom)
    """
    return {
        'xgb': get_xgb_gpu_params(aggressive=False),
        'lgb': get_lgb_gpu_params(aggressive=False),
        'catboost': get_catboost_gpu_params(aggressive=False),
        'tabnet': get_tabnet_gpu_params(aggressive=False),
        'cnn': get_cnn_gpu_params(aggressive=False),
        'mlp': get_mlp_gpu_params(aggressive=False),
        'meta_learner': {
            'type': 'xgboost',
            **get_xgb_gpu_params(aggressive=False),
        },
        'total_vram_budget_gb': 12.0,
        'cpu_workers': 4,
    }


# ============================================================================
# MAX PARALLEL CONFIGURATION (RTX 5080 16GB + Ryzen 9 9900X at 100% utilization)
# ============================================================================

def get_max_parallel_config():
    """
    Config for MAXIMUM system utilization.

    Target:
    - GPU: 85-95% utilization, 280-320W power draw
    - CPU: 75-85% utilization (all 12 cores)
    - RAM: 20-25 GB used (from 32GB available)

    Architecture:
    - 3 GPU training jobs in parallel (5GB VRAM each)
    - 4 CPU feature workers (3 threads each = 12 cores)
    - Pipeline parallelism: Stage 1 (CPU) feeds Stage 2 (GPU)

    Memory Budget (16GB VRAM):
    - Job 1: XGBoost ~3GB + LightGBM ~2GB = 5GB
    - Job 2: CatBoost ~3GB + buffer = 4GB
    - Job 3: TabNet ~1.5GB + CNN ~0.5GB + MLP ~0.5GB + buffer = 4GB
    - Overhead: ~3GB
    - Total: ~16GB
    """
    return {
        # System utilization targets
        'gpu_target_utilization': 0.90,  # 90%
        'cpu_target_utilization': 0.80,  # 80%
        'ram_target_gb': 22,              # 22 GB of 32 GB

        # Parallel workers
        'gpu_workers': 3,                 # 3 concurrent GPU training jobs
        'cpu_workers': 4,                 # 4 feature generation processes
        'threads_per_cpu_worker': 3,      # 4 * 3 = 12 cores total

        # Data loading
        'batch_size': 4096,               # Large batch for GPU efficiency
        'max_samples': 80000,             # More samples (32GB RAM allows it)
        'prefetch_pairs': 2,              # Prefetch next 2 pairs

        # XGBoost concurrent params (3 jobs fit in 16GB VRAM)
        'xgb': {
            'tree_method': 'hist',        # XGBoost 2.0+
            'device': 'cuda',             # GPU acceleration
            'max_depth': 10,              # Balanced for concurrent training
            'max_bin': 256,               # Good GPU efficiency
            'n_estimators': 1200,         # More trees for accuracy
            'learning_rate': 0.03,        # Lower LR, more trees
            'subsample': 0.8,
            'colsample_bytree': 0.7,
            'colsample_bylevel': 0.8,
            'min_child_weight': 3,
            'gamma': 0.1,
            'reg_alpha': 0.1,
            'reg_lambda': 1.0,
            'nthread': 4,                 # 12 cores / 3 jobs
            'early_stopping_rounds': 50,
            'objective': 'binary:logistic',
            'eval_metric': 'auc',
        },

        # LightGBM concurrent params
        'lgb': {
            'device': 'gpu',
            'gpu_platform_id': 0,
            'gpu_device_id': 0,
            'max_depth': 10,              # Balanced depth
            'num_leaves': 512,            # 2^9 (moderate for concurrency)
            'n_estimators': 1200,
            'learning_rate': 0.03,
            'feature_fraction': 0.7,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'lambda_l1': 0.1,
            'lambda_l2': 1.0,
            'n_jobs': 4,                  # 12 cores / 3 jobs
            'histogram_pool_size': 512,   # Reduced for concurrent GPU
            'verbose': -1,
            'objective': 'binary',
            'metric': 'auc',
        },

        # CatBoost concurrent params
        'cb': {
            'task_type': 'GPU',
            'devices': '0',
            'depth': 8,                   # Reduced for concurrent training
            'border_count': 32,
            'iterations': 1200,
            'learning_rate': 0.03,
            'boosting_type': 'Plain',     # Faster than Ordered
            'l2_leaf_reg': 3.0,
            'random_strength': 1.0,
            'bagging_temperature': 0.5,
            'thread_count': 4,            # 12 cores / 3 jobs
            'early_stopping_rounds': 50,
            'verbose': False,             # Suppress output for cleaner logs
        },

        # Neural nets concurrent (smaller for concurrent training)
        'tabnet': {
            'n_d': 48,                    # Reduced from 64
            'n_a': 48,
            'n_steps': 4,                 # Reduced from 5
            'gamma': 1.5,
            'n_independent': 2,
            'n_shared': 2,
            'lambda_sparse': 1e-4,
            'batch_size': 2048,           # Larger batch for GPU
            'virtual_batch_size': 256,
            'max_epochs': 80,
            'patience': 10,
        },

        'cnn': {
            'conv_layers': [48, 96, 192], # Reduced from [64, 128, 256]
            'kernel_size': 3,
            'dropout': 0.3,
            'fc_layers': [192, 96],
            'max_epochs': 40,
            'batch_size': 2048,
            'learning_rate': 0.001,
        },

        'mlp': {
            'hidden_layers': [384, 192, 96, 48],  # Reduced
            'dropout': 0.3,
            'batch_norm': True,
            'max_epochs': 40,
            'batch_size': 2048,
            'learning_rate': 0.001,
        },

        # Meta-learner for stacking
        'meta_learner': {
            'tree_method': 'hist',
            'device': 'cuda',
            'max_depth': 6,
            'learning_rate': 0.05,
            'n_estimators': 400,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
        },

        # Training modes
        'mode': {
            # Quick mode: XGB + LGB + CB only (fastest)
            'quick': {
                'enable_tabnet': False,
                'enable_cnn': False,
                'enable_mlp': False,
                'n_estimators': 300,  # Reduced for faster iteration
            },
            # Standard mode: All 6 models
            'standard': {
                'enable_tabnet': True,
                'enable_cnn': True,
                'enable_mlp': True,
                'n_estimators': 1200,
            },
            # Full mode: More trees/epochs (best accuracy)
            'full': {
                'enable_tabnet': True,
                'enable_cnn': True,
                'enable_mlp': True,
                'n_estimators': 2000,
                'neural_epochs': 100,
            },
        },
    }


def verify_max_parallel_ready() -> dict:
    """
    Verify system is ready for max parallel training.

    Returns dict with status of each component.
    """
    import torch
    import psutil

    results = {}

    # GPU check
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        vram_gb = props.total_memory / 1024**3
        results['gpu'] = {
            'available': True,
            'name': torch.cuda.get_device_name(0),
            'vram_gb': vram_gb,
            'sufficient': vram_gb >= 14,  # Need 14+ GB for max parallel
        }
    else:
        results['gpu'] = {'available': False, 'sufficient': False}

    # CPU check
    results['cpu'] = {
        'physical_cores': psutil.cpu_count(logical=False),
        'logical_cores': psutil.cpu_count(logical=True),
        'sufficient': psutil.cpu_count(logical=False) >= 8,
    }

    # RAM check
    mem = psutil.virtual_memory()
    results['ram'] = {
        'total_gb': mem.total / 1024**3,
        'available_gb': mem.available / 1024**3,
        'sufficient': mem.total / 1024**3 >= 24,  # Need 24+ GB
    }

    # Overall
    results['ready'] = all([
        results['gpu'].get('sufficient', False),
        results['cpu'].get('sufficient', False),
        results['ram'].get('sufficient', False),
    ])

    return results


def print_system_info():
    """Print current system configuration."""
    import torch
    import psutil

    print("=" * 60)
    print("SYSTEM CONFIGURATION")
    print("=" * 60)

    # CPU
    print(f"\n--- CPU ---")
    print(f"Physical cores: {psutil.cpu_count(logical=False)}")
    print(f"Logical cores: {psutil.cpu_count(logical=True)}")
    print(f"Current frequency: {psutil.cpu_freq().current:.0f} MHz")

    # Memory
    mem = psutil.virtual_memory()
    print(f"\n--- Memory ---")
    print(f"Total: {mem.total / 1024**3:.1f} GB")
    print(f"Available: {mem.available / 1024**3:.1f} GB")
    print(f"Used: {mem.percent}%")

    # GPU
    if torch.cuda.is_available():
        print(f"\n--- GPU ---")
        print(f"Device: {torch.cuda.get_device_name(0)}")
        props = torch.cuda.get_device_properties(0)
        print(f"VRAM: {props.total_memory / 1024**3:.1f} GB")
        print(f"Compute capability: {props.major}.{props.minor}")
        print(f"SM count: {props.multi_processor_count}")

        # Current memory usage
        allocated = torch.cuda.memory_allocated(0) / 1024**3
        reserved = torch.cuda.memory_reserved(0) / 1024**3
        print(f"Memory allocated: {allocated:.2f} GB")
        print(f"Memory reserved: {reserved:.2f} GB")

    print("=" * 60)


if __name__ == "__main__":
    print_system_info()
    print()
    optimize_for_training()
