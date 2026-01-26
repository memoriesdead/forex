"""
MAXIMUM GPU CONFIGURATION for RTX 5080 16GB
============================================

This config pushes the RTX 5080 to 95%+ utilization for:
1. LLM Inference (Ollama)
2. LoRA Training
3. ML Ensemble (XGBoost + LightGBM + CatBoost)

Target: $3000 GPU at 95%+ utilization, 280-350W power draw

System: AMD Ryzen 9 9900X (12c/24t) + RTX 5080 16GB + 32GB DDR5
"""

import os
import subprocess

# ============================================================================
# MAXIMUM GPU ENVIRONMENT (set before ANY imports)
# ============================================================================

# CUDA optimization for max performance
os.environ['CUDA_LAUNCH_BLOCKING'] = '0'  # Async GPU ops
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# PyTorch max performance
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True,garbage_collection_threshold:0.9,max_split_size_mb:512'
os.environ['TORCH_CUDNN_V8_API_ENABLED'] = '1'

# CPU threads (Ryzen 9 9900X = 12 physical cores)
os.environ['OMP_NUM_THREADS'] = '12'
os.environ['MKL_NUM_THREADS'] = '12'
os.environ['OPENBLAS_NUM_THREADS'] = '12'
os.environ['NUMEXPR_NUM_THREADS'] = '12'
os.environ['NUMEXPR_MAX_THREADS'] = '24'  # Use hyperthreading

# Max threads for parallel
os.environ['OMP_SCHEDULE'] = 'dynamic'
os.environ['OMP_PROC_BIND'] = 'close'
os.environ['OMP_WAIT_POLICY'] = 'active'

# Disable debugging for speed
os.environ['CUDA_DEBUG'] = '0'
os.environ['TORCH_SHOW_CPP_STACKTRACES'] = '0'
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'


def configure_max_gpu():
    """Configure GPU for MAXIMUM performance."""
    try:
        import torch

        if not torch.cuda.is_available():
            print("[MAX_GPU] CUDA not available!")
            return False

        # Enable max speed settings
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.deterministic = False  # Faster

        # Enable TF32 for max speed on Ampere+ GPUs
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

        # Enable Flash Attention if available
        torch.backends.cuda.enable_flash_sdp(True)
        torch.backends.cuda.enable_mem_efficient_sdp(True)

        # Prefer fastest algorithms
        torch.backends.cuda.preferred_linalg_library('cusolver')

        # Set device
        torch.cuda.set_device(0)

        # Print config
        props = torch.cuda.get_device_properties(0)
        print(f"[MAX_GPU] Device: {torch.cuda.get_device_name(0)}")
        print(f"[MAX_GPU] VRAM: {props.total_memory / 1024**3:.1f} GB")
        print(f"[MAX_GPU] SM Count: {props.multi_processor_count}")
        print(f"[MAX_GPU] Compute: {props.major}.{props.minor}")
        print(f"[MAX_GPU] cuDNN benchmark: ON")
        print(f"[MAX_GPU] TF32: ON")
        print(f"[MAX_GPU] Flash Attention: ON")

        return True

    except Exception as e:
        print(f"[MAX_GPU] Error: {e}")
        return False


def get_max_ollama_params():
    """
    Maximum Ollama settings for forex-r1-v3 on RTX 5080 16GB.

    These settings push GPU to 90%+ during inference.
    """
    return {
        'num_gpu': 99,          # Use all GPU layers (max)
        'num_ctx': 8192,        # Maximum context window
        'num_batch': 1024,      # Larger batch = faster generation
        'num_predict': 4096,    # Allow long responses
        'num_thread': 12,       # CPU threads for tokenization
        'low_vram': False,      # Don't use low VRAM mode
        'f16_kv': True,         # FP16 key-value cache (saves VRAM)
        'use_mmap': True,       # Memory map model
        'use_mlock': True,      # Lock model in RAM
        'temperature': 0.3,     # Low temp for consistent outputs
        'top_p': 0.9,
        'repeat_penalty': 1.1,
    }


def get_max_lora_training_params():
    """
    Maximum LoRA training params for RTX 5080 16GB.

    Optimized for fastest training while using all VRAM.
    """
    return {
        # LoRA config
        'r': 128,                    # Higher rank = more capacity
        'lora_alpha': 256,           # alpha = 2 * r
        'lora_dropout': 0.05,
        'target_modules': [
            'q_proj', 'k_proj', 'v_proj', 'o_proj',
            'gate_proj', 'up_proj', 'down_proj',
        ],

        # Training config (maximize GPU util)
        'per_device_train_batch_size': 8,    # Max batch for 16GB
        'gradient_accumulation_steps': 2,     # Effective batch = 16
        'gradient_checkpointing': True,       # Trade compute for VRAM
        'learning_rate': 1e-5,
        'warmup_ratio': 0.1,
        'num_train_epochs': 1,
        'max_steps': -1,

        # Precision for speed
        'bf16': True,
        'tf32': True,

        # Memory optimization
        'max_grad_norm': 1.0,
        'optim': 'paged_adamw_8bit',  # 8-bit optimizer saves VRAM

        # Flash attention
        'attn_implementation': 'flash_attention_2',

        # Sequence lengths
        'max_length': 2048,
        'max_prompt_length': 1024,

        # Logging
        'logging_steps': 10,
        'save_strategy': 'steps',
        'save_steps': 100,
    }


def get_max_xgb_params():
    """Maximum XGBoost params for RTX 5080 (1239 features, 100k samples)."""
    return {
        'tree_method': 'hist',
        'device': 'cuda',
        'max_depth': 10,              # Reduced to fit in 16GB VRAM
        'max_bin': 256,               # Reduced for memory
        'learning_rate': 0.03,        # Slightly higher to compensate
        'n_estimators': 2000,         # Reduced for memory
        'subsample': 0.8,
        'colsample_bytree': 0.7,      # Reduced for 1239 features
        'colsample_bylevel': 0.7,
        'min_child_weight': 5,        # Higher for stability
        'gamma': 0.1,
        'reg_alpha': 0.1,
        'reg_lambda': 1.0,
        'early_stopping_rounds': 100,
        'n_jobs': 12,                 # All physical cores
        'objective': 'binary:logistic',
        'eval_metric': 'auc',
        'grow_policy': 'depthwise',
    }


def get_max_lgb_params():
    """Maximum LightGBM params for RTX 5080."""
    return {
        'device': 'gpu',
        'gpu_platform_id': 0,
        'gpu_device_id': 0,
        'max_depth': 12,              # Reduced for GPU stability
        'num_leaves': 1023,           # 2^10 - 1 (stable with GPU)
        'learning_rate': 0.02,
        'n_estimators': 3000,
        'feature_fraction': 0.85,
        'bagging_fraction': 0.85,
        'bagging_freq': 1,
        'min_data_in_leaf': 50,       # Increased for stability
        'min_data_in_bin': 10,        # Prevent empty bins
        'lambda_l1': 0.05,
        'lambda_l2': 0.5,
        'histogram_pool_size': 2048,  # Reduced for stability
        'gpu_use_dp': False,          # FP32 for speed
        'early_stopping_rounds': 100,
        'n_jobs': 12,
        'verbose': -1,
        'objective': 'binary',
        'metric': 'auc',
    }


def get_max_catboost_params():
    """Maximum CatBoost params for RTX 5080 (stable with 1239 features)."""
    return {
        'task_type': 'GPU',
        'devices': '0',
        'depth': 8,                   # Reduced for memory stability
        'border_count': 128,          # Reduced for memory
        'learning_rate': 0.03,
        'iterations': 2000,           # Reduced for memory
        'boosting_type': 'Plain',     # Faster than Ordered
        'l2_leaf_reg': 3.0,
        'random_strength': 0.5,
        'bagging_temperature': 0.3,
        'grow_policy': 'SymmetricTree',
        'max_ctr_complexity': 2,      # Reduced for memory
        'thread_count': 12,
        'early_stopping_rounds': 100,
        'verbose': 100,
        'loss_function': 'Logloss',
        'eval_metric': 'AUC',
    }


def get_max_ensemble_config():
    """
    Full max ensemble config for RTX 5080 16GB.

    Memory budget (16GB):
    - XGBoost: ~4GB
    - LightGBM: ~3GB
    - CatBoost: ~4GB
    - Neural nets: ~3GB
    - Overhead: ~2GB
    """
    return {
        'gpu_utilization_target': 0.95,
        'cpu_utilization_target': 0.85,
        'ram_target_gb': 28,  # Use 28GB of 32GB RAM

        # Parallel training
        'enable_parallel_training': True,
        'max_concurrent_gpu_jobs': 2,  # 2 models training at once
        'cpu_workers': 6,              # Feature generation workers

        # Model configs
        'xgb': get_max_xgb_params(),
        'lgb': get_max_lgb_params(),
        'catboost': get_max_catboost_params(),

        # Data loading
        'batch_size': 8192,
        'max_samples': 200000,         # Use all available data
        'prefetch_batches': 4,

        # Training
        'n_folds': 5,
        'stratified': True,
        'shuffle': True,
    }


def set_windows_high_performance():
    """Set Windows to High Performance power plan."""
    try:
        # Activate high performance power plan
        subprocess.run(
            ['powercfg', '/setactive', '8c5e7fda-e8bf-4a96-9a85-a6e23a8c635c'],
            capture_output=True
        )
        print("[MAX_GPU] Windows High Performance power plan activated")

        # Set GPU to prefer maximum performance
        subprocess.run(
            ['powershell', '-Command',
             'Set-ItemProperty -Path "HKLM:\\SYSTEM\\CurrentControlSet\\Control\\Power\\PowerSettings\\54533251-82be-4824-96c1-47b60b740d00\\be337238-0d82-4146-a960-4f3749d470c7" -Name Attributes -Value 0'],
            capture_output=True
        )

        return True
    except Exception as e:
        print(f"[MAX_GPU] Power plan error: {e}")
        return False


def set_process_max_priority():
    """Set current process to maximum priority."""
    try:
        import psutil
        p = psutil.Process()
        p.nice(psutil.REALTIME_PRIORITY_CLASS)  # Maximum priority
        print("[MAX_GPU] Process priority: REALTIME")
        return True
    except Exception as e:
        try:
            import psutil
            p = psutil.Process()
            p.nice(psutil.HIGH_PRIORITY_CLASS)
            print("[MAX_GPU] Process priority: HIGH")
            return True
        except Exception as e2:
            print(f"[MAX_GPU] Priority error: {e2}")
            return False


def maximize_system():
    """
    One-call function to maximize EVERYTHING.

    Call this at the start of any training or trading script.
    """
    print("=" * 70)
    print("MAXIMIZING SYSTEM FOR RTX 5080 16GB + RYZEN 9 9900X")
    print("=" * 70)

    # Windows power settings
    set_windows_high_performance()

    # Process priority
    set_process_max_priority()

    # GPU configuration
    configure_max_gpu()

    # Verify GPU
    try:
        import torch
        if torch.cuda.is_available():
            # Warm up GPU
            print("[MAX_GPU] Warming up GPU...")
            dummy = torch.randn(1000, 1000, device='cuda')
            for _ in range(10):
                dummy = torch.matmul(dummy, dummy)
            del dummy
            torch.cuda.empty_cache()
            print("[MAX_GPU] GPU warmed up")
    except Exception as e:
        print(f"[MAX_GPU] GPU warmup error: {e}")

    print("=" * 70)
    print("SYSTEM MAXIMIZED - TARGET: 95% GPU UTILIZATION")
    print("=" * 70)


def print_gpu_status():
    """Print current GPU status."""
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw',
             '--format=csv,noheader,nounits'],
            capture_output=True,
            text=True
        )

        if result.returncode == 0:
            values = result.stdout.strip().split(',')
            print(f"\n[GPU STATUS]")
            print(f"  Utilization: {values[0].strip()}%")
            print(f"  Memory: {values[1].strip()} / {values[2].strip()} MB")
            print(f"  Temperature: {values[3].strip()}C")
            print(f"  Power: {values[4].strip()}W")

    except Exception as e:
        print(f"[GPU STATUS] Error: {e}")


if __name__ == "__main__":
    maximize_system()
    print_gpu_status()
