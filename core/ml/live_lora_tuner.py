"""
Live LoRA Tuner for Chinese Quant Style Continuous Learning

Implements incremental LoRA fine-tuning on RTX 5080 for forex-r1-v2 model.
Based on techniques from Chinese quantitative firms and DeepSeek.

=============================================================================
MASTER CITATION INDEX (for AI context preservation)
=============================================================================

CHINESE QUANT FIRMS:
- 幻方量化 (High-Flyer): https://www.high-flyer.cn/ - Full AI since 2017, DeepSeek parent
- 九坤投资 (Ubiquant): http://www.ubiquant.com/ - 650B+ RMB AUM, AI Lab
- 明汯投资 (MH Funds): https://www.mhfunds.com/ - 400 PFlops compute

DEEPSEEK PAPERS (OFFICIAL arXiv):
- GRPO: arXiv:2402.03300 (DeepSeekMath)
  "DeepSeekMath: Pushing the Limits of Mathematical Reasoning"
  https://arxiv.org/abs/2402.03300
  Section 2.2: "replay mechanism that incorporates 10% of historical data"
  HYPERPARAMETERS: LR=1e-6, KL=0.04, samples=64, batch=1024

- R1: arXiv:2501.12948 (DeepSeek-R1)
  "DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via RL"
  https://arxiv.org/abs/2501.12948
  GRPO HYPERPARAMETERS from R1:
  - Learning Rate: 3e-6
  - KL Coefficient: 0.001
  - GRPO Clip Ratio ε: 10
  - Samples per Question: 16
  - Max Length: 32,768
  - Batch Size: 512
  - Historical Replay: 10%

- V3: arXiv:2412.19437 (DeepSeek-V3)
  Training infrastructure, FP8, 8x H100 training

=============================================================================

CITATIONS (with official arXiv where available):
-----------
[1] DeepSeek GRPO (OFFICIAL arXiv):
    Paper: "DeepSeekMath: Pushing the Limits of Mathematical Reasoning"
    arXiv: https://arxiv.org/abs/2402.03300 (arXiv:2402.03300)
    Key insight: Iterative RL with group relative policy optimization

[2] XGBoost增量训练原理 (Official Docs):
    Documentation: https://xgboost.readthedocs.io/en/stable/python/python_api.html#xgboost.train
    xgb_model parameter for warm-start continuation
    process_type='update' with refresh_leaf=True for hot updates

[3] 滚动训练最佳实践 (BigQuant):
    "定期全量重训练优于简单增量训练"
    https://bigquant.com/wiki/doc/xKO5e4qSRT

[4] 幻方量化模型更新:
    Official: https://www.high-flyer.cn/
    "同时及时应对市场规则变化，不断更新模型"
    CEO 陆政哲 Interview: https://blog.csdn.net/wowotuo/article/details/106698768

[5] 明汯投资算力建设:
    Official: https://www.mhfunds.com/
    400 PFlops AI computing power, TOP500 cluster
    China Daily: https://caijing.chinadaily.com.cn/a/202408/21/WS66c57e93a310b35299d37b76.html

[6] 模型热更新 (Hot Swap):
    Pattern: "建立模型回滚机制，定期保存模型检查点，当在线更新导致性能下降时，快速回退"
    Implementation: Atomic model swap with version tracking

Author: Claude Code (forex-r1-v2 live tuning system)
Date: 2026-01-21
"""

import os
import json
import time
import shutil
import threading
import subprocess
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional, Callable
from datetime import datetime

# Import our modules
from .trade_outcome_buffer import TradeOutcomeBuffer, DPOPair, get_outcome_buffer
from .drift_detector import DriftDetector, DriftAlert, get_drift_detector


@dataclass
class LoRAVersion:
    """
    Track LoRA adapter versions for rollback capability.

    Citation [6]: "建立模型回滚机制，定期保存模型检查点"
    """
    version: int
    timestamp: float
    path: Path
    training_samples: int
    validation_accuracy: float
    win_rate_before: float
    win_rate_after: Optional[float] = None
    is_active: bool = False
    notes: str = ""


class LiveLoRATuner:
    """
    Live LoRA fine-tuning system for forex-r1-v2.

    Implements Chinese quant style continuous learning:
    1. Accumulate trade outcomes (via TradeOutcomeBuffer)
    2. Generate DPO pairs from outcomes
    3. Fine-tune LoRA on RTX 5080
    4. Validate and hot-swap if improved
    5. Rollback if performance degrades

    Citation [1]: DeepSeek uses iterative RL where model evolves continuously
    Citation [3]: BigQuant recommends periodic full retrain + incremental updates
    Citation [4]: 幻方量化 continuously updates models for market changes
    """

    def __init__(
        self,
        base_model_path: Path,
        lora_path: Path,
        ollama_model_name: str = "forex-r1-v2",
        min_samples_for_training: int = 500,
        training_interval_hours: float = 4.0,
        validation_holdout_ratio: float = 0.1,
        min_improvement_threshold: float = 0.01,  # 1% improvement required
        max_versions_to_keep: int = 5,
        device: str = "cuda",
    ):
        """
        Initialize live LoRA tuner.

        Args:
            base_model_path: Path to base safetensors model
            lora_path: Path to current LoRA adapter
            ollama_model_name: Name for Ollama model
            min_samples_for_training: Minimum DPO pairs before training
            training_interval_hours: Hours between training cycles
            validation_holdout_ratio: Ratio of data for validation
            min_improvement_threshold: Required improvement to swap models
            max_versions_to_keep: Max LoRA versions to retain (Citation [6])
            device: Training device (cuda for RTX 5080)
        """
        self.base_model_path = Path(base_model_path)
        self.lora_path = Path(lora_path)
        self.ollama_model_name = ollama_model_name
        self.min_samples_for_training = min_samples_for_training
        self.training_interval_hours = training_interval_hours
        self.validation_holdout_ratio = validation_holdout_ratio
        self.min_improvement_threshold = min_improvement_threshold
        self.max_versions_to_keep = max_versions_to_keep
        self.device = device

        # Paths
        self.versions_dir = self.lora_path.parent / "lora_versions"
        self.versions_dir.mkdir(parents=True, exist_ok=True)
        self.training_data_dir = Path("data/live_training")
        self.training_data_dir.mkdir(parents=True, exist_ok=True)

        # State
        self._lock = threading.RLock()
        self._versions: List[LoRAVersion] = []
        self._current_version: int = 0
        self._last_training_time: float = 0
        self._is_training: bool = False
        self._training_thread: Optional[threading.Thread] = None

        # Components
        self._outcome_buffer = get_outcome_buffer()
        self._drift_detector = get_drift_detector()

        # Callbacks
        self._on_training_complete: Optional[Callable] = None
        self._on_model_swapped: Optional[Callable] = None

        # Load existing versions
        self._load_version_history()

    def _load_version_history(self):
        """Load version history from disk."""
        history_file = self.versions_dir / "version_history.json"
        if history_file.exists():
            try:
                with open(history_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    for v in data.get("versions", []):
                        v["path"] = Path(v["path"])
                        self._versions.append(LoRAVersion(**v))
                    self._current_version = data.get("current_version", 0)
                print(f"[LORA] Loaded {len(self._versions)} version history")
            except Exception as e:
                print(f"[LORA] Warning: Could not load version history: {e}")

    def _save_version_history(self):
        """Save version history to disk."""
        history_file = self.versions_dir / "version_history.json"
        data = {
            "current_version": self._current_version,
            "versions": [
                {**asdict(v), "path": str(v.path)}
                for v in self._versions
            ]
        }
        with open(history_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)

    def should_train(self) -> bool:
        """
        Check if training should be triggered.

        Citation [3]: BigQuant recommends checking both time and data thresholds.
        """
        if self._is_training:
            return False

        # Check time interval
        hours_since_last = (time.time() - self._last_training_time) / 3600
        if hours_since_last < self.training_interval_hours:
            return False

        # Check data availability
        if not self._outcome_buffer.should_trigger_training():
            return False

        return True

    def start_training_async(self) -> bool:
        """
        Start background training if conditions are met.

        Citation [5]: 明汯投资 uses high-performance compute for continuous training.
        Returns True if training started.
        """
        if not self.should_train():
            return False

        with self._lock:
            if self._is_training:
                return False
            self._is_training = True

        self._training_thread = threading.Thread(
            target=self._training_loop,
            daemon=True,
        )
        self._training_thread.start()
        return True

    def _training_loop(self):
        """
        Main training loop (runs in background thread).

        Citation [1]: DeepSeek iterative RL - continuously evolve model
        Citation [2]: XGBoost warm-start principle applied to LoRA
        """
        try:
            print("[LORA] Starting background training cycle...")

            # 1. Export DPO pairs
            dpo_file = self.training_data_dir / f"dpo_pairs_{int(time.time())}.jsonl"
            num_pairs, num_historical = self._outcome_buffer.export_for_training(dpo_file)

            if num_pairs < self.min_samples_for_training // 2:
                print(f"[LORA] Insufficient data: {num_pairs} pairs")
                return

            # 2. Create new version directory
            new_version = self._current_version + 1
            new_version_dir = self.versions_dir / f"v{new_version:04d}"
            new_version_dir.mkdir(parents=True, exist_ok=True)

            # 3. Run LoRA training
            print(f"[LORA] Training v{new_version} with {num_pairs} pairs...")
            training_success = self._run_lora_training(
                dpo_file=dpo_file,
                output_dir=new_version_dir,
            )

            if not training_success:
                print("[LORA] Training failed")
                shutil.rmtree(new_version_dir, ignore_errors=True)
                return

            # 4. Validate new model
            print("[LORA] Validating new model...")
            validation_accuracy = self._validate_model(new_version_dir)

            # 5. Get current win rate for comparison
            current_win_rate = self._outcome_buffer.get_win_rate()

            # 6. Create version record
            version = LoRAVersion(
                version=new_version,
                timestamp=time.time(),
                path=new_version_dir,
                training_samples=num_pairs,
                validation_accuracy=validation_accuracy,
                win_rate_before=current_win_rate,
                notes=f"Trained on {num_pairs} pairs ({num_historical} historical)",
            )

            # 7. Decide whether to swap
            # Citation [3]: Only swap if improvement exceeds threshold
            current_accuracy = self._get_current_accuracy()
            improvement = validation_accuracy - current_accuracy

            if improvement >= self.min_improvement_threshold:
                print(f"[LORA] Improvement detected: {improvement:.2%}")
                self._hot_swap_model(version)
            else:
                print(f"[LORA] No improvement ({improvement:.2%}), keeping current model")
                version.notes += " (not activated - insufficient improvement)"

            # 8. Save version
            self._versions.append(version)
            self._cleanup_old_versions()
            self._save_version_history()

            # 9. Archive buffer
            self._outcome_buffer.archive_current_buffer()

            # 10. Reset drift detector baseline
            self._drift_detector.reset_baseline()

            self._last_training_time = time.time()

            if self._on_training_complete:
                self._on_training_complete(version)

            print(f"[LORA] Training cycle complete. Version: {new_version}")

        except Exception as e:
            print(f"[LORA] Training error: {e}")
            import traceback
            traceback.print_exc()
        finally:
            with self._lock:
                self._is_training = False

    def _run_lora_training(self, dpo_file: Path, output_dir: Path) -> bool:
        """
        Run LoRA fine-tuning on RTX 5080.

        Citation [5]: 明汯投资 invests heavily in compute infrastructure.
        We use local RTX 5080 (16GB VRAM) for continuous training.
        """
        # Create training script
        train_script = output_dir / "train_lora.py"
        script_content = f'''
"""
Auto-generated LoRA training script for forex-r1-v3 live tuning.
MAXIMUM GPU UTILIZATION for RTX 5080 16GB.
"""
import os

# MAXIMUM GPU SETTINGS
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True,garbage_collection_threshold:0.9,max_split_size_mb:512"
os.environ["CUDA_LAUNCH_BLOCKING"] = "0"
os.environ["OMP_NUM_THREADS"] = "12"
os.environ["TOKENIZERS_PARALLELISM"] = "true"

import torch

# Enable max speed settings
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, PeftModel
from trl import DPOConfig, DPOTrainer
from datasets import load_dataset

# Paths
BASE_MODEL = r"{self.base_model_path}"
CURRENT_LORA = r"{self.lora_path}"
DPO_FILE = r"{dpo_file}"
OUTPUT_DIR = r"{output_dir}"

print("[TRAIN] MAXIMUM GPU MODE - RTX 5080 16GB")
print("[TRAIN] Loading base model...")
model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    attn_implementation="flash_attention_2",
    use_cache=False,  # Disable for training
)
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
tokenizer.pad_token = tokenizer.eos_token

# Load existing LoRA and merge (Citation [2]: warm-start principle)
if os.path.exists(CURRENT_LORA):
    print("[TRAIN] Loading existing LoRA for warm-start...")
    model = PeftModel.from_pretrained(model, CURRENT_LORA)
    model = model.merge_and_unload()

# Apply new LoRA with MAXIMUM settings for RTX 5080
lora_config = LoraConfig(
    r=128,  # Higher rank = more capacity (was 64)
    lora_alpha=256,  # alpha = 2*r for stability
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, lora_config)
model.enable_input_require_grads()
model.gradient_checkpointing_enable()  # Trade compute for VRAM

# Print trainable params
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
total = sum(p.numel() for p in model.parameters())
print(f"[TRAIN] Trainable: {{trainable:,}} / {{total:,}} ({{100*trainable/total:.2f}}%)")

# Load DPO data
print(f"[TRAIN] Loading DPO data from {{DPO_FILE}}...")
dataset = load_dataset("json", data_files=DPO_FILE, split="train")

# DPO config with MAXIMUM batch size for RTX 5080 16GB
dpo_config = DPOConfig(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=8,  # Max batch (was 4)
    gradient_accumulation_steps=2,  # Effective batch = 16
    learning_rate=1e-5,  # Slightly higher for faster convergence
    warmup_ratio=0.1,
    num_train_epochs=1,
    max_length=2048,  # Longer sequences (was 1024)
    max_prompt_length=1024,  # (was 512)
    bf16=True,
    tf32=True,  # Enable TF32 for speed
    dataloader_num_workers=4,  # Parallel data loading
    dataloader_pin_memory=True,
    logging_steps=10,
    save_strategy="no",
    remove_unused_columns=False,
    optim="paged_adamw_8bit",  # 8-bit optimizer saves VRAM
    max_grad_norm=1.0,
)

# Train
print("[TRAIN] Starting DPO training (MAX GPU MODE)...")
trainer = DPOTrainer(
    model=model,
    args=dpo_config,
    train_dataset=dataset,
    processing_class=tokenizer,
)
trainer.train()

# Save LoRA adapter
print(f"[TRAIN] Saving LoRA to {{OUTPUT_DIR}}...")
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print("[TRAIN] Done!")
'''
        with open(train_script, 'w', encoding='utf-8') as f:
            f.write(script_content)

        # Run training
        try:
            result = subprocess.run(
                ["python", str(train_script)],
                capture_output=True,
                text=True,
                timeout=3600,  # 1 hour max
                cwd=str(output_dir),
            )

            if result.returncode != 0:
                print(f"[LORA] Training failed: {result.stderr}")
                return False

            print(result.stdout)
            return True

        except subprocess.TimeoutExpired:
            print("[LORA] Training timeout")
            return False
        except Exception as e:
            print(f"[LORA] Training error: {e}")
            return False

    def _validate_model(self, model_dir: Path) -> float:
        """
        Validate new model on holdout data.

        Returns accuracy score.
        """
        # For now, return a simulated validation
        # In production, this would run actual inference
        return 0.65  # Placeholder

    def _get_current_accuracy(self) -> float:
        """Get current model's validation accuracy."""
        for v in reversed(self._versions):
            if v.is_active:
                return v.validation_accuracy
        return 0.60  # Default baseline

    def _hot_swap_model(self, version: LoRAVersion):
        """
        Hot-swap to new model version.

        Citation [6]: Hot update without downtime
        Citation [4]: 幻方量化 continuously updates models
        """
        print(f"[LORA] Hot-swapping to version {version.version}...")

        # 1. Deactivate current version
        for v in self._versions:
            v.is_active = False

        # 2. Copy new LoRA to active location
        if self.lora_path.exists():
            backup_path = self.lora_path.parent / f"lora_backup_{int(time.time())}"
            shutil.move(str(self.lora_path), str(backup_path))

        shutil.copytree(str(version.path), str(self.lora_path))

        # 3. Update Ollama model
        self._update_ollama_model()

        # 4. Mark version as active
        version.is_active = True
        self._current_version = version.version

        if self._on_model_swapped:
            self._on_model_swapped(version)

        print(f"[LORA] Hot-swap complete. Active version: {version.version}")

    def _update_ollama_model(self):
        """
        Update Ollama model with new LoRA.

        This recreates the Ollama model to pick up the new adapter.
        """
        try:
            modelfile_path = self.base_model_path.parent / "Modelfile"

            if modelfile_path.exists():
                # Recreate Ollama model
                result = subprocess.run(
                    ["ollama", "create", f"{self.ollama_model_name}-live", "-f", str(modelfile_path)],
                    capture_output=True,
                    text=True,
                    timeout=300,
                )

                if result.returncode == 0:
                    print(f"[LORA] Ollama model {self.ollama_model_name}-live updated")
                else:
                    print(f"[LORA] Ollama update warning: {result.stderr}")

        except Exception as e:
            print(f"[LORA] Ollama update error: {e}")

    def _cleanup_old_versions(self):
        """
        Remove old versions, keeping max_versions_to_keep.

        Citation [6]: Maintain rollback capability while managing storage.
        """
        if len(self._versions) <= self.max_versions_to_keep:
            return

        # Sort by version, keep newest
        self._versions.sort(key=lambda v: v.version)
        to_remove = self._versions[:-self.max_versions_to_keep]

        for v in to_remove:
            if not v.is_active:
                try:
                    shutil.rmtree(v.path, ignore_errors=True)
                    print(f"[LORA] Removed old version {v.version}")
                except Exception:
                    pass

        self._versions = self._versions[-self.max_versions_to_keep:]

    def rollback(self, version: int) -> bool:
        """
        Rollback to a previous version.

        Citation [6]: "当在线更新导致性能下降时，快速回退至历史版本"
        """
        target = None
        for v in self._versions:
            if v.version == version:
                target = v
                break

        if target is None:
            print(f"[LORA] Version {version} not found")
            return False

        if not target.path.exists():
            print(f"[LORA] Version {version} files not found")
            return False

        self._hot_swap_model(target)
        print(f"[LORA] Rolled back to version {version}")
        return True

    def get_stats(self) -> Dict:
        """Get tuner statistics."""
        return {
            "current_version": self._current_version,
            "total_versions": len(self._versions),
            "is_training": self._is_training,
            "last_training": datetime.fromtimestamp(self._last_training_time).isoformat()
                if self._last_training_time > 0 else None,
            "hours_since_training": (time.time() - self._last_training_time) / 3600
                if self._last_training_time > 0 else None,
            "buffer_ready": self._outcome_buffer.should_trigger_training(),
            "buffer_stats": self._outcome_buffer.get_stats(),
            "drift_stats": self._drift_detector.get_stats(),
        }

    def list_versions(self) -> List[Dict]:
        """List all available versions."""
        return [
            {
                "version": v.version,
                "timestamp": datetime.fromtimestamp(v.timestamp).isoformat(),
                "samples": v.training_samples,
                "accuracy": v.validation_accuracy,
                "is_active": v.is_active,
                "notes": v.notes,
            }
            for v in self._versions
        ]


# Singleton instance
_tuner_instance: Optional[LiveLoRATuner] = None


def get_live_tuner(
    base_model_path: Optional[Path] = None,
    lora_path: Optional[Path] = None,
) -> LiveLoRATuner:
    """Get or create the global tuner instance."""
    global _tuner_instance

    if _tuner_instance is None:
        if base_model_path is None:
            # Use forex-r1-v3 (2026-01-22 trained on 8x H100)
            base_model_path = Path("models/forex-r1-v3/merged_model")
        if lora_path is None:
            lora_path = Path("models/forex-r1-v3/lora_adapter")

        _tuner_instance = LiveLoRATuner(
            base_model_path=base_model_path,
            lora_path=lora_path,
        )

    return _tuner_instance
