#!/usr/bin/env python3
"""
GRPO-Style Fine-Tuning for Forex LLM
====================================
Based on DeepSeek's approach and Chinese quant techniques.

Stage 1: Supervised Fine-Tuning (SFT) on formula data
Stage 2: GRPO reinforcement learning (optional, for trading feedback)

Key settings from research:
- LoRA rank: 8 (FinGPT optimal)
- Target modules: q_proj, v_proj only (not all layers)
- Learning rate: 1e-4 (lower for stability)
- Multiple epochs: 3-5 for domain adaptation
- Proper train/val split: 80/20
"""

import os
import sys
import json
import torch
from pathlib import Path
from typing import Dict, List, Optional

# Set working directory
os.chdir(Path(__file__).parent.parent)

print("=" * 70)
print("GRPO-STYLE FOREX LLM FINE-TUNING (DeepSeek Method)")
print("=" * 70)

# ============================================================================
# CONFIGURATION
# ============================================================================

class Config:
    """Fine-tuning configuration based on Chinese quant research."""

    # Model
    BASE_MODEL = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
    MAX_SEQ_LENGTH = 2048

    # LoRA Settings (FinGPT research: r=8 is optimal)
    LORA_R = 8                          # NOT 32! Research shows 8 is optimal
    LORA_ALPHA = 32                     # alpha = 4 * r
    LORA_DROPOUT = 0.05
    TARGET_MODULES = ["q_proj", "v_proj"]  # Only attention, not all layers

    # Training
    LEARNING_RATE = 1e-4                # Lower than typical (stability)
    BATCH_SIZE = 1                      # Small for 16GB VRAM
    GRADIENT_ACCUMULATION = 16          # Effective batch = 16
    NUM_EPOCHS = 5                      # More epochs for domain learning
    WARMUP_RATIO = 0.1
    WEIGHT_DECAY = 0.01
    MAX_GRAD_NORM = 1.0

    # Data
    TRAIN_DATA = "training_data/grpo_forex/train.jsonl"
    VAL_DATA = "training_data/grpo_forex/val.jsonl"

    # Output
    OUTPUT_DIR = "models/grpo-forex-checkpoints"
    FINAL_MODEL_DIR = "models/grpo-forex-lora"
    GGUF_OUTPUT_DIR = "models/grpo-forex"

    # GRPO specific (for Stage 2)
    GRPO_GROUP_SIZE = 4                 # Number of outputs to sample per input
    GRPO_CLIP_EPSILON = 0.2             # PPO-style clipping


def check_environment():
    """Check CUDA and dependencies."""
    print(f"\nPyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")

    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        props = torch.cuda.get_device_properties(0)
        print(f"VRAM: {props.total_memory / 1e9:.1f} GB")
        print(f"Compute capability: {props.major}.{props.minor}")
    else:
        print("WARNING: No CUDA GPU detected. Training will be slow.")

    return torch.cuda.is_available()


def load_training_data(train_path: str, val_path: str) -> tuple:
    """Load training and validation data."""
    from datasets import load_dataset

    train_ds = load_dataset('json', data_files=train_path, split='train')
    val_ds = load_dataset('json', data_files=val_path, split='train')

    print(f"\nTraining samples: {len(train_ds)}")
    print(f"Validation samples: {len(val_ds)}")

    return train_ds, val_ds


def format_for_training(example: Dict) -> Dict:
    """Format example for DeepSeek chat template."""
    instruction = example.get('instruction', '')
    input_text = example.get('input', '')
    output = example.get('output', '')

    if input_text:
        prompt = f"{instruction}\n\nContext: {input_text}"
    else:
        prompt = instruction

    # DeepSeek format
    text = f"""<|im_start|>user
{prompt}<|im_end|>
<|im_start|>assistant
{output}<|im_end|>"""

    return {"text": text}


def stage1_sft(config: Config, train_ds, val_ds):
    """Stage 1: Supervised Fine-Tuning."""
    print("\n" + "=" * 70)
    print("STAGE 1: SUPERVISED FINE-TUNING (SFT)")
    print("=" * 70)

    try:
        from unsloth import FastLanguageModel
        from unsloth.chat_templates import get_chat_template
        use_unsloth = True
        print("Using Unsloth for 2x faster training")
    except ImportError:
        print("Unsloth not available, using standard HuggingFace")
        use_unsloth = False
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from peft import get_peft_model, LoraConfig, TaskType

    # Load model
    print(f"\nLoading {config.BASE_MODEL}...")
    print(f"  LoRA rank: {config.LORA_R}")
    print(f"  Target modules: {config.TARGET_MODULES}")

    if use_unsloth:
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=config.BASE_MODEL,
            max_seq_length=config.MAX_SEQ_LENGTH,
            dtype=None,  # Auto-detect
            load_in_4bit=True,
        )

        # Add LoRA
        model = FastLanguageModel.get_peft_model(
            model,
            r=config.LORA_R,
            target_modules=config.TARGET_MODULES,
            lora_alpha=config.LORA_ALPHA,
            lora_dropout=config.LORA_DROPOUT,
            bias="none",
            use_gradient_checkpointing="unsloth",
            random_state=42,
        )
    else:
        from transformers import BitsAndBytesConfig

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )

        model = AutoModelForCausalLM.from_pretrained(
            config.BASE_MODEL,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
        )

        tokenizer = AutoTokenizer.from_pretrained(
            config.BASE_MODEL,
            trust_remote_code=True,
        )

        # Add LoRA
        lora_config = LoraConfig(
            r=config.LORA_R,
            lora_alpha=config.LORA_ALPHA,
            target_modules=config.TARGET_MODULES,
            lora_dropout=config.LORA_DROPOUT,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )
        model = get_peft_model(model, lora_config)

    # Print trainable parameters
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"\nTrainable parameters: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")

    # Format datasets
    print("\nFormatting datasets...")
    train_ds = train_ds.map(format_for_training)
    val_ds = val_ds.map(format_for_training)

    # Training arguments
    from transformers import TrainingArguments
    from trl import SFTTrainer

    training_args = TrainingArguments(
        output_dir=config.OUTPUT_DIR,
        per_device_train_batch_size=config.BATCH_SIZE,
        gradient_accumulation_steps=config.GRADIENT_ACCUMULATION,
        learning_rate=config.LEARNING_RATE,
        num_train_epochs=config.NUM_EPOCHS,
        warmup_ratio=config.WARMUP_RATIO,
        weight_decay=config.WEIGHT_DECAY,
        max_grad_norm=config.MAX_GRAD_NORM,
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        logging_steps=5,
        save_strategy="epoch",
        eval_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        optim="adamw_8bit" if use_unsloth else "adamw_torch",
        report_to="none",
        dataloader_pin_memory=True,
    )

    print(f"\nTraining configuration:")
    print(f"  Epochs: {config.NUM_EPOCHS}")
    print(f"  Effective batch size: {config.BATCH_SIZE * config.GRADIENT_ACCUMULATION}")
    print(f"  Learning rate: {config.LEARNING_RATE}")
    print(f"  Warmup ratio: {config.WARMUP_RATIO}")

    # Create trainer (num_proc=1 to avoid Windows multiprocessing issues)
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        dataset_text_field="text",
        max_seq_length=config.MAX_SEQ_LENGTH,
        args=training_args,
        dataset_num_proc=1,  # Disable multiprocessing for Windows
    )

    # Train
    print("\nStarting training...")
    train_result = trainer.train()

    # Log results
    print(f"\nTraining complete!")
    print(f"  Final train loss: {train_result.training_loss:.4f}")

    # Evaluate
    print("\nEvaluating on validation set...")
    eval_result = trainer.evaluate()
    print(f"  Validation loss: {eval_result['eval_loss']:.4f}")

    # Save model
    print(f"\nSaving model to {config.FINAL_MODEL_DIR}...")
    model.save_pretrained(config.FINAL_MODEL_DIR)
    tokenizer.save_pretrained(config.FINAL_MODEL_DIR)

    return model, tokenizer, train_result, eval_result


def export_to_gguf(config: Config, model, tokenizer):
    """Export model to GGUF for Ollama."""
    print("\n" + "=" * 70)
    print("EXPORTING TO GGUF FOR OLLAMA")
    print("=" * 70)

    try:
        from unsloth import FastLanguageModel
        use_unsloth = True
    except ImportError:
        use_unsloth = False

    if use_unsloth:
        print("\nExporting with Unsloth (Q4_K_M quantization)...")
        Path(config.GGUF_OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

        model.save_pretrained_gguf(
            config.GGUF_OUTPUT_DIR,
            tokenizer,
            quantization_method="q4_k_m"
        )
        print(f"GGUF saved to {config.GGUF_OUTPUT_DIR}/")
    else:
        print("\nUnsloth not available for GGUF export.")
        print("Use llama.cpp to convert manually:")
        print(f"  python convert.py {config.FINAL_MODEL_DIR} --outtype f16")
        print(f"  ./quantize {config.FINAL_MODEL_DIR}/model.gguf Q4_K_M")


def create_modelfile(config: Config):
    """Create Ollama Modelfile."""
    modelfile_content = '''FROM ./grpo-forex-q4_k_m.gguf

TEMPLATE """{{ if .System }}<|im_start|>system
{{ .System }}<|im_end|>
{{ end }}{{ if .Prompt }}<|im_start|>user
{{ .Prompt }}<|im_end|>
{{ end }}<|im_start|>assistant
{{ .Response }}<|im_end|>
"""

SYSTEM """You are a quantitative forex trading expert fine-tuned on 800+ mathematical formulas.

Your knowledge includes:

ALPHA FACTORS (WorldQuant 101 Alphas - Kakushadze 2016):
- Alpha001-020 and beyond
- Cross-sectional and time-series operations
- rank(), correlation(), delta(), stddev(), sum()

ALPHA191 (国泰君安 Guotai Junan):
- Chinese A-share market factors
- Volume-price relationships

VOLATILITY MODELS:
- HAR-RV (Corsi 2009): RV_t+1 = β₀ + β_d*RV_d + β_w*RV_w + β_m*RV_m
- GARCH family (Bollerslev 1986): σ²_t = ω + α*ε² + β*σ²
- Garman-Klass, Parkinson, Rogers-Satchell, Yang-Zhang

MICROSTRUCTURE:
- VPIN (Easley et al 2012): Order flow toxicity
- Kyle Lambda (1985): Market depth
- OFI (Cont et al 2014): Order Flow Imbalance

RISK MANAGEMENT:
- Kelly Criterion (1956): f* = (p*b - q) / b
- VaR, CVaR, Sharpe, Sortino, Calmar ratios

EXECUTION:
- Almgren-Chriss (2001): Optimal execution
- TWAP, VWAP algorithms
- Market impact models

REINFORCEMENT LEARNING:
- GRPO (DeepSeek 2025): Group Relative Policy Optimization
- PPO (Schulman 2017): Proximal Policy Optimization
- TD learning, GAE, DQN, SAC

Always cite academic sources and provide exact mathematical formulas."""

PARAMETER temperature 0.3
PARAMETER num_ctx 4096
PARAMETER num_predict 2048
PARAMETER repeat_penalty 1.2
'''

    modelfile_path = Path(config.GGUF_OUTPUT_DIR) / "Modelfile"
    modelfile_path.parent.mkdir(parents=True, exist_ok=True)

    with open(modelfile_path, 'w') as f:
        f.write(modelfile_content)

    print(f"\nModelfile created at {modelfile_path}")
    print("\nTo create Ollama model:")
    print(f"  cd {config.GGUF_OUTPUT_DIR}")
    print(f"  ollama create grpo-forex -f Modelfile")


def validate_model(config: Config, model, tokenizer):
    """Validate that model learned the formulas."""
    print("\n" + "=" * 70)
    print("VALIDATING MODEL KNOWLEDGE")
    print("=" * 70)

    test_prompts = [
        "What is the Alpha001 formula from WorldQuant 101 Alphas?",
        "Explain HAR-RV volatility model.",
        "How do I calculate Kelly Criterion?",
        "What is VPIN and how is it calculated?",
        "Explain GRPO from DeepSeek.",
    ]

    try:
        from unsloth import FastLanguageModel
        FastLanguageModel.for_inference(model)
    except:
        model.eval()

    print("\nTesting model responses:\n")

    for i, prompt in enumerate(test_prompts, 1):
        print(f"Q{i}: {prompt}")

        inputs = tokenizer(
            f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n",
            return_tensors="pt"
        ).to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=200,
                temperature=0.3,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
            )

        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Extract just the assistant response
        if "<|im_start|>assistant" in response:
            response = response.split("<|im_start|>assistant")[-1]

        print(f"A{i}: {response[:300]}...")
        print("-" * 50)


def main():
    """Main training pipeline."""

    # Check environment
    has_cuda = check_environment()

    if not has_cuda:
        print("\nWARNING: No GPU detected. Fine-tuning requires a GPU.")
        response = input("Continue anyway? (y/n): ")
        if response.lower() != 'y':
            return

    # Configuration
    config = Config()

    # Check data exists
    if not Path(config.TRAIN_DATA).exists():
        print(f"\nTraining data not found at {config.TRAIN_DATA}")
        print("Run: python scripts/generate_comprehensive_training_data.py")
        return

    # Load data
    print("\nLoading training data...")
    train_ds, val_ds = load_training_data(config.TRAIN_DATA, config.VAL_DATA)

    # Stage 1: SFT
    model, tokenizer, train_result, eval_result = stage1_sft(config, train_ds, val_ds)

    # Export to GGUF
    export_to_gguf(config, model, tokenizer)

    # Create Modelfile
    create_modelfile(config)

    # Validate
    validate_model(config, model, tokenizer)

    # Summary
    print("\n" + "=" * 70)
    print("FINE-TUNING COMPLETE!")
    print("=" * 70)
    print(f"""
Results:
  - Train loss: {train_result.training_loss:.4f}
  - Val loss: {eval_result['eval_loss']:.4f}

Files created:
  - LoRA adapter: {config.FINAL_MODEL_DIR}/
  - GGUF model: {config.GGUF_OUTPUT_DIR}/
  - Modelfile: {config.GGUF_OUTPUT_DIR}/Modelfile

Next steps:
  1. cd {config.GGUF_OUTPUT_DIR}
  2. ollama create grpo-forex -f Modelfile
  3. ollama run grpo-forex "What is Alpha001?"

The model is now ACTUALLY fine-tuned on your formulas!
""")


if __name__ == "__main__":
    # Windows multiprocessing fix
    import multiprocessing
    multiprocessing.freeze_support()

    main()
