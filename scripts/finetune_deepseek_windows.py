#!/usr/bin/env python3
"""
DeepSeek-R1 Fine-Tuning for Forex Formulas (Windows Compatible)
================================================================
Actual LoRA fine-tuning on RTX 5080 (16GB VRAM)
Trains on 806+ mathematical formulas from quant research

Key Windows fixes:
- multiprocessing.freeze_support()
- num_workers=0 in DataLoader
- if __name__ == "__main__" guard
"""

import os
import sys
import json
import torch
import logging
from pathlib import Path
from datetime import datetime

# Windows multiprocessing fix
if sys.platform == "win32":
    import multiprocessing
    multiprocessing.freeze_support()

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# =============================================================================
# CONFIG
# =============================================================================
BASE_MODEL = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"  # Smaller for 16GB VRAM
# Alternative: "Qwen/Qwen2.5-3B-Instruct" if DeepSeek fails
MAX_SEQ_LENGTH = 2048
BATCH_SIZE = 2  # Conservative for 16GB
GRADIENT_ACCUMULATION = 8  # Effective batch = 16
LEARNING_RATE = 2e-4
NUM_EPOCHS = 3
LORA_R = 64
LORA_ALPHA = 128
LORA_DROPOUT = 0.05

OUTPUT_DIR = Path("models/forex-finetuned")
DATA_DIR = Path("training_data")


def check_gpu():
    """Verify GPU is available and has enough VRAM."""
    if not torch.cuda.is_available():
        logger.error("CUDA not available! Fine-tuning requires GPU.")
        return False

    device = torch.cuda.get_device_name(0)
    vram = torch.cuda.get_device_properties(0).total_memory / 1e9
    logger.info(f"GPU: {device}")
    logger.info(f"VRAM: {vram:.1f} GB")

    if vram < 12:
        logger.warning("Less than 12GB VRAM - may run out of memory")

    return True


def load_training_data():
    """Load all formula training data."""
    all_samples = []

    # Priority order of data files
    data_files = [
        "high_quality_formulas.jsonl",  # Best - actual formulas
        "fingpt_forex_formulas.jsonl",  # FinGPT format
        "llm_finetune_samples.jsonl",   # Additional samples
    ]

    for filename in data_files:
        filepath = DATA_DIR / filename
        if filepath.exists():
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    samples = [json.loads(line) for line in f if line.strip()]
                    all_samples.extend(samples)
                    logger.info(f"Loaded {len(samples)} samples from {filename}")
            except Exception as e:
                logger.warning(f"Error loading {filename}: {e}")

    logger.info(f"Total training samples: {len(all_samples)}")
    return all_samples


def format_for_training(samples):
    """Format samples for instruction fine-tuning."""
    formatted = []

    for sample in samples:
        instruction = sample.get('instruction', '')
        input_text = sample.get('input', '')
        output = sample.get('output', '')

        if not instruction or not output:
            continue

        # Create chat format
        if input_text:
            user_content = f"{instruction}\n\nContext: {input_text}"
        else:
            user_content = instruction

        # DeepSeek chat format
        text = f"""<|im_start|>system
You are a quantitative forex trading expert trained on 806+ mathematical formulas from academic research. Always provide exact formulas with citations.<|im_end|>
<|im_start|>user
{user_content}<|im_end|>
<|im_start|>assistant
{output}<|im_end|>"""

        formatted.append({"text": text})

    return formatted


def main():
    """Main fine-tuning function."""
    logger.info("=" * 60)
    logger.info("DeepSeek Forex Formula Fine-Tuning")
    logger.info("=" * 60)

    # Check GPU
    if not check_gpu():
        return

    # Try to import training libraries
    try:
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
            TrainingArguments,
            Trainer,
            DataCollatorForLanguageModeling,
            BitsAndBytesConfig
        )
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
        from datasets import Dataset
    except ImportError as e:
        logger.error(f"Missing dependency: {e}")
        logger.info("Install with: pip install transformers peft datasets bitsandbytes accelerate")
        return

    # Load data
    logger.info("\n[1/5] Loading training data...")
    raw_samples = load_training_data()
    if len(raw_samples) < 100:
        logger.error("Not enough training samples!")
        return

    formatted_samples = format_for_training(raw_samples)
    logger.info(f"Formatted {len(formatted_samples)} training samples")

    # Create dataset
    dataset = Dataset.from_list(formatted_samples)
    dataset = dataset.shuffle(seed=42)

    # Split train/val
    split = dataset.train_test_split(test_size=0.1, seed=42)
    train_dataset = split['train']
    val_dataset = split['test']
    logger.info(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}")

    # Load tokenizer
    logger.info("\n[2/5] Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL,
        trust_remote_code=True,
        padding_side="right"
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Tokenize function
    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=MAX_SEQ_LENGTH,
            padding="max_length",
            return_tensors=None
        )

    # Tokenize datasets
    logger.info("Tokenizing...")
    train_dataset = train_dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=["text"],
        num_proc=1  # Single process for Windows
    )
    val_dataset = val_dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=["text"],
        num_proc=1
    )

    # 4-bit quantization config for memory efficiency
    logger.info("\n[3/5] Loading model with 4-bit quantization...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    )

    # Prepare for k-bit training
    model = prepare_model_for_kbit_training(model)

    # LoRA config
    logger.info("\n[4/5] Applying LoRA adapters...")
    lora_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj"
        ]
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Training arguments
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=str(OUTPUT_DIR),
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION,
        learning_rate=LEARNING_RATE,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        weight_decay=0.01,
        logging_steps=10,
        save_strategy="epoch",
        eval_strategy="epoch",
        load_best_model_at_end=True,
        bf16=True,
        optim="paged_adamw_8bit",
        gradient_checkpointing=True,
        max_grad_norm=1.0,
        report_to="none",
        dataloader_num_workers=0,  # Windows fix
        dataloader_pin_memory=False,  # Windows fix
        remove_unused_columns=False,
    )

    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )

    # Trainer
    logger.info("\n[5/5] Starting training...")
    logger.info(f"Effective batch size: {BATCH_SIZE * GRADIENT_ACCUMULATION}")

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
    )

    # Train
    try:
        trainer.train()

        # Save
        logger.info("\nSaving model...")
        trainer.save_model(str(OUTPUT_DIR / "final"))
        tokenizer.save_pretrained(str(OUTPUT_DIR / "final"))

        logger.info("\n" + "=" * 60)
        logger.info("FINE-TUNING COMPLETE!")
        logger.info("=" * 60)
        logger.info(f"Model saved to: {OUTPUT_DIR / 'final'}")
        logger.info("\nNext steps:")
        logger.info("1. Convert to GGUF: python scripts/convert_to_gguf.py")
        logger.info("2. Create Ollama model: ollama create forex-trained -f Modelfile")

    except Exception as e:
        logger.error(f"Training failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
