#!/usr/bin/env python3
"""
Convert merged safetensors model to GGUF format for Ollama.
Uses llama-cpp-python if available, otherwise provides instructions.
"""

import os
import sys
import subprocess
from pathlib import Path

MODEL_DIR = Path(__file__).parent.parent / "models" / "forex-quant"
OUTPUT_GGUF = MODEL_DIR / "forex-quant-q4_k_m.gguf"


def check_llama_cpp():
    """Check if llama.cpp tools are available."""
    try:
        result = subprocess.run(
            ["where", "llama-quantize"],
            capture_output=True,
            text=True,
            shell=True
        )
        return result.returncode == 0
    except:
        return False


def try_convert_with_ctransformers():
    """Try using ctransformers for conversion."""
    try:
        from ctransformers import AutoModelForCausalLM
        print("Using ctransformers for conversion...")
        model = AutoModelForCausalLM.from_pretrained(
            str(MODEL_DIR),
            model_type="llama"
        )
        # ctransformers doesn't directly export GGUF, just loads them
        return False
    except ImportError:
        return False
    except Exception as e:
        print(f"ctransformers failed: {e}")
        return False


def manual_conversion_instructions():
    """Print manual conversion instructions."""
    print("\n" + "=" * 70)
    print("MANUAL GGUF CONVERSION INSTRUCTIONS")
    print("=" * 70)
    print("""
Option 1: Use Hugging Face Hub (Easiest)
---------------------------------------
1. Upload your model to Hugging Face Hub
2. Use their GGUF conversion service
3. Download the GGUF file

Option 2: Use llama.cpp on WSL
-----------------------------
1. Install WSL if not installed: wsl --install
2. In WSL terminal:
   git clone https://github.com/ggerganov/llama.cpp
   cd llama.cpp
   make
   python3 convert_hf_to_gguf.py /mnt/c/Users/kevin/forex/forex/models/forex-quant \\
       --outfile /mnt/c/Users/kevin/forex/forex/models/forex-quant/forex-quant-f16.gguf \\
       --outtype f16
   ./llama-quantize /mnt/c/Users/kevin/forex/forex/models/forex-quant/forex-quant-f16.gguf \\
       /mnt/c/Users/kevin/forex/forex/models/forex-quant/forex-quant-q4_k_m.gguf q4_k_m

Option 3: Use pre-built Windows llama.cpp
----------------------------------------
1. Download from: https://github.com/ggerganov/llama.cpp/releases
2. Extract and run the conversion tools

Option 4: Use the LoRA adapter directly with Ollama
--------------------------------------------------
Since you have the base model already in Ollama (deepseek-r1:8b),
you can merge the adapter at runtime.

For now, the model is ready to use with transformers/peft:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    base_model = AutoModelForCausalLM.from_pretrained("deepseek-ai/DeepSeek-R1-Distill-Qwen-7B")
    model = PeftModel.from_pretrained(base_model, "forex-quant-lora")
    tokenizer = AutoTokenizer.from_pretrained("forex-quant-lora")
""")
    print("=" * 70)
    print(f"\nMerged model location: {MODEL_DIR}")
    print(f"LoRA adapter location: {MODEL_DIR.parent.parent / 'forex-quant-lora'}")
    print("=" * 70)


def main():
    print("GGUF Conversion for forex-quant model")
    print("=" * 50)
    print(f"Model directory: {MODEL_DIR}")

    # Check if model files exist
    safetensor_files = list(MODEL_DIR.glob("model-*.safetensors"))
    if not safetensor_files:
        print("ERROR: No safetensor files found in model directory!")
        sys.exit(1)

    print(f"Found {len(safetensor_files)} safetensor files")

    # Try different conversion methods
    if check_llama_cpp():
        print("llama.cpp tools found!")
        # Run conversion
        subprocess.run([
            "llama-quantize",
            str(MODEL_DIR),
            str(OUTPUT_GGUF),
            "q4_k_m"
        ])
    else:
        print("llama.cpp not found in PATH")
        manual_conversion_instructions()


if __name__ == "__main__":
    main()
