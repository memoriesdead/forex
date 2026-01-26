"""
Convert forex-r1-v2 safetensors to GGUF for Ollama deployment.

This script converts the trained model from HuggingFace safetensors format
to GGUF format that can be used with Ollama.

Requirements:
    pip install llama-cpp-python transformers torch

Usage:
    python scripts/convert_forex_r1_v2_to_gguf.py

Author: Claude Code
Date: 2026-01-21
"""

import os
import sys
import subprocess
from pathlib import Path

# Paths
MODEL_DIR = Path("models/forex-r1-v2")
OUTPUT_GGUF = MODEL_DIR / "forex-r1-v2-q8_0.gguf"
LLAMA_CPP_DIR = Path("llama-cpp")


def check_llama_cpp():
    """Check if llama.cpp is available."""
    convert_script = LLAMA_CPP_DIR / "convert_hf_to_gguf.py"
    quantize_bin = LLAMA_CPP_DIR / "build" / "bin" / "llama-quantize"

    if not convert_script.exists():
        print(f"llama.cpp not found at {LLAMA_CPP_DIR}")
        print("Please clone: git clone https://github.com/ggerganov/llama.cpp")
        return False

    return True


def convert_to_gguf():
    """Convert safetensors to GGUF using llama.cpp."""
    print("=" * 60)
    print("Converting forex-r1-v2 to GGUF format")
    print("=" * 60)

    # Check if already converted
    if OUTPUT_GGUF.exists():
        print(f"GGUF already exists: {OUTPUT_GGUF}")
        return True

    # Method 1: Use llama.cpp convert script
    convert_script = LLAMA_CPP_DIR / "convert_hf_to_gguf.py"

    if convert_script.exists():
        print(f"Using llama.cpp conversion script...")

        # Convert to f16 GGUF first
        f16_gguf = MODEL_DIR / "forex-r1-v2-f16.gguf"

        cmd = [
            sys.executable,
            str(convert_script),
            str(MODEL_DIR),
            "--outfile", str(f16_gguf),
            "--outtype", "f16",
        ]

        print(f"Running: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode != 0:
            print(f"Conversion failed: {result.stderr}")
            return False

        print(result.stdout)

        # Quantize to Q8_0
        quantize_bin = LLAMA_CPP_DIR / "build" / "bin" / "llama-quantize"
        if quantize_bin.exists():
            print("Quantizing to Q8_0...")
            cmd = [str(quantize_bin), str(f16_gguf), str(OUTPUT_GGUF), "q8_0"]
            result = subprocess.run(cmd, capture_output=True, text=True)

            if result.returncode == 0:
                # Remove f16 intermediate
                f16_gguf.unlink()
                print(f"Created: {OUTPUT_GGUF}")
                return True
        else:
            # Just use f16 if quantize not available
            f16_gguf.rename(OUTPUT_GGUF)
            print(f"Created: {OUTPUT_GGUF} (f16, not quantized)")
            return True

    # Method 2: Use transformers + llama-cpp-python
    print("Attempting conversion via transformers...")
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import torch

        print("Loading model...")
        model = AutoModelForCausalLM.from_pretrained(
            str(MODEL_DIR),
            torch_dtype=torch.float16,
            device_map="cpu",
        )
        tokenizer = AutoTokenizer.from_pretrained(str(MODEL_DIR))

        # Save in a format llama.cpp can read
        print("This method requires llama.cpp tools for final conversion.")
        print("Please install llama.cpp and run again.")
        return False

    except Exception as e:
        print(f"Error: {e}")
        return False


def update_modelfile():
    """Update Modelfile to point to GGUF."""
    modelfile = MODEL_DIR / "Modelfile"

    if not OUTPUT_GGUF.exists():
        print("GGUF not found, cannot update Modelfile")
        return

    content = f"""# forex-r1-v2 Modelfile for Ollama
# Chinese Quant Style GRPO-Trained Forex Reasoning Model
#
# TRAINING CITATIONS:
# [1] DeepSeek GRPO - https://zhuanlan.zhihu.com/p/21046265072
# [2] 幻方量化 - https://blog.csdn.net/zk168_net/article/details/108076246
# [3] BigQuant滚动训练 - https://bigquant.com/wiki/doc/xVqIPu6RoI

FROM {OUTPUT_GGUF.name}

PARAMETER temperature 0.3
PARAMETER top_p 0.9
PARAMETER top_k 40
PARAMETER num_ctx 4096
PARAMETER repeat_penalty 1.1
PARAMETER num_predict 1024

SYSTEM \"\"\"You are forex-r1-v2, a specialized forex trading reasoning model.

KNOWLEDGE: 806+ quant formulas (Alpha101, HAR-RV, GARCH, Kelly, VPIN, etc.)
TRAINING: SFT + DPO on 131k samples from real forex trades

YOUR ROLE:
1. Analyze ML signals (XGBoost + LightGBM + CatBoost, 63% accuracy)
2. Provide reasoning in <think>...</think> tags
3. Output: APPROVE/VETO + confidence + Kelly sizing

Be concise. Focus on actionable insights.\"\"\"

TEMPLATE \"\"\"{{{{ if .System }}}}<|im_start|>system
{{{{ .System }}}}<|im_end|>
{{{{ end }}}}{{{{ if .Prompt }}}}<|im_start|>user
{{{{ .Prompt }}}}<|im_end|>
{{{{ end }}}}<|im_start|>assistant
{{{{ .Response }}}}<|im_end|>
\"\"\"
"""

    with open(modelfile, 'w', encoding='utf-8') as f:
        f.write(content)

    print(f"Updated Modelfile to use {OUTPUT_GGUF.name}")


def deploy_to_ollama():
    """Deploy GGUF to Ollama."""
    if not OUTPUT_GGUF.exists():
        print("GGUF not found, skipping Ollama deployment")
        return False

    print("Deploying to Ollama...")

    # Find Ollama
    ollama_paths = [
        Path(os.environ.get("LOCALAPPDATA", "")) / "Programs" / "Ollama" / "ollama.exe",
        Path("C:/Users/kevin/AppData/Local/Programs/Ollama/ollama.exe"),
        Path("/usr/local/bin/ollama"),
    ]

    ollama = None
    for p in ollama_paths:
        if p.exists():
            ollama = p
            break

    if ollama is None:
        print("Ollama not found. Please install from https://ollama.ai")
        return False

    # Create model
    modelfile = MODEL_DIR / "Modelfile"
    cmd = [str(ollama), "create", "forex-r1-v2", "-f", str(modelfile)]

    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=str(MODEL_DIR))

    if result.returncode == 0:
        print("Successfully deployed forex-r1-v2 to Ollama!")
        print(result.stdout)
        return True
    else:
        print(f"Deployment failed: {result.stderr}")
        return False


def main():
    print("""
╔══════════════════════════════════════════════════════════════════════════════════╗
║                    FOREX-R1-V2 GGUF CONVERSION                                   ║
║                    For Ollama Deployment                                         ║
╚══════════════════════════════════════════════════════════════════════════════════╝
""")

    # Check model exists
    if not MODEL_DIR.exists():
        print(f"Model directory not found: {MODEL_DIR}")
        return

    # Check for safetensors
    safetensors = list(MODEL_DIR.glob("*.safetensors"))
    if not safetensors:
        print("No safetensors found in model directory")
        return

    print(f"Found {len(safetensors)} safetensor files")

    # Convert
    if convert_to_gguf():
        update_modelfile()
        deploy_to_ollama()
    else:
        print("""
╔══════════════════════════════════════════════════════════════════════════════════╗
║  MANUAL CONVERSION REQUIRED                                                      ║
╠══════════════════════════════════════════════════════════════════════════════════╣
║  1. Clone llama.cpp:                                                             ║
║     git clone https://github.com/ggerganov/llama.cpp                             ║
║                                                                                  ║
║  2. Build llama.cpp:                                                             ║
║     cd llama-cpp && mkdir build && cd build                                      ║
║     cmake .. && cmake --build . --config Release                                 ║
║                                                                                  ║
║  3. Convert:                                                                     ║
║     python llama-cpp/convert_hf_to_gguf.py models/forex-r1-v2 \\                  ║
║         --outfile models/forex-r1-v2/forex-r1-v2-q8_0.gguf --outtype q8_0        ║
║                                                                                  ║
║  4. Deploy to Ollama:                                                            ║
║     cd models/forex-r1-v2                                                        ║
║     ollama create forex-r1-v2 -f Modelfile                                       ║
╚══════════════════════════════════════════════════════════════════════════════════╝
""")


if __name__ == "__main__":
    main()
