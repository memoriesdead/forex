# Vast.ai ML Ensemble Training - ALL 55 Pairs with 1239 Features

## Quick Start

```bash
# 1. Upload this folder + training_package to Vast.ai
# 2. SSH into instance
# 3. Install dependencies
pip install -r requirements.txt

# 4. Run training
python train_all_pairs.py 2>&1 | tee training.log
```

## Hardware Requirements

- **GPU:** 1x H100/A100 (16GB+ VRAM)
- **RAM:** 64GB+
- **Storage:** 50GB+

## Expected Time (8x H100)

| Pairs | Time per Pair | Total |
|-------|---------------|-------|
| 55 | ~5-10 min | ~5-9 hours |

## Features Generated (1239 total)

| Source | Count |
|--------|-------|
| Alpha101 | 72 |
| Alpha158 | 179 |
| Alpha360 | 276 |
| Barra CNE6 | 46 |
| Alpha191 | 201 |
| Chinese HFT | 22 |
| Chinese Additions | 20 |
| Chinese Gold | 27 |
| US Academic | 50 |
| MLFinLab | 17 |
| Time-Series | 41 |
| Renaissance | 51 |
| India Quant | 25 |
| Japan Quant | 30 |
| Europe Quant | 15 |
| Emerging | 20 |
| Universal Math | 29 |
| RL | 35 |
| MOE | 20 |
| GNN | 15 |
| Korea Quant | 20 |
| Asia FX | 15 |
| MARL | 15 |

## Output

Models saved to: `models/<PAIR>_models.pkl`
Results saved to: `models/<PAIR>_results.json`

## Download After Training

```bash
# From local machine
scp -r -P <PORT> root@<HOST>:/workspace/vastai_ml_training/models/* C:\Users\kevin\forex\forex\models\production\
```

## Checkpointing

Training saves checkpoints after each target (direction_1, direction_5, direction_10).
If training crashes, resume with same command - it will skip completed targets.
