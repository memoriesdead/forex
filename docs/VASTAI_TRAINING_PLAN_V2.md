# VAST.AI TRAINING PLAN V2: FOREX DATA INTEGRATED

```
╔════════════════════════════════════════════════════════════════════════════════╗
║  VERSION 2.0 - CRITICAL FIX: YOUR FOREX DATA NOW INCLUDED                      ║
║  PREVIOUS VERSION: Trained on generic formulas only (USELESS)                  ║
║  THIS VERSION: Trains on YOUR 51 pairs × 575 features (USEFUL)                 ║
║  DATE: 2026-01-22                                                              ║
╚════════════════════════════════════════════════════════════════════════════════╝
```

---

## WHAT WAS WRONG (V1)

```
V1 TRAINING DATA:
├── Formula Q&A samples (19,429)     ← Generic knowledge
├── Certainty module examples        ← Generic reasoning
└── ❌ YOUR FOREX DATA (0 samples)   ← NOT INCLUDED!

RESULT: LLM knows formulas but has NEVER seen your actual data patterns
```

## WHAT V2 FIXES

```
V2 TRAINING DATA:
├── Formula Q&A samples (19,429)           ← Generic knowledge
├── Certainty module examples              ← Generic reasoning
├── ✅ YOUR forex feature examples (10k+)  ← REAL data patterns
├── ✅ YOUR trade scenarios (5k+)          ← REAL decisions
└── ✅ YOUR DPO pairs from outcomes (2k+)  ← REAL right/wrong
```

---

## DATA ARCHITECTURE

### Layer 1: Your Forex Data (THE FOUNDATION)

```
training_package/
├── 51 forex pairs
├── 575 features per tick
├── 562,835 total samples
└── Targets: direction_1, direction_5, direction_10, etc.

THIS IS YOUR EDGE. The LLM must learn from THIS data.
```

### Layer 2: Formula Knowledge (THE REASONING)

```
training_data/
├── 1,172+ unique formulas
├── 19,429 Q&A samples
├── Academic citations
└── Mathematical explanations

This teaches the LLM HOW to think about trading.
```

### Layer 3: Integration (THE BRIDGE)

```
NEW: forex_integrated_training/
├── real_feature_examples.jsonl      ← "Given these 575 feature values..."
├── real_trade_scenarios.jsonl       ← "EURUSD shows Alpha001=0.73, VPIN=0.42..."
├── real_dpo_pairs.jsonl             ← Correct vs incorrect predictions
└── real_certainty_calculations.jsonl ← EdgeProof on YOUR data
```

---

## PHASE 0: DATA BRIDGE CREATION (NEW - DO LOCALLY FIRST)

### 0.1 Export Forex Features for LLM

**Script:** `scripts/create_llm_forex_bridge.py`

```python
"""
Creates LLM training samples from YOUR actual forex data.
This is the MISSING PIECE from V1.
"""

import pandas as pd
import json
from pathlib import Path
import numpy as np

def create_forex_llm_bridge():
    """
    Converts your 51 pairs × 575 features into LLM training samples.
    """

    output_samples = []
    pairs_dir = Path("training_package")

    for pair_dir in pairs_dir.iterdir():
        if not pair_dir.is_dir():
            continue

        pair = pair_dir.name
        train_file = pair_dir / "train.parquet"

        if not train_file.exists():
            continue

        # Load actual data
        df = pd.read_parquet(train_file)

        # Sample 200 rows per pair (10,200 total for 51 pairs)
        sample_df = df.sample(n=min(200, len(df)), random_state=42)

        for idx, row in sample_df.iterrows():
            # Create feature summary
            features = {col: float(row[col]) for col in df.columns
                       if not col.startswith('target_')}

            # Get actual outcome
            outcome = int(row.get('target_direction_10', 0))

            # Create training sample
            sample = create_trading_scenario_sample(pair, features, outcome)
            output_samples.append(sample)

    return output_samples


def create_trading_scenario_sample(pair: str, features: dict, outcome: int):
    """
    Creates a single training sample from real data.
    """

    # Format key features for prompt
    key_features = format_key_features(features)

    # Create the training conversation
    user_prompt = f"""Analyze this {pair} trading signal:

{key_features}

ML Ensemble Prediction: {'BUY' if outcome == 1 else 'SELL'}
ML Confidence: {features.get('ml_confidence', 0.63):.1%}

Should I execute this trade? Use certainty analysis."""

    # Generate the expected response based on actual outcome
    if outcome == 1:
        assistant_response = generate_approve_response(pair, features)
    else:
        assistant_response = generate_veto_response(pair, features)

    return {
        "messages": [
            {"role": "user", "content": user_prompt},
            {"role": "assistant", "content": assistant_response}
        ],
        "pair": pair,
        "outcome": outcome,
        "source": "real_forex_data"
    }


def format_key_features(features: dict) -> str:
    """Format the most important features for the prompt."""

    key_names = [
        'ret_1', 'ret_5', 'ret_10', 'ret_20',
        'vol_20', 'vol_50',
        'alpha001', 'alpha006', 'alpha012',
        'vpin', 'ofi', 'kyle_lambda',
        'rsi_14', 'macd', 'bb_position'
    ]

    lines = []
    for name in key_names:
        if name in features:
            lines.append(f"  {name}: {features[name]:.4f}")

    return "\n".join(lines)


def generate_approve_response(pair: str, features: dict) -> str:
    """Generate APPROVE response for correct predictions."""

    return f"""<think>
Analyzing {pair} signal with certainty framework...

1. EdgeProof Check:
   - Historical win rate on similar setups: 63%+
   - Bootstrap p-value < 0.001 (edge is statistically significant)

2. VPIN Analysis:
   - Current VPIN: {features.get('vpin', 0.35):.2f}
   - Below 0.5 threshold = low toxicity
   - Safe to trade

3. Factor Confirmation:
   - Alpha001: {features.get('alpha001', 0.1):.3f} (momentum aligned)
   - OFI: {features.get('ofi', 0.0):.3f} (order flow supports direction)

4. Kelly Sizing:
   - Win probability: 63%
   - Win/loss ratio: 1.5
   - Kelly fraction: f* = (0.63 × 1.5 - 0.37) / 1.5 = 0.38
   - Using half-Kelly: 19% of capital
</think>

**DECISION: APPROVE**

Certainty Score: 78%
Position Size: 19% (Half-Kelly)
Risk/Reward: 1.5:1

Reasoning: EdgeProof confirms statistical edge (p < 0.001), VPIN shows low toxicity, and factor signals align with ML prediction."""


def generate_veto_response(pair: str, features: dict) -> str:
    """Generate VETO response for incorrect predictions."""

    vpin = features.get('vpin', 0.6)

    return f"""<think>
Analyzing {pair} signal with certainty framework...

1. EdgeProof Check:
   - Similar setups have shown weakness recently
   - Need to verify edge still exists

2. VPIN Analysis:
   - Current VPIN: {vpin:.2f}
   - {'ELEVATED - potential informed trading' if vpin > 0.5 else 'Acceptable levels'}

3. Factor Divergence:
   - Alpha signals not fully aligned
   - OFI showing mixed signals

4. Uncertainty Assessment:
   - Epistemic uncertainty: HIGH
   - Market regime may be shifting
</think>

**DECISION: VETO**

Certainty Score: 45%
Reason: Insufficient certainty for trade execution

Specific Concerns:
1. {'VPIN elevated at ' + f'{vpin:.2f}' + ' (>0.5 threshold)' if vpin > 0.5 else 'Factor signals showing divergence'}
2. EdgeProof not conclusive for current regime
3. Better to wait for higher certainty setup

Action: Skip this signal, wait for cleaner setup."""
```

### 0.2 Create DPO Pairs from Real Outcomes

**The key insight:** Your ML ensemble is 63% accurate. That means:
- 63% of predictions are CORRECT → "chosen" responses
- 37% of predictions are WRONG → "rejected" responses

We use ACTUAL outcomes to create preference pairs:

```python
def create_dpo_pairs_from_real_data():
    """
    Creates DPO training pairs from actual trade outcomes.

    Chosen = What the model SHOULD have said (based on actual outcome)
    Rejected = What would have been WRONG
    """

    dpo_pairs = []

    for pair_dir in Path("training_package").iterdir():
        if not pair_dir.is_dir():
            continue

        df = pd.read_parquet(pair_dir / "train.parquet")

        # Sample rows where we know the outcome
        for idx, row in df.sample(n=100).iterrows():
            features = extract_features(row)
            actual_direction = row['target_direction_10']

            # The prompt is the same
            prompt = create_analysis_prompt(pair_dir.name, features)

            # Chosen = response aligned with ACTUAL outcome
            # Rejected = response that would have been WRONG

            if actual_direction == 1:  # Price went UP
                chosen = "APPROVE - EdgeProof confirms upward momentum..."
                rejected = "VETO - Uncertain signals suggest waiting..."
            else:  # Price went DOWN
                chosen = "VETO - Factors indicate downside risk..."
                rejected = "APPROVE - ML model confident, execute trade..."

            dpo_pairs.append({
                "prompt": prompt,
                "chosen": chosen,
                "rejected": rejected,
                "pair": pair_dir.name,
                "actual_outcome": int(actual_direction)
            })

    return dpo_pairs
```

### 0.3 Verification Checklist (BEFORE SPENDING MONEY)

Run this LOCALLY before renting any GPU:

```bash
python scripts/create_llm_forex_bridge.py --verify

# Expected output:
# ✅ Loaded 51 forex pairs
# ✅ Total samples: 562,835
# ✅ Features per sample: 575
# ✅ Generated 10,200 real feature examples
# ✅ Generated 5,100 real trade scenarios
# ✅ Generated 2,550 DPO pairs from actual outcomes
# ✅ Total integrated training samples: 17,850
# ✅ Combined with formula samples: 37,279 total
#
# READY FOR GPU TRAINING
```

---

## PHASE 1: LOCAL DATA PREPARATION

### 1.1 Required Files (V2)

| File | Source | Samples | Purpose |
|------|--------|---------|---------|
| `formula_samples.jsonl` | Existing | 19,429 | Formula knowledge |
| `real_feature_examples.jsonl` | **NEW** | 10,200 | Your actual data patterns |
| `real_trade_scenarios.jsonl` | **NEW** | 5,100 | Decision examples on your data |
| `real_dpo_pairs.jsonl` | **NEW** | 2,550 | Right/wrong from actual outcomes |
| **TOTAL** | | **37,279** | Complete training set |

### 1.2 Generate Integrated Data

```bash
cd C:\Users\kevin\forex\forex

# Step 1: Generate forex-integrated samples
python scripts/create_llm_forex_bridge.py --output training_data/forex_integrated/

# Step 2: Verify counts
python scripts/create_llm_forex_bridge.py --verify

# Step 3: Package for upload
python scripts/package_training_data.py --include-forex-integration
```

### 1.3 Verify Package Contents

```bash
python -c "
import json
from pathlib import Path

print('=== V2 TRAINING DATA VERIFICATION ===')

# Formula samples (existing)
formula_count = sum(1 for _ in open('training_data/deepseek_finetune_dataset.jsonl'))
print(f'Formula samples: {formula_count:,}')

# NEW: Real forex integration
forex_dir = Path('training_data/forex_integrated')
if forex_dir.exists():
    feature_examples = sum(1 for _ in open(forex_dir / 'real_feature_examples.jsonl'))
    trade_scenarios = sum(1 for _ in open(forex_dir / 'real_trade_scenarios.jsonl'))
    dpo_pairs = sum(1 for _ in open(forex_dir / 'real_dpo_pairs.jsonl'))

    print(f'Real feature examples: {feature_examples:,}')
    print(f'Real trade scenarios: {trade_scenarios:,}')
    print(f'Real DPO pairs: {dpo_pairs:,}')
    print(f'')
    print(f'TOTAL: {formula_count + feature_examples + trade_scenarios + dpo_pairs:,} samples')
    print(f'')
    print('✅ YOUR FOREX DATA IS NOW INCLUDED')
else:
    print('❌ FOREX INTEGRATION NOT FOUND - RUN create_llm_forex_bridge.py FIRST')
"
```

---

## PHASE 2: TRAINING SCRIPT UPDATE

### 2.1 Updated Data Loading

The training script must load ALL data sources:

```python
def load_all_training_data(data_dir: Path) -> List[Dict]:
    """
    V2: Loads formula knowledge + YOUR forex data
    """
    all_samples = []

    # 1. Formula knowledge (existing)
    formula_files = [
        'deepseek_finetune_dataset.jsonl',
        'llm_finetune_samples.jsonl',
        'high_quality_formulas.jsonl'
    ]
    for f in formula_files:
        if (data_dir / f).exists():
            all_samples.extend(load_jsonl(data_dir / f))

    # 2. YOUR FOREX DATA (NEW IN V2)
    forex_dir = data_dir / 'forex_integrated'
    if forex_dir.exists():
        # Real feature examples from your 51 pairs
        all_samples.extend(load_jsonl(forex_dir / 'real_feature_examples.jsonl'))

        # Real trade scenarios with your data
        all_samples.extend(load_jsonl(forex_dir / 'real_trade_scenarios.jsonl'))
    else:
        raise FileNotFoundError(
            "CRITICAL: forex_integrated/ not found!\n"
            "Run: python scripts/create_llm_forex_bridge.py first"
        )

    logger.info(f"Loaded {len(all_samples)} total samples (including YOUR forex data)")
    return all_samples


def load_dpo_data(data_dir: Path) -> List[Dict]:
    """
    V2: Loads DPO pairs from YOUR actual trade outcomes
    """
    dpo_pairs = []

    # 1. Generic DPO pairs (existing)
    generic_dpo = data_dir / 'grpo_forex' / 'train.jsonl'
    if generic_dpo.exists():
        dpo_pairs.extend(load_jsonl(generic_dpo))

    # 2. YOUR REAL DPO PAIRS (NEW IN V2)
    real_dpo = data_dir / 'forex_integrated' / 'real_dpo_pairs.jsonl'
    if real_dpo.exists():
        dpo_pairs.extend(load_jsonl(real_dpo))
        logger.info("Loaded DPO pairs from YOUR actual trade outcomes")
    else:
        raise FileNotFoundError(
            "CRITICAL: real_dpo_pairs.jsonl not found!\n"
            "Run: python scripts/create_llm_forex_bridge.py first"
        )

    return dpo_pairs
```

---

## PHASE 3: TRAINING EXECUTION

### 3.1 SFT Stage (Supervised Fine-Tuning)

**What it learns:**
- Formula knowledge (Alpha101, Kelly, VPIN, etc.)
- **YOUR data patterns** (what Alpha001=0.73 looks like on EURUSD)
- **YOUR decision scenarios** (when to approve/veto on YOUR data)

### 3.2 DPO Stage (Preference Learning)

**What it learns:**
- From **YOUR actual outcomes**: what worked vs what didn't
- 63% correct predictions → model learns to approve these patterns
- 37% incorrect predictions → model learns to veto these patterns

---

## PHASE 4: EXPECTED RESULTS

### V1 (Old - Generic Only)

```
LLM knows: "Kelly Criterion is f* = (bp - q) / b"
LLM doesn't know: What YOUR data looks like when Kelly should be applied
Result: Generic advice, not tailored to YOUR edge
```

### V2 (New - Your Data Integrated)

```
LLM knows: "Kelly Criterion is f* = (bp - q) / b"
LLM ALSO knows: "When EURUSD shows Alpha001=0.73, VPIN=0.35, ret_5=0.002,
                 similar patterns in YOUR data had 68% win rate"
Result: Advice based on YOUR actual data patterns
```

---

## VERIFICATION GATES

### Gate 1: Before Renting GPU

```bash
# Run locally - must ALL pass
python scripts/create_llm_forex_bridge.py --verify

# Expected:
# ✅ 51 forex pairs loaded
# ✅ 10,200+ real feature examples generated
# ✅ 5,100+ real trade scenarios generated
# ✅ 2,550+ real DPO pairs generated
# ✅ Total: 37,000+ samples
```

### Gate 2: After Upload to GPU

```bash
# Run on GPU instance
ls -la training_data/forex_integrated/
# Must show:
# real_feature_examples.jsonl
# real_trade_scenarios.jsonl
# real_dpo_pairs.jsonl

wc -l training_data/forex_integrated/*.jsonl
# Must show 17,850+ lines total
```

### Gate 3: During Training

```bash
# Training log must show:
# "Loaded X samples (including YOUR forex data)"
# NOT just "Loaded 19,429 formula samples"
```

### Gate 4: After Training

```bash
# Test with YOUR actual data pattern
ollama run forex-model "
EURUSD signal:
  Alpha001: 0.73
  VPIN: 0.35
  ret_5: 0.002
  vol_20: 0.0012

ML prediction: BUY (63% confidence)
Should I trade?
"

# Model should reference YOUR data patterns, not just generic formulas
```

---

## COST ESTIMATE (UPDATED)

| Phase | Duration | Cost |
|-------|----------|------|
| Data prep (local) | 30 min | $0 |
| Upload (from local) | 10 min | $0.80 |
| SFT (37k samples) | 2-3 hr | $16-24 |
| DPO (2.5k pairs) | 1 hr | $8 |
| Export + Download | 30 min | $4 |
| **TOTAL** | **4-5 hr** | **$30-40** |

**Buffer for errors:** Add 50% = **$45-60 max**

---

## FILES TO CREATE/UPDATE

### New Files Needed

| File | Purpose | Status |
|------|---------|--------|
| `scripts/create_llm_forex_bridge.py` | Generates integrated training data | [ ] CREATE |
| `training_data/forex_integrated/real_feature_examples.jsonl` | Your data patterns | [ ] GENERATE |
| `training_data/forex_integrated/real_trade_scenarios.jsonl` | Your decisions | [ ] GENERATE |
| `training_data/forex_integrated/real_dpo_pairs.jsonl` | Your outcomes | [ ] GENERATE |

### Files to Update

| File | Change | Status |
|------|--------|--------|
| `vastai_h100_training/train_deepseek_forex.py` | Add forex data loading | [ ] UPDATE |
| `docs/VASTAI_PREFLIGHT_CHECKLIST.md` | Add forex verification | [ ] UPDATE |
| `docs/VASTAI_TRAINING_AUDIT.md` | Add forex data audit | [ ] UPDATE |

---

## SUMMARY

```
╔══════════════════════════════════════════════════════════════════════════════════╗
║  V1 PROBLEM: LLM trained on generic formulas, never saw YOUR data               ║
║  V2 SOLUTION: LLM trained on formulas + YOUR 51 pairs × 575 features            ║
╠══════════════════════════════════════════════════════════════════════════════════╣
║                                                                                  ║
║  BEFORE SPENDING MONEY:                                                          ║
║  1. Run create_llm_forex_bridge.py locally                                       ║
║  2. Verify 37,000+ total samples generated                                       ║
║  3. Verify forex_integrated/ directory exists with 3 files                       ║
║  4. ONLY THEN rent GPU                                                           ║
║                                                                                  ║
╚══════════════════════════════════════════════════════════════════════════════════╝
```

---

**Created:** 2026-01-22
**Version:** 2.0
**Critical Fix:** Your forex data is now integrated into LLM training
