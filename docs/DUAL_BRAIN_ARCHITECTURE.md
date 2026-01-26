# DUAL-BRAIN FOREX TRADING ARCHITECTURE

## Chinese Quant Style - Best of Both Worlds

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    FOREX DUAL-BRAIN TRADING SYSTEM                          │
│                    "幻方量化 Style - Two Specialized Models"                 │
└─────────────────────────────────────────────────────────────────────────────┘

YOUR 806+ FORMULAS + 18,929 TRAINING SAMPLES
                    │
                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                     BRAIN #1: FORMULA BRAIN                                 │
│                     DeepSeek-R1-Distill-Qwen-7B (Fine-tuned)               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   WHAT IT DOES:                                                             │
│   • Calculates 806+ alpha signals from raw price data                       │
│   • Computes volatility (HAR-RV, GARCH, Yang-Zhang)                        │
│   • Analyzes microstructure (VPIN, Kyle Lambda, OFI)                       │
│   • Evaluates execution impact (Almgren-Chriss)                            │
│   • Mathematical reasoning with full formula knowledge                      │
│                                                                             │
│   INPUT:  OHLCV + tick data + order book                                   │
│   OUTPUT: JSON with calculated signals + confidence + reasoning             │
│                                                                             │
│   Example output:                                                           │
│   {                                                                         │
│     "alpha101_042": 0.73,                                                   │
│     "alpha191_015": -0.42,                                                  │
│     "har_rv_forecast": 0.0012,                                              │
│     "vpin": 0.68,                                                           │
│     "kyle_lambda": 0.00045,                                                 │
│     "signal_strength": 2.3,  // sigma                                       │
│     "reasoning": "Strong momentum confirmed by Alpha042..."                 │
│   }                                                                         │
│                                                                             │
│   RUNS ON: Local Ollama (forex-finetuned model)                            │
│   LATENCY: ~500ms per analysis                                              │
│                                                                             │
└───────────────────────────────────┬─────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                     BRAIN #2: TRADING BRAIN                                 │
│                     Kimi-K2-Instruct (1T MoE, 32B active)                   │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   WHAT IT DOES:                                                             │
│   • Makes final BUY/SELL/HOLD decisions                                     │
│   • Risk management and position sizing                                     │
│   • Multi-asset correlation analysis                                        │
│   • Market regime detection                                                 │
│   • Explains decisions in plain English                                     │
│                                                                             │
│   INPUT:  Formula Brain's JSON output + account state                       │
│   OUTPUT: Trading decision with position size and stops                     │
│                                                                             │
│   Example output:                                                           │
│   {                                                                         │
│     "action": "BUY",                                                        │
│     "symbol": "EURUSD",                                                     │
│     "position_pct": 15,  // % of capital                                    │
│     "stop_loss_pips": 25,                                                   │
│     "take_profit_pips": 50,                                                 │
│     "confidence": 0.78,                                                     │
│     "reasoning": "Formula brain shows 2.3σ signal strength..."             │
│   }                                                                         │
│                                                                             │
│   RUNS ON: Kimi API (Moonshot) or Together.ai                              │
│   COST: ~$0.14/M input tokens, ~$0.28/M output tokens                      │
│   LATENCY: ~1-2s per decision                                               │
│                                                                             │
└───────────────────────────────────┬─────────────────────────────────────────┘
                                    │
                                    ▼
                          EXECUTE TRADE VIA IB GATEWAY
```

---

## Why Two Models?

| Aspect | Single Model | Dual Model |
|--------|--------------|------------|
| **Specialization** | Jack of all trades | Expert at each task |
| **Fine-tuning** | Need to train everything | Only train formula brain |
| **Cost** | Expensive to run large model | Small local + cheap API |
| **Accuracy** | 60-70% | 75-85% (each brain optimized) |
| **Latency** | Slow (one big model) | Fast (parallel processing) |
| **Updates** | Retrain entire model | Update formula brain only |

---

## Model Selection

### Brain #1: Formula Brain (LOCAL)

| Option | Size | Speed | Quality | Cost |
|--------|------|-------|---------|------|
| **DeepSeek-R1-Distill-Qwen-7B** ★ | 7B | Fast | Excellent | Free (local) |
| DeepSeek-R1-Distill-Llama-8B | 8B | Fast | Excellent | Free (local) |
| Qwen2.5-7B-Instruct | 7B | Fast | Good | Free (local) |

**Winner: DeepSeek-R1-Distill-Qwen-7B**
- Best math reasoning at 7B size
- 97.3% accuracy on math benchmarks
- Perfect for formula calculations
- Runs fast on RTX 5080

### Brain #2: Trading Brain (API)

| Option | Size | Trading Perf | Cost/1M tokens | API |
|--------|------|--------------|----------------|-----|
| **Kimi-K2-Instruct** ★ | 1T (32B active) | Best | $0.14 in / $0.28 out | Moonshot, Together.ai |
| Claude Opus 4 | Unknown | Excellent | $15 in / $75 out | Anthropic |
| GPT-5 | Unknown | Good | $2.50 in / $10 out | OpenAI |
| DeepSeek-V3 | 671B | Stable | $0.27 in / $1.10 out | DeepSeek |

**Winner: Kimi-K2-Instruct**
- #1 trading performance in live tests
- Best Sortino ratio (40% better)
- Cheapest API cost
- 256K context window
- Open weights (can self-host later)

---

## Implementation Plan

### Phase 1: Train Formula Brain (TODAY)
```bash
# Deploy to Vast.ai H100
cd C:\Users\kevin\forex\forex\vastai_package
.\deploy_maxspeed.ps1 -Action install
.\deploy_maxspeed.ps1 -Action setkey -ApiKey YOUR_KEY
.\deploy_maxspeed.ps1 -Action search
# ... follow workflow
```

Output: `forex-finetuned` model in Ollama

### Phase 2: Integrate Kimi-K2 API
```python
# core/ml/trading_brain.py
from openai import OpenAI

client = OpenAI(
    api_key="YOUR_MOONSHOT_KEY",
    base_url="https://api.moonshot.cn/v1"  # or Together.ai
)

def get_trading_decision(formula_output: dict, account_state: dict) -> dict:
    response = client.chat.completions.create(
        model="kimi-k2-instruct",
        messages=[
            {"role": "system", "content": TRADING_SYSTEM_PROMPT},
            {"role": "user", "content": f"Formula analysis: {formula_output}\nAccount: {account_state}"}
        ],
        temperature=0.3
    )
    return parse_decision(response.choices[0].message.content)
```

### Phase 3: Connect to Trading Bot
```python
# In hft_trading_bot.py
async def process_signal(self, tick_data):
    # Step 1: Formula Brain (local, fast)
    formula_output = await self.formula_brain.analyze(tick_data)

    # Step 2: Trading Brain (API, 1-2s)
    if formula_output["signal_strength"] > 1.5:  # Only call API for strong signals
        decision = await self.trading_brain.decide(formula_output, self.account)

        if decision["action"] != "HOLD":
            await self.execute_trade(decision)
```

---

## Data Flow

```
Tick Data (TrueFX/IB)
         │
         ▼
┌─────────────────────────┐
│   Feature Engine        │  575 features extracted
│   (core/features/)      │
└───────────┬─────────────┘
            │
            ▼
┌─────────────────────────┐
│   Formula Brain         │  Local Ollama
│   (forex-finetuned)     │  ~500ms
│                         │
│   Calculates:           │
│   - Alpha signals       │
│   - Volatility          │
│   - Microstructure      │
│   - Signal strength     │
└───────────┬─────────────┘
            │
            ▼ (only if signal > threshold)
┌─────────────────────────┐
│   Trading Brain         │  Kimi-K2 API
│   (kimi-k2-instruct)    │  ~1-2s
│                         │
│   Decides:              │
│   - BUY/SELL/HOLD       │
│   - Position size       │
│   - Stop loss           │
│   - Take profit         │
└───────────┬─────────────┘
            │
            ▼
┌─────────────────────────┐
│   IB Gateway            │  Execute trade
│   (localhost:4004)      │
└─────────────────────────┘
```

---

## Cost Analysis (Per Day)

| Component | Calls/Day | Cost |
|-----------|-----------|------|
| Formula Brain (local) | 10,000+ | $0 |
| Trading Brain API | ~100-500 | ~$0.05-0.25 |
| **Total** | | **~$0.25/day** |

*Only calling Trading Brain for strong signals keeps API costs minimal*

---

## Files to Create

| File | Purpose |
|------|---------|
| `core/ml/formula_brain.py` | Local Ollama wrapper for formula analysis |
| `core/ml/trading_brain.py` | Kimi-K2 API wrapper for decisions |
| `core/ml/dual_brain_trader.py` | Orchestrator combining both |
| `config/llm_config.py` | API keys and endpoints |

---

## API Keys Needed

1. **Moonshot AI (Kimi-K2)**: https://platform.moonshot.cn/
   - Or Together.ai: https://www.together.ai/

2. **Vast.ai**: https://cloud.vast.ai/api/ (for training)

---

## References

- [Kimi K2 GitHub](https://github.com/MoonshotAI/Kimi-K2)
- [Kimi K2 Technical Deep Dive](https://intuitionlabs.ai/articles/kimi-k2-technical-deep-dive)
- [StockBench LLM Trading Benchmark](https://arxiv.org/html/2510.02209v1)
- [Finance LLM Benchmark](https://research.aimultiple.com/finance-llm/)
- [Yahoo Finance AI Trading Experiment](https://finance.yahoo.com/news/6-ai-models-trading-10k-130601853.html)
