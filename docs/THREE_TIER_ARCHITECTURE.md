# THREE-TIER FOREX ARCHITECTURE

## Opus 4.5 Recommended - Adapted for Forex

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    YOUR RTX 5080 LOCAL SYSTEM                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌───────────────┐     ┌──────────────────┐     ┌─────────────────┐        │
│  │   TIER 1      │     │     TIER 2       │     │    TIER 3       │        │
│  │   LightGBM    │────>│   Forex-R1       │────>│    Kimi-K2      │        │
│  │   Ensemble    │     │   (7B Custom)    │     │   (1T API)      │        │
│  │               │     │                  │     │                 │        │
│  │  • XGBoost    │     │  • 806+ formulas │     │  • Complex      │        │
│  │  • LightGBM   │     │  • Alpha101/191  │     │    decisions    │        │
│  │  • CatBoost   │     │  • HAR-RV/GARCH  │     │  • Risk mgmt    │        │
│  │               │     │  • VPIN/Kyle     │     │  • Overnight    │        │
│  │  63% accuracy │     │  ~2 sec/trade    │     │  • Position     │        │
│  │  <50ms        │     │  LOCAL Ollama    │     │    sizing       │        │
│  └───────────────┘     └──────────────────┘     └─────────────────┘        │
│         │                      │                        │                  │
│         │ ALL ticks            │ Strong signals         │ Major decisions  │
│         │ (screening)          │ (reasoning)            │ (optional)       │
│         ▼                      ▼                        ▼                  │
│                                                                             │
│  Retrain Daily:          Retrain Weekly:           API Only:               │
│  Chinese Quant           Vast.ai H200              Together.ai             │
│  Online Learning         ~$1.50/week               ~$0.25/day              │
│  (FREE on RTX 5080)      (when formulas change)    (complex trades)        │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Why 3 Tiers Beat 2 Tiers for Forex

| Aspect | Dual-Brain | Three-Tier |
|--------|------------|------------|
| **Speed** | ~2.5s per tick | **<50ms** screening |
| **Cost** | API every signal | API only complex trades |
| **Accuracy** | LLM for everything | ML + LLM + LLM |
| **HFT Ready** | No (too slow) | **Yes** (Tier 1 instant) |

---

## Data Flow

```
Every Tick (10-100/sec)
         │
         ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  TIER 1: ML ENSEMBLE (Already Built!)                                       │
│  XGBoost + LightGBM + CatBoost                                              │
│  63% accuracy, <50ms                                                        │
│                                                                             │
│  Input: 575 features                                                        │
│  Output: direction, probability, confidence                                 │
│                                                                             │
│  IF confidence < 0.6 → SKIP (no trade)                                      │
│  IF confidence 0.6-0.8 → Tier 2                                             │
│  IF confidence > 0.8 → Direct execute (proven edge)                         │
└───────────────────────────────────┬─────────────────────────────────────────┘
                                    │
                                    ▼ (~5% of ticks)
┌─────────────────────────────────────────────────────────────────────────────┐
│  TIER 2: FOREX-R1 (Fine-tuned DeepSeek, LOCAL)                              │
│  806+ mathematical formulas embedded                                        │
│  ~2 sec per analysis                                                        │
│                                                                             │
│  Input: ML prediction + tick features + OHLCV                               │
│  Output: {                                                                  │
│    alpha_signals: {...},                                                    │
│    volatility_forecast: 0.0012,                                             │
│    signal_strength: 2.3σ,                                                   │
│    reasoning: "Alpha042 confirms momentum..."                               │
│  }                                                                          │
│                                                                             │
│  IF signal_strength < 1.5σ → SKIP                                           │
│  IF signal_strength 1.5-2.5σ → Execute with formula sizing                  │
│  IF signal_strength > 2.5σ OR overnight → Tier 3                            │
└───────────────────────────────────┬─────────────────────────────────────────┘
                                    │
                                    ▼ (~1% of trades)
┌─────────────────────────────────────────────────────────────────────────────┐
│  TIER 3: KIMI-K2 (1T MoE, API)                                              │
│  #1 trading performance in benchmarks                                       │
│  ~2 sec per decision                                                        │
│                                                                             │
│  Use for:                                                                   │
│  • Major position changes (>20% of capital)                                 │
│  • Overnight holds (risk of gap)                                            │
│  • High correlation situations                                              │
│  • Unusual market regimes                                                   │
│  • Weekly portfolio rebalancing                                             │
│                                                                             │
│  Output: Final BUY/SELL/HOLD + position % + stops                           │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Forex-Specific Adaptations

### vs Stocks Architecture

| Stocks | Forex | Why Different |
|--------|-------|---------------|
| 29M rows historical | 200M+ ticks | Forex is 24/5, way more data |
| Daily rebalance | Tick-by-tick | Forex = continuous market |
| Overnight = closed | Overnight = gap risk | Forex opens Sunday |
| Single exchange | Multi-broker | Need best execution |
| Earnings events | Central bank events | Different catalysts |

### Forex-Specific Tier 2 Prompts

```python
# Tier 2 should understand forex concepts
FOREX_CONTEXT = """
FOREX MARKET STRUCTURE:
- 24/5 continuous trading (Sun 5PM - Fri 5PM ET)
- Sessions: Sydney → Tokyo → London → New York
- Current session affects volatility and spreads
- Pip values vary by pair (USDJPY = 0.01, EURUSD = 0.0001)
- Leverage typically 50:1 (retail) to 500:1 (institutional)

PAIRS YOU TRADE:
- Majors: EURUSD, GBPUSD, USDJPY, USDCHF, AUDUSD, USDCAD, NZDUSD
- Crosses: EURJPY, GBPJPY, EURGBP, AUDJPY
- Exotics: EURTRY, USDZAR, USDMXN

CORRELATION GROUPS:
- USD pairs move together (DXY correlation)
- JPY pairs = risk-off indicator
- AUD/NZD = commodity currencies
- EUR/GBP = Brexit sensitivity
"""
```

### Tier 3 Kimi-K2 Use Cases for Forex

```python
# Only call Tier 3 for these scenarios
TIER3_TRIGGERS = {
    "large_position": position_pct > 20,          # Big trade
    "overnight_hold": is_friday_afternoon,         # Weekend gap risk
    "high_correlation": corr_with_open > 0.7,      # Too much exposure
    "central_bank": news_event_within_hours(2),    # Fed/ECB/BOJ
    "regime_change": volatility_spike > 2.0,       # Market shift
    "weekly_rebalance": is_sunday_open,            # Fresh week
}
```

---

## Implementation Changes

### Update DualBrainTrader → ThreeTierTrader

```python
class ThreeTierTrader:
    def __init__(self):
        # Tier 1: Already built! (core/ml/adaptive_ensemble.py)
        self.ml_ensemble = AdaptiveMLEnsemble()

        # Tier 2: Local Forex-R1 (core/ml/formula_brain.py)
        self.formula_brain = FormulaBrain("forex-finetuned")

        # Tier 3: Kimi-K2 API (core/ml/trading_brain.py)
        self.trading_brain = TradingBrain("moonshot")

    async def process_tick(self, tick):
        # Tier 1: Fast ML (<50ms)
        ml_result = self.ml_ensemble.predict(tick.symbol, tick.features)

        if ml_result.confidence < 0.6:
            return None  # Skip

        if ml_result.confidence > 0.8:
            # High confidence - direct execute
            return self.execute(ml_result)

        # Tier 2: Formula reasoning (~2s)
        formula = await self.formula_brain.analyze(tick, ml_result)

        if formula.signal_strength < 1.5:
            return None  # Skip

        if formula.signal_strength < 2.5 and not self.needs_tier3(tick):
            # Normal trade - use formula sizing
            return self.execute_with_formula(ml_result, formula)

        # Tier 3: Kimi-K2 for complex decisions (~2s)
        decision = await self.trading_brain.decide(tick, formula)
        return self.execute(decision)
```

---

## Cost & Speed Summary

| Tier | Calls/Day | Latency | Cost/Day |
|------|-----------|---------|----------|
| **Tier 1** | 100,000+ | <50ms | $0 (local) |
| **Tier 2** | ~5,000 | ~2s | $0 (local) |
| **Tier 3** | ~50-100 | ~2s | ~$0.05 |
| **TOTAL** | | | **~$0.05/day** |

---

## Training Schedule

| Component | Frequency | Where | Cost |
|-----------|-----------|-------|------|
| ML Ensemble | Continuous | RTX 5080 | $0 |
| Forex-R1 | Weekly | Vast.ai H200 | ~$1.50/week |
| Kimi-K2 | Never (API) | - | API only |

---

## Files to Create/Update

| File | Change |
|------|--------|
| `core/ml/three_tier_trader.py` | New orchestrator |
| `core/ml/formula_brain.py` | Add forex context |
| `core/ml/trading_brain.py` | Add tier 3 triggers |
| `scripts/hft_trading_bot.py` | Integrate 3-tier |
