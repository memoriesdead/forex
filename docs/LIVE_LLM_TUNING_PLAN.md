# Live LLM Tuning Plan (Chinese Quant Style)

## Goal

**Continuous online learning for forex-r1-v2** - the LLM adapts in real-time based on live trading outcomes, exactly like 幻方量化, 九坤投资, 明汯投资.

```
╔══════════════════════════════════════════════════════════════════════════════════╗
║                    LIVE LLM TUNING ARCHITECTURE                                  ║
║                    Chinese Quant Style (幻方/九坤/明汯)                           ║
╠══════════════════════════════════════════════════════════════════════════════════╣
║                                                                                  ║
║   Live Trading                    Background Tuning                              ║
║   ────────────                    ─────────────────                              ║
║   Tick → ML Ensemble (63%)        Every 4 hours:                                 ║
║        ↓                          ┌─────────────────────────┐                    ║
║   LLM Validation (forex-r1-v2)    │ 1. Collect outcomes     │                    ║
║        ↓                          │ 2. Generate DPO pairs   │                    ║
║   Execute Trade                   │ 3. LoRA fine-tune       │                    ║
║        ↓                          │ 4. Validate on holdout  │                    ║
║   Record Outcome ───────────────► │ 5. Hot-swap if better   │                    ║
║                                   └─────────────────────────┘                    ║
║                                            ↓                                     ║
║                                   New LoRA Adapter                               ║
║                                            ↓                                     ║
║                                   Ollama Hot-Reload                              ║
║                                                                                  ║
╚══════════════════════════════════════════════════════════════════════════════════╝
```

---

## Current State

| Component | Status | Location |
|-----------|--------|----------|
| forex-r1-v2 base | ✅ Trained | `models/forex-r1-v2/` (16GB) |
| LoRA adapter | ✅ Trained | `models/forex-r1-v2/lora_adapter/` (1.3GB) |
| ML Ensemble | ✅ Live tuning | `core/ml/chinese_online_learning.py` |
| LLM Reasoner | ⚠️ Static | `core/ml/llm_reasoner.py` |
| Live Tuning MCP | ❌ Missing | Need to create |

---

## Architecture: 3-Layer Adaptive System

### Layer 1: Immediate Adaptation (RAG - milliseconds)

```python
# Vector store of successful trade reasoning
# Retrieved at inference time for context

class LiveTradeMemory:
    """Store and retrieve successful trade patterns"""

    def store_outcome(self, signal, reasoning, outcome, pnl):
        """Store trade with outcome for retrieval"""
        embedding = self.embed(f"{signal} {reasoning}")
        self.vector_db.add(embedding, {
            "signal": signal,
            "reasoning": reasoning,
            "outcome": outcome,
            "pnl": pnl,
            "timestamp": time.time()
        })

    def get_similar_trades(self, current_signal, k=3):
        """Retrieve similar past trades for context"""
        embedding = self.embed(current_signal)
        return self.vector_db.search(embedding, k=k)
```

**Speed:** <10ms retrieval
**Learning:** Immediate pattern matching

### Layer 2: Session Adaptation (Prompt Engineering - seconds)

```python
# Accumulate session stats and inject into system prompt

class SessionAdapter:
    """Track session performance and adapt prompts"""

    def __init__(self):
        self.session_stats = {
            "trades": 0,
            "wins": 0,
            "losses": 0,
            "regime": "unknown",
            "hot_patterns": [],
            "cold_patterns": []
        }

    def get_adapted_prompt(self):
        """Generate prompt with session context"""
        return f"""
Current Session Stats:
- Win rate: {self.win_rate:.1%}
- Market regime: {self.regime}
- Working patterns: {self.hot_patterns}
- Avoid patterns: {self.cold_patterns}

Adjust your analysis accordingly.
"""
```

**Speed:** <100ms
**Learning:** Within-session adaptation

### Layer 3: Deep Adaptation (LoRA Tuning - hours)

```python
# Background LoRA fine-tuning every 4 hours

class LiveLoRATuner:
    """Chinese quant style continuous LoRA updates"""

    def __init__(self, base_model_path, lora_path):
        self.base_model_path = base_model_path
        self.current_lora = lora_path
        self.outcome_buffer = []  # Accumulate trade outcomes
        self.min_samples = 500    # Minimum before retraining

    def record_outcome(self, signal, reasoning, outcome, pnl):
        """Buffer trade outcomes for training"""
        self.outcome_buffer.append({
            "prompt": signal,
            "chosen": reasoning if outcome > 0 else None,
            "rejected": reasoning if outcome < 0 else None,
            "reward": pnl
        })

    def should_retrain(self):
        """Check if we have enough data for retraining"""
        return len(self.outcome_buffer) >= self.min_samples

    async def retrain_lora(self):
        """Background LoRA fine-tuning on RTX 5080"""
        # Generate DPO pairs from outcomes
        dpo_pairs = self.generate_dpo_pairs()

        # Fine-tune LoRA (uses local GPU)
        new_lora = await self.train_lora_increment(dpo_pairs)

        # Validate on holdout
        if self.validate(new_lora) > self.validate(self.current_lora):
            self.hot_swap_lora(new_lora)

    def hot_swap_lora(self, new_lora_path):
        """Atomically swap LoRA adapter in Ollama"""
        # 1. Save new adapter
        # 2. Update Modelfile
        # 3. Recreate Ollama model
        # 4. Trading continues with new model
```

**Speed:** 30-60 min training, instant swap
**Learning:** Deep pattern learning from outcomes

---

## MCP Server Design

### New Endpoints

```python
# mcp_servers/llm_live_tuning_mcp.py

@app.route('/api/llm/record_outcome', methods=['POST'])
def record_outcome():
    """Record trade outcome for learning"""
    data = request.json
    # {symbol, signal, ml_confidence, llm_reasoning, llm_decision,
    #  actual_outcome, pnl, timestamp}
    tuner.record_outcome(data)
    return {"status": "recorded", "buffer_size": len(tuner.outcome_buffer)}

@app.route('/api/llm/get_context', methods=['POST'])
def get_context():
    """Get RAG context for current signal"""
    signal = request.json['signal']
    similar = memory.get_similar_trades(signal, k=3)
    session = adapter.get_adapted_prompt()
    return {"similar_trades": similar, "session_context": session}

@app.route('/api/llm/trigger_retrain', methods=['POST'])
def trigger_retrain():
    """Manually trigger LoRA retraining"""
    if tuner.should_retrain():
        task_id = tuner.start_background_retrain()
        return {"status": "started", "task_id": task_id}
    return {"status": "insufficient_data", "current": len(tuner.outcome_buffer)}

@app.route('/api/llm/get_tuning_stats', methods=['GET'])
def get_tuning_stats():
    """Get live tuning statistics"""
    return {
        "current_lora_version": tuner.current_version,
        "buffer_size": len(tuner.outcome_buffer),
        "last_retrain": tuner.last_retrain_time,
        "win_rate_improvement": tuner.calculate_improvement(),
        "next_retrain_eta": tuner.estimate_next_retrain()
    }

@app.route('/api/llm/lora_versions', methods=['GET'])
def list_lora_versions():
    """List all LoRA adapter versions"""
    return {"versions": tuner.list_versions()}

@app.route('/api/llm/rollback', methods=['POST'])
def rollback_lora():
    """Rollback to previous LoRA version"""
    version = request.json['version']
    tuner.rollback_to(version)
    return {"status": "rolled_back", "current": tuner.current_version}
```

### Background Tasks

```python
# Scheduled tasks

@scheduler.task('interval', hours=4)
async def scheduled_retrain():
    """Every 4 hours, check and retrain if needed"""
    if tuner.should_retrain():
        await tuner.retrain_lora()

@scheduler.task('interval', minutes=30)
async def drift_detection():
    """Check for concept drift"""
    if tuner.detect_drift():
        # More aggressive retraining
        tuner.min_samples = 200  # Lower threshold
        await tuner.retrain_lora()
        tuner.min_samples = 500  # Reset
```

---

## Data Flow

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                              LIVE TRADING LOOP                                  │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│   1. Tick arrives                                                               │
│      ↓                                                                          │
│   2. ML Ensemble predicts (63% accuracy)                                        │
│      ↓                                                                          │
│   3. [NEW] Fetch RAG context from LiveTradeMemory                               │
│      ↓                                                                          │
│   4. [NEW] Get session adaptation prompt                                        │
│      ↓                                                                          │
│   5. LLM validates with context (forex-r1-v2 + LoRA)                            │
│      ↓                                                                          │
│   6. Execute or veto trade                                                      │
│      ↓                                                                          │
│   7. [NEW] Wait for outcome (next tick or timeout)                              │
│      ↓                                                                          │
│   8. [NEW] Record outcome to:                                                   │
│      ├── LiveTradeMemory (immediate RAG updates)                                │
│      ├── SessionAdapter (session stats)                                         │
│      └── LiveLoRATuner (buffer for retraining)                                  │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────────┐
│                           BACKGROUND TUNING LOOP                                │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│   Every 4 hours (or 500+ outcomes):                                             │
│                                                                                 │
│   1. Check buffer size >= 500                                                   │
│      ↓                                                                          │
│   2. Generate DPO pairs from outcomes                                           │
│      - Chosen: reasoning that led to profit                                     │
│      - Rejected: reasoning that led to loss                                     │
│      ↓                                                                          │
│   3. LoRA fine-tune on RTX 5080 (30-60 min)                                     │
│      - Load base model + current LoRA                                           │
│      - Train new LoRA delta                                                     │
│      - Merge: current_lora + delta = new_lora                                   │
│      ↓                                                                          │
│   4. Validate on holdout set (last 100 outcomes)                                │
│      ↓                                                                          │
│   5. If improved:                                                               │
│      - Save new LoRA with version tag                                           │
│      - Update Ollama Modelfile                                                  │
│      - Hot-reload model (ollama create forex-r1-v2-live)                        │
│      - Trading continues with new model                                         │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
```

---

## Files to Create

| File | Purpose |
|------|---------|
| `core/ml/live_llm_tuner.py` | LiveLoRATuner, LiveTradeMemory, SessionAdapter |
| `core/ml/lora_trainer.py` | Local RTX 5080 LoRA training |
| `mcp_servers/llm_live_tuning_mcp.py` | MCP endpoints for live tuning |
| `scripts/start_live_tuning.py` | Start live tuning system |

---

## Implementation Phases

### Phase 1: RAG Layer (2 hours)

1. Create `LiveTradeMemory` with ChromaDB/FAISS
2. Integrate into `llm_reasoner.py`
3. Store outcomes, retrieve similar trades

### Phase 2: Session Adapter (1 hour)

1. Create `SessionAdapter` class
2. Track session stats
3. Generate adaptive prompts

### Phase 3: LoRA Tuner (4 hours)

1. Create `LiveLoRATuner` with outcome buffer
2. Create `lora_trainer.py` for RTX 5080 training
3. Implement hot-swap mechanism
4. Version management

### Phase 4: MCP Server (2 hours)

1. Create `llm_live_tuning_mcp.py`
2. Add all endpoints
3. Background scheduler integration

### Phase 5: Integration (2 hours)

1. Update `hft_trading_bot.py` to record outcomes
2. Connect all layers
3. End-to-end testing

---

## Hardware Requirements

| Component | Requirement |
|-----------|-------------|
| Base model | 16GB disk (forex-r1-v2 safetensors) |
| LoRA training | RTX 5080 16GB (local) |
| Vector DB | ~1GB RAM (ChromaDB) |
| Outcome buffer | ~100MB RAM |
| Training time | 30-60 min per cycle |

---

## Chinese Quant References

| Firm | Technique | Our Implementation |
|------|-----------|-------------------|
| 幻方量化 | 萤火平台实时更新 | LiveLoRATuner background updates |
| 九坤投资 | AI Lab增量学习 | LoRA delta training |
| 明汯投资 | 400P Flops实时计算 | RTX 5080 local training |

---

## Success Metrics

| Metric | Target |
|--------|--------|
| RAG retrieval | <10ms |
| Session adaptation | Real-time |
| LoRA retrain cycle | Every 4 hours |
| Win rate improvement | +5% per week |
| Model drift detection | <1 hour |

---

## Risk Mitigation

| Risk | Mitigation |
|------|------------|
| Overfitting to recent data | Holdout validation, version rollback |
| Training during peak hours | Schedule for low-activity periods |
| Bad LoRA degrades performance | Automatic rollback if win rate drops |
| GPU contention with trading | Training uses separate CUDA stream |

