# C++ Hybrid Architecture Plan

## Current State (Python Only)
- Tick rate: ~1-5/sec per symbol
- Total: ~50-250 ticks/sec
- Python handling it, but no headroom

## Phase 1: Python Optimization (NOW)
Already implemented:
- Async buffered writes
- Background thread for disk I/O
- Per-symbol file rotation

## Phase 2: C++ Tick Capture Service (RECOMMENDED)

### Architecture
```
┌─────────────────────────────────────────────────────────────────────────┐
│                    HYBRID ARCHITECTURE                                  │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  C++ TICK CAPTURE SERVICE (New)                                         │
│  ├── UDP/WebSocket Feed Handler                                         │
│  ├── Lock-Free Ring Buffer (SPSC queue)                                 │
│  ├── Memory-Mapped File Writer (zero-copy)                              │
│  ├── Binary Tick Format (compact, fast)                                 │
│  └── ZeroMQ/IPC to Python                                               │
│                                                                         │
│  PYTHON ML ENGINE (Existing)                                            │
│  ├── Feature Engineering                                                │
│  ├── ML Predictions                                                     │
│  ├── Strategy Logic                                                     │
│  └── Order Management                                                   │
│                                                                         │
│  SHARED DATA                                                            │
│  ├── Memory-mapped tick files (C++ writes, Python reads)                │
│  ├── ZeroMQ for real-time tick stream                                   │
│  └── Redis for state (optional)                                         │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### C++ Libraries to Use
- **Boost.Asio** - Async networking
- **SPSCQueue** - Lock-free ring buffer (folly or custom)
- **FlatBuffers/Cap'n Proto** - Zero-copy serialization
- **ZeroMQ** - IPC to Python
- **mmap** - Memory-mapped file writes

### Binary Tick Format (32 bytes per tick)
```cpp
struct __attribute__((packed)) Tick {
    uint64_t timestamp_ns;  // 8 bytes - nanosecond precision
    uint32_t symbol_id;     // 4 bytes - symbol hash
    float bid;              // 4 bytes
    float ask;              // 4 bytes
    float volume;           // 4 bytes
    uint8_t source;         // 1 byte - TrueFX=1, OANDA=2, IB=3
    uint8_t flags;          // 1 byte - reserved
    uint16_t spread_bps;    // 2 bytes - spread in 0.1 bps
    uint32_t sequence;      // 4 bytes - for gap detection
};  // Total: 32 bytes (cache-line friendly)
```

### Performance Targets
| Metric | Python Current | C++ Target |
|--------|---------------|------------|
| Tick latency | ~1-10ms | <100µs |
| Max throughput | ~1000/sec | >100,000/sec |
| Data loss risk | Low | Zero |
| GC pauses | Yes | No |

## Phase 3: Rust Alternative (Optional)

Rust offers C++ performance with memory safety:
- No null pointer issues
- No data races (compile-time checked)
- Modern tooling (cargo)
- Great Python interop (PyO3)

### Rust Libraries
- **tokio** - Async runtime
- **crossbeam** - Lock-free data structures
- **mmap-rs** - Memory-mapped files
- **pyo3** - Python bindings

## Implementation Priority

1. **Week 1**: C++ tick capture service (standalone)
2. **Week 2**: ZeroMQ bridge to Python
3. **Week 3**: Memory-mapped file sharing
4. **Week 4**: Full integration + testing

## Open Source References

- [hftbacktest](https://github.com/nkaz001/hftbacktest) - Rust/Python hybrid HFT framework
- [LMAX Disruptor](https://github.com/LMAX-Exchange/disruptor) - Java, but pattern is universal
- [folly SPSC Queue](https://github.com/facebook/folly) - Facebook's lock-free queue
- [Low-Latency Market Data Processor](https://github.com/SourenaMOOSAVI/Low-Latency-Market-Data-Processor) - C++ example

## Decision Matrix

| Factor | Stay Python | Add C++ | Switch to Rust |
|--------|-------------|---------|----------------|
| Current needs | ✅ Sufficient | Overkill | Overkill |
| 100+ symbols | ❌ Risky | ✅ Safe | ✅ Safe |
| Development time | Fast | Medium | Medium |
| Maintenance | Easy | Harder | Medium |
| Memory safety | N/A | Manual | Automatic |

## Recommendation

**For now**: Python is fine at 50-250 ticks/sec.

**When to upgrade**:
- Adding 100+ symbols
- Moving to true HFT (sub-ms decisions)
- Experiencing data loss or GC pauses
- IB Gateway provides L2/L3 order book data

**What to build first** (if upgrading):
1. C++ tick capture → binary file → Python reads
2. This decouples capture from processing
3. Zero data loss even if Python is slow
