"""
LLM Live Tuning MCP Server

Provides API endpoints for Chinese quant style continuous learning:
- Record trade outcomes
- Trigger LoRA training
- Monitor drift detection
- Manage model versions

CITATIONS:
-----------
[1] 幻方量化: "用人工智能技术深度分析数据，同时及时应对市场规则变化，不断更新模型。"
    - https://blog.csdn.net/zk168_net/article/details/108076246

[2] 九坤投资: "基于可解释AI模型实现因子选择和因子组合在风格变化中的自动切换。"
    - https://news.qq.com/rain/a/20250122A085K600

[3] DeepSeek GRPO: "引入了带有组相对策略优化的迭代强化学习方法，包含重播机制，
    历史数据占比10%。"
    - https://zhuanlan.zhihu.com/p/21046265072

[4] BigQuant滚动训练: "为了避免策略失效，定期更新训练集数据，通过滚动训练的方式
    更新预测模型以适应最新市场行情。"
    - https://bigquant.com/wiki/doc/xVqIPu6RoI

Author: Claude Code (forex-r1-v2 live tuning system)
Date: 2026-01-21

Usage:
    python mcp_servers/llm_live_tuning_mcp.py --port 8082
"""

import os
import sys
import json
import argparse
from pathlib import Path
from datetime import datetime
from flask import Flask, request, jsonify
from flask_cors import CORS

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.ml.trade_outcome_buffer import (
    TradeOutcomeBuffer,
    TradeOutcome,
    get_outcome_buffer,
)
from core.ml.drift_detector import (
    DriftDetector,
    DriftAlert,
    DriftType,
    MarketRegime,
    get_drift_detector,
)
from core.ml.live_lora_tuner import (
    LiveLoRATuner,
    LoRAVersion,
    get_live_tuner,
)

app = Flask(__name__)
CORS(app)

# Global instances
outcome_buffer: TradeOutcomeBuffer = None
drift_detector: DriftDetector = None
lora_tuner: LiveLoRATuner = None


def init_components():
    """Initialize all components."""
    global outcome_buffer, drift_detector, lora_tuner

    outcome_buffer = get_outcome_buffer()
    drift_detector = get_drift_detector()

    # Initialize tuner with paths - forex-r1-v3 (2026-01-22 trained on 8x H100)
    base_model = Path("models/forex-r1-v3/merged_model")
    lora_path = Path("models/forex-r1-v3/lora_adapter")

    if base_model.exists():
        lora_tuner = get_live_tuner(base_model, lora_path)
        print(f"[MCP] Initialized with forex-r1-v3 model")
    else:
        print(f"[MCP] Warning: Base model not found at {base_model}")
        lora_tuner = None


# ============================================================================
# Health & Status Endpoints
# ============================================================================

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint."""
    return jsonify({
        "status": "healthy",
        "service": "llm-live-tuning",
        "timestamp": datetime.now().isoformat(),
    })


@app.route('/api/tuning/status', methods=['GET'])
def get_status():
    """
    Get overall status of live tuning system.

    Citation [1]: 幻方量化 monitors model performance continuously.
    """
    status = {
        "timestamp": datetime.now().isoformat(),
        "components": {
            "outcome_buffer": outcome_buffer is not None,
            "drift_detector": drift_detector is not None,
            "lora_tuner": lora_tuner is not None,
        },
    }

    if outcome_buffer:
        status["buffer"] = outcome_buffer.get_stats()

    if drift_detector:
        status["drift"] = drift_detector.get_stats()

    if lora_tuner:
        status["tuner"] = lora_tuner.get_stats()

    return jsonify(status)


# ============================================================================
# Trade Outcome Recording
# ============================================================================

@app.route('/api/tuning/record_outcome', methods=['POST'])
def record_outcome():
    """
    Record a trade outcome for learning.

    Citation [3]: DeepSeek uses trade outcomes to generate DPO pairs.
    Citation [4]: BigQuant accumulates data for rolling retraining.

    Request body:
    {
        "symbol": "EURUSD",
        "ml_direction": 1,  // 1=BUY, -1=SELL
        "ml_confidence": 0.65,
        "llm_reasoning": "<think>...</think>",
        "llm_decision": "APPROVE",
        "llm_confidence": 0.7,
        "actual_direction": 1,  // 1=UP, -1=DOWN
        "pnl_pips": 5.2,
        "pnl_dollars": 52.0
    }
    """
    try:
        data = request.json

        # Record to buffer
        outcome = outcome_buffer.record(
            symbol=data["symbol"],
            ml_direction=data["ml_direction"],
            ml_confidence=data["ml_confidence"],
            llm_reasoning=data.get("llm_reasoning", ""),
            llm_decision=data.get("llm_decision", "APPROVE"),
            llm_confidence=data.get("llm_confidence", 0.5),
            actual_direction=data["actual_direction"],
            pnl_pips=data.get("pnl_pips", 0.0),
            pnl_dollars=data.get("pnl_dollars", 0.0),
        )

        # Also feed to drift detector
        prediction = data["ml_confidence"] if data["ml_direction"] == 1 else 1 - data["ml_confidence"]
        actual = 1.0 if data["actual_direction"] == 1 else 0.0
        returns = data.get("pnl_dollars", 0.0) / 1000  # Normalize

        alert = drift_detector.add_observation(prediction, actual, returns)

        response = {
            "status": "recorded",
            "outcome": {
                "was_correct": outcome.was_correct,
                "reward": outcome.reward,
            },
            "buffer_size": outcome_buffer.get_buffer_size(),
            "ready_for_training": outcome_buffer.should_trigger_training(),
        }

        if alert:
            response["drift_alert"] = {
                "type": alert.drift_type.value,
                "severity": alert.severity,
                "metric": alert.metric,
                "recommendation": alert.recommendation,
            }

        return jsonify(response)

    except Exception as e:
        return jsonify({"error": str(e)}), 400


@app.route('/api/tuning/buffer_stats', methods=['GET'])
def get_buffer_stats():
    """Get outcome buffer statistics."""
    return jsonify(outcome_buffer.get_stats())


# ============================================================================
# Drift Detection
# ============================================================================

@app.route('/api/tuning/drift_status', methods=['GET'])
def get_drift_status():
    """
    Get drift detection status.

    Citation [2]: 九坤投资 auto-switches based on style changes.
    """
    stats = drift_detector.get_stats()
    alerts = drift_detector.get_recent_alerts(5)

    return jsonify({
        "stats": stats,
        "current_regime": drift_detector.get_current_regime().value,
        "recent_alerts": [
            {
                "timestamp": datetime.fromtimestamp(a.timestamp).isoformat(),
                "type": a.drift_type.value,
                "severity": a.severity,
                "metric": a.metric,
                "recommendation": a.recommendation,
            }
            for a in alerts
        ]
    })


@app.route('/api/tuning/check_drift', methods=['POST'])
def check_drift():
    """Force drift check."""
    alert = drift_detector._check_all_drift()

    if alert:
        return jsonify({
            "drift_detected": True,
            "alert": {
                "type": alert.drift_type.value,
                "severity": alert.severity,
                "metric": alert.metric,
                "current_value": alert.current_value,
                "baseline_value": alert.baseline_value,
                "recommendation": alert.recommendation,
            }
        })

    return jsonify({"drift_detected": False})


# ============================================================================
# LoRA Training
# ============================================================================

@app.route('/api/tuning/trigger_training', methods=['POST'])
def trigger_training():
    """
    Manually trigger LoRA training.

    Citation [1]: 幻方量化 continuously updates models.
    Citation [4]: BigQuant recommends periodic model updates.
    """
    if lora_tuner is None:
        return jsonify({"error": "LoRA tuner not initialized"}), 500

    if not outcome_buffer.should_trigger_training():
        stats = outcome_buffer.get_stats()
        return jsonify({
            "status": "insufficient_data",
            "buffer_size": stats["buffer_size"],
            "ready": False,
            "message": "Need more trade outcomes before training",
        })

    started = lora_tuner.start_training_async()

    if started:
        return jsonify({
            "status": "training_started",
            "message": "Background training initiated",
        })
    else:
        return jsonify({
            "status": "already_training",
            "message": "Training already in progress",
        })


@app.route('/api/tuning/training_status', methods=['GET'])
def get_training_status():
    """Get current training status."""
    if lora_tuner is None:
        return jsonify({"error": "LoRA tuner not initialized"}), 500

    return jsonify(lora_tuner.get_stats())


@app.route('/api/tuning/should_train', methods=['GET'])
def should_train():
    """Check if training should be triggered."""
    if lora_tuner is None:
        return jsonify({"should_train": False, "reason": "tuner_not_initialized"})

    should = lora_tuner.should_train()
    stats = lora_tuner.get_stats()

    return jsonify({
        "should_train": should,
        "is_training": stats["is_training"],
        "buffer_ready": stats["buffer_ready"],
        "hours_since_training": stats["hours_since_training"],
    })


# ============================================================================
# Model Version Management
# ============================================================================

@app.route('/api/tuning/versions', methods=['GET'])
def list_versions():
    """
    List all LoRA adapter versions.

    Citation [6]: Maintain version history for rollback capability.
    """
    if lora_tuner is None:
        return jsonify({"error": "LoRA tuner not initialized"}), 500

    return jsonify({
        "current_version": lora_tuner._current_version,
        "versions": lora_tuner.list_versions(),
    })


@app.route('/api/tuning/rollback', methods=['POST'])
def rollback_version():
    """
    Rollback to a previous version.

    Citation [6]: "当在线更新导致性能下降时，快速回退至历史版本"
    """
    if lora_tuner is None:
        return jsonify({"error": "LoRA tuner not initialized"}), 500

    try:
        data = request.json
        version = data["version"]

        success = lora_tuner.rollback(version)

        if success:
            return jsonify({
                "status": "rolled_back",
                "version": version,
            })
        else:
            return jsonify({
                "status": "failed",
                "message": f"Could not rollback to version {version}",
            }), 400

    except Exception as e:
        return jsonify({"error": str(e)}), 400


# ============================================================================
# Export for Training
# ============================================================================

@app.route('/api/tuning/export_dpo', methods=['POST'])
def export_dpo():
    """
    Export DPO pairs for external training.

    Citation [3]: DeepSeek uses DPO pairs with 90% new + 10% historical.
    """
    try:
        data = request.json or {}
        output_path = Path(data.get("output_path", "data/live_training/dpo_export.jsonl"))

        num_pairs, num_historical = outcome_buffer.export_for_training(output_path)

        return jsonify({
            "status": "exported",
            "path": str(output_path),
            "total_pairs": num_pairs,
            "historical_pairs": num_historical,
            "new_pairs": num_pairs - num_historical,
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 400


# ============================================================================
# Automatic Training Loop
# ============================================================================

@app.route('/api/tuning/auto_train', methods=['POST'])
def configure_auto_train():
    """Configure automatic training."""
    try:
        data = request.json

        if lora_tuner:
            if "interval_hours" in data:
                lora_tuner.training_interval_hours = data["interval_hours"]
            if "min_samples" in data:
                lora_tuner.min_samples_for_training = data["min_samples"]

        return jsonify({
            "status": "configured",
            "interval_hours": lora_tuner.training_interval_hours if lora_tuner else None,
            "min_samples": lora_tuner.min_samples_for_training if lora_tuner else None,
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 400


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="LLM Live Tuning MCP Server")
    parser.add_argument("--port", type=int, default=8082, help="Port to run on")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    args = parser.parse_args()

    print(f"""
================================================================================
                    LLM LIVE TUNING MCP SERVER
                    Chinese Quant Style (HuanFang/JiuKun/MingHui)
================================================================================
  Port: {args.port}
  Endpoints:
    POST /api/tuning/record_outcome  - Record trade outcome
    GET  /api/tuning/status          - Get overall status
    GET  /api/tuning/drift_status    - Get drift detection status
    POST /api/tuning/trigger_training - Trigger LoRA training
    GET  /api/tuning/versions        - List model versions
    POST /api/tuning/rollback        - Rollback to previous version
================================================================================
  Citations:
  [1] HuanFang Quant - Continuous model updates
  [2] JiuKun Investment - Auto-switching based on style changes
  [3] DeepSeek GRPO - Iterative RL with replay mechanism
  [4] BigQuant - Rolling training best practices
================================================================================
""")

    # Initialize components
    init_components()

    # Run server
    app.run(host=args.host, port=args.port, debug=False)


if __name__ == "__main__":
    main()
