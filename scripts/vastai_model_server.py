"""
Vast.ai Model Server for Forex Trading

Runs on vast.ai GPU instance, serves ML model predictions via HTTP
Reads configuration from Oracle Cloud MCP server API

Deploy to vast.ai and run:
    python vastai_model_server.py --mcp-api http://89.168.65.47:8080

Client (paper trading bot) sends features, server returns predictions
"""

from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
import requests
import logging
from pathlib import Path
import json
from datetime import datetime
import os

# Setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Global state
MODELS = {}  # {pair: model}
MCP_API_URL = None


def load_model_from_mcp(pair: str) -> tf.keras.Model:
    """Download and load model from Oracle Cloud via MCP API"""
    try:
        logger.info(f"Fetching model for {pair} from MCP server...")

        # Search for model config in memory
        response = requests.post(
            f"{MCP_API_URL}/api/memory/search",
            json={'query': f'model {pair}', 'category': 'note', 'limit': 5},
            timeout=10
        )

        if response.status_code == 200:
            results = response.json()
            logger.info(f"Found {len(results.get('items', []))} model configs")

        # Get list of available models
        response = requests.get(
            f"{MCP_API_URL}/api/models/list",
            timeout=10
        )

        if response.status_code != 200:
            raise Exception(f"Failed to list models: {response.text}")

        models_data = response.json()
        models = models_data.get('models', [])

        # Find model for this pair
        model_name = None
        for model in models:
            if pair.upper() in model['name'].upper():
                model_name = model['name']
                break

        if not model_name:
            raise Exception(f"No model found for {pair}")

        logger.info(f"Downloading model: {model_name}")

        # Download model file
        response = requests.get(
            f"{MCP_API_URL}/api/models/download/{model_name}",
            timeout=60
        )

        if response.status_code != 200:
            raise Exception(f"Failed to download model: {response.text}")

        # Save to local file
        local_path = Path(f"/workspace/{model_name}")
        local_path.write_bytes(response.content)

        logger.info(f"Model saved to {local_path}")

        # Load model
        model = tf.keras.models.load_model(str(local_path))
        logger.info(f"Model loaded: {model.summary()}")

        return model

    except Exception as e:
        logger.error(f"Failed to load model for {pair}: {e}")
        return None


def initialize_models(pairs: list):
    """Load models for all trading pairs"""
    logger.info("Initializing models...")

    for pair in pairs:
        try:
            model = load_model_from_mcp(pair)
            if model:
                MODELS[pair] = model
                logger.info(f"✓ {pair} model loaded")
            else:
                logger.warning(f"✗ {pair} model not available")
        except Exception as e:
            logger.error(f"✗ {pair} model failed: {e}")

    logger.info(f"Loaded {len(MODELS)}/{len(pairs)} models")


@app.route('/health', methods=['GET'])
def health_check():
    """Health check"""
    return jsonify({
        'status': 'healthy',
        'models_loaded': len(MODELS),
        'pairs': list(MODELS.keys()),
        'timestamp': datetime.utcnow().isoformat()
    })


@app.route('/predict', methods=['POST'])
def predict():
    """
    Make prediction

    POST /predict
    {
        "pair": "EURUSD",
        "features": {
            "returns_1": 0.0001,
            "returns_5": 0.0005,
            "returns_20": 0.001,
            "ma_5": 1.09845,
            "ma_20": 1.09820,
            "volatility": 0.0002,
            "spread": 0.00002,
            "avg_spread": 0.00002,
            "vol_imbalance": 0.00001
        }
    }

    Returns:
    {
        "pair": "EURUSD",
        "signal": "long",  # or "short" or "neutral"
        "confidence": 0.75,
        "prediction_value": 0.625,  # raw model output
        "timestamp": "2026-01-08T12:34:56"
    }
    """
    try:
        data = request.json
        pair = data.get('pair')
        features = data.get('features')

        if not pair or not features:
            return jsonify({'error': 'pair and features required'}), 400

        # Check if model loaded
        if pair not in MODELS:
            return jsonify({'error': f'Model not loaded for {pair}'}), 404

        model = MODELS[pair]

        # Convert features dict to array
        feature_array = np.array([[
            features.get('returns_1', 0),
            features.get('returns_5', 0),
            features.get('returns_20', 0),
            features.get('ma_5', 0),
            features.get('ma_20', 0),
            features.get('volatility', 0),
            features.get('spread', 0),
            features.get('avg_spread', 0),
            features.get('vol_imbalance', 0)
        ]], dtype=np.float32)

        # Make prediction
        prediction = model.predict(feature_array, verbose=0)[0][0]

        # Convert to signal
        if prediction > 0.55:
            signal = 'long'
        elif prediction < 0.45:
            signal = 'short'
        else:
            signal = 'neutral'

        confidence = abs(prediction - 0.5) * 2  # 0 to 1

        result = {
            'pair': pair,
            'signal': signal,
            'confidence': float(confidence),
            'prediction_value': float(prediction),
            'timestamp': datetime.utcnow().isoformat()
        }

        logger.info(f"{pair}: {signal} (conf={confidence:.2f}, pred={prediction:.4f})")

        return jsonify(result)

    except Exception as e:
        logger.error(f"Prediction error: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500


@app.route('/batch_predict', methods=['POST'])
def batch_predict():
    """
    Batch prediction for multiple pairs

    POST /batch_predict
    {
        "predictions": [
            {"pair": "EURUSD", "features": {...}},
            {"pair": "GBPUSD", "features": {...}}
        ]
    }
    """
    try:
        data = request.json
        predictions_input = data.get('predictions', [])

        results = []
        for item in predictions_input:
            pair = item.get('pair')
            features = item.get('features')

            if pair in MODELS:
                # Use single prediction endpoint logic
                pred_response = predict_single(pair, features)
                results.append(pred_response)
            else:
                results.append({
                    'pair': pair,
                    'error': 'Model not loaded'
                })

        return jsonify({'results': results})

    except Exception as e:
        return jsonify({'error': str(e)}), 500


def predict_single(pair: str, features: dict) -> dict:
    """Helper for single prediction"""
    model = MODELS[pair]

    feature_array = np.array([[
        features.get('returns_1', 0),
        features.get('returns_5', 0),
        features.get('returns_20', 0),
        features.get('ma_5', 0),
        features.get('ma_20', 0),
        features.get('volatility', 0),
        features.get('spread', 0),
        features.get('avg_spread', 0),
        features.get('vol_imbalance', 0)
    ]], dtype=np.float32)

    prediction = model.predict(feature_array, verbose=0)[0][0]

    if prediction > 0.55:
        signal = 'long'
    elif prediction < 0.45:
        signal = 'short'
    else:
        signal = 'neutral'

    confidence = abs(prediction - 0.5) * 2

    return {
        'pair': pair,
        'signal': signal,
        'confidence': float(confidence),
        'prediction_value': float(prediction)
    }


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Vast.ai Model Server')
    parser.add_argument('--mcp-api', required=True, help='MCP API URL (e.g., http://89.168.65.47:8080)')
    parser.add_argument('--port', type=int, default=5000, help='Port to listen on')
    parser.add_argument('--pairs', nargs='+', default=['EURUSD', 'GBPUSD', 'USDJPY', 'AUDUSD', 'USDCAD'],
                        help='Pairs to load models for')

    args = parser.parse_args()

    global MCP_API_URL
    MCP_API_URL = args.mcp_api

    logger.info("="*60)
    logger.info("VAST.AI MODEL SERVER")
    logger.info(f"MCP API: {MCP_API_URL}")
    logger.info(f"Port: {args.port}")
    logger.info(f"Pairs: {args.pairs}")
    logger.info("="*60)

    # Test MCP connection
    try:
        response = requests.get(f"{MCP_API_URL}/health", timeout=5)
        if response.status_code == 200:
            logger.info("✓ MCP API connection successful")
        else:
            logger.error("✗ MCP API returned non-200 status")
            return 1
    except Exception as e:
        logger.error(f"✗ Cannot connect to MCP API: {e}")
        return 1

    # Load models
    initialize_models(args.pairs)

    if len(MODELS) == 0:
        logger.error("No models loaded, cannot start server")
        return 1

    # Start server
    logger.info("="*60)
    logger.info("Server starting...")
    logger.info(f"Inference endpoint: http://0.0.0.0:{args.port}/predict")
    logger.info("="*60)

    app.run(host='0.0.0.0', port=args.port, debug=False, threaded=True)


if __name__ == "__main__":
    exit(main())
