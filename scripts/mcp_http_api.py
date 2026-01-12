"""
MCP Server HTTP API for Vast.ai Access

Exposes memory-keeper and claude-mem MCP servers via HTTP REST API
Allows vast.ai instances to read training configs, model metadata, etc.

Deploy to Oracle Cloud and expose via public port for vast.ai access

Usage:
    python scripts/mcp_http_api.py --port 8080 --host 0.0.0.0
"""

from flask import Flask, jsonify, request
from flask_cors import CORS
import subprocess
import json
import os
from pathlib import Path
from datetime import datetime
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)  # Allow cross-origin requests from vast.ai

# MCP server configuration
PROJECT_DIR = Path("/home/ubuntu/projects/forex")
MEMORY_KEEPER_PORT = 3847


def query_memory_keeper(method: str, params: dict) -> dict:
    """Query memory-keeper MCP server via stdio"""
    try:
        # Use mcp CLI to query memory-keeper
        cmd = [
            'node',
            f'{PROJECT_DIR}/mcp_servers/memory-keeper-client.js',
            method,
            json.dumps(params)
        ]

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=10
        )

        if result.returncode == 0:
            return json.loads(result.stdout)
        else:
            logger.error(f"Memory-keeper query failed: {result.stderr}")
            return {'error': result.stderr}

    except Exception as e:
        logger.error(f"Exception querying memory-keeper: {e}")
        return {'error': str(e)}


@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.utcnow().isoformat(),
        'service': 'mcp-http-api'
    })


@app.route('/api/memory/search', methods=['POST'])
def search_memory():
    """
    Search memory-keeper context

    POST /api/memory/search
    {
        "query": "training config",
        "category": "note",
        "limit": 10
    }
    """
    try:
        data = request.json
        query = data.get('query', '')
        category = data.get('category')
        limit = data.get('limit', 10)

        params = {
            'query': query,
            'limit': limit,
            'sessionId': None  # Search across all sessions
        }

        if category:
            params['category'] = category

        result = query_memory_keeper('context_search', params)

        return jsonify(result)

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/memory/get', methods=['POST'])
def get_memory():
    """
    Get specific memory item by key

    POST /api/memory/get
    {
        "key": "model_config_eurusd"
    }
    """
    try:
        data = request.json
        key = data.get('key')

        if not key:
            return jsonify({'error': 'key parameter required'}), 400

        result = query_memory_keeper('context_get', {'key': key})

        return jsonify(result)

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/memory/list-channels', methods=['GET'])
def list_channels():
    """List all available channels"""
    try:
        result = query_memory_keeper('context_list_channels', {})
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/models/list', methods=['GET'])
def list_models():
    """List available trained models"""
    try:
        models_dir = PROJECT_DIR / "models"

        if not models_dir.exists():
            return jsonify({'models': []})

        models = []
        for model_file in models_dir.glob('*.h5'):
            stat = model_file.stat()
            models.append({
                'name': model_file.name,
                'path': str(model_file),
                'size_mb': round(stat.st_size / (1024 * 1024), 2),
                'modified': datetime.fromtimestamp(stat.st_mtime).isoformat()
            })

        return jsonify({'models': models})

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/models/download/<model_name>', methods=['GET'])
def download_model(model_name: str):
    """Download model file (for vast.ai to pull)"""
    try:
        models_dir = PROJECT_DIR / "models"
        model_path = models_dir / model_name

        if not model_path.exists():
            return jsonify({'error': 'Model not found'}), 404

        # Return file
        from flask import send_file
        return send_file(model_path, as_attachment=True)

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/data/latest', methods=['GET'])
def get_latest_data():
    """Get latest captured data info"""
    try:
        data_dir = PROJECT_DIR / "data" / "live"
        today = datetime.utcnow().strftime('%Y-%m-%d')
        today_dir = data_dir / today

        if not today_dir.exists():
            return jsonify({'error': 'No data for today'}), 404

        files = []
        for csv_file in today_dir.glob('*.csv'):
            stat = csv_file.stat()

            # Read last line to get latest timestamp
            try:
                with open(csv_file, 'r') as f:
                    lines = f.readlines()
                    if len(lines) > 1:
                        last_line = lines[-1]
                        timestamp = last_line.split(',')[0]
                    else:
                        timestamp = None
            except:
                timestamp = None

            files.append({
                'pair': csv_file.stem.split('_')[0],
                'file': csv_file.name,
                'size_mb': round(stat.st_size / (1024 * 1024), 2),
                'rows': len(lines) - 1 if lines else 0,
                'latest_timestamp': timestamp
            })

        return jsonify({
            'date': today,
            'files': files
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/config/trading', methods=['GET'])
def get_trading_config():
    """Get trading configuration"""
    try:
        # Read from memory-keeper
        result = query_memory_keeper('context_search', {
            'query': 'trading config',
            'category': 'note',
            'limit': 10
        })

        return jsonify(result)

    except Exception as e:
        return jsonify({'error': str(e)}), 500


def main():
    import argparse

    parser = argparse.ArgumentParser(description='MCP HTTP API Server')
    parser.add_argument('--port', type=int, default=8080, help='Port to listen on')
    parser.add_argument('--host', default='0.0.0.0', help='Host to bind to')

    args = parser.parse_args()

    logger.info("="*60)
    logger.info("MCP HTTP API SERVER")
    logger.info(f"Listening on {args.host}:{args.port}")
    logger.info(f"Project: {PROJECT_DIR}")
    logger.info("="*60)

    app.run(host=args.host, port=args.port, debug=False)


if __name__ == "__main__":
    main()
