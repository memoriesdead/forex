"""
MCP Server for US Quant Gold Standard RL

Exposes reinforcement learning capabilities via MCP protocol:
- Multi-Agent RL (A3C, MAPPO, JPMorgan MARL)
- Offline RL (CQL, BCQ, CFCQL)
- World Model RL (Dreamer, SafeDreamer)
- Imitation Learning (BC, GAIL, DAgger)
- Meta-Learning (MAML, Reptile, RL²)

Usage:
    python scripts/mcp_rl_server.py --port 8081

MCP Tools:
    - rl_list_algorithms: List all available RL algorithms
    - rl_create_agent: Create an RL agent instance
    - rl_train: Train an agent on forex data
    - rl_predict: Get trading action from agent
    - rl_save_agent: Save agent to disk
    - rl_load_agent: Load agent from disk
"""

import json
import asyncio
import logging
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Dict, Any, Optional, List
from datetime import datetime
import pickle
import numpy as np

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add project root to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

# ═══════════════════════════════════════════════════════════════════════════════
# RL Algorithm Registry
# ═══════════════════════════════════════════════════════════════════════════════

RL_ALGORITHMS = {
    # Multi-Agent RL
    "a3c": {
        "name": "Asynchronous Advantage Actor-Critic",
        "module": "core.rl.multi_agent",
        "class": "A3CTrader",
        "category": "multi_agent",
        "paper": "Mnih et al. 2016 - Asynchronous Methods for Deep RL",
        "description": "Parallel workers with shared network, lock-free updates"
    },
    "mappo": {
        "name": "Multi-Agent PPO",
        "module": "core.rl.multi_agent",
        "class": "MAPPOTrader",
        "category": "multi_agent",
        "paper": "Yu et al. 2022 - The Surprising Effectiveness of PPO",
        "description": "CTDE paradigm with centralized critic, VDN mixer"
    },
    "jpmorgan_marl": {
        "name": "JPMorgan Multi-Agent RL",
        "module": "core.rl.multi_agent",
        "class": "JPMorganMARL",
        "category": "multi_agent",
        "paper": "JPMorgan AI Research 2023",
        "description": "Liquidity provider/taker agents with market impact"
    },

    # Offline RL
    "cql": {
        "name": "Conservative Q-Learning",
        "module": "core.rl.offline_rl",
        "class": "CQLTrader",
        "category": "offline",
        "paper": "Kumar et al. 2020 - Conservative Q-Learning",
        "description": "Penalizes OOD actions, safe for historical data"
    },
    "bcq": {
        "name": "Batch-Constrained Q-Learning",
        "module": "core.rl.offline_rl",
        "class": "BCQTrader",
        "category": "offline",
        "paper": "Fujimoto et al. 2019 - Off-Policy Deep RL",
        "description": "VAE for action generation, perturbation network"
    },
    "cfcql": {
        "name": "Counterfactual CQL",
        "module": "core.rl.offline_rl",
        "class": "CFCQLTrader",
        "category": "offline",
        "paper": "Two Sigma Research 2023",
        "description": "Multi-agent credit assignment with counterfactuals"
    },

    # World Model RL
    "dreamer": {
        "name": "DreamerV3",
        "module": "core.rl.world_model",
        "class": "DreamerTrader",
        "category": "world_model",
        "paper": "Hafner et al. 2023 - DreamerV3",
        "description": "RSSM world model, imagination-based planning"
    },
    "safe_dreamer": {
        "name": "SafeDreamer",
        "module": "core.rl.world_model",
        "class": "SafeDreamerTrader",
        "category": "world_model",
        "paper": "Wu et al. 2024 - Safe World Model RL",
        "description": "Lagrangian constraints for safe policy learning"
    },

    # Imitation Learning
    "bc": {
        "name": "Behavioral Cloning",
        "module": "core.rl.imitation",
        "class": "BehavioralCloningTrader",
        "category": "imitation",
        "paper": "Pomerleau 1991 - Efficient Training",
        "description": "Supervised learning from expert demonstrations"
    },
    "gail": {
        "name": "Generative Adversarial Imitation Learning",
        "module": "core.rl.imitation",
        "class": "GAILTrader",
        "category": "imitation",
        "paper": "Ho & Ermon 2016 - GAIL",
        "description": "Adversarial imitation with discriminator as reward"
    },
    "dagger": {
        "name": "Dataset Aggregation",
        "module": "core.rl.imitation",
        "class": "DAggerTrader",
        "category": "imitation",
        "paper": "Ross et al. 2011 - DAgger",
        "description": "Iterative expert querying for distribution shift"
    },

    # Meta-Learning
    "maml": {
        "name": "Model-Agnostic Meta-Learning",
        "module": "core.rl.meta_learning",
        "class": "MAMLTrader",
        "category": "meta",
        "paper": "Finn et al. 2017 - MAML",
        "description": "Second-order meta-learning, fast adaptation"
    },
    "reptile": {
        "name": "Reptile",
        "module": "core.rl.meta_learning",
        "class": "ReptileTrader",
        "category": "meta",
        "paper": "Nichol et al. 2018 - Reptile",
        "description": "First-order approximation of MAML"
    },
    "rl2": {
        "name": "RL²",
        "module": "core.rl.meta_learning",
        "class": "RL2Trader",
        "category": "meta",
        "paper": "Duan et al. 2016 - RL²",
        "description": "RNN-based meta-RL with implicit task encoding"
    },
}

# ═══════════════════════════════════════════════════════════════════════════════
# Agent Manager
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class AgentInstance:
    """Tracks a running agent instance"""
    id: str
    algorithm: str
    config: Dict[str, Any]
    created_at: str
    state_dim: int
    action_dim: int
    agent: Any = None


class RLAgentManager:
    """Manages RL agent lifecycle"""

    def __init__(self, models_dir: Path = None):
        self.agents: Dict[str, AgentInstance] = {}
        self.models_dir = models_dir or Path("models/rl_agents")
        self.models_dir.mkdir(parents=True, exist_ok=True)

    def list_algorithms(self) -> List[Dict[str, Any]]:
        """List all available RL algorithms"""
        return [
            {
                "id": algo_id,
                **{k: v for k, v in info.items() if k != "class"}
            }
            for algo_id, info in RL_ALGORITHMS.items()
        ]

    def create_agent(
        self,
        algorithm: str,
        state_dim: int = 575,
        action_dim: int = 3,
        config: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """Create a new RL agent instance"""

        if algorithm not in RL_ALGORITHMS:
            return {"error": f"Unknown algorithm: {algorithm}. Available: {list(RL_ALGORITHMS.keys())}"}

        algo_info = RL_ALGORITHMS[algorithm]

        try:
            # Dynamic import
            module = __import__(algo_info["module"], fromlist=[algo_info["class"]])
            agent_class = getattr(module, algo_info["class"])

            # Create agent
            agent_config = config or {}

            # Handle different constructor signatures
            if algorithm in ["a3c", "mappo"]:
                agent = agent_class(state_dim=state_dim, action_dim=action_dim)
            elif algorithm == "jpmorgan_marl":
                agent = agent_class(state_dim=state_dim, n_agents=agent_config.get("n_agents", 5))
            elif algorithm in ["cql", "bcq", "cfcql"]:
                agent = agent_class(state_dim=state_dim, action_dim=action_dim)
            elif algorithm in ["dreamer", "safe_dreamer"]:
                agent = agent_class(obs_dim=state_dim, action_dim=action_dim)
            elif algorithm in ["bc", "gail", "dagger"]:
                agent = agent_class(state_dim=state_dim, action_dim=action_dim)
            elif algorithm in ["maml", "reptile", "rl2"]:
                agent = agent_class(state_dim=state_dim, action_dim=action_dim)
            else:
                agent = agent_class(state_dim=state_dim)

            # Generate unique ID
            agent_id = f"{algorithm}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

            instance = AgentInstance(
                id=agent_id,
                algorithm=algorithm,
                config=agent_config,
                created_at=datetime.now().isoformat(),
                state_dim=state_dim,
                action_dim=action_dim,
                agent=agent
            )

            self.agents[agent_id] = instance

            return {
                "success": True,
                "agent_id": agent_id,
                "algorithm": algorithm,
                "name": algo_info["name"],
                "category": algo_info["category"],
                "state_dim": state_dim,
                "action_dim": action_dim
            }

        except Exception as e:
            logger.exception(f"Failed to create agent: {e}")
            return {"error": str(e)}

    def get_agent(self, agent_id: str) -> Optional[AgentInstance]:
        """Get agent instance by ID"""
        return self.agents.get(agent_id)

    def list_agents(self) -> List[Dict[str, Any]]:
        """List all active agents"""
        return [
            {
                "id": inst.id,
                "algorithm": inst.algorithm,
                "created_at": inst.created_at,
                "state_dim": inst.state_dim,
                "action_dim": inst.action_dim
            }
            for inst in self.agents.values()
        ]

    def predict(
        self,
        agent_id: str,
        state: List[float],
        deterministic: bool = True
    ) -> Dict[str, Any]:
        """Get trading action from agent"""

        instance = self.agents.get(agent_id)
        if not instance:
            return {"error": f"Agent not found: {agent_id}"}

        try:
            state_array = np.array(state, dtype=np.float32)

            # Different agents have different prediction methods
            agent = instance.agent
            algorithm = instance.algorithm

            if hasattr(agent, "select_action"):
                action = agent.select_action(state_array, deterministic=deterministic)
            elif hasattr(agent, "predict"):
                action = agent.predict(state_array)
            elif hasattr(agent, "act"):
                action = agent.act(state_array)
            else:
                return {"error": f"Agent has no prediction method"}

            # Convert action to trading signal
            if isinstance(action, (int, np.integer)):
                action_map = {0: "hold", 1: "buy", 2: "sell"}
                signal = action_map.get(int(action), "hold")
            elif isinstance(action, np.ndarray):
                if len(action) == 1:
                    signal = "buy" if action[0] > 0 else "sell" if action[0] < 0 else "hold"
                else:
                    signal = action_map.get(int(np.argmax(action)), "hold")
            else:
                signal = "hold"

            return {
                "success": True,
                "agent_id": agent_id,
                "action": int(action) if isinstance(action, (int, np.integer)) else action.tolist(),
                "signal": signal
            }

        except Exception as e:
            logger.exception(f"Prediction failed: {e}")
            return {"error": str(e)}

    def train_step(
        self,
        agent_id: str,
        batch: Dict[str, List]
    ) -> Dict[str, Any]:
        """Execute one training step"""

        instance = self.agents.get(agent_id)
        if not instance:
            return {"error": f"Agent not found: {agent_id}"}

        try:
            agent = instance.agent

            # Convert batch to numpy
            states = np.array(batch["states"], dtype=np.float32)
            actions = np.array(batch["actions"])
            rewards = np.array(batch["rewards"], dtype=np.float32)
            next_states = np.array(batch["next_states"], dtype=np.float32)
            dones = np.array(batch["dones"], dtype=np.float32)

            # Different training methods
            if hasattr(agent, "train_step"):
                metrics = agent.train_step(states, actions, rewards, next_states, dones)
            elif hasattr(agent, "update"):
                metrics = agent.update(states, actions, rewards, next_states, dones)
            elif hasattr(agent, "learn"):
                metrics = agent.learn(states, actions, rewards, next_states, dones)
            else:
                return {"error": "Agent has no training method"}

            return {
                "success": True,
                "agent_id": agent_id,
                "metrics": metrics if isinstance(metrics, dict) else {"loss": float(metrics) if metrics else 0}
            }

        except Exception as e:
            logger.exception(f"Training failed: {e}")
            return {"error": str(e)}

    def save_agent(self, agent_id: str, path: str = None) -> Dict[str, Any]:
        """Save agent to disk"""

        instance = self.agents.get(agent_id)
        if not instance:
            return {"error": f"Agent not found: {agent_id}"}

        try:
            save_path = Path(path) if path else self.models_dir / f"{agent_id}.pkl"
            save_path.parent.mkdir(parents=True, exist_ok=True)

            # Save agent state
            save_data = {
                "id": instance.id,
                "algorithm": instance.algorithm,
                "config": instance.config,
                "created_at": instance.created_at,
                "state_dim": instance.state_dim,
                "action_dim": instance.action_dim,
            }

            # Try to get agent state dict
            if hasattr(instance.agent, "state_dict"):
                save_data["state_dict"] = instance.agent.state_dict()
            elif hasattr(instance.agent, "get_weights"):
                save_data["weights"] = instance.agent.get_weights()
            else:
                save_data["agent"] = instance.agent

            with open(save_path, "wb") as f:
                pickle.dump(save_data, f)

            return {
                "success": True,
                "agent_id": agent_id,
                "path": str(save_path)
            }

        except Exception as e:
            logger.exception(f"Save failed: {e}")
            return {"error": str(e)}

    def load_agent(self, path: str) -> Dict[str, Any]:
        """Load agent from disk"""

        try:
            load_path = Path(path)
            if not load_path.exists():
                return {"error": f"File not found: {path}"}

            with open(load_path, "rb") as f:
                save_data = pickle.load(f)

            algorithm = save_data["algorithm"]
            algo_info = RL_ALGORITHMS[algorithm]

            # Recreate agent
            module = __import__(algo_info["module"], fromlist=[algo_info["class"]])
            agent_class = getattr(module, algo_info["class"])

            state_dim = save_data["state_dim"]
            action_dim = save_data["action_dim"]

            # Create agent instance
            if algorithm in ["a3c", "mappo"]:
                agent = agent_class(state_dim=state_dim, action_dim=action_dim)
            elif algorithm == "jpmorgan_marl":
                agent = agent_class(state_dim=state_dim, n_agents=save_data["config"].get("n_agents", 5))
            elif algorithm in ["dreamer", "safe_dreamer"]:
                agent = agent_class(obs_dim=state_dim, action_dim=action_dim)
            else:
                agent = agent_class(state_dim=state_dim, action_dim=action_dim)

            # Load weights
            if "state_dict" in save_data and hasattr(agent, "load_state_dict"):
                agent.load_state_dict(save_data["state_dict"])
            elif "weights" in save_data and hasattr(agent, "set_weights"):
                agent.set_weights(save_data["weights"])
            elif "agent" in save_data:
                agent = save_data["agent"]

            # Register agent
            agent_id = save_data["id"]
            instance = AgentInstance(
                id=agent_id,
                algorithm=algorithm,
                config=save_data["config"],
                created_at=save_data["created_at"],
                state_dim=state_dim,
                action_dim=action_dim,
                agent=agent
            )

            self.agents[agent_id] = instance

            return {
                "success": True,
                "agent_id": agent_id,
                "algorithm": algorithm,
                "path": str(load_path)
            }

        except Exception as e:
            logger.exception(f"Load failed: {e}")
            return {"error": str(e)}

    def delete_agent(self, agent_id: str) -> Dict[str, Any]:
        """Delete agent instance"""

        if agent_id not in self.agents:
            return {"error": f"Agent not found: {agent_id}"}

        del self.agents[agent_id]
        return {"success": True, "agent_id": agent_id}


# ═══════════════════════════════════════════════════════════════════════════════
# MCP Server (JSON-RPC over stdio)
# ═══════════════════════════════════════════════════════════════════════════════

class MCPRLServer:
    """MCP Server for RL operations"""

    def __init__(self):
        self.manager = RLAgentManager()
        self.tools = {
            "rl_list_algorithms": self._list_algorithms,
            "rl_create_agent": self._create_agent,
            "rl_list_agents": self._list_agents,
            "rl_predict": self._predict,
            "rl_train_step": self._train_step,
            "rl_save_agent": self._save_agent,
            "rl_load_agent": self._load_agent,
            "rl_delete_agent": self._delete_agent,
        }

    def get_manifest(self) -> Dict[str, Any]:
        """Return MCP server manifest"""
        return {
            "name": "forex-rl",
            "version": "1.0.0",
            "description": "US Quant Gold Standard RL for Forex Trading",
            "tools": [
                {
                    "name": "rl_list_algorithms",
                    "description": "List all available RL algorithms (A3C, MAPPO, CQL, Dreamer, MAML, etc.)",
                    "inputSchema": {"type": "object", "properties": {}}
                },
                {
                    "name": "rl_create_agent",
                    "description": "Create a new RL agent instance",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "algorithm": {"type": "string", "description": "Algorithm ID (a3c, mappo, cql, dreamer, maml, etc.)"},
                            "state_dim": {"type": "integer", "default": 575, "description": "State dimension (575 for forex features)"},
                            "action_dim": {"type": "integer", "default": 3, "description": "Action dimension (3 for buy/hold/sell)"},
                            "config": {"type": "object", "description": "Algorithm-specific configuration"}
                        },
                        "required": ["algorithm"]
                    }
                },
                {
                    "name": "rl_list_agents",
                    "description": "List all active RL agent instances",
                    "inputSchema": {"type": "object", "properties": {}}
                },
                {
                    "name": "rl_predict",
                    "description": "Get trading action from an RL agent",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "agent_id": {"type": "string", "description": "Agent instance ID"},
                            "state": {"type": "array", "items": {"type": "number"}, "description": "Current state (575 features)"},
                            "deterministic": {"type": "boolean", "default": True, "description": "Use deterministic policy"}
                        },
                        "required": ["agent_id", "state"]
                    }
                },
                {
                    "name": "rl_train_step",
                    "description": "Execute one training step on an RL agent",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "agent_id": {"type": "string", "description": "Agent instance ID"},
                            "batch": {
                                "type": "object",
                                "properties": {
                                    "states": {"type": "array"},
                                    "actions": {"type": "array"},
                                    "rewards": {"type": "array"},
                                    "next_states": {"type": "array"},
                                    "dones": {"type": "array"}
                                },
                                "description": "Training batch"
                            }
                        },
                        "required": ["agent_id", "batch"]
                    }
                },
                {
                    "name": "rl_save_agent",
                    "description": "Save an RL agent to disk",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "agent_id": {"type": "string", "description": "Agent instance ID"},
                            "path": {"type": "string", "description": "Optional save path"}
                        },
                        "required": ["agent_id"]
                    }
                },
                {
                    "name": "rl_load_agent",
                    "description": "Load an RL agent from disk",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "path": {"type": "string", "description": "Path to saved agent"}
                        },
                        "required": ["path"]
                    }
                },
                {
                    "name": "rl_delete_agent",
                    "description": "Delete an RL agent instance",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "agent_id": {"type": "string", "description": "Agent instance ID"}
                        },
                        "required": ["agent_id"]
                    }
                }
            ]
        }

    def _list_algorithms(self, params: Dict) -> Dict:
        return {"algorithms": self.manager.list_algorithms()}

    def _create_agent(self, params: Dict) -> Dict:
        return self.manager.create_agent(
            algorithm=params["algorithm"],
            state_dim=params.get("state_dim", 575),
            action_dim=params.get("action_dim", 3),
            config=params.get("config")
        )

    def _list_agents(self, params: Dict) -> Dict:
        return {"agents": self.manager.list_agents()}

    def _predict(self, params: Dict) -> Dict:
        return self.manager.predict(
            agent_id=params["agent_id"],
            state=params["state"],
            deterministic=params.get("deterministic", True)
        )

    def _train_step(self, params: Dict) -> Dict:
        return self.manager.train_step(
            agent_id=params["agent_id"],
            batch=params["batch"]
        )

    def _save_agent(self, params: Dict) -> Dict:
        return self.manager.save_agent(
            agent_id=params["agent_id"],
            path=params.get("path")
        )

    def _load_agent(self, params: Dict) -> Dict:
        return self.manager.load_agent(path=params["path"])

    def _delete_agent(self, params: Dict) -> Dict:
        return self.manager.delete_agent(agent_id=params["agent_id"])

    async def handle_request(self, request: Dict) -> Dict:
        """Handle JSON-RPC request"""

        method = request.get("method", "")
        params = request.get("params", {})
        req_id = request.get("id")

        if method == "initialize":
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {"tools": {}},
                    "serverInfo": self.get_manifest()
                }
            }

        elif method == "tools/list":
            manifest = self.get_manifest()
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {"tools": manifest["tools"]}
            }

        elif method == "tools/call":
            tool_name = params.get("name", "")
            tool_args = params.get("arguments", {})

            if tool_name in self.tools:
                try:
                    result = self.tools[tool_name](tool_args)
                    return {
                        "jsonrpc": "2.0",
                        "id": req_id,
                        "result": {"content": [{"type": "text", "text": json.dumps(result, indent=2)}]}
                    }
                except Exception as e:
                    return {
                        "jsonrpc": "2.0",
                        "id": req_id,
                        "error": {"code": -32603, "message": str(e)}
                    }
            else:
                return {
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "error": {"code": -32601, "message": f"Unknown tool: {tool_name}"}
                }

        else:
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "error": {"code": -32601, "message": f"Unknown method: {method}"}
            }

    async def run_stdio(self):
        """Run MCP server over stdio"""

        logger.info("Starting MCP RL Server (stdio mode)")

        reader = asyncio.StreamReader()
        protocol = asyncio.StreamReaderProtocol(reader)
        await asyncio.get_event_loop().connect_read_pipe(lambda: protocol, sys.stdin)

        writer_transport, writer_protocol = await asyncio.get_event_loop().connect_write_pipe(
            asyncio.streams.FlowControlMixin, sys.stdout
        )
        writer = asyncio.StreamWriter(writer_transport, writer_protocol, reader, asyncio.get_event_loop())

        while True:
            try:
                line = await reader.readline()
                if not line:
                    break

                request = json.loads(line.decode())
                response = await self.handle_request(request)

                writer.write((json.dumps(response) + "\n").encode())
                await writer.drain()

            except Exception as e:
                logger.exception(f"Error handling request: {e}")


# ═══════════════════════════════════════════════════════════════════════════════
# HTTP Server (for REST access)
# ═══════════════════════════════════════════════════════════════════════════════

def create_http_app():
    """Create Flask HTTP app for RL operations"""

    from flask import Flask, jsonify, request
    from flask_cors import CORS

    app = Flask(__name__)
    CORS(app)

    manager = RLAgentManager()

    @app.route('/health', methods=['GET'])
    def health():
        return jsonify({
            "status": "healthy",
            "service": "mcp-rl-server",
            "timestamp": datetime.utcnow().isoformat(),
            "algorithms": len(RL_ALGORITHMS)
        })

    @app.route('/api/rl/algorithms', methods=['GET'])
    def list_algorithms():
        return jsonify({"algorithms": manager.list_algorithms()})

    @app.route('/api/rl/agents', methods=['GET'])
    def list_agents():
        return jsonify({"agents": manager.list_agents()})

    @app.route('/api/rl/agents', methods=['POST'])
    def create_agent():
        data = request.json
        result = manager.create_agent(
            algorithm=data["algorithm"],
            state_dim=data.get("state_dim", 575),
            action_dim=data.get("action_dim", 3),
            config=data.get("config")
        )
        return jsonify(result)

    @app.route('/api/rl/agents/<agent_id>/predict', methods=['POST'])
    def predict(agent_id):
        data = request.json
        result = manager.predict(
            agent_id=agent_id,
            state=data["state"],
            deterministic=data.get("deterministic", True)
        )
        return jsonify(result)

    @app.route('/api/rl/agents/<agent_id>/train', methods=['POST'])
    def train(agent_id):
        data = request.json
        result = manager.train_step(agent_id=agent_id, batch=data["batch"])
        return jsonify(result)

    @app.route('/api/rl/agents/<agent_id>/save', methods=['POST'])
    def save(agent_id):
        data = request.json or {}
        result = manager.save_agent(agent_id=agent_id, path=data.get("path"))
        return jsonify(result)

    @app.route('/api/rl/agents/<agent_id>', methods=['DELETE'])
    def delete(agent_id):
        result = manager.delete_agent(agent_id=agent_id)
        return jsonify(result)

    @app.route('/api/rl/load', methods=['POST'])
    def load():
        data = request.json
        result = manager.load_agent(path=data["path"])
        return jsonify(result)

    return app


def main():
    import argparse

    parser = argparse.ArgumentParser(description='MCP RL Server')
    parser.add_argument('--mode', choices=['stdio', 'http'], default='http', help='Server mode')
    parser.add_argument('--port', type=int, default=8081, help='HTTP port')
    parser.add_argument('--host', default='0.0.0.0', help='HTTP host')

    args = parser.parse_args()

    if args.mode == 'stdio':
        server = MCPRLServer()
        asyncio.run(server.run_stdio())
    else:
        app = create_http_app()

        logger.info("=" * 60)
        logger.info("MCP RL SERVER - US Quant Gold Standard")
        logger.info(f"Listening on {args.host}:{args.port}")
        logger.info(f"Algorithms: {len(RL_ALGORITHMS)}")
        logger.info("=" * 60)
        logger.info("")
        logger.info("Available endpoints:")
        logger.info("  GET  /health")
        logger.info("  GET  /api/rl/algorithms")
        logger.info("  GET  /api/rl/agents")
        logger.info("  POST /api/rl/agents")
        logger.info("  POST /api/rl/agents/<id>/predict")
        logger.info("  POST /api/rl/agents/<id>/train")
        logger.info("  POST /api/rl/agents/<id>/save")
        logger.info("  DELETE /api/rl/agents/<id>")
        logger.info("  POST /api/rl/load")
        logger.info("")

        app.run(host=args.host, port=args.port, debug=False)


if __name__ == "__main__":
    main()
