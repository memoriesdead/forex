# Core Module
# Backward-compatible exports for forex trading system
#
# New structure (2026-01):
#   core/ml/        - ML training (gpu_config, ensemble, retrainer)
#   core/rl/        - Reinforcement Learning (PPO, SAC, TD3, CVaR) - NEW 2026-01-17
#   core/data/      - Data loading (loader, buffer)
#   core/features/  - Feature engineering (engine, alpha101, renaissance)
#   core/execution/ - Order execution (order_book, queue, fill, backtest)
#   core/models/    - Model management (cache, loader)
#   core/risk/      - Risk management (limits, per_symbol, portfolio)
#   core/symbol/    - Symbol management (registry)
#   core/trading/   - Trading logic (signal, position, executor, bot)
#
# Legacy imports below allow old code to work during transition.

# ML Module exports
from core.ml.gpu_config import (
    get_xgb_gpu_params,
    get_lgb_gpu_params,
    get_catboost_gpu_params,
    get_tabnet_gpu_params,
    get_cnn_gpu_params,
    get_mlp_gpu_params,
    get_transformer_gpu_params,
    get_stacking_training_config,
    configure_gpu,
)
from core.ml.ensemble import HotSwappableEnsemble, ModelVersion, get_ensemble
from core.ml.retrainer import HybridRetrainer, get_hybrid_retrainer
from core.ml.live_retrainer import LiveRetrainer, get_retrainer

# Feature Selection exports (World-Class Pipeline - 2026-01-17)
from core.ml.feature_selector import (
    FeatureSelector,
    AdaptiveFeatureSelector,
    FeatureSelectionConfig,
    create_feature_selector,
)

# Stacking Ensemble exports (6-model stacking - 2026-01-17)
from core.ml.stacking_ensemble import (
    StackingEnsemble,
    StackingConfig,
    create_stacking_ensemble,
)

# Neural Models exports (TabNet, CNN, MLP - 2026-01-17)
# Optional - requires torch
try:
    from core.ml.neural_models import (
        CNN1DClassifier,
        MLPClassifier,
        EnsembleNeuralClassifier,
        create_neural_classifier,
    )
except ImportError:
    CNN1DClassifier = None
    MLPClassifier = None
    EnsembleNeuralClassifier = None
    create_neural_classifier = None

# Data Module exports
from core.data.loader import UnifiedDataLoader, TrueFXLiveLoader, TrueFXHistoricalLoader
from core.data.buffer import LiveTickBuffer, TickRecord, get_tick_buffer

# Features Module exports
from core.features.engine import HFTFeatureEngine, FeatureConfig, FastTechnicalFeatures, create_hft_feature_engine
from core.features.fast_engine import FastFeatureEngine, create_targets
from core.features.alpha101 import Alpha101Complete
from core.features.renaissance import RenaissanceSignalGenerator
from core.features.cross_asset import CrossAssetSignalGenerator, CrossAssetSignal

# Experimental Features exports (1,500+ features - 2026-01-17)
from core.features.experimental_engine import (
    ExperimentalFeatureEngine,
    ExperimentalConfig,
    create_experimental_engine,
)

# Execution Module exports
from core.execution.order_book import OrderBookL3, PriceLevel, Order, OrderBookSignals
from core.execution.queue_position import (
    RiskAverseQueueModel,
    ProbabilisticQueueModel,
    L3FIFOQueueModel,
    QueuePositionTracker,
)
from core.execution.fill_probability import PoissonFillModel, MarketImpactModel, FillProbabilityEngine
from core.execution.latency import LatencySimulator, LatencyConfig
from core.execution.backtest import TickBacktestEngine, BacktestConfig

# RL Module exports (Gold Standard RL - 2026-01-17)
# Academic Citations:
#   - PPO: Schulman et al. (2017), arXiv:1707.06347
#   - SAC: Haarnoja et al. (2018), ICML
#   - TD3: Fujimoto et al. (2018), ICML
#   - CVaR-RL: Tamar et al. (2015), ICML
#   - FinRL: Liu et al. (2020), NeurIPS
#   - ElegantRL: Liu et al. (2021), ICAIF
#
# Chinese Quant RL Innovations (2026-01-17):
#   - MARL Order Execution: Wei et al. (2023), Microsoft Research Asia
#   - QMIX: Rashid et al. (2018), ICML
#   - GraphMIX: 控制与决策期刊 (2022), Chinese GNN extension
#   - GAT: Veličković et al. (2018), ICLR
#   - Option-Critic: Bacon et al. (2017), AAAI
#   - FeUdal Networks: Vezhnevets et al. (2017), ICML
#   - TradeMaster: Sun et al. (2023), NeurIPS Workshop (南洋理工 NTU)
# RL imports are optional - require gymnasium and torch
try:
    from core.rl.environments import (
        ForexTradingEnv,
        ForexTradingEnvConfig,
        MultiAssetForexEnv,
        create_forex_env,
    )
    from core.rl.agents import (
        PPOAgent,
        SACAgent,
        A2CAgent,
        TD3Agent,
        DQNAgent,
        DDPGAgent,
        BaseRLAgent,
        AgentConfig,
        create_agent,
    )
    from core.rl.risk_sensitive import (
        CVaRPPO,
        DrawdownConstrainedPPO,
        TailSafePPO,
        RiskSensitiveConfig,
        create_risk_sensitive_agent,
    )
    from core.rl.portfolio import (
        RLPortfolioOptimizer,
        PortfolioEnv,
        PortfolioConfig,
        optimize_portfolio_rl,
    )
    from core.rl.ensemble import (
        RLEnsemble,
        TournamentEnsemble,
        create_rl_ensemble,
    )
    from core.rl.trainer import (
        RLTrainer,
        TrainingConfig,
        train_rl_agent,
        evaluate_agent,
    )
    # Chinese Quant RL exports
    from core.rl.chinese_marl import (
        MARLConfig,
        OrderExecutionEnv,
        QMIXMixer,
        MARLAgent,
        MARLOrderExecution,
        create_marl_order_execution,
    )
    from core.rl.graph_rl import (
        GraphRLConfig,
        GraphAttentionLayer,
        GraphConvLayer,
        HierarchicalGAT,
        GraphMIX,
        GraphRLAgent,
        create_graph_rl_agent,
    )
    from core.rl.hierarchical_rl import (
        HierarchicalConfig,
        ManagerNetwork,
        WorkerNetwork,
        OptionCritic,
        HierarchicalTradingAgent,
        OptionCriticTrader,
        create_hierarchical_agent,
        create_option_critic_trader,
    )
    from core.rl.trademaster import (
        TradeMasterConfig,
        DeepScalper,
        DeepTrader,
        EIIE,
        SARL,
        InvestorImitator,
        TradeMasterAgent,
        create_trademaster_agent,
    )
    _HAS_RL = True
except ImportError:
    # RL not available (missing gymnasium/torch)
    _HAS_RL = False
    ForexTradingEnv = None
    ForexTradingEnvConfig = None
    MultiAssetForexEnv = None
    create_forex_env = None
    PPOAgent = None
    SACAgent = None
    A2CAgent = None
    TD3Agent = None
    DQNAgent = None
    DDPGAgent = None
    BaseRLAgent = None
    AgentConfig = None
    create_agent = None
    CVaRPPO = None
    DrawdownConstrainedPPO = None
    TailSafePPO = None
    RiskSensitiveConfig = None
    create_risk_sensitive_agent = None
    RLPortfolioOptimizer = None
    PortfolioEnv = None
    PortfolioConfig = None
    optimize_portfolio_rl = None
    RLEnsemble = None
    TournamentEnsemble = None
    create_rl_ensemble = None
    RLTrainer = None
    TrainingConfig = None
    train_rl_agent = None
    evaluate_agent = None
    MARLConfig = None
    OrderExecutionEnv = None
    QMIXMixer = None
    MARLAgent = None
    MARLOrderExecution = None
    create_marl_order_execution = None
    GraphRLConfig = None
    GraphAttentionLayer = None
    GraphConvLayer = None
    HierarchicalGAT = None
    GraphMIX = None
    GraphRLAgent = None
    create_graph_rl_agent = None
    HierarchicalConfig = None
    ManagerNetwork = None
    WorkerNetwork = None
    OptionCritic = None
    HierarchicalTradingAgent = None
    OptionCriticTrader = None
    create_hierarchical_agent = None
    create_option_critic_trader = None
    TradeMasterConfig = None
    DeepScalper = None
    DeepTrader = None
    EIIE = None
    SARL = None
    InvestorImitator = None
    TradeMasterAgent = None
    create_trademaster_agent = None

__all__ = [
    # ML (GPU Config)
    'get_xgb_gpu_params',
    'get_lgb_gpu_params',
    'get_catboost_gpu_params',
    'get_tabnet_gpu_params',
    'get_cnn_gpu_params',
    'get_mlp_gpu_params',
    'get_transformer_gpu_params',
    'get_stacking_training_config',
    'configure_gpu',
    'HotSwappableEnsemble',
    'ModelVersion',
    'get_ensemble',
    'HybridRetrainer',
    'get_hybrid_retrainer',
    'LiveRetrainer',
    'get_retrainer',
    # ML (Feature Selection - World-Class 2026-01-17)
    'FeatureSelector',
    'AdaptiveFeatureSelector',
    'FeatureSelectionConfig',
    'create_feature_selector',
    # ML (Stacking Ensemble - 6-model 2026-01-17)
    'StackingEnsemble',
    'StackingConfig',
    'create_stacking_ensemble',
    # ML (Neural Models - TabNet, CNN, MLP 2026-01-17)
    'CNN1DClassifier',
    'MLPClassifier',
    'EnsembleNeuralClassifier',
    'create_neural_classifier',
    # Data
    'UnifiedDataLoader',
    'TrueFXLiveLoader',
    'TrueFXHistoricalLoader',
    'LiveTickBuffer',
    'TickRecord',
    'get_tick_buffer',
    # Features (Core)
    'HFTFeatureEngine',
    'FeatureConfig',
    'FastTechnicalFeatures',
    'create_hft_feature_engine',
    'FastFeatureEngine',
    'create_targets',
    'Alpha101Complete',
    'RenaissanceSignalGenerator',
    'CrossAssetSignalGenerator',
    'CrossAssetSignal',
    # Features (Experimental - 1,500+ features 2026-01-17)
    'ExperimentalFeatureEngine',
    'ExperimentalConfig',
    'create_experimental_engine',
    # Execution
    'OrderBookL3',
    'PriceLevel',
    'Order',
    'OrderBookSignals',
    'RiskAverseQueueModel',
    'ProbabilisticQueueModel',
    'L3FIFOQueueModel',
    'QueuePositionTracker',
    'PoissonFillModel',
    'MarketImpactModel',
    'FillProbabilityEngine',
    'LatencySimulator',
    'LatencyConfig',
    'TickBacktestEngine',
    'BacktestConfig',
    # RL (Gold Standard - 2026-01-17)
    # Environments
    'ForexTradingEnv',
    'ForexTradingEnvConfig',
    'MultiAssetForexEnv',
    'create_forex_env',
    # Agents (FinRL/SB3 style)
    'PPOAgent',
    'SACAgent',
    'A2CAgent',
    'TD3Agent',
    'DQNAgent',
    'DDPGAgent',
    'BaseRLAgent',
    'AgentConfig',
    'create_agent',
    # Risk-Sensitive (CVaR, Drawdown)
    'CVaRPPO',
    'DrawdownConstrainedPPO',
    'TailSafePPO',
    'RiskSensitiveConfig',
    'create_risk_sensitive_agent',
    # Portfolio Optimization
    'RLPortfolioOptimizer',
    'PortfolioEnv',
    'PortfolioConfig',
    'optimize_portfolio_rl',
    # Ensemble (ElegantRL style)
    'RLEnsemble',
    'TournamentEnsemble',
    'create_rl_ensemble',
    # Training
    'RLTrainer',
    'TrainingConfig',
    'train_rl_agent',
    'evaluate_agent',
    # ═══════════════════════════════════════════════════════════════════
    # Chinese Quant RL Innovations (2026-01-17)
    # ═══════════════════════════════════════════════════════════════════
    # MARL Order Execution (Microsoft Qlib / 微软亚研)
    'MARLConfig',
    'OrderExecutionEnv',
    'QMIXMixer',
    'MARLAgent',
    'MARLOrderExecution',
    'create_marl_order_execution',
    # Graph RL (GraphMIX / 控制与决策)
    'GraphRLConfig',
    'GraphAttentionLayer',
    'GraphConvLayer',
    'HierarchicalGAT',
    'GraphMIX',
    'GraphRLAgent',
    'create_graph_rl_agent',
    # Hierarchical RL (分层强化学习 / 智能系统学报)
    'HierarchicalConfig',
    'ManagerNetwork',
    'WorkerNetwork',
    'OptionCritic',
    'HierarchicalTradingAgent',
    'OptionCriticTrader',
    'create_hierarchical_agent',
    'create_option_critic_trader',
    # TradeMaster (南洋理工 NTU)
    'TradeMasterConfig',
    'DeepScalper',
    'DeepTrader',
    'EIIE',
    'SARL',
    'InvestorImitator',
    'TradeMasterAgent',
    'create_trademaster_agent',
]
