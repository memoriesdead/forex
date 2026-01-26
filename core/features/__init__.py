# Features Module
# Feature engineering for forex trading
#
# ══════════════════════════════════════════════════════════════
# CHINESE QUANT ALPHA LIBRARIES:
# ══════════════════════════════════════════════════════════════
# - Alpha101: WorldQuant (62 factors)
# - Alpha158: Microsoft Qlib (179 factors)
# - Alpha191: 国泰君安 Guotaijunan (191 factors) - in _experimental
# - Alpha360: Microsoft Qlib (360 factors)
# - Barra CNE6: MSCI Risk Factors (46 factors)
# - Renaissance: Proprietary signals (51)
#
# ══════════════════════════════════════════════════════════════
# USA ACADEMIC FACTORS (Peer-Reviewed):
# ══════════════════════════════════════════════════════════════
# - Carry: Lustig, Roussanov, Verdelhan (2011) - RFS
# - Momentum: Menkhoff et al (2012) - BIS Working Papers
# - Value: Asness, Moskowitz, Pedersen (2013) - JFE
# - Macro: Bybee, Gomes, Valente (2023) - SSRN
# - US Academic Factors: 50 factors
#
# ══════════════════════════════════════════════════════════════
# GITHUB/GITEE GOLD STANDARD:
# ══════════════════════════════════════════════════════════════
# - MLFinLab: Lopez de Prado (Triple Barrier, Meta-Label) - 17 factors
# - Time-Series-Library: Tsinghua ML (iTransformer, TimeMixer) - 45 factors
# - GitHub Gold Standard: Microstructure formulas - 12 factors
#   * Order Flow Imbalance (Cont et al. 2014)
#   * VPIN (Easley, Lopez de Prado, O'Hara 2012)
#   * Volatility Estimators (Garman-Klass, Parkinson, Rogers-Satchell)
#   * Risk Metrics (Sharpe, Sortino, Calmar)
#   * Kelly Criterion position sizing
#   * Hawkes Process intensity
#
# ══════════════════════════════════════════════════════════════
# GITEE CHINESE QUANTITATIVE FACTORS (2026-01-17):
# ══════════════════════════════════════════════════════════════
# Sources: QUANTAXIS, QuantsPlaybook, Alpha-101-GTJA-191
# - Alpha191 Forex Subset: 国泰君安 (16 factors)
# - Smart Money Factor 2.0: 开源证券 (4 factors)
# - VPIN Enhanced: 招商证券/Easley (6 factors)
# - Integrated OFI: 兴业证券/Cont (7 factors)
# - Volatility Estimators: Yang-Zhang, Garman-Klass, etc. (7 factors)
# - Market Impact: Kyle Lambda, Amihud Illiquidity (5 factors)
# - Microprice: Stoikov (6 factors)
# Total: ~60-80 additional factors with full academic citations
#
# ══════════════════════════════════════════════════════════════
# GLOBAL EXPANSION (2026-01-17):
# ══════════════════════════════════════════════════════════════
# - India Quant: IIQF, NSE Academy, RBI research (25 factors)
# - Japan Quant: JPX, J-Quants, TSE Arrowhead (20 factors)
# - Europe Quant: ETH, Imperial, Heston, SABR (15 factors)
# - Emerging Markets: Brazil, Russia, SE Asia (20 factors)
# - Universal Math: OU, Kelly, Cointegration (30 factors)
#
# ══════════════════════════════════════════════════════════════
# REINFORCEMENT LEARNING (2026-01-17):
# ══════════════════════════════════════════════════════════════
# Pure mathematical RL without neural networks:
# - Q-Learning (Watkins 1989): 6 features
# - SARSA (Rummery & Niranjan 1994): 5 features
# - TD(λ) (Sutton 1988): 5 features
# - Dyna-Q (Sutton 1991): 5 features
# - Thompson Sampling (Thompson 1933): 5 features
# - Almgren-Chriss Execution (2001): 5 features
# - Kelly-RL Integration: 4 features
# Total: 35 features
#
# ══════════════════════════════════════════════════════════════
# DEEP REINFORCEMENT LEARNING (2026-01-17):
# ══════════════════════════════════════════════════════════════
# Gold standard deep RL algorithms from top conferences:
# - TD3: Twin Delayed DDPG (Fujimoto ICML 2018)
# - SAC: Soft Actor-Critic (Haarnoja ICML 2018)
# - EarnHFT: Hierarchical RL for HFT (Qin AAAI 2024) +30% vs baselines
# - MacroHFT: Memory-Augmented Context-Aware RL (Zong KDD 2024)
# - DeepScalper: Risk-Aware RL (Sun CIKM 2022)
# Total: 15 features + full trading agents
#
# ══════════════════════════════════════════════════════════════
# TOTAL: 1300+ features
# ══════════════════════════════════════════════════════════════

from .engine import HFTFeatureEngine, FeatureConfig, FastTechnicalFeatures, create_hft_feature_engine
from .fast_engine import FastFeatureEngine, create_targets
from .alpha101 import Alpha101Complete
from .alpha158 import Alpha158, generate_alpha158
from .alpha360 import Alpha360, Alpha360Compact, generate_alpha360, generate_alpha360_compact
from .barra_cne6 import BarraCNE6Forex, generate_barra_cne6
from .us_academic_factors import USAcademicFactors, generate_us_academic_factors
from .mlfinlab_features import MLFinLabFeatures, generate_mlfinlab_features, TripleBarrierLabeling, MetaLabeling
from .timeseries_features import TimeSeriesLibraryFeatures, generate_timeseries_features
from .renaissance import RenaissanceSignalGenerator
from .cross_asset import CrossAssetSignalGenerator, CrossAssetSignal
from .mega_generator import (
    MegaFeatureGenerator,
    generate_mega_features,
    generate_chinese_quant_features,
    generate_academic_features,
    generate_renaissance_features
)
from .github_gold_standard import (
    OrderFlowImbalance,
    OFIResult,
    VPIN,
    VPINResult,
    VolatilityEstimators,
    RiskMetrics,
    KellyCriterion,
    HawkesProcess,
    GoldStandardFeatures,
    generate_gold_standard_features,
    calculate_kelly_position
)
from .gitee_chinese_factors import (
    Alpha191ForexFactors,
    SmartMoneyFactor2,
    VPINEnhanced,
    IntegratedOFI,
    VolatilityEstimators as GiteeVolatilityEstimators,
    MarketImpactFactors,
    MicropriceCalculator,
    GiteeChineseFactorGenerator,
    generate_gitee_chinese_factors,
    get_citation_info
)
# Global Expansion (2026-01-17)
from .india_quant import IndiaQuantFeatures, generate_india_features
from .japan_quant import JapanQuantFeatures, generate_japan_features
from .europe_quant import EuropeQuantFeatures, generate_europe_features
from .emerging_quant import EmergingMarketsFeatures, generate_emerging_features
from .universal_math import UniversalMathFeatures, generate_universal_math_features

# ══════════════════════════════════════════════════════════════
# REINFORCEMENT LEARNING ALGORITHMS (2026-01-17):
# ══════════════════════════════════════════════════════════════
# Pure mathematical RL without neural networks:
# - Q-Learning: Off-policy value-based learning (Watkins 1989)
# - SARSA: On-policy safe learning (Rummery & Niranjan 1994)
# - TD(λ): Temporal difference with eligibility traces (Sutton 1988)
# - Dyna-Q: Model-based RL with planning (Sutton 1991)
# - Thompson Sampling: Bayesian bandit algorithm (Thompson 1933)
# - Almgren-Chriss: Optimal execution framework (2001)
# - Kelly-RL: Position sizing integration
from .rl_algorithms import (
    RLState,
    QLearningTrader,
    SARSATrader,
    TDLambdaTrader,
    DynaQTrader,
    ThompsonSamplingBandit,
    AlmgrenChrisExecutor,
    RLFeatureGenerator,
    generate_rl_features,
    get_rl_citations,
    print_citations as print_rl_citations
)

# ══════════════════════════════════════════════════════════════
# DEEP REINFORCEMENT LEARNING ALGORITHMS (2026-01-17):
# ══════════════════════════════════════════════════════════════
# Gold standard deep RL from ICML, AAAI, KDD:
# - TD3: Twin Delayed DDPG (Fujimoto et al. ICML 2018)
# - SAC: Soft Actor-Critic (Haarnoja et al. ICML 2018)
# - EarnHFT: Hierarchical RL for HFT (Qin et al. AAAI 2024)
# - MacroHFT: Memory-Augmented RL (Zong et al. KDD 2024)
# - DeepScalper: Risk-Aware RL (Sun et al. CIKM 2022)
from .deep_rl_trading import (
    # Configs
    TD3Config,
    SACConfig,
    EarnHFTConfig,
    MacroHFTConfig,
    DeepScalperConfig,
    # Replay Buffers
    ReplayBuffer,
    PrioritizedReplayBuffer,
    # TD3 (Fujimoto ICML 2018)
    TD3Actor,
    TD3Critic,
    TD3Trader,
    # SAC (Haarnoja ICML 2018)
    SACGaussianActor,
    SACCritic,
    SACTrader,
    # EarnHFT (Qin AAAI 2024) - +30% vs baselines
    EarnHFTSubAgent,
    EarnHFTRouter,
    EarnHFTTrader,
    # MacroHFT (Zong KDD 2024)
    MemoryAugmentedNetwork,
    ConditionalAdapter,
    MacroHFTTrader,
    # DeepScalper (Sun CIKM 2022)
    RiskAwareNetwork,
    DeepScalperTrader,
    # Features
    DeepRLFeatureGenerator,
    generate_deep_rl_features,
)

# ══════════════════════════════════════════════════════════════
# RL RESEARCH (USA QUANT PEER-REVIEWED) - 2026-01-17:
# ══════════════════════════════════════════════════════════════
# 22 Academic Papers from IEEE, ICML, AAAI, NeurIPS, JMLR:
# - Reward Shaping: Moody & Saffell (2001) IEEE Trans Neural Networks
# - Risk-Sensitive RL: Tamar (2015), Chow (2017), Greenberg (2024)
# - CVaR/EVaR: Rockafellar & Uryasev (2000), Lim & Malik (2024)
# - Optimal Execution: Almgren & Chriss (2001), Nevmyvaka (2006)
# - Market Making: Avellaneda & Stoikov (2008), Spooner (2018)
# - Distributional RL: Dabney (2018) AAAI, Zhou (2020) NeurIPS
# - Meta-RL: Finn (2017) ICML - MAML [10,000+ citations]
from .rl_research import (
    # Reward Shaping (Moody & Saffell 2001)
    DifferentialSharpeRatio,
    DifferentialDownsideDeviation,
    MaxDrawdownPenalty,
    RewardConfig,
    CombinedRewardFunction,
    # Risk-Sensitive RL (Tamar 2015, Chow 2017)
    ValueAtRisk,
    ConditionalValueAtRisk,
    EntropicValueAtRisk,
    CVaRConstrainedKelly,
    RiskMetricsResult,
    compute_risk_metrics,
    # Optimal Execution (Almgren-Chriss 2001)
    ExecutionConfig,
    AlmgrenChriss as AlmgrenChrissRL,
    RLOptimalExecution,
    # Market Making (Avellaneda-Stoikov 2008)
    MarketMakingConfig as RLMarketMakingConfig,
    AvellanedaStoikovRL,
    # Distributional RL (Dabney 2018)
    QuantileDistribution,
    QuantileHuberLoss,
    # Meta-RL (Finn 2017)
    MAMLTrader,
    RegimeAwareRL,
    # Position Sizing
    PositionSizerConfig,
    RLPositionSizer,
    # Features
    RLFeatureGenerator as RLResearchFeatureGenerator,
    # Factory Functions
    create_reward_function,
    create_execution_model,
    create_market_maker,
    generate_rl_features as generate_rl_research_features,
)

# ══════════════════════════════════════════════════════════════
# ACADEMIC RESEARCH MODULES (2026-01-17):
# ══════════════════════════════════════════════════════════════
# Peer-reviewed implementations with full citations:
# - Microstructure: OFI, VPIN, Kyle Lambda (Cont 2014, Easley 2012, Kyle 1985)
# - Volatility: HAR-RV, Range-based estimators (Corsi 2009, Parkinson 1980)
# - Regime Detection: HMM (Hamilton 1989)
# - Market Making: Avellaneda-Stoikov (2008)
# - Deep Learning: iTransformer, TFT, N-BEATS (ICLR/NeurIPS 2020-2024)
from .academic_microstructure import (
    OrderFlowImbalance as AcademicOFI,
    VPIN as AcademicVPIN,
    KyleLambda,
    AmihudIlliquidity,
    RollSpread,
    AcademicMicrostructureFeatures,
    generate_microstructure_features
)
from .academic_volatility import (
    RealizedVolatility,
    HARRV,
    RangeBasedVolatility,
    AcademicVolatilityFeatures,
    generate_volatility_features
)
from .academic_regime import (
    GaussianHMM,
    RegimeFeatures,
    RollingRegimeDetector,
    AcademicRegimeFeatures,
    generate_regime_features
)
from .academic_market_making import (
    MarketMakingConfig,
    AvellanedaStoikov,
    MarketMakingFeatures,
    AcademicMarketMakingFeatures,
    generate_market_making_features
)
from .academic_deep_learning import (
    TFTConfig,
    iTransformerConfig,
    TimeXerConfig,
    NBEATSConfig,
    TimeSeriesLibraryModels,
    DeepLearningFeatures,
    generate_deep_learning_features
)

# ══════════════════════════════════════════════════════════════
# EASTERN ASIA GOLD STANDARD (2026-01-17):
# ══════════════════════════════════════════════════════════════
# Research from China, Japan, Korea, Singapore/HK/Taiwan:
# - MOE (Mixture of Experts): MIGA architecture - 24% excess return
# - GNN Temporal: Graph attention, message passing from Tsinghua
# - Korea Quant: MS-GARCH, CGMY Lévy, VKOSPI (KAIST/KRX)
# - Asia FX Spread: CNH-CNY, HKD peg exploitation (HK academia)
# - MARL Trading: Multi-agent RL dynamics (Fudan/国泰君安)
from .moe_trading import (
    MomentumExpert,
    MeanReversionExpert,
    VolatilityExpert,
    TrendExpert,
    GatingNetwork,
    MixtureOfExpertsFeatures,
    generate_moe_features
)
from .gnn_temporal import (
    GraphAttentionLayer,
    TemporalGNNFeatures,
    generate_gnn_features
)
from .korea_quant import (
    KoreaQuantFeatures,
    generate_korea_features
)
from .asia_fx_spread import (
    AsiaFXSpreadFeatures,
    generate_asia_fx_features
)
from .marl_trading import (
    AgentTypeDetector,
    MultiAgentRLFeatures,
    generate_marl_features
)

# ══════════════════════════════════════════════════════════════
# FUNDAMENTAL LIMITS ANALYSIS (2026-01-22):
# ══════════════════════════════════════════════════════════════
# Academic research on prediction limits with full citations:
# - Information Theory: Shannon (1948), Cover & Thomas (2006)
# - Chaos Detection: Lorenz (1963), Rosenstein (1993), BDS (1996)
# These prove why 99.99% accuracy is mathematically impossible
from .information_theory import (
    InformationTheoryFeatures,
    compute_market_efficiency_score,
    create_information_features,
)
from .chaos_detection import (
    ChaosDetector,
    estimate_prediction_horizon,
    create_chaos_features,
)

__all__ = [
    # Engine
    'HFTFeatureEngine',
    'FeatureConfig',
    'FastTechnicalFeatures',
    'create_hft_feature_engine',
    'FastFeatureEngine',
    'create_targets',
    # Chinese Alpha Libraries
    'Alpha101Complete',
    'Alpha158',
    'generate_alpha158',
    'Alpha360',
    'Alpha360Compact',
    'generate_alpha360',
    'generate_alpha360_compact',
    'BarraCNE6Forex',
    'generate_barra_cne6',
    # US Academic Factors
    'USAcademicFactors',
    'generate_us_academic_factors',
    # GitHub/Gitee Gold Standard
    'MLFinLabFeatures',
    'generate_mlfinlab_features',
    'TripleBarrierLabeling',
    'MetaLabeling',
    'TimeSeriesLibraryFeatures',
    'generate_timeseries_features',
    # Signals
    'RenaissanceSignalGenerator',
    'CrossAssetSignalGenerator',
    'CrossAssetSignal',
    # Mega Generator (806+ features)
    'MegaFeatureGenerator',
    'generate_mega_features',
    'generate_chinese_quant_features',
    'generate_academic_features',
    'generate_renaissance_features',
    # GitHub Gold Standard Microstructure (12 features)
    'OrderFlowImbalance',
    'OFIResult',
    'VPIN',
    'VPINResult',
    'VolatilityEstimators',
    'RiskMetrics',
    'KellyCriterion',
    'HawkesProcess',
    'GoldStandardFeatures',
    'generate_gold_standard_features',
    'calculate_kelly_position',
    # Gitee Chinese Factors (60-80 features) - 2026-01-17
    'Alpha191ForexFactors',
    'SmartMoneyFactor2',
    'VPINEnhanced',
    'IntegratedOFI',
    'GiteeVolatilityEstimators',
    'MarketImpactFactors',
    'MicropriceCalculator',
    'GiteeChineseFactorGenerator',
    'generate_gitee_chinese_factors',
    'get_citation_info',
    # Global Expansion (110 features) - 2026-01-17
    'IndiaQuantFeatures',
    'generate_india_features',
    'JapanQuantFeatures',
    'generate_japan_features',
    'EuropeQuantFeatures',
    'generate_europe_features',
    'EmergingMarketsFeatures',
    'generate_emerging_features',
    'UniversalMathFeatures',
    'generate_universal_math_features',
    # Reinforcement Learning Algorithms (35 features) - 2026-01-17
    'RLState',
    'QLearningTrader',
    'SARSATrader',
    'TDLambdaTrader',
    'DynaQTrader',
    'ThompsonSamplingBandit',
    'AlmgrenChrisExecutor',
    'RLFeatureGenerator',
    'generate_rl_features',
    'get_rl_citations',       # 17 academic references
    'print_rl_citations',     # Print formatted citations
    # Deep Reinforcement Learning - Gold Standard (2018-2024)
    # TD3 - Fujimoto et al. ICML 2018
    'TD3Config',
    'TD3Actor',
    'TD3Critic',
    'TD3Trader',
    # SAC - Haarnoja et al. ICML 2018
    'SACConfig',
    'SACGaussianActor',
    'SACCritic',
    'SACTrader',
    # EarnHFT - Qin et al. AAAI 2024 (+30% over baselines)
    'EarnHFTConfig',
    'EarnHFTSubAgent',
    'EarnHFTRouter',
    'EarnHFTTrader',
    # MacroHFT - Zong et al. KDD 2024
    'MacroHFTConfig',
    'MemoryAugmentedNetwork',
    'ConditionalAdapter',
    'MacroHFTTrader',
    # DeepScalper - Sun et al. CIKM 2022
    'DeepScalperConfig',
    'RiskAwareNetwork',
    'DeepScalperTrader',
    # Replay Buffers
    'ReplayBuffer',
    'PrioritizedReplayBuffer',
    # Deep RL Features
    'DeepRLFeatureGenerator',
    'generate_deep_rl_features',
    # Academic Research Modules (2026-01-17)
    # Microstructure (Cont 2014, Easley 2012, Kyle 1985)
    'AcademicOFI',
    'AcademicVPIN',
    'KyleLambda',
    'AmihudIlliquidity',
    'RollSpread',
    'AcademicMicrostructureFeatures',
    'generate_microstructure_features',
    # Volatility (Corsi 2009, Parkinson 1980, Garman-Klass 1980)
    'RealizedVolatility',
    'HARRV',
    'RangeBasedVolatility',
    'AcademicVolatilityFeatures',
    'generate_volatility_features',
    # Regime Detection (Hamilton 1989)
    'GaussianHMM',
    'RegimeFeatures',
    'RollingRegimeDetector',
    'AcademicRegimeFeatures',
    'generate_regime_features',
    # Market Making (Avellaneda-Stoikov 2008)
    'MarketMakingConfig',
    'AvellanedaStoikov',
    'MarketMakingFeatures',
    'AcademicMarketMakingFeatures',
    'generate_market_making_features',
    # Deep Learning (iTransformer ICLR 2024, TFT, N-BEATS)
    'TFTConfig',
    'iTransformerConfig',
    'TimeXerConfig',
    'NBEATSConfig',
    'TimeSeriesLibraryModels',
    'DeepLearningFeatures',
    'generate_deep_learning_features',
    # RL Research (22 Papers - IEEE, ICML, AAAI, NeurIPS, JMLR) - 2026-01-17
    # Reward Shaping (Moody & Saffell 2001)
    'DifferentialSharpeRatio',
    'DifferentialDownsideDeviation',
    'MaxDrawdownPenalty',
    'RewardConfig',
    'CombinedRewardFunction',
    # Risk-Sensitive RL (Tamar 2015, Chow 2017, Rockafellar 2000)
    'ValueAtRisk',
    'ConditionalValueAtRisk',
    'EntropicValueAtRisk',
    'CVaRConstrainedKelly',
    'RiskMetricsResult',
    'compute_risk_metrics',
    # Optimal Execution (Almgren-Chriss 2001, Nevmyvaka 2006)
    'ExecutionConfig',
    'AlmgrenChrissRL',
    'RLOptimalExecution',
    # Market Making RL (Avellaneda-Stoikov 2008, Spooner 2018)
    'RLMarketMakingConfig',
    'AvellanedaStoikovRL',
    # Distributional RL (Dabney AAAI 2018, Zhou NeurIPS 2020)
    'QuantileDistribution',
    'QuantileHuberLoss',
    # Meta-RL (Finn ICML 2017 - MAML)
    'MAMLTrader',
    'RegimeAwareRL',
    # Position Sizing RL
    'PositionSizerConfig',
    'RLPositionSizer',
    # Features & Factories
    'RLResearchFeatureGenerator',
    'create_reward_function',
    'create_execution_model',
    'create_market_maker',
    'generate_rl_research_features',
    # ══════════════════════════════════════════════════════════════
    # EASTERN ASIA GOLD STANDARD (2026-01-17) - 85 features
    # ══════════════════════════════════════════════════════════════
    # MOE (Mixture of Experts) - MIGA 24% excess return (20 features)
    'MomentumExpert',
    'MeanReversionExpert',
    'VolatilityExpert',
    'TrendExpert',
    'GatingNetwork',
    'MixtureOfExpertsFeatures',
    'generate_moe_features',
    # GNN Temporal (15 features)
    'GraphAttentionLayer',
    'TemporalGNNFeatures',
    'generate_gnn_features',
    # Korea Quant - MS-GARCH, CGMY, VKOSPI (20 features)
    'KoreaQuantFeatures',
    'generate_korea_features',
    # Asia FX Spread - CNH-CNY, HKD peg (15 features)
    'AsiaFXSpreadFeatures',
    'generate_asia_fx_features',
    # MARL Trading - Multi-Agent RL (15 features)
    'AgentTypeDetector',
    'MultiAgentRLFeatures',
    'generate_marl_features',
    # ══════════════════════════════════════════════════════════════
    # FUNDAMENTAL LIMITS ANALYSIS (2026-01-22) - Shannon, Lorenz
    # ══════════════════════════════════════════════════════════════
    # Information Theory (Shannon 1948, Cover & Thomas 2006)
    # Use: InformationTheoryFeatures().shannon_entropy(returns) for methods
    'InformationTheoryFeatures',
    'compute_market_efficiency_score',
    'create_information_features',
    # Chaos Detection (Lorenz 1963, Rosenstein 1993, BDS 1996)
    # Use: ChaosDetector().lyapunov_exponent(returns) for methods
    'ChaosDetector',
    'estimate_prediction_horizon',
    'create_chaos_features',
]
