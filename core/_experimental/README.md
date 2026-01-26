# Experimental Modules

This directory contains non-production modules that are experimental, deprecated, or under development.

## Contents (38 modules)

These modules were moved here during the 2026-01-16 codebase reorganization:

- Advanced ML models (gold_standard_models, institutional_predictor, etc.)
- Specialized features (alpha191_guotaijunan, attention_factors, etc.)
- Research tools (genetic_factor_mining, graph_neural_network, etc.)
- Legacy implementations (various deprecated approaches)

## Usage

These modules may have broken imports or dependencies. If you need functionality from here:

1. Check if equivalent functionality exists in the production modules (core/ml/, core/features/, etc.)
2. If not, fix imports and test thoroughly before using
3. Consider migrating useful code to production modules

## Production Modules

Use these instead:

| Need | Use |
|------|-----|
| GPU config | `core.ml.gpu_config` |
| ML ensemble | `core.ml.ensemble` |
| Retraining | `core.ml.retrainer` |
| Data loading | `core.data.loader` |
| Tick buffer | `core.data.buffer` |
| Features | `core.features.engine` |
| Alpha101 | `core.features.alpha101` |
| Renaissance | `core.features.renaissance` |
| Order book | `core.execution.order_book` |
| Backtesting | `core.execution.backtest` |
