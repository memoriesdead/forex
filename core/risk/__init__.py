"""
Risk Management Module
======================
Per-symbol and portfolio-level risk management.

FUNDAMENTAL LIMITS ANALYSIS (2026-01-22):
- Tail Risk: Taleb (2007), Mandelbrot (2004), Hill (1975)
- Power Law: Pareto VaR, Expected Shortfall, Knightian Uncertainty
- Black Swan Detection: EVT, GEV distribution fitting
"""

from .limits import RiskLimits
from .per_symbol import PerSymbolRiskManager
from .portfolio import PortfolioRiskManager

# Fundamental Limits - Tail Risk Analysis (2026-01-22)
# Citations: Taleb (2007), Mandelbrot (2004), Hill (1975), Knight (1921)
from .tail_risk import (
    TailRiskAnalyzer,
    TailRiskAnalysis,
    calculate_knightian_uncertainty,
    create_tail_risk_features,
)

__all__ = [
    'RiskLimits',
    'PerSymbolRiskManager',
    'PortfolioRiskManager',
    # Fundamental Limits - Tail Risk (2026-01-22)
    # Citations: Taleb (2007), Mandelbrot (2004), Hill (1975)
    # Use: TailRiskAnalyzer().hill_estimator(returns) for methods
    'TailRiskAnalyzer',
    'TailRiskAnalysis',
    'calculate_knightian_uncertainty',
    'create_tail_risk_features',
]
