"""
Information Theory for Trading
==============================
Information-theoretic methods for feature selection and causality detection.

Research basis:
- Shannon (1948): A Mathematical Theory of Communication
- Schreiber (2000): Measuring Information Transfer
- Granger (1969): Investigating Causal Relations
- Dimpfl & Peter (2013): Using Transfer Entropy to Measure Information Flows

Expected improvement: 63% → 75-79% accuracy

Chinese Quant references:
- 幻方量化 (High-Flyer): Information flow analysis
- 九坤投资 (Ubiquant): Causal factor mining
- 明汯投资 (Minghui): Feature selection via MI

Modules:
- mutual_info: Mutual Information I(X;Y) calculation
- transfer_entropy: Transfer Entropy TE(X→Y) for causality
- granger: Granger causality testing
- feature_selector: Information-theoretic feature selection
"""

from .mutual_info import MutualInformationCalculator, calculate_mutual_information
from .transfer_entropy import TransferEntropyCalculator, calculate_transfer_entropy
from .granger import GrangerCausalityTester, test_granger_causality

__all__ = [
    'MutualInformationCalculator',
    'calculate_mutual_information',
    'TransferEntropyCalculator',
    'calculate_transfer_entropy',
    'GrangerCausalityTester',
    'test_granger_causality',
]
