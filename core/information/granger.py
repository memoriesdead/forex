"""
Granger Causality Tester
========================
Tests if time series X "Granger-causes" Y.

X Granger-causes Y if past values of X provide statistically significant
information about future values of Y, beyond what is provided by past values of Y alone.

Research basis:
- Granger (1969): Investigating Causal Relations by Econometric Models
- Toda & Yamamoto (1995): Statistical inference in VAR models
- Dimpfl & Peter (2013): Comparison with Transfer Entropy

Limitations (why Transfer Entropy is better):
- Only detects LINEAR causality
- Misses nonlinear relationships
- Fixed lag structure (less flexible)

Chinese Quant usage:
- 幻方量化: Pre-filter for TE analysis
- 九坤投资: Feature pair validation
- Fast screening before expensive TE calculation

Expected gain: +1-2% accuracy (use as preprocessing for TE)
