# Deep Research Prompt: 100% Certainty Trading Framework

**Copy this prompt for deep research sessions:**

---

## PROMPT

I have an HFT forex trading system showing 82% accuracy. But accuracy isn't enough - Renaissance Technologies has only 50.75% accuracy but 100% CERTAINTY about that edge. I need to achieve 99.99% mathematical certainty about my edge.

Research the following with academic rigor, mathematical proofs, and implementable code:

### 1. CONFORMAL PREDICTION FOR TRADING
How can I apply conformal prediction to get GUARANTEED coverage on my forex predictions? I want prediction sets where I know with 99% probability the true outcome is in my set. If the set is {UP, DOWN}, I don't trade. If the set is {UP}, I trade with certainty.

Find: Mathematical framework, time series adaptations, Python implementation.

### 2. CALIBRATED UNCERTAINTY QUANTIFICATION
How do I get confidence scores that are CALIBRATED - meaning when my model says 90% confident, it's actually right 90% of the time? Cover ensemble disagreement, Bayesian methods, temperature scaling, and expected calibration error.

Find: Methods used by Two Sigma, Citadel. Calibration techniques. Implementation.

### 3. STATISTICAL PROOF OF EDGE
Given 820 wins out of 1000 trades (82%), what statistical tests PROVE (not suggest) this edge is real and not luck? Account for multiple hypothesis testing, look-ahead bias, and overfitting. I need p < 0.0001.

Find: Deflated Sharpe Ratio, probability of backtest overfitting, bootstrap methods, minimum sample size for 99.99% certainty.

### 4. REGIME VALIDITY DETECTION
My model might be 95% accurate in Regime A but 55% in Regime B. How do I detect in real-time which regime I'm in, and ONLY trade when I'm in a regime where my model is provably valid?

Find: Hidden Markov Models, online changepoint detection, regime-conditional performance analysis.

### 5. REAL-TIME EDGE DECAY DETECTION
How do I detect within 50 trades that my edge has decayed from 82% to 77%? I need sequential statistical tests that give early warning with 99% certainty.

Find: CUSUM, SPRT, Bayesian online changepoint detection. Speed vs accuracy tradeoffs.

### 6. INFORMATION-THEORETIC EDGE
Can I measure my edge in bits/trade using information theory? This would be a fundamental, unfakeable measure. Cover mutual information between predictions and outcomes, and how bits relate to betting edge.

Find: Information theory in finance, Kelly criterion connection to information, implementation.

### 7. OPTIMAL TRADE FILTERING
If I only take trades where confidence > threshold, what's the optimal threshold? There's a tradeoff: fewer trades but higher certainty. Find the mathematical optimum.

Find: Signal detection theory, optimal stopping, threshold optimization.

### 8. ROBUST KELLY CRITERION
Standard Kelly assumes I KNOW my true win probability. But I only have an estimate with uncertainty (82% +/- 4%). How do I size positions accounting for this uncertainty?

Find: Bayesian Kelly, worst-case Kelly, fractional Kelly justification, robust optimization.

### SYNTHESIS
How do I combine all 8 components into ONE trading decision framework where:
- Every trade has 99.99% certainty of positive EV
- I know my exact edge with tight confidence intervals
- I automatically stop when certainty drops
- Position size reflects certainty level

Give me the mathematical framework and Python architecture.

### OUTPUT FORMAT
For each topic:
1. Mathematical foundation (theorems, proofs)
2. Key academic references (papers with citations)
3. Python implementation sketch
4. How it integrates with the other components

### CONTEXT
- HFT forex system, 51 currency pairs
- XGBoost + LightGBM + CatBoost ensemble
- 575 features per prediction
- Need real-time (sub-second) certainty scoring
- Python 3.11, numpy, scipy, sklearn available

---

## KEY INSIGHT TO EMPHASIZE

**Renaissance doesn't try to be right on every trade. They try to be CERTAIN about their edge.**

A 50.75% win rate with 100% certainty beats an 82% win rate with unknown certainty.

My goal: 82% accuracy with 99.99% certainty that it's really 82%, with the ability to detect instantly when it's not.

**When we trade, we are mathematically certain. When we're not certain, we don't trade.**
