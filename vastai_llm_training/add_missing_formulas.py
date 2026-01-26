"""Add missing HIGH priority formulas from Chinese Quant research."""
import json
from pathlib import Path

new_formulas = [
    {
        "instruction": "What is the Treynor Ratio?",
        "input": "",
        "output": """The Treynor Ratio was developed by Jack Treynor (1965) and measures risk-adjusted return using systematic risk.

**Formula:**
```
TR = (R_p - R_f) / beta_p
```

**Where:**
- R_p = Portfolio return
- R_f = Risk-free rate
- beta_p = Portfolio beta (systematic risk)

**Beta calculation:**
```
beta = Cov(R_p, R_m) / Var(R_m)
```

**Comparison to Sharpe:**
- Sharpe uses total risk (std dev)
- Treynor uses systematic risk only (beta)
- Use Treynor when portfolio is well-diversified

**Implementation:**
```python
def treynor_ratio(returns, benchmark_returns, rf_rate=0.02):
    excess = returns - rf_rate/252
    bench_excess = benchmark_returns - rf_rate/252
    cov = np.cov(excess, bench_excess)
    beta = cov[0,1] / cov[1,1]
    return np.mean(excess) * 252 / beta
```

**Citation:** Treynor, J. (1965). How to Rate Management of Investment Funds."""
    },
    {
        "instruction": "Explain Factor Turnover and Half-Life",
        "input": "",
        "output": """Factor Turnover and Half-Life are critical metrics for alpha decay analysis used by Chinese quant firms.

**Factor Turnover Formula:**
```
TO_t = Sum(|w_t - w_{t-1}|)
```

**Factor Half-Life Formula:**
```
HL = -log(2) / log(autocorr)
autocorr = corr(IC_t, IC_{t-1})
```

**Interpretation:**
- Half-life = Time for IC to decay to 50%
- Short HL (< 5 days) = Fast decay, need frequent rebalancing
- Long HL (> 20 days) = Slow decay, can hold positions longer

**Implementation:**
```python
def factor_turnover(weights_t, weights_prev):
    return np.sum(np.abs(weights_t - weights_prev))

def factor_half_life(ic_series):
    autocorr = np.corrcoef(ic_series[:-1], ic_series[1:])[0,1]
    if autocorr <= 0 or autocorr >= 1:
        return np.inf
    return -np.log(2) / np.log(autocorr)
```

**Used by:** 幻方量化, 九坤投资, 长江证券

**Citation:** Qlib factor analysis, 长江证券 Research (2024)"""
    },
    {
        "instruction": "What is Rank IC (Spearman IC)?",
        "input": "",
        "output": """Rank IC uses Spearman correlation for robust prediction quality measurement.

**Formula:**
```
Rank_IC = corr(rank(predictions), rank(actuals))
```

**ICIR (IC Information Ratio):**
```
ICIR = mean(IC) / std(IC)
```

**Advantages over Pearson IC:**
- More robust to outliers
- Captures monotonic relationships
- Better for fat-tailed distributions

**Implementation:**
```python
from scipy.stats import spearmanr

def rank_ic(predictions, actuals):
    return spearmanr(predictions, actuals)[0]

def ic_ir(ic_series):
    return np.mean(ic_series) / np.std(ic_series)
```

**Targets:**
- IC > 0.03: Weak but usable signal
- IC > 0.05: Good signal
- ICIR > 0.5: Production-ready

**Used by:** 明汯投资, All major Chinese quant firms

**Citation:** Qlib documentation, Microsoft Research"""
    },
    {
        "instruction": "Explain Lee-Mykland Jump Detection",
        "input": "",
        "output": """Lee-Mykland (2008) is a nonparametric test for detecting jumps in high-frequency data.

**Test Statistic:**
```
L_i = |r_i| / sigma_i
sigma_i = sqrt(BV_i)
```

**Bipower Variation:**
```
BV = (pi/2) * Sum(|r_t| * |r_{t-1}|) / (n-1)
```

**Critical Value:**
```
C_n = sqrt(2*log(n)) - (log(pi) + log(log(n))) / (2*sqrt(2*log(n)))
```

**Jump Detection:** Jump if L_i > C_n

**Implementation:**
```python
def lee_mykland_jump(returns, k=270):
    n = len(returns)
    jumps = np.zeros(n, dtype=bool)
    for i in range(k, n):
        window = returns[i-k:i]
        bv = (np.pi/2) * np.sum(np.abs(window[1:]) * np.abs(window[:-1])) / (k-1)
        sigma = np.sqrt(bv)
        L = np.abs(returns[i]) / sigma if sigma > 0 else 0
        c_n = np.sqrt(2*np.log(n)) - (np.log(np.pi) + np.log(np.log(n))) / (2*np.sqrt(2*np.log(n)))
        jumps[i] = L > c_n
    return jumps
```

**Citation:** Lee & Mykland (2008). Jumps in Financial Markets. Review of Financial Studies."""
    },
    {
        "instruction": "What is MAD Winsorization?",
        "input": "",
        "output": """MAD Winsorization is a robust outlier handling technique for factor preprocessing.

**Formula:**
```
x_win = clip(x, median - n*MAD*1.4826, median + n*MAD*1.4826)
MAD = median(|x - median(x)|)
```

**Where:**
- 1.4826 = consistency constant for normal distribution
- n = Number of MADs (typically 3)

**Why MAD over Standard Deviation:**
- MAD has 50% breakdown point (robust to outliers)
- Std dev is sensitive to extreme values
- Better for fat-tailed financial data

**Implementation:**
```python
def winsorize_mad(x, n_mad=3.0):
    median = np.median(x)
    mad = np.median(np.abs(x - median))
    lower = median - n_mad * mad * 1.4826
    upper = median + n_mad * mad * 1.4826
    return np.clip(x, lower, upper)
```

**Used by:** All Chinese quant firms for factor preprocessing

**Citation:** Standard factor engineering, Barra methodology"""
    },
    {
        "instruction": "Explain FEDformer Frequency Enhanced Attention",
        "input": "",
        "output": """FEDformer (ICML 2022) uses frequency domain attention for efficient time series forecasting.

**Key Innovation:**
- Standard attention: O(L^2) complexity
- FEDformer: O(L) complexity via Fourier transform

**Frequency Enhanced Attention:**
```
FEA(Q, K, V) = IFFT(FFT(Q) * FFT(K)^*) * V
```

**Key insight:** Most time series have sparse representation in Fourier basis.

**Implementation:**
```python
class FEA(nn.Module):
    def __init__(self, d_model, modes=32):
        self.modes = modes  # Keep top-k Fourier modes

    def forward(self, q, k, v):
        q_fft = torch.fft.rfft(q, dim=-2)[:, :, :self.modes]
        k_fft = torch.fft.rfft(k, dim=-2)[:, :, :self.modes]
        attn = q_fft * k_fft.conj()
        return torch.fft.irfft(attn, dim=-2)
```

**Results:** 14.8% error reduction vs SOTA.

**Citation:** Zhou et al. (2022). FEDformer. ICML."""
    },
    {
        "instruction": "What are Tick Imbalance Bars?",
        "input": "",
        "output": """Tick Imbalance Bars (TIB) trigger based on order flow imbalance, not time.

**Formula:**
```
TI = Sum(sign(Delta_P_i) * V_i)
Trigger new bar when |TI| > E[TI]
```

**Advantages over time bars:**
- Synchronized with information arrival
- More uniform statistical properties
- Better for ML models

**Implementation:**
```python
def tick_imbalance_bars(prices, volumes, expected_imb):
    bars = []
    tick_imb = 0
    current_start = 0

    for i in range(1, len(prices)):
        sign = np.sign(prices[i] - prices[i-1])
        tick_imb += sign * volumes[i]

        if np.abs(tick_imb) > expected_imb:
            bars.append({
                'start': current_start, 'end': i,
                'open': prices[current_start],
                'close': prices[i],
                'imbalance': tick_imb
            })
            current_start = i + 1
            tick_imb = 0
    return bars
```

**Used by:** 九坤投资 for HFT

**Citation:** Lopez de Prado (2018). Advances in Financial Machine Learning."""
    },
    {
        "instruction": "Explain BNS Jump Test (Barndorff-Nielsen Shephard)",
        "input": "",
        "output": """BNS Jump Test detects jumps using the difference between realized and bipower variation.

**Test Statistic:**
```
Z = (RV - BV) / sqrt(Theta * max(RQ, BQ))
```

**Where:**
- RV = Realized Variance = Sum(r_i^2)
- BV = Bipower Variation = (pi/2) * Sum(|r_i| * |r_{i-1}|)
- RQ = Realized Quarticity
- BQ = Bipower Quarticity
- Theta = adjustment factor

**Key Insight:**
- RV captures both continuous and jump variation
- BV is robust to jumps (only continuous)
- Difference RV - BV isolates jump component

**Implementation:**
```python
def bns_jump_test(returns):
    n = len(returns)
    rv = np.sum(returns**2)
    bv = (np.pi/2) * np.sum(np.abs(returns[1:]) * np.abs(returns[:-1]))

    # Realized quarticity
    rq = (n/3) * np.sum(returns**4)

    # Test statistic
    theta = (np.pi**2/4 + np.pi - 5) * max(rq, bv**2)
    z = (rv - bv) / np.sqrt(theta) if theta > 0 else 0

    # p-value (standard normal)
    from scipy.stats import norm
    p_value = 1 - norm.cdf(z)
    return z, p_value
```

**Citation:** Barndorff-Nielsen & Shephard (2006). Econometrica."""
    },
    {
        "instruction": "What is Factor Neutralization?",
        "input": "",
        "output": """Factor Neutralization removes unwanted exposures (industry, size) from alpha factors.

**Formula:**
```
factor_neutral = factor - beta_ind * r_ind - beta_size * r_size
```

**Regression-based neutralization:**
```
factor = alpha + beta_1*industry + beta_2*log(mcap) + epsilon
factor_neutral = epsilon (residual)
```

**Types of Neutralization:**
1. Industry neutralization: Remove sector effects
2. Size neutralization: Remove market cap effects
3. Beta neutralization: Remove market exposure

**Implementation:**
```python
from sklearn.linear_model import LinearRegression

def neutralize_factor(factor, industry, market_cap):
    X = np.column_stack([
        pd.get_dummies(industry),  # Industry dummies
        np.log(market_cap)         # Log market cap
    ])
    model = LinearRegression().fit(X, factor)
    return factor - model.predict(X)
```

**Why Neutralize:**
- Isolate pure alpha from risk factors
- Required by Barra risk model
- Reduces unintended bets

**Used by:** All institutional quant firms

**Citation:** Barra CNE5/CNE6 methodology, MSCI"""
    },
    {
        "instruction": "Explain IC Decay Analysis",
        "input": "",
        "output": """IC Decay measures how factor predictive power decreases over time horizons.

**Formula:**
```
IC_tau = E[corr(signal_t, return_{t+tau})]
for tau = 1, 5, 10, 20, 60 days
```

**IC Decay Curve:**
```
IC(tau) = IC_0 * exp(-tau / lambda)
```
Where lambda = decay constant

**Key Metrics:**
1. Peak IC horizon: tau where IC is maximum
2. Half-life: tau where IC drops to 50%
3. Decay rate: -d(IC)/d(tau)

**Implementation:**
```python
def ic_decay_curve(signals, returns, horizons=[1,5,10,20,60]):
    ic_curve = {}
    for h in horizons:
        future_returns = returns.shift(-h)
        ic = signals.corrwith(future_returns).mean()
        ic_curve[h] = ic
    return ic_curve

def fit_decay_constant(ic_curve):
    from scipy.optimize import curve_fit
    def exp_decay(x, ic0, lam):
        return ic0 * np.exp(-x / lam)

    x = np.array(list(ic_curve.keys()))
    y = np.array(list(ic_curve.values()))
    popt, _ = curve_fit(exp_decay, x, y, p0=[y[0], 20])
    return popt[1]  # lambda (decay constant)
```

**Trading Implications:**
- Fast decay (HL < 5d): High turnover strategy
- Slow decay (HL > 20d): Low turnover, lower costs

**Used by:** 幻方量化, 长江证券

**Citation:** Qlib factor analysis framework"""
    },
    {
        "instruction": "What is WaveNet for Financial Time Series?",
        "input": "",
        "output": """WaveNet uses dilated causal convolutions for financial time series prediction.

**Dilated Causal Convolution:**
```
y_t = Sum_k(w_k * x_{t - d*k})
```
Where d = 2^layer (dilation rate doubles each layer)

**Key Properties:**
- Receptive field grows exponentially: RF = 2^n * kernel_size
- Causal: No future information leakage
- Efficient: Fewer parameters than RNNs

**Gated Activation:**
```
z = tanh(W_f * x) * sigmoid(W_g * x)
```

**Architecture:**
```
Input -> [Dilated Conv (d=1) -> Gated -> Residual] ->
      -> [Dilated Conv (d=2) -> Gated -> Residual] ->
      -> [Dilated Conv (d=4) -> Gated -> Residual] ->
      -> Skip connections -> Output
```

**Implementation:**
```python
class WaveNetBlock(nn.Module):
    def __init__(self, channels, kernel_size, dilation):
        self.conv = nn.Conv1d(channels, channels*2,
                              kernel_size, dilation=dilation,
                              padding=(kernel_size-1)*dilation)

    def forward(self, x):
        out = self.conv(x)[:, :, :x.size(2)]  # Causal
        gate, filt = out.chunk(2, dim=1)
        return x + torch.tanh(filt) * torch.sigmoid(gate)
```

**Results:** 20-37% better than naive baselines for price prediction

**Citation:** van den Oord et al. (2016). WaveNet.
Borovykh et al. (2017). arXiv:1703.04691"""
    }
]

# Append to high_quality_formulas.jsonl
data_path = Path(__file__).parent / "data" / "high_quality_formulas.jsonl"

with open(data_path, 'a', encoding='utf-8') as f:
    for formula in new_formulas:
        f.write(json.dumps(formula, ensure_ascii=False) + '\n')

print(f"Added {len(new_formulas)} new HIGH priority formulas from Chinese Quant research")
print(f"File: {data_path}")

# Verify total count
with open(data_path, 'r', encoding='utf-8') as f:
    total = sum(1 for _ in f)
print(f"Total formulas now: {total}")
