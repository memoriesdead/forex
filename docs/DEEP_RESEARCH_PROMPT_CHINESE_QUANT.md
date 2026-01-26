# DEEP RESEARCH PROMPT: Chinese Quant Gold Standard

**Use this prompt with Claude, GPT-4, Gemini, or any AI with web search/deep research capabilities.**

---

## RESEARCH PROMPT

```
I am building a world-class forex trading LLM fine-tuned on DeepSeek-R1-Distill-Qwen-7B.

My current training data includes:
- 33,538 training samples
- 1,172+ mathematical formulas
- 18 certainty validation modules
- 51 forex pairs with real trade outcomes

FORMULAS I ALREADY HAVE:
- Alpha Factors: Alpha101 (WorldQuant), Alpha191 (国泰君安), Alpha360/Alpha158 (Microsoft Qlib)
- Volatility: HAR-RV, GARCH, EGARCH, GJR-GARCH, Yang-Zhang, Garman-Klass, Parkinson, Rogers-Satchell
- Microstructure: VPIN, Kyle Lambda, OFI, Microprice, Amihud ILLIQ, Roll Spread, Adverse Selection
- Risk: Kelly Criterion, Sharpe, Sortino, Calmar, VaR, CVaR, Expected Shortfall, Max Drawdown
- Execution: Almgren-Chriss, Avellaneda-Stoikov, TWAP, VWAP, Implementation Shortfall
- RL: GRPO (DeepSeek), PPO, DQN, SAC, TD3, A2C, DDPG, TD Error
- Deep Learning: Transformer, Attention, LSTM, GRU, TCN, Informer, Autoformer, PatchTST, TimesNet, iTransformer
- Chinese Quant: Barra CNE6, ICIR, Factor Decay, Neutralization, Z-Score
- Statistical: Conformal Prediction, BOCPD, HMM Regime Detection, Kalman Filter, Bootstrap

CHINESE QUANT SOURCES I'M USING:
- 幻方量化 (High-Flyer Quant) - $8B AUM
- 九坤投资 (Ubiquant) - 600B RMB AUM
- 明汯投资 (Minghui Capital) - 400 PFlops AI
- 国泰君安 (Guotai Junan) - Alpha191
- DeepSeek (深度求索) - GRPO algorithm

YOUR TASK:
Search Chinese GitHub, Gitee, CSDN, Zhihu, and academic sources to find:

1. **MISSING ALPHA FACTORS** - Any alpha factors from Chinese quant firms NOT in my list
   - Search: "量化因子", "alpha因子", "选股因子", "因子挖掘"
   - Look for: 聚宽因子, 米筐因子, 优矿因子, BigQuant因子

2. **MISSING VOLATILITY/RISK MODELS** - Chinese-specific risk models
   - Search: "波动率模型", "风险模型", "VaR模型"
   - Look for: 中国市场特有的风险因子

3. **MISSING RL/ONLINE LEARNING** - Chinese quant online learning techniques
   - Search: "在线学习", "增量学习", "模型热更新", "概念漂移"
   - Look for: 幻方/九坤的实时更新方法

4. **MISSING DEEP LEARNING ARCHITECTURES** - SOTA Chinese time series models
   - Search: "时间序列预测", "股票预测模型", "Transformer金融"
   - Look for: 清华/北大/中科院最新模型

5. **MISSING EXECUTION ALGORITHMS** - Chinese market-specific execution
   - Search: "算法交易", "最优执行", "交易成本模型"
   - Look for: 中国市场的流动性模型, 冲击成本

6. **MISSING CERTAINTY/VALIDATION** - Chinese model validation techniques
   - Search: "模型验证", "回测框架", "样本外测试"
   - Look for: 过拟合检测, 因子有效性检验

7. **GITEE REPOSITORIES** - Search these specifically:
   - https://gitee.com/explore/ai
   - https://gitee.com/explore/machine-learning
   - Search: "量化交易", "quant", "alpha", "因子"

8. **CSDN/ZHIHU ARTICLES** - Search for:
   - "幻方量化 技术", "九坤 因子", "量化私募 算法"
   - "机器学习 量化", "深度学习 股票"

OUTPUT FORMAT:
For each technique/formula found that I DON'T already have, provide:

| Technique | Formula/Algorithm | Source | Citation | Priority |
|-----------|-------------------|--------|----------|----------|
| Name | Mathematical formula or pseudocode | GitHub/Gitee URL or paper | Author/Year | High/Medium/Low |

Also provide:
1. Python implementation snippet if available
2. Why it's important for forex trading
3. Which Chinese firm uses it (if known)

FOCUS ON:
- Techniques with proven track record at Chinese hedge funds
- Mathematical formulas with clear definitions
- Open-source implementations I can integrate
- Anything that could improve certainty/accuracy

DO NOT INCLUDE:
- Techniques I already have (listed above)
- Generic ML without financial application
- Theoretical concepts without formulas
- Techniques without implementation details
```

---

## ALTERNATIVE SHORTER PROMPT

```
Search Chinese quant sources (Gitee, CSDN, Zhihu, 幻方, 九坤) for alpha factors, volatility models, RL algorithms, and certainty validation techniques NOT in this list:

ALREADY HAVE: Alpha101, Alpha191, Alpha360, Alpha158, HAR-RV, GARCH variants, VPIN, Kyle Lambda, OFI, Kelly, Sharpe, Sortino, VaR, CVaR, Almgren-Chriss, Avellaneda-Stoikov, GRPO, PPO, DQN, SAC, Transformer, LSTM, iTransformer, HMM, Kalman, Conformal Prediction, BOCPD, Barra CNE6, ICIR

Find formulas I'm MISSING with:
1. Mathematical definition
2. Python code or pseudocode
3. Source URL (Gitee/GitHub/paper)
4. Why it matters for forex

Output as markdown table.
```

---

## SPECIFIC SEARCH QUERIES TO TRY

### Gitee (Chinese GitHub)
```
site:gitee.com 量化因子 alpha
site:gitee.com 高频交易 因子
site:gitee.com qlib alpha
site:gitee.com 机器学习 股票预测
site:gitee.com 深度学习 时间序列
site:gitee.com 强化学习 交易
```

### CSDN (Chinese Stack Overflow)
```
site:csdn.net 幻方量化 因子
site:csdn.net 九坤投资 算法
site:csdn.net Alpha191 实现
site:csdn.net 量化交易 深度学习
site:csdn.net 在线学习 量化
```

### Zhihu (Chinese Quora)
```
site:zhihu.com 量化私募 技术栈
site:zhihu.com 高频交易 因子
site:zhihu.com 机器学习 量化投资
site:zhihu.com 深度学习 股票
```

### Academic (Chinese Papers)
```
site:arxiv.org chinese stock prediction
site:arxiv.org alpha factor mining
site:semanticscholar.org quantitative trading china
site:papers.ssrn.com chinese quant factors
```

---

## KNOWN GAPS TO RESEARCH

Based on my audit, these specific techniques are MISSING or WEAK:

| Gap | What to Search | Why Important |
|-----|----------------|---------------|
| Treynor Ratio | "特雷诺比率" | Risk-adjusted return metric |
| Actor-Critic | "Actor-Critic 交易" | RL architecture variant |
| WaveNet | "WaveNet 时间序列" | Dilated causal convolutions |
| FEDformer | "FEDformer" | Frequency domain transformer |
| CNE5 | "Barra CNE5" | Earlier Barra China model |
| Factor Turnover | "因子换手率" | Factor decay measurement |
| Winsorization | "缩尾处理 因子" | Outlier handling |

---

## EXPECTED OUTPUT EXAMPLE

After research, you should get results like:

| Technique | Formula | Source | Priority |
|-----------|---------|--------|----------|
| Turnover Factor | `TO = Σ|w_t - w_{t-1}|` | Gitee/xxx/alpha | High |
| Jump Detection | `J_t = I(|r_t| > 3σ)` | CSDN article | Medium |
| Tick Imbalance | `TI = Σsign(ΔP) × V` | 九坤 paper | High |
| ... | ... | ... | ... |

---

## AFTER RESEARCH

Once you have the research results, bring them back and I will:

1. Evaluate each technique for forex applicability
2. Add high-priority ones to training data
3. Update the training script
4. Re-run the audit to verify

**Goal: Mathematical science perfection with every known Chinese quant technique included.**
