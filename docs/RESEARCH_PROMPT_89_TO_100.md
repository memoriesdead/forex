# Deep Research Prompt: 89% → 99.999% Certainty

```
╔══════════════════════════════════════════════════════════════════════════════════╗
║                         STRICT RESEARCH PROTOCOL                                 ║
╠══════════════════════════════════════════════════════════════════════════════════╣
║  RULE 1: Search in NATIVE LANGUAGE of the source country                         ║
║  RULE 2: Only GOLD STANDARD sources (top funds, top conferences, top journals)   ║
║  RULE 3: DO NOT suggest anything in "WHAT WE HAVE" section                       ║
║  RULE 4: Must provide CITATION + IMPLEMENTATION for each technique               ║
║  RULE 5: Focus on 8 SPECIFIC GAPS only - nothing else                            ║
╚══════════════════════════════════════════════════════════════════════════════════╝
```

---

# PART 1: WHAT WE ALREADY HAVE (DO NOT SUGGEST THESE)

## Complete Inventory - ALREADY IMPLEMENTED

### 1.1 Base Model
- ✅ DeepSeek-R1-Distill-Qwen-7B (幻方量化 base)
- ✅ LoRA: r=64, alpha=128, 7 modules
- ✅ SFT + DPO pipeline
- ✅ GGUF Q8_0 (7.54GB)
- ✅ Ollama deployment

### 1.2 Training Data (33,310 samples)
| Dataset | Count |
|---------|-------|
| deepseek_finetune_dataset.jsonl | 11,628 |
| high_quality_formulas.jsonl | 53 |
| llm_finetune_samples.jsonl | 3,529 |
| real_feature_examples.jsonl | 10,200 |
| real_dpo_pairs.jsonl | 2,550 |
| real_trade_scenarios.jsonl | 5,100 |
| GRPO data | 250 |

### 1.3 Formulas We Have (DO NOT SUGGEST)

**Alpha Factors (HAVE):**
- ❌ Alpha101 (WorldQuant) - 62 formulas
- ❌ Alpha158 (Qlib) - 158 formulas
- ❌ Alpha191 (国泰君安) - 191 formulas
- ❌ Alpha360 (Qlib) - 360 formulas

**Volatility (HAVE):**
- ❌ HAR-RV (Corsi 2009)
- ❌ GARCH, EGARCH, GJR-GARCH
- ❌ Garman-Klass, Parkinson, Rogers-Satchell, Yang-Zhang
- ❌ TSRV, Realized Variance, Bipower Variation

**Microstructure (HAVE):**
- ❌ VPIN (Easley 2012)
- ❌ Kyle Lambda (Kyle 1985)
- ❌ OFI (Cont 2014)
- ❌ PIN, Amihud ILLIQ, Roll Spread, Microprice

**Risk (HAVE):**
- ❌ Kelly Criterion, Fractional Kelly
- ❌ VaR, CVaR/ES
- ❌ Sharpe, Sortino, Calmar, Treynor, Max DD

**Execution (HAVE):**
- ❌ Almgren-Chriss (2001)
- ❌ TWAP, VWAP
- ❌ Avellaneda-Stoikov (2008)

**Regime (HAVE):**
- ❌ HMM 3-state
- ❌ BOCPD (Adams & MacKay 2007)

**Deep Learning (HAVE):**
- ❌ iTransformer (ICLR 2024)
- ❌ TimeXer (NeurIPS 2024)
- ❌ TimeMixer, PatchTST
- ❌ FEDformer (ICML 2022)
- ❌ WaveNet

**RL (HAVE):**
- ❌ GRPO (DeepSeek 2025)
- ❌ PPO, DQN, SAC, TD, GAE

**Chinese Quant (HAVE):**
- ❌ Factor Turnover/Half-Life (长江证券)
- ❌ Rank IC/Spearman IC (明汯)
- ❌ Lee-Mykland Jump, BNS Jump Test
- ❌ MAD Winsorization (Barra)
- ❌ Tick Imbalance Bars (Lopez de Prado)
- ❌ Factor Neutralization (Barra CNE5/6)
- ❌ IC Decay Analysis (幻方)

### 1.4 ML Models (HAVE)
- ❌ XGBoost GPU (depth=12)
- ❌ LightGBM GPU (leaves=511)
- ❌ CatBoost GPU (depth=12)
- ❌ 54 pairs trained, 575 features

### 1.5 Certainty Modules (HAVE)
- ❌ EdgeProof
- ❌ VPIN Toxicity
- ❌ BOCPD Changepoint
- ❌ CertaintyValidator
- ❌ BayesianSizing
- ❌ UncertaintyDecomposition
- ❌ Basic conformal prediction
- ❌ Temperature scaling concept

---

# PART 2: WHERE TO SEARCH (BY COUNTRY & LANGUAGE)

## 2.1 CHINA (Search in Chinese - 中文)

### Gold Standard Sources - Chinese Quant

**Top Funds to Research:**
| Fund | Chinese | AUM | Specialty |
|------|---------|-----|-----------|
| High-Flyer | 幻方量化 | $8B+ | AI全自动化, 萤火平台 |
| Ubiquant | 九坤投资 | 600B RMB | HFT, 机器学习 |
| Minghui | 明汯投资 | $10B+ | 400P算力, 深度学习 |
| Ruitian | 锐天投资 | - | 做市商, 高频 |
| Lingjun | 灵均投资 | 500B RMB | 量化多头 |

**Chinese Search Terms:**
```
量化交易置信度校准 (quant trading confidence calibration)
模型不确定性量化 (model uncertainty quantification)
在线学习实盘更新 (online learning live update)
LoRA微调生产环境 (LoRA fine-tuning production)
外汇另类数据 (forex alternative data)
央行沟通NLP分析 (central bank communication NLP)
订单流预测模型 (order flow prediction model)
极端风险EVT (extreme risk EVT)
尾部风险管理 (tail risk management)
集成学习元模型 (ensemble learning meta-model)
模型动态加权 (dynamic model weighting)
概念漂移检测 (concept drift detection)
```

**Chinese Platforms to Search:**
- 知乎 (Zhihu) - Search: "量化 置信度" "在线学习 实盘"
- CSDN - Search: "模型校准" "不确定性估计"
- BigQuant Wiki - https://bigquant.com/wiki/
- 聚宽 (JoinQuant) - https://www.joinquant.com/
- 米筐 (RiceQuant) - https://www.ricequant.com/
- 优矿 (Uqer) - https://uqer.datayes.com/
- 万矿 (WindQuant) - https://www.windquant.com/

**Chinese Academic:**
- 中国知网 CNKI - https://www.cnki.net/
- 万方数据 - https://www.wanfangdata.com.cn/
- Search: "量化投资" "机器学习" "置信区间"

**Chinese Conferences:**
- 中国量化投资学会年会
- 金融科技峰会
- AI金融应用大会

---

## 2.2 USA (Search in English)

### Gold Standard Sources - US Quant

**Top Funds/Firms:**
| Firm | Specialty | Public Research |
|------|-----------|-----------------|
| Renaissance Technologies | Statistical arbitrage | Jim Simons interviews |
| Two Sigma | ML/AI quant | Two Sigma blog, papers |
| DE Shaw | Systematic trading | D.E. Shaw research |
| Citadel | Market making, HFT | Ken Griffin talks |
| Jump Trading | HFT | Jump research blog |
| Jane Street | Market making | Jane Street tech blog |
| AQR | Factor investing | AQR insights (public) |
| Point72 | Multi-strategy | Steve Cohen interviews |

**US Academic - GOLD STANDARD ONLY:**

**Conferences (Tier 1 only):**
- NeurIPS (Neural Information Processing Systems)
- ICML (International Conference on Machine Learning)
- ICLR (International Conference on Learning Representations)
- KDD (Knowledge Discovery and Data Mining)
- AAAI (Association for AI)
- ACL (Association for Computational Linguistics) - for NLP

**Journals (Tier 1 only):**
- Journal of Finance (JF)
- Review of Financial Studies (RFS)
- Journal of Financial Economics (JFE)
- Journal of Financial and Quantitative Analysis (JFQA)
- Quantitative Finance
- Journal of Machine Learning Research (JMLR)

**Search Terms:**
```
"confidence calibration" "trading systems"
"uncertainty quantification" "financial machine learning"
"online learning" "production trading"
"model drift detection" "concept drift"
"FX alternative data" "forex NLP"
"central bank communication" "FOMC analysis"
"order flow prediction" "forex"
"Hawkes process" "FX markets"
"extreme value theory" "currency"
"flash crash prediction"
"ensemble meta-learner" "time series"
"dynamic model weighting" "regime"
"conformal prediction" "finance"
```

**US Platforms:**
- arXiv q-fin (https://arxiv.org/list/q-fin/recent)
- SSRN Finance (https://www.ssrn.com/index.cfm/en/fin/)
- Google Scholar
- Semantic Scholar
- Papers with Code (https://paperswithcode.com/)

**US Quant Blogs (Gold Standard):**
- QuantStart (https://www.quantstart.com/)
- Quantitative Research and Trading
- Two Sigma Engineering Blog
- Jane Street Tech Blog
- AQR Insights

---

## 2.3 UK (Search in English)

**Top Funds/Institutions:**
| Institution | Specialty |
|-------------|-----------|
| Man Group/AHL | Systematic trading, ML research |
| Winton Group | Statistical research |
| Oxford-Man Institute | Academic quant research |
| LSE Finance | Academic papers |
| Imperial College | ML for finance |

**Search Terms:**
```
"model calibration" "trading"
"uncertainty estimation" "portfolio"
"Bank of England" "communication analysis"
"FX microstructure" "London session"
```

---

## 2.4 JAPAN (Search in Japanese - 日本語)

**Search Terms:**
```
機械学習 信頼度校正 (machine learning confidence calibration)
量子取引 不確実性 (quant trading uncertainty)
為替予測 オルタナティブデータ (FX prediction alternative data)
日銀 コミュニケーション分析 (BOJ communication analysis)
オンライン学習 本番環境 (online learning production)
```

**Japanese Platforms:**
- CiNii (Japanese academic)
- J-STAGE (Japanese journals)
- Qiita (Japanese tech blog)

---

## 2.5 GERMANY (Search in German)

**Search Terms:**
```
Modellkalibrierung Handel (model calibration trading)
Unsicherheitsquantifizierung Finanzen (uncertainty quantification finance)
EZB Kommunikation Analyse (ECB communication analysis)
```

**German Institutions:**
- Deutsche Bundesbank research
- Frankfurt School of Finance

---

## 2.6 FRANCE (Search in French)

**Search Terms:**
```
calibration modèle trading
quantification incertitude finance
BCE communication analyse
```

**French Institutions:**
- Paris-Dauphine Finance
- HEC Paris

---

# PART 3: THE 8 GAPS TO RESEARCH

```
╔══════════════════════════════════════════════════════════════════════════════════╗
║  ONLY RESEARCH THESE 8 GAPS - NOTHING ELSE                                       ║
║  For each gap, search ALL countries in their NATIVE LANGUAGE                     ║
╚══════════════════════════════════════════════════════════════════════════════════╝
```

## GAP 1: CERTAINTY CALIBRATION (CRITICAL)

**What we have:** Basic conformal, temperature scaling concept
**What we need:** Production-grade calibration that makes "95% confident" = 95% accurate

**Search queries by language:**

| Language | Search Query |
|----------|--------------|
| English | "confidence calibration neural networks trading" site:arxiv.org OR site:neurips.cc |
| English | "Platt scaling" OR "isotonic regression" "financial" |
| English | "expected calibration error" "trading system" |
| Chinese | "置信度校准" "神经网络" "交易" site:zhihu.com |
| Chinese | "模型校准" "金融" site:cnki.net |
| Japanese | "信頼度校正" "機械学習" "取引" |

**Specific papers to find:**
- Guo et al. (2017) "On Calibration of Modern Neural Networks" - and its extensions to finance
- Any paper applying calibration to trading systems
- Chinese quant papers on 置信度校准

**Expected output:** Implementation of ECE measurement + Platt/isotonic scaling for our models

---

## GAP 2: FOREX-SPECIFIC ALTERNATIVE DATA

**What we have:** Price, volume, technicals only
**What we need:** Central bank NLP, positioning data, carry signals

**Search queries by language:**

| Language | Search Query |
|----------|--------------|
| English | "FOMC text analysis" "machine learning" "FX" |
| English | "central bank communication" "NLP" "forex prediction" |
| English | "COT report" "machine learning" "currency" |
| English | "carry trade signal" "construction" site:ssrn.com |
| Chinese | "央行沟通" "NLP" "外汇" |
| Chinese | "利差交易" "信号构建" |
| Japanese | "日銀" "テキスト分析" "為替予測" |
| German | "EZB Kommunikation" "NLP" "Devisen" |

**Specific data sources to find:**
- Free/cheap FOMC transcript APIs
- COT report APIs
- FX sentiment data sources
- Interest rate differential data

**Expected output:** 2-3 alternative data sources with APIs we can use

---

## GAP 3: LIVE CONTINUOUS LEARNING

**What we have:** Static model, LoRA concept
**What we need:** Production protocol for continuous retraining

**Search queries by language:**

| Language | Search Query |
|----------|--------------|
| English | "online learning" "production" "trading system" |
| English | "concept drift" "detection" "finance" |
| English | "model update" "live trading" "without downtime" |
| Chinese | "在线学习" "实盘" "量化" site:bigquant.com |
| Chinese | "模型更新" "生产环境" "幻方" OR "九坤" |
| Chinese | "概念漂移" "检测" "金融" |

**Specific questions:**
- How often does 幻方量化 retrain? (幻方 多久训练一次)
- What triggers retraining at 九坤? (九坤 模型更新触发条件)
- LoRA hot-swap in production (LoRA 热更新 生产)

**Expected output:** Protocol with frequency, triggers, and implementation

---

## GAP 4: MULTI-AGENT DEBATE

**What we have:** Concept only
**What we need:** Actual implementation with prompts

**Search queries by language:**

| Language | Search Query |
|----------|--------------|
| English | "multi-agent debate" "trading" "LLM" |
| English | "TradingAgents" "UCLA" site:arxiv.org |
| English | "adversarial" "bull bear" "analysis" "LLM" |
| Chinese | "多智能体辩论" "交易" |
| Chinese | "LLM" "多空分析" "对抗" |

**Specific papers:**
- TradingAgents (UCLA/MIT) paper
- Any multi-agent trading papers from 2024-2025

**Expected output:** Working prompt templates for Bull/Bear/Risk agents

---

## GAP 5: ADVANCED ENSEMBLE

**What we have:** Simple voting
**What we need:** Meta-learner, dynamic weighting

**Search queries by language:**

| Language | Search Query |
|----------|--------------|
| English | "stacking" "meta-learner" "time series" "finance" |
| English | "dynamic ensemble" "regime" "trading" |
| English | "model confidence set" "Hansen" |
| Chinese | "堆叠" "元学习器" "时间序列" |
| Chinese | "动态集成" "市场状态" |

**Specific techniques to find:**
- Stacking for financial time series
- Regime-conditional model weighting
- Model Confidence Set (Hansen et al.)

**Expected output:** Meta-learner architecture + dynamic weighting scheme

---

## GAP 6: ORDER FLOW PREDICTION

**What we have:** VPIN/OFI as features (measurement)
**What we need:** Prediction of future order flow

**Search queries by language:**

| Language | Search Query |
|----------|--------------|
| English | "order flow prediction" "forex" "machine learning" |
| English | "Hawkes process" "FX" "order arrival" |
| English | "adverse selection" "prediction" "currency" |
| Chinese | "订单流预测" "外汇" |
| Chinese | "Hawkes过程" "高频" "九坤" |

**Expected output:** Model that predicts order flow direction

---

## GAP 7: TAIL RISK / EXTREME EVENTS

**What we have:** VaR, CVaR
**What we need:** EVT, flash crash detection

**Search queries by language:**

| Language | Search Query |
|----------|--------------|
| English | "extreme value theory" "forex" "tail risk" |
| English | "flash crash" "detection" "prediction" "FX" |
| English | "correlation breakdown" "currency" "crisis" |
| Chinese | "极值理论" "外汇" "尾部风险" |
| Chinese | "闪崩" "检测" "预警" |
| Japanese | "極値理論" "為替" "テールリスク" |

**Expected output:** EVT implementation for forex + crash detector

---

## GAP 8: EXECUTION TIMING

**What we have:** Almgren-Chriss, TWAP, VWAP
**What we need:** Session timing, spread prediction

**Search queries by language:**

| Language | Search Query |
|----------|--------------|
| English | "optimal execution" "time of day" "forex" |
| English | "spread prediction" "FX" "machine learning" |
| English | "London session" "execution" "optimal" |
| Chinese | "最优执行" "交易时段" "外汇" |
| Japanese | "最適執行" "ロンドン時間" "為替" |

**Expected output:** ML model for optimal execution timing by session

---

# PART 4: OUTPUT FORMAT (STRICT)

For EACH technique found, you MUST provide:

```
## [TECHNIQUE NAME]

### Source
- Country: [China/USA/UK/Japan/etc.]
- Language searched: [Chinese/English/Japanese/etc.]
- Citation: [Full academic citation OR fund source]
- URL: [Link to paper/blog/wiki]

### What it does
[2-3 sentences explaining the technique]

### Why we don't have it
[How it's DIFFERENT from what we already have]

### Mathematical Formula
```
[LaTeX or code formula]
```

### Implementation
```python
[Working Python code snippet]
```

### Expected Improvement
- Accuracy gain: [X%]
- Certainty gain: [Y%]
- Evidence: [Citation showing this improvement]

### Complexity
- Implementation time: [hours/days]
- Dependencies: [libraries needed]
- Compute requirements: [GPU/CPU/memory]
```

---

# PART 5: VALIDATION RULES

```
╔══════════════════════════════════════════════════════════════════════════════════╗
║  BEFORE SUGGESTING ANY TECHNIQUE, VERIFY:                                        ║
╠══════════════════════════════════════════════════════════════════════════════════╣
║  □ It's NOT in "WHAT WE HAVE" section                                            ║
║  □ It has academic citation OR credible fund source                              ║
║  □ It has mathematical formula                                                   ║
║  □ It has implementation code                                                    ║
║  □ It addresses one of the 8 GAPS specifically                                   ║
║  □ It's been used in production OR has empirical results                         ║
╚══════════════════════════════════════════════════════════════════════════════════╝
```

**REJECT techniques that:**
- ❌ Are already in our inventory
- ❌ Have no citation/source
- ❌ Are purely theoretical with no implementation
- ❌ Don't address one of the 8 gaps
- ❌ Have no evidence of effectiveness

---

# PART 6: QUICK COPY-PASTE PROMPT

Use this exact prompt with ChatGPT/Perplexity/Claude:

```
I need to improve a forex trading system from 89% to 99.999% certainty.

STRICT RULES:
1. Search in NATIVE LANGUAGE of source country (Chinese for China, Japanese for Japan, etc.)
2. Only GOLD STANDARD sources (NeurIPS, ICML, JF, RFS, top quant funds)
3. DO NOT suggest: Alpha101/158/191/360, GARCH family, VPIN, Kelly, Almgren-Chriss, HMM, iTransformer, TimeXer, GRPO, PPO - I ALREADY HAVE THESE

RESEARCH THESE 8 GAPS ONLY:

1. CERTAINTY CALIBRATION - How to make "95% confident" = 95% accurate
   - Search: "置信度校准" (Chinese), "confidence calibration trading" (English)

2. FOREX ALTERNATIVE DATA - Central bank NLP, positioning, carry signals
   - Search: "央行沟通NLP" (Chinese), "FOMC text analysis forex" (English)

3. LIVE CONTINUOUS LEARNING - Production LoRA updates, drift detection
   - Search: "在线学习实盘" (Chinese), "online learning production trading" (English)

4. MULTI-AGENT DEBATE - Bull/Bear/Risk agent implementation
   - Search: "多智能体辩论交易" (Chinese), "TradingAgents LLM" (English)

5. ADVANCED ENSEMBLE - Meta-learner stacking, dynamic weighting
   - Search: "堆叠元学习器" (Chinese), "stacking meta-learner time series" (English)

6. ORDER FLOW PREDICTION - Predict future flow, not measure past
   - Search: "订单流预测" (Chinese), "order flow prediction forex ML" (English)

7. TAIL RISK - EVT for forex, flash crash detection
   - Search: "极值理论外汇" (Chinese), "extreme value theory forex" (English)

8. EXECUTION TIMING - Session-optimal, spread prediction
   - Search: "最优执行交易时段" (Chinese), "optimal execution time forex" (English)

FOR EACH TECHNIQUE FOUND, PROVIDE:
- Full citation (paper/fund source)
- Mathematical formula
- Python implementation
- Expected accuracy improvement
- How it's different from what I have

SEARCH THESE SOURCES:
- China: 知乎, CNKI, BigQuant, 聚宽
- USA: arXiv q-fin, SSRN, NeurIPS/ICML proceedings
- UK: Oxford-Man Institute, Man AHL research
- Japan: CiNii, J-STAGE
```

---

# PART 7: SUCCESS METRICS

Research is COMPLETE when you have:

| Gap | Deliverable | Validation |
|-----|-------------|------------|
| Calibration | ECE implementation | ECE < 0.05 on test set |
| Alt Data | 2+ data source APIs | Can fetch data |
| Live Learning | Update protocol doc | Tested hot-swap |
| Multi-Agent | Prompt templates | Run debate successfully |
| Ensemble | Meta-learner code | +2% accuracy |
| Order Flow | Prediction model | +1% edge |
| Tail Risk | EVT implementation | Detects 90% of crashes |
| Execution | Timing model | Saves 0.5 pips |

**TOTAL EXPECTED IMPROVEMENT: 89% → 95-99%**
(99.999% requires multiplicative certainty from all layers)
