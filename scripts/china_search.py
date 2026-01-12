#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
China Search Engine - Access Baidu, Zhihu, Gitee
Uses OpenSERP Docker container (already running on localhost:7000)

Usage:
    python scripts/china_search.py "量化交易公式"
    python scripts/china_search.py kalman
    python scripts/china_search.py --list
"""

import requests
import json
import sys
import io
from urllib.parse import quote

# Fix Windows console encoding
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

OPENSERP_URL = "http://localhost:7000"

# Pre-built quant research queries
QUANT_QUERIES = {
    # Mathematical Models
    "kalman": "卡尔曼滤波 量化交易 公式推导",
    "hawkes": "Hawkes过程 订单簿 高频交易 公式",
    "hmm": "隐马尔可夫模型 量化 regime detection",
    "ou": "Ornstein-Uhlenbeck 均值回归 公式",
    "garch": "GARCH模型 波动率预测 公式",

    # Factor Models
    "factor": "因子模型 多因子选股 alpha因子 IC",
    "barra": "Barra风险模型 因子暴露",
    "fama": "Fama-French 三因子 五因子",

    # Execution
    "vwap": "VWAP算法 成交量加权 最优执行 公式",
    "twap": "TWAP算法 时间加权",
    "almgren": "Almgren-Chriss 最优执行 市场冲击",

    # ML
    "ml": "机器学习 量化交易 特征工程",
    "deep": "深度学习 LSTM 股票预测",
    "rl": "强化学习 交易策略 DQN PPO",
    "transformer": "Transformer 时间序列 金融预测",

    # Microstructure
    "orderbook": "限价订单簿 市场微观结构 做市",
    "spread": "买卖价差 流动性 逆向选择",
    "kyle": "Kyle模型 信息交易",

    # Risk
    "var": "VaR风险价值 CVaR 尾部风险",
    "kelly": "Kelly准则 最优仓位 资金管理",
    "sharpe": "夏普比率 风险调整收益",

    # HFT
    "hft": "高频交易 算法交易 低延迟",
    "mm": "做市策略 Avellaneda-Stoikov 库存",
    "arb": "统计套利 配对交易 协整",

    # Formulas
    "formulas": "量化交易指标公式大全",
    "indicators": "技术指标 公式 源码",
    "backtest": "回测 策略 公式",
}


def search_baidu(query: str, limit: int = 10) -> list:
    """Search Baidu via OpenSERP"""
    url = f"{OPENSERP_URL}/baidu/search"
    params = {"text": query, "limit": limit}

    try:
        resp = requests.get(url, params=params, timeout=30)
        return resp.json()
    except Exception as e:
        return [{"error": str(e)}]


def search_all(query: str, limit: int = 10) -> dict:
    """Search multiple engines"""
    results = {
        "query": query,
        "baidu": search_baidu(query, limit),
    }
    return results


def print_results(results: list, source: str = "baidu"):
    """Pretty print results"""
    print(f"\n{'='*70}")
    print(f"  {source.upper()} SEARCH RESULTS")
    print(f"{'='*70}\n")

    for i, item in enumerate(results, 1):
        if "error" in item:
            print(f"Error: {item['error']}")
            continue

        title = item.get("title", "No title")
        url = item.get("url", "")
        desc = item.get("description", "")

        print(f"{i}. {title}")
        if desc:
            # Clean and truncate description
            desc_clean = desc.replace('\n', ' ').strip()[:200]
            print(f"   {desc_clean}...")
        print(f"   URL: {url[:80]}...")
        print()


def main():
    if len(sys.argv) < 2:
        print("Usage: python china_search.py <query>")
        print("       python china_search.py --list")
        print("\nExamples:")
        print("  python china_search.py kalman")
        print("  python china_search.py '高频交易算法'")
        sys.exit(1)

    query = sys.argv[1]

    if query == "--list":
        print("\nAvailable preset queries:")
        print("-" * 50)
        for key, value in QUANT_QUERIES.items():
            print(f"  {key:15} → {value}")
        print("\nUsage: python china_search.py <preset>")
        sys.exit(0)

    # Check if it's a preset
    if query in QUANT_QUERIES:
        actual_query = QUANT_QUERIES[query]
        print(f"Using preset '{query}': {actual_query}")
    else:
        actual_query = query

    print(f"\nSearching Baidu for: {actual_query}")

    results = search_baidu(actual_query, limit=10)
    print_results(results)


if __name__ == "__main__":
    main()
