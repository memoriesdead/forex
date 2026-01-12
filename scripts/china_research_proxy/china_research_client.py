#!/usr/bin/env python3
"""
China Research Client
Connect to Alibaba Cloud HK VPS to search Chinese academic content

Usage:
    python china_research_client.py search "量化交易公式"
    python china_research_client.py formulas kalman
    python china_research_client.py fetch "https://zhuanlan.zhihu.com/p/xxxxx"
"""

import requests
import json
import argparse
from pathlib import Path

# Configuration - UPDATE AFTER VPS SETUP
CHINA_PROXY_URL = "http://YOUR_VPS_IP:8888"
SOCKS5_PROXY = "socks5://YOUR_VPS_IP:1080"


class ChinaResearchClient:
    def __init__(self, proxy_url: str = None):
        self.proxy_url = proxy_url or CHINA_PROXY_URL
        self.session = requests.Session()

    def health_check(self) -> dict:
        """Check if proxy is running"""
        try:
            resp = self.session.get(f"{self.proxy_url}/health", timeout=5)
            return resp.json()
        except Exception as e:
            return {"error": str(e), "status": "offline"}

    def search_baidu(self, query: str, limit: int = 20) -> dict:
        """Search Baidu for Chinese content"""
        resp = self.session.get(
            f"{self.proxy_url}/search/baidu",
            params={"q": query, "limit": limit},
            timeout=30
        )
        return resp.json()

    def search_zhihu(self, query: str, limit: int = 20) -> dict:
        """Search Zhihu for quant discussions"""
        resp = self.session.get(
            f"{self.proxy_url}/search/zhihu",
            params={"q": query, "limit": limit},
            timeout=30
        )
        return resp.json()

    def search_scholar(self, query: str, limit: int = 20) -> dict:
        """Search Baidu Scholar for academic papers"""
        resp = self.session.get(
            f"{self.proxy_url}/search/xueshu",
            params={"q": query, "limit": limit},
            timeout=30
        )
        return resp.json()

    def search_wenku(self, query: str, limit: int = 20) -> dict:
        """Search Baidu Wenku for documents"""
        resp = self.session.get(
            f"{self.proxy_url}/search/wenku",
            params={"q": query, "limit": limit},
            timeout=30
        )
        return resp.json()

    def search_gitee(self, query: str, limit: int = 20) -> dict:
        """Search Gitee for Chinese code repos"""
        resp = self.session.get(
            f"{self.proxy_url}/search/gitee",
            params={"q": query, "limit": limit},
            timeout=30
        )
        return resp.json()

    def fetch_url(self, url: str) -> dict:
        """Fetch any Chinese URL"""
        resp = self.session.post(
            f"{self.proxy_url}/fetch",
            json={"url": url},
            timeout=60
        )
        return resp.json()

    def get_quant_formulas(self, category: str = "all") -> dict:
        """Get pre-built quant formula searches"""
        resp = self.session.get(
            f"{self.proxy_url}/quant/formulas",
            params={"category": category},
            timeout=30
        )
        return resp.json()

    def search_all(self, query: str) -> dict:
        """Search all Chinese sources at once"""
        results = {
            "query": query,
            "baidu": self.search_baidu(query, 10),
            "zhihu": self.search_zhihu(query, 10),
            "scholar": self.search_scholar(query, 10),
            "wenku": self.search_wenku(query, 5),
            "gitee": self.search_gitee(query, 5)
        }
        return results


# Pre-defined quant research queries
QUANT_QUERIES = {
    # Mathematical Models
    "kalman": "卡尔曼滤波 量化交易 状态空间模型 公式推导",
    "hawkes": "Hawkes过程 自激励点过程 订单流 高频交易",
    "hmm": "隐马尔可夫模型 regime detection 量化 状态转移",
    "ou": "Ornstein-Uhlenbeck 均值回归 随机微分方程",
    "garch": "GARCH模型 波动率预测 条件异方差",
    "jump": "跳跃扩散模型 Merton 期权定价",

    # Factor Models
    "factor": "因子模型 多因子选股 alpha因子 IC IR",
    "barra": "Barra风险模型 因子暴露 协方差矩阵",
    "fama": "Fama-French 三因子 五因子 因子溢价",

    # Execution Algorithms
    "vwap": "VWAP算法 成交量加权 最优执行",
    "twap": "TWAP算法 时间加权 执行策略",
    "is": "Implementation Shortfall 执行差额 最优化",
    "almgren": "Almgren-Chriss 最优执行 市场冲击",

    # Machine Learning
    "ml_quant": "机器学习 量化交易 特征工程 模型选择",
    "deep_quant": "深度学习 LSTM 时序预测 量化",
    "rl_trading": "强化学习 交易策略 DQN PPO",
    "transformer": "Transformer 时间序列 股票预测",

    # Market Microstructure
    "orderbook": "限价订单簿 市场微观结构 做市商",
    "spread": "买卖价差 流动性 逆向选择",
    "kyle": "Kyle模型 信息交易 价格发现",
    "glosten": "Glosten-Milgrom 知情交易者 价差模型",

    # Risk Management
    "var": "VaR风险价值 CVaR 尾部风险",
    "kelly": "Kelly准则 最优仓位 资金管理",
    "sharpe": "夏普比率 风险调整收益 绩效评估",

    # High Frequency
    "hft": "高频交易 算法交易 低延迟",
    "market_making": "做市策略 库存管理 Avellaneda-Stoikov",
    "stat_arb": "统计套利 配对交易 协整",

    # Chinese Specific
    "a_share": "A股量化 中国市场 特色因子",
    "t0": "T+0策略 日内交易 中国股市",
    "limit": "涨跌停 中国市场 流动性"
}


def print_results(results: dict, format_type: str = "summary"):
    """Pretty print search results"""
    if "error" in results:
        print(f"Error: {results['error']}")
        return

    if format_type == "json":
        print(json.dumps(results, ensure_ascii=False, indent=2))
        return

    # Summary format
    if "results" in results:
        print(f"\nQuery: {results.get('query', 'N/A')}")
        print(f"Found: {results.get('count', len(results['results']))} results\n")
        print("-" * 60)

        for i, item in enumerate(results["results"], 1):
            print(f"{i}. {item.get('title', 'No title')}")
            if item.get("url"):
                print(f"   URL: {item['url'][:80]}...")
            if item.get("abstract"):
                print(f"   {item['abstract'][:150]}...")
            if item.get("authors"):
                print(f"   Authors: {item['authors']}")
            print()
    elif "categories" in results:
        print("\nAvailable formula categories:")
        for cat in results["categories"]:
            print(f"  - {cat}")


def main():
    parser = argparse.ArgumentParser(description="China Research Client")
    parser.add_argument("command", choices=[
        "search", "baidu", "zhihu", "scholar", "wenku", "gitee",
        "formulas", "fetch", "health", "queries"
    ])
    parser.add_argument("query", nargs="?", default="")
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--proxy", default=None, help="Override proxy URL")

    args = parser.parse_args()

    client = ChinaResearchClient(args.proxy)

    if args.command == "health":
        result = client.health_check()
        print(json.dumps(result, indent=2))

    elif args.command == "queries":
        print("\nPre-defined quant research queries:")
        print("-" * 60)
        for key, query in QUANT_QUERIES.items():
            print(f"  {key:15} → {query}")
        print("\nUsage: python china_research_client.py search kalman")

    elif args.command == "search":
        # Check if query is a preset
        query = QUANT_QUERIES.get(args.query, args.query)
        result = client.search_all(query)

        if args.json:
            print(json.dumps(result, ensure_ascii=False, indent=2))
        else:
            for source, data in result.items():
                if source == "query":
                    continue
                print(f"\n{'='*60}")
                print(f"SOURCE: {source.upper()}")
                print(f"{'='*60}")
                print_results(data)

    elif args.command == "baidu":
        query = QUANT_QUERIES.get(args.query, args.query)
        result = client.search_baidu(query, args.limit)
        print_results(result, "json" if args.json else "summary")

    elif args.command == "zhihu":
        query = QUANT_QUERIES.get(args.query, args.query)
        result = client.search_zhihu(query, args.limit)
        print_results(result, "json" if args.json else "summary")

    elif args.command == "scholar":
        query = QUANT_QUERIES.get(args.query, args.query)
        result = client.search_scholar(query, args.limit)
        print_results(result, "json" if args.json else "summary")

    elif args.command == "wenku":
        query = QUANT_QUERIES.get(args.query, args.query)
        result = client.search_wenku(query, args.limit)
        print_results(result, "json" if args.json else "summary")

    elif args.command == "gitee":
        query = QUANT_QUERIES.get(args.query, args.query)
        result = client.search_gitee(query, args.limit)
        print_results(result, "json" if args.json else "summary")

    elif args.command == "formulas":
        result = client.get_quant_formulas(args.query or "all")
        print_results(result, "json" if args.json else "summary")

    elif args.command == "fetch":
        result = client.fetch_url(args.query)
        if args.json:
            print(json.dumps(result, ensure_ascii=False, indent=2))
        else:
            print(f"Title: {result.get('title', 'N/A')}")
            print(f"URL: {result.get('url', 'N/A')}")
            print("-" * 60)
            print(result.get("text", "")[:5000])


if __name__ == "__main__":
    main()
