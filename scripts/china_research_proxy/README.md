# China Research Proxy

Access Chinese academic content, quant formulas, and research papers from behind the Great Firewall.

## Architecture

```
Your PC ──SSH Tunnel──► Alibaba Cloud HK VPS ──► Chinese Internet
                              │
                              ├── Baidu (百度)
                              ├── Zhihu (知乎)
                              ├── Baidu Scholar (百度学术)
                              ├── Baidu Wenku (百度文库)
                              ├── Gitee (码云)
                              └── CNKI (中国知网)
```

## Setup

### Step 1: Get Alibaba Cloud Free VPS

1. Go to https://www.alibabacloud.com/campaign/free-trial
2. Sign up with email (no Chinese ID needed)
3. Create ECS instance:
   - Region: **Hong Kong**
   - Instance: t5 or t6 (free tier)
   - OS: Ubuntu 22.04
   - Download SSH key (.pem file)

### Step 2: Deploy Proxy

```powershell
powershell -File deploy_to_alibaba.ps1 -IP "YOUR_VPS_IP" -KeyPath "path/to/key.pem"
```

### Step 3: Use It

```bash
# Check health
python china_research_client.py health

# List pre-built quant queries
python china_research_client.py queries

# Search for Kalman filter quant formulas
python china_research_client.py search kalman

# Search all sources for custom query
python china_research_client.py search "高频交易算法"

# Search specific source
python china_research_client.py baidu "量化交易数学公式"
python china_research_client.py zhihu "因子模型"
python china_research_client.py scholar "Hawkes过程"

# Fetch specific URL
python china_research_client.py fetch "https://zhuanlan.zhihu.com/p/618456485"
```

## Pre-Built Quant Queries

| Key | Search Query |
|-----|--------------|
| `kalman` | 卡尔曼滤波 量化交易 状态空间模型 |
| `hawkes` | Hawkes过程 自激励点过程 订单流 |
| `hmm` | 隐马尔可夫模型 regime detection |
| `ou` | Ornstein-Uhlenbeck 均值回归 |
| `garch` | GARCH模型 波动率预测 |
| `factor` | 因子模型 多因子选股 alpha |
| `vwap` | VWAP算法 成交量加权 最优执行 |
| `ml_quant` | 机器学习 量化交易 特征工程 |
| `orderbook` | 限价订单簿 市场微观结构 |
| `hft` | 高频交易 算法交易 低延迟 |
| `market_making` | 做市策略 Avellaneda-Stoikov |

## API Endpoints

| Endpoint | Description |
|----------|-------------|
| `GET /health` | Health check |
| `GET /search/baidu?q=` | Search Baidu |
| `GET /search/zhihu?q=` | Search Zhihu |
| `GET /search/xueshu?q=` | Search Baidu Scholar |
| `GET /search/wenku?q=` | Search Baidu Wenku |
| `GET /search/gitee?q=` | Search Gitee |
| `POST /fetch` | Fetch any Chinese URL |
| `GET /quant/formulas` | Pre-built quant searches |

## SOCKS5 Proxy

For browser access to Chinese sites:

```bash
# Start SSH tunnel
ssh -i "key.pem" -D 1080 -N root@YOUR_VPS_IP

# Configure browser to use SOCKS5 proxy: localhost:1080
```

## Cost

- Alibaba Cloud HK: **FREE for 12 months**
- After free tier: ~$5-10/month

## Files

- `setup_china_proxy.sh` - Server setup script
- `china_research_client.py` - Python client
- `deploy_to_alibaba.ps1` - Deployment script
