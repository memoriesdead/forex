#!/bin/bash
# China Research Proxy Setup for Alibaba Cloud HK VPS
# Provides full access to Chinese academic/research content

set -e

echo "=========================================="
echo "CHINA RESEARCH PROXY SETUP"
echo "=========================================="

# Update system
apt-get update && apt-get upgrade -y

# Install dependencies
apt-get install -y \
    python3 python3-pip python3-venv \
    nodejs npm \
    nginx \
    dante-server \
    git curl wget \
    chromium-browser \
    fonts-wqy-zenhei fonts-wqy-microhei

# Create project directory
mkdir -p /opt/china-research-proxy
cd /opt/china-research-proxy

# Create Python virtual environment
python3 -m venv venv
source venv/bin/activate

# Install Python packages
pip install --upgrade pip
pip install \
    flask flask-cors \
    requests beautifulsoup4 \
    selenium playwright \
    akshare \
    fake-useragent \
    googletrans==4.0.0-rc1

# Install Playwright browsers
playwright install chromium

# Create the search API server
cat > /opt/china-research-proxy/server.py << 'PYEOF'
#!/usr/bin/env python3
"""
China Research Proxy Server
Search Baidu, Zhihu, Wenku, CNKI from outside China
"""

from flask import Flask, jsonify, request
from flask_cors import CORS
import requests
from bs4 import BeautifulSoup
from fake_useragent import UserAgent
import json
import re
from urllib.parse import quote
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

ua = UserAgent()

HEADERS = {
    'User-Agent': ua.chrome,
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
    'Accept-Language': 'zh-CN,zh;q=0.9,en;q=0.8',
    'Accept-Encoding': 'gzip, deflate',
    'Connection': 'keep-alive',
}

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok', 'location': 'Hong Kong', 'access': 'China Internet'})

@app.route('/search/baidu', methods=['GET'])
def search_baidu():
    """Search Baidu for Chinese content"""
    query = request.args.get('q', '')
    limit = int(request.args.get('limit', 10))

    try:
        url = f"https://www.baidu.com/s?wd={quote(query)}&rn={limit}"
        response = requests.get(url, headers=HEADERS, timeout=10)
        soup = BeautifulSoup(response.text, 'html.parser')

        results = []
        for item in soup.select('.result.c-container'):
            title_elem = item.select_one('h3 a')
            abstract_elem = item.select_one('.c-abstract') or item.select_one('.content-right_2s-H4')

            if title_elem:
                results.append({
                    'title': title_elem.get_text(strip=True),
                    'url': title_elem.get('href', ''),
                    'abstract': abstract_elem.get_text(strip=True) if abstract_elem else ''
                })

        return jsonify({'query': query, 'results': results[:limit], 'count': len(results)})

    except Exception as e:
        logger.error(f"Baidu search error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/search/zhihu', methods=['GET'])
def search_zhihu():
    """Search Zhihu for quant discussions"""
    query = request.args.get('q', '')
    limit = int(request.args.get('limit', 10))

    try:
        url = f"https://www.zhihu.com/search?type=content&q={quote(query)}"
        response = requests.get(url, headers=HEADERS, timeout=10)

        # Zhihu returns JSON in script tag
        soup = BeautifulSoup(response.text, 'html.parser')

        results = []
        # Parse search results
        for script in soup.find_all('script'):
            if script.string and 'initialData' in str(script.string):
                # Extract JSON data
                match = re.search(r'initialData.*?=\s*({.*?});', script.string, re.DOTALL)
                if match:
                    try:
                        data = json.loads(match.group(1))
                        # Extract search results from data structure
                        # This is simplified - actual structure may vary
                    except:
                        pass

        # Fallback: parse HTML directly
        for item in soup.select('.SearchResult-Card'):
            title = item.select_one('.ContentItem-title')
            excerpt = item.select_one('.RichText')
            if title:
                results.append({
                    'title': title.get_text(strip=True),
                    'excerpt': excerpt.get_text(strip=True)[:200] if excerpt else ''
                })

        return jsonify({'query': query, 'results': results[:limit], 'count': len(results)})

    except Exception as e:
        logger.error(f"Zhihu search error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/search/xueshu', methods=['GET'])
def search_xueshu():
    """Search Baidu Scholar for academic papers"""
    query = request.args.get('q', '')
    limit = int(request.args.get('limit', 10))

    try:
        url = f"https://xueshu.baidu.com/s?wd={quote(query)}&rsv_bp=0&tn=SE_baiduxueshu_c1gjeupa"
        response = requests.get(url, headers=HEADERS, timeout=10)
        soup = BeautifulSoup(response.text, 'html.parser')

        results = []
        for item in soup.select('.result'):
            title_elem = item.select_one('.t a')
            authors_elem = item.select_one('.author_text')
            abstract_elem = item.select_one('.c_abstract')

            if title_elem:
                results.append({
                    'title': title_elem.get_text(strip=True),
                    'url': title_elem.get('href', ''),
                    'authors': authors_elem.get_text(strip=True) if authors_elem else '',
                    'abstract': abstract_elem.get_text(strip=True) if abstract_elem else ''
                })

        return jsonify({'query': query, 'results': results[:limit], 'count': len(results)})

    except Exception as e:
        logger.error(f"Baidu Scholar search error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/search/wenku', methods=['GET'])
def search_wenku():
    """Search Baidu Wenku for documents"""
    query = request.args.get('q', '')
    limit = int(request.args.get('limit', 10))

    try:
        url = f"https://wenku.baidu.com/search?word={quote(query)}&lm=0&od=0"
        response = requests.get(url, headers=HEADERS, timeout=10)
        soup = BeautifulSoup(response.text, 'html.parser')

        results = []
        for item in soup.select('.search-result-item'):
            title_elem = item.select_one('.title a')
            desc_elem = item.select_one('.desc')

            if title_elem:
                results.append({
                    'title': title_elem.get_text(strip=True),
                    'url': 'https://wenku.baidu.com' + title_elem.get('href', ''),
                    'description': desc_elem.get_text(strip=True) if desc_elem else ''
                })

        return jsonify({'query': query, 'results': results[:limit], 'count': len(results)})

    except Exception as e:
        logger.error(f"Wenku search error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/search/gitee', methods=['GET'])
def search_gitee():
    """Search Gitee for Chinese code repos"""
    query = request.args.get('q', '')
    limit = int(request.args.get('limit', 10))

    try:
        url = f"https://search.gitee.com/?type=repository&q={quote(query)}"
        response = requests.get(url, headers=HEADERS, timeout=10)
        soup = BeautifulSoup(response.text, 'html.parser')

        results = []
        for item in soup.select('.item'):
            title_elem = item.select_one('.title a')
            desc_elem = item.select_one('.desc')
            stars_elem = item.select_one('.stars-count')

            if title_elem:
                results.append({
                    'title': title_elem.get_text(strip=True),
                    'url': 'https://gitee.com' + title_elem.get('href', ''),
                    'description': desc_elem.get_text(strip=True) if desc_elem else '',
                    'stars': stars_elem.get_text(strip=True) if stars_elem else '0'
                })

        return jsonify({'query': query, 'results': results[:limit], 'count': len(results)})

    except Exception as e:
        logger.error(f"Gitee search error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/fetch', methods=['POST'])
def fetch_url():
    """Fetch any Chinese URL and return content"""
    data = request.json
    url = data.get('url', '')

    try:
        response = requests.get(url, headers=HEADERS, timeout=30)
        soup = BeautifulSoup(response.text, 'html.parser')

        # Remove scripts and styles
        for tag in soup(['script', 'style', 'nav', 'footer']):
            tag.decompose()

        text = soup.get_text(separator='\n', strip=True)

        return jsonify({
            'url': url,
            'title': soup.title.string if soup.title else '',
            'text': text[:50000],  # Limit to 50k chars
            'html': str(soup)[:100000]
        })

    except Exception as e:
        logger.error(f"Fetch error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/quant/formulas', methods=['GET'])
def quant_formulas():
    """Pre-built searches for quant formulas"""
    category = request.args.get('category', 'all')

    searches = {
        'kalman': '卡尔曼滤波 量化交易 公式',
        'hawkes': 'Hawkes过程 订单簿 高频',
        'hmm': '隐马尔可夫模型 量化 regime',
        'mean_reversion': '均值回归 Ornstein-Uhlenbeck 公式',
        'factor': '因子模型 alpha 公式 推导',
        'execution': 'VWAP TWAP 算法 最优执行',
        'cointegration': '协整 配对交易 ADF检验',
        'kelly': 'Kelly准则 仓位管理 公式',
        'microstructure': '市场微观结构 做市商 公式',
        'ml': '机器学习 量化 特征工程'
    }

    if category == 'all':
        return jsonify({'categories': list(searches.keys()), 'searches': searches})
    elif category in searches:
        # Actually perform the search
        query = searches[category]
        # Redirect to baidu search
        return search_baidu_internal(query, 20)
    else:
        return jsonify({'error': 'Unknown category', 'available': list(searches.keys())}), 400

def search_baidu_internal(query, limit):
    """Internal baidu search"""
    try:
        url = f"https://www.baidu.com/s?wd={quote(query)}&rn={limit}"
        response = requests.get(url, headers=HEADERS, timeout=10)
        soup = BeautifulSoup(response.text, 'html.parser')

        results = []
        for item in soup.select('.result.c-container'):
            title_elem = item.select_one('h3 a')
            abstract_elem = item.select_one('.c-abstract')

            if title_elem:
                results.append({
                    'title': title_elem.get_text(strip=True),
                    'url': title_elem.get('href', ''),
                    'abstract': abstract_elem.get_text(strip=True) if abstract_elem else ''
                })

        return jsonify({'query': query, 'results': results, 'count': len(results)})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8888, debug=False)
PYEOF

# Create systemd service
cat > /etc/systemd/system/china-research-proxy.service << 'SVCEOF'
[Unit]
Description=China Research Proxy Server
After=network.target

[Service]
Type=simple
User=root
WorkingDirectory=/opt/china-research-proxy
ExecStart=/opt/china-research-proxy/venv/bin/python /opt/china-research-proxy/server.py
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
SVCEOF

# Configure SOCKS5 proxy (dante-server)
cat > /etc/danted.conf << 'DANTEEOF'
logoutput: syslog
internal: 0.0.0.0 port = 1080
external: eth0

socksmethod: none
clientmethod: none

client pass {
    from: 0.0.0.0/0 to: 0.0.0.0/0
    log: connect disconnect error
}

socks pass {
    from: 0.0.0.0/0 to: 0.0.0.0/0
    log: connect disconnect error
}
DANTEEOF

# Enable and start services
systemctl daemon-reload
systemctl enable china-research-proxy
systemctl start china-research-proxy
systemctl enable danted
systemctl start danted

# Open firewall ports
ufw allow 22
ufw allow 1080
ufw allow 8888
ufw --force enable

echo ""
echo "=========================================="
echo "SETUP COMPLETE"
echo "=========================================="
echo ""
echo "Services running:"
echo "  - Research API: http://YOUR_IP:8888"
echo "  - SOCKS5 Proxy: YOUR_IP:1080"
echo ""
echo "Test endpoints:"
echo "  curl http://localhost:8888/health"
echo "  curl 'http://localhost:8888/search/baidu?q=量化交易'"
echo "  curl 'http://localhost:8888/quant/formulas?category=all'"
echo ""
