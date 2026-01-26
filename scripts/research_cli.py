#!/usr/bin/env python3
"""
Research CLI - Local Claude Code Alternative for Online Research

Uses GLM-4.7 (or other Ollama models) with web search for research tasks.
Designed to save tokens on expensive API plans by running research locally.

Usage:
    python scripts/research_cli.py "What are the best LLMs for coding in 2026?"
    python scripts/research_cli.py --plan "Research forex ML techniques"
    python scripts/research_cli.py --interactive
"""

import argparse
import json
import os
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

# Check for required packages
try:
    import requests
except ImportError:
    print("Installing requests...")
    os.system(f"{sys.executable} -m pip install requests -q")
    import requests

try:
    from duckduckgo_search import DDGS
except ImportError:
    print("Installing duckduckgo-search...")
    os.system(f"{sys.executable} -m pip install duckduckgo-search -q")
    from duckduckgo_search import DDGS


# Configuration
DEFAULT_MODEL = "deepseek-r1:8b"  # Reasoning model - already installed, great for research
OLLAMA_URL = "http://localhost:11434"
MAX_SEARCH_RESULTS = 8
OUTPUT_DIR = Path("research_outputs")


class ResearchCLI:
    """Local research assistant using Ollama + web search."""

    def __init__(self, model: str = DEFAULT_MODEL, verbose: bool = False):
        self.model = model
        self.verbose = verbose
        self.search_client = DDGS()
        self.conversation_history = []

    def log(self, msg: str):
        """Print if verbose mode."""
        if self.verbose:
            print(f"[DEBUG] {msg}")

    def check_ollama(self) -> bool:
        """Verify Ollama is running and model is available."""
        try:
            resp = requests.get(f"{OLLAMA_URL}/api/tags", timeout=5)
            if resp.status_code == 200:
                models = [m['name'].split(':')[0] for m in resp.json().get('models', [])]
                if self.model.split(':')[0] not in models:
                    print(f"Model {self.model} not found. Available: {models}")
                    print(f"Run: ollama pull {self.model}")
                    return False
                return True
        except requests.exceptions.ConnectionError:
            print("Ollama not running. Start with: ollama serve")
            return False
        return False

    def web_search(self, query: str, max_results: int = MAX_SEARCH_RESULTS) -> list[dict]:
        """Search the web using DuckDuckGo."""
        self.log(f"Searching: {query}")
        try:
            results = list(self.search_client.text(query, max_results=max_results))
            return results
        except Exception as e:
            print(f"Search error: {e}")
            return []

    def fetch_page(self, url: str) -> str:
        """Fetch and extract text from a webpage."""
        try:
            headers = {'User-Agent': 'Mozilla/5.0 (Research Bot)'}
            resp = requests.get(url, headers=headers, timeout=10)

            # Simple HTML to text conversion
            text = resp.text
            # Remove scripts and styles
            text = re.sub(r'<script[^>]*>.*?</script>', '', text, flags=re.DOTALL)
            text = re.sub(r'<style[^>]*>.*?</style>', '', text, flags=re.DOTALL)
            # Remove HTML tags
            text = re.sub(r'<[^>]+>', ' ', text)
            # Clean whitespace
            text = re.sub(r'\s+', ' ', text).strip()

            return text[:8000]  # Limit length
        except Exception as e:
            self.log(f"Fetch error for {url}: {e}")
            return ""

    def query_llm(self, prompt: str, system: str = None) -> str:
        """Query the local LLM via Ollama."""
        messages = []

        if system:
            messages.append({"role": "system", "content": system})

        # Add conversation history
        messages.extend(self.conversation_history)
        messages.append({"role": "user", "content": prompt})

        try:
            resp = requests.post(
                f"{OLLAMA_URL}/api/chat",
                json={
                    "model": self.model,
                    "messages": messages,
                    "stream": True,
                    "options": {
                        "temperature": 0.7,
                        "num_predict": 4096,
                    }
                },
                stream=True,
                timeout=300
            )

            full_response = ""
            for line in resp.iter_lines():
                if line:
                    data = json.loads(line)
                    if 'message' in data:
                        chunk = data['message'].get('content', '')
                        full_response += chunk
                        print(chunk, end='', flush=True)
            print()  # Newline after response

            # Update history
            self.conversation_history.append({"role": "user", "content": prompt})
            self.conversation_history.append({"role": "assistant", "content": full_response})

            return full_response

        except Exception as e:
            print(f"LLM error: {e}")
            return ""

    def research(self, query: str, depth: str = "standard") -> str:
        """
        Conduct research on a topic.

        depth options:
        - quick: 1 search, summarize results
        - standard: 2-3 searches, fetch top pages
        - deep: 5+ searches, comprehensive analysis
        """
        print(f"\n{'='*60}")
        print(f"RESEARCHING: {query}")
        print(f"Depth: {depth} | Model: {self.model}")
        print(f"{'='*60}\n")

        # Phase 1: Initial search
        print("[1/4] Searching the web...")
        results = self.web_search(query)

        if not results:
            return "No search results found."

        # Format search results
        search_summary = "\n".join([
            f"- [{r['title']}]({r['href']})\n  {r['body'][:200]}..."
            for r in results[:5]
        ])

        print(f"\nFound {len(results)} results.\n")

        # Phase 2: Fetch top pages (if not quick)
        page_contents = []
        if depth != "quick":
            print("[2/4] Fetching top pages...")
            for i, r in enumerate(results[:3]):
                self.log(f"Fetching {r['href']}")
                content = self.fetch_page(r['href'])
                if content:
                    page_contents.append({
                        'title': r['title'],
                        'url': r['href'],
                        'content': content[:3000]
                    })
                print(f"  - Fetched: {r['title'][:50]}...")

        # Phase 3: Generate sub-queries (if deep)
        if depth == "deep":
            print("\n[3/4] Generating follow-up searches...")
            sub_query_prompt = f"""Based on the initial search for "{query}", generate 3 specific follow-up search queries that would provide deeper understanding. Return only the queries, one per line."""

            sub_queries = self.query_llm(sub_query_prompt)

            for sq in sub_queries.strip().split('\n')[:3]:
                sq = sq.strip().lstrip('0123456789.-) ')
                if sq:
                    print(f"  Searching: {sq}")
                    extra_results = self.web_search(sq, max_results=3)
                    results.extend(extra_results)

        # Phase 4: Synthesize findings
        print("\n[4/4] Synthesizing findings...\n")
        print("-" * 40)

        synthesis_prompt = f"""You are a research assistant. Based on the following search results and page contents, provide a comprehensive answer to: "{query}"

## Search Results:
{search_summary}

## Page Contents:
{chr(10).join([f"### {p['title']}{chr(10)}{p['content'][:2000]}" for p in page_contents[:3]])}

## Instructions:
1. Synthesize the key findings
2. Cite sources with [Source Name](URL)
3. Highlight any conflicting information
4. Provide actionable conclusions
5. List any gaps in the research

Be thorough but concise. Focus on facts from the sources."""

        system_prompt = """You are a research analyst. Your job is to synthesize information from web searches and provide accurate, well-sourced answers. Always cite your sources."""

        response = self.query_llm(synthesis_prompt, system=system_prompt)

        return response

    def plan_mode(self, topic: str) -> str:
        """
        Plan mode - create a research plan before executing.
        Similar to Claude Code's plan mode.
        """
        print(f"\n{'='*60}")
        print(f"PLAN MODE: {topic}")
        print(f"{'='*60}\n")

        # Generate research plan
        plan_prompt = f"""Create a research plan for: "{topic}"

Structure your plan as:
1. **Objectives**: What we need to learn
2. **Key Questions**: 5-7 specific questions to answer
3. **Search Strategy**: What to search for
4. **Expected Sources**: Types of sources needed
5. **Success Criteria**: How we'll know research is complete

Be specific and actionable."""

        print("Generating research plan...\n")
        plan = self.query_llm(plan_prompt)

        print("\n" + "="*40)
        user_input = input("\nExecute this plan? (y/n/modify): ").strip().lower()

        if user_input == 'y':
            # Extract questions from plan and research each
            print("\nExecuting research plan...\n")

            # Extract key questions (simple regex)
            questions = re.findall(r'\d+\.\s*\*?\*?([^?\n]+\?)', plan)
            if not questions:
                questions = [topic]

            all_findings = []
            for i, q in enumerate(questions[:5], 1):
                print(f"\n--- Researching Question {i}/{len(questions[:5])} ---")
                finding = self.research(q.strip(), depth="quick")
                all_findings.append(f"## Question {i}: {q}\n{finding}")

            # Final synthesis
            print("\n--- Final Synthesis ---\n")
            final_prompt = f"""Synthesize all findings into a comprehensive research report on: "{topic}"

## Individual Findings:
{chr(10).join(all_findings)}

Create a final report with:
1. Executive Summary (2-3 sentences)
2. Key Findings (bullet points)
3. Detailed Analysis
4. Recommendations
5. Sources"""

            return self.query_llm(final_prompt)

        elif user_input == 'modify':
            modification = input("What should be modified? ")
            return self.plan_mode(f"{topic} - {modification}")

        return plan

    def interactive(self):
        """Interactive research session."""
        print(f"""
{'='*60}
RESEARCH CLI - Interactive Mode
Model: {self.model}
Commands:
  /search <query>  - Quick web search
  /research <topic> - Full research
  /plan <topic>    - Plan mode research
  /deep <topic>    - Deep research
  /clear           - Clear history
  /save            - Save session
  /quit            - Exit
{'='*60}
""")

        while True:
            try:
                user_input = input("\nYou: ").strip()

                if not user_input:
                    continue

                if user_input.startswith('/'):
                    cmd_parts = user_input.split(' ', 1)
                    cmd = cmd_parts[0].lower()
                    arg = cmd_parts[1] if len(cmd_parts) > 1 else ""

                    if cmd == '/quit':
                        print("Goodbye!")
                        break
                    elif cmd == '/clear':
                        self.conversation_history = []
                        print("History cleared.")
                    elif cmd == '/save':
                        self.save_session()
                    elif cmd == '/search':
                        results = self.web_search(arg)
                        for r in results[:5]:
                            print(f"\n- {r['title']}\n  {r['href']}\n  {r['body'][:150]}...")
                    elif cmd == '/research':
                        self.research(arg, depth="standard")
                    elif cmd == '/plan':
                        self.plan_mode(arg)
                    elif cmd == '/deep':
                        self.research(arg, depth="deep")
                    else:
                        print(f"Unknown command: {cmd}")
                else:
                    # Regular chat
                    self.query_llm(user_input)

            except KeyboardInterrupt:
                print("\n\nInterrupted. Type /quit to exit.")
            except Exception as e:
                print(f"Error: {e}")

    def save_session(self, filename: str = None):
        """Save research session to file."""
        OUTPUT_DIR.mkdir(exist_ok=True)

        if not filename:
            filename = f"research_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"

        filepath = OUTPUT_DIR / filename

        content = f"""# Research Session
Date: {datetime.now().isoformat()}
Model: {self.model}

## Conversation
"""
        for msg in self.conversation_history:
            role = "**User**" if msg['role'] == 'user' else "**Assistant**"
            content += f"\n{role}:\n{msg['content']}\n"

        filepath.write_text(content, encoding='utf-8')
        print(f"Session saved to: {filepath}")


def main():
    parser = argparse.ArgumentParser(
        description="Research CLI - Local Claude Code alternative for online research",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python research_cli.py "Best LLMs for coding 2026"
  python research_cli.py --plan "Forex ML techniques"
  python research_cli.py --deep "Kimi K2 vs GLM-4.7"
  python research_cli.py --interactive
  python research_cli.py --model deepseek-r1:8b "Your query"
"""
    )

    parser.add_argument('query', nargs='?', help='Research query')
    parser.add_argument('--plan', '-p', metavar='TOPIC', help='Plan mode research')
    parser.add_argument('--deep', '-d', metavar='TOPIC', help='Deep research mode')
    parser.add_argument('--interactive', '-i', action='store_true', help='Interactive mode')
    parser.add_argument('--model', '-m', default=DEFAULT_MODEL, help=f'Ollama model (default: {DEFAULT_MODEL})')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')

    args = parser.parse_args()

    # Initialize CLI
    cli = ResearchCLI(model=args.model, verbose=args.verbose)

    # Check Ollama
    if not cli.check_ollama():
        sys.exit(1)

    # Execute based on mode
    if args.interactive:
        cli.interactive()
    elif args.plan:
        result = cli.plan_mode(args.plan)
        cli.save_session()
    elif args.deep:
        result = cli.research(args.deep, depth="deep")
        cli.save_session()
    elif args.query:
        result = cli.research(args.query)
        cli.save_session()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
