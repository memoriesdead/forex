"""
LLM Fine-Tuning Dataset Extraction from Quant Code
===================================================
Extracts instruction-response pairs from 65k+ lines of quant code with
40+ academic citations for fine-tuning DeepSeek-R1-8B.

Sources:
- core/features/ - Alpha101, Renaissance signals, academic volatility
- core/_experimental/ - 191 Guotai Junan alphas, gold standard models
- core/rl/ - GRPO, TradeMaster implementations
- core/execution/ - Almgren-Chriss, market impact models
"""

import ast
import json
import re
import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class TrainingSample:
    """Single training sample for LLM fine-tuning."""
    instruction: str
    response: str
    source_file: str
    category: str


class QuantCodeExtractor:
    """Extract training samples from quantitative finance code."""

    # Directories to scan
    SOURCE_DIRS = [
        "core/features",
        "core/_experimental",
        "core/rl",
        "core/execution",
        "core/ml",
        "core/trading",
        "core/risk",
        "core/information",
    ]

    # Categories for classification
    CATEGORIES = {
        "alpha": ["alpha", "factor", "signal"],
        "volatility": ["volatility", "vol", "garch", "stochastic"],
        "risk": ["risk", "kelly", "sharpe", "drawdown", "var"],
        "execution": ["execution", "twap", "vwap", "almgren", "impact"],
        "microstructure": ["microstructure", "spread", "bid", "ask", "tick"],
        "rl": ["rl", "reinforcement", "agent", "dqn", "ppo", "grpo"],
        "regime": ["regime", "hmm", "markov", "state"],
        "ensemble": ["ensemble", "stacking", "blend", "meta"],
        "time_series": ["time_series", "arima", "kalman", "forecast"],
    }

    def __init__(self, base_path: Path):
        self.base_path = base_path
        self.samples: List[TrainingSample] = []

    def extract_all(self) -> List[TrainingSample]:
        """Extract training samples from all source directories."""
        for dir_name in self.SOURCE_DIRS:
            dir_path = self.base_path / dir_name
            if dir_path.exists():
                self._process_directory(dir_path)

        logger.info(f"Extracted {len(self.samples)} training samples")
        return self.samples

    def _process_directory(self, dir_path: Path) -> None:
        """Process all Python files in a directory."""
        for py_file in dir_path.glob("*.py"):
            if py_file.name.startswith("__"):
                continue
            try:
                self._process_file(py_file)
            except Exception as e:
                logger.warning(f"Error processing {py_file}: {e}")

    def _process_file(self, file_path: Path) -> None:
        """Process a single Python file to extract training samples."""
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()

        try:
            tree = ast.parse(content)
        except SyntaxError as e:
            logger.warning(f"Syntax error in {file_path}: {e}")
            return

        source_file = str(file_path.relative_to(self.base_path))

        # Extract module-level docstring
        module_doc = ast.get_docstring(tree)
        if module_doc:
            self._extract_from_module_doc(module_doc, source_file)

        # Extract class and method docstrings
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                self._extract_from_class(node, content, source_file)
            elif isinstance(node, ast.FunctionDef):
                self._extract_from_function(node, content, source_file)

    def _extract_from_module_doc(self, docstring: str, source_file: str) -> None:
        """Extract samples from module docstring."""
        category = self._categorize(source_file + " " + docstring)

        # Extract paper references
        papers = self._extract_paper_references(docstring)
        for paper in papers:
            self.samples.append(TrainingSample(
                instruction=f"What is the source paper for {paper['topic']}?",
                response=f"The source is: {paper['reference']}. {paper['description']}",
                source_file=source_file,
                category=category
            ))

        # Create overview question
        lines = docstring.strip().split("\n")
        title = lines[0] if lines else "Unknown"
        self.samples.append(TrainingSample(
            instruction=f"Explain the purpose of {title}",
            response=docstring,
            source_file=source_file,
            category=category
        ))

    def _extract_from_class(self, node: ast.ClassDef, content: str, source_file: str) -> None:
        """Extract samples from class definitions."""
        docstring = ast.get_docstring(node)
        if not docstring:
            return

        category = self._categorize(node.name + " " + (docstring or ""))

        # Class overview
        self.samples.append(TrainingSample(
            instruction=f"What is the {node.name} class and how do I use it?",
            response=f"The {node.name} class:\n\n{docstring}",
            source_file=source_file,
            category=category
        ))

        # Extract usage example if present
        if "usage:" in docstring.lower() or "example:" in docstring.lower():
            usage = self._extract_usage(docstring)
            if usage:
                self.samples.append(TrainingSample(
                    instruction=f"Show me a code example for {node.name}",
                    response=usage,
                    source_file=source_file,
                    category=category
                ))

    def _extract_from_function(self, node: ast.FunctionDef, content: str, source_file: str) -> None:
        """Extract samples from function definitions."""
        docstring = ast.get_docstring(node)
        if not docstring or node.name.startswith("_"):
            return

        category = self._categorize(node.name + " " + (docstring or ""))

        # Function implementation question
        func_name = node.name

        # Check if docstring contains a formula
        if any(c in docstring for c in ["=", "*", "/", "+", "-", "^", "sqrt"]):
            self.samples.append(TrainingSample(
                instruction=f"What is the formula for {self._humanize_name(func_name)}?",
                response=docstring,
                source_file=source_file,
                category=category
            ))

        # General explanation
        self.samples.append(TrainingSample(
            instruction=f"Explain the {self._humanize_name(func_name)} function",
            response=f"Function: {func_name}\n\n{docstring}",
            source_file=source_file,
            category=category
        ))

        # Check for specific quant concepts
        self._extract_quant_concepts(func_name, docstring, source_file, category)

    def _extract_quant_concepts(self, func_name: str, docstring: str,
                                 source_file: str, category: str) -> None:
        """Extract specific quant concept Q&As."""
        text = (func_name + " " + docstring).lower()

        concepts = {
            "kelly": ("Kelly Criterion", "position sizing formula that maximizes long-term growth"),
            "sharpe": ("Sharpe Ratio", "risk-adjusted return measure"),
            "vpin": ("VPIN", "Volume-Synchronized Probability of Informed Trading"),
            "ofi": ("OFI", "Order Flow Imbalance"),
            "twap": ("TWAP", "Time-Weighted Average Price execution"),
            "vwap": ("VWAP", "Volume-Weighted Average Price execution"),
            "almgren": ("Almgren-Chriss", "optimal execution model minimizing market impact"),
            "hmm": ("HMM", "Hidden Markov Model for regime detection"),
            "garch": ("GARCH", "volatility forecasting model"),
            "kalman": ("Kalman Filter", "state estimation and signal filtering"),
            "grpo": ("GRPO", "Group Relative Policy Optimization for RL"),
            "dqn": ("DQN", "Deep Q-Network for trading"),
            "ppo": ("PPO", "Proximal Policy Optimization"),
        }

        for key, (name, description) in concepts.items():
            if key in text:
                self.samples.append(TrainingSample(
                    instruction=f"What is {name} in quantitative trading?",
                    response=f"{name} ({description}).\n\nImplementation details:\n{docstring}",
                    source_file=source_file,
                    category=category
                ))

    def _extract_paper_references(self, text: str) -> List[Dict]:
        """Extract academic paper references from text."""
        papers = []

        # Pattern for papers like "Paper: ... (year)"
        paper_pattern = r"(?:Paper|Source|Reference):\s*[\"']?([^\n\"']+)[\"']?\s*(?:\((\d{4})\))?"

        for match in re.finditer(paper_pattern, text, re.IGNORECASE):
            reference = match.group(1).strip()
            year = match.group(2) or ""

            # Extract topic from context
            start = max(0, match.start() - 100)
            context = text[start:match.start()]
            topic_match = re.search(r"(\w+(?:\s+\w+){0,3})\s*$", context)
            topic = topic_match.group(1) if topic_match else "this technique"

            papers.append({
                "reference": reference,
                "year": year,
                "topic": topic,
                "description": ""
            })

        # Pattern for arxiv links
        arxiv_pattern = r"https?://arxiv\.org/abs/(\d+\.\d+)"
        for match in re.finditer(arxiv_pattern, text):
            papers.append({
                "reference": f"arXiv:{match.group(1)}",
                "year": "",
                "topic": "research paper",
                "description": ""
            })

        return papers

    def _extract_usage(self, docstring: str) -> Optional[str]:
        """Extract usage example from docstring."""
        usage_patterns = [
            r"Usage:\s*\n(.*?)(?=\n\n|\Z)",
            r"Example:\s*\n(.*?)(?=\n\n|\Z)",
            r"```python\s*(.*?)\s*```",
            r">>>\s*(.*?)(?=\n\n|\Z)",
        ]

        for pattern in usage_patterns:
            match = re.search(pattern, docstring, re.DOTALL | re.IGNORECASE)
            if match:
                return match.group(1).strip()

        return None

    def _categorize(self, text: str) -> str:
        """Categorize text based on keywords."""
        text_lower = text.lower()

        for category, keywords in self.CATEGORIES.items():
            if any(kw in text_lower for kw in keywords):
                return category

        return "general"

    def _humanize_name(self, name: str) -> str:
        """Convert function name to human-readable form."""
        # alpha001 -> Alpha 1
        alpha_match = re.match(r"alpha(\d+)", name)
        if alpha_match:
            return f"Alpha {int(alpha_match.group(1))}"

        # snake_case to Title Case
        return name.replace("_", " ").title()


class ConversationGenerator:
    """Generate multi-turn conversations from training samples."""

    TRADING_SCENARIOS = [
        "I'm building a forex HFT system",
        "I need to implement risk management",
        "How should I size my positions",
        "I want to detect market regimes",
        "Help me build a signal validation system",
        "I need to optimize trade execution",
        "How do I measure volatility",
        "I want to implement Alpha factors",
    ]

    def __init__(self, samples: List[TrainingSample]):
        self.samples = samples

    def generate_conversations(self, n_conversations: int = 200) -> List[Dict]:
        """Generate multi-turn trading conversations."""
        import random
        conversations = []

        # Group samples by category
        by_category = {}
        for sample in self.samples:
            by_category.setdefault(sample.category, []).append(sample)

        for _ in range(n_conversations):
            scenario = random.choice(self.TRADING_SCENARIOS)
            category = random.choice(list(by_category.keys()))
            category_samples = by_category[category]

            if len(category_samples) < 3:
                continue

            selected = random.sample(category_samples, min(3, len(category_samples)))

            turns = [{"role": "user", "content": scenario}]

            for i, sample in enumerate(selected):
                if i == 0:
                    turns.append({
                        "role": "assistant",
                        "content": f"I'll help you with that. Let me explain a key concept:\n\n{sample.response}"
                    })
                else:
                    turns.append({
                        "role": "user",
                        "content": sample.instruction
                    })
                    turns.append({
                        "role": "assistant",
                        "content": sample.response
                    })

            conversations.append({
                "conversation": turns,
                "category": category,
                "scenario": scenario
            })

        return conversations


def create_training_dataset(base_path: Path, output_dir: Path) -> Tuple[int, int]:
    """Create complete training dataset."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Extract samples
    extractor = QuantCodeExtractor(base_path)
    samples = extractor.extract_all()

    # Save individual samples as JSONL (Alpaca format)
    samples_file = output_dir / "llm_finetune_samples.jsonl"
    with open(samples_file, "w", encoding="utf-8") as f:
        for sample in samples:
            entry = {
                "instruction": sample.instruction,
                "input": "",
                "output": sample.response,
                "source": sample.source_file,
                "category": sample.category
            }
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    logger.info(f"Saved {len(samples)} samples to {samples_file}")

    # Generate multi-turn conversations
    conv_generator = ConversationGenerator(samples)
    conversations = conv_generator.generate_conversations(200)

    conv_file = output_dir / "llm_finetune_conversations.jsonl"
    with open(conv_file, "w", encoding="utf-8") as f:
        for conv in conversations:
            f.write(json.dumps(conv, ensure_ascii=False) + "\n")

    logger.info(f"Saved {len(conversations)} conversations to {conv_file}")

    # Create combined dataset (Alpaca format for Unsloth)
    combined_file = output_dir / "llm_finetune_dataset.jsonl"
    with open(combined_file, "w", encoding="utf-8") as f:
        for sample in samples:
            entry = {
                "instruction": sample.instruction,
                "input": "",
                "output": sample.response
            }
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    logger.info(f"Saved combined dataset to {combined_file}")

    # Print statistics
    categories = {}
    for sample in samples:
        categories[sample.category] = categories.get(sample.category, 0) + 1

    print("\n" + "="*60)
    print("Dataset Statistics")
    print("="*60)
    print(f"Total samples: {len(samples)}")
    print(f"Total conversations: {len(conversations)}")
    print("\nSamples by category:")
    for cat, count in sorted(categories.items(), key=lambda x: -x[1]):
        print(f"  {cat}: {count}")

    return len(samples), len(conversations)


def main():
    """Main entry point."""
    base_path = Path(__file__).parent.parent
    output_dir = base_path / "training_data"

    print("="*60)
    print("LLM Fine-Tuning Dataset Extraction")
    print("="*60)
    print(f"Base path: {base_path}")
    print(f"Output dir: {output_dir}")
    print()

    n_samples, n_convs = create_training_dataset(base_path, output_dir)

    print("\n" + "="*60)
    print("Dataset creation complete!")
    print(f"Files created in: {output_dir}")
    print("  - llm_finetune_samples.jsonl (individual Q&A pairs)")
    print("  - llm_finetune_conversations.jsonl (multi-turn dialogues)")
    print("  - llm_finetune_dataset.jsonl (combined Alpaca format)")
    print("="*60)


if __name__ == "__main__":
    main()
