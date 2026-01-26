#!/usr/bin/env python3
"""
Comprehensive Formula Extraction - ALL 1500+ Formulas
=====================================================
Extracts EVERY formula from the entire forex codebase for fine-tuning.

Target: 1500+ unique formulas, 10,000+ training samples
"""

import ast
import re
import json
import hashlib
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass, asdict
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class Formula:
    """Extracted formula entry."""
    name: str
    category: str
    formula: str
    explanation: str
    python_code: str
    source_file: str
    academic_reference: Optional[str] = None
    class_name: Optional[str] = None


@dataclass
class TrainingSample:
    """Training sample for fine-tuning."""
    instruction: str
    input: str
    output: str


class ComprehensiveExtractor:
    """Extract ALL formulas from the codebase."""

    # Categories based on keywords
    CATEGORY_KEYWORDS = {
        'alpha': 'alpha_factor',
        'volatility': 'volatility',
        'vol_': 'volatility',
        'garch': 'volatility',
        'har_': 'volatility',
        'parkinson': 'volatility',
        'yang_zhang': 'volatility',
        'garman': 'volatility',
        'rogers': 'volatility',
        'vpin': 'microstructure',
        'ofi': 'microstructure',
        'kyle': 'microstructure',
        'amihud': 'microstructure',
        'spread': 'microstructure',
        'bid_ask': 'microstructure',
        'order_flow': 'microstructure',
        'impact': 'microstructure',
        'twap': 'execution',
        'vwap': 'execution',
        'almgren': 'execution',
        'execution': 'execution',
        'slippage': 'execution',
        'kelly': 'risk',
        'sharpe': 'risk',
        'sortino': 'risk',
        'drawdown': 'risk',
        'var_': 'risk',
        'cvar': 'risk',
        'position_size': 'risk',
        'rsi': 'technical',
        'macd': 'technical',
        'bollinger': 'technical',
        'atr': 'technical',
        'ema': 'technical',
        'sma': 'technical',
        'momentum': 'technical',
        'stochastic': 'technical',
        'zscore': 'statistical',
        'correlation': 'statistical',
        'covariance': 'statistical',
        'rank': 'statistical',
        'decay': 'statistical',
        'normalize': 'statistical',
        'rolling': 'statistical',
        'loss': 'reinforcement_learning',
        'reward': 'reinforcement_learning',
        'advantage': 'reinforcement_learning',
        'policy': 'reinforcement_learning',
        'value_function': 'reinforcement_learning',
        'actor': 'reinforcement_learning',
        'critic': 'reinforcement_learning',
        'ppo': 'reinforcement_learning',
        'sac': 'reinforcement_learning',
        'dqn': 'reinforcement_learning',
        'grpo': 'reinforcement_learning',
        'predict': 'machine_learning',
        'train': 'machine_learning',
        'fit': 'machine_learning',
        'ensemble': 'machine_learning',
        'attention': 'deep_learning',
        'transformer': 'deep_learning',
        'lstm': 'deep_learning',
        'gru': 'deep_learning',
        'neural': 'deep_learning',
        'conv': 'deep_learning',
        'hmm': 'regime',
        'regime': 'regime',
        'kalman': 'filtering',
        'filter': 'filtering',
    }

    # Academic references
    ACADEMIC_REFS = {
        'alpha101': "Kakushadze (2016) '101 Formulaic Alphas' arXiv:1601.00991",
        'alpha158': "Qlib (Microsoft) Alpha158 Factor Library",
        'alpha360': "Qlib (Microsoft) Alpha360 Extended Factors",
        'alpha191': "国泰君安 (Guotai Junan) 191 Alpha Factors",
        'har': "Corsi (2009) 'HAR-RV Model' J. Financial Econometrics",
        'garch': "Bollerslev (1986) 'GARCH' J. Econometrics",
        'parkinson': "Parkinson (1980) 'Extreme Value Variance' J. Business",
        'yang_zhang': "Yang & Zhang (2000) 'Drift-Independent Volatility'",
        'garman_klass': "Garman & Klass (1980) 'Security Price Volatilities'",
        'rogers_satchell': "Rogers & Satchell (1991) 'Volatility Estimator'",
        'vpin': "Easley, Lopez de Prado, O'Hara (2012) 'Flow Toxicity' RFS",
        'ofi': "Cont, Kukanov, Stoikov (2014) 'Order Book Events' JFE",
        'kyle': "Kyle (1985) 'Continuous Auctions' Econometrica",
        'almgren': "Almgren & Chriss (2001) 'Optimal Execution' J. Risk",
        'kelly': "Kelly (1956) 'Information Rate' Bell System Tech J.",
        'sharpe': "Sharpe (1966) 'Mutual Fund Performance' J. Business",
        'sortino': "Sortino & Price (1994) 'Performance Measurement'",
        'grpo': "DeepSeek (2025) 'Group Relative Policy Optimization'",
        'ppo': "Schulman (2017) 'PPO Algorithms' arXiv:1707.06347",
        'sac': "Haarnoja (2018) 'Soft Actor-Critic' ICML",
        'dqn': "Mnih (2015) 'Human-level Control' Nature",
        'td3': "Fujimoto (2018) 'Twin Delayed DDPG' ICML",
        'transformer': "Vaswani (2017) 'Attention Is All You Need' NeurIPS",
        'lstm': "Hochreiter & Schmidhuber (1997) 'LSTM' Neural Computation",
        'hmm': "Rabiner (1989) 'HMM Tutorial' IEEE Proceedings",
        'kalman': "Kalman (1960) 'Filtering and Prediction' J. Basic Engineering",
        'barra': "Barra Risk Model (MSCI)",
        'fama_french': "Fama & French (1993) 'Common Risk Factors' JFE",
        'momentum': "Jegadeesh & Titman (1993) 'Momentum Strategies' JF",
        'mean_reversion': "Poterba & Summers (1988) 'Mean Reversion' JFE",
    }

    def __init__(self, base_path: Path):
        self.base_path = base_path
        self.formulas: List[Formula] = []
        self.samples: List[TrainingSample] = []
        self.seen_hashes: Set[str] = set()

    def extract_all(self) -> Tuple[List[Formula], List[TrainingSample]]:
        """Extract all formulas from all Python files."""
        # Find all Python files in core/
        core_path = self.base_path / "core"
        py_files = list(core_path.rglob("*.py"))

        logger.info(f"Found {len(py_files)} Python files in {core_path}")

        for py_file in py_files:
            if py_file.name.startswith("__"):
                continue
            try:
                self._process_file(py_file)
            except Exception as e:
                logger.warning(f"Error processing {py_file.name}: {e}")

        logger.info(f"Extracted {len(self.formulas)} unique formulas")
        logger.info(f"Generated {len(self.samples)} training samples")

        return self.formulas, self.samples

    def _process_file(self, file_path: Path) -> None:
        """Process a single Python file."""
        try:
            content = file_path.read_text(encoding='utf-8', errors='ignore')
        except Exception:
            return

        try:
            tree = ast.parse(content)
        except SyntaxError:
            return

        source_file = str(file_path.relative_to(self.base_path))
        module_doc = ast.get_docstring(tree) or ""

        # Process all nodes
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                self._process_class(node, content, source_file, module_doc)
            elif isinstance(node, ast.FunctionDef):
                self._process_function(node, content, source_file, module_doc)

    def _process_class(self, node: ast.ClassDef, content: str, source_file: str, module_doc: str) -> None:
        """Process class and its methods."""
        class_name = node.name
        class_doc = ast.get_docstring(node) or ""

        for item in node.body:
            if isinstance(item, ast.FunctionDef):
                self._process_function(item, content, source_file, module_doc, class_name, class_doc)

    def _process_function(self, node: ast.FunctionDef, content: str, source_file: str,
                         module_doc: str, class_name: str = "", class_doc: str = "") -> None:
        """Process function and extract formula."""
        func_name = node.name

        # Skip dunder methods except __init__ with formulas
        if func_name.startswith("__") and func_name != "__init__":
            return

        # Get docstring and source
        func_doc = ast.get_docstring(node) or ""

        try:
            func_source = ast.get_source_segment(content, node)
            if func_source and len(func_source) > 5000:
                func_source = func_source[:5000] + "\n# ... (truncated)"
        except:
            func_source = ""

        # Categorize
        category = self._categorize(func_name, func_doc, source_file, class_name)

        # Extract formula from docstring or code
        formula = self._extract_formula(func_doc, func_source)

        # Skip if no useful content
        if not func_source and not formula and not func_doc:
            return

        # Find academic reference
        ref = self._find_reference(func_name, func_doc, class_doc, module_doc, source_file)

        # Create entry
        entry = Formula(
            name=func_name,
            category=category,
            formula=formula[:1000] if formula else "",
            explanation=func_doc[:1000] if func_doc else "",
            python_code=func_source[:3000] if func_source else "",
            source_file=source_file,
            academic_reference=ref,
            class_name=class_name if class_name else None,
        )

        # Dedupe by name + formula + source
        entry_hash = hashlib.md5(f"{func_name}:{formula[:100]}:{source_file}".encode()).hexdigest()
        if entry_hash in self.seen_hashes:
            return
        self.seen_hashes.add(entry_hash)

        self.formulas.append(entry)
        self._generate_samples(entry)

    def _categorize(self, func_name: str, func_doc: str, source_file: str, class_name: str) -> str:
        """Categorize the function."""
        text = f"{func_name} {class_name} {func_doc} {source_file}".lower()

        for keyword, category in self.CATEGORY_KEYWORDS.items():
            if keyword in text:
                return category

        # Fallback based on file path
        if 'feature' in source_file.lower():
            return 'feature_engineering'
        if 'rl' in source_file.lower():
            return 'reinforcement_learning'
        if 'ml' in source_file.lower():
            return 'machine_learning'
        if 'execution' in source_file.lower():
            return 'execution'
        if 'risk' in source_file.lower():
            return 'risk'
        if 'data' in source_file.lower():
            return 'data_processing'

        return 'quantitative'

    def _extract_formula(self, docstring: str, source: str) -> str:
        """Extract mathematical formula."""
        formulas = []

        # Look for explicit formula patterns in docstring
        patterns = [
            r"Formula:\s*(.+?)(?:\n\n|\n[A-Z]|\Z)",
            r"Mathematical[^:]*:\s*(.+?)(?:\n\n|\Z)",
            r"Equation:\s*(.+?)(?:\n\n|\Z)",
            r"([A-Za-z_]+\s*=\s*[^\n]+)",
        ]

        for pattern in patterns:
            for match in re.finditer(pattern, docstring, re.DOTALL | re.IGNORECASE):
                formulas.append(match.group(1).strip())

        # Extract from return statements
        return_matches = re.findall(r"return\s+(.+?)$", source, re.MULTILINE)
        for match in return_matches:
            if len(match) < 200 and not match.startswith('self') and not match.startswith('None'):
                formulas.append(match.strip())

        return " | ".join(formulas[:3]) if formulas else ""

    def _find_reference(self, func_name: str, func_doc: str, class_doc: str,
                       module_doc: str, source_file: str) -> Optional[str]:
        """Find academic reference."""
        text = f"{func_name} {func_doc} {class_doc} {module_doc} {source_file}".lower()

        for key, ref in self.ACADEMIC_REFS.items():
            if key in text:
                return ref

        # Look for arXiv
        arxiv = re.search(r"arXiv[:\s]*(\d+\.\d+)", text, re.IGNORECASE)
        if arxiv:
            return f"arXiv:{arxiv.group(1)}"

        return None

    def _generate_samples(self, entry: Formula) -> None:
        """Generate multiple training samples from formula."""
        display_name = self._humanize(entry.name, entry.class_name)

        # Sample 1: Implementation
        if entry.python_code:
            self.samples.append(TrainingSample(
                instruction=f"Implement the {display_name} formula for quantitative forex trading",
                input=f"Category: {entry.category}",
                output=self._format_implementation(entry),
            ))

        # Sample 2: Explanation
        if entry.explanation:
            self.samples.append(TrainingSample(
                instruction=f"Explain {display_name} and how it's used in forex trading",
                input="",
                output=self._format_explanation(entry),
            ))

        # Sample 3: Formula query
        if entry.formula:
            self.samples.append(TrainingSample(
                instruction=f"What is the mathematical formula for {display_name}?",
                input="",
                output=f"The formula for {display_name} is:\n\n`{entry.formula}`\n\nCategory: {entry.category}",
            ))

        # Sample 4: Code example
        if entry.python_code and len(entry.python_code) > 50:
            self.samples.append(TrainingSample(
                instruction=f"Show Python code to calculate {display_name}",
                input="",
                output=f"```python\n{entry.python_code}\n```",
            ))

        # Sample 5: Academic source
        if entry.academic_reference:
            self.samples.append(TrainingSample(
                instruction=f"What is the academic source for {display_name}?",
                input="",
                output=f"Academic reference: {entry.academic_reference}\n\nThis {entry.category} formula is used in quantitative finance for trading and risk management.",
            ))

    def _humanize(self, name: str, class_name: str = None) -> str:
        """Convert to human-readable name."""
        # Handle alpha names
        alpha_match = re.match(r"alpha(\d+)", name, re.IGNORECASE)
        if alpha_match:
            return f"Alpha {int(alpha_match.group(1))}"

        display = name.replace("_", " ").title()
        if class_name:
            display = f"{class_name}.{display}"

        return display

    def _format_implementation(self, entry: Formula) -> str:
        """Format implementation response."""
        parts = [f"# {self._humanize(entry.name, entry.class_name)}\n"]

        if entry.formula:
            parts.append(f"**Formula**: `{entry.formula}`\n")
        if entry.explanation:
            parts.append(f"**Explanation**: {entry.explanation[:500]}\n")
        if entry.python_code:
            parts.append(f"**Implementation**:\n```python\n{entry.python_code}\n```\n")
        if entry.academic_reference:
            parts.append(f"**Reference**: {entry.academic_reference}")

        return "\n".join(parts)

    def _format_explanation(self, entry: Formula) -> str:
        """Format explanation response."""
        parts = [f"{self._humanize(entry.name, entry.class_name)} is a {entry.category} technique used in quantitative trading.\n"]
        parts.append(entry.explanation)

        if entry.academic_reference:
            parts.append(f"\n\n**Academic Reference**: {entry.academic_reference}")

        return "\n".join(parts)


def main():
    """Main entry point."""
    base_path = Path(__file__).parent.parent
    output_dir = base_path / "training_data"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("COMPREHENSIVE FORMULA EXTRACTION - Target 1500+ Formulas")
    print("=" * 70)

    extractor = ComprehensiveExtractor(base_path)
    formulas, samples = extractor.extract_all()

    # Save formulas
    formulas_file = output_dir / "all_formulas.json"
    with open(formulas_file, "w", encoding="utf-8") as f:
        json.dump([asdict(f) for f in formulas], f, indent=2, ensure_ascii=False)
    print(f"\nSaved {len(formulas)} formulas to {formulas_file}")

    # Save training samples
    samples_file = output_dir / "deepseek_finetune_dataset.jsonl"
    with open(samples_file, "w", encoding="utf-8") as f:
        for sample in samples:
            f.write(json.dumps(asdict(sample), ensure_ascii=False) + "\n")
    print(f"Saved {len(samples)} training samples to {samples_file}")

    # Statistics
    print("\n" + "=" * 70)
    print("EXTRACTION STATISTICS")
    print("=" * 70)

    categories = {}
    for formula in formulas:
        categories[formula.category] = categories.get(formula.category, 0) + 1

    print("\nFormulas by category:")
    for cat, count in sorted(categories.items(), key=lambda x: -x[1]):
        print(f"  {cat}: {count}")

    refs = sum(1 for f in formulas if f.academic_reference)
    print(f"\nWith academic references: {refs}/{len(formulas)} ({refs/len(formulas)*100:.1f}%)")

    print("\n" + "=" * 70)
    print(f"TOTAL FORMULAS: {len(formulas)}")
    print(f"TOTAL TRAINING SAMPLES: {len(samples)}")
    print("=" * 70)

    if len(formulas) < 1500:
        print(f"\nWARNING: Only {len(formulas)} formulas extracted. Target was 1500+")
    else:
        print(f"\nSUCCESS: Extracted {len(formulas)} formulas (target: 1500+)")


if __name__ == "__main__":
    main()
