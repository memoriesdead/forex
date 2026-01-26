"""
Complete Formula Extraction for DeepSeek-R1 Fine-Tuning
========================================================
Extracts ALL 806+ mathematical formulas from the forex codebase.

Target files:
- core/features/ (37 files) - Alpha101, Alpha158, Alpha360, Renaissance, etc.
- core/_experimental/ (30+ files) - Alpha191, GARCH, Kalman, etc.
- core/execution/ (10+ files) - Almgren-Chriss, TWAP, VWAP
- core/rl/ (19 files) - GRPO, PPO, SAC, DQN
- core/risk/ (4 files) - Kelly, Sharpe, Sortino
- core/ml/ (12 files) - Ensemble, online learning

Output: 3,000+ training samples for Unsloth fine-tuning
"""

import ast
import re
import json
import inspect
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, asdict
import logging
import hashlib

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class FormulaEntry:
    """Extracted mathematical formula."""
    name: str
    category: str
    formula: str
    explanation: str
    python_code: str
    source_file: str
    academic_reference: Optional[str] = None
    parameters: Optional[Dict[str, str]] = None


@dataclass
class TrainingSample:
    """Single training sample for fine-tuning."""
    instruction: str
    input: str
    output: str
    category: str
    source: str


class ComprehensiveFormulaExtractor:
    """
    Extracts ALL mathematical formulas from the forex codebase.

    Coverage:
    - Alpha factors (101 + 158 + 360 + 191 = 810 alphas)
    - Volatility models (HAR-RV, GARCH, Parkinson, etc.)
    - Microstructure (OFI, VPIN, Kyle Lambda, etc.)
    - Execution (Almgren-Chriss, TWAP, VWAP)
    - Risk (Kelly, Sharpe, Sortino)
    - RL algorithms (GRPO, PPO, SAC, etc.)
    """

    # All directories to scan
    SCAN_DIRS = [
        "core/features",
        "core/_experimental",
        "core/execution",
        "core/execution/optimization",
        "core/rl",
        "core/risk",
        "core/ml",
        "core/trading",
        "core/information",
    ]

    # Formula patterns to detect
    FORMULA_PATTERNS = {
        # Alpha factors
        r"def alpha(\d+)": "alpha_factor",
        r"def calc_alpha": "alpha_factor",

        # Volatility
        r"def .*volatility|def .*vol_|def har_|def garch|def parkinson|def yang_zhang|def garman_klass|def rogers_satchell": "volatility",

        # Microstructure
        r"def .*ofi|def .*vpin|def .*kyle|def .*amihud|def .*roll_spread|def .*bid_ask": "microstructure",

        # Execution
        r"def .*twap|def .*vwap|def .*almgren|def .*execution|def .*impact|def optimal_": "execution",

        # Risk
        r"def .*kelly|def .*sharpe|def .*sortino|def .*drawdown|def .*var_|def .*cvar": "risk",

        # Technical indicators
        r"def .*rsi|def .*macd|def .*bollinger|def .*atr|def .*ema|def .*sma|def .*momentum": "technical",

        # Statistical
        r"def .*zscore|def .*correlation|def .*covariance|def .*rank|def .*decay": "statistical",

        # RL
        r"def .*loss|def .*reward|def .*advantage|def .*policy|def .*value_function": "reinforcement_learning",

        # ML
        r"def .*predict|def .*train|def .*fit|def .*ensemble": "machine_learning",
    }

    # Academic references to extract
    ACADEMIC_REFS = {
        "alpha101": "Kakushadze, Z. (2016). '101 Formulaic Alphas'. arXiv:1601.00991",
        "har": "Corsi, F. (2009). 'A Simple Approximate Long-Memory Model of Realized Volatility'. Journal of Financial Econometrics 7(2)",
        "garch": "Bollerslev, T. (1986). 'Generalized Autoregressive Conditional Heteroskedasticity'. Journal of Econometrics 31(3)",
        "parkinson": "Parkinson, M. (1980). 'The Extreme Value Method for Estimating the Variance of the Rate of Return'. Journal of Business 53(1)",
        "yang_zhang": "Yang, D. & Zhang, Q. (2000). 'Drift-Independent Volatility Estimation'. Journal of Business 73(3)",
        "garman_klass": "Garman, M.B. & Klass, M.J. (1980). 'On the Estimation of Security Price Volatilities'. Journal of Business 53(1)",
        "ofi": "Cont, R., Kukanov, A. & Stoikov, S. (2014). 'The Price Impact of Order Book Events'. Journal of Financial Econometrics 12(1)",
        "vpin": "Easley, D., López de Prado, M. & O'Hara, M. (2012). 'Flow Toxicity and Liquidity in a High Frequency World'. Review of Financial Studies 25(5)",
        "kyle": "Kyle, A.S. (1985). 'Continuous Auctions and Insider Trading'. Econometrica 53(6)",
        "almgren": "Almgren, R. & Chriss, N. (2001). 'Optimal Execution of Portfolio Transactions'. Journal of Risk 3(2)",
        "kelly": "Kelly, J.L. (1956). 'A New Interpretation of Information Rate'. Bell System Technical Journal",
        "sharpe": "Sharpe, W.F. (1966). 'Mutual Fund Performance'. Journal of Business 39(1)",
        "grpo": "DeepSeek-AI (2025). 'DeepSeek-R1'. arXiv:2501.12948",
        "ppo": "Schulman, J. et al. (2017). 'Proximal Policy Optimization Algorithms'. arXiv:1707.06347",
    }

    def __init__(self, base_path: Path):
        self.base_path = base_path
        self.formulas: List[FormulaEntry] = []
        self.samples: List[TrainingSample] = []
        self.seen_hashes = set()  # Dedupe

    def extract_all(self) -> Tuple[List[FormulaEntry], List[TrainingSample]]:
        """Extract all formulas and generate training samples."""
        for dir_name in self.SCAN_DIRS:
            dir_path = self.base_path / dir_name
            if dir_path.exists():
                self._process_directory(dir_path)

        logger.info(f"Extracted {len(self.formulas)} formulas")
        logger.info(f"Generated {len(self.samples)} training samples")

        return self.formulas, self.samples

    def _process_directory(self, dir_path: Path) -> None:
        """Process all Python files in directory."""
        for py_file in dir_path.glob("*.py"):
            if py_file.name.startswith("__"):
                continue
            try:
                self._process_file(py_file)
            except Exception as e:
                logger.warning(f"Error processing {py_file}: {e}")

    def _process_file(self, file_path: Path) -> None:
        """Process single Python file."""
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read()

        try:
            tree = ast.parse(content)
        except SyntaxError:
            return

        source_file = str(file_path.relative_to(self.base_path))

        # Extract module docstring for academic references
        module_doc = ast.get_docstring(tree) or ""

        # Process all classes and functions
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                self._process_class(node, content, source_file, module_doc)
            elif isinstance(node, ast.FunctionDef):
                self._process_function(node, content, source_file, module_doc)

    def _process_class(self, node: ast.ClassDef, content: str, source_file: str, module_doc: str) -> None:
        """Process class definition."""
        class_doc = ast.get_docstring(node) or ""

        # Check if this is an alpha/formula class
        class_name = node.name.lower()

        # Process all methods in the class
        for item in node.body:
            if isinstance(item, ast.FunctionDef):
                self._process_function(item, content, source_file, module_doc, class_name, class_doc)

    def _process_function(self, node: ast.FunctionDef, content: str, source_file: str,
                         module_doc: str, class_name: str = "", class_doc: str = "") -> None:
        """Process function definition and extract formula."""
        func_name = node.name

        # Skip private methods except alpha methods
        if func_name.startswith("_") and not func_name.startswith("__"):
            if not re.match(r"_?alpha\d+", func_name):
                return

        # Get function docstring
        func_doc = ast.get_docstring(node) or ""

        # Get function source code
        try:
            func_source = ast.get_source_segment(content, node)
        except:
            func_source = ""

        # Categorize the function
        category = self._categorize_function(func_name, func_doc, source_file)
        if not category:
            return

        # Extract formula from docstring
        formula = self._extract_formula(func_doc, func_source)

        # Find academic reference
        ref = self._find_reference(func_name, func_doc, module_doc, class_doc)

        # Create formula entry
        entry = FormulaEntry(
            name=func_name,
            category=category,
            formula=formula,
            explanation=func_doc[:500] if func_doc else "",
            python_code=func_source[:2000] if func_source else "",
            source_file=source_file,
            academic_reference=ref,
            parameters=self._extract_parameters(node),
        )

        # Dedupe
        entry_hash = hashlib.md5(f"{entry.name}:{entry.formula}".encode()).hexdigest()
        if entry_hash in self.seen_hashes:
            return
        self.seen_hashes.add(entry_hash)

        self.formulas.append(entry)

        # Generate training samples
        self._generate_samples(entry, class_name)

    def _categorize_function(self, func_name: str, func_doc: str, source_file: str) -> Optional[str]:
        """Categorize function based on name and content."""
        text = f"{func_name} {func_doc} {source_file}".lower()

        for pattern, category in self.FORMULA_PATTERNS.items():
            if re.search(pattern, func_name, re.IGNORECASE):
                return category
            if re.search(pattern, text, re.IGNORECASE):
                return category

        # Additional categorization from file path
        if "alpha" in source_file.lower():
            return "alpha_factor"
        if "volatility" in source_file.lower():
            return "volatility"
        if "microstructure" in source_file.lower():
            return "microstructure"
        if "execution" in source_file.lower():
            return "execution"
        if "risk" in source_file.lower():
            return "risk"
        if "/rl/" in source_file.lower():
            return "reinforcement_learning"

        return None

    def _extract_formula(self, docstring: str, source: str) -> str:
        """Extract mathematical formula from docstring or code."""
        # Look for explicit formula patterns
        patterns = [
            r"Formula:\s*(.+?)(?:\n\n|\Z)",
            r"([A-Za-z_]+\s*=\s*[^\n]+(?:[\+\-\*/\^]\s*[^\n]+)*)",
            r"((?:rank|ts_|delta|correlation|stddev)\([^\)]+\)(?:\s*[\+\-\*/]\s*[^\n]+)?)",
        ]

        for pattern in patterns:
            match = re.search(pattern, docstring, re.DOTALL | re.IGNORECASE)
            if match:
                return match.group(1).strip()

        # Extract from return statement in source
        return_match = re.search(r"return\s+(.+?)$", source, re.MULTILINE)
        if return_match:
            return return_match.group(1).strip()[:200]

        return ""

    def _find_reference(self, func_name: str, func_doc: str, module_doc: str, class_doc: str) -> Optional[str]:
        """Find academic reference for this formula."""
        all_text = f"{func_name} {func_doc} {module_doc} {class_doc}".lower()

        for key, ref in self.ACADEMIC_REFS.items():
            if key in all_text:
                return ref

        # Look for arxiv/paper references in docstrings
        arxiv_match = re.search(r"arXiv[:\s]*(\d+\.\d+)", all_text, re.IGNORECASE)
        if arxiv_match:
            return f"arXiv:{arxiv_match.group(1)}"

        paper_match = re.search(r"(?:paper|reference|source):\s*([^\n]+)", all_text, re.IGNORECASE)
        if paper_match:
            return paper_match.group(1).strip()

        return None

    def _extract_parameters(self, node: ast.FunctionDef) -> Dict[str, str]:
        """Extract function parameters."""
        params = {}
        for arg in node.args.args:
            arg_name = arg.arg
            if arg_name != "self":
                params[arg_name] = str(arg.annotation) if arg.annotation else "Any"
        return params

    def _generate_samples(self, entry: FormulaEntry, class_name: str) -> None:
        """Generate training samples from formula entry."""

        # Sample 1: Implementation question
        self.samples.append(TrainingSample(
            instruction=f"Implement the {self._humanize_name(entry.name)} formula used in quantitative trading",
            input=f"Category: {entry.category}" + (f"\nClass: {class_name}" if class_name else ""),
            output=self._format_implementation_response(entry),
            category=entry.category,
            source=entry.source_file,
        ))

        # Sample 2: Explanation question
        if entry.explanation:
            self.samples.append(TrainingSample(
                instruction=f"Explain {self._humanize_name(entry.name)} and when to use it in forex trading",
                input="",
                output=self._format_explanation_response(entry),
                category=entry.category,
                source=entry.source_file,
            ))

        # Sample 3: Formula derivation
        if entry.formula:
            self.samples.append(TrainingSample(
                instruction=f"What is the mathematical formula for {self._humanize_name(entry.name)}?",
                input="",
                output=self._format_formula_response(entry),
                category=entry.category,
                source=entry.source_file,
            ))

        # Sample 4: Code example
        if entry.python_code:
            self.samples.append(TrainingSample(
                instruction=f"Show me Python code to calculate {self._humanize_name(entry.name)}",
                input="",
                output=f"Here's the implementation:\n\n```python\n{entry.python_code}\n```",
                category=entry.category,
                source=entry.source_file,
            ))

        # Sample 5: Academic reference
        if entry.academic_reference:
            self.samples.append(TrainingSample(
                instruction=f"What is the academic source for {self._humanize_name(entry.name)}?",
                input="",
                output=f"The academic reference is: {entry.academic_reference}\n\nThis formula is widely used in quantitative finance for {entry.category} calculations.",
                category=entry.category,
                source=entry.source_file,
            ))

    def _humanize_name(self, name: str) -> str:
        """Convert function name to human-readable form."""
        # Handle alpha names
        alpha_match = re.match(r"alpha(\d+)", name, re.IGNORECASE)
        if alpha_match:
            return f"Alpha {int(alpha_match.group(1))}"

        # Snake case to title case
        return name.replace("_", " ").title()

    def _format_implementation_response(self, entry: FormulaEntry) -> str:
        """Format implementation response."""
        response = f"# {self._humanize_name(entry.name)}\n\n"

        if entry.formula:
            response += f"**Formula**: `{entry.formula}`\n\n"

        if entry.explanation:
            response += f"**Explanation**: {entry.explanation}\n\n"

        if entry.python_code:
            response += f"**Python Implementation**:\n```python\n{entry.python_code}\n```\n\n"

        if entry.academic_reference:
            response += f"**Reference**: {entry.academic_reference}\n"

        return response

    def _format_explanation_response(self, entry: FormulaEntry) -> str:
        """Format explanation response."""
        response = f"{self._humanize_name(entry.name)} is a {entry.category} formula used in quantitative trading.\n\n"
        response += entry.explanation

        if entry.academic_reference:
            response += f"\n\n**Academic Reference**: {entry.academic_reference}"

        return response

    def _format_formula_response(self, entry: FormulaEntry) -> str:
        """Format formula response."""
        response = f"The formula for {self._humanize_name(entry.name)} is:\n\n"
        response += f"```\n{entry.formula}\n```\n\n"

        if entry.parameters:
            response += "**Parameters**:\n"
            for param, ptype in entry.parameters.items():
                response += f"- `{param}`: {ptype}\n"

        return response


def main():
    """Main entry point."""
    base_path = Path(__file__).parent.parent
    output_dir = base_path / "training_data"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("Comprehensive Formula Extraction for DeepSeek-R1 Fine-Tuning")
    print("=" * 70)
    print(f"Base path: {base_path}")
    print(f"Output: {output_dir}")
    print()

    # Extract formulas
    extractor = ComprehensiveFormulaExtractor(base_path)
    formulas, samples = extractor.extract_all()

    # Save formulas
    formulas_file = output_dir / "all_formulas.json"
    with open(formulas_file, "w", encoding="utf-8") as f:
        json.dump([asdict(f) for f in formulas], f, indent=2, ensure_ascii=False)
    print(f"Saved {len(formulas)} formulas to {formulas_file}")

    # Save training samples (Alpaca format for Unsloth)
    samples_file = output_dir / "deepseek_finetune_dataset.jsonl"
    with open(samples_file, "w", encoding="utf-8") as f:
        for sample in samples:
            entry = {
                "instruction": sample.instruction,
                "input": sample.input,
                "output": sample.output,
            }
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    print(f"Saved {len(samples)} training samples to {samples_file}")

    # Print statistics
    print("\n" + "=" * 70)
    print("Extraction Statistics")
    print("=" * 70)

    # By category
    categories = {}
    for formula in formulas:
        categories[formula.category] = categories.get(formula.category, 0) + 1

    print("\nFormulas by category:")
    for cat, count in sorted(categories.items(), key=lambda x: -x[1]):
        print(f"  {cat}: {count}")

    # By source file
    sources = {}
    for formula in formulas:
        src = formula.source_file.split("/")[0] if "/" in formula.source_file else formula.source_file
        sources[src] = sources.get(src, 0) + 1

    print("\nFormulas by source directory:")
    for src, count in sorted(sources.items(), key=lambda x: -x[1]):
        print(f"  {src}: {count}")

    # Academic references
    refs = sum(1 for f in formulas if f.academic_reference)
    print(f"\nFormulas with academic references: {refs}/{len(formulas)} ({refs/len(formulas)*100:.1f}%)")

    print("\n" + "=" * 70)
    print("Extraction complete!")
    print(f"Total formulas: {len(formulas)}")
    print(f"Total training samples: {len(samples)}")
    print("=" * 70)


if __name__ == "__main__":
    main()
