"""
Extraction agent domain diversity analysis.

Usage:
  python eval/extraction_diversity/eval_extraction_diversity.py
  python eval/extraction_diversity/eval_extraction_diversity.py --trials 10
  python eval/extraction_diversity/eval_extraction_diversity.py --config path/to/config.yaml
  python eval/extraction_diversity/eval_extraction_diversity.py --question "New question here"
  python eval/extraction_diversity/eval_extraction_diversity.py --load path/to/run_dir
"""

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

# Add project root to path
ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))

# Load environment variables
from dotenv import load_dotenv
load_dotenv(ROOT / ".env")

import yaml
import matplotlib.pyplot as plt
from agents.extraction import extract_problem
from config import get_config, config as config_singleton
from metrics import compute_all_metrics


# ANSI color codes
def _init_colors():
    if get_config("output.use_colors", True):
        return {
            'C': '\033[96m',  # Cyan
            'B': '\033[94m',  # Blue
            'G': '\033[92m',  # Green
            'Y': '\033[93m',  # Yellow
            'R': '\033[0m'    # Reset
        }
    return {k: '' for k in ['C', 'B', 'G', 'Y', 'R']}


def apply_extraction_overrides(eval_config: dict):
    """Apply extraction settings from eval config to main config singleton."""
    if "extraction" not in eval_config:
        return

    # Ensure main config is loaded
    main_config = config_singleton.get_all()
    if "extraction" not in main_config:
        main_config["extraction"] = {}

    # Override extraction settings
    for key, value in eval_config["extraction"].items():
        if value is not None:
            main_config["extraction"][key] = value


def load_config(config_path: str = None) -> dict:
    """Load config from yaml file."""
    path = Path(config_path) if config_path else Path(__file__).parent / "eval_config.yaml"
    with open(path) as f:
        return yaml.safe_load(f)


def get_model_pricing(model_name: str) -> tuple[float, float]:
    """Get pricing for a model.

    Args:
        model_name: Model identifier

    Returns:
        (input_price_per_mtok, output_price_per_mtok)
    """
    model_lower = model_name.lower()

    if 'haiku' in model_lower:
        return (1.0, 5.0)
    elif 'sonnet' in model_lower:
        return (3.0, 15.0)
    elif 'opus' in model_lower:
        return (15.0, 75.0)
    else:
        return (3.0, 15.0)


def calculate_cost(tokens: dict, model: str = "claude-sonnet-4-5-20250929") -> float:
    """Calculate cost in USD based on model pricing.

    Args:
        tokens: Token usage dict
        model: Model name

    Returns:
        Cost in USD
    """
    input_price, output_price = get_model_pricing(model)
    return (tokens.get("input", 0) / 1_000_000) * input_price + (tokens.get("output", 0) / 1_000_000) * output_price


def run_analysis(config: dict, extra_questions: list = None, run_id: str = None) -> tuple[list, list, dict, str, float]:
    """Run extraction trials and return runs list, output text, tokens, run_id, and runtime."""
    start_time = time.time()
    questions = config["questions"] + (extra_questions or [])
    num_trials = config["analysis"]["num_trials"]
    model = config.get("model") or get_config("model.name", "claude-sonnet-4-5-20250929")

    # Generate run ID if not provided
    if run_id is None:
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    output_lines = []
    def log(msg=""):
        print(msg)
        output_lines.append(msg)

    # Determine prompt being used
    custom_prompt = get_config("extraction.prompt_file", None)
    if custom_prompt:
        prompt_used = custom_prompt
    else:
        use_diverse = get_config("extraction.use_diverse_prompts", False)
        prompt_used = "prompts/extraction_diverse.txt" if use_diverse else "prompts/extraction.txt"

    log("=" * 60)
    log("EXTRACTION DIVERSITY ANALYSIS")
    log("=" * 60)
    log(f"Run ID: {run_id}")
    log(f"Config: {len(questions)} questions x {num_trials} trials = {len(questions) * num_trials} runs")
    log(f"Model: {model}")
    log(f"Prompt: {prompt_used}")
    log()
    log("Running trials...")

    runs = []
    total_tokens = {"input": 0, "output": 0}

    for qi, question in enumerate(questions, 1):
        log(f"  [Q{qi}] {question}")

        for trial in range(1, num_trials + 1):
            extraction, tokens = extract_problem(question)
            domains = extraction.get("target_domains", [])
            cost = calculate_cost(tokens, model)

            runs.append({
                "question": question,
                "trial": trial,
                "target_domains": domains,
                "tokens": tokens,
                "cost": cost,
                "timestamp": datetime.now().isoformat(),
            })

            total_tokens["input"] += tokens.get("input", 0)
            total_tokens["output"] += tokens.get("output", 0)

            log(f"    Trial {trial}/{num_trials}: {', '.join(domains)}")

    log()
    runtime = time.time() - start_time
    return runs, output_lines, total_tokens, run_id, runtime


def format_results(runs: list, total_tokens: dict, output_lines: list, runtime: float, model: str = "claude-sonnet-4-5-20250929") -> dict:
    """Format and print results summary."""
    c = _init_colors()

    def log(msg=""):
        """Log with colors to both console and file."""
        print(msg)
        output_lines.append(msg)

    metrics = compute_all_metrics(runs)
    total_cost = calculate_cost(total_tokens, model)

    log("=" * 60)
    log("RESULTS SUMMARY")
    log("=" * 60)
    log()
    log("OVERALL METRICS")
    log(f"  Total runs: {len(runs)}")
    log(f"  Total unique domains: {len(metrics['unique_domains'])}")
    dpr = metrics["domains_per_run"]
    log(f"  Domains per run: mean={dpr['mean']:.1f}, std={dpr['std']:.1f}")
    if metrics["unique_domains"] and dpr["mean"] > 0:
        # Expected % of trials a domain appears in if sampling uniformly
        random_pct = (dpr["mean"] / len(metrics["unique_domains"])) * 100.0
        log(f"  Random expected (if uniform): {random_pct:.1f}% of trials per domain")
    log()

    log("DOMAIN FREQUENCY BY QUESTION")
    for question, q_data in metrics["per_question"].items():
        avg_per_trial = q_data["avg_domains_per_trial"]
        num_unique = len(q_data["unique_domains"])
        log(f"  [{question}] ({q_data['trials']} trials, {num_unique} unique domains, {avg_per_trial:.1f} avg/trial)")

        # Expected % of trials a domain appears in if sampling uniformly
        random_pct = (avg_per_trial / num_unique) * 100.0 if num_unique > 0 else 0
        log(f"    {'Domain':<30} {'Count':>6} {'Observed%':>10} {'Random%':>10}")

        for domain, count in q_data["domain_counts"].items():
            observed_pct = 100.0 * count / q_data["trials"]
            if observed_pct > random_pct * 1.5:
                marker = " <- overrepresented"
            elif observed_pct < random_pct * 0.5:
                marker = " <- underrepresented"
            else:
                marker = ""
            log(f"    {domain:<30} {count:>6} {observed_pct:>9.0f}% {random_pct:>9.1f}%{marker}")

        log(f"    Mean Jaccard similarity: {q_data['mean_jaccard']:.2f}")
        log()

    log("CROSS-QUESTION ANALYSIS")
    cross = metrics["cross_question"]
    if cross["universal_domains"]:
        log(f"  Universal domains (appear for all questions):")
        log(f"    {', '.join(cross['universal_domains'])}")
    else:
        log("  Universal domains: (none)")
    log()

    if cross["question_specific"]:
        log("  Question-specific domains:")
        for q, domains in cross["question_specific"].items():
            log(f"    {q}: {', '.join(domains)}")
    log()

    log(f"{c['B']}COST & RUNTIME{c['R']}")
    log(f"  {c['G']}Total tokens: {total_tokens['input']:,} input / {total_tokens['output']:,} output{c['R']}")
    log(f"  {c['Y']}Total cost: ${total_cost:.4f} USD{c['R']}")
    log(f"  {c['G']}Runtime: {runtime:.2f}s{c['R']}")
    log("=" * 60)

    return metrics


def load_results(run_dir: str) -> tuple[dict, Path]:
    """Load saved results from a run directory."""
    path = Path(run_dir)
    with open(path / "results.json") as f:
        return json.load(f), path


def create_domain_frequency_graphs(metrics: dict, output_dir: Path):
    """Create bar graphs of domain frequency by question."""
    for qi, (question, q_data) in enumerate(metrics["per_question"].items(), 1):
        if not q_data["domain_counts"]:
            continue

        domains = list(q_data["domain_counts"].keys())
        counts = list(q_data["domain_counts"].values())
        num_trials = q_data["trials"]
        avg_per_trial = q_data["avg_domains_per_trial"]

        fig, ax = plt.subplots(figsize=(10, max(4, len(domains) * 0.3)))
        ax.barh(domains, counts, color='steelblue')
        ax.set_xlabel('Count')
        ax.set_title(f'Domain Frequency ({num_trials} trials, {len(domains)} unique domains)')

        # Add random baseline
        if len(domains) > 0 and avg_per_trial > 0:
            random_expected = (avg_per_trial / len(domains)) * num_trials
            ax.axvline(x=random_expected, color='red', linestyle='--', label=f'Random baseline ({random_expected:.1f})')
            ax.legend(loc='lower right')

        # Add question and Jaccard similarity textbox
        jaccard = q_data.get("mean_jaccard", 0)
        fig.text(0.5, -0.02, f'Q: {question}\nMean Jaccard Similarity: {jaccard:.2f}', ha='center', fontsize=11, wrap=True)

        plt.tight_layout()
        plt.savefig(output_dir / f"domain_freq_q{qi}.png", dpi=150, bbox_inches='tight')
        plt.close()


def save_results(runs: list, metrics: dict, config: dict, total_tokens: dict, output_lines: list, run_id: str):
    """Save results to JSON and text files in a run-specific directory."""
    output_dir = Path(ROOT) / config["analysis"]["output_dir"] / run_id
    output_dir.mkdir(parents=True, exist_ok=True)

    # Get model from config
    model = config.get("model") or get_config("model.name", "claude-sonnet-4-5-20250929")

    # Save JSON
    json_data = {
        "run_id": run_id,
        "config": config,
        "runs": runs,
        "summary": {
            "total_runs": len(runs),
            "total_tokens": total_tokens,
            "total_cost": calculate_cost(total_tokens, model),
            **metrics,
        },
    }
    json_path = output_dir / "results.json"
    with open(json_path, "w") as f:
        json.dump(json_data, f, indent=2)
    print(f"\nSaved JSON: {json_path}")

    # Save log (with ANSI colors)
    log_path = output_dir / "results.log"
    with open(log_path, "w") as f:
        f.write("\n".join(output_lines))
    print(f"Saved log: {log_path}")

    # Create bar graphs
    create_domain_frequency_graphs(metrics, output_dir)
    print(f"Saved domain frequency graphs to: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Extraction agent domain diversity analysis")
    parser.add_argument("--trials", type=int, help="Number of trials per question")
    parser.add_argument("--config", type=str, help="Path to config file")
    parser.add_argument("--question", type=str, action="append", help="Additional question(s)")
    parser.add_argument("--load", type=str, help="Load saved results from run directory and regenerate plots")
    args = parser.parse_args()

    config = load_config(args.config)

    # Load saved results and regenerate plots
    load_run = args.load or config["analysis"].get("load_run")
    if load_run:
        run_dir = Path(ROOT) / config["analysis"]["output_dir"] / str(load_run)
        print(f"Loading pre-generated results from: {run_dir}")
        data, _ = load_results(run_dir)
        create_domain_frequency_graphs(data["summary"], run_dir)
        print(f"Saved plots to: {run_dir}")
        return

    if args.trials:
        config["analysis"]["num_trials"] = args.trials

    # Apply extraction settings to main config singleton
    apply_extraction_overrides(config)

    runs, output_lines, total_tokens, run_id, runtime = run_analysis(config, args.question)
    model = config.get("model") or get_config("model.name", "claude-sonnet-4-5-20250929")
    metrics = format_results(runs, total_tokens, output_lines, runtime, model)
    save_results(runs, metrics, config, total_tokens, output_lines, run_id)


if __name__ == "__main__":
    main()
