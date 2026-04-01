#!/usr/bin/env python3
"""
Compare AR-discovered analogies to ground truth analogies from the dataset.

Scores both AR and ground truth analogies on creativity, novelty, and structural depth
using an LLM judge.
"""

import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any
from anthropic import Anthropic
import numpy as np
from dotenv import load_dotenv
from statistics import mean

# Get project root
ROOT = Path(__file__).parent.parent.parent

# Load environment variables
load_dotenv(ROOT / '.env')

# ANSI color codes
C = '\033[96m'  # Cyan
B = '\033[94m'  # Blue
G = '\033[92m'  # Green
Y = '\033[93m'  # Yellow
R = '\033[0m'   # Reset

CONFIG = {
    'ar_run': '',
    'cross_domain_run': '',
    'no_domain_run': '',
    'dataset_path': '',
    'paper_ids': None,
    'output_dir': None,
    'judge_model': '',
    'extraction_prompt_path': '',
    'baseline_extraction_prompt_path': '',
    'judge_prompt_path': '',
    'cache_path': None,
    'test_extract_only': False,
    'test_extract_baseline_only': False,
}

def resolve_run_path(run_id_or_path):
    """Resolve run ID or path to full path to results.json.

    Args:
        run_id_or_path: Run ID or full path

    Returns:
        Full path to results.json
    """
    if run_id_or_path is None or run_id_or_path == 'None':
        return None

    # Already a full path
    if '/' in run_id_or_path or run_id_or_path.endswith('.json'):
        return run_id_or_path

    # Try relative to repo root
    results_path = ROOT / 'eval' / 'results' / 'dataset_eval' / run_id_or_path / 'results.json'
    if results_path.exists():
        return str(results_path)

    # Fall back to simple relative path
    return f'eval/results/dataset_eval/{run_id_or_path}/results.json'

def load_judge_prompt(prompt_path: str) -> str:
    """Load judge prompt from file."""
    path = ROOT / prompt_path
    with open(path, 'r') as f:
        return f.read()

def format_object_mappings(mappings: List[Dict]) -> str:
    """Format object mappings for judge prompt."""
    formatted = []
    for m in mappings:
        formatted.append(f"• {m['source']} → {m['target']}")
        formatted.append(f"  Rationale: {m['mapping_rationale']}")
    return "\n".join(formatted)

def score_analogy_with_llm(
    analogy: Dict,
    problem: str,
    target_domain: str,
    client: Anthropic,
    model: str,
    prompt_template: str
) -> tuple:
    """
    Score a single analogy using LLM judge.

    Returns:
        (scores_dict, tokens_dict)
    """
    # Format the prompt
    object_mappings_str = format_object_mappings(analogy['object_mappings'])

    prompt = prompt_template.format(
        problem=problem,
        source_domain=target_domain,
        target_domain=analogy['target_domain'],
        object_mappings=object_mappings_str,
        shared_relations=analogy['shared_relations']
    )

    # Call API with retry logic
    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = client.messages.create(
                model=model,
                max_tokens=2000,
                temperature=0.0,
                messages=[{"role": "user", "content": prompt}]
            )

            # Extract response text
            response_text = response.content[0].text

            # Parse JSON
            # Find JSON object in response
            start = response_text.find('{')
            end = response_text.rfind('}') + 1
            if start >= 0 and end > start:
                json_str = response_text[start:end]
                scores = json.loads(json_str)
            else:
                scores = json.loads(response_text)

            # Track tokens
            tokens = {
                'input': response.usage.input_tokens,
                'output': response.usage.output_tokens
            }

            return scores, tokens

        except Exception as e:
            if attempt < max_retries - 1:
                print(f"    Retry {attempt + 1}/{max_retries} after error: {e}")
                time.sleep(2 ** attempt)  # Exponential backoff
            else:
                print(f"    ERROR: Failed after {max_retries} attempts: {e}")
                return {
                    'error': str(e),
                    'structural_depth': {'score': 0, 'explanation': 'Error'},
                    'domain_distance': {'score': 0, 'explanation': 'Error'},
                    'applicability': {'score': 0, 'explanation': 'Error'},
                    'novelty': {'score': 0, 'explanation': 'Error'},
                    'unexpectedness': {'score': 0, 'explanation': 'Error'},
                    'non_obviousness': {'score': 0, 'explanation': 'Error'},
                    'overall_assessment': f'Error: {e}'
                }, {'input': 0, 'output': 0}

def extract_ground_truth_analogy(
    paper: Dict,
    client: Anthropic,
    model: str,
    prompt_template: str
) -> tuple:
    """
    Extract structured analogy from ground truth paper.

    Returns:
        (analogy_dict, tokens_dict)
    """
    prompt = prompt_template.format(
        source_domain=paper['target_domain'],
        target_domain=paper['base_domain'],
        method_name=paper.get('method_name', 'Unknown'),
        analogy_description=paper['analogy_description'],
        concrete_example=paper.get('concrete_example', 'N/A')
    )

    # Call API with retry logic
    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = client.messages.create(
                model=model,
                max_tokens=2000,
                temperature=0.0,
                messages=[{"role": "user", "content": prompt}]
            )

            response_text = response.content[0].text

            # Parse JSON
            start = response_text.find('{')
            end = response_text.rfind('}') + 1
            if start >= 0 and end > start:
                json_str = response_text[start:end]
                analogy = json.loads(json_str)
            else:
                analogy = json.loads(response_text)

            tokens = {
                'input': response.usage.input_tokens,
                'output': response.usage.output_tokens
            }

            return analogy, tokens

        except Exception as e:
            if attempt < max_retries - 1:
                print(f"    Retry {attempt + 1}/{max_retries} after error: {e}")
                time.sleep(2 ** attempt)
            else:
                print(f"    ERROR: Failed to extract ground truth analogy: {e}")
                return None, {'input': 0, 'output': 0}

def load_or_extract_ground_truth(
    paper: Dict,
    cache_path: Path,
    client: Anthropic,
    model: str,
    extraction_prompt: str
) -> tuple:
    """
    Load ground truth analogy from cache or extract with LLM.

    Returns:
        (analogy_dict, tokens_dict, was_cached)
    """
    # Load cache if exists
    cache = {}
    if cache_path.exists():
        with open(cache_path, 'r') as f:
            cache = json.load(f)

    paper_title = paper.get('title') or paper.get('paper_title')

    # Check if already cached
    if paper_title in cache:
        print(f"  [Cached] Ground truth analogy for: {paper_title}")
        return cache[paper_title], {'input': 0, 'output': 0}, True

    # Extract with LLM
    print(f"  [Extracting] Ground truth analogy for: {paper_title}")
    analogy, tokens = extract_ground_truth_analogy(paper, client, model, extraction_prompt)

    if analogy is not None:
        # Save to cache
        cache[paper_title] = analogy
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_path, 'w') as f:
            json.dump(cache, f, indent=2)
        print(f"  [Saved to cache] {cache_path}")

    return analogy, tokens, False

def extract_baseline_analogy(
    solution: Dict,
    problem: str,
    problem_domain: str,
    client: Anthropic,
    model: str,
    prompt_template: str
) -> tuple:
    """Extract analogy structure from baseline solution description.

    Args:
        solution: Baseline solution dict with source_domain (where solution comes from)
        problem: Problem statement
        problem_domain: Domain where problem originates (source in mapping)

    Returns:
        (analogy_dict, tokens_dict)
    """
    # Handle missing fields gracefully (some baseline results may be incomplete)
    try:
        description = solution.get('description', 'N/A')
        key_concepts = solution.get('key_concepts', [])
        key_concepts_str = ', '.join(key_concepts) if key_concepts else 'N/A'
        source_domain = solution.get('source_domain', 'Unknown')
        title = solution.get('title', 'Unknown')

        prompt = prompt_template.format(
            problem=problem,
            source_domain=problem_domain,
            target_domain=source_domain,
            solution_title=title,
            description=description,
            key_concepts=key_concepts_str,
            relevance=solution.get('relevance', 'N/A')
        )
    except Exception as e:
        print(f"    ERROR: Failed to format prompt for baseline analogy: {e}")
        print(f"    Solution keys: {list(solution.keys())}")
        return None, {'input': 0, 'output': 0}

    # Same retry logic as extract_ground_truth_analogy
    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = client.messages.create(
                model=model,
                max_tokens=2000,
                temperature=0.0,
                messages=[{"role": "user", "content": prompt}]
            )

            response_text = response.content[0].text

            # Parse JSON
            start = response_text.find('{')
            end = response_text.rfind('}') + 1
            if start >= 0 and end > start:
                json_str = response_text[start:end]
                analogy = json.loads(json_str)
            else:
                analogy = json.loads(response_text)

            tokens = {
                'input': response.usage.input_tokens,
                'output': response.usage.output_tokens
            }

            return analogy, tokens

        except Exception as e:
            if attempt < max_retries - 1:
                print(f"    Retry {attempt + 1}/{max_retries} after error: {e}")
                time.sleep(2 ** attempt)
            else:
                print(f"    ERROR: Failed to extract baseline analogy: {e}")
                return None, {'input': 0, 'output': 0}

def load_or_extract_baseline_analogy(
    solution: Dict,
    problem: str,
    problem_domain: str,
    cache_path: Path,
    paper_title: str,
    attempt_num: int,
    client: Anthropic,
    model: str,
    extraction_prompt: str
) -> tuple:
    """Load baseline analogy from cache or extract with LLM.

    Returns:
        (analogy_dict, tokens_dict, was_cached)
    """
    # Load cache
    cache = {}
    if cache_path.exists():
        with open(cache_path, 'r') as f:
            cache = json.load(f)

    # Cache key: paper_title + attempt + solution title
    cache_key = f"{paper_title}_{attempt_num}_{solution['title']}"

    if cache_key in cache:
        return cache[cache_key], {'input': 0, 'output': 0}, True

    # Extract with LLM
    analogy, tokens = extract_baseline_analogy(solution, problem, problem_domain, client, model, extraction_prompt)

    if analogy is not None:
        cache[cache_key] = analogy
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_path, 'w') as f:
            json.dump(cache, f, indent=2)

    return analogy, tokens, False

def compute_statistics(scores: List[float]) -> Dict:
    """Compute statistics for a list of scores."""
    if not scores:
        return {'mean': 0, 'std': 0, 'min': 0, 'max': 0, 'median': 0}

    return {
        'mean': float(np.mean(scores)),
        'std': float(np.std(scores)),
        'min': float(np.min(scores)),
        'max': float(np.max(scores)),
        'median': float(np.median(scores))
    }

def main():
    import argparse

    parser = argparse.ArgumentParser(description='Compare AR/baseline analogies to ground truth')

    # Run paths (specify one to compare against ground truth)
    parser.add_argument('--ar-run', type=str, default=CONFIG['ar_run'],
                       help='Path or run ID for AR run (loads analogies directly)')
    parser.add_argument('--cross-domain-run', type=str, default=CONFIG['cross_domain_run'],
                       help='Path or run ID for cross-domain baseline run (extracts analogies from solutions)')
    parser.add_argument('--no-domain-run', type=str, default=CONFIG['no_domain_run'],
                       help='Path or run ID for no-domain baseline run (extracts analogies from solutions)')

    parser.add_argument('--dataset', default=CONFIG['dataset_path'], help='Path to dataset.json with ground truth')
    parser.add_argument('--paper-ids', type=str, default=CONFIG['paper_ids'],
                       help='Comma-separated list of original_paper_ids to process (optional, default: all)')
    parser.add_argument('--output', default=CONFIG['output_dir'], help='Output directory for comparison results (None = auto-generate)')
    parser.add_argument('--model', default=CONFIG['judge_model'], help='Model for judging')
    parser.add_argument('--test-extract-only', action='store_true', default=CONFIG['test_extract_only'],
                       help='Test mode: only extract ground truth, skip analogy scoring')
    parser.add_argument('--test-extract-baseline-only', action='store_true', default=CONFIG['test_extract_baseline_only'],
                       help='Test mode: only extract baseline analogies, skip scoring')
    args = parser.parse_args()

    # Parse paper IDs if provided
    paper_ids = None
    if args.paper_ids:
        # Handle both list and string formats
        if isinstance(args.paper_ids, list):
            paper_ids = args.paper_ids
        elif isinstance(args.paper_ids, str):
            paper_ids = [int(pid.strip()) for pid in args.paper_ids.split(',')]

    # Collect all specified runs
    runs_to_process = []

    if args.ar_run:
        runs_to_process.append({
            'path': resolve_run_path(args.ar_run),
            'type': 'ar',
            'label': 'AR',
            'is_baseline': False
        })

    if args.cross_domain_run:
        runs_to_process.append({
            'path': resolve_run_path(args.cross_domain_run),
            'type': 'cross_domain_baseline',
            'label': 'Cross-Domain Baseline',
            'is_baseline': True
        })

    if args.no_domain_run:
        runs_to_process.append({
            'path': resolve_run_path(args.no_domain_run),
            'type': 'no_domain_baseline',
            'label': 'No-Domain Baseline',
            'is_baseline': True
        })

    if not runs_to_process:
        print("ERROR: Must specify at least one of --ar-run, --cross-domain-run, or --no-domain-run")
        sys.exit(1)

    # Validate test mode compatibility
    if args.test_extract_baseline_only:
        # Only baseline runs should be specified
        has_ar = any(r['type'] == 'ar' for r in runs_to_process)
        has_baseline = any(r['is_baseline'] for r in runs_to_process)

        if has_ar and not has_baseline:
            print(f"\n{Y}ERROR: --test-extract-baseline-only requires baseline runs{R}")
            print(f"Specified run types: {[r['label'] for r in runs_to_process]}")
            print(f"Use --cross-domain-run or --no-domain-run instead of --ar-run")
            sys.exit(1)

        # Filter out AR runs if both are specified
        if has_ar and has_baseline:
            print(f"\n{Y}WARNING: Filtering out AR runs for baseline extraction test mode{R}")
            runs_to_process = [r for r in runs_to_process if r['is_baseline']]

    # In test modes, process all runs sequentially
    # In comparison mode, process all runs together for side-by-side comparison
    if args.test_extract_only or args.test_extract_baseline_only:
        # Test mode: process all runs
        if len(runs_to_process) > 1:
            print(f"\n{C}Processing {len(runs_to_process)} runs in test mode:{R}")
            for run in runs_to_process:
                print(f"  - {run['label']}")
            print()
    else:
        # Comparison mode: process all runs for multi-run comparison
        if len(runs_to_process) > 1:
            print(f"\n{C}Processing {len(runs_to_process)} runs for comparison:{R}")
            for run in runs_to_process:
                print(f"  - {run['label']}")
            print()

    # Load data from all runs
    all_runs_data = []
    for run_info in runs_to_process:
        results_path = run_info['path']
        run_type = run_info['type']
        run_label = run_info['label']
        expected_baseline = run_info['is_baseline']

        print(f"Loading {run_label}...")
        print(f"  Results: {results_path}")
        with open(results_path, 'r') as f:
            results = json.load(f)

        # Validate run type matches expectation
        actual_baseline = results['config'].get('baseline', {}).get('enabled', False)

        if expected_baseline and not actual_baseline:
            print(f"\n{Y}ERROR: Expected baseline results but got AR results{R}")
            print(f"  Specified: {run_label}")
            print(f"  Actual: AR results (baseline.enabled=false)")
            sys.exit(1)
        elif not expected_baseline and actual_baseline:
            print(f"\n{Y}ERROR: Expected AR results but got baseline results{R}")
            print(f"  Specified: {run_label}")
            print(f"  Actual: Baseline results (baseline.enabled=true)")
            sys.exit(1)

        # Setup baseline cache path for this run
        if expected_baseline:
            run_results_dir = Path(results_path).parent
            run_baseline_cache_path = run_results_dir / "structured_baseline_analogies.json"
        else:
            run_baseline_cache_path = None

        # Store run data
        all_runs_data.append({
            'info': run_info,
            'results': results,
            'baseline_cache_path': run_baseline_cache_path
        })
        print()

    with open(args.dataset, 'r') as f:
        dataset_raw = json.load(f)
        # Handle both list format and dict with 'papers' key
        if isinstance(dataset_raw, list):
            dataset = dataset_raw
        elif isinstance(dataset_raw, dict) and 'papers' in dataset_raw:
            dataset = dataset_raw['papers']
        else:
            raise ValueError(f"Unexpected dataset format: {type(dataset_raw)}")

    # Initialize API client
    client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

    # Load prompts
    extraction_prompt = load_judge_prompt(CONFIG['extraction_prompt_path'])
    baseline_extraction_prompt = load_judge_prompt(CONFIG['baseline_extraction_prompt_path'])
    judge_prompt = load_judge_prompt(CONFIG['judge_prompt_path'])

    # Setup ground truth cache
    if CONFIG['cache_path']:
        cache_path = Path(CONFIG['cache_path'])
    else:
        dataset_dir = Path(args.dataset).parent
        cache_path = dataset_dir / "structured_ground_truth_analogies.json"

    # Determine output directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    if args.output is None:
        # Auto-generate: eval/results/analogy_comparison/{timestamp}/
        output_dir = ROOT / 'eval' / 'results' / 'analogy_comparison' / timestamp
    else:
        # User-specified directory
        output_dir = Path(args.output)

    output_dir.mkdir(parents=True, exist_ok=True)

    # Set up terminal output logging
    class TeeOutput:
        """Write to both terminal and log file."""
        def __init__(self, log_path):
            self.terminal = sys.stdout
            self.log = open(log_path, 'w', encoding='utf-8')

        def write(self, message):
            self.terminal.write(message)
            if not self.log.closed:
                self.log.write(message)

        def flush(self):
            self.terminal.flush()
            if not self.log.closed:
                self.log.flush()

        def close(self):
            if not self.log.closed:
                self.log.close()

    # Redirect stdout to both terminal and log file
    log_path = output_dir / "terminal_output.log"
    tee = TeeOutput(log_path)
    sys.stdout = tee

    # Print header
    print(f"\n{'='*100}")
    print(f"{C}MULTI-RUN ANALOGY COMPARISON{R}")
    print(f"{'='*100}")
    print(f"Timestamp: {timestamp}")
    print(f"Runs: {', '.join([r['info']['label'] for r in all_runs_data])}")
    print(f"Papers: {len(all_runs_data[0]['results']['paper_results']) if all_runs_data else 0}")
    print(f"Judge Model: {args.model}")
    print(f"Output directory: {output_dir}")
    print(f"Ground truth cache: {cache_path}")
    for run_data in all_runs_data:
        if run_data['baseline_cache_path']:
            print(f"{run_data['info']['label']} cache: {run_data['baseline_cache_path']}")
            print(f"  Cache exists: {run_data['baseline_cache_path'].exists()}")

    if args.test_extract_only:
        print(f"\n{'='*100}")
        print(f"{Y}TEST MODE: Extract ground truth only (no AR analogy scoring){R}")
        print(f"{'='*100}\n")
    else:
        print(f"{'='*100}")

    # Track tokens and costs
    total_tokens = {
        'extraction': {'input': 0, 'output': 0},
        'extraction_baseline': {},
        'judge': {},
        'judge_gt': {'input': 0, 'output': 0}
    }

    start_time = time.time()

    # Validate all runs have same papers
    first_titles = [pr['paper_title'] for pr in all_runs_data[0]['results']['paper_results']]
    for run_data in all_runs_data[1:]:
        titles = [pr['paper_title'] for pr in run_data['results']['paper_results']]
        if titles != first_titles:
            print(f"\n{Y}ERROR: Runs have different papers{R}")
            sys.exit(1)

    # Get paper count from first run (all runs should have same papers)
    if not all_runs_data:
        print(f"\n{Y}ERROR: No runs loaded{R}")
        sys.exit(1)

    first_results = all_runs_data[0]['results']
    num_papers = len(first_results['paper_results'])

    # Get original paper indices from first run config
    original_paper_indices = None
    if 'config' in first_results and 'evaluation' in first_results['config']:
        original_paper_indices = first_results['config']['evaluation'].get('paper_indices')

    # Show paper ID filter if specified
    if paper_ids:
        print(f"\n{C}Filtering to paper IDs: {paper_ids}{R}\n")

    # Helper functions for processing runs
    def process_baseline_run(paper_result, problem, target_domain, paper_title,
                             cache_path, client, model, extraction_prompt,
                             judge_prompt, total_tokens, run_type):
        """Extract and score baseline analogies from solutions."""
        analogies_scored = []

        for attempt in paper_result['attempts']:
            try:
                attempt_num = attempt['attempt_num']
                solution = attempt.get('best_solution') or (
                    attempt['evaluations'][0] if attempt.get('evaluations') else None
                )

                if not solution:
                    print(f"    Attempt {attempt_num}: No solution found, skipping")
                    continue

                # Extract or load from cache
                analogy, extract_tokens, was_cached = load_or_extract_baseline_analogy(
                    solution, problem, target_domain, cache_path, paper_title,
                    attempt_num, client, model, extraction_prompt
                )

                if not analogy:
                    print(f"    Attempt {attempt_num}: Failed to extract analogy, skipping")
                    continue

                # Skip if not an analogy (no object mappings)
                if not analogy.get('object_mappings') or len(analogy['object_mappings']) == 0:
                    print(f"    Attempt {attempt_num}: No object mappings found, skipping")
                    continue

                # Track tokens
                if run_type not in total_tokens['extraction_baseline']:
                    total_tokens['extraction_baseline'][run_type] = {'input': 0, 'output': 0}

                total_tokens['extraction_baseline'][run_type]['input'] += extract_tokens['input']
                total_tokens['extraction_baseline'][run_type]['output'] += extract_tokens['output']

                if not was_cached:
                    time.sleep(1)

                # Score
                scores, judge_tokens = score_analogy_with_llm(
                    analogy, problem, target_domain, client, model, judge_prompt
                )

                if run_type not in total_tokens['judge']:
                    total_tokens['judge'][run_type] = {'input': 0, 'output': 0}

                total_tokens['judge'][run_type]['input'] += judge_tokens['input']
                total_tokens['judge'][run_type]['output'] += judge_tokens['output']

                analogies_scored.append({
                    'attempt_num': attempt_num,
                    'analogy': analogy,
                    'scores': scores
                })
                time.sleep(1)

            except Exception as e:
                print(f"    ERROR in attempt {attempt.get('attempt_num', '?')}: {str(e)}")
                print(f"    Continuing with next attempt...")
                continue

        return analogies_scored

    def process_ar_run(paper_result, problem, target_domain, client, model,
                       judge_prompt, total_tokens, run_type):
        """Score AR analogies (already extracted)."""
        analogies_scored = []

        for attempt in paper_result['attempts']:
            try:
                attempt_num = attempt['attempt_num']
                analogies = attempt.get('analogies', [])

                if not analogies:
                    print(f"    Attempt {attempt_num}: No analogies found, skipping")
                    continue

                for idx, analogy in enumerate(analogies):
                    try:
                        scores, judge_tokens = score_analogy_with_llm(
                            analogy, problem, target_domain, client, model, judge_prompt
                        )

                        if run_type not in total_tokens['judge']:
                            total_tokens['judge'][run_type] = {'input': 0, 'output': 0}

                        total_tokens['judge'][run_type]['input'] += judge_tokens['input']
                        total_tokens['judge'][run_type]['output'] += judge_tokens['output']

                        analogies_scored.append({
                            'attempt_num': attempt_num,
                            'analogy_index': idx,
                            'analogy': analogy,
                            'scores': scores
                        })
                        time.sleep(1)

                    except Exception as e:
                        print(f"    ERROR scoring analogy {idx} in attempt {attempt_num}: {str(e)}")
                        print(f"    Continuing with next analogy...")
                        continue

            except Exception as e:
                print(f"    ERROR in attempt {attempt.get('attempt_num', '?')}: {str(e)}")
                print(f"    Continuing with next attempt...")
                continue

        return analogies_scored

    def compute_run_statistics(analogies_scored):
        """Compute statistics for a run's analogies."""
        if not analogies_scored:
            return {
                'num_analogies': 0,
                'structural_depth': compute_statistics([]),
                'domain_distance': compute_statistics([]),
                'applicability': compute_statistics([]),
                'novelty': compute_statistics([]),
                'unexpectedness': compute_statistics([]),
                'non_obviousness': compute_statistics([])
            }

        valid = [a for a in analogies_scored if 'error' not in a['scores']]
        return {
            'num_analogies': len(analogies_scored),
            'structural_depth': compute_statistics([a['scores']['structural_depth']['score'] for a in valid]),
            'domain_distance': compute_statistics([a['scores']['domain_distance']['score'] for a in valid]),
            'applicability': compute_statistics([a['scores']['applicability']['score'] for a in valid]),
            'novelty': compute_statistics([a['scores']['novelty']['score'] for a in valid]),
            'unexpectedness': compute_statistics([a['scores']['unexpectedness']['score'] for a in valid]),
            'non_obviousness': compute_statistics([a['scores']['non_obviousness']['score'] for a in valid])
        }

    def format_score_with_color(score, comparison_score):
        """Format score with green color if it beats comparison."""
        score_str = f"{score:.1f}"
        padding = ' ' * (5 - len(score_str))
        if score > comparison_score:
            return f"{G}{score_str}{R}{padding}"
        return f"{score_str}{padding}"

    # Process each paper across all runs
    per_paper_results = []
    extracted_papers = []  # Track extracted papers in test mode

    for paper_idx in range(num_papers):
        try:
            # Get paper info from first run (should be same across all runs)
            paper_result = first_results['paper_results'][paper_idx]
            paper_title = paper_result['paper_title']
            base_domain = paper_result['base_domain']
            target_domain = paper_result['target_domain']

            # Use preprocessed/rewritten problem for fair comparison
            # Both GT and AR are judged against the same problem statement
            if 'preprocessing' in paper_result and paper_result['preprocessing']:
                problem_to_use = paper_result['preprocessing'].get('rewritten_problem', paper_result['problem'])
            else:
                # No preprocessing - use original problem
                problem_to_use = paper_result['problem']

            # Find matching paper in dataset
            dataset_paper = None
            for p in dataset:
                if p['title'] == paper_title:
                    dataset_paper = p
                    break

            if dataset_paper is None:
                print(f"\n{'=' * 100}")
                print(f"{Y}WARNING: Paper not found in dataset, skipping{R}")
                print(f"{'=' * 100}\n")
                continue

            # Get original paper ID
            if original_paper_indices and paper_idx < len(original_paper_indices):
                paper_id = original_paper_indices[paper_idx]
            else:
                paper_id = dataset_paper.get('id', paper_idx)

            # Skip if not in requested paper_ids
            if paper_ids and paper_id not in paper_ids:
                continue

            # Print paper header with metadata
            print(f"\n{'=' * 100}")
            print(f"{B}PAPER ID {paper_id}: {paper_title}{R}")
            print(f"{'=' * 100}")
            print(f"Analogy: {base_domain} → {target_domain}")

            # Add metadata if available
            metadata_parts = []
            if 'difficulty' in dataset_paper:
                metadata_parts.append(f"Difficulty: {dataset_paper['difficulty']}")
            if 'well_known' in dataset_paper:
                metadata_parts.append(f"Well-known: {dataset_paper['well_known']}")
            if 'year' in dataset_paper:
                metadata_parts.append(f"Year: {dataset_paper['year']}")

            if metadata_parts:
                print(" | ".join(metadata_parts))
            print()

            # Extract ground truth analogy
            gt_analogy, extract_tokens, was_cached = load_or_extract_ground_truth(
                dataset_paper, cache_path, client, args.model, extraction_prompt
            )

            if gt_analogy is None:
                print(f"  ERROR: Failed to extract ground truth, skipping\n")
                continue

            total_tokens['extraction']['input'] += extract_tokens['input']
            total_tokens['extraction']['output'] += extract_tokens['output']

            if not was_cached:
                time.sleep(1)  # Rate limiting

            # In test mode, just print the extracted structure and continue
            if args.test_extract_only:
                print(f"  ✓ Extracted ground truth structure:")
                print(f"    - Domain: {gt_analogy['target_domain']}")
                print(f"    - Object mappings: {len(gt_analogy['object_mappings'])}")
                print(f"    - Shared relations: {len(gt_analogy['shared_relations'])} chars")
                print()

                extracted_papers.append({
                    'paper_title': paper_title,
                    'base_domain': base_domain,
                    'extracted_analogy': gt_analogy,
                    'was_cached': was_cached
                })
                continue

            # Score ground truth
            print(f"  Scoring ground truth analogy...")
            print(f"    Problem: {problem_to_use[:100]}...")
            gt_scores, gt_tokens = score_analogy_with_llm(
                gt_analogy, problem_to_use, target_domain, client, args.model, judge_prompt
            )
            total_tokens['judge_gt']['input'] += gt_tokens['input']
            total_tokens['judge_gt']['output'] += gt_tokens['output']
            time.sleep(1)

            # Process all runs for this paper
            paper_runs_data = {}

            for run_data in all_runs_data:
                run_info = run_data['info']
                run_label = run_info['label']
                run_type = run_info['type']
                is_baseline = run_info['is_baseline']
                results = run_data['results']
                baseline_cache_path = run_data['baseline_cache_path']

                # Get this run's paper result
                paper_result = results['paper_results'][paper_idx]

                print(f"\n  Processing {run_label}...")

                # Extract and score analogies
                if is_baseline:
                    analogies_scored = process_baseline_run(
                        paper_result, problem_to_use, target_domain, paper_title,
                        baseline_cache_path, client, args.model,
                        baseline_extraction_prompt, judge_prompt,
                        total_tokens, run_type
                    )
                else:
                    analogies_scored = process_ar_run(
                        paper_result, problem_to_use, target_domain,
                        client, args.model, judge_prompt,
                        total_tokens, run_type
                    )

                # Compute statistics
                run_stats = compute_run_statistics(analogies_scored)

                # Store results
                paper_runs_data[run_type] = {
                    'label': run_label,
                    'analogies': analogies_scored,
                    'statistics': run_stats
                }

            # Store complete paper results
            per_paper_results.append({
                'paper_id': paper_id,
                'paper_title': paper_title,
                'ground_truth': {
                    'analogy': gt_analogy,
                    'scores': gt_scores
                },
                'runs': paper_runs_data
            })

            # Print per-attempt score table
            print()  # Add newline before table
            print(f"{C}ANALOGY QUALITY SCORES{R}")
            print(f"{'Type':<25} | {'Att':<3} | {'Domain':<20} | {'SD':<5} | {'DD':<5} | {'AP':<5} | {'NV':<5} | {'UN':<5} | {'NO':<5}")
            print(f"{'-'*25}-|-{'-'*3}-|-{'-'*20}-|-{'-'*5}-|-{'-'*5}-|-{'-'*5}-|-{'-'*5}-|-{'-'*5}-|-{'-'*5}")

            # Get GT scores for comparison
            paper_data = per_paper_results[-1]
            gt_scores = paper_data['ground_truth']['scores']
            gt_vals = {m: gt_scores[m]['score'] for m in ['structural_depth', 'domain_distance',
                       'applicability', 'novelty', 'unexpectedness', 'non_obviousness']}

            # Print all individual analogies first
            for run_type, run_data in paper_runs_data.items():
                run_label = run_data['label']

                # Individual analogies
                for analogy in run_data['analogies']:
                    if 'error' in analogy['scores']:
                        continue
                    att = analogy['attempt_num']
                    domain = analogy['analogy']['target_domain'][:20]
                    s = analogy['scores']
                    print(f"{run_label:<25} | {att:<3} | {domain:<20} | "
                          f"{s['structural_depth']['score']:<5.1f} | {s['domain_distance']['score']:<5.1f} | "
                          f"{s['applicability']['score']:<5.1f} | {s['novelty']['score']:<5.1f} | "
                          f"{s['unexpectedness']['score']:<5.1f} | {s['non_obviousness']['score']:<5.1f}")

            # Print separator before averaged rows
            print(f"{'-'*25}-|-{'-'*3}-|-{'-'*20}-|-{'-'*5}-|-{'-'*5}-|-{'-'*5}-|-{'-'*5}-|-{'-'*5}-|-{'-'*5}")

            # Calculate max values across all approaches (including GT) for each metric
            max_vals = {}
            for m in gt_vals.keys():
                all_vals = [gt_vals[m]]
                for run_data in paper_runs_data.values():
                    if run_data['statistics']['num_analogies'] > 0:
                        all_vals.append(run_data['statistics'][m]['mean'])
                max_vals[m] = max(all_vals)

            # Helper function to format score with green highlighting if it's the max
            def format_score_max(score, max_score):
                score_str = f"{score:.1f}"
                padding = ' ' * (5 - len(score_str))
                if score == max_score:
                    return f"{G}{score_str}{R}{padding}"
                return f"{score_str}{padding}"

            # Print all averaged rows (skip if no analogies found)
            for run_type, run_data in paper_runs_data.items():
                stats = run_data['statistics']
                # Skip average row if no analogies were found for this run
                if stats['num_analogies'] == 0:
                    continue
                avg_label = f"{run_data['label']} Avg"
                print(f"{avg_label:<25} | {'-':<3} | {'-':<20} | "
                      f"{format_score_max(stats['structural_depth']['mean'], max_vals['structural_depth'])} | "
                      f"{format_score_max(stats['domain_distance']['mean'], max_vals['domain_distance'])} | "
                      f"{format_score_max(stats['applicability']['mean'], max_vals['applicability'])} | "
                      f"{format_score_max(stats['novelty']['mean'], max_vals['novelty'])} | "
                      f"{format_score_max(stats['unexpectedness']['mean'], max_vals['unexpectedness'])} | "
                      f"{format_score_max(stats['non_obviousness']['mean'], max_vals['non_obviousness'])}")

            # GT row
            gt_domain = gt_analogy['target_domain'][:20]
            print(f"{'GT':<25} | {'-':<3} | {gt_domain:<20} | "
                  f"{format_score_max(gt_vals['structural_depth'], max_vals['structural_depth'])} | "
                  f"{format_score_max(gt_vals['domain_distance'], max_vals['domain_distance'])} | "
                  f"{format_score_max(gt_vals['applicability'], max_vals['applicability'])} | "
                  f"{format_score_max(gt_vals['novelty'], max_vals['novelty'])} | "
                  f"{format_score_max(gt_vals['unexpectedness'], max_vals['unexpectedness'])} | "
                  f"{format_score_max(gt_vals['non_obviousness'], max_vals['non_obviousness'])}")

            # Performance summary
            print(f"\n{C}PERFORMANCE SUMMARY{R}")
            print("-" * 100)

            # Show number of valid analogies found per run
            print(f"\n{C}Valid Analogies Found:{R}")
            for run_type, run_data in paper_runs_data.items():
                num_analogies = run_data['statistics']['num_analogies']
                print(f"{run_data['label']}: {num_analogies} analogies")

            metrics = ['structural_depth', 'domain_distance', 'applicability', 'novelty', 'unexpectedness', 'non_obviousness']
            print(f"\n{C}Metric Wins vs GT:{R}")
            for run_type, run_data in paper_runs_data.items():
                wins = sum(1 for m in metrics if run_data['statistics'][m]['mean'] > gt_vals[m])
                print(f"{run_data['label']}: {wins}/{len(metrics)} metrics beat GT")

            # Best approach
            best_label = max(paper_runs_data.values(),
                             key=lambda x: sum(1 for m in metrics if x['statistics'][m]['mean'] > gt_vals[m]))['label']
            best_wins = max(sum(1 for m in metrics if run_data['statistics'][m]['mean'] > gt_vals[m])
                            for run_data in paper_runs_data.values())
            print(f"\nBest approach: {best_label} ({best_wins}/{len(metrics)} wins)\n")

        except Exception as e:
            print(f"\n{Y}ERROR processing paper {paper_idx + 1}/{num_papers}: {str(e)}{R}")
            print(f"Paper title: {paper_result.get('paper_title', 'Unknown')}")
            print(f"Skipping this paper and continuing with next one...\n")
            import traceback
            traceback.print_exc()
            continue

    # Handle test mode separately
    if args.test_extract_only:
        runtime = time.time() - start_time

        # Calculate extraction cost
        extraction_cost = (total_tokens['extraction']['input'] * 3.0 / 1_000_000 +
                          total_tokens['extraction']['output'] * 15.0 / 1_000_000)

        print(f"{'=' * 70}")
        print("TEST MODE SUMMARY - Ground Truth Extraction")
        print(f"{'=' * 70}\n")
        if paper_ids:
            print(f"Paper IDs filter: {paper_ids}")
        print(f"Papers processed: {len(extracted_papers)}")
        print(f"Successfully extracted: {len([p for p in extracted_papers if not p['was_cached']])}")
        print(f"Loaded from cache: {len([p for p in extracted_papers if p['was_cached']])}")
        print(f"\nExtraction tokens: {total_tokens['extraction']['input']} input / {total_tokens['extraction']['output']} output")
        print(f"Extraction cost: ${extraction_cost:.4f} USD")
        print(f"Runtime: {runtime:.1f}s ({runtime/60:.1f} min)")
        print(f"\nCache file: {cache_path}")
        print(f"{'=' * 70}\n")

        # Save test results
        test_results = {
            'run_id': timestamp,
            'mode': 'test_extract_only',
            'paper_ids_filter': paper_ids,
            'extracted_papers': extracted_papers,
            'cache_path': str(cache_path),
            'extraction_tokens': total_tokens['extraction'],
            'extraction_cost_usd': extraction_cost,
            'runtime_seconds': runtime
        }

        output_file = output_dir / "test_extraction_results.json"
        with open(output_file, 'w') as f:
            json.dump(test_results, f, indent=2)

        print(f"Test results saved to: {output_file}\n")

        # Close log file and exit
        if hasattr(sys.stdout, 'close'):
            sys.stdout.flush()
            sys.stdout.close()
        return

    # Check for baseline extraction test mode
    if args.test_extract_baseline_only:
        print(f"\n{'='*100}")
        print(f"{Y}TEST MODE: Extract baseline analogies only (no scoring){R}")
        print(f"{'='*100}\n")

        # Process each baseline run
        all_runs_extracted = []

        for run_idx, run_info in enumerate(runs_to_process):
            if run_idx > 0:
                print(f"\n{'='*100}")
                print(f"{C}Processing run {run_idx + 1}/{len(runs_to_process)}: {run_info['label']}{R}")
                print(f"{'='*100}\n")
            else:
                print(f"{C}Processing run 1/{len(runs_to_process)}: {run_info['label']}{R}\n")

            # Load this run's results
            run_results_path = run_info['path']
            with open(run_results_path, 'r') as f:
                run_results = json.load(f)

            # Validate this is a baseline run
            is_baseline = run_results['config'].get('baseline', {}).get('enabled', False)

            if not is_baseline:
                print(f"\n{Y}ERROR: Run {run_info['label']} is not a baseline run{R}")
                print(f"The results do not have baseline.enabled=true in config")
                continue

            # Setup cache for this run
            run_results_dir = Path(run_results_path).parent
            run_baseline_cache_path = run_results_dir / "structured_baseline_analogies.json"
            print(f"Using cache: {run_baseline_cache_path}")
            print(f"Cache exists: {run_baseline_cache_path.exists()}\n")

            # Get original paper indices from results config
            original_paper_indices = None
            if 'config' in run_results and 'evaluation' in run_results['config']:
                original_paper_indices = run_results['config']['evaluation'].get('paper_indices')

            if paper_ids and run_idx == 0:
                print(f"Filtering to paper IDs: {paper_ids}\n")

            extracted_baseline = []

            # Track analogy statistics for no-domain baseline
            analogy_stats = {
                'total_solutions': 0,
                'solutions_with_analogies': 0,
                'solutions_without_analogies': 0
            } if run_info['type'] == 'no_domain_baseline' else None

            for paper_idx, paper_result in enumerate(run_results['paper_results']):
                # Get original paper ID
                if original_paper_indices and paper_idx < len(original_paper_indices):
                    original_paper_id = original_paper_indices[paper_idx]
                else:
                    original_paper_id = paper_idx

                # Skip if not in requested paper_ids
                if paper_ids and original_paper_id not in paper_ids:
                    continue
                paper_title = paper_result['paper_title']
                target_domain = paper_result['target_domain']

                # Use preprocessed problem if available
                if 'preprocessing' in paper_result and paper_result['preprocessing']:
                    problem_to_use = paper_result['preprocessing'].get('rewritten_problem', paper_result['problem'])
                else:
                    problem_to_use = paper_result['problem']

                print(f"\n{'=' * 100}")
                print(f"{B}PAPER {paper_idx + 1}: {paper_title}{R}")
                print(f"{'=' * 100}")

                # Extract baseline analogies for all attempts
                paper_analogies = []
                for attempt in paper_result['attempts']:
                    attempt_num = attempt['attempt_num']

                    # Get solution from baseline results
                    # Try best_solution first, then evaluations array
                    solution = attempt.get('best_solution')
                    if solution is None and 'evaluations' in attempt and attempt['evaluations']:
                        # Get the top-ranked solution from evaluations
                        evaluations = attempt['evaluations']
                        solution = evaluations[0]  # Already sorted by rank

                    if solution is None:
                        print(f"  Attempt {attempt_num}: No solution, skipping")
                        continue

                    # Extract or load analogy
                    analogy, extract_tokens, was_cached = load_or_extract_baseline_analogy(
                        solution, problem_to_use, target_domain,
                        run_baseline_cache_path, paper_title, attempt_num,
                        client, args.model, baseline_extraction_prompt
                    )

                    if analogy is None:
                        print(f"  Attempt {attempt_num}: ERROR - Failed to extract from '{solution['title'][:60]}...'")
                        continue

                    # Show cache status
                    cache_status = "[Cached]" if was_cached else "[Extracted]"
                    print(f"  Attempt {attempt_num}: {cache_status} '{solution['title'][:60]}...'")

                    total_tokens['extraction']['input'] += extract_tokens['input']
                    total_tokens['extraction']['output'] += extract_tokens['output']

                    # Track analogy statistics for no-domain baseline
                    if analogy_stats is not None:
                        analogy_stats['total_solutions'] += 1
                        has_analogy = len(analogy.get('object_mappings', [])) > 0
                        if has_analogy:
                            analogy_stats['solutions_with_analogies'] += 1
                        else:
                            analogy_stats['solutions_without_analogies'] += 1

                    paper_analogies.append({
                        'attempt_num': attempt_num,
                        'solution_title': solution['title'],
                        'solution_domain': solution['source_domain'],
                        'analogy': analogy,
                        'has_analogy': len(analogy.get('object_mappings', [])) > 0,
                        'was_cached': was_cached
                    })

                    if not was_cached:
                        time.sleep(1)

                extracted_baseline.append({
                    'paper_title': paper_title,
                    'target_domain': target_domain,
                    'num_attempts': len(paper_analogies),
                    'analogies': paper_analogies
                })

                print(f"  ✓ Extracted {len(paper_analogies)} baseline analogies")

            # Store results for this run
            run_total_extracted = sum(len(p['analogies']) for p in extracted_baseline)
            run_total_cached = sum(sum(1 for a in p['analogies'] if a['was_cached']) for p in extracted_baseline)

            run_data = {
                'run_label': run_info['label'],
                'run_type': run_info['type'],
                'source_results': str(run_results_path),
                'cache_path': str(run_baseline_cache_path),
                'papers_processed': len(extracted_baseline),
                'total_extracted': run_total_extracted,
                'total_cached': run_total_cached,
                'extracted_baseline': extracted_baseline
            }

            # Add analogy statistics for no-domain baseline
            if analogy_stats is not None:
                run_data['analogy_statistics'] = analogy_stats

            all_runs_extracted.append(run_data)

            print(f"\n{C}Summary for {run_info['label']}:{R}")
            print(f"  Papers: {len(extracted_baseline)}")
            print(f"  Analogies: {run_total_extracted} ({run_total_extracted - run_total_cached} new, {run_total_cached} cached)")

            # Display analogy statistics for no-domain baseline
            if analogy_stats is not None and analogy_stats['total_solutions'] > 0:
                pct_with = analogy_stats['solutions_with_analogies'] / analogy_stats['total_solutions'] * 100
                pct_without = analogy_stats['solutions_without_analogies'] / analogy_stats['total_solutions'] * 100
                print(f"\n  {Y}Analogy Statistics (No-Domain Baseline):{R}")
                print(f"    Solutions with analogies: {analogy_stats['solutions_with_analogies']}/{analogy_stats['total_solutions']} ({pct_with:.1f}%)")
                print(f"    Solutions without analogies: {analogy_stats['solutions_without_analogies']}/{analogy_stats['total_solutions']} ({pct_without:.1f}%)")

            print(f"  Cache: {run_baseline_cache_path}\n")

        # End of runs loop - print overall summary
        runtime = time.time() - start_time
        extraction_cost = (total_tokens['extraction']['input'] * 3.0 / 1_000_000 +
                          total_tokens['extraction']['output'] * 15.0 / 1_000_000)

        print(f"\n{'=' * 100}")
        print(f"{C}TEST MODE SUMMARY - Baseline Analogy Extraction{R}")
        print(f"{'=' * 100}")
        if paper_ids:
            print(f"Paper IDs filter: {paper_ids}")
        print(f"Runs processed: {len(all_runs_extracted)}")
        total_all_extracted = sum(r['total_extracted'] for r in all_runs_extracted)
        total_all_cached = sum(r['total_cached'] for r in all_runs_extracted)
        print(f"Total analogies extracted across all runs: {total_all_extracted}")
        print(f"  - Newly extracted: {total_all_extracted - total_all_cached}")
        print(f"  - Loaded from cache: {total_all_cached}")
        print(f"\nExtraction tokens: {total_tokens['extraction']['input']} input / {total_tokens['extraction']['output']} output")
        print(f"Extraction cost: ${extraction_cost:.4f} USD")
        print(f"Runtime: {runtime:.1f}s ({runtime/60:.1f} min)")

        # Show analogy statistics for no-domain runs
        no_domain_runs = [r for r in all_runs_extracted if r['run_type'] == 'no_domain_baseline' and 'analogy_statistics' in r]
        if no_domain_runs:
            print(f"\n{Y}No-Domain Baseline Analogy Statistics:{R}")
            for run in no_domain_runs:
                stats = run['analogy_statistics']
                if stats['total_solutions'] > 0:
                    pct_with = stats['solutions_with_analogies'] / stats['total_solutions'] * 100
                    print(f"  {run['run_label']}: {stats['solutions_with_analogies']}/{stats['total_solutions']} solutions have analogies ({pct_with:.1f}%)")

        print(f"{'=' * 100}\n")

        # Save test results
        test_results = {
            'run_id': timestamp,
            'mode': 'test_extract_baseline_only',
            'paper_ids_filter': paper_ids,
            'runs': all_runs_extracted,
            'extraction_tokens': total_tokens['extraction'],
            'extraction_cost_usd': extraction_cost,
            'runtime_seconds': runtime
        }

        output_file = output_dir / "test_baseline_extraction_results.json"
        with open(output_file, 'w') as f:
            json.dump(test_results, f, indent=2)

        print(f"Test results saved to: {output_file}\n")

        # Close log and exit
        if hasattr(sys.stdout, 'close'):
            sys.stdout.flush()
            sys.stdout.close()
        return

    # Compute aggregate statistics (average of paper averages)
    # Collect per-run statistics across papers
    metrics = ['structural_depth', 'domain_distance', 'applicability', 'novelty', 'unexpectedness', 'non_obviousness']

    run_types = set()
    for paper in per_paper_results:
        run_types.update(paper['runs'].keys())

    per_run_stats = {rt: {m: [] for m in metrics} for rt in run_types}
    gt_stats = {m: [] for m in metrics}
    wins_papers = {rt: {m: 0 for m in metrics} for rt in run_types}

    # Track paper-level wins per metric (which approach had highest score on each paper for each metric)
    paper_level_wins_per_metric = {m: {rt: 0 for rt in run_types} for m in metrics}
    for m in metrics:
        paper_level_wins_per_metric[m]['gt'] = 0
    papers_with_comparisons_per_metric = {m: 0 for m in metrics}

    for paper in per_paper_results:
        gt_scores = paper['ground_truth']['scores']
        for m in metrics:
            gt_stats[m].append(gt_scores[m]['score'])

        for run_type, run_data in paper['runs'].items():
            stats = run_data['statistics']
            # Only include papers where analogies were actually found
            if stats['num_analogies'] == 0:
                continue

            for m in metrics:
                run_mean = stats[m]['mean']
                per_run_stats[run_type][m].append(run_mean)

                # Track which approach won this metric on this specific paper
                if run_mean > gt_scores[m]['score']:
                    wins_papers[run_type][m] += 1

        # For each metric, determine which approach had the highest score on this paper
        for m in metrics:
            # Collect scores from all approaches that have analogies for this paper
            metric_scores = {'gt': gt_scores[m]['score']}

            for run_type, run_data in paper['runs'].items():
                stats = run_data['statistics']
                if stats['num_analogies'] > 0:
                    metric_scores[run_type] = stats[m]['mean']

            # Only count if at least one approach had analogies
            if len(metric_scores) > 1:
                papers_with_comparisons_per_metric[m] += 1
                # Find winner for this metric on this paper (only if no tie)
                max_score = max(metric_scores.values())
                # Count how many approaches have the max score
                approaches_with_max = [approach for approach, score in metric_scores.items() if score == max_score]
                # Only give the win if there's exactly one winner (no tie)
                if len(approaches_with_max) == 1:
                    paper_level_wins_per_metric[m][approaches_with_max[0]] += 1

    # Calculate costs (Sonnet 4.5 pricing: $3/MTok input, $15/MTok output)
    input_cost_per_mtok = 3.0
    output_cost_per_mtok = 15.0

    cost_breakdown = {
        'extraction_gt': (total_tokens['extraction']['input'] * input_cost_per_mtok / 1_000_000 +
                         total_tokens['extraction']['output'] * output_cost_per_mtok / 1_000_000),
        'judge_gt': (total_tokens['judge_gt']['input'] * input_cost_per_mtok / 1_000_000 +
                    total_tokens['judge_gt']['output'] * output_cost_per_mtok / 1_000_000)
    }

    # Add extraction costs for baseline runs
    for run_type, tokens in total_tokens.get('extraction_baseline', {}).items():
        cost_breakdown[f'extraction_{run_type}'] = (
            tokens['input'] * input_cost_per_mtok / 1_000_000 +
            tokens['output'] * output_cost_per_mtok / 1_000_000
        )

    # Add judge costs for all runs
    for run_type, tokens in total_tokens.get('judge', {}).items():
        cost_breakdown[f'judge_{run_type}'] = (
            tokens['input'] * input_cost_per_mtok / 1_000_000 +
            tokens['output'] * output_cost_per_mtok / 1_000_000
        )

    cost_breakdown['total'] = sum(cost_breakdown.values())

    runtime = time.time() - start_time

    # Build final results
    comparison_results = {
        'run_id': timestamp,
        'config': {
            'runs': [r['info'] for r in all_runs_data],
            'dataset_path': args.dataset,
            'judge_model': args.model,
            'num_papers': len(per_paper_results),
            'cache_path': str(cache_path)
        },
        'per_paper_results': per_paper_results,
        'wins_count_papers': wins_papers,
        'tokens': total_tokens,
        'cost_breakdown_usd': cost_breakdown,
        'runtime_seconds': runtime
    }

    # Save results
    output_file = output_dir / "comparison_results.json"
    with open(output_file, 'w') as f:
        json.dump(comparison_results, f, indent=2)

    # Count papers with valid analogies per run and total analogies
    papers_with_analogies = {}
    total_analogies = {}
    for rt in run_types:
        papers_with_analogies[rt] = sum(1 for p in per_paper_results
                                        if rt in p['runs'] and p['runs'][rt]['statistics']['num_analogies'] > 0)
        total_analogies[rt] = sum(p['runs'][rt]['statistics']['num_analogies']
                                 for p in per_paper_results if rt in p['runs'])

    print(f"\n{'='*150}")
    print(f"{C}AGGREGATE STATISTICS ACROSS {len(per_paper_results)} PAPERS{R}")
    print(f"{C}(Only papers with valid analogies included in means){R}")
    print(f"{'='*150}")
    # Show count of papers with analogies per run
    for rt in sorted(run_types):
        label = next(p['runs'][rt]['label'] for p in per_paper_results if rt in p['runs'])
        print(f"{label}: {papers_with_analogies[rt]}/{len(per_paper_results)} papers with analogies, {total_analogies[rt]} total analogies")
    print()

    # Build header dynamically
    metric_display = [
        ('Structural Depth ↑', 'structural_depth'),
        ('Domain Distance ↑', 'domain_distance'),
        ('Applicability ↑', 'applicability'),
        ('Novelty ↑', 'novelty'),
        ('Unexpectedness ↑', 'unexpectedness'),
        ('Non-Obviousness ↑', 'non_obviousness')
    ]

    headers = ['Metric']
    run_labels = []
    for rt in sorted(run_types):
        label = next(p['runs'][rt]['label'] for p in per_paper_results if rt in p['runs'])
        run_labels.append(label)
        headers.append(f"{label} Mean")
    headers.extend(['GT Mean', 'Best'])

    # Calculate column widths dynamically
    col_widths = [22]  # Metric column
    for label in run_labels:
        col_widths.append(max(len(f"{label} Mean"), 8))  # Min width 8 for numbers
    col_widths.append(8)  # GT Mean
    col_widths.append(max(max(len(label) for label in run_labels), 4))  # Best column

    # Print header
    print(' | '.join(f"{h:<{w}}" for h, w in zip(headers, col_widths)))
    print('-|-'.join(['-'*w for w in col_widths]))

    # Print metrics
    for display, key in metric_display:
        # Collect all means for this metric
        run_means = {}
        for rt in sorted(run_types):
            if per_run_stats[rt][key]:  # Only if there are values
                mean = np.mean(per_run_stats[rt][key])
                run_means[rt] = mean
            else:
                run_means[rt] = 0

        gt_mean = np.mean(gt_stats[key])

        # Find max value across all approaches (including GT)
        all_means = list(run_means.values()) + [gt_mean]
        max_mean = max(all_means)

        # Build row with highlighting for max values
        row_parts = [f"{display:<{col_widths[0]}}"]
        col_idx = 1

        # Run means with highlighting
        for rt in sorted(run_types):
            mean = run_means[rt]
            mean_str = f"{mean:.2f}"
            padded = f"{mean_str:<{col_widths[col_idx]}}"
            if mean == max_mean:
                row_parts.append(f"{G}{padded}{R}")
            else:
                row_parts.append(padded)
            col_idx += 1

        # GT mean with highlighting
        gt_str = f"{gt_mean:.2f}"
        padded = f"{gt_str:<{col_widths[col_idx]}}"
        if gt_mean == max_mean:
            row_parts.append(f"{G}{padded}{R}")
        else:
            row_parts.append(padded)
        col_idx += 1

        # Best column (find who has the max)
        best_rt, best_val = max(run_means.items(), key=lambda x: x[1])
        best_label = next(p['runs'][best_rt]['label'] for p in per_paper_results if best_rt in p['runs'])
        if best_val >= gt_mean:
            best_plain = best_label
        else:
            best_plain = "GT"
        best_padding = ' ' * (col_widths[-1] - len(best_plain))
        row_parts.append(f"{G}{best_plain}{R}{best_padding}")

        # Print row
        print(' | '.join(row_parts))

    # Delta table (Run - GT)
    print(f"\n{C}DELTA FROM GT (Run - GT){R}")
    print("-" * 150)

    # Build delta table
    delta_headers = ['Metric']
    for rt in sorted(run_types):
        label = next(p['runs'][rt]['label'] for p in per_paper_results if rt in p['runs'])
        delta_headers.append(f"Δ({label}-GT)")

    # Calculate delta column widths dynamically based on header lengths
    # Need more space for percentage display
    delta_col_widths = [22]  # Metric column
    for header in delta_headers[1:]:
        delta_col_widths.append(max(len(header), 20))  # At least 20 for delta + percentage

    # Print delta header
    print(' | '.join(f"{h:<{w}}" for h, w in zip(delta_headers, delta_col_widths)))
    print('-|-'.join(['-'*w for w in delta_col_widths]))

    # Print delta rows
    for display, key in metric_display:
        gt_mean = np.mean(gt_stats[key])

        # Build row parts with proper padding before adding color codes
        row_parts = [f"{display:<{delta_col_widths[0]}}"]
        col_idx = 1

        for rt in sorted(run_types):
            if per_run_stats[rt][key]:  # Only if there are values
                run_mean = np.mean(per_run_stats[rt][key])
                delta = run_mean - gt_mean

                # Calculate percentage change
                if gt_mean != 0:
                    pct_change = (delta / gt_mean) * 100
                    pct_str = f"({pct_change:+.1f}%)"
                else:
                    pct_str = "(N/A)"

                # Format: delta + percentage
                delta_str = f"{delta:+.2f}" if delta != 0 else "0.00"
                combined_str = f"{delta_str} {pct_str}"
                padded = f"{combined_str:<{delta_col_widths[col_idx]}}"

                # Color positive deltas green, negative deltas yellow
                if delta > 0:
                    row_parts.append(f"{G}{padded}{R}")
                elif delta < 0:
                    row_parts.append(f"{Y}{padded}{R}")
                else:
                    row_parts.append(padded)
            else:
                row_parts.append(f"{'N/A':<{delta_col_widths[col_idx]}}")
            col_idx += 1

        # Print row
        print(' | '.join(row_parts))

    # Paper-level win rates per metric (what % of papers each approach had highest score for each metric)
    print(f"\n{C}PAPER-LEVEL WIN RATES PER METRIC{R}")
    print(f"{C}(% of papers where approach had highest average score for that metric){R}")
    print("-" * 150)

    # Build dynamic header with all run types + GT
    plw_headers = ['Metric']
    for rt in sorted(run_types):
        label = next(p['runs'][rt]['label'] for p in per_paper_results if rt in p['runs'])
        plw_headers.append(f"{label} WR")
    plw_headers.append('GT WR')

    # Calculate column widths dynamically based on header lengths
    plw_col_widths = [22]  # Metric column
    for header in plw_headers[1:]:
        plw_col_widths.append(max(len(header), 12))  # At least 12, or header length

    # Print header
    print(' | '.join(f"{h:<{w}}" for h, w in zip(plw_headers, plw_col_widths)))
    print('-|-'.join(['-'*w for w in plw_col_widths]))

    # Print rows
    for display, key in metric_display:
        num_papers = papers_with_comparisons_per_metric[key]

        # Collect all win percentages to find the max
        win_percentages = {}
        for rt in sorted(run_types):
            wins = paper_level_wins_per_metric[key][rt]
            win_pct = wins / num_papers * 100 if num_papers > 0 else 0
            win_percentages[rt] = win_pct

        # GT win rate
        gt_wins = paper_level_wins_per_metric[key]['gt']
        gt_win_pct = gt_wins / num_papers * 100 if num_papers > 0 else 0
        win_percentages['gt'] = gt_win_pct

        # Find the maximum win percentage
        max_win_pct = max(win_percentages.values())

        # Build row - pad first, then add colors
        row_parts = [f"{display:<{plw_col_widths[0]}}"]
        col_idx = 1

        for rt in sorted(run_types):
            win_pct = win_percentages[rt]
            pct_str = f"{win_pct:.1f}%"
            padded = f"{pct_str:<{plw_col_widths[col_idx]}}"
            if win_pct == max_win_pct and win_pct > 0:
                row_parts.append(f"{G}{padded}{R}")
            else:
                row_parts.append(padded)
            col_idx += 1

        # GT win rate with highlighting
        pct_str = f"{gt_win_pct:.1f}%"
        padded = f"{pct_str:<{plw_col_widths[col_idx]}}"
        if gt_win_pct == max_win_pct and gt_win_pct > 0:
            row_parts.append(f"{G}{padded}{R}")
        else:
            row_parts.append(padded)

        # Print row
        print(' | '.join(row_parts))

    print(f"\n{'=' * 150}")
    print(f"{C}EXECUTION SUMMARY{R}")
    print(f"{'=' * 150}")
    print(f"Total Cost: ${cost_breakdown['total']:.2f} USD")
    for key, cost in sorted(cost_breakdown.items()):
        if key != 'total':
            print(f"  - {key}: ${cost:.2f}")
    print(f"Runtime: {runtime:.1f}s ({runtime/60:.1f} min)")
    print(f"Results saved to: {output_file}")
    print(f"{'=' * 150}")

    # Close log file
    if hasattr(sys.stdout, 'close'):
        sys.stdout.flush()
        sys.stdout.close()

if __name__ == '__main__':
    main()
