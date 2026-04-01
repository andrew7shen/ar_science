"""
Evaluate analogous reasoning system against dataset of ground truth papers.

Configuration Settings Used:
  - judge_model: Model for LLM judge evaluations
  - dataset_path: Path to dataset JSON file
  - evaluation.num_papers: Number of papers to evaluate (null = all)
  - evaluation.paper_indices: Specific paper indices to evaluate (e.g., [15, 16])
  - evaluation.num_attempts_per_paper: Attempts per paper
  - evaluation.num_solutions_per_attempt: Solutions per attempt
  - evaluation.match_threshold: Score threshold for exact match
  - agents.*: Agent enable/disable settings
  - model.*: Model settings for workflow
  - extraction.*: Extraction agent settings
  - search.*: Search agent settings (including abstraction_level, use_llm_fallback)
  - output.*: Output settings (including save_runs)

Token Tracking:
  - Workflow tokens: extraction + search (no assessment)
  - Judge tokens: LLM judge API calls
  - All tokens logged per run with separate cost breakdown

Usage:
  python eval/dataset_eval/evaluate_on_papers.py \\
    --config eval/dataset_eval/eval_config.yaml \\
    --output eval/results/dataset_eval/
"""

import json
import argparse
import sys
import time
import os
import yaml
from pathlib import Path
from datetime import datetime
import numpy as np
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

# Add src to path
ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))

from dotenv import load_dotenv
# Load environment variables BEFORE other imports so API keys are available
load_dotenv(ROOT / ".env")

from anthropic import Anthropic
from config import config as config_singleton
from orchestrator import run_workflow

# Import preprocessing utilities
sys.path.insert(0, str(ROOT / "eval" / "dataset_eval"))
from preprocessing import preprocess_paper

# Global lock for config singleton mutations (prevents race conditions in parallel execution)
_config_lock = threading.Lock()


# ANSI color codes
def _init_colors():
    """Initialize color codes based on config."""
    # Always use colors for dataset eval
    return {
        'C': '\033[96m',  # Cyan
        'B': '\033[94m',  # Blue
        'G': '\033[92m',  # Green
        'Y': '\033[93m',  # Yellow
        'M': '\033[95m',  # Magenta
        'R': '\033[0m'    # Reset
    }


class ThreadSafeTokenCounter:
    """Thread-safe counter for tracking tokens across parallel evaluations."""
    def __init__(self):
        self.lock = threading.Lock()
        self.workflow = {'input': 0, 'output': 0}
        self.judge = {
            'input': 0,
            'output': 0,
            'cache_creation_input_tokens': 0,
            'cache_read_input_tokens': 0
        }
        self.domain_judge = {
            'input': 0,
            'output': 0
        }

    def add_workflow_tokens(self, input_tokens: int, output_tokens: int):
        with self.lock:
            self.workflow['input'] += input_tokens
            self.workflow['output'] += output_tokens

    def add_judge_tokens(self, tokens: dict):
        # For solution matching (Sonnet)
        with self.lock:
            self.judge['input'] += tokens.get('input', 0)
            self.judge['output'] += tokens.get('output', 0)
            self.judge['cache_creation_input_tokens'] += tokens.get('cache_creation_input_tokens', 0)
            self.judge['cache_read_input_tokens'] += tokens.get('cache_read_input_tokens', 0)

    def add_domain_judge_tokens(self, tokens: dict):
        # For domain matching (Haiku)
        with self.lock:
            self.domain_judge['input'] += tokens.get('input', 0)
            self.domain_judge['output'] += tokens.get('output', 0)

    def to_dict(self) -> dict:
        """Convert to standard dict for cost calculation."""
        with self.lock:
            return {
                'workflow': self.workflow.copy(),
                'judge': self.judge.copy(),
                'domain_judge': self.domain_judge.copy()
            }


class ThreadSafePrinter:
    """Thread-safe printer that prevents garbled output."""
    def __init__(self):
        self.lock = threading.Lock()

    def print(self, *args, **kwargs):
        with self.lock:
            print(*args, **kwargs)


class TeeOutput:
    """Write to both terminal and file simultaneously."""
    def __init__(self, file_path, original_stdout):
        self.file = open(file_path, 'w')
        self.stdout = original_stdout

    def write(self, message):
        self.stdout.write(message)
        self.file.write(message)
        self.file.flush()

    def flush(self):
        self.stdout.flush()
        self.file.flush()

    def close(self):
        self.file.close()


def load_eval_config(config_path=None) -> dict:
    """Load evaluation configuration from YAML file."""
    if config_path is None:
        config_path = ROOT / "eval" / "dataset_eval" / "eval_config.yaml"

    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def load_dataset(dataset_path: Path) -> list:
    """Load dataset and filter to papers with analogical reasoning."""
    with open(dataset_path, 'r') as f:
        data = json.load(f)

    # Filter to papers using analogical reasoning
    ar_papers = [
        p for p in data['papers']
        if p.get('uses_analogical_reasoning', False)
    ]

    return ar_papers


def load_datasets_multi(datasets_config: list) -> list:
    """Load papers from multiple datasets with per-dataset filtering.

    Args:
        datasets_config: List of dataset configs, each with:
            - path: Path to dataset JSON
            - paper_indices: Optional list of indices (1-based, AR-filtered)

    Returns:
        List of paper dicts with added metadata:
            - _source_dataset: Dataset path for provenance tracking
            - _source_index: Original index in source dataset (1-based)
    """
    all_papers = []

    for ds_config in datasets_config:
        dataset_path = Path(ds_config['path'])
        paper_indices = ds_config.get('paper_indices')

        # Load dataset and filter to AR papers
        ar_papers = load_dataset(dataset_path)

        # Select papers by indices if specified
        if paper_indices is not None:
            selected = []
            for idx in paper_indices:
                if 1 <= idx <= len(ar_papers):
                    paper = ar_papers[idx - 1].copy()
                    paper['_source_dataset'] = str(dataset_path)
                    paper['_source_index'] = idx
                    selected.append(paper)
                else:
                    print(f"  Warning: Index {idx} out of range for {dataset_path.name} (max: {len(ar_papers)})")
        else:
            # Use all AR papers from this dataset
            selected = []
            for idx, paper in enumerate(ar_papers, 1):
                paper_copy = paper.copy()
                paper_copy['_source_dataset'] = str(dataset_path)
                paper_copy['_source_index'] = idx
                selected.append(paper_copy)

        all_papers.extend(selected)
        print(f"  Loaded {len(selected)} papers from {dataset_path.name}")

    return all_papers


def apply_config_overrides(eval_config: dict):
    """Apply eval config overrides to main config singleton."""
    cfg = config_singleton.get_all()

    # Apply agent settings
    if 'agents' in eval_config:
        for agent_name, agent_settings in eval_config['agents'].items():
            if isinstance(agent_settings, dict):
                # Agent configuration (extraction, search, assessment, etc.)
                if agent_name not in cfg['agents']:
                    cfg['agents'][agent_name] = {}
                cfg['agents'][agent_name].update(agent_settings)
            else:
                # Non-dict settings (like domain_judge_stop_on_first_match)
                cfg['agents'][agent_name] = agent_settings

    # Apply model settings
    if 'model' in eval_config:
        cfg['model'].update(eval_config['model'])

    # Apply extraction settings
    if 'extraction' in eval_config:
        cfg['extraction'].update(eval_config['extraction'])

    # Apply search settings
    if 'search' in eval_config:
        cfg['search'].update(eval_config['search'])

    # Apply baseline settings
    if 'baseline' in eval_config:
        cfg['baseline'].update(eval_config['baseline'])

    # Apply output settings
    if 'output' in eval_config:
        cfg['output'].update(eval_config['output'])

    # Apply assessment settings
    if 'assessment' in eval_config:
        cfg['assessment'].update(eval_config['assessment'])


def apply_paper_config(paper: dict, eval_config: dict, verbose: bool = True):
    """
    Apply config overrides for a paper. Must be called ONCE per paper before running attempts.

    Args:
        paper: Paper data
        eval_config: Evaluation configuration
        verbose: Whether to print configuration info
    """
    with _config_lock:
        # Apply config overrides
        apply_config_overrides(eval_config)

        # Additional overrides for this paper
        cfg = config_singleton.get_all()

        # Get evaluation mode first to determine how to handle preprocessing
        eval_mode = eval_config.get('evaluation', {}).get('mode', 'solution_search')

        # Determine base_domain for mode-specific config
        # In solution_search mode, use paper's original base_domain (ground truth)
        # In domain_search mode, we'll discover domains naturally
        base_domain = paper['base_domain']

        # Check if baseline mode is enabled
        baseline_enabled = eval_config.get('baseline', {}).get('enabled', False)

        if eval_mode == 'domain_search':
            # ========== DOMAIN SEARCH MODE ==========
            # System discovers domains naturally - NO override
            # Tests domain discovery + solution quality
            if verbose:
                print(f"  Evaluation Mode: DOMAIN SEARCH (discovering domains naturally)")

            if baseline_enabled:
                cfg['baseline']['num_solutions_per_domain'] = eval_config['evaluation'].get('num_solutions_per_domain', 10)
                # Only clear override_domains if not explicitly set in eval_config
                if eval_config.get('baseline', {}).get('override_domains') is None:
                    cfg['baseline']['override_domains'] = None
            else:
                cfg['search']['num_solutions_per_domain'] = eval_config['evaluation'].get('num_solutions_per_domain', 10)
                # Only clear override_domains if not explicitly set in eval_config
                if eval_config.get('extraction', {}).get('override_domains') is None:
                    cfg['extraction']['override_domains'] = None

        elif eval_mode == 'solution_search':
            # ========== SOLUTION SEARCH MODE ==========
            # Domain constrained to ground truth - WITH override
            # Tests solution quality only (original behavior)
            if verbose:
                print(f"  Evaluation Mode: SOLUTION SEARCH (constraining to ground truth domain: {base_domain})")

            if baseline_enabled:
                # Only set to base_domain if not explicitly set in eval_config
                if eval_config.get('baseline', {}).get('override_domains') is None:
                    cfg['baseline']['override_domains'] = [base_domain]
                cfg['baseline']['num_solutions_per_domain'] = eval_config['evaluation'].get('num_solutions_per_attempt', 10)
            else:
                # Only set to base_domain if not explicitly set in eval_config
                if eval_config.get('extraction', {}).get('override_domains') is None:
                    cfg['extraction']['override_domains'] = [base_domain]
                cfg['search']['num_solutions_per_domain'] = eval_config['evaluation'].get('num_solutions_per_attempt', 10)

        else:
            raise ValueError(f"Unknown evaluation mode: {eval_mode}. Must be 'solution_search' or 'domain_search'")


def run_constrained_workflow(paper: dict, attempt_num: int, eval_config: dict, verbose: bool = True) -> tuple:
    """
    Run workflow constrained to paper's base_domain (or with natural domain discovery).

    NOTE: Config must be applied BEFORE calling this function (via apply_config_overrides).
    This function does NOT hold the config lock, enabling parallel execution.

    Args:
        paper: Paper data
        attempt_num: Current attempt number
        eval_config: Evaluation configuration
        verbose: Whether to print workflow progress (set False for parallel execution)

    Returns:
        (solutions, tokens_dict, preprocessing_info, discovered_domains, analogies)
    """
    try:
        # Get evaluation mode first to determine how to handle preprocessing
        eval_mode = eval_config.get('evaluation', {}).get('mode', 'solution_search')

        # Apply preprocessing if enabled
        preprocessing_enabled = eval_config.get('preprocessing', {}).get('remove_domain_hints', False)

        # Skip preprocessing for custom questions (user controls exact input)
        if paper.get('_source_dataset') == 'custom_question':
            preprocessing_enabled = False
            if verbose:
                print("  Skipping preprocessing for custom question (user controls exact input)")

        if preprocessing_enabled:
            paper_to_use = preprocess_paper(paper, eval_config, verbose=verbose)
            problem = paper_to_use['problem']
            # IMPORTANT: In solution_search mode, use ORIGINAL base_domain (ground truth)
            # In domain_search mode, use extracted/preprocessed base_domain
            if eval_mode == 'solution_search':
                base_domain = paper['base_domain']  # Use original ground truth domain
            else:
                base_domain = paper_to_use['base_domain']  # Use preprocessed/extracted domain
        else:
            paper_to_use = paper
            problem = paper['problem']
            base_domain = paper['base_domain']

        # Get domain judge settings
        domain_judge_client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
        domain_judge_model = eval_config.get('domain_judge_model', eval_config.get('judge_model'))

        if verbose:
            print(f"  Domain judge configured: client={domain_judge_client is not None}, model={domain_judge_model}")

        # Run workflow WITHOUT lock (config already applied by caller)
        abstraction_level = eval_config['search']['abstraction_level']
        baseline_enabled = eval_config.get('baseline', {}).get('enabled', False)

        state = run_workflow(problem, abstraction_level=abstraction_level, verbose=verbose, ground_truth_domain=base_domain, domain_judge_client=domain_judge_client, domain_judge_model=domain_judge_model)

        # Extract solutions and tokens
        solutions = state.get('solutions', [])
        metrics = state.get('metrics', {})

        tokens = {
            'input': metrics.get('total_input', 0),
            'output': metrics.get('total_output', 0),
            'domain_judge_input': metrics.get('domain_judge_input', 0),
            'domain_judge_output': metrics.get('domain_judge_output', 0)
        }

        # Track preprocessing info if applied
        preprocessing_info = None
        if preprocessing_enabled and '_original_problem' in paper_to_use:
            preprocessing_info = {
                'original_problem': paper_to_use['_original_problem'],
                'rewritten_problem': paper_to_use['problem'],
                'base_domain': paper_to_use['base_domain']
            }

        # Extract discovered domains from state
        discovered_domains = state.get('extraction', {}).get('target_domains', [])
        if baseline_enabled:
            # For baseline, domains come from _discover_domains() call
            discovered_domains = state.get('baseline_domains', [])

        # Extract analogies if save_analogies flag is enabled
        analogies = state.get('extraction', {}).get('analogies', [])

        return solutions, tokens, preprocessing_info, discovered_domains, analogies

    except Exception as e:
        if verbose:
            import traceback
            print(f"  ERROR: Workflow failed: {str(e)}")
            print(f"  Traceback:")
            traceback.print_exc()
        return [], {'input': 0, 'output': 0, 'domain_judge_input': 0, 'domain_judge_output': 0}, None, [], []


def evaluate_domain_match(discovered_domain: str, ground_truth_domain: str, client: Anthropic, judge_model: str, c: dict) -> dict:
    """
    Use LLM judge to evaluate if discovered domain matches ground truth domain.

    Args:
        discovered_domain: Domain discovered by the system
        ground_truth_domain: Ground truth domain from dataset
        client: Anthropic client
        judge_model: Model to use for judging
        c: Color codes

    Returns:
        {
            'is_match': bool,
            'explanation': str,
            'tokens': {'input': int, 'output': int}
        }
    """
    # Load prompt template
    prompt_path = ROOT / "eval" / "prompts" / "domain_match_judge.txt"
    with open(prompt_path, 'r') as f:
        template = f.read()

    # Format prompt
    prompt = template.format(
        ground_truth_domain=ground_truth_domain,
        discovered_domain=discovered_domain
    )

    try:
        response = client.messages.create(
            model=judge_model,
            max_tokens=100,
            temperature=0,
            messages=[{"role": "user", "content": prompt}]
        )

        # Check if response has content
        if not hasattr(response, 'content') or not response.content:
            tokens = {
                'input': getattr(response.usage, 'input_tokens', 0) if hasattr(response, 'usage') else 0,
                'output': getattr(response.usage, 'output_tokens', 0) if hasattr(response, 'usage') else 0
            }
            return {
                'is_match': False,
                'explanation': 'API refused to respond',
                'tokens': tokens,
                'error': True
            }

        response_text = response.content[0].text.strip()

        # Parse Yes/No response - handle markdown formatting (bold/italic)
        # Different judge prompts may put verdict at beginning or end
        # Check both first and last few lines
        lines = [line.strip() for line in response_text.split('\n') if line.strip()]

        # Try to find Yes/No verdict
        is_match = False
        verdict_found = False

        # First check the first few lines (some prompts put verdict at start)
        for line in lines[:5]:
            cleaned_line = line.lstrip('*').strip().lower()
            if cleaned_line == 'yes' or cleaned_line.startswith('yes'):
                is_match = True
                verdict_found = True
                break
            elif cleaned_line == 'no' or cleaned_line.startswith('no'):
                is_match = False
                verdict_found = True
                break

        # If not found at start, check the last few lines (some prompts put verdict at end)
        if not verdict_found:
            for line in reversed(lines[-5:]):
                cleaned_line = line.lstrip('*').strip().lower()
                if cleaned_line == 'yes' or cleaned_line.startswith('yes'):
                    is_match = True
                    break
                elif cleaned_line == 'no' or cleaned_line.startswith('no'):
                    is_match = False
                    break

        return {
            'is_match': is_match,
            'explanation': response_text,
            'tokens': {
                'input': response.usage.input_tokens,
                'output': response.usage.output_tokens
            }
        }

    except Exception as e:
        print(f"    {c['Y']}Warning: Domain match evaluation failed: {str(e)}{c['R']}")
        return {
            'is_match': False,
            'explanation': f'Evaluation failed: {str(e)}',
            'tokens': {'input': 0, 'output': 0},
            'error': True
        }


def evaluate_solution_match_single(generated_solution: dict, paper: dict, client: Anthropic, judge_model: str, judge_prompt_path: str, c: dict, attempt_num: int = 1) -> dict:
    """
    Use LLM judge to evaluate if generated solution matches ground truth using a single scoring method.

    Args:
        generated_solution: Solution dict from workflow
        paper: Ground truth paper dict
        client: Anthropic client
        judge_model: Model to use for judging
        judge_prompt_path: Path to judge prompt template
        c: Color codes
        attempt_num: Attempt number (for per-attempt caching)

    Returns:
        {
            'is_match': bool,
            'explanation': str,
            'tokens': {'input': int, 'output': int, 'cache_creation_input_tokens': int, 'cache_read_input_tokens': int}
        }
    """
    # Load prompt template
    prompt_path = ROOT / judge_prompt_path
    with open(prompt_path, 'r') as f:
        template = f.read()

    # Check if template has caching markers
    use_caching = '---GROUND_TRUTH_MARKER---' in template and '---GENERATED_SOLUTION_MARKER---' in template

    # Note: Domain fields removed from most prompts to focus on method matching regardless of domain
    if 'relaxed' in judge_prompt_path:
        # Relaxed prompt uses old format with domains and justification
        prompt = template.format(
            generated_title=generated_solution.get('title', 'Unknown'),
            generated_domain=generated_solution.get('source_domain', 'Unknown'),
            generated_description=generated_solution.get('description', ''),
            generated_concepts=', '.join(generated_solution.get('key_concepts', [])),
            ground_truth_title=paper['title'],
            ground_truth_domain=paper['base_domain'],
            ground_truth_justification=paper['analogy_justification']
        )
    elif 'analogical_structure' in judge_prompt_path:
        # Analogical structure prompt needs base_domain and target_domain
        prompt = template.format(
            generated_title=generated_solution.get('title', 'Unknown'),
            source_domain=generated_solution.get('source_domain', 'Unknown'),
            generated_description=generated_solution.get('description', ''),
            generated_concepts=', '.join(generated_solution.get('key_concepts', [])),
            ground_truth_title=paper['title'],
            base_domain=paper['base_domain'],
            target_domain=paper['target_domain'],
            ground_truth_method=paper['analogy_description']
        )
    else:
        # Use new format without domains for exact/principle prompts
        prompt = template.format(
            generated_title=generated_solution.get('title', 'Unknown'),
            generated_description=generated_solution.get('description', ''),
            generated_concepts=', '.join(generated_solution.get('key_concepts', [])),
            ground_truth_title=paper['title'],
            ground_truth_method=paper['analogy_description']
        )

    # Create API call with or without caching based on prompt structure
    if use_caching:
        # Split prompt into cacheable blocks using markers
        parts = prompt.split('---GROUND_TRUTH_MARKER---')
        criteria_section = parts[0].strip()

        remaining = parts[1].split('---GENERATED_SOLUTION_MARKER---')
        ground_truth_section = remaining[0].strip()
        generated_solution_section = remaining[1].strip()

        # Use multi-block message with caching
        # Cache the criteria (static) and ground truth (static per paper)
        # Don't cache the generated solution (changes per solution)
        response = client.messages.create(
            model=judge_model,
            max_tokens=1000,
            temperature=0,
            messages=[{
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": criteria_section,
                        "cache_control": {"type": "ephemeral"}
                    },
                    {
                        "type": "text",
                        "text": ground_truth_section,
                        "cache_control": {"type": "ephemeral"}
                    },
                    {
                        "type": "text",
                        "text": generated_solution_section
                    }
                ]
            }]
        )
    else:
        # Use simple single-string prompt (no caching)
        response = client.messages.create(
            model=judge_model,
            max_tokens=1000,  # Increased from 300 to allow full step-by-step reasoning + verdict
            temperature=0,
            messages=[{"role": "user", "content": prompt}]
        )

    try:
        # Check if response has content (API may refuse for safety reasons)
        if not hasattr(response, 'content') or not response.content:
            # Track actual token usage even for refusals
            tokens = {
                'input': getattr(response.usage, 'input_tokens', 0) if hasattr(response, 'usage') else 0,
                'output': getattr(response.usage, 'output_tokens', 0) if hasattr(response, 'usage') else 0,
                'cache_creation_input_tokens': getattr(response.usage, 'cache_creation_input_tokens', 0) if hasattr(response, 'usage') else 0,
                'cache_read_input_tokens': getattr(response.usage, 'cache_read_input_tokens', 0) if hasattr(response, 'usage') else 0
            }
            stop_reason = getattr(response, 'stop_reason', 'unknown')
            # Print concise warning without full response object
            print(f"    {c['Y']}Warning: API refusal (stop_reason: {stop_reason}){c['R']}")
            return {
                'match_score': 0.0,
                'is_exact_match': False,
                'explanation': f'API refused to respond (stop_reason: {stop_reason})',
                'tokens': tokens,
                'error': True
            }

        response_text = response.content[0].text.strip()

        # Parse Yes/No response - handle markdown formatting (bold/italic)
        # Different judge prompts may put verdict at beginning or end
        # Check both first and last few lines
        lines = [line.strip() for line in response_text.split('\n') if line.strip()]

        # Try to find Yes/No verdict
        is_match = False
        verdict_found = False

        # First check the first few lines (some prompts put verdict at start)
        for line in lines[:5]:
            cleaned_line = line.lstrip('*').strip().lower()
            if cleaned_line == 'yes' or cleaned_line.startswith('yes'):
                is_match = True
                verdict_found = True
                break
            elif cleaned_line == 'no' or cleaned_line.startswith('no'):
                is_match = False
                verdict_found = True
                break

        # If not found at start, check the last few lines (some prompts put verdict at end)
        if not verdict_found:
            for line in reversed(lines[-5:]):
                cleaned_line = line.lstrip('*').strip().lower()
                if cleaned_line == 'yes' or cleaned_line.startswith('yes'):
                    is_match = True
                    break
                elif cleaned_line == 'no' or cleaned_line.startswith('no'):
                    is_match = False
                    break

        return {
            'is_match': is_match,
            'explanation': response_text,
            'tokens': {
                'input': response.usage.input_tokens,
                'output': response.usage.output_tokens,
                'cache_creation_input_tokens': getattr(response.usage, 'cache_creation_input_tokens', 0),
                'cache_read_input_tokens': getattr(response.usage, 'cache_read_input_tokens', 0)
            }
        }

    except Exception as e:
        print(f"    {c['Y']}Warning: LLM judge failed: {str(e)}{c['R']}")
        print(f"    {c['Y']}Solution title: {generated_solution.get('title', 'Unknown')}{c['R']}")
        print(f"    {c['Y']}Solution fields: {list(generated_solution.keys())}{c['R']}")
        return {
            'is_match': False,
            'explanation': f'Evaluation failed: {str(e)}',
            'tokens': {'input': 0, 'output': 0, 'cache_creation_input_tokens': 0, 'cache_read_input_tokens': 0},
            'error': True
        }


def extract_method_name(prompt_path: str) -> str:
    """Extract method name from prompt filename.

    Examples:
        "eval/prompts/dataset_eval_match_exact.txt" -> "exact"
        "eval/prompts/dataset_eval_match_exact_cached.txt" -> "exact"
        "eval/prompts/dataset_eval_match_relaxed.txt" -> "relaxed"
    """
    from pathlib import Path
    filename = Path(prompt_path).stem  # e.g., "dataset_eval_match_exact" or "dataset_eval_match_exact_cached"
    # Remove "dataset_eval_match_" prefix if present
    if filename.startswith("dataset_eval_match_"):
        method_name = filename[len("dataset_eval_match_"):]
        # Remove "_cached" suffix if present
        if method_name.endswith("_cached"):
            method_name = method_name[:-len("_cached")]
        return method_name
    return filename


def evaluate_solution_match(generated_solution: dict, paper: dict, client: Anthropic, judge_model: str, judge_prompts: list, c: dict, attempt_num: int = 1) -> dict:
    """
    Evaluate solution using configured scoring methods.

    Args:
        generated_solution: Solution dict from workflow
        paper: Ground truth paper dict
        client: Anthropic client
        judge_model: Model to use for judging
        judge_prompts: List of prompt paths
        c: Color codes
        attempt_num: Attempt number (for per-attempt caching)

    Returns:
        {
            'is_{method}_match': bool (for each method),
            '{method}_explanation': str (for each method),
            'tokens': {'input': int, 'output': int, 'cache_creation_input_tokens': int, 'cache_read_input_tokens': int}
        }
    """
    results = {}
    total_tokens = {'input': 0, 'output': 0, 'cache_creation_input_tokens': 0, 'cache_read_input_tokens': 0}

    for prompt_path in judge_prompts:
        method_name = extract_method_name(prompt_path)
        eval_result = evaluate_solution_match_single(generated_solution, paper, client, judge_model, prompt_path, c, attempt_num)

        # Store match result
        results[f'is_{method_name}_match'] = eval_result['is_match']
        results[f'{method_name}_explanation'] = eval_result['explanation']

        # Accumulate tokens
        for key in total_tokens.keys():
            total_tokens[key] += eval_result['tokens'].get(key, 0)

        # Propagate error flag if present
        if eval_result.get('error', False):
            results['error'] = True

    results['tokens'] = total_tokens
    return results


def evaluate_attempt(solutions: list, paper: dict, client: Anthropic, judge_model: str, judge_prompts: list, total_tokens: dict, c: dict, attempt_num: int = 1) -> dict:
    """
    Evaluate all solutions from one attempt against ground truth using configured scoring methods.

    Returns:
        {
            'is_{method}_match': bool (for each method),
            'best_solution': dict,
            'evaluations': list,
            'num_{method}_matches': int (for each method),
            '{method}_match_rate': float (for each method)
        }
    """
    evaluations = []
    # Initialize counters for each method
    method_names = [extract_method_name(p) for p in judge_prompts]
    match_counters = {method: 0 for method in method_names}

    for rank, solution in enumerate(solutions, 1):
        eval_result = evaluate_solution_match(solution, paper, client, judge_model, judge_prompts, c, attempt_num)

        # Track tokens (including cache metrics)
        total_tokens['judge']['input'] += eval_result['tokens']['input']
        total_tokens['judge']['output'] += eval_result['tokens']['output']
        total_tokens['judge']['cache_creation_input_tokens'] += eval_result['tokens'].get('cache_creation_input_tokens', 0)
        total_tokens['judge']['cache_read_input_tokens'] += eval_result['tokens'].get('cache_read_input_tokens', 0)

        # Store evaluation with FULL solution object
        # Preserves assessment fields: overall_score, code_availability_score, rationale,
        # overall_score_breakdown, code_availability_score_breakdown, novelty_data
        evaluation = solution.copy()  # Shallow copy preserves all fields

        # Filter novelty_data to only keep novelty scores (remove papers, queries, etc.)
        if 'novelty_data' in evaluation and evaluation['novelty_data']:
            novelty_data = evaluation['novelty_data'].copy()

            # Handle both scoring_methods (all mode) and simple novelty_score
            if 'scoring_methods' in novelty_data:
                # Extract just the novelty_score from each method
                filtered_methods = {}
                for method, data in novelty_data['scoring_methods'].items():
                    if isinstance(data, dict) and 'novelty_score' in data:
                        filtered_methods[method] = data['novelty_score']
                filtered_novelty = {'scoring_methods': filtered_methods}
            elif 'novelty_score' in novelty_data:
                # Single score mode
                filtered_novelty = {'novelty_score': novelty_data['novelty_score']}
            else:
                filtered_novelty = {}

            evaluation['novelty_data'] = filtered_novelty

        evaluation['rank'] = rank
        # Rename 'title' to 'solution_title' for backward compatibility
        if 'title' in evaluation and 'solution_title' not in evaluation:
            evaluation['solution_title'] = evaluation['title']

        # Add match results for each method
        for method in method_names:
            evaluation[f'is_{method}_match'] = eval_result.get(f'is_{method}_match', False)
            evaluation[f'{method}_explanation'] = eval_result.get(f'{method}_explanation', '')

        # Propagate error flag if present
        if eval_result.get('error', False):
            evaluation['error'] = True

        evaluations.append(evaluation)

        # Count matches by method
        for method in method_names:
            if eval_result.get(f'is_{method}_match', False):
                match_counters[method] += 1

    # Calculate match rates and attempt-level matches
    result = {
        'best_solution': solutions[0] if solutions else None,
        'evaluations': evaluations
    }

    for method in method_names:
        num_matches = match_counters[method]
        match_rate = num_matches / len(solutions) if solutions else 0.0
        result[f'is_{method}_match'] = num_matches > 0
        result[f'num_{method}_matches'] = num_matches
        result[f'{method}_match_rate'] = match_rate

    return result


def _evaluate_single_attempt(
    paper: dict,
    attempt_num: int,
    num_attempts: int,
    paper_idx: int,
    total_papers: int,
    eval_config: dict,
    client: Anthropic,
    token_counter: ThreadSafeTokenCounter,
    printer: ThreadSafePrinter,
    c: dict,
    workflow_verbose: bool
) -> dict:
    """
    Execute and evaluate a single attempt. Thread-safe.

    Returns:
        dict with keys:
            - attempt_data: Dict to append to attempts list
            - preprocessing_info: Preprocessing info (only for first attempt)
            - judge_tokens: Token counts for judge calls
            - domain_judge_tokens: Token counts for domain judge calls
            - match_results: Dict of method -> bool for updating any_matches
    """
    printer.print(f"\n  {c['B']}[Paper {paper_idx}/{total_papers}] Attempt {attempt_num}/{num_attempts}:{c['R']}")

    # Run workflow
    solutions, workflow_tokens, attempt_preprocessing_info, discovered_domains, analogies = run_constrained_workflow(
        paper, attempt_num, eval_config, verbose=workflow_verbose
    )

    # Track workflow tokens
    token_counter.add_workflow_tokens(workflow_tokens['input'], workflow_tokens['output'])

    # Extract domain judge tokens from workflow (will be added to counter later in evaluate_paper)
    domain_judge_tokens = {
        'input': workflow_tokens.get('domain_judge_input', 0),
        'output': workflow_tokens.get('domain_judge_output', 0)
    }
    # NOTE: Do NOT add to token_counter here - these are returned and added in evaluate_paper()
    # to avoid double-counting when results are aggregated

    if workflow_verbose:
        printer.print(f"    Generated {len(solutions)} solutions")

    # Initialize return values
    judge_tokens = {'input': 0, 'output': 0, 'cache_creation_input_tokens': 0, 'cache_read_input_tokens': 0}
    match_results = {}
    judge_prompts = eval_config.get('judge_prompts', ['eval/prompts/dataset_eval_match_exact.txt'])
    method_names = [extract_method_name(p) for p in judge_prompts]

    if not solutions:
        # Check if we're in domain-only mode
        cfg = config_singleton.get_all()
        baseline_enabled = eval_config.get('baseline', {}).get('enabled', False)

        if baseline_enabled:
            is_domain_only_mode = eval_config.get('baseline', {}).get('domain_only', False)
        else:
            is_domain_only_mode = not cfg['agents'].get('search', {}).get('enabled', True)

        # If domain-only mode AND we have discovered domains, evaluate them directly
        if is_domain_only_mode and discovered_domains:
            if workflow_verbose:
                printer.print(f"    {c['Y']}Domain-only mode - evaluating {len(discovered_domains)} discovered domains using LLM judge{c['R']}")

            # Evaluate domains using LLM judge
            ground_truth_domain = paper['base_domain']
            domain_judge_model = eval_config.get('domain_judge_model', eval_config['judge_model'])
            evaluations = []
            num_exact_matches = 0

            for i, domain in enumerate(discovered_domains, 1):
                eval_result = evaluate_domain_match(domain, ground_truth_domain, client, domain_judge_model, c)

                # Track domain judge tokens
                domain_judge_tokens['input'] += eval_result['tokens']['input']
                domain_judge_tokens['output'] += eval_result['tokens']['output']

                is_match = eval_result['is_match']
                explanation = eval_result['explanation']

                if is_match:
                    num_exact_matches += 1
                    if workflow_verbose:
                        printer.print(f"      {c['G']}✓{c['R']} Domain #{i}: {domain} - MATCHES ground truth")
                else:
                    if workflow_verbose:
                        printer.print(f"      Domain #{i}: {domain} - No match")

                evaluations.append({
                    'rank': i,
                    'solution_title': f'Domain: {domain}',
                    'source_domain': domain,
                    'match_score': 10.0 if is_match else 0.0,
                    'is_exact_match': is_match,
                    'explanation': explanation
                })

            is_exact_match = num_exact_matches > 0
            best_match_score = 10.0 if is_exact_match else 0.0
            match_rate = num_exact_matches / len(discovered_domains) if discovered_domains else 0.0

            if workflow_verbose:
                match_summary = (
                    f"    Domain match: {c['G'] if is_exact_match else c['Y']}{'YES' if is_exact_match else 'NO'}{c['R']}\n"
                    f"    Match rate: {match_rate:.1%} ({num_exact_matches}/{len(discovered_domains)} domains match)"
                )
                printer.print(match_summary)

            attempt_data = {
                'attempt_num': attempt_num,
                'num_solutions': len(discovered_domains),
                'best_match_score': best_match_score,
                'is_exact_match': is_exact_match,
                'evaluations': evaluations,
                'num_exact_matches': num_exact_matches,
                'match_rate': match_rate,
                'failed': False,
                'discovered_domains': discovered_domains,
                'ground_truth_domain': paper['base_domain']
            }

            # Add analogies if save_analogies flag is enabled
            if eval_config.get('evaluation', {}).get('save_analogies', False):
                attempt_data['analogies'] = analogies

            return {
                'attempt_data': attempt_data,
                'preprocessing_info': attempt_preprocessing_info,
                'judge_tokens': judge_tokens,
                'domain_judge_tokens': domain_judge_tokens,
                'match_results': {}  # No method-based matching for domain-only mode
            }
        else:
            # Not domain-only mode, or no domains discovered - record as failed attempt
            if workflow_verbose:
                if discovered_domains:
                    printer.print(f"    {c['Y']}No solutions generated (discovered {len(discovered_domains)} domains but none matched ground truth){c['R']}")
                else:
                    printer.print(f"    {c['Y']}No solutions generated (workflow failed){c['R']}")

            attempt_data = {
                'attempt_num': attempt_num,
                'num_solutions': 0,
                'best_match_score': 0.0,
                'is_exact_match': False,
                'evaluations': [],
                'num_exact_matches': 0,
                'match_rate': 0.0,
                'failed': True,
                'discovered_domains': discovered_domains,
                'ground_truth_domain': paper['base_domain']
            }

            # Add analogies if save_analogies flag is enabled
            if eval_config.get('evaluation', {}).get('save_analogies', False):
                attempt_data['analogies'] = analogies

            return {
                'attempt_data': attempt_data,
                'preprocessing_info': attempt_preprocessing_info,
                'judge_tokens': judge_tokens,
                'domain_judge_tokens': domain_judge_tokens,
                'match_results': {}
            }

    # Evaluate solutions
    judge_model = eval_config['judge_model']
    # Need to pass total_tokens as a local dict that will be updated
    local_judge_tokens = {'judge': judge_tokens.copy(), 'domain_judge': domain_judge_tokens.copy()}
    attempt_result = evaluate_attempt(solutions, paper, client, judge_model, judge_prompts, local_judge_tokens, c, attempt_num)

    # Extract judge tokens
    judge_tokens = local_judge_tokens['judge']

    # Print result
    printer.print(f"    Matches by scoring method:")
    for method in method_names:
        num_matches = attempt_result.get(f'num_{method}_matches', 0)
        match_rate = attempt_result.get(f'{method}_match_rate', 0.0)
        method_label = method.replace('_', ' ').title()
        printer.print(f"      {method_label}: {match_rate:.1%} ({num_matches}/{len(solutions)})")

    # Print all solution results
    printer.print(f"\n    {c['B']}[Attempt {attempt_num}] Solution matches:{c['R']}")
    for i, eval_result in enumerate(attempt_result['evaluations'], 1):
        solution_title = eval_result.get('solution_title') or eval_result.get('title', 'Unknown')
        printer.print(f"      {i}. {solution_title}")

        match_statuses = []
        for method in method_names:
            is_match = eval_result.get(f'is_{method}_match', False)
            method_label = method.replace('_', ' ').title()
            status = f"{c['G']}✓{c['R']}" if is_match else "✗"
            match_statuses.append(f"{method_label}: {status}")

        printer.print(f"         {' | '.join(match_statuses)}")
    printer.print("")  # Add blank line after results

    # Check if any evaluations had errors
    has_judge_errors = any('error' in e for e in attempt_result.get('evaluations', []))

    # Build attempt data
    attempt_data = {
        'attempt_num': attempt_num,
        'num_solutions': len(solutions),
        'failed': has_judge_errors,
        'discovered_domains': discovered_domains,
        'ground_truth_domain': paper['base_domain'],
        **attempt_result
    }

    # Add analogies if save_analogies flag is enabled
    if eval_config.get('evaluation', {}).get('save_analogies', False):
        attempt_data['analogies'] = analogies

    # Build match results for paper-level tracking
    match_results = {method: attempt_result.get(f'is_{method}_match', False) for method in method_names}

    return {
        'attempt_data': attempt_data,
        'preprocessing_info': attempt_preprocessing_info,
        'judge_tokens': judge_tokens,
        'domain_judge_tokens': domain_judge_tokens,
        'match_results': match_results
    }


def evaluate_paper(
    paper: dict,
    paper_idx: int,
    total_papers: int,
    num_attempts: int,
    eval_config: dict,
    client: Anthropic,
    token_counter: ThreadSafeTokenCounter,
    printer: ThreadSafePrinter,
    c: dict
) -> dict:
    """
    Run multiple attempts for one paper and evaluate.

    Returns:
        {
            'paper_title': str,
            'base_domain': str,
            'difficulty': str,
            'attempts': list,
            'any_exact_match': bool
        }
    """
    # Get verbose setting from config (defaults to False for parallel mode if not specified)
    max_workers = eval_config.get('evaluation', {}).get('max_workers', 1)
    workflow_verbose = eval_config.get('evaluation', {}).get('verbose', max_workers <= 1)

    # Print paper header as single atomic operation to prevent interleaving
    if workflow_verbose:
        # Detailed header for sequential mode
        header = (
            f"\n{c['C']}[Paper {paper_idx}/{total_papers}] {paper['title']}{c['R']}\n"
            f"  Base domain: {paper['base_domain']}\n"
            f"  Difficulty: {paper['difficulty']}\n"
            f"\n  {c['M']}Analogy Description:{c['R']}\n"
            f"  {paper.get('analogy_description', 'N/A')}\n"
            f"\n  {c['M']}Original Problem:{c['R']}\n"
            f"  {paper['problem']}"
        )
        printer.print(header)
    else:
        # Minimal header for parallel mode
        header = f"\n{c['C']}[Paper {paper_idx}/{total_papers}] {paper['title']}{c['R']}"
        printer.print(header)

    attempts = []
    # Initialize match tracking for all configured methods
    judge_prompts = eval_config.get('judge_prompts', ['eval/prompts/dataset_eval_match_exact.txt'])
    method_names = [extract_method_name(p) for p in judge_prompts]
    any_matches = {method: False for method in method_names}
    preprocessing_info = None  # Track preprocessing info from first attempt
    preprocessing_printed = False

    # Initialize token counter for LLM judge calls
    total_tokens = {
        'judge': {
            'input': 0,
            'output': 0,
            'cache_creation_input_tokens': 0,
            'cache_read_input_tokens': 0
        },
        'domain_judge': {
            'input': 0,
            'output': 0
        }
    }

    # Apply config once for all attempts in this paper
    # This enables attempt-level parallelization (config no longer locked per-attempt)
    apply_paper_config(paper, eval_config, verbose=workflow_verbose)

    # Determine parallelization strategy
    parallelize_over = eval_config.get('evaluation', {}).get('parallelize_over', 'papers')

    # Run attempts (parallel or sequential based on config)
    attempt_results = []
    if parallelize_over == 'attempts' and max_workers > 1:
        # PARALLEL: Run attempts in parallel using ThreadPoolExecutor
        from concurrent.futures import ThreadPoolExecutor, as_completed

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {}
            for attempt_num in range(1, num_attempts + 1):
                future = executor.submit(
                    _evaluate_single_attempt,
                    paper, attempt_num, num_attempts, paper_idx, total_papers,
                    eval_config, client, token_counter, printer, c, workflow_verbose
                )
                futures[future] = attempt_num

            # Collect results as they complete
            results_dict = {}
            for future in as_completed(futures):
                attempt_num = futures[future]
                try:
                    result = future.result()
                    results_dict[attempt_num] = result
                except Exception as e:
                    import traceback
                    printer.print(f"  ERROR: Attempt {attempt_num} failed: {str(e)}")
                    traceback.print_exc()
                    # Create a failed attempt result
                    results_dict[attempt_num] = {
                        'attempt_data': {
                            'attempt_num': attempt_num,
                            'num_solutions': 0,
                            'failed': True,
                            'discovered_domains': [],
                            'ground_truth_domain': paper['base_domain']
                        },
                        'preprocessing_info': None,
                        'judge_tokens': {'input': 0, 'output': 0, 'cache_creation_input_tokens': 0, 'cache_read_input_tokens': 0},
                        'domain_judge_tokens': {'input': 0, 'output': 0},
                        'match_results': {}
                    }

            # Sort results by attempt number for consistent output
            attempt_results = [results_dict[i] for i in sorted(results_dict.keys())]
    else:
        # SEQUENTIAL: Run attempts one at a time (original behavior)
        for attempt_num in range(1, num_attempts + 1):
            result = _evaluate_single_attempt(
                paper, attempt_num, num_attempts, paper_idx, total_papers,
                eval_config, client, token_counter, printer, c, workflow_verbose
            )
            attempt_results.append(result)

    # Process results (same for both parallel and sequential)
    for result in attempt_results:
        # Capture preprocessing info from first attempt and print it
        if result['attempt_data']['attempt_num'] == 1 and result['preprocessing_info'] and not preprocessing_printed:
            preprocessing_info = result['preprocessing_info']
            if workflow_verbose:
                preprocessing_msg = (
                    f"\n  {c['Y']}>>> PREPROCESSING APPLIED <<<{c['R']}\n"
                    f"  {c['M']}Rewritten Problem (used in workflow):{c['R']}\n"
                    f"  {preprocessing_info['rewritten_problem']}\n"
                    f"\n  {c['M']}Base Domain: {preprocessing_info['base_domain']}{c['R']}"
                )
                printer.print(preprocessing_msg)
            preprocessing_printed = True

        # Accumulate tokens
        total_tokens['judge']['input'] += result['judge_tokens']['input']
        total_tokens['judge']['output'] += result['judge_tokens']['output']
        total_tokens['judge']['cache_creation_input_tokens'] += result['judge_tokens'].get('cache_creation_input_tokens', 0)
        total_tokens['judge']['cache_read_input_tokens'] += result['judge_tokens'].get('cache_read_input_tokens', 0)
        total_tokens['domain_judge']['input'] += result['domain_judge_tokens']['input']
        total_tokens['domain_judge']['output'] += result['domain_judge_tokens']['output']

        # Track attempt
        attempts.append(result['attempt_data'])

        # Update paper-level tracking
        for method, has_match in result['match_results'].items():
            if has_match:
                any_matches[method] = True

    # Count attempts with judge errors (API refusals, etc.)
    num_attempts_with_judge_errors = sum(1 for a in attempts if a.get('failed', False))

    result = {
        'paper_title': paper['title'],
        'base_domain': paper['base_domain'],
        'target_domain': paper.get('target_domain', 'Unknown'),
        'difficulty': paper['difficulty'],
        'problem': paper['problem'],
        'analogy_description': paper.get('analogy_description', 'N/A'),
        'source_dataset': paper.get('_source_dataset', 'Unknown'),
        'source_index': paper.get('_source_index', 'Unknown'),
        'attempts': attempts,
        'num_attempts_with_judge_errors': num_attempts_with_judge_errors
    }

    # Add match tracking for each configured method
    for method, has_match in any_matches.items():
        result[f'any_{method}_match'] = has_match

    # Add preprocessing info if preprocessing was applied
    if preprocessing_info:
        result['preprocessing'] = preprocessing_info

    # Track judge tokens
    token_counter.add_judge_tokens(total_tokens['judge'])
    token_counter.add_domain_judge_tokens(total_tokens['domain_judge'])

    return result


def calculate_metrics(all_results: list, method_names: list) -> dict:
    """Calculate aggregate metrics across all papers for configured scoring methods."""
    total_papers = len(all_results)
    total_attempts = sum(len(r['attempts']) for r in all_results)
    total_solutions = sum(
        attempt.get('num_solutions', 0)
        for result in all_results
        for attempt in result['attempts']
    )

    metrics = {
        'total_papers': total_papers,
        'total_attempts': total_attempts,
        'total_solutions': total_solutions
    }

    # Calculate metrics for each configured method
    for method in method_names:
        # Count attempts with match
        match_count = sum(
            1 for r in all_results
            for a in r['attempts']
            if a.get(f'is_{method}_match', False)
        )

        # Hit rate (% attempts with match)
        hit_rate = match_count / total_attempts if total_attempts > 0 else 0.0

        # Count total individual solution matches
        total_solution_matches = sum(
            attempt.get(f'num_{method}_matches', 0)
            for result in all_results
            for attempt in result['attempts']
        )

        # Count papers with at least one match
        papers_with_match = sum(1 for r in all_results if r.get(f'any_{method}_match', False))

        # Coverage (% papers with match)
        coverage = papers_with_match / total_papers if total_papers > 0 else 0.0

        # Calculate average match rate
        match_rates = [
            a.get(f'{method}_match_rate', 0.0)
            for r in all_results
            for a in r['attempts']
            if f'{method}_match_rate' in a
        ]
        avg_match_rate = float(np.mean(match_rates)) if match_rates else 0.0

        # Store metrics for this method
        metrics[f'{method}_match_count'] = match_count
        metrics[f'{method}_hit_rate'] = hit_rate
        metrics[f'total_{method}_solution_matches'] = total_solution_matches
        metrics[f'papers_with_{method}_match'] = papers_with_match
        metrics[f'{method}_coverage'] = coverage
        metrics[f'avg_{method}_match_rate'] = avg_match_rate

    # Error tracking
    metrics['total_judge_errors'] = sum(r.get('num_attempts_with_judge_errors', 0) for r in all_results)
    metrics['papers_with_judge_errors'] = sum(1 for r in all_results if r.get('num_attempts_with_judge_errors', 0) > 0)

    return metrics


def calculate_domain_metrics(all_results: list) -> dict:
    """Calculate domain discovery accuracy metrics."""
    total_attempts = 0
    domain_exact_matches = 0
    domain_partial_matches = 0

    for result in all_results:
        ground_truth_domain = result['base_domain']

        for attempt in result['attempts']:
            total_attempts += 1
            discovered = attempt.get('discovered_domains', [])

            if not discovered:
                continue

            # Exact match: ground truth in discovered list
            if ground_truth_domain in discovered:
                domain_exact_matches += 1

            # Partial match: substring or stem match (e.g., "ecology" matches "evolutionary_ecology")
            else:
                gt_lower = ground_truth_domain.lower().replace('_', '')
                for d in discovered:
                    d_lower = d.lower().replace('_', '')
                    if gt_lower in d_lower or d_lower in gt_lower:
                        domain_partial_matches += 1
                        break

    return {
        'total_attempts': total_attempts,
        'domain_exact_match_rate': domain_exact_matches / total_attempts if total_attempts > 0 else 0.0,
        'domain_partial_match_rate': domain_partial_matches / total_attempts if total_attempts > 0 else 0.0,
        'domain_miss_rate': (total_attempts - domain_exact_matches - domain_partial_matches) / total_attempts if total_attempts > 0 else 0.0
    }


def get_model_pricing(model_name: str) -> tuple[float, float]:
    """Get pricing for a model.

    Args:
        model_name: Model identifier

    Returns:
        (input_price_per_mtok, output_price_per_mtok)
    """
    # Pricing in USD per million tokens
    if 'haiku' in model_name.lower():
        return (1.0, 5.0)  # Haiku: $1 input, $5 output
    elif 'sonnet' in model_name.lower():
        return (3.0, 15.0)  # Sonnet: $3 input, $15 output
    elif 'opus' in model_name.lower():
        return (15.0, 75.0)  # Opus: $15 input, $75 output
    else:
        # Default to Sonnet pricing if unknown
        return (3.0, 15.0)


def calculate_cost(tokens: dict, workflow_model: str, judge_model: str, domain_judge_model: str = None) -> dict:
    """Calculate total cost in USD with breakdown.

    Args:
        tokens: Token usage dict with workflow, judge, and domain_judge breakdowns
        workflow_model: Model used for workflow
        judge_model: Model used for solution judging
        domain_judge_model: Model used for domain matching (optional, defaults to judge_model)

    Returns:
        {
            'workflow_cost': float,
            'judge_cost': float,
            'domain_judge_cost': float,
            'total_cost': float,
            'cache_savings': float (optional)
        }
    """
    workflow_input_price, workflow_output_price = get_model_pricing(workflow_model)
    judge_input_price, judge_output_price = get_model_pricing(judge_model)

    # Domain judge defaults to same model as judge if not specified
    if domain_judge_model is None:
        domain_judge_model = judge_model
    domain_judge_input_price, domain_judge_output_price = get_model_pricing(domain_judge_model)

    # Get cache token counts
    cache_read_tokens = tokens['judge'].get('cache_read_input_tokens', 0)

    # Cache read pricing: 10% of regular input price
    judge_cache_read_price = judge_input_price * 0.1

    # Calculate workflow cost
    workflow_cost = (
        (tokens['workflow']['input'] / 1_000_000) * workflow_input_price +
        (tokens['workflow']['output'] / 1_000_000) * workflow_output_price
    )

    # Calculate judge cost (accounting for cached tokens)
    # Note: tokens['judge']['input'] from Anthropic API already excludes cached tokens
    # It only includes non-cached input tokens charged at full price
    judge_regular_input_tokens = tokens['judge']['input']

    judge_cost = (
        (judge_regular_input_tokens / 1_000_000) * judge_input_price +  # Regular input (full price)
        (cache_read_tokens / 1_000_000) * judge_cache_read_price +  # Cached input (10% price)
        (tokens['judge']['output'] / 1_000_000) * judge_output_price  # Output
    )

    # Calculate domain judge cost (no caching)
    domain_judge_cost = 0.0
    if 'domain_judge' in tokens:
        domain_judge_cost = (
            (tokens['domain_judge']['input'] / 1_000_000) * domain_judge_input_price +
            (tokens['domain_judge']['output'] / 1_000_000) * domain_judge_output_price
        )

    # Calculate cache savings (90% of what we would have paid for cached tokens)
    cache_savings = (cache_read_tokens / 1_000_000) * judge_input_price * 0.9

    result = {
        'workflow_cost': workflow_cost,
        'judge_cost': judge_cost,
        'domain_judge_cost': domain_judge_cost,
        'total_cost': workflow_cost + judge_cost + domain_judge_cost
    }

    # Add cache savings if any caching was used
    if cache_savings > 0:
        result['cache_savings'] = cache_savings

    return result


def generate_report(
    all_results: list,
    metrics: dict,
    output_dir: Path,
    run_id: str,
    runtime: float,
    cost_breakdown: dict,
    total_tokens: dict,
    eval_config: dict,
    c: dict
):
    """Generate JSON results and print summary."""

    # Save JSON results
    results_json = {
        'run_id': run_id,
        'timestamp': datetime.now().isoformat(),
        'num_papers': metrics['total_papers'],
        'num_attempts_per_paper': eval_config['evaluation']['num_attempts_per_paper'],
        'config': eval_config,
        'paper_results': all_results,
        'metrics': metrics,
        'total_tokens': total_tokens,
        'cost_breakdown': cost_breakdown,
        'runtime_seconds': runtime,
        'total_cost_usd': cost_breakdown['total_cost']
    }

    # Add dataset info (handle both single and multi-dataset modes)
    if 'dataset_path' in eval_config:
        results_json['dataset_path'] = eval_config['dataset_path']
    elif 'datasets' in eval_config:
        results_json['datasets'] = eval_config['datasets']

    # Remove fields we don't want in the saved JSON
    for paper_result in results_json['paper_results']:
        for attempt in paper_result.get('attempts', []):
            # Remove best_solution from attempts (redundant with evaluations)
            attempt.pop('best_solution', None)

    with open(output_dir / "results.json", 'w') as f:
        json.dump(results_json, f, indent=2)

    # Print summary
    print("\n" + "=" * 70)
    print(f"{c['C']}EVALUATION SUMMARY{c['R']}")
    print("=" * 70)

    print(f"\n{c['C']}[OVERALL METRICS]{c['R']}")
    print(f"  Total papers: {metrics['total_papers']}")
    print(f"  Total attempts: {metrics['total_attempts']}")
    print(f"  Total solutions evaluated: {metrics['total_solutions']}")
    print()

    # Get method names from config
    judge_prompts = eval_config.get('judge_prompts', ['eval/prompts/dataset_eval_match_exact.txt'])
    method_names = [extract_method_name(p) for p in judge_prompts]

    # Show metrics for each configured method
    for method in method_names:
        method_label = method.replace('_', ' ').title()
        print(f"  {c['B']}{method_label.upper()}:{c['R']}")
        print(f"    Attempts with match: {metrics[f'{method}_match_count']}/{metrics['total_attempts']} ({metrics[f'{method}_hit_rate']:.1%})")
        print(f"    Papers with match: {metrics[f'papers_with_{method}_match']}/{metrics['total_papers']} ({metrics[f'{method}_coverage']:.1%})")
        print(f"    Total solution matches: {metrics[f'total_{method}_solution_matches']}/{metrics['total_solutions']}")
        print(f"    Avg match rate per attempt: {metrics[f'avg_{method}_match_rate']:.1%}")
        print()

    # Only show judge error stats if there were any errors
    if metrics['total_judge_errors'] > 0:
        print(f"  {c['Y']}Judge Errors:{c['R']}")
        print(f"    Attempts with errors: {metrics['total_judge_errors']}/{metrics['total_attempts']} (API refusals)")
        print(f"    Papers affected: {metrics['papers_with_judge_errors']}/{metrics['total_papers']}")
        print()

    # Print domain discovery metrics if available
    if 'domain_discovery' in metrics:
        print(f"\n{c['C']}[DOMAIN DISCOVERY]{c['R']}")
        dm = metrics['domain_discovery']
        print(f"  Domain exact match rate: {dm['domain_exact_match_rate']:.1%}")
        print(f"  Domain partial match rate: {dm['domain_partial_match_rate']:.1%}")
        print(f"  Domain miss rate: {dm['domain_miss_rate']:.1%}")

    # Print by difficulty
    if metrics.get('by_difficulty'):
        print(f"\n{c['C']}[BY DIFFICULTY]{c['R']}")
        for difficulty in ['easy', 'medium', 'hard']:
            if difficulty in metrics['by_difficulty']:
                diff = metrics['by_difficulty'][difficulty]
                print(f"\n  {c['B']}{difficulty.upper()}:{c['R']}")
                print(f"    Papers: {diff['papers']}")
                print(f"    Hit rate: {diff['hit_rate']:.1%}")
                print(f"    Coverage: {diff['coverage']:.1%}")
                print(f"    Avg match rate: {diff['avg_match_rate']:.1%}")
                # Only show judge errors if there were any
                if diff.get('attempts_with_judge_errors', 0) > 0:
                    print(f"    Attempts with judge errors: {diff['attempts_with_judge_errors']}/{diff['attempts']}")

    # Print per-paper summary
    print(f"\n{c['C']}[PER-PAPER RESULTS]{c['R']}")
    for i, result in enumerate(all_results, 1):
        # Show match status for all configured methods
        match_statuses = []
        for method in method_names:
            has_match = result.get(f'any_{method}_match', False)
            status = f"{c['G']}✓{c['R']}" if has_match else "✗"
            method_label = method.replace('_', ' ').title()
            match_statuses.append(f"{method_label}: {status}")

        num_judge_errors = result.get('num_attempts_with_judge_errors', 0)
        judge_error_str = f" {c['Y']}[JUDGE ERRORS: {num_judge_errors}]{c['R']}" if num_judge_errors > 0 else ""

        print(f"\n  {i}. {result['paper_title']}")
        print(f"     {' | '.join(match_statuses)} | Difficulty: {result['difficulty']}{judge_error_str}")

        # Show analogy description
        print(f"     {c['M']}Analogy:{c['R']} {result.get('analogy_description', 'N/A')}")

        # Show problem statement
        print(f"     {c['M']}Original Problem:{c['R']} {result['problem']}")
        if 'preprocessing' in result:
            print(f"     {c['Y']}Rewritten Problem:{c['R']} {result['preprocessing']['rewritten_problem']}")
            print(f"     {c['M']}Base Domain:{c['R']} {result['preprocessing']['base_domain']}")

        # Show attempt-level details
        for attempt in result['attempts']:
            attempt_num = attempt['attempt_num']
            num_solutions = attempt['num_solutions']
            has_judge_errors = attempt.get('failed', False)

            if num_solutions == 0:
                print(f"     Attempt {attempt_num}: No solutions generated")
                continue

            # Get match counts and rates for all configured methods
            judge_prompts = eval_config.get('judge_prompts', ['eval/prompts/dataset_eval_match_exact.txt'])
            method_names = [extract_method_name(p) for p in judge_prompts]

            # Only show judge error count if there were errors
            judge_error_marker = ""
            if has_judge_errors:
                num_errors = sum(1 for e in attempt.get('evaluations', []) if e.get('error', False))
                judge_error_marker = f" {c['Y']}[{num_errors} judge refusals]{c['R']}"

            print(f"     Attempt {attempt_num}:")

            # Show match rates for each method
            rate_strs = []
            for method in method_names:
                num_matches = attempt.get(f'num_{method}_matches', 0)
                match_rate = attempt.get(f'{method}_match_rate', 0.0)
                method_label = method.replace('_', ' ').title()
                rate_strs.append(f"{method_label}: {match_rate:.1%} ({num_matches}/{num_solutions})")
            print(f"       {' | '.join(rate_strs)}{judge_error_marker}")

            # Show solution-level details
            evaluations = attempt.get('evaluations', [])
            if evaluations:
                print(f"       Solution matches:")
                for eval_result in evaluations:
                    solution_title = eval_result.get('solution_title') or eval_result.get('title', 'Unknown')
                    print(f"         • {solution_title}")

                    # Show assessment scores if present
                    overall_score = eval_result.get('overall_score')
                    code_score = eval_result.get('code_availability_score')
                    novelty_data = eval_result.get('novelty_data', {})

                    score_parts = []
                    if overall_score is not None:
                        score_parts.append(f"Overall: {overall_score:.1f}/10")
                    if code_score is not None:
                        score_parts.append(f"Code: {code_score:.1f}/10")

                    # Add novelty scores if present
                    if novelty_data:
                        # Handle scoring_methods format (all mode)
                        if 'scoring_methods' in novelty_data:
                            scoring_methods = novelty_data['scoring_methods']
                            novelty_strs = []
                            for score_type, value in scoring_methods.items():
                                if value is not None:
                                    # Format binary as 0/1, others as 0.00-1.00
                                    if score_type == 'binary':
                                        novelty_strs.append(f"{score_type}: {int(value)}")
                                    else:
                                        novelty_strs.append(f"{score_type}: {value:.2f}")
                            if novelty_strs:
                                score_parts.append(f"Novelty ({', '.join(novelty_strs)})")
                        # Handle simple novelty_score format
                        elif 'novelty_score' in novelty_data:
                            score = novelty_data['novelty_score']
                            if score is not None:
                                score_parts.append(f"Novelty: {score:.1f}")

                    if score_parts:
                        print(f"           {c['B']}Scores:{c['R']} {' | '.join(score_parts)}")

                    # Show match status for each method
                    match_statuses = []
                    for method in method_names:
                        is_match = eval_result.get(f'is_{method}_match', False)
                        status = f"{c['G']}✓{c['R']}" if is_match else "✗"
                        method_label = method.replace('_', ' ').title()
                        match_statuses.append(f"{method_label}: {status}")
                    if match_statuses:  # Only print if there are judge methods configured
                        print(f"           {' | '.join(match_statuses)}")
                print()  # Blank line after scores

    # Print domain analysis
    print(f"\n{c['C']}[DOMAIN ANALYSIS]{c['R']}")

    # Check if domain filtering was enabled
    baseline_enabled = eval_config.get('baseline', {}).get('enabled', False)
    if baseline_enabled:
        filtering_enabled = eval_config.get('baseline', {}).get('only_search_matched_domains', False)
    else:
        filtering_enabled = eval_config.get('agents', {}).get('search', {}).get('only_search_matched_domains', False)

    for i, result in enumerate(all_results, 1):
        print(f"\n  {i}. {result['paper_title']}")
        print(f"     {c['M']}Ground Truth Domain:{c['R']} {result['base_domain']}")

        # Collect all solution domains and discovered domains from all attempts
        solution_domains = {}
        all_discovered_domains = set()
        for attempt in result['attempts']:
            # Track discovered domains (these are the ones that were matched when filtering is enabled)
            for disc_domain in attempt.get('discovered_domains', []):
                all_discovered_domains.add(disc_domain.lower().strip())

            for evaluation in attempt.get('evaluations', []):
                domain = evaluation.get('source_domain', 'Unknown')
                is_match = evaluation.get('is_exact_match', False)
                if domain not in solution_domains:
                    solution_domains[domain] = {'count': 0, 'matches': 0}
                solution_domains[domain]['count'] += 1
                if is_match:
                    solution_domains[domain]['matches'] += 1

        # Print solution domains sorted by match count
        if solution_domains:
            print(f"     {c['B']}Solution Domains Found:{c['R']}")
            sorted_domains = sorted(solution_domains.items(),
                                   key=lambda x: (x[1]['matches'], x[1]['count']),
                                   reverse=True)
            for domain, stats in sorted_domains:
                domain_lower = domain.lower().strip()

                # Determine if this domain was matched by the LLM judge
                if filtering_enabled:
                    # When filtering is enabled, check if domain is in discovered_domains (matched by judge)
                    is_matched_domain = domain_lower in all_discovered_domains
                else:
                    # When filtering is disabled, fall back to substring matching
                    ground_truth_domain = result['base_domain'].lower().strip()
                    is_matched_domain = ground_truth_domain in domain_lower

                # Highlight domain in green if it was matched
                domain_display = f"{c['G']}{domain}{c['R']}" if is_matched_domain else domain

                match_marker = f" {c['G']}({stats['matches']} matches){c['R']}" if stats['matches'] > 0 else ""
                matched_domain_marker = f" {c['G']}✓ MATCHED DOMAIN{c['R']}" if is_matched_domain else ""
                print(f"       • {domain_display}: {stats['count']} solutions{match_marker}{matched_domain_marker}")
        else:
            print(f"     {c['Y']}No solutions generated{c['R']}")

    print(f"\n{c['C']}[COST & RUNTIME]{c['R']}")

    # Label tokens appropriately based on workflow type
    baseline_enabled = eval_config.get('baseline', {}).get('enabled', False)
    workflow_label = "Baseline tokens" if baseline_enabled else "Workflow tokens"
    cost_label = "Baseline cost" if baseline_enabled else "Workflow cost"

    print(f"  {workflow_label}: {total_tokens['workflow']['input']:,} input / {total_tokens['workflow']['output']:,} output")
    print(f"  {cost_label}: ${cost_breakdown['workflow_cost']:.2f} USD")

    # Solution judge tokens (Sonnet)
    print(f"  Solution judge tokens: {total_tokens['judge']['input']:,} input / {total_tokens['judge']['output']:,} output")

    # Show cache metrics if caching was used
    cache_read = total_tokens['judge'].get('cache_read_input_tokens', 0)
    cache_creation = total_tokens['judge'].get('cache_creation_input_tokens', 0)
    if cache_read > 0 or cache_creation > 0:
        print(f"  Solution judge cache: {cache_read:,} tokens read (90% discount), {cache_creation:,} tokens created")
        if 'cache_savings' in cost_breakdown:
            # Calculate what cost would have been without caching
            judge_cost_with_cache = cost_breakdown['judge_cost']
            cache_savings = cost_breakdown['cache_savings']
            judge_cost_without_cache = judge_cost_with_cache + cache_savings
            savings_pct = (cache_savings / judge_cost_without_cache * 100) if judge_cost_without_cache > 0 else 0
            print(f"  {c['G']}Cache savings: ${cache_savings:.2f} USD ({savings_pct:.1f}% reduction in judge cost){c['R']}")

    print(f"  Solution judge cost: ${cost_breakdown['judge_cost']:.2f} USD ({eval_config['judge_model']})")

    # Domain judge tokens (Haiku)
    if 'domain_judge' in total_tokens and (total_tokens['domain_judge']['input'] > 0 or total_tokens['domain_judge']['output'] > 0):
        domain_judge_model = eval_config.get('domain_judge_model', eval_config['judge_model'])
        print(f"  Domain judge tokens: {total_tokens['domain_judge']['input']:,} input / {total_tokens['domain_judge']['output']:,} output")
        print(f"  Domain judge cost: ${cost_breakdown['domain_judge_cost']:.2f} USD ({domain_judge_model})")

    # Show total cost with cache savings highlighted
    total_cost = cost_breakdown['total_cost']
    print(f"  Total cost: ${total_cost:.2f} USD", end="")
    if 'cache_savings' in cost_breakdown:
        cache_savings = cost_breakdown['cache_savings']
        cost_without_cache = total_cost + cache_savings
        print(f" (saved ${cache_savings:.2f} from ${cost_without_cache:.2f})")
    else:
        print()

    print(f"  Runtime: {runtime:.1f}s ({runtime/60:.1f} min)")

    print(f"\n{c['G']}Results saved to: {output_dir}{c['R']}")
    print("=" * 70 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Evaluate system on dataset of papers")
    parser.add_argument("--config", default=None, help="Path to eval config YAML")
    parser.add_argument("--output", default=None, help="Output directory override")

    args = parser.parse_args()

    # Load eval config
    eval_config = load_eval_config(args.config)

    # Generate run ID
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Create output directory
    if args.output:
        output_dir = Path(args.output)
    else:
        output_dir = ROOT / eval_config['output_dir'] / run_id

    output_dir.mkdir(parents=True, exist_ok=True)

    # Set up logging
    terminal_log_path = output_dir / "terminal_output.log"
    original_stdout = sys.stdout
    tee_output = TeeOutput(terminal_log_path, original_stdout)
    sys.stdout = tee_output

    try:
        # Initialize colors
        c = _init_colors()

        print("=" * 70)
        print(f"{c['C']}DATASET EVALUATION{c['R']}")
        print("=" * 70)
        print(f"Run ID: {c['B']}{run_id}{c['R']}")
        print(f"Output: {output_dir}")
        print()
        print(f"{c['C']}Configuration:{c['R']}")

        # Show mode (baseline or AR)
        baseline_enabled = eval_config.get('baseline', {}).get('enabled', False)
        mode_str = "BASELINE" if baseline_enabled else "ANALOGOUS REASONING"
        print(f"  Mode: {mode_str}")

        print(f"  Judge model: {eval_config['judge_model']}")
        print(f"  Workflow model: {eval_config['model']['name']}")

        if baseline_enabled:
            baseline_mode = eval_config['baseline'].get('mode', 'simple_llm')
            use_deep_research = eval_config['baseline'].get('use_deep_research', False)
            print(f"  Baseline: mode={baseline_mode}, deep_research={use_deep_research}")
        else:
            print(f"  Extraction: {eval_config['extraction']['reasoning_type']}")
            print(f"  Search: LLM fallback={eval_config['search']['use_llm_fallback']}")

        print(f"  Assessment: enabled={eval_config['agents']['assessment']['enabled']}")
        print(f"  Attempts per paper: {eval_config['evaluation']['num_attempts_per_paper']}")

        # Print solutions count based on mode
        eval_mode = eval_config.get('evaluation', {}).get('mode', 'solution_search')
        if eval_mode == 'domain_search':
            print(f"  Solutions per domain: {eval_config['evaluation']['num_solutions_per_domain']}")
        else:
            print(f"  Solutions per attempt: {eval_config['evaluation']['num_solutions_per_attempt']}")

        print(f"  Match threshold: {eval_config['evaluation']['match_threshold']}")
        print()

        # Load dataset(s) or custom question
        custom_question = eval_config.get('evaluation', {}).get('custom_question')
        datasets_config = eval_config.get('datasets')
        paper_indices_config = eval_config['evaluation'].get('paper_indices')

        # Validate mutual exclusivity
        config_count = sum([
            custom_question is not None,
            datasets_config is not None,
            paper_indices_config is not None
        ])

        if config_count == 0:
            print(f"{c['Y']}ERROR: Must specify either 'custom_question', 'datasets', or 'paper_indices' in evaluation config{c['R']}")
            sys.exit(1)
        elif config_count > 1:
            print(f"{c['Y']}ERROR: Cannot specify multiple of 'custom_question', 'datasets', and 'paper_indices' simultaneously{c['R']}")
            sys.exit(1)

        if custom_question:
            # Custom question mode
            print(f"{c['C']}Custom question mode enabled{c['R']}")

            # Validate required field
            if 'problem' not in custom_question:
                print(f"{c['Y']}ERROR: custom_question must have 'problem' field{c['R']}")
                sys.exit(1)

            # Create synthetic paper
            paper = {
                'title': custom_question.get('title', 'Custom Question'),
                'problem': custom_question['problem'],
                'base_domain': custom_question.get('base_domain', 'unknown'),
                'target_domain': custom_question.get('target_domain', 'unknown'),
                'difficulty': 'custom',
                'analogy_description': custom_question.get('analogy_description', 'N/A'),
                'uses_analogical_reasoning': True,
                '_source_dataset': 'custom_question',
                '_source_index': 1
            }

            papers = [paper]
            num_attempts = eval_config['evaluation'].get('num_attempts_per_paper', 1)

            print(f"\nUsing custom question: '{paper['title']}'")
            print(f"  Problem: {paper['problem'][:100]}...")
            print(f"  Target domain: {paper['target_domain']}")
            print(f"  Ground truth base domain: {paper['base_domain']}")
            print(f"  Attempts: {num_attempts}\n")

        elif datasets_config:
            # Multi-dataset mode
            print(f"{c['C']}Multi-dataset mode enabled{c['R']}")
            print(f"Loading from {len(datasets_config)} datasets:")
            for ds in datasets_config:
                ds_path = Path(ds['path'])
                indices = ds.get('paper_indices')
                if indices:
                    print(f"  - {ds_path.name}: indices {indices}")
                else:
                    print(f"  - {ds_path.name}: all AR papers")
            print()
            papers = load_datasets_multi(datasets_config)
            print(f"{c['G']}Total papers loaded: {len(papers)}{c['R']}\n")
        else:
            # Single dataset mode (backward compatible)
            dataset_path = ROOT / eval_config['dataset_path']
            print(f"Loading dataset from: {dataset_path}")
            all_papers = load_dataset(dataset_path)

            # Select papers based on paper_indices or num_papers
            if paper_indices_config is not None:
                # Select specific papers by index (1-based indexing)
                papers = []
                for idx in paper_indices_config:
                    if 1 <= idx <= len(all_papers):
                        papers.append(all_papers[idx - 1])
                    else:
                        print(f"{c['Y']}Warning: Paper index {idx} out of range (1-{len(all_papers)}), skipping{c['R']}")
                print(f"Selected {len(papers)} specific papers: {paper_indices_config}")
            else:
                # Limit to first N papers if specified
                num_papers = eval_config['evaluation'].get('num_papers')
                if num_papers is not None:
                    papers = all_papers[:num_papers]
                    print(f"Limited to first {num_papers} papers")
                else:
                    papers = all_papers

            print(f"Evaluating {len(papers)} papers with analogical reasoning\n")

        if not papers:
            print(f"{c['Y']}ERROR: No papers with analogical reasoning found{c['R']}")
            sys.exit(1)

        # Initialize client
        client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

        # Initialize thread-safe token tracking
        token_counter = ThreadSafeTokenCounter()
        printer = ThreadSafePrinter()

        # Get parallelization config
        max_workers = eval_config.get('evaluation', {}).get('max_workers', 5)
        parallelize_over = eval_config.get('evaluation', {}).get('parallelize_over', 'papers')

        # Print parallelization strategy
        if max_workers > 1:
            if parallelize_over == 'attempts':
                printer.print(f"\n🚀 Running with {max_workers} parallel workers PER PAPER (parallelizing attempts)\n")
            else:
                printer.print(f"\n🚀 Running with {max_workers} parallel workers ACROSS PAPERS (parallelizing papers)\n")
        else:
            printer.print(f"\n📋 Running in sequential mode (max_workers=1)\n")

        # Run evaluation
        start_time = time.time()
        all_results = []

        num_attempts = eval_config['evaluation']['num_attempts_per_paper']

        # Route parallelization based on strategy
        if parallelize_over == 'papers' and max_workers > 1:
            # PAPER-LEVEL PARALLELIZATION: Run multiple papers in parallel
            # Each paper runs its attempts sequentially
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                # Submit all papers
                future_to_paper = {}
                for idx, paper in enumerate(papers, 1):
                    future = executor.submit(
                        evaluate_paper,
                        paper, idx, len(papers), num_attempts,
                        eval_config, client, token_counter, printer, c
                    )
                    future_to_paper[future] = (idx, paper)

                # Collect results as they complete
                for future in as_completed(future_to_paper):
                    idx, paper_info = future_to_paper[future]
                    try:
                        result = future.result()
                        all_results.append(result)
                        printer.print(f"\n✓ Completed paper {idx}/{len(papers)}: {paper_info['title'][:60]}...")
                    except Exception as e:
                        printer.print(f"\n✗ ERROR evaluating paper {idx}/{len(papers)}: {str(e)}")
                        import traceback
                        printer.print(traceback.format_exc())
        else:
            # SEQUENTIAL PAPERS (but may parallelize attempts inside evaluate_paper)
            # This is used when:
            # - parallelize_over == 'attempts': Papers run sequentially, attempts run in parallel
            # - max_workers == 1: Everything runs sequentially
            for idx, paper in enumerate(papers, 1):
                result = evaluate_paper(
                    paper, idx, len(papers), num_attempts,
                    eval_config, client, token_counter, printer, c
                )
                all_results.append(result)

        # Convert token counter to dict for cost calculation
        total_tokens = token_counter.to_dict()

        runtime = time.time() - start_time

        # Calculate metrics
        print(f"\n{c['C']}Calculating metrics...{c['R']}")
        judge_prompts = eval_config.get('judge_prompts', ['eval/prompts/dataset_eval_match_exact.txt'])
        method_names = [extract_method_name(p) for p in judge_prompts]
        metrics = calculate_metrics(all_results, method_names)

        # Add domain discovery metrics if in domain_search mode
        eval_mode = eval_config.get('evaluation', {}).get('mode', 'solution_search')
        if eval_mode == 'domain_search':
            domain_metrics = calculate_domain_metrics(all_results)
            metrics['domain_discovery'] = domain_metrics

        # Calculate cost
        cost_breakdown = calculate_cost(
            total_tokens,
            workflow_model=eval_config['model']['name'],
            judge_model=eval_config['judge_model'],
            domain_judge_model=eval_config.get('domain_judge_model')
        )

        # Generate report
        generate_report(
            all_results, metrics, output_dir, run_id,
            runtime, cost_breakdown, total_tokens, eval_config, c
        )

    finally:
        sys.stdout = original_stdout
        tee_output.close()


if __name__ == "__main__":
    main()
