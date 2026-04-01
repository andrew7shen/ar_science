#!/usr/bin/env python3
"""Compare embedding diversity analysis results across different runs.

This script compares embedding diversity metrics across three runs (No-Domain,
Cross-Domain, AR) with pretty-printed output similar to baseline_eval format.

Usage:
    # Option 1: Edit CONFIG section at top of script, then run:
    python compare_embedding_diversity.py

    # Option 2: Pass arguments via command-line:
    python compare_embedding_diversity.py \\
      --run-dirs /path/to/run1 /path/to/run2 /path/to/run3 \\
      [--paper-ids 1,28,92,206,228] \\
      [--output-dir results/comparison_TIMESTAMP]
"""

import json
import argparse
import sys
from pathlib import Path
from datetime import datetime
from statistics import mean
import os
import numpy as np
from dotenv import load_dotenv

# Import utilities from analyze_embedding_diversity and embedding_viz_utils
from analyze_embedding_diversity import (
    EmbeddingCache,
    normalize_domain,
    normalize_title,
    extract_all_domains,
    extract_all_solutions,
    load_dataset,
    safe_mean,
    safe_std,
)
from embedding_viz_utils import calculate_pairwise_distances, calculate_vendi_score

# Load environment variables
load_dotenv()

# ANSI color codes
C = '\033[96m'  # Cyan
B = '\033[94m'  # Blue
G = '\033[92m'  # Green
Y = '\033[93m'  # Yellow
R = '\033[0m'   # Reset

CONFIG = {
    'run_dirs': [],
    'paper_ids': None,
    'output_dir': None,
    'use_cleaned_titles': False,
}


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


def is_multi_model_mode(run_dirs):
    """Check if run_dirs is in multi-model format (list of lists).

    Args:
        run_dirs: Either a flat list or list of lists

    Returns:
        bool: True if multi-model mode (list of lists), False otherwise
    """
    return run_dirs and isinstance(run_dirs[0], list)


def extract_run_id_from_dirname(dirname):
    """Parse analysis directory name to extract run IDs.

    Format: {timestamp}_papers_{range}_AR_{run_id}_Cross_{run_id}_No_{run_id}
    Run IDs are in format YYYYMMDD_HHMMSS (2 parts joined by underscore)

    Args:
        dirname: Analysis directory name

    Returns:
        Dict mapping run_type -> run_id (or None if not found)
    """
    parts = dirname.split('_')
    run_ids = {}

    # Find AR run ID
    try:
        ar_idx = parts.index('AR')
        if ar_idx + 2 < len(parts):
            # Check if next part is "None"
            if parts[ar_idx + 1] != 'None':
                # Run ID is two parts: timestamp_time
                run_id = f"{parts[ar_idx + 1]}_{parts[ar_idx + 2]}"
                run_ids['AR'] = run_id
    except (ValueError, IndexError):
        pass

    # Find Cross-domain run ID
    try:
        cross_idx = parts.index('Cross')
        if cross_idx + 2 < len(parts):
            if parts[cross_idx + 1] != 'None':
                run_id = f"{parts[cross_idx + 1]}_{parts[cross_idx + 2]}"
                run_ids['Cross-domain'] = run_id
    except (ValueError, IndexError):
        pass

    # Find No-domain run ID
    try:
        no_idx = parts.index('No')
        if no_idx + 2 < len(parts):
            if parts[no_idx + 1] != 'None':
                run_id = f"{parts[no_idx + 1]}_{parts[no_idx + 2]}"
                run_ids['No-domain'] = run_id
    except (ValueError, IndexError):
        pass

    return run_ids


def load_results_for_model(model_dirs, dataset_path='', use_cleaned_titles=True):
    """Load results.json files for a single model across multiple analysis directories.

    Args:
        model_dirs: List of analysis directory paths for one model
        dataset_path: Path to dataset.json for metadata lookup
        use_cleaned_titles: If True, use cleaned_solution_title (fallback to solution_title)

    Returns:
        Dict mapping original_paper_id -> {run_name: paper_data}
    """
    repo_root = Path(__file__).parent.parent.parent
    model_results = {}

    # Load dataset for metadata
    paper_index = load_dataset(dataset_path)

    for analysis_dir in model_dirs:
        dirname = Path(analysis_dir).name
        run_ids = extract_run_id_from_dirname(dirname)

        # Load results.json or results_cleaned.json for each run type
        filename = 'results_cleaned.json' if use_cleaned_titles else 'results.json'
        for run_name, run_id in run_ids.items():
            results_path = repo_root / 'eval' / 'results' / 'dataset_eval' / run_id / filename

            if not results_path.exists():
                raise FileNotFoundError(f"{filename} not found at {results_path}")

            with open(results_path, 'r') as f:
                run_data = json.load(f)

            # Extract original paper indices from config
            original_paper_indices = None
            if 'config' in run_data and 'evaluation' in run_data['config']:
                original_paper_indices = run_data['config']['evaluation'].get('paper_indices')

            # Process each paper in the results
            for paper_idx, paper_result in enumerate(run_data['paper_results']):
                # Get original paper ID
                original_paper_id = None
                if original_paper_indices and paper_idx < len(original_paper_indices):
                    original_paper_id = original_paper_indices[paper_idx]
                else:
                    original_paper_id = paper_idx

                # Initialize paper entry if needed
                if original_paper_id not in model_results:
                    model_results[original_paper_id] = {}

                # Store paper data with extracted domains/solutions
                domains = extract_all_domains(paper_result)
                solutions = extract_all_solutions(paper_result, use_cleaned_titles=use_cleaned_titles)

                # Get metadata from dataset
                paper_title = paper_result['paper_title']
                dataset_metadata = paper_index.get(paper_title.strip(), {})

                model_results[original_paper_id][run_name] = {
                    'paper_title': paper_title,
                    'domains': domains,
                    'solutions': solutions,
                    'metadata': {
                        'paper_title': paper_title,
                        'target_domain': paper_result.get('target_domain') or dataset_metadata.get('target_domain'),
                        'base_domain': dataset_metadata.get('base_domain'),
                        'difficulty': dataset_metadata.get('difficulty'),
                        'well_known': dataset_metadata.get('well_known', False),
                        'year': dataset_metadata.get('year'),
                    }
                }

    return model_results


def merge_multi_model_papers(all_models_data, paper_ids=None):
    """Merge papers across models by original_paper_id.

    Args:
        all_models_data: List of dicts, each mapping original_paper_id -> {run_name: paper_data}
        paper_ids: Optional list of paper IDs to filter

    Returns:
        Dict mapping original_paper_id -> {run_name: {'domains': [...], 'solutions': [...], 'metadata': {...}}}
    """
    # Find common papers across all models
    common_paper_ids = set(all_models_data[0].keys())
    for model_data in all_models_data[1:]:
        common_paper_ids &= set(model_data.keys())

    # Filter by paper_ids if provided
    if paper_ids:
        common_paper_ids &= set(paper_ids)
        missing = set(paper_ids) - common_paper_ids
        if missing:
            print(f"{Y}Warning: Papers not found in all models: {sorted(missing)}{R}")

    # Merge data for common papers
    merged_papers = {}

    for paper_id in sorted(common_paper_ids):
        merged_papers[paper_id] = {}

        # Get all run types across all models for this paper
        all_run_types = set()
        for model_data in all_models_data:
            if paper_id in model_data:
                all_run_types.update(model_data[paper_id].keys())

        # Merge each run type
        for run_name in all_run_types:
            combined_domains = []
            combined_solutions = []
            metadata = None

            # Collect domains and solutions from all models
            for model_data in all_models_data:
                if paper_id in model_data and run_name in model_data[paper_id]:
                    paper_data = model_data[paper_id][run_name]
                    combined_domains.extend(paper_data['domains'])
                    combined_solutions.extend(paper_data['solutions'])

                    # Keep metadata from first model
                    if metadata is None:
                        metadata = paper_data['metadata']

            merged_papers[paper_id][run_name] = {
                'domains': combined_domains,
                'solutions': combined_solutions,
                'metadata': metadata
            }

    return merged_papers


def recalculate_paper_metrics(combined_domains, combined_solutions, target_domain, cache, api_key):
    """Recalculate metrics on combined multi-model data.

    Args:
        combined_domains: List of all domains from all models
        combined_solutions: List of all solutions from all models
        target_domain: Target domain for distance calculation
        cache: EmbeddingCache instance
        api_key: OpenAI API key

    Returns:
        Dict with recalculated metrics
    """
    metrics = {}

    # Normalize domains and solutions
    normalized_domains = [normalize_domain(d) for d in combined_domains if d]
    normalized_solutions = [normalize_title(s) for s in combined_solutions if s]

    # Remove duplicates for unique counts
    unique_domains = list(set(normalized_domains))
    unique_solutions = list(set(normalized_solutions))

    # Basic counts
    metrics['num_domains_discovered'] = len(normalized_domains)
    metrics['num_unique_domains'] = len(unique_domains)
    metrics['num_solutions_discovered'] = len(normalized_solutions)
    metrics['num_unique_solutions'] = len(unique_solutions)

    # Get embeddings for domains
    if len(unique_domains) >= 2:
        domain_embeddings, _, _ = cache.get_embeddings(unique_domains, api_key)
        distances = calculate_pairwise_distances(domain_embeddings)
        metrics['domain_embedding_avg_pairwise_distance'] = safe_mean(distances.tolist())
        metrics['domain_embedding_std_pairwise_distance'] = safe_std(distances.tolist())

        # Domain Vendi score - use ALL domains including duplicates
        all_domain_embeddings, _, _ = cache.get_embeddings(normalized_domains, api_key)
        domain_vendi_result = calculate_vendi_score(all_domain_embeddings, kernel='cosine')
        metrics['domain_vendi_cosine'] = domain_vendi_result['vendi_score']
    else:
        metrics['domain_embedding_avg_pairwise_distance'] = None
        metrics['domain_embedding_std_pairwise_distance'] = None
        metrics['domain_vendi_cosine'] = 1.0 if len(unique_domains) == 1 else None

    # Calculate distance to target domain
    if target_domain and len(unique_domains) > 0:
        normalized_target = normalize_domain(target_domain)
        target_emb, _, _ = cache.get_embeddings([normalized_target], api_key)
        domain_embeddings, _, _ = cache.get_embeddings(unique_domains, api_key)

        target_distances = [np.linalg.norm(emb - target_emb[0]) for emb in domain_embeddings]
        metrics['domain_distance_to_target'] = safe_mean(target_distances)
    else:
        metrics['domain_distance_to_target'] = None

    # Get embeddings for solutions
    if len(unique_solutions) >= 2:
        solution_embeddings, _, _ = cache.get_embeddings(unique_solutions, api_key)
        distances = calculate_pairwise_distances(solution_embeddings)
        metrics['solution_embedding_avg_pairwise_distance'] = safe_mean(distances.tolist())
        metrics['solution_embedding_std_pairwise_distance'] = safe_std(distances.tolist())

        # Solution Vendi score - use ALL solutions including duplicates
        all_solution_embeddings, _, _ = cache.get_embeddings(normalized_solutions, api_key)
        solution_vendi_result = calculate_vendi_score(all_solution_embeddings, kernel='cosine')
        metrics['solution_vendi_cosine'] = solution_vendi_result['vendi_score']
    else:
        metrics['solution_embedding_avg_pairwise_distance'] = None
        metrics['solution_embedding_std_pairwise_distance'] = None
        metrics['solution_vendi_cosine'] = 1.0 if len(unique_solutions) == 1 else None

    return metrics


def process_multi_model_runs(run_dirs, paper_ids, cache, api_key, use_cleaned_titles=True):
    """Process multi-model runs and recalculate metrics.

    Args:
        run_dirs: List of lists, each inner list contains directories for one model
        paper_ids: Optional list of paper IDs to filter
        cache: EmbeddingCache instance
        api_key: OpenAI API key
        use_cleaned_titles: If True, use cleaned_solution_title (fallback to solution_title)

    Returns:
        Dict mapping run_type -> run_data (compatible with single-model format)
    """
    print(f"Multi-model mode detected: {len(run_dirs)} models")

    # Load results for each model
    all_models_data = []
    for model_idx, model_dirs in enumerate(run_dirs):
        print(f"  Loading model {model_idx + 1} from {len(model_dirs)} directories...")
        model_data = load_results_for_model(model_dirs, use_cleaned_titles=use_cleaned_titles)
        all_models_data.append(model_data)
        print(f"    Found {len(model_data)} papers")

    # Merge papers across models
    print("\nMerging papers across models...")
    merged_papers = merge_multi_model_papers(all_models_data, paper_ids)
    print(f"  {len(merged_papers)} papers found in all models")

    # Recalculate metrics for combined data
    print("\nRecalculating metrics on combined data...")
    all_per_paper_metrics = []

    total_papers = len(merged_papers)
    for paper_idx, (original_paper_id, run_data) in enumerate(merged_papers.items()):
        if (paper_idx + 1) % 10 == 0:
            print(f"  Processing paper {paper_idx + 1}/{total_papers}...")

        for run_name, data in run_data.items():
            combined_domains = data['domains']
            combined_solutions = data['solutions']
            metadata = data['metadata']
            target_domain = metadata.get('target_domain')

            # Recalculate metrics
            metrics = recalculate_paper_metrics(
                combined_domains,
                combined_solutions,
                target_domain,
                cache,
                api_key
            )

            # Add metadata - include all fields needed for display
            metrics['paper_title'] = metadata['paper_title']
            metrics['original_paper_id'] = original_paper_id
            metrics['run_name'] = run_name
            metrics['base_domain'] = metadata.get('base_domain', 'Unknown')
            metrics['target_domain'] = target_domain or 'Unknown'
            metrics['difficulty'] = metadata.get('difficulty', 'Unknown')
            metrics['well_known'] = metadata.get('well_known', False)
            metrics['year'] = metadata.get('year', 'Unknown')

            all_per_paper_metrics.append(metrics)

    # Save cache
    cache.save()
    print("  Saved embedding cache")

    # Build structure compatible with single-model mode
    runs = {}
    for run_name in ['No-domain', 'Cross-domain', 'AR']:
        run_papers = [p for p in all_per_paper_metrics if p['run_name'] == run_name]

        if run_papers:
            runs[run_name] = {
                'data': {
                    'metadata': {},
                    'per_paper_metrics': run_papers,
                    'per_run_aggregates': {},
                    'metadata_correlations': {}
                },
                'paths': [str(d) for model_dirs in run_dirs for d in model_dirs]
            }

    return runs


def load_and_validate_runs(run_dirs, cache=None, api_key=None, use_cleaned_titles=True):
    """Load analysis.json from directories and validate run types.

    Args:
        run_dirs: List of 1-3 directory paths containing analysis.json
                 Can handle multiple run types in same directory
                 OR list of lists for multi-model mode
        cache: Optional EmbeddingCache for multi-model mode
        api_key: Optional API key for multi-model mode
        use_cleaned_titles: If True, use cleaned_solution_title (fallback to solution_title)

    Returns:
        Dict mapping run_type -> analysis_data

    Raises:
        ValueError: If missing required run types
    """
    # Check for multi-model mode
    if is_multi_model_mode(run_dirs):
        if cache is None or api_key is None:
            raise ValueError("Multi-model mode requires cache and api_key parameters")
        return process_multi_model_runs(run_dirs, paper_ids=None, cache=cache, api_key=api_key, use_cleaned_titles=use_cleaned_titles)

    # Single-model mode (existing logic)
    runs = {}

    for run_dir in run_dirs:
        analysis_path = Path(run_dir) / 'analysis.json'
        if not analysis_path.exists():
            raise FileNotFoundError(f"analysis.json not found in {run_dir}")

        with open(analysis_path, 'r') as f:
            data = json.load(f)

        # Get all papers and group by run_name
        per_paper = data.get('per_paper_metrics', [])
        if not per_paper:
            raise ValueError(f"No per_paper_metrics found in {run_dir}")

        # Group papers by run_name (handles mixed runs in same directory)
        papers_by_run = {}
        for paper in per_paper:
            run_name = paper.get('run_name')
            if not run_name:
                raise ValueError(f"No run_name found in paper metrics in {run_dir}")

            if run_name not in papers_by_run:
                papers_by_run[run_name] = []
            papers_by_run[run_name].append(paper)

        # Add each run type found in this directory
        for run_name, papers in papers_by_run.items():
            if run_name in runs:
                # Merge papers if run type already exists from another directory
                existing_data = runs[run_name]['data']
                existing_data['per_paper_metrics'].extend(papers)
                # Track additional directory
                runs[run_name]['paths'].append(run_dir)
            else:
                # Create new entry for this run type
                runs[run_name] = {
                    'data': {
                        'metadata': data.get('metadata', {}),
                        'per_paper_metrics': papers,
                        'per_run_aggregates': data.get('per_run_aggregates', {}),
                        'metadata_correlations': data.get('metadata_correlations', {})
                    },
                    'paths': [run_dir]  # Track all contributing directories
                }

    # Validate all three run types present
    required_types = {'No-domain', 'Cross-domain', 'AR'}
    found_types = set(runs.keys())

    if found_types != required_types:
        missing = required_types - found_types
        raise ValueError(f"Missing run types: {missing}. Found: {found_types}")

    return runs


def build_comparison_structure(runs, paper_ids=None):
    """Match papers by original_paper_id across runs.

    Args:
        runs: Dict mapping run_type -> run_data
        paper_ids: Optional list of original_paper_ids to filter

    Returns:
        Dict mapping original_paper_id -> comparison data
    """
    # Build lookup: {original_paper_id: {run_name: paper_data}}
    paper_lookup = {}

    for run_name, run_info in runs.items():
        per_paper = run_info['data'].get('per_paper_metrics', [])

        for paper in per_paper:
            orig_id = paper.get('original_paper_id')
            if orig_id is None:
                continue

            if orig_id not in paper_lookup:
                paper_lookup[orig_id] = {}

            paper_lookup[orig_id][run_name] = paper

    # Filter papers that exist in all three runs
    complete_papers = {
        pid: data for pid, data in paper_lookup.items()
        if len(data) == 3  # Must have all three runs
    }

    # Filter by paper_ids if provided
    if paper_ids:
        complete_papers = {
            pid: data for pid, data in complete_papers.items()
            if pid in paper_ids
        }

        # Warn about missing papers
        missing = set(paper_ids) - set(complete_papers.keys())
        if missing:
            print(f"{Y}Warning: Papers not found in all runs: {sorted(missing)}{R}")

    # Sort by original_paper_id
    sorted_papers = dict(sorted(complete_papers.items()))

    return sorted_papers


def format_delta(value, baseline, is_int=False):
    """Format delta value with color and percentage.

    Args:
        value: Current value
        baseline: Baseline value for comparison
        is_int: Whether to format as integer

    Returns:
        Tuple of (formatted_string_with_colors, raw_string_without_colors)
    """
    if value is None or baseline is None:
        return "N/A", "N/A"

    delta = value - baseline

    # Calculate percentage
    if baseline == 0:
        if delta == 0:
            pct_str = "0.0%"
        else:
            pct_str = "N/A"
    else:
        pct = (delta / baseline) * 100
        pct_str = f"{pct:+.1f}%"

    # Format delta
    if is_int:
        delta_str = f"{delta:+d}"
    else:
        delta_str = f"{delta:+.4f}"

    # Build raw string (no colors)
    if pct_str == "N/A":
        raw_str = f"{delta_str} ({pct_str})"
    else:
        raw_str = f"{delta_str} ({pct_str})"

    # Apply color
    color = G if delta > 0 else Y
    colored_str = f"{color}{raw_str}{R}"

    return colored_str, raw_str


def print_per_paper_comparison(paper_id, papers_data, output=None):
    """Print comparison for a single paper.

    Args:
        paper_id: Original paper ID
        papers_data: Dict mapping run_name -> paper_data
        output: Optional TeeOutput for dual output
    """
    def p(msg=""):
        """Print to output or stdout."""
        if output:
            output.write(msg + "\n")
        else:
            print(msg)

    # Get baseline (No-domain) and comparison runs
    no_domain = papers_data.get('No-domain', {})
    cross_domain = papers_data.get('Cross-domain', {})
    ar = papers_data.get('AR', {})

    # Extract metadata (same across all runs)
    title = no_domain.get('paper_title', 'Unknown')
    base_dom = no_domain.get('base_domain', 'Unknown')
    target_dom = no_domain.get('target_domain', 'Unknown')
    difficulty = no_domain.get('difficulty', 'Unknown')
    well_known = no_domain.get('well_known', False)
    year = no_domain.get('year', 'Unknown')

    # Print header
    p()
    p("=" * 100)
    p(f"{B}PAPER ID {paper_id}: {title}{R}")
    p("=" * 100)
    p(f"Analogy: {base_dom} → {target_dom}")
    p(f"Difficulty: {difficulty} | Well-known: {well_known} | Year: {year}")

    # Domain exploration metrics
    p()
    p(f"{C}DOMAIN EXPLORATION METRICS{R}")
    p(f"{'Metric':<40}{'No-Domain':<15}{'Cross-Domain':<15}{'AR':<15}{'Δ Cross':<25}{'Δ AR':<25}")
    p("-" * 135)

    # Unique Domains
    metric = "Unique Domains ↑"
    no_val = no_domain.get('num_unique_domains')
    cross_val = cross_domain.get('num_unique_domains')
    ar_val = ar.get('num_unique_domains')

    cross_delta, cross_raw = format_delta(cross_val, no_val, is_int=True)
    ar_delta, ar_raw = format_delta(ar_val, no_val, is_int=True)

    no_str = str(no_val) if no_val is not None else "N/A"
    cross_str = str(cross_val) if cross_val is not None else "N/A"
    ar_str = str(ar_val) if ar_val is not None else "N/A"

    p(f"{metric:<40}{no_str:<15}{cross_str:<15}{ar_str:<15}{cross_delta}{' '*(25-len(cross_raw))}{ar_delta}")

    # Domain Diversity
    metric = "Domain Diversity ↑"
    no_val = no_domain.get('domain_embedding_avg_pairwise_distance')
    cross_val = cross_domain.get('domain_embedding_avg_pairwise_distance')
    ar_val = ar.get('domain_embedding_avg_pairwise_distance')

    cross_delta, cross_raw = format_delta(cross_val, no_val)
    ar_delta, ar_raw = format_delta(ar_val, no_val)

    no_str = f"{no_val:.4f}" if no_val is not None else "N/A"
    cross_str = f"{cross_val:.4f}" if cross_val is not None else "N/A"
    ar_str = f"{ar_val:.4f}" if ar_val is not None else "N/A"

    p(f"{metric:<40}{no_str:<15}{cross_str:<15}{ar_str:<15}{cross_delta}{' '*(25-len(cross_raw))}{ar_delta}")

    # Domain Vendi Score
    metric = "Domain Vendi Score ↑"
    no_val = no_domain.get('domain_vendi_cosine')
    cross_val = cross_domain.get('domain_vendi_cosine')
    ar_val = ar.get('domain_vendi_cosine')

    cross_delta, cross_raw = format_delta(cross_val, no_val)
    ar_delta, ar_raw = format_delta(ar_val, no_val)

    no_str = f"{no_val:.4f}" if no_val is not None else "N/A"
    cross_str = f"{cross_val:.4f}" if cross_val is not None else "N/A"
    ar_str = f"{ar_val:.4f}" if ar_val is not None else "N/A"

    p(f"{metric:<40}{no_str:<15}{cross_str:<15}{ar_str:<15}{cross_delta}{' '*(25-len(cross_raw))}{ar_delta}")

    # Distance to Target
    metric = "Distance to Target ↑"
    no_val = no_domain.get('domain_distance_to_target')
    cross_val = cross_domain.get('domain_distance_to_target')
    ar_val = ar.get('domain_distance_to_target')

    cross_delta, cross_raw = format_delta(cross_val, no_val)
    ar_delta, ar_raw = format_delta(ar_val, no_val)

    no_str = f"{no_val:.4f}" if no_val is not None else "N/A"
    cross_str = f"{cross_val:.4f}" if cross_val is not None else "N/A"
    ar_str = f"{ar_val:.4f}" if ar_val is not None else "N/A"

    p(f"{metric:<40}{no_str:<15}{cross_str:<15}{ar_str:<15}{cross_delta}{' '*(25-len(cross_raw))}{ar_delta}")

    # Solution generation metrics
    p()
    p(f"{C}SOLUTION GENERATION METRICS{R}")
    p(f"{'Metric':<40}{'No-Domain':<15}{'Cross-Domain':<15}{'AR':<15}{'Δ Cross':<25}{'Δ AR':<25}")
    p("-" * 135)

    # Unique Solutions
    metric = "Unique Solutions ↑"
    no_val = no_domain.get('num_unique_solutions')
    cross_val = cross_domain.get('num_unique_solutions')
    ar_val = ar.get('num_unique_solutions')

    cross_delta, cross_raw = format_delta(cross_val, no_val, is_int=True)
    ar_delta, ar_raw = format_delta(ar_val, no_val, is_int=True)

    no_str = str(no_val) if no_val is not None else "N/A"
    cross_str = str(cross_val) if cross_val is not None else "N/A"
    ar_str = str(ar_val) if ar_val is not None else "N/A"

    p(f"{metric:<40}{no_str:<15}{cross_str:<15}{ar_str:<15}{cross_delta}{' '*(25-len(cross_raw))}{ar_delta}")

    # Solution Diversity
    metric = "Solution Diversity ↑"
    no_val = no_domain.get('solution_embedding_avg_pairwise_distance')
    cross_val = cross_domain.get('solution_embedding_avg_pairwise_distance')
    ar_val = ar.get('solution_embedding_avg_pairwise_distance')

    cross_delta, cross_raw = format_delta(cross_val, no_val)
    ar_delta, ar_raw = format_delta(ar_val, no_val)

    no_str = f"{no_val:.4f}" if no_val is not None else "N/A"
    cross_str = f"{cross_val:.4f}" if cross_val is not None else "N/A"
    ar_str = f"{ar_val:.4f}" if ar_val is not None else "N/A"

    p(f"{metric:<40}{no_str:<15}{cross_str:<15}{ar_str:<15}{cross_delta}{' '*(25-len(cross_raw))}{ar_delta}")

    # Solution Vendi Score
    metric = "Solution Vendi Score ↑"
    no_val = no_domain.get('solution_vendi_cosine')
    cross_val = cross_domain.get('solution_vendi_cosine')
    ar_val = ar.get('solution_vendi_cosine')

    cross_delta, cross_raw = format_delta(cross_val, no_val)
    ar_delta, ar_raw = format_delta(ar_val, no_val)

    no_str = f"{no_val:.4f}" if no_val is not None else "N/A"
    cross_str = f"{cross_val:.4f}" if cross_val is not None else "N/A"
    ar_str = f"{ar_val:.4f}" if ar_val is not None else "N/A"

    p(f"{metric:<40}{no_str:<15}{cross_str:<15}{ar_str:<15}{cross_delta}{' '*(25-len(cross_raw))}{ar_delta}")

    # Performance summary
    p()
    p(f"{C}PERFORMANCE SUMMARY{R}")
    p("-" * 100)

    # Count which method is best for key metrics (higher is better)
    ar_best = 0
    cross_best = 0
    no_best = 0
    total_metrics = 0

    # Metrics where higher is better
    metrics = [
        ('num_unique_domains', 'Unique Domains'),
        ('domain_embedding_avg_pairwise_distance', 'Domain Diversity'),
        ('domain_vendi_cosine', 'Domain Vendi Score'),
        ('num_unique_solutions', 'Unique Solutions'),
        ('solution_embedding_avg_pairwise_distance', 'Solution Diversity'),
        ('solution_vendi_cosine', 'Solution Vendi Score')
    ]

    for metric_key, _ in metrics:
        no_val = no_domain.get(metric_key)
        cross_val = cross_domain.get(metric_key)
        ar_val_m = ar.get(metric_key)

        if all(v is not None for v in [no_val, cross_val, ar_val_m]):
            total_metrics += 1
            max_val = max(no_val, cross_val, ar_val_m)
            if ar_val_m == max_val:
                ar_best += 1
            elif cross_val == max_val:
                cross_best += 1
            else:
                no_best += 1

    p(f"Metrics where AR is best: {ar_best}/{total_metrics}")
    p(f"Metrics where Cross-Domain is best: {cross_best}/{total_metrics}")
    p(f"Metrics where No-Domain is best: {no_best}/{total_metrics}")


def print_aggregate_statistics(comparison_data, output=None):
    """Print aggregate statistics across all papers.

    Args:
        comparison_data: Dict mapping paper_id -> papers_data
        output: Optional TeeOutput for dual output

    Returns:
        Tuple of (improvements_count, deltas) for JSON output
    """
    def p(msg=""):
        """Print to output or stdout."""
        if output:
            output.write(msg + "\n")
        else:
            print(msg)

    num_papers = len(comparison_data)

    p()
    p("=" * 100)
    p(f"{C}AGGREGATE STATISTICS ACROSS {num_papers} PAPERS{R}")
    p("=" * 100)

    # Collect all metrics across papers
    metrics_data = {
        'No-domain': {
            'num_unique_domains': [],
            'domain_diversity': [],
            'domain_vendi_cosine': [],
            'num_unique_solutions': [],
            'solution_diversity': [],
            'solution_vendi_cosine': []
        },
        'Cross-domain': {
            'num_unique_domains': [],
            'domain_diversity': [],
            'domain_vendi_cosine': [],
            'num_unique_solutions': [],
            'solution_diversity': [],
            'solution_vendi_cosine': []
        },
        'AR': {
            'num_unique_domains': [],
            'domain_diversity': [],
            'domain_vendi_cosine': [],
            'num_unique_solutions': [],
            'solution_diversity': [],
            'solution_vendi_cosine': []
        }
    }

    # Collect deltas
    deltas = {
        'cross': {
            'num_unique_domains': [],
            'domain_diversity': [],
            'domain_vendi_cosine': [],
            'num_unique_solutions': [],
            'solution_diversity': [],
            'solution_vendi_cosine': []
        },
        'ar': {
            'num_unique_domains': [],
            'domain_diversity': [],
            'domain_vendi_cosine': [],
            'num_unique_solutions': [],
            'solution_diversity': [],
            'solution_vendi_cosine': []
        }
    }

    # Count wins for each method (which method has the max value)
    wins_count = {
        'No-domain': {
            'num_unique_domains': 0,
            'domain_diversity': 0,
            'domain_vendi_cosine': 0,
            'num_unique_solutions': 0,
            'solution_diversity': 0,
            'solution_vendi_cosine': 0
        },
        'Cross-domain': {
            'num_unique_domains': 0,
            'domain_diversity': 0,
            'domain_vendi_cosine': 0,
            'num_unique_solutions': 0,
            'solution_diversity': 0,
            'solution_vendi_cosine': 0
        },
        'AR': {
            'num_unique_domains': 0,
            'domain_diversity': 0,
            'domain_vendi_cosine': 0,
            'num_unique_solutions': 0,
            'solution_diversity': 0,
            'solution_vendi_cosine': 0
        }
    }

    for paper_id, papers_data in comparison_data.items():
        for run_name in ['No-domain', 'Cross-domain', 'AR']:
            paper = papers_data.get(run_name, {})

            # Collect metrics
            if paper.get('num_unique_domains') is not None:
                metrics_data[run_name]['num_unique_domains'].append(paper['num_unique_domains'])
            if paper.get('domain_embedding_avg_pairwise_distance') is not None:
                metrics_data[run_name]['domain_diversity'].append(paper['domain_embedding_avg_pairwise_distance'])
            if paper.get('domain_vendi_cosine') is not None:
                metrics_data[run_name]['domain_vendi_cosine'].append(paper['domain_vendi_cosine'])
            if paper.get('num_unique_solutions') is not None:
                metrics_data[run_name]['num_unique_solutions'].append(paper['num_unique_solutions'])
            if paper.get('solution_embedding_avg_pairwise_distance') is not None:
                metrics_data[run_name]['solution_diversity'].append(paper['solution_embedding_avg_pairwise_distance'])
            if paper.get('solution_vendi_cosine') is not None:
                metrics_data[run_name]['solution_vendi_cosine'].append(paper['solution_vendi_cosine'])

        # Calculate deltas
        no_domain = papers_data.get('No-domain', {})
        cross_domain = papers_data.get('Cross-domain', {})
        ar_paper = papers_data.get('AR', {})

        # Unique domains delta and wins
        if all(p.get('num_unique_domains') is not None for p in [no_domain, cross_domain, ar_paper]):
            cross_delta = cross_domain['num_unique_domains'] - no_domain['num_unique_domains']
            ar_delta = ar_paper['num_unique_domains'] - no_domain['num_unique_domains']
            deltas['cross']['num_unique_domains'].append(cross_delta)
            deltas['ar']['num_unique_domains'].append(ar_delta)

            # Find which method has max value (only count if no tie)
            values = {
                'No-domain': no_domain['num_unique_domains'],
                'Cross-domain': cross_domain['num_unique_domains'],
                'AR': ar_paper['num_unique_domains']
            }
            max_val = max(values.values())
            methods_with_max = [method for method, val in values.items() if val == max_val]
            if len(methods_with_max) == 1:
                wins_count[methods_with_max[0]]['num_unique_domains'] += 1

        # Domain diversity delta and wins
        if all(p.get('domain_embedding_avg_pairwise_distance') is not None for p in [no_domain, cross_domain, ar_paper]):
            cross_delta = cross_domain['domain_embedding_avg_pairwise_distance'] - no_domain['domain_embedding_avg_pairwise_distance']
            ar_delta = ar_paper['domain_embedding_avg_pairwise_distance'] - no_domain['domain_embedding_avg_pairwise_distance']
            deltas['cross']['domain_diversity'].append(cross_delta)
            deltas['ar']['domain_diversity'].append(ar_delta)

            # Find which method has max value (only count if no tie)
            values = {
                'No-domain': no_domain['domain_embedding_avg_pairwise_distance'],
                'Cross-domain': cross_domain['domain_embedding_avg_pairwise_distance'],
                'AR': ar_paper['domain_embedding_avg_pairwise_distance']
            }
            max_val = max(values.values())
            methods_with_max = [method for method, val in values.items() if val == max_val]
            if len(methods_with_max) == 1:
                wins_count[methods_with_max[0]]['domain_diversity'] += 1

        # Domain Vendi score delta and wins
        if all(p.get('domain_vendi_cosine') is not None for p in [no_domain, cross_domain, ar_paper]):
            cross_delta = cross_domain['domain_vendi_cosine'] - no_domain['domain_vendi_cosine']
            ar_delta = ar_paper['domain_vendi_cosine'] - no_domain['domain_vendi_cosine']
            deltas['cross']['domain_vendi_cosine'].append(cross_delta)
            deltas['ar']['domain_vendi_cosine'].append(ar_delta)

            # Find which method has max value (only count if no tie)
            values = {
                'No-domain': no_domain['domain_vendi_cosine'],
                'Cross-domain': cross_domain['domain_vendi_cosine'],
                'AR': ar_paper['domain_vendi_cosine']
            }
            max_val = max(values.values())
            methods_with_max = [method for method, val in values.items() if val == max_val]
            if len(methods_with_max) == 1:
                wins_count[methods_with_max[0]]['domain_vendi_cosine'] += 1

        # Unique solutions delta and wins
        if all(p.get('num_unique_solutions') is not None for p in [no_domain, cross_domain, ar_paper]):
            cross_delta = cross_domain['num_unique_solutions'] - no_domain['num_unique_solutions']
            ar_delta = ar_paper['num_unique_solutions'] - no_domain['num_unique_solutions']
            deltas['cross']['num_unique_solutions'].append(cross_delta)
            deltas['ar']['num_unique_solutions'].append(ar_delta)

            # Find which method has max value (only count if no tie)
            values = {
                'No-domain': no_domain['num_unique_solutions'],
                'Cross-domain': cross_domain['num_unique_solutions'],
                'AR': ar_paper['num_unique_solutions']
            }
            max_val = max(values.values())
            methods_with_max = [method for method, val in values.items() if val == max_val]
            if len(methods_with_max) == 1:
                wins_count[methods_with_max[0]]['num_unique_solutions'] += 1

        # Solution diversity delta and wins
        if all(p.get('solution_embedding_avg_pairwise_distance') is not None for p in [no_domain, cross_domain, ar_paper]):
            cross_delta = cross_domain['solution_embedding_avg_pairwise_distance'] - no_domain['solution_embedding_avg_pairwise_distance']
            ar_delta = ar_paper['solution_embedding_avg_pairwise_distance'] - no_domain['solution_embedding_avg_pairwise_distance']
            deltas['cross']['solution_diversity'].append(cross_delta)
            deltas['ar']['solution_diversity'].append(ar_delta)

            # Find which method has max value (only count if no tie)
            values = {
                'No-domain': no_domain['solution_embedding_avg_pairwise_distance'],
                'Cross-domain': cross_domain['solution_embedding_avg_pairwise_distance'],
                'AR': ar_paper['solution_embedding_avg_pairwise_distance']
            }
            max_val = max(values.values())
            methods_with_max = [method for method, val in values.items() if val == max_val]
            if len(methods_with_max) == 1:
                wins_count[methods_with_max[0]]['solution_diversity'] += 1

        # Solution Vendi score delta and wins
        if all(p.get('solution_vendi_cosine') is not None for p in [no_domain, cross_domain, ar_paper]):
            cross_delta = cross_domain['solution_vendi_cosine'] - no_domain['solution_vendi_cosine']
            ar_delta = ar_paper['solution_vendi_cosine'] - no_domain['solution_vendi_cosine']
            deltas['cross']['solution_vendi_cosine'].append(cross_delta)
            deltas['ar']['solution_vendi_cosine'].append(ar_delta)

            # Find which method has max value (only count if no tie)
            values = {
                'No-domain': no_domain['solution_vendi_cosine'],
                'Cross-domain': cross_domain['solution_vendi_cosine'],
                'AR': ar_paper['solution_vendi_cosine']
            }
            max_val = max(values.values())
            methods_with_max = [method for method, val in values.items() if val == max_val]
            if len(methods_with_max) == 1:
                wins_count[methods_with_max[0]]['solution_vendi_cosine'] += 1

    # Helper to format improvement
    def format_improvement(delta_list, baseline_mean):
        if not delta_list:
            return "N/A", "N/A"
        delta_mean = mean(delta_list)
        if baseline_mean == 0:
            pct_str = "N/A"
        else:
            pct = (delta_mean / baseline_mean) * 100
            pct_str = f"{pct:+.1f}%"
        return delta_mean, pct_str

    # Print combined table with mean values and deltas
    p()
    p(f"{C}AGGREGATE METRICS{R}")
    p(f"{'Metric':<40}{'No-Domain':<15}{'Cross-Domain':<15}{'AR':<15}{'Δ Cross':<25}{'Δ AR':<25}{'WR No':<15}{'WR Cross':<15}{'WR AR':<15}")
    p("-" * 180)

    # Unique Domains
    metric = "Unique Domains ↑"
    no_mean = mean(metrics_data['No-domain']['num_unique_domains']) if metrics_data['No-domain']['num_unique_domains'] else 0
    cross_mean = mean(metrics_data['Cross-domain']['num_unique_domains']) if metrics_data['Cross-domain']['num_unique_domains'] else 0
    ar_mean = mean(metrics_data['AR']['num_unique_domains']) if metrics_data['AR']['num_unique_domains'] else 0

    cross_delta_mean, cross_pct = format_improvement(deltas['cross']['num_unique_domains'], no_mean)
    ar_delta_mean, ar_pct = format_improvement(deltas['ar']['num_unique_domains'], no_mean)

    if cross_delta_mean != "N/A":
        cross_raw = f"{cross_delta_mean:+.2f} ({cross_pct})"
        cross_color = G if cross_delta_mean > 0 else Y
        cross_delta_str = f"{cross_color}{cross_raw}{R}"
    else:
        cross_delta_str = "N/A"
        cross_raw = "N/A"

    if ar_delta_mean != "N/A":
        ar_raw = f"{ar_delta_mean:+.2f} ({ar_pct})"
        ar_color = G if ar_delta_mean > 0 else Y
        ar_delta_str = f"{ar_color}{ar_raw}{R}"
    else:
        ar_delta_str = "N/A"
        ar_raw = "N/A"

    # Calculate win rates for all three methods
    no_wins = wins_count['No-domain']['num_unique_domains']
    cross_wins = wins_count['Cross-domain']['num_unique_domains']
    ar_wins = wins_count['AR']['num_unique_domains']
    total = len(deltas['cross']['num_unique_domains'])
    no_win_pct = (no_wins / total * 100) if total > 0 else 0
    cross_win_pct = (cross_wins / total * 100) if total > 0 else 0
    ar_win_pct = (ar_wins / total * 100) if total > 0 else 0

    # Determine best method and color appropriately
    best_pct = max(no_win_pct, cross_win_pct, ar_win_pct)
    no_win_str = f"{no_win_pct:.1f}%"
    cross_win_str = f"{cross_win_pct:.1f}%"
    ar_win_str = f"{ar_win_pct:.1f}%"
    no_win_colored = f"{G}{no_win_str}{R}" if no_win_pct == best_pct else no_win_str
    cross_win_colored = f"{G}{cross_win_str}{R}" if cross_win_pct == best_pct else cross_win_str
    ar_win_colored = f"{G}{ar_win_str}{R}" if ar_win_pct == best_pct else ar_win_str

    p(f"{metric:<40}{no_mean:<15.2f}{cross_mean:<15.2f}{ar_mean:<15.2f}{cross_delta_str}{' '*(25-len(cross_raw))}{ar_delta_str}{' '*(25-len(ar_raw))}{no_win_colored}{' '*(15-len(no_win_str))}{cross_win_colored}{' '*(15-len(cross_win_str))}{ar_win_colored}")

    # Domain Diversity
    metric = "Domain Diversity ↑"
    no_mean = mean(metrics_data['No-domain']['domain_diversity']) if metrics_data['No-domain']['domain_diversity'] else 0
    cross_mean = mean(metrics_data['Cross-domain']['domain_diversity']) if metrics_data['Cross-domain']['domain_diversity'] else 0
    ar_mean = mean(metrics_data['AR']['domain_diversity']) if metrics_data['AR']['domain_diversity'] else 0

    cross_delta_mean, cross_pct = format_improvement(deltas['cross']['domain_diversity'], no_mean)
    ar_delta_mean, ar_pct = format_improvement(deltas['ar']['domain_diversity'], no_mean)

    if cross_delta_mean != "N/A":
        cross_raw = f"{cross_delta_mean:+.4f} ({cross_pct})"
        cross_color = G if cross_delta_mean > 0 else Y
        cross_delta_str = f"{cross_color}{cross_raw}{R}"
    else:
        cross_delta_str = "N/A"
        cross_raw = "N/A"

    if ar_delta_mean != "N/A":
        ar_raw = f"{ar_delta_mean:+.4f} ({ar_pct})"
        ar_color = G if ar_delta_mean > 0 else Y
        ar_delta_str = f"{ar_color}{ar_raw}{R}"
    else:
        ar_delta_str = "N/A"
        ar_raw = "N/A"

    # Calculate win rates for all three methods
    no_wins = wins_count['No-domain']['domain_diversity']
    cross_wins = wins_count['Cross-domain']['domain_diversity']
    ar_wins = wins_count['AR']['domain_diversity']
    total = len(deltas['cross']['domain_diversity'])
    no_win_pct = (no_wins / total * 100) if total > 0 else 0
    cross_win_pct = (cross_wins / total * 100) if total > 0 else 0
    ar_win_pct = (ar_wins / total * 100) if total > 0 else 0

    # Determine best method and color appropriately
    best_pct = max(no_win_pct, cross_win_pct, ar_win_pct)
    no_win_str = f"{no_win_pct:.1f}%"
    cross_win_str = f"{cross_win_pct:.1f}%"
    ar_win_str = f"{ar_win_pct:.1f}%"
    no_win_colored = f"{G}{no_win_str}{R}" if no_win_pct == best_pct else no_win_str
    cross_win_colored = f"{G}{cross_win_str}{R}" if cross_win_pct == best_pct else cross_win_str
    ar_win_colored = f"{G}{ar_win_str}{R}" if ar_win_pct == best_pct else ar_win_str

    p(f"{metric:<40}{no_mean:<15.4f}{cross_mean:<15.4f}{ar_mean:<15.4f}{cross_delta_str}{' '*(25-len(cross_raw))}{ar_delta_str}{' '*(25-len(ar_raw))}{no_win_colored}{' '*(15-len(no_win_str))}{cross_win_colored}{' '*(15-len(cross_win_str))}{ar_win_colored}")

    # Domain Vendi Score
    metric = "Domain Vendi Score ↑"
    no_mean = mean(metrics_data['No-domain']['domain_vendi_cosine']) if metrics_data['No-domain']['domain_vendi_cosine'] else 0
    cross_mean = mean(metrics_data['Cross-domain']['domain_vendi_cosine']) if metrics_data['Cross-domain']['domain_vendi_cosine'] else 0
    ar_mean = mean(metrics_data['AR']['domain_vendi_cosine']) if metrics_data['AR']['domain_vendi_cosine'] else 0

    cross_delta_mean, cross_pct = format_improvement(deltas['cross']['domain_vendi_cosine'], no_mean)
    ar_delta_mean, ar_pct = format_improvement(deltas['ar']['domain_vendi_cosine'], no_mean)

    if cross_delta_mean != "N/A":
        cross_raw = f"{cross_delta_mean:+.4f} ({cross_pct})"
        cross_color = G if cross_delta_mean > 0 else Y
        cross_delta_str = f"{cross_color}{cross_raw}{R}"
    else:
        cross_delta_str = "N/A"
        cross_raw = "N/A"

    if ar_delta_mean != "N/A":
        ar_raw = f"{ar_delta_mean:+.4f} ({ar_pct})"
        ar_color = G if ar_delta_mean > 0 else Y
        ar_delta_str = f"{ar_color}{ar_raw}{R}"
    else:
        ar_delta_str = "N/A"
        ar_raw = "N/A"

    # Calculate win rates for all three methods
    no_wins = wins_count['No-domain']['domain_vendi_cosine']
    cross_wins = wins_count['Cross-domain']['domain_vendi_cosine']
    ar_wins = wins_count['AR']['domain_vendi_cosine']
    total = len(deltas['cross']['domain_vendi_cosine'])
    no_win_pct = (no_wins / total * 100) if total > 0 else 0
    cross_win_pct = (cross_wins / total * 100) if total > 0 else 0
    ar_win_pct = (ar_wins / total * 100) if total > 0 else 0

    # Determine best method and color appropriately
    best_pct = max(no_win_pct, cross_win_pct, ar_win_pct)
    no_win_str = f"{no_win_pct:.1f}%"
    cross_win_str = f"{cross_win_pct:.1f}%"
    ar_win_str = f"{ar_win_pct:.1f}%"
    no_win_colored = f"{G}{no_win_str}{R}" if no_win_pct == best_pct else no_win_str
    cross_win_colored = f"{G}{cross_win_str}{R}" if cross_win_pct == best_pct else cross_win_str
    ar_win_colored = f"{G}{ar_win_str}{R}" if ar_win_pct == best_pct else ar_win_str

    p(f"{metric:<40}{no_mean:<15.4f}{cross_mean:<15.4f}{ar_mean:<15.4f}{cross_delta_str}{' '*(25-len(cross_raw))}{ar_delta_str}{' '*(25-len(ar_raw))}{no_win_colored}{' '*(15-len(no_win_str))}{cross_win_colored}{' '*(15-len(cross_win_str))}{ar_win_colored}")

    # Unique Solutions
    metric = "Unique Solutions ↑"
    no_mean = mean(metrics_data['No-domain']['num_unique_solutions']) if metrics_data['No-domain']['num_unique_solutions'] else 0
    cross_mean = mean(metrics_data['Cross-domain']['num_unique_solutions']) if metrics_data['Cross-domain']['num_unique_solutions'] else 0
    ar_mean = mean(metrics_data['AR']['num_unique_solutions']) if metrics_data['AR']['num_unique_solutions'] else 0

    cross_delta_mean, cross_pct = format_improvement(deltas['cross']['num_unique_solutions'], no_mean)
    ar_delta_mean, ar_pct = format_improvement(deltas['ar']['num_unique_solutions'], no_mean)

    if cross_delta_mean != "N/A":
        cross_raw = f"{cross_delta_mean:+.2f} ({cross_pct})"
        cross_color = G if cross_delta_mean > 0 else Y
        cross_delta_str = f"{cross_color}{cross_raw}{R}"
    else:
        cross_delta_str = "N/A"
        cross_raw = "N/A"

    if ar_delta_mean != "N/A":
        ar_raw = f"{ar_delta_mean:+.2f} ({ar_pct})"
        ar_color = G if ar_delta_mean > 0 else Y
        ar_delta_str = f"{ar_color}{ar_raw}{R}"
    else:
        ar_delta_str = "N/A"
        ar_raw = "N/A"

    # Calculate win rates for all three methods
    no_wins = wins_count['No-domain']['num_unique_solutions']
    cross_wins = wins_count['Cross-domain']['num_unique_solutions']
    ar_wins = wins_count['AR']['num_unique_solutions']
    total = len(deltas['cross']['num_unique_solutions'])
    no_win_pct = (no_wins / total * 100) if total > 0 else 0
    cross_win_pct = (cross_wins / total * 100) if total > 0 else 0
    ar_win_pct = (ar_wins / total * 100) if total > 0 else 0

    # Determine best method and color appropriately
    best_pct = max(no_win_pct, cross_win_pct, ar_win_pct)
    no_win_str = f"{no_win_pct:.1f}%"
    cross_win_str = f"{cross_win_pct:.1f}%"
    ar_win_str = f"{ar_win_pct:.1f}%"
    no_win_colored = f"{G}{no_win_str}{R}" if no_win_pct == best_pct else no_win_str
    cross_win_colored = f"{G}{cross_win_str}{R}" if cross_win_pct == best_pct else cross_win_str
    ar_win_colored = f"{G}{ar_win_str}{R}" if ar_win_pct == best_pct else ar_win_str

    p(f"{metric:<40}{no_mean:<15.2f}{cross_mean:<15.2f}{ar_mean:<15.2f}{cross_delta_str}{' '*(25-len(cross_raw))}{ar_delta_str}{' '*(25-len(ar_raw))}{no_win_colored}{' '*(15-len(no_win_str))}{cross_win_colored}{' '*(15-len(cross_win_str))}{ar_win_colored}")

    # Solution Diversity
    metric = "Solution Diversity ↑"
    no_mean = mean(metrics_data['No-domain']['solution_diversity']) if metrics_data['No-domain']['solution_diversity'] else 0
    cross_mean = mean(metrics_data['Cross-domain']['solution_diversity']) if metrics_data['Cross-domain']['solution_diversity'] else 0
    ar_mean = mean(metrics_data['AR']['solution_diversity']) if metrics_data['AR']['solution_diversity'] else 0

    cross_delta_mean, cross_pct = format_improvement(deltas['cross']['solution_diversity'], no_mean)
    ar_delta_mean, ar_pct = format_improvement(deltas['ar']['solution_diversity'], no_mean)

    if cross_delta_mean != "N/A":
        cross_raw = f"{cross_delta_mean:+.4f} ({cross_pct})"
        cross_color = G if cross_delta_mean > 0 else Y
        cross_delta_str = f"{cross_color}{cross_raw}{R}"
    else:
        cross_delta_str = "N/A"
        cross_raw = "N/A"

    if ar_delta_mean != "N/A":
        ar_raw = f"{ar_delta_mean:+.4f} ({ar_pct})"
        ar_color = G if ar_delta_mean > 0 else Y
        ar_delta_str = f"{ar_color}{ar_raw}{R}"
    else:
        ar_delta_str = "N/A"
        ar_raw = "N/A"

    # Calculate win rates for all three methods
    no_wins = wins_count['No-domain']['solution_diversity']
    cross_wins = wins_count['Cross-domain']['solution_diversity']
    ar_wins = wins_count['AR']['solution_diversity']
    total = len(deltas['cross']['solution_diversity'])
    no_win_pct = (no_wins / total * 100) if total > 0 else 0
    cross_win_pct = (cross_wins / total * 100) if total > 0 else 0
    ar_win_pct = (ar_wins / total * 100) if total > 0 else 0

    # Determine best method and color appropriately
    best_pct = max(no_win_pct, cross_win_pct, ar_win_pct)
    no_win_str = f"{no_win_pct:.1f}%"
    cross_win_str = f"{cross_win_pct:.1f}%"
    ar_win_str = f"{ar_win_pct:.1f}%"
    no_win_colored = f"{G}{no_win_str}{R}" if no_win_pct == best_pct else no_win_str
    cross_win_colored = f"{G}{cross_win_str}{R}" if cross_win_pct == best_pct else cross_win_str
    ar_win_colored = f"{G}{ar_win_str}{R}" if ar_win_pct == best_pct else ar_win_str

    p(f"{metric:<40}{no_mean:<15.4f}{cross_mean:<15.4f}{ar_mean:<15.4f}{cross_delta_str}{' '*(25-len(cross_raw))}{ar_delta_str}{' '*(25-len(ar_raw))}{no_win_colored}{' '*(15-len(no_win_str))}{cross_win_colored}{' '*(15-len(cross_win_str))}{ar_win_colored}")

    # Solution Vendi Score
    metric = "Solution Vendi Score ↑"
    no_mean = mean(metrics_data['No-domain']['solution_vendi_cosine']) if metrics_data['No-domain']['solution_vendi_cosine'] else 0
    cross_mean = mean(metrics_data['Cross-domain']['solution_vendi_cosine']) if metrics_data['Cross-domain']['solution_vendi_cosine'] else 0
    ar_mean = mean(metrics_data['AR']['solution_vendi_cosine']) if metrics_data['AR']['solution_vendi_cosine'] else 0

    cross_delta_mean, cross_pct = format_improvement(deltas['cross']['solution_vendi_cosine'], no_mean)
    ar_delta_mean, ar_pct = format_improvement(deltas['ar']['solution_vendi_cosine'], no_mean)

    if cross_delta_mean != "N/A":
        cross_raw = f"{cross_delta_mean:+.4f} ({cross_pct})"
        cross_color = G if cross_delta_mean > 0 else Y
        cross_delta_str = f"{cross_color}{cross_raw}{R}"
    else:
        cross_delta_str = "N/A"
        cross_raw = "N/A"

    if ar_delta_mean != "N/A":
        ar_raw = f"{ar_delta_mean:+.4f} ({ar_pct})"
        ar_color = G if ar_delta_mean > 0 else Y
        ar_delta_str = f"{ar_color}{ar_raw}{R}"
    else:
        ar_delta_str = "N/A"
        ar_raw = "N/A"

    # Calculate win rates for all three methods
    no_wins = wins_count['No-domain']['solution_vendi_cosine']
    cross_wins = wins_count['Cross-domain']['solution_vendi_cosine']
    ar_wins = wins_count['AR']['solution_vendi_cosine']
    total = len(deltas['cross']['solution_vendi_cosine'])
    no_win_pct = (no_wins / total * 100) if total > 0 else 0
    cross_win_pct = (cross_wins / total * 100) if total > 0 else 0
    ar_win_pct = (ar_wins / total * 100) if total > 0 else 0

    # Determine best method and color appropriately
    best_pct = max(no_win_pct, cross_win_pct, ar_win_pct)
    no_win_str = f"{no_win_pct:.1f}%"
    cross_win_str = f"{cross_win_pct:.1f}%"
    ar_win_str = f"{ar_win_pct:.1f}%"
    no_win_colored = f"{G}{no_win_str}{R}" if no_win_pct == best_pct else no_win_str
    cross_win_colored = f"{G}{cross_win_str}{R}" if cross_win_pct == best_pct else cross_win_str
    ar_win_colored = f"{G}{ar_win_str}{R}" if ar_win_pct == best_pct else ar_win_str

    p(f"{metric:<40}{no_mean:<15.4f}{cross_mean:<15.4f}{ar_mean:<15.4f}{cross_delta_str}{' '*(25-len(cross_raw))}{ar_delta_str}{' '*(25-len(ar_raw))}{no_win_colored}{' '*(15-len(no_win_str))}{cross_win_colored}{' '*(15-len(cross_win_str))}{ar_win_colored}")

    p()
    p("=" * 180)

    return wins_count, deltas


def save_json_results(comparison_data, runs, output_dir, wins_count, deltas):
    """Save comparison results to JSON.

    Args:
        comparison_data: Dict mapping paper_id -> papers_data
        runs: Dict mapping run_type -> run_data
        output_dir: Output directory path
        wins_count: Dict of win counts for each method per metric
        deltas: Dict of delta lists per metric
    """
    # Calculate aggregate win rates for all methods
    win_rates = {}
    for method in ['No-domain', 'Cross-domain', 'AR']:
        win_rates[method] = {}
        for metric_key in ['num_unique_domains', 'domain_diversity', 'domain_vendi_cosine', 'num_unique_solutions', 'solution_diversity', 'solution_vendi_cosine']:
            wins = wins_count[method][metric_key]
            total = len(deltas['cross'][metric_key])  # Use any deltas list for total count
            win_rate = (wins / total * 100) if total > 0 else 0
            win_rates[method][metric_key] = {
                'wins': wins,
                'total': total,
                'win_rate_pct': win_rate
            }

    results = {
        'metadata': {
            'timestamp': datetime.now().isoformat(),
            'num_papers': len(comparison_data),
            'runs': {
                run_name: [str(p) for p in run_info['paths']]
                for run_name, run_info in runs.items()
            }
        },
        'per_paper_comparisons': comparison_data,
        'aggregate_win_rates': win_rates
    }

    output_path = Path(output_dir) / 'comparison_results.json'
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    return output_path


def main():
    parser = argparse.ArgumentParser(
        description='Compare embedding diversity analysis results across different runs'
    )

    parser.add_argument(
        '--run-dirs',
        nargs='+',
        default=CONFIG['run_dirs'],
        help='One or more directory paths containing analysis.json files (script will find No-Domain, Cross-Domain, and AR runs by run_name)'
    )

    parser.add_argument(
        '--paper-ids',
        type=str,
        default=CONFIG['paper_ids'],
        help='Comma-separated list of original_paper_ids to compare (optional, default: all)'
    )

    parser.add_argument(
        '--output-dir',
        type=str,
        default=CONFIG['output_dir'],
        help='Output directory for results (optional, default: auto-generated with timestamp)'
    )

    parser.add_argument(
        '--use-cleaned-titles',
        action='store_true',
        default=CONFIG['use_cleaned_titles'],
        help='Use cleaned_solution_title instead of solution_title for diversity analysis'
    )

    args = parser.parse_args()

    # Validate run_dirs are configured
    run_dirs = args.run_dirs
    if not isinstance(run_dirs, list) or len(run_dirs) == 0:
        print(f"{Y}Error: Please configure run_dirs in the CONFIG section at the top of the script,{R}")
        print(f"{Y}       or pass them via --run-dirs command-line argument.{R}")
        print()
        print("Example:")
        print("  python compare_embedding_diversity.py \\")
        print("    --run-dirs dir1 dir2 dir3")
        return 1

    # Parse paper IDs if provided
    paper_ids = None
    if args.paper_ids:
        # Handle both list and string formats
        if isinstance(args.paper_ids, list):
            paper_ids = args.paper_ids
        elif isinstance(args.paper_ids, str):
            paper_ids = [int(pid.strip()) for pid in args.paper_ids.split(',')]

    # Initialize cache and API key if multi-model mode
    cache = None
    api_key = None
    if is_multi_model_mode(run_dirs):
        print(f"{C}Multi-model mode detected: Combining solutions from {len(run_dirs)} models{R}")
        print("Initializing embedding cache and API...")

        # Get API key
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            print(f"{Y}Error: OPENAI_API_KEY not found in environment.{R}")
            print(f"{Y}       Multi-model mode requires API access to recalculate metrics.{R}")
            return 1

        # Initialize cache
        cache = EmbeddingCache('.embedding_cache.json')
        print(f"  ✓ Loaded {len(cache.cache)} cached embeddings")

    # Load and validate runs
    print("\nLoading and validating runs...")
    try:
        runs = load_and_validate_runs(run_dirs, cache=cache, api_key=api_key, use_cleaned_titles=args.use_cleaned_titles)
        print(f"  ✓ Loaded 3 runs: {', '.join(sorted(runs.keys()))}")
    except Exception as e:
        print(f"Error loading runs: {e}")
        import traceback
        traceback.print_exc()
        return 1

    # Build comparison structure
    print("\nMatching papers across runs...")
    comparison_data = build_comparison_structure(runs, paper_ids)
    print(f"  ✓ Found {len(comparison_data)} papers present in all three runs")

    if not comparison_data:
        print("No papers found in all three runs. Exiting.")
        return 1

    # Create output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        repo_root = Path(__file__).parent.parent.parent
        output_dir = repo_root / 'eval' / 'results' / 'dataset_eval' / 'embedding_diversity_comparison' / timestamp

    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nOutput directory: {output_dir}")

    # Set up TeeOutput for dual output
    terminal_output_path = output_dir / 'terminal_output.log'
    original_stdout = sys.stdout
    tee = TeeOutput(terminal_output_path, original_stdout)
    sys.stdout = tee

    try:
        # Print header
        print()
        print("=" * 100)
        print(f"{C}EMBEDDING DIVERSITY COMPARISON{R}")
        print("=" * 100)
        print(f"Timestamp: {datetime.now().isoformat()}")
        print(f"Papers: {len(comparison_data)}")
        print(f"Runs: {', '.join(sorted(runs.keys()))}")
        print("=" * 100)

        # Print per-paper comparisons
        for paper_id in sorted(comparison_data.keys()):
            print_per_paper_comparison(paper_id, comparison_data[paper_id], output=tee)

        # Print aggregate statistics
        wins_count, deltas = print_aggregate_statistics(comparison_data, output=tee)

        # Save JSON results
        json_path = save_json_results(comparison_data, runs, output_dir, wins_count, deltas)

        print()
        print(f"{G}Results saved to:{R}")
        print(f"  - {terminal_output_path}")
        print(f"  - {json_path}")
        print()

    finally:
        sys.stdout = original_stdout
        tee.close()

    return 0


if __name__ == '__main__':
    sys.exit(main())
