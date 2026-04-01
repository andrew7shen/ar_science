#!/usr/bin/env python3
"""Analyze embedding diversity metrics without generating visualizations.

This script calculates embedding diversity metrics for domains and solutions
across multiple runs, analyzes correlations with paper metadata, and outputs
results to JSON and terminal.
"""

import json
import argparse
from pathlib import Path
from datetime import datetime
from collections import Counter
import numpy as np
from scipy.spatial.distance import pdist
import os
from dotenv import load_dotenv

# Import shared utilities
from embedding_viz_utils import (
    generate_embeddings,
    calculate_pairwise_distances,
    calculate_vendi_score
)

# Load environment variables
load_dotenv()

CONFIG = {
    'no_domain_run': '',
    'ar_run': '',
    'cross_domain_run': '',
    'dataset_path': '',
    'output_dir': None,
    'paper_indices': '',
    'cache_embeddings': True,
    'embedding_cache_file': '',
    'use_cleaned_titles': False,
}


class EmbeddingCache:
    """Cache text embeddings to avoid duplicate API calls."""

    def __init__(self, cache_file=None):
        self.cache_file = cache_file
        self.cache = {}
        if cache_file and Path(cache_file).exists():
            try:
                with open(cache_file, 'r') as f:
                    self.cache = json.load(f)
                print(f"Loaded {len(self.cache)} cached embeddings from {cache_file}")
            except json.JSONDecodeError as e:
                print(f"Warning: Cache file {cache_file} is corrupted (JSON error at char {e.pos})")
                print(f"Starting with empty cache. You may want to delete the corrupted file.")
                self.cache = {}
            except Exception as e:
                print(f"Warning: Could not load cache file {cache_file}: {e}")
                print(f"Starting with empty cache.")
                self.cache = {}

    def get(self, text):
        """Get embedding from cache if it exists."""
        return self.cache.get(text)

    def set(self, text, embedding):
        """Store embedding in cache."""
        self.cache[text] = embedding

    def save(self):
        """Save cache to file."""
        if self.cache_file:
            with open(self.cache_file, 'w') as f:
                json.dump(self.cache, f)

    def get_embeddings(self, texts, api_key):
        """Get embeddings from cache or generate new ones via API.

        Returns:
            Tuple of (embeddings_array, total_tokens, cost)
        """
        embeddings = []
        uncached_texts = []
        uncached_indices = []

        # Check cache first
        for i, text in enumerate(texts):
            cached = self.get(text)
            if cached:
                embeddings.append(cached)
            else:
                embeddings.append(None)
                uncached_texts.append(text)
                uncached_indices.append(i)

        # Generate embeddings for uncached texts
        total_tokens = 0
        total_cost = 0.0

        if uncached_texts:
            new_embeddings, tokens, cost = generate_embeddings(uncached_texts, api_key)
            total_tokens = tokens
            total_cost = cost

            # Update cache and embeddings list
            for idx, embedding in zip(uncached_indices, new_embeddings):
                embeddings[idx] = embedding.tolist()
                self.set(texts[idx], embedding.tolist())

        return np.array(embeddings), total_tokens, total_cost


def normalize_domain(domain):
    """Normalize domain name to lowercase with underscores."""
    if domain is None:
        return None
    return domain.lower().replace(' ', '_')


def normalize_title(title):
    """Normalize solution title to lowercase for comparison."""
    if title is None:
        return None
    return title.lower().strip()


def resolve_run_path(run_id_or_path, use_cleaned_titles=False):
    """Resolve run ID or path to full path to results.json or results_cleaned.json.

    Args:
        run_id_or_path: Run ID or full path
        use_cleaned_titles: If True, use results_cleaned.json instead of results.json
    """
    if run_id_or_path is None or run_id_or_path == 'None':
        return None

    # Determine which file to use
    filename = 'results_cleaned.json' if use_cleaned_titles else 'results.json'

    # Already a full path
    if '/' in run_id_or_path or run_id_or_path.endswith('.json'):
        # If it's a full path to results.json and we want cleaned, replace it
        if use_cleaned_titles and run_id_or_path.endswith('results.json'):
            return run_id_or_path.replace('results.json', 'results_cleaned.json')
        return run_id_or_path

    # Try relative to repo root
    repo_root_path = Path(__file__).parent.parent.parent
    results_path = repo_root_path / 'eval' / 'results' / 'dataset_eval' / run_id_or_path / filename
    if results_path.exists():
        return str(results_path)

    # Fall back to simple relative path
    return f'eval/results/dataset_eval/{run_id_or_path}/{filename}'


def parse_paper_indices(indices_arg, max_papers):
    """Parse paper indices argument into a list of indices.

    Args:
        indices_arg: Can be:
            - "all" or None: all papers
            - "0-5": range of papers (inclusive)
            - "0,2,4": comma-separated indices
            - [0, 1, 2]: list of indices

    Returns:
        List of paper indices to analyze
    """
    if indices_arg is None or indices_arg == 'all':
        return list(range(max_papers))

    if isinstance(indices_arg, list):
        return indices_arg

    # Parse string argument
    indices = []

    # Handle comma-separated values
    parts = str(indices_arg).split(',')

    for part in parts:
        part = part.strip()

        # Handle range (e.g., "0-5")
        if '-' in part:
            start, end = part.split('-')
            start = int(start.strip())
            end = int(end.strip())
            indices.extend(range(start, end + 1))
        else:
            # Single index
            indices.append(int(part))

    # Validate indices
    valid_indices = [i for i in indices if 0 <= i < max_papers]

    if len(valid_indices) != len(indices):
        print(f"Warning: Some indices were out of range (0-{max_papers-1})")

    return sorted(list(set(valid_indices)))  # Remove duplicates and sort


def load_dataset(dataset_path):
    """Load dataset.json and build paper title -> metadata index.

    Returns:
        Dictionary mapping paper titles to metadata
    """
    repo_root = Path(__file__).parent.parent.parent
    full_path = repo_root / dataset_path

    with open(full_path, 'r') as f:
        dataset = json.load(f)

    # Build index
    paper_index = {}
    for paper in dataset.get('papers', []):
        title = paper.get('title', '').strip()
        paper_index[title] = {
            'base_domain': paper.get('base_domain'),
            'target_domain': paper.get('target_domain'),
            'domain_distance': paper.get('domain_distance'),
            'difficulty': paper.get('difficulty'),
            'well_known': paper.get('likely_well_known_example', False),
            'structural_reasoning': paper.get('structural_reasoning'),
            'analogy_depth': paper.get('analogy_depth'),
            'year': paper.get('year'),
            'analogy_description': paper.get('analogy_description', '')
        }

    return paper_index


def load_run_results(run_path):
    """Load results.json from directory or file.

    Returns:
        Tuple of (run_data, original_paper_indices)
        - run_data: Full results JSON
        - original_paper_indices: List of original paper IDs from dataset, or None
    """
    if run_path is None:
        return None, None

    path = Path(run_path)
    if path.is_dir():
        path = path / 'results.json'

    with open(path, 'r') as f:
        data = json.load(f)

    # Try to extract original paper indices from config
    original_paper_indices = None
    if 'config' in data and 'evaluation' in data['config']:
        original_paper_indices = data['config']['evaluation'].get('paper_indices')

    return data, original_paper_indices


def extract_all_domains(paper_data):
    """Extract all discovered domains from paper results."""
    domains = []
    for attempt in paper_data.get('attempts', []):
        if 'discovered_domains' in attempt:
            normalized = [normalize_domain(d) for d in attempt['discovered_domains']]
            domains.extend(normalized)
    return domains


def extract_all_solutions(paper_data, use_cleaned_titles=True):
    """Extract all solution titles from paper results.

    Args:
        paper_data: Paper results data
        use_cleaned_titles: If True, use cleaned_solution_title (fallback to solution_title)
                           If False, use solution_title only
    """
    solutions = []
    for attempt in paper_data.get('attempts', []):
        for evaluation in attempt.get('evaluations', []):
            if use_cleaned_titles:
                # Use cleaned title if available, fallback to original
                title = evaluation.get('cleaned_solution_title') or evaluation.get('solution_title', '')
            else:
                # Use original title only
                title = evaluation.get('solution_title', '')
            solutions.append(title)
    return solutions


def find_paper_metadata(paper_title, paper_index):
    """Lookup metadata by paper title."""
    return paper_index.get(paper_title.strip())


def safe_mean(values):
    """Calculate mean, handling empty lists."""
    if values is None or len(values) == 0:
        return None
    return float(np.mean(values))


def safe_median(values):
    """Calculate median, handling empty lists."""
    if values is None or len(values) == 0:
        return None
    return float(np.median(values))


def safe_std(values):
    """Calculate std, handling empty lists."""
    if values is None or len(values) == 0:
        return None
    return float(np.std(values))


def calculate_percentiles(values, percentiles):
    """Calculate percentiles, handling empty lists."""
    if not values or len(values) == 0:
        return {p: None for p in percentiles}
    return {p: float(np.percentile(values, p)) for p in percentiles}


def calculate_paper_metrics(domains, solutions, domain_embeddings, solution_embeddings,
                           target_domain, target_embedding):
    """Calculate metrics for a single paper.

    Returns:
        Dictionary with per-paper metrics
    """
    metrics = {}

    # Domain metrics
    metrics['num_domains_discovered'] = len(domains)
    metrics['num_unique_domains'] = len(set(domains))

    if len(domain_embeddings) >= 2:
        distances = calculate_pairwise_distances(domain_embeddings)
        metrics['domain_embedding_avg_pairwise_distance'] = safe_mean(distances)
        metrics['domain_embedding_std_pairwise_distance'] = safe_std(distances)

        # Domain Vendi score (uses ALL embeddings including duplicates)
        domain_vendi_result = calculate_vendi_score(domain_embeddings, kernel='cosine')
        metrics['domain_vendi_cosine'] = domain_vendi_result['vendi_score']
    else:
        metrics['domain_embedding_avg_pairwise_distance'] = None
        metrics['domain_embedding_std_pairwise_distance'] = None
        metrics['domain_vendi_cosine'] = 1.0 if len(domain_embeddings) == 1 else None

    # Distance to target domain
    if target_embedding is not None and len(domain_embeddings) > 0:
        target_distances = [np.linalg.norm(emb - target_embedding) for emb in domain_embeddings]
        metrics['domain_distance_to_target'] = safe_mean(target_distances)
    else:
        metrics['domain_distance_to_target'] = None

    # Solution metrics
    metrics['num_solutions_discovered'] = len(solutions)
    metrics['num_unique_solutions'] = len(set([normalize_title(s) for s in solutions]))

    if len(solution_embeddings) >= 2:
        distances = calculate_pairwise_distances(solution_embeddings)
        metrics['solution_embedding_avg_pairwise_distance'] = safe_mean(distances)
        metrics['solution_embedding_std_pairwise_distance'] = safe_std(distances)

        # Solution Vendi score (uses ALL embeddings including duplicates)
        solution_vendi_result = calculate_vendi_score(solution_embeddings, kernel='cosine')
        metrics['solution_vendi_cosine'] = solution_vendi_result['vendi_score']
    else:
        metrics['solution_embedding_avg_pairwise_distance'] = None
        metrics['solution_embedding_std_pairwise_distance'] = None
        metrics['solution_vendi_cosine'] = 1.0 if len(solution_embeddings) == 1 else None

    return metrics


def calculate_run_aggregates(per_paper_metrics):
    """Calculate aggregated statistics across all papers in a run.

    Returns:
        Dictionary with per-run aggregate metrics
    """
    aggregates = {}

    # Collect all domains across papers
    all_domains = []
    domains_per_paper = []
    domain_diversity_values = []
    solution_diversity_values = []
    target_distance_values = []

    for paper in per_paper_metrics:
        # Domains per paper
        domains_per_paper.append(paper['num_unique_domains'])

        # Domain diversity
        if paper['domain_embedding_avg_pairwise_distance'] is not None:
            domain_diversity_values.append(paper['domain_embedding_avg_pairwise_distance'])

        # Solution diversity
        if paper['solution_embedding_avg_pairwise_distance'] is not None:
            solution_diversity_values.append(paper['solution_embedding_avg_pairwise_distance'])

        # Target distance
        if paper['domain_distance_to_target'] is not None:
            target_distance_values.append(paper['domain_distance_to_target'])

    # Domains per paper distribution
    aggregates['domains_per_paper_distribution'] = {
        'mean': safe_mean(domains_per_paper),
        'median': safe_median(domains_per_paper),
        'std': safe_std(domains_per_paper),
        'min': min(domains_per_paper) if domains_per_paper else None,
        'max': max(domains_per_paper) if domains_per_paper else None,
        'percentiles': calculate_percentiles(domains_per_paper, [25, 50, 75])
    }

    # Domain diversity stats
    aggregates['domain_diversity_stats'] = {
        'mean': safe_mean(domain_diversity_values),
        'median': safe_median(domain_diversity_values),
        'std': safe_std(domain_diversity_values),
        'min': min(domain_diversity_values) if domain_diversity_values else None,
        'max': max(domain_diversity_values) if domain_diversity_values else None,
        'percentiles': calculate_percentiles(domain_diversity_values, [25, 50, 75])
    }

    # Solution diversity stats
    aggregates['solution_diversity_stats'] = {
        'mean': safe_mean(solution_diversity_values),
        'median': safe_median(solution_diversity_values),
        'std': safe_std(solution_diversity_values),
        'min': min(solution_diversity_values) if solution_diversity_values else None,
        'max': max(solution_diversity_values) if solution_diversity_values else None,
        'percentiles': calculate_percentiles(solution_diversity_values, [25, 50, 75])
    }

    # Target distance stats
    aggregates['target_distance_stats'] = {
        'mean': safe_mean(target_distance_values),
        'median': safe_median(target_distance_values),
        'std': safe_std(target_distance_values),
        'min': min(target_distance_values) if target_distance_values else None,
        'max': max(target_distance_values) if target_distance_values else None,
        'percentiles': calculate_percentiles(target_distance_values, [25, 50, 75])
    }

    return aggregates


def calculate_metadata_correlations(per_paper_metrics):
    """Group papers by metadata categories and compute descriptive statistics.

    Returns:
        Dictionary with correlations by metadata category
    """
    correlations = {}

    # Define grouping functions
    def group_by_difficulty(papers):
        groups = {'easy': [], 'medium': [], 'hard': []}
        for paper in papers:
            diff = paper.get('difficulty')
            if diff in groups:
                groups[diff].append(paper)
        return groups

    def group_by_well_known(papers):
        groups = {True: [], False: []}
        for paper in papers:
            wk = paper.get('well_known', False)
            groups[wk].append(paper)
        return groups

    def group_by_structural_reasoning(papers):
        groups = {'requires': [], 'does_not_require': []}
        for paper in papers:
            sr = paper.get('structural_reasoning')
            if sr in groups:
                groups[sr].append(paper)
        return groups

    def group_by_domain_distance(papers):
        groups = {'close': [], 'medium': [], 'distant': []}
        for paper in papers:
            dd = paper.get('domain_distance')
            if dd in groups:
                groups[dd].append(paper)
        return groups

    def group_by_analogy_depth(papers):
        groups = {'surface_metaphor': [], 'moderate_transfer': [], 'deep_structural_transfer': []}
        for paper in papers:
            ad = paper.get('analogy_depth')
            if ad in groups:
                groups[ad].append(paper)
        return groups

    def group_by_year(papers):
        groups = {'pre_2000': [], '2000-2010': [], '2011-2020': [], 'post_2020': []}
        for paper in papers:
            year = paper.get('year')
            if year is None:
                continue
            if year < 2000:
                groups['pre_2000'].append(paper)
            elif year <= 2010:
                groups['2000-2010'].append(paper)
            elif year <= 2020:
                groups['2011-2020'].append(paper)
            else:
                groups['post_2020'].append(paper)
        return groups

    def compute_group_stats(group_papers):
        """Compute descriptive stats for a group of papers."""
        if not group_papers:
            return None

        unique_domains = [p['num_unique_domains'] for p in group_papers]
        unique_solutions = [p['num_unique_solutions'] for p in group_papers]
        domain_diversity = [p['domain_embedding_avg_pairwise_distance']
                          for p in group_papers
                          if p['domain_embedding_avg_pairwise_distance'] is not None]
        solution_diversity = [p['solution_embedding_avg_pairwise_distance']
                            for p in group_papers
                            if p['solution_embedding_avg_pairwise_distance'] is not None]
        target_distance = [p['domain_distance_to_target']
                         for p in group_papers
                         if p['domain_distance_to_target'] is not None]

        return {
            'count': len(group_papers),
            'unique_domains': {
                'mean': safe_mean(unique_domains),
                'median': safe_median(unique_domains),
                'std': safe_std(unique_domains),
                'min': min(unique_domains) if unique_domains else None,
                'max': max(unique_domains) if unique_domains else None,
                'percentiles': calculate_percentiles(unique_domains, [25, 50, 75])
            },
            'unique_solutions': {
                'mean': safe_mean(unique_solutions),
                'median': safe_median(unique_solutions),
                'std': safe_std(unique_solutions),
                'min': min(unique_solutions) if unique_solutions else None,
                'max': max(unique_solutions) if unique_solutions else None,
                'percentiles': calculate_percentiles(unique_solutions, [25, 50, 75])
            },
            'domain_diversity': {
                'mean': safe_mean(domain_diversity),
                'median': safe_median(domain_diversity),
                'std': safe_std(domain_diversity),
                'min': min(domain_diversity) if domain_diversity else None,
                'max': max(domain_diversity) if domain_diversity else None,
                'percentiles': calculate_percentiles(domain_diversity, [25, 50, 75])
            },
            'solution_diversity': {
                'mean': safe_mean(solution_diversity),
                'median': safe_median(solution_diversity),
                'std': safe_std(solution_diversity),
                'min': min(solution_diversity) if solution_diversity else None,
                'max': max(solution_diversity) if solution_diversity else None,
                'percentiles': calculate_percentiles(solution_diversity, [25, 50, 75])
            },
            'target_distance': {
                'mean': safe_mean(target_distance),
                'median': safe_median(target_distance),
                'std': safe_std(target_distance),
                'min': min(target_distance) if target_distance else None,
                'max': max(target_distance) if target_distance else None,
                'percentiles': calculate_percentiles(target_distance, [25, 50, 75])
            }
        }

    # Apply groupings
    groupings = {
        'by_difficulty': group_by_difficulty(per_paper_metrics),
        'by_well_known': group_by_well_known(per_paper_metrics),
        'by_structural_reasoning': group_by_structural_reasoning(per_paper_metrics),
        'by_domain_distance': group_by_domain_distance(per_paper_metrics),
        'by_analogy_depth': group_by_analogy_depth(per_paper_metrics),
        'by_year': group_by_year(per_paper_metrics)
    }

    # Compute stats for each grouping
    for grouping_name, groups in groupings.items():
        correlations[grouping_name] = {}
        for group_name, group_papers in groups.items():
            stats = compute_group_stats(group_papers)
            if stats:
                correlations[grouping_name][str(group_name)] = stats

    return correlations


def print_terminal_summary(analysis_results):
    """Print formatted summary to terminal."""
    metadata = analysis_results['metadata']

    print("\n" + "="*80)
    print("EMBEDDING DIVERSITY ANALYSIS SUMMARY")
    print("="*80)

    print(f"\nAnalysis Timestamp: {metadata['analysis_timestamp']}")
    print(f"Dataset: {metadata['dataset_path']}")
    print(f"Papers Analyzed: {metadata['num_papers_analyzed']}")
    print(f"Embedding Model: {metadata['embedding_model']}")
    print(f"Runs Analyzed: {', '.join(metadata['runs_analyzed'])}")
    print(f"Total API Cost: ${metadata['total_api_cost']:.4f}")

    # Per-run aggregates
    print("\n" + "-"*80)
    print("PER-RUN AGGREGATES")
    print("-"*80)

    for run_name, aggregates in analysis_results['per_run_aggregates'].items():
        print(f"\n{run_name}:")
        print(f"  Total Unique Domains: {aggregates.get('total_unique_domains_discovered', 'N/A')}")

        dist = aggregates['domains_per_paper_distribution']
        print(f"  Domains per Paper: mean={dist['mean']:.2f}, median={dist['median']:.2f}, std={dist['std']:.2f}")

        dd = aggregates['domain_diversity_stats']
        if dd['mean'] is not None:
            print(f"  Domain Diversity: mean={dd['mean']:.4f}, median={dd['median']:.4f}, std={dd['std']:.4f}, range=[{dd['min']:.4f}, {dd['max']:.4f}]")

        sd = aggregates['solution_diversity_stats']
        if sd['mean'] is not None:
            print(f"  Solution Diversity: mean={sd['mean']:.4f}, median={sd['median']:.4f}, std={sd['std']:.4f}, range=[{sd['min']:.4f}, {sd['max']:.4f}]")

        td = aggregates['target_distance_stats']
        if td['mean'] is not None:
            print(f"  Target Distance: mean={td['mean']:.4f}, median={td['median']:.4f}, std={td['std']:.4f}, range=[{td['min']:.4f}, {td['max']:.4f}]")

        # Top domains
        if 'most_frequent_domains' in aggregates:
            print(f"\n  Top 10 Most Frequent Domains:")
            for domain, count in aggregates['most_frequent_domains'][:10]:
                print(f"    {domain}: {count}")

    # Metadata correlations
    print("\n" + "-"*80)
    print("METADATA CORRELATIONS")
    print("-"*80)

    for run_name, correlations in analysis_results['metadata_correlations'].items():
        print(f"\n{run_name}:")

        for grouping_name, groups in correlations.items():
            print(f"\n  {grouping_name}:")
            for group_name, stats in groups.items():
                print(f"    {group_name} (n={stats['count']}):")

                # Domain metrics
                ud = stats['unique_domains']
                if ud['mean'] is not None:
                    print(f"      Unique Domains: mean={ud['mean']:.2f}, median={ud['median']:.2f}, range=[{ud['min']}, {ud['max']}]")
                dd = stats['domain_diversity']
                if dd['mean'] is not None:
                    print(f"      Domain Diversity: mean={dd['mean']:.4f}, median={dd['median']:.4f}, range=[{dd['min']:.4f}, {dd['max']:.4f}]")

                # Solution metrics
                us = stats['unique_solutions']
                if us['mean'] is not None:
                    print(f"      Unique Solutions: mean={us['mean']:.2f}, median={us['median']:.2f}, range=[{us['min']}, {us['max']}]")
                sd = stats['solution_diversity']
                if sd['mean'] is not None:
                    print(f"      Solution Diversity: mean={sd['mean']:.4f}, median={sd['median']:.4f}, range=[{sd['min']:.4f}, {sd['max']:.4f}]")

                # Target distance (applies to domains)
                td = stats['target_distance']
                if td['mean'] is not None:
                    print(f"      Target Distance: mean={td['mean']:.4f}, median={td['median']:.4f}, range=[{td['min']:.4f}, {td['max']:.4f}]")

    print("\n" + "="*80)
    print(f"Results saved to: {metadata['output_file']}")
    print("="*80 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description='Analyze embedding diversity metrics without generating visualizations'
    )

    # Run paths
    parser.add_argument('--ar-run', type=str, default=CONFIG['ar_run'],
                       help='Path or run ID for AR run')
    parser.add_argument('--cross-domain-run', type=str, default=CONFIG['cross_domain_run'],
                       help='Path or run ID for cross-domain run')
    parser.add_argument('--no-domain-run', type=str, default=CONFIG['no_domain_run'],
                       help='Path or run ID for no-domain run')

    # Dataset and output
    parser.add_argument('--dataset-path', type=str, default=CONFIG['dataset_path'],
                       help='Path to dataset.json')
    parser.add_argument('--output-dir', type=str, default=CONFIG['output_dir'],
                       help='Output directory (will be auto-generated with timestamp if None)')

    # Paper selection
    parser.add_argument('--paper-indices', type=str, default=CONFIG['paper_indices'],
                       help='Paper indices to analyze: "all", "0-5", "0,2,4", or comma-separated list')

    # Embedding cache
    parser.add_argument('--cache-embeddings', action='store_true',
                       default=CONFIG['cache_embeddings'],
                       help='Cache embeddings to avoid regenerating')
    parser.add_argument('--embedding-cache-file', type=str,
                       default=CONFIG['embedding_cache_file'],
                       help='Path to embedding cache file')

    # Solution title cleaning
    parser.add_argument('--use-cleaned-titles', action='store_true',
                       default=CONFIG['use_cleaned_titles'],
                       help='Use cleaned_solution_title instead of solution_title for diversity analysis')

    args = parser.parse_args()

    # Get API key
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        raise ValueError("OPENAI_API_KEY not found in environment")

    # Resolve run paths
    ar_run_path = resolve_run_path(args.ar_run, use_cleaned_titles=args.use_cleaned_titles)
    cross_domain_run_path = resolve_run_path(args.cross_domain_run, use_cleaned_titles=args.use_cleaned_titles)
    no_domain_run_path = resolve_run_path(args.no_domain_run, use_cleaned_titles=args.use_cleaned_titles)

    # Load dataset
    print("Loading dataset...")
    paper_index = load_dataset(args.dataset_path)
    print(f"Loaded {len(paper_index)} papers from dataset")

    # Load run results
    print("\nLoading run results...")
    runs = {}
    run_paper_mappings = {}  # Store original paper ID mappings for each run

    if ar_run_path:
        runs['AR'], run_paper_mappings['AR'] = load_run_results(ar_run_path)
        print(f"  Loaded AR run: {ar_run_path}")
    if cross_domain_run_path:
        runs['Cross-domain'], run_paper_mappings['Cross-domain'] = load_run_results(cross_domain_run_path)
        print(f"  Loaded Cross-domain run: {cross_domain_run_path}")
    if no_domain_run_path:
        runs['No-domain'], run_paper_mappings['No-domain'] = load_run_results(no_domain_run_path)
        print(f"  Loaded No-domain run: {no_domain_run_path}")

    if not runs:
        raise ValueError("No run results loaded. Please specify at least one run.")

    # Parse paper indices to analyze
    # Get max paper count from first available run
    first_run = list(runs.values())[0]
    max_papers = len(first_run['paper_results'])
    paper_indices_to_analyze = parse_paper_indices(args.paper_indices, max_papers)

    print(f"\nPaper selection:")
    print(f"  Total papers available: {max_papers}")
    print(f"  Papers to analyze: {len(paper_indices_to_analyze)}")
    if len(paper_indices_to_analyze) <= 10:
        print(f"  Indices: {paper_indices_to_analyze}")
    else:
        print(f"  Indices: {paper_indices_to_analyze[:5]} ... {paper_indices_to_analyze[-5:]}")

    # Determine output directory with timestamp
    if args.output_dir is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        # Extract run IDs for directory name
        ar_id = Path(ar_run_path).parent.name if ar_run_path else 'None'
        cross_id = Path(cross_domain_run_path).parent.name if cross_domain_run_path else 'None'
        no_id = Path(no_domain_run_path).parent.name if no_domain_run_path else 'None'

        # Build directory name
        paper_range_str = f"papers_{min(paper_indices_to_analyze)}-{max(paper_indices_to_analyze)}"
        dir_name = f"{timestamp}_{paper_range_str}_AR_{ar_id}_Cross_{cross_id}_No_{no_id}"

        # Use absolute path based on repo root
        repo_root = Path(__file__).parent.parent.parent
        output_dir = repo_root / 'eval' / 'results' / 'dataset_eval' / 'embedding_diversity' / dir_name
    else:
        output_dir = Path(args.output_dir)

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nOutput directory: {output_dir}")

    # Initialize embedding cache
    cache = EmbeddingCache(args.embedding_cache_file if args.cache_embeddings else None)

    # Collect all unique domains and solutions for embedding generation
    print("\nExtracting domains and solutions from all runs...")
    all_domains = set()
    all_solutions = set()
    all_target_domains = set()

    per_paper_data = {}  # Store extracted data per paper per run

    for run_name, run_data in runs.items():
        per_paper_data[run_name] = []
        original_paper_indices = run_paper_mappings.get(run_name)

        for paper_idx, paper_result in enumerate(run_data['paper_results']):
            # Filter by paper indices
            if paper_idx not in paper_indices_to_analyze:
                continue
            paper_title = paper_result['paper_title']

            # Get original paper ID from mapping if available
            original_paper_id = None
            if original_paper_indices and paper_idx < len(original_paper_indices):
                original_paper_id = original_paper_indices[paper_idx]

            # Extract domains and solutions
            domains = extract_all_domains(paper_result)
            solutions = extract_all_solutions(paper_result, use_cleaned_titles=args.use_cleaned_titles)

            # Get metadata
            metadata = find_paper_metadata(paper_title, paper_index)
            target_domain = metadata.get('target_domain') if metadata else None

            # Add to sets
            all_domains.update(domains)
            all_solutions.update([normalize_title(s) for s in solutions])
            if target_domain:
                all_target_domains.add(normalize_domain(target_domain))

            # Store for later processing
            per_paper_data[run_name].append({
                'paper_title': paper_title,
                'paper_index': paper_idx,
                'original_paper_id': original_paper_id,
                'domains': domains,
                'solutions': solutions,
                'metadata': metadata,
                'target_domain': normalize_domain(target_domain) if target_domain else None
            })

    print(f"  Total unique domains: {len(all_domains)}")
    print(f"  Total unique solutions: {len(all_solutions)}")
    print(f"  Total target domains: {len(all_target_domains)}")

    # Generate embeddings
    print("\nGenerating embeddings...")
    all_domains_list = sorted(list(all_domains))
    all_solutions_list = sorted(list(all_solutions))
    all_targets_list = sorted(list(all_target_domains))

    total_tokens = 0
    total_cost = 0.0

    print("  Embedding domains...")
    domain_embeddings_array, tokens, cost = cache.get_embeddings(all_domains_list, api_key)
    total_tokens += tokens
    total_cost += cost
    domain_embeddings = {d: emb for d, emb in zip(all_domains_list, domain_embeddings_array)}

    print("  Embedding solutions...")
    solution_embeddings_array, tokens, cost = cache.get_embeddings(all_solutions_list, api_key)
    total_tokens += tokens
    total_cost += cost
    solution_embeddings = {s: emb for s, emb in zip(all_solutions_list, solution_embeddings_array)}

    print("  Embedding target domains...")
    target_embeddings_array, tokens, cost = cache.get_embeddings(all_targets_list, api_key)
    total_tokens += tokens
    total_cost += cost
    target_embeddings = {t: emb for t, emb in zip(all_targets_list, target_embeddings_array)}

    # Save cache
    if args.cache_embeddings:
        cache.save()
        print(f"  Saved embeddings to cache: {args.embedding_cache_file}")

    print(f"  Total tokens: {total_tokens}")
    print(f"  Total cost: ${total_cost:.4f}")

    # Calculate per-paper metrics
    print("\nCalculating per-paper metrics...")
    all_per_paper_metrics = []

    for run_name, papers in per_paper_data.items():
        for paper_data in papers:
            # Get embeddings for this paper
            paper_domain_embeddings = np.array([
                domain_embeddings[d] for d in paper_data['domains'] if d in domain_embeddings
            ])
            paper_solution_embeddings = np.array([
                solution_embeddings[normalize_title(s)]
                for s in paper_data['solutions']
                if normalize_title(s) in solution_embeddings
            ])

            # Get target embedding
            target_embedding = None
            if paper_data['target_domain']:
                target_embedding = target_embeddings.get(paper_data['target_domain'])

            # Calculate metrics
            metrics = calculate_paper_metrics(
                paper_data['domains'],
                paper_data['solutions'],
                paper_domain_embeddings,
                paper_solution_embeddings,
                paper_data['target_domain'],
                target_embedding
            )

            # Add metadata and run info
            metrics['paper_title'] = paper_data['paper_title']
            metrics['paper_index'] = paper_data['paper_index']
            metrics['original_paper_id'] = paper_data['original_paper_id']
            metrics['run_name'] = run_name
            if paper_data['metadata']:
                metrics.update(paper_data['metadata'])

            all_per_paper_metrics.append(metrics)

    print(f"  Calculated metrics for {len(all_per_paper_metrics)} paper-run combinations")

    # Calculate per-run aggregates
    print("\nCalculating per-run aggregates...")
    per_run_aggregates = {}

    for run_name in runs.keys():
        run_papers = [p for p in all_per_paper_metrics if p['run_name'] == run_name]
        aggregates = calculate_run_aggregates(run_papers)

        # Calculate most frequent domains
        all_run_domains = []
        for paper_data in per_paper_data[run_name]:
            all_run_domains.extend(paper_data['domains'])

        domain_counts = Counter(all_run_domains)
        aggregates['most_frequent_domains'] = domain_counts.most_common(20)
        aggregates['total_unique_domains_discovered'] = len(set(all_run_domains))

        per_run_aggregates[run_name] = aggregates

    # Calculate metadata correlations
    print("\nCalculating metadata correlations...")
    metadata_correlations = {}

    for run_name in runs.keys():
        run_papers = [p for p in all_per_paper_metrics if p['run_name'] == run_name]
        correlations = calculate_metadata_correlations(run_papers)
        metadata_correlations[run_name] = correlations

    # Build final results
    output_json_path = output_dir / 'analysis.json'
    analysis_results = {
        'metadata': {
            'analysis_timestamp': datetime.now().isoformat(),
            'dataset_path': args.dataset_path,
            'runs_analyzed': list(runs.keys()),
            'num_papers_analyzed': len(paper_indices_to_analyze),
            'paper_indices': paper_indices_to_analyze,
            'embedding_model': 'text-embedding-3-small',
            'total_api_cost': total_cost,
            'output_dir': str(output_dir),
            'output_file': str(output_json_path)
        },
        'per_paper_metrics': all_per_paper_metrics,
        'per_run_aggregates': per_run_aggregates,
        'metadata_correlations': metadata_correlations
    }

    # Save to JSON
    print(f"\nSaving results to {output_json_path}...")
    with open(output_json_path, 'w') as f:
        json.dump(analysis_results, f, indent=2)

    # Print terminal summary
    print_terminal_summary(analysis_results)

    print(f"\nAnalysis complete!")
    print(f"Results saved to: {output_dir}")
    print(f"  - analysis.json: Full analysis results")


if __name__ == '__main__':
    main()
