"""
Main orchestration script for dataset creation.
"""

import os
import sys
import argparse
import uuid
import json
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables from .env file
env_path = Path(__file__).parent.parent / ".env"
load_dotenv(env_path)

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from utils import (
    load_config, _init_colors, TeeOutput,
    _get_pricing, save_state, load_state, get_config_value
)
from schema import AnalogicalReasoningExample, DifficultyLevel
import discovery
import verification
import extraction
import difficulty


def run_multi_template_discovery(config: dict, base_domains: list, target_domains: list,
                                generic_template: str, papers_per_template: int, c: dict) -> tuple[list, dict]:
    """Run discovery across multiple domain pairs with de-duplication.

    Args:
        config: Configuration dictionary
        base_domains: List of base domains
        target_domains: List of target domains
        generic_template: Path to generic template file
        papers_per_template: Number of papers to request per domain pair
        c: Color codes dictionary

    Returns:
        Tuple of (combined_papers_list, combined_metrics_dict)
    """
    all_papers = []
    seen_titles = set()  # De-duplication by title
    combined_metrics = {
        "by_domain_pair": {},
        "total_input": 0,
        "total_output": 0,
        "total_runtime": 0.0,
        "total_papers_discovered": 0,
        "papers_after_dedup": 0
    }

    # Generate all pairwise domain mappings
    domain_pairs = [(base, target) for base in base_domains for target in target_domains]

    for idx, (base_domain, target_domain) in enumerate(domain_pairs):
        pair_name = f"{base_domain}_to_{target_domain}"

        print(f"\n{c['B']}[Domain Pair {idx+1}/{len(domain_pairs)}] {pair_name}{c['R']}")
        print(f"  Base domain: {base_domain}")
        print(f"  Target domain: {target_domain}")

        # Run discovery for this domain pair
        try:
            max_retries = get_config_value(config, "apis.perplexity.max_retries", 3)
            papers, tokens = discovery.discover_papers(
                config,
                target_count=papers_per_template,
                template_path=generic_template,
                base_domain=base_domain,
                target_domain=target_domain,
                max_retries=max_retries
            )
        except RuntimeError as e:
            # Discovery failed after all retries - skip this domain pair
            print(f"  {c['Y']}⚠ SKIPPED: {str(e)}{c['R']}")
            combined_metrics['by_domain_pair'][pair_name] = {
                "base_domain": base_domain,
                "target_domain": target_domain,
                "papers_discovered": 0,
                "papers_unique": 0,
                "papers_duplicates": 0,
                "error": str(e),
                "tokens": {"input": 0, "output": 0, "runtime": 0.0}
            }
            continue

        # De-duplicate by title (case-insensitive, normalized)
        unique_papers = []
        for paper in papers:
            # Normalize: lowercase, strip whitespace, remove extra spaces
            title_normalized = ' '.join(paper['title'].lower().strip().split())

            if title_normalized not in seen_titles:
                seen_titles.add(title_normalized)
                unique_papers.append(paper)

        # Track metrics
        combined_metrics['by_domain_pair'][pair_name] = {
            "base_domain": base_domain,
            "target_domain": target_domain,
            "papers_discovered": len(papers),
            "papers_unique": len(unique_papers),
            "papers_duplicates": len(papers) - len(unique_papers),
            "tokens": tokens
        }
        combined_metrics['total_input'] += tokens['input']
        combined_metrics['total_output'] += tokens['output']
        combined_metrics['total_runtime'] += tokens['runtime']
        combined_metrics['total_papers_discovered'] += len(papers)

        all_papers.extend(unique_papers)

        print(f"  {c['G']}→ Discovered: {len(papers)} papers{c['R']}")
        print(f"  {c['G']}→ Unique: {len(unique_papers)} papers{c['R']}")
        if len(papers) - len(unique_papers) > 0:
            print(f"  {c['Y']}→ Duplicates: {len(papers) - len(unique_papers)} papers{c['R']}")
        print(f"  {c['G']}→ Tokens: {tokens['input']:,} input / {tokens['output']:,} output | Runtime: {tokens['runtime']:.2f}s{c['R']}")

    combined_metrics['papers_after_dedup'] = len(all_papers)

    return all_papers, combined_metrics


def main():
    """Main entry point for dataset creation."""
    parser = argparse.ArgumentParser(description="Create analogical reasoning paper dataset")
    parser.add_argument("--config", default="dataset_creation/config.yaml", help="Path to config file")
    parser.add_argument("--count", type=int, help="Override discovery count")
    parser.add_argument("--resume", help="Resume from previous run directory")
    parser.add_argument("--start-stage", type=int, choices=[1, 2, 3, 4], help="Start from specific stage")

    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config)

    # Override discovery count if specified
    if args.count:
        if 'dataset' not in config:
            config['dataset'] = {}
        config['dataset']['discovery_count'] = args.count

    # Initialize colors
    use_colors = get_config_value(config, "logging.use_colors", True)
    c = _init_colors(use_colors)

    # Determine if resuming or loading from previous run
    resuming = args.resume is not None
    load_from_run = get_config_value(config, "dataset.load_from_run", "")
    loading_from_config = load_from_run and not resuming

    # Determine start stage
    if args.start_stage:
        start_stage = args.start_stage
    elif loading_from_config:
        start_stage = get_config_value(config, "dataset.start_stage", 1)
    else:
        start_stage = 1

    # Load state if resuming
    state = {}
    source_dir = None

    if resuming:
        resume_dir = Path(args.resume)
        try:
            state = load_state(resume_dir)
            last_completed_stage = state.get('last_completed_stage', '')

            # Determine start stage based on last completed
            stage_map = {"discovery": 2, "verification": 3, "extraction": 4, "assessment": 5}
            start_stage = stage_map.get(last_completed_stage, 1)

            # Use existing output directory
            output_dir = resume_dir
            run_id = state.get('run_id')
            timestamp = state.get('timestamp')

            print(f"\n{c['C']}Resuming from previous run: {resume_dir.name}{c['R']}")

        except FileNotFoundError:
            print(f"Error: Could not find state file in {resume_dir}")
            return

    elif loading_from_config:
        # Load data from previous run but create new output directory
        output_base = get_config_value(config, "dataset.output_directory", "data/dataset_outputs")
        source_dir = Path(f"{output_base}/{load_from_run}")

        try:
            source_state = load_state(source_dir)

            # Create new run
            run_id = str(uuid.uuid4())
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = Path(f"{output_base}/{timestamp}_{run_id[:8]}")
            output_dir.mkdir(parents=True, exist_ok=True)

            # Initialize new state with data from source run
            state = {
                "run_id": run_id,
                "timestamp": timestamp,
                "started_at": datetime.now().isoformat(),
                "discovery_count": source_state.get('discovery_count', 10),
                "loaded_from_run": load_from_run,
            }

            # Load data from appropriate stages based on start_stage
            if start_stage >= 2 and 'discovery_data' in source_state:
                state['discovery_data'] = source_state['discovery_data']
                state['discovery_tokens'] = source_state.get('discovery_tokens', {})
                state['discovery_cost'] = source_state.get('discovery_cost', 0)
            if start_stage >= 3 and 'verification_data' in source_state:
                state['verification_data'] = source_state['verification_data']
                state['verification_runtime'] = source_state.get('verification_runtime', 0)
            if start_stage >= 4 and 'extraction_data' in source_state:
                state['extraction_data'] = source_state['extraction_data']
                state['extraction_tokens'] = source_state.get('extraction_tokens', {})
                state['extraction_cost'] = source_state.get('extraction_cost', 0)

            print(f"\n{c['C']}Loading data from previous run: {load_from_run}{c['R']}")
            print(f"{c['C']}Starting from stage {start_stage}{c['R']}")

        except FileNotFoundError:
            print(f"Error: Could not find state file in {source_dir}")
            return

    else:
        # Create new run
        run_id = str(uuid.uuid4())
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        output_base = get_config_value(config, "dataset.output_directory", "data/dataset_outputs")
        output_dir = Path(f"{output_base}/{timestamp}_{run_id[:8]}")
        output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize state
        state = {
            "run_id": run_id,
            "timestamp": timestamp,
            "started_at": datetime.now().isoformat(),
            "discovery_count": get_config_value(config, "dataset.discovery_count", 10),
        }

    # Set up terminal output logging
    save_terminal = get_config_value(config, "logging.save_terminal_output", True)
    original_stdout = sys.stdout
    if save_terminal:
        log_file = output_dir / "terminal_output.log"
        sys.stdout = TeeOutput(log_file, original_stdout)

    try:
        # Print header
        print(f"\n{c['C']}{'='*60}{c['R']}")
        print(f"{c['C']}DATASET CREATION: {run_id[:8]}{c['R']}")
        print(f"{c['C']}{'='*60}{c['R']}")
        print(f"Discovery: {state['discovery_count']} candidate papers")

        discovery_model = get_config_value(config, "apis.perplexity.model", "sonar-pro")
        extraction_model = get_config_value(config, "apis.anthropic.extraction_model", "claude-sonnet-4-5")
        assessment_model = get_config_value(config, "apis.anthropic.assessment_model", "claude-sonnet-4-5")
        print(f"Models: {discovery_model} (discovery), {extraction_model} (extraction), {assessment_model} (assessment)")

        # Print prompt configuration and verify files exist
        discovery_prompt = get_config_value(config, "dataset.discovery_prompt", "dataset_creation/prompts/discover_papers.txt")
        extraction_prompt = get_config_value(config, "apis.anthropic.extraction_prompt", "dataset_creation/prompts/extract_analogy.txt")
        difficulty_prompt = get_config_value(config, "apis.anthropic.difficulty_prompt", "dataset_creation/prompts/assess_difficulty.txt")
        print(f"Prompts:")
        print(f"  Discovery: {discovery_prompt}")
        print(f"  Extraction: {extraction_prompt}")
        print(f"  Difficulty: {difficulty_prompt}")

        # Verify prompt files exist
        for prompt_name, prompt_path in [("Discovery", discovery_prompt), ("Extraction", extraction_prompt), ("Difficulty", difficulty_prompt)]:
            if not Path(prompt_path).exists():
                print(f"{c['Y']}  ⚠ WARNING: {prompt_name} prompt file not found: {prompt_path}{c['R']}")

        # Print domain configuration if multi-template enabled
        enable_multi_template = get_config_value(config, "dataset.enable_multi_template_discovery", False)
        if enable_multi_template:
            base_domains = get_config_value(config, "dataset.base_domains", [])
            target_domains = get_config_value(config, "dataset.target_domains", [])
            print(f"Domains:")
            print(f"  Base domains ({len(base_domains)}): {', '.join(base_domains)}")
            print(f"  Target domains ({len(target_domains)}): {', '.join(target_domains)}")

        if resuming:
            print(f"Resuming from: {state.get('last_completed_stage', 'start')}")
            print(f"Starting at stage: {start_stage}")
        elif loading_from_config:
            print(f"Loading data from: {load_from_run}")
            print(f"Starting at stage: {start_stage}")

        print(f"{'='*60}\n")

        # Stage 1: Discovery
        if start_stage <= 1:
            # Check if multi-template mode enabled
            enable_multi_template = get_config_value(config, "dataset.enable_multi_template_discovery", False)

            if enable_multi_template:
                # Multi-template discovery with domain pairs
                base_domains = get_config_value(config, "dataset.base_domains", [])
                target_domains = get_config_value(config, "dataset.target_domains", [])
                discovery_prompt = get_config_value(config, "dataset.discovery_prompt", "dataset_creation/prompts/discover_papers_generic_domains.txt")
                papers_per_template = get_config_value(config, "dataset.papers_per_template", 10)

                print(f"{c['B']}[1/4] Multi-Domain Discovery...{c['R']}")
                print(f"  Model: {discovery_model}")
                print(f"  Prompt: {discovery_prompt}")
                print(f"  Base domains: {len(base_domains)}")
                print(f"  Target domains: {len(target_domains)}")
                print(f"  Domain pairs: {len(base_domains) * len(target_domains)}")
                print(f"  Papers per domain pair: {papers_per_template}")

                papers, discovery_metrics = run_multi_template_discovery(
                    config, base_domains, target_domains, discovery_prompt, papers_per_template, c
                )

                state['discovery_data'] = papers
                state['discovery_metrics'] = discovery_metrics
                state['discovery_tokens'] = {
                    "input": discovery_metrics['total_input'],
                    "output": discovery_metrics['total_output'],
                    "runtime": discovery_metrics['total_runtime']
                }
                state['discovery_cost'] = calculate_cost("sonar-pro", state['discovery_tokens'])

                # Print summary
                print(f"\n  {c['G']}DISCOVERY SUMMARY{c['R']}")
                print(f"  Total discovered: {discovery_metrics['total_papers_discovered']}")
                print(f"  After de-duplication: {discovery_metrics['papers_after_dedup']}")
                print(f"  Per-domain-pair breakdown:")
                for pair_name, metrics in discovery_metrics['by_domain_pair'].items():
                    if 'error' in metrics:
                        print(f"    [{pair_name}] {c['Y']}FAILED: {metrics.get('error', 'Unknown error')}{c['R']}")
                    else:
                        print(f"    [{pair_name}] {metrics['papers_unique']}/{metrics['papers_discovered']} unique")
                print(f"  {c['G']}→ Total tokens: {state['discovery_tokens']['input']:,} input / {state['discovery_tokens']['output']:,} output (perplexity){c['R']}")
                print(f"  {c['G']}→ Total runtime: {state['discovery_tokens']['runtime']:.2f}s{c['R']}")
                print(f"  {c['Y']}→ Cost: ${state['discovery_cost']:.4f} USD{c['R']}\n")

                # Check if no papers were found at all
                if len(papers) == 0:
                    print(f"\n{c['Y']}{'='*60}{c['R']}")
                    print(f"{c['Y']}⚠ WARNING: No papers discovered across any domain pairs!{c['R']}")
                    print(f"{c['Y']}{'='*60}{c['R']}")
                    print(f"\nPossible reasons:")
                    print(f"  - Domain combinations may be too specific or rare")
                    print(f"  - Perplexity API may be having issues")
                    print(f"  - Consider trying different domain combinations")
                    print(f"\nExiting early - no papers to process.\n")
                    return
            else:
                # Single-template discovery (backward compatible)
                print(f"{c['B']}[1/4] Discovering papers...{c['R']}")
                print(f"  Model: {discovery_model}")
                print(f"  Prompt: dataset_creation/prompts/discover_papers.txt")
                max_retries = get_config_value(config, "apis.perplexity.max_retries", 3)
                try:
                    papers, tokens = discovery.discover_papers(config, state['discovery_count'],
                                                              max_retries=max_retries)
                except RuntimeError as e:
                    print(f"\n{c['Y']}{'='*60}{c['R']}")
                    print(f"{c['Y']}⚠ ERROR: Discovery failed - {str(e)}{c['R']}")
                    print(f"{c['Y']}{'='*60}{c['R']}")
                    print(f"\nNo papers were discovered. Exiting early.\n")
                    return

                state['discovery_data'] = papers
                state['discovery_tokens'] = tokens
                state['discovery_cost'] = calculate_cost("sonar-pro", tokens)

                print(f"  {c['G']}→ Discovered: {len(papers)} papers{c['R']}")
                print(f"  {c['G']}→ Tokens: {tokens['input']:,} input / {tokens['output']:,} output (perplexity) | Runtime: {tokens['runtime']:.2f}s{c['R']}")
                print(f"  {c['Y']}→ Cost: ${state['discovery_cost']:.4f} USD{c['R']}\n")

                # Check if no papers were found
                if len(papers) == 0:
                    print(f"\n{c['Y']}{'='*60}{c['R']}")
                    print(f"{c['Y']}⚠ WARNING: No papers discovered!{c['R']}")
                    print(f"{c['Y']}{'='*60}{c['R']}")
                    print(f"\nExiting early - no papers to process.\n")
                    return

            save_state(output_dir, state, "discovery")
        else:
            print(f"[1/4] Discovery SKIPPED - loading from saved state\n")

        # Stage 2: Verification
        if start_stage <= 2:
            print(f"{c['B']}[2/4] Verifying papers...{c['R']}")
            print(f"  API: Semantic Scholar / arXiv")
            papers_to_verify = state.get('discovery_data', [])

            verified_papers, verification_runtime = verification.verify_papers(papers_to_verify, config)

            state['verification_data'] = verified_papers
            state['verification_runtime'] = verification_runtime

            print(f"  {c['G']}→ Verified: {len(verified_papers)}/{len(papers_to_verify)} papers{c['R']}\n")

            save_state(output_dir, state, "verification")
        else:
            print(f"[2/4] Verification SKIPPED - loading from saved state\n")

        # Stage 3: Extraction
        if start_stage <= 3:
            print(f"{c['B']}[3/4] Extracting analogy details...{c['R']}")
            papers_to_extract = state.get('verification_data', [])

            papers_with_analogies, tokens = extraction.extract_analogies(papers_to_extract, config)

            state['extraction_data'] = papers_with_analogies
            state['extraction_tokens'] = tokens
            state['extraction_cost'] = calculate_cost(extraction_model, tokens)

            print(f"  {c['G']}→ Extracted: {len(papers_with_analogies)}/{len(papers_to_extract)} papers{c['R']}")
            print(f"  {c['G']}→ Tokens: {tokens['input']:,} input / {tokens['output']:,} output (anthropic) | Runtime: {tokens['runtime']:.2f}s{c['R']}")
            print(f"  {c['Y']}→ Cost: ${state['extraction_cost']:.4f} USD{c['R']}\n")

            save_state(output_dir, state, "extraction")
        else:
            print(f"[3/4] Extraction SKIPPED - loading from saved state\n")

        # Stage 4: Assessment
        if start_stage <= 4:
            print(f"{c['B']}[4/4] Assessing difficulty...{c['R']}")
            papers_to_assess = state.get('extraction_data', [])

            papers_with_difficulty, tokens = difficulty.assess_difficulty(papers_to_assess, config)

            state['assessment_data'] = papers_with_difficulty
            state['assessment_tokens'] = tokens
            state['assessment_cost'] = calculate_cost(assessment_model, tokens)

            print(f"  {c['G']}→ Assessed: {len(papers_with_difficulty)}/{len(papers_to_assess)} papers{c['R']}")
            print(f"  {c['G']}→ Tokens: {tokens['input']:,} input / {tokens['output']:,} output (anthropic) | Runtime: {tokens['runtime']:.2f}s{c['R']}")
            print(f"  {c['Y']}→ Cost: ${state['assessment_cost']:.4f} USD{c['R']}\n")

            save_state(output_dir, state, "assessment")
        else:
            print(f"[4/4] Assessment SKIPPED - loading from saved state\n")

        # Finalize dataset
        print(f"{c['B']}Finalizing dataset...{c['R']}")
        finalize_dataset(output_dir, state, config)

        # Print summary
        print_summary(state, output_dir, c)

    finally:
        # Restore stdout
        if save_terminal and isinstance(sys.stdout, TeeOutput):
            sys.stdout.close()
            sys.stdout = original_stdout


def calculate_cost(model: str, tokens: dict) -> float:
    """Calculate cost for API call.

    Args:
        model: Model name
        tokens: Token usage dictionary

    Returns:
        Cost in USD
    """
    input_price, output_price = _get_pricing(model)
    cost = (tokens.get("input", 0) / 1_000_000) * input_price + \
           (tokens.get("output", 0) / 1_000_000) * output_price
    return cost


def finalize_dataset(output_dir: Path, state: dict, config: dict):
    """Generate final dataset and metadata files.

    Args:
        output_dir: Output directory
        state: Complete state dictionary
        config: Configuration dictionary
    """
    papers = state.get('assessment_data', [])

    # Separate papers with and without analogical reasoning, and add binary field
    papers_with_analogy = []
    papers_without_analogy = []
    difficulty_dist = {"easy": 0, "medium": 0, "hard": 0}
    annotated_papers = []

    for paper in papers:
        problem = paper.get('problem', '')
        rejected_at_extraction = paper.get('_rejected_at_extraction', False)

        # Determine if paper uses analogical reasoning
        uses_ar = not (
            rejected_at_extraction or
            problem.lower().startswith('this paper does not use analogical reasoning')
        )

        # Add binary field for analogical reasoning
        paper['uses_analogical_reasoning'] = uses_ar

        # Remove internal flags before saving
        paper.pop('_rejected_at_extraction', None)

        if uses_ar:
            papers_with_analogy.append(paper)
            diff = paper.get('difficulty', 'medium')
            if diff in difficulty_dist:
                difficulty_dist[diff] = difficulty_dist.get(diff, 0) + 1
        else:
            papers_without_analogy.append(paper)

        annotated_papers.append(paper)

    # Count non-original papers
    non_original_count = sum(1 for p in papers_with_analogy if not p.get('is_original_paper', True))

    # Create dataset
    dataset = {
        "metadata": {
            "run_id": state.get('run_id'),
            "timestamp": state.get('timestamp'),
            "created_at": state.get('started_at'),
            "total_papers": len(annotated_papers),
            "papers_with_analogical_reasoning": len(papers_with_analogy),
            "papers_without_analogical_reasoning": len(papers_without_analogy),
            "original_papers": len(papers_with_analogy) - non_original_count,
            "non_original_papers": non_original_count,
            "non_original_papers_note": "Non-original papers use analogical reasoning but are reviews/evaluations, not the first presentation of the method",
            "difficulty_distribution": difficulty_dist,
            "difficulty_distribution_note": "Difficulty counts only include papers with analogical reasoning",
            "version": "1.0"
        },
        "papers": annotated_papers
    }

    # Save dataset.json
    dataset_file = output_dir / "dataset.json"
    with open(dataset_file, 'w') as f:
        json.dump(dataset, f, indent=2)

    # Calculate totals
    total_input = sum([
        state.get('discovery_tokens', {}).get('input', 0),
        state.get('extraction_tokens', {}).get('input', 0),
        state.get('assessment_tokens', {}).get('input', 0)
    ])
    total_output = sum([
        state.get('discovery_tokens', {}).get('output', 0),
        state.get('extraction_tokens', {}).get('output', 0),
        state.get('assessment_tokens', {}).get('output', 0)
    ])
    total_cost = sum([
        state.get('discovery_cost', 0),
        state.get('extraction_cost', 0),
        state.get('assessment_cost', 0)
    ])

    # Calculate costs by provider
    perplexity_cost = state.get('discovery_cost', 0)
    claude_cost = state.get('extraction_cost', 0) + state.get('assessment_cost', 0)

    # Check if multi-template discovery was used
    discovery_metrics = state.get('discovery_metrics')
    multi_template_enabled = discovery_metrics is not None

    # Create metadata
    metadata = {
        "run_id": state.get('run_id'),
        "timestamp": state.get('timestamp'),
        "started_at": state.get('started_at'),
        "completed_at": datetime.now().isoformat(),
        "discovery_count": state.get('discovery_count', 10),
        "discovery_method": "multi_domain" if multi_template_enabled else "single_template",
        "domain_pairs_used": list(discovery_metrics['by_domain_pair'].keys()) if multi_template_enabled else None,
        "papers_discovered": len(state.get('discovery_data', [])),
        "papers_verified": len(state.get('verification_data', [])),
        "papers_extracted": len(state.get('extraction_data', [])),
        "papers_completed": len(papers),
        "papers_with_analogical_reasoning": len(papers_with_analogy),
        "papers_without_analogical_reasoning": len(papers_without_analogy),
        "difficulty_distribution": difficulty_dist,
        "tokens": {
            "by_stage": {
                "discovery": state.get('discovery_tokens', {}),
                "extraction": state.get('extraction_tokens', {}),
                "assessment": state.get('assessment_tokens', {})
            },
            "by_provider": {
                "perplexity": {
                    "input": state.get('discovery_tokens', {}).get('input', 0),
                    "output": state.get('discovery_tokens', {}).get('output', 0),
                    "total": state.get('discovery_tokens', {}).get('input', 0) + state.get('discovery_tokens', {}).get('output', 0)
                },
                "claude": {
                    "input": state.get('extraction_tokens', {}).get('input', 0) + state.get('assessment_tokens', {}).get('input', 0),
                    "output": state.get('extraction_tokens', {}).get('output', 0) + state.get('assessment_tokens', {}).get('output', 0),
                    "total": (state.get('extraction_tokens', {}).get('input', 0) + state.get('assessment_tokens', {}).get('input', 0)) + (state.get('extraction_tokens', {}).get('output', 0) + state.get('assessment_tokens', {}).get('output', 0))
                }
            },
            "total_input": total_input,
            "total_output": total_output,
            "total": total_input + total_output
        },
        "costs": {
            "by_stage": {
                "discovery": state.get('discovery_cost', 0),
                "extraction": state.get('extraction_cost', 0),
                "assessment": state.get('assessment_cost', 0)
            },
            "by_provider": {
                "perplexity": perplexity_cost,
                "claude": claude_cost
            },
            "total_cost": total_cost
        },
        "config": {
            "discovery_model": get_config_value(config, "apis.perplexity.model", "sonar-pro"),
            "extraction_model": get_config_value(config, "apis.anthropic.extraction_model", "claude-sonnet-4-5"),
            "assessment_model": get_config_value(config, "apis.anthropic.assessment_model", "claude-sonnet-4-5")
        }
    }

    # Add multi-template discovery details if available
    if multi_template_enabled:
        metadata["discovery_details"] = discovery_metrics

    # Save metadata.json
    metadata_file = output_dir / "metadata.json"
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)


def print_summary(state: dict, output_dir: Path, c: dict):
    """Print final summary.

    Args:
        state: Complete state dictionary
        output_dir: Output directory
        c: Color codes dictionary
    """
    total_input = sum([
        state.get('discovery_tokens', {}).get('input', 0),
        state.get('extraction_tokens', {}).get('input', 0),
        state.get('assessment_tokens', {}).get('input', 0)
    ])
    total_output = sum([
        state.get('discovery_tokens', {}).get('output', 0),
        state.get('extraction_tokens', {}).get('output', 0),
        state.get('assessment_tokens', {}).get('output', 0)
    ])
    total_cost = sum([
        state.get('discovery_cost', 0),
        state.get('extraction_cost', 0),
        state.get('assessment_cost', 0)
    ])

    # Calculate by provider
    perplexity_cost = state.get('discovery_cost', 0)
    claude_cost = state.get('extraction_cost', 0) + state.get('assessment_cost', 0)

    perplexity_tokens = state.get('discovery_tokens', {}).get('input', 0) + state.get('discovery_tokens', {}).get('output', 0)
    claude_tokens = (state.get('extraction_tokens', {}).get('input', 0) + state.get('assessment_tokens', {}).get('input', 0)) + \
                    (state.get('extraction_tokens', {}).get('output', 0) + state.get('assessment_tokens', {}).get('output', 0))

    # Get pipeline counts
    papers_discovered = len(state.get('discovery_data', []))
    papers_verified = len(state.get('verification_data', []))
    papers_extracted = len(state.get('extraction_data', []))
    papers_completed = len(state.get('assessment_data', []))

    # Analyze papers with actual analogical reasoning
    assessment_data = state.get('assessment_data', [])
    papers_with_analogy = []
    papers_without_analogy = []
    difficulty_dist = {"easy": 0, "medium": 0, "hard": 0}

    for paper in assessment_data:
        # Use the uses_analogical_reasoning field set during finalization
        uses_ar = paper.get('uses_analogical_reasoning', False)

        if uses_ar:
            papers_with_analogy.append(paper)
            diff = paper.get('difficulty', 'medium')
            if diff in difficulty_dist:
                difficulty_dist[diff] = difficulty_dist.get(diff, 0) + 1
        else:
            papers_without_analogy.append(paper)

    print(f"\n{c['G']}{'='*60}{c['R']}")
    print(f"{c['G']}✓ DATASET CREATION COMPLETE{c['R']}")
    print(f"{c['G']}{'='*60}{c['R']}")
    print(f"Papers processed: {papers_completed}")
    print(f"Pipeline: {papers_discovered} discovered → {papers_verified} verified → {papers_extracted} extracted → {papers_completed} assessed\n")

    # Dataset quality metrics
    print(f"{c['C']}DATASET QUALITY{c['R']}")
    print(f"{'-'*60}")
    print(f"Papers with analogical reasoning: {c['G']}{len(papers_with_analogy)}{c['R']}/{papers_verified}")
    print(f"Papers without analogical reasoning: {c['Y']}{papers_verified - len(papers_with_analogy)}{c['R']}/{papers_verified}")

    if papers_with_analogy:
        print(f"\n{c['G']}Papers with analogical reasoning:{c['R']}")
        for paper in papers_with_analogy:
            print(f"  - {paper.get('title', 'Unknown')}")

    if papers_without_analogy:
        print(f"\n{c['Y']}Papers without analogical reasoning:{c['R']}")
        for paper in papers_without_analogy:
            print(f"  - {paper.get('title', 'Unknown')}")

    # Show non-original papers that need original paper lookup
    non_original_papers = [p for p in papers_with_analogy if not p.get('is_original_paper', True)]
    if non_original_papers:
        print(f"\n{c['Y']}⚠ Papers that are NOT the original (need original paper lookup):{c['R']}")
        for paper in non_original_papers:
            print(f"  - {paper.get('title', 'Unknown')}")
            original_info = paper.get('original_paper_info', '')
            if original_info:
                print(f"    → {original_info}")

    print(f"\n{c['C']}Difficulty distribution (analogical reasoning papers only):{c['R']}")
    print(f"  Easy:   {difficulty_dist['easy']}")
    print(f"  Medium: {difficulty_dist['medium']}")
    print(f"  Hard:   {difficulty_dist['hard']}")

    # Show examples by difficulty
    if papers_with_analogy:
        print(f"\n{c['C']}Sample analogies by difficulty:{c['R']}")

        # Show one example from each difficulty level (if available)
        for diff_level in ['hard', 'medium', 'easy']:
            examples = [p for p in papers_with_analogy if p.get('difficulty') == diff_level]
            if examples:
                paper = examples[0]
                print(f"\n  [{diff_level.upper()}] {paper.get('title', 'Unknown')}")
                base = paper.get('base_domain', 'Unknown')
                target = paper.get('target_domain', 'Unknown')
                print(f"    {base} → {target}")

    print()

    # Print detailed metrics
    print(f"\n{c['C']}{'='*60}{c['R']}")
    print(f"{c['C']}METRICS{c['R']}")
    print(f"{c['C']}{'='*60}{c['R']}\n")

    # Discovery metrics
    discovery_tok = state.get('discovery_tokens', {})
    print(f"[DISCOVERY] (perplexity)")
    print(f"  Input tokens:  {discovery_tok.get('input', 0):,}")
    print(f"  Output tokens: {discovery_tok.get('output', 0):,}")
    print(f"  Total tokens:  {discovery_tok.get('input', 0) + discovery_tok.get('output', 0):,}")
    print(f"  Runtime:       {discovery_tok.get('runtime', 0):.2f}s\n")

    # Verification metrics (no tokens)
    verification_runtime = state.get('verification_runtime', 0)
    print(f"[VERIFICATION] (academic APIs)")
    print(f"  Runtime:       {verification_runtime:.2f}s\n")

    # Extraction metrics
    extraction_tok = state.get('extraction_tokens', {})
    print(f"[EXTRACTION] (anthropic)")
    print(f"  Input tokens:  {extraction_tok.get('input', 0):,}")
    print(f"  Output tokens: {extraction_tok.get('output', 0):,}")
    print(f"  Total tokens:  {extraction_tok.get('input', 0) + extraction_tok.get('output', 0):,}")
    print(f"  Runtime:       {extraction_tok.get('runtime', 0):.2f}s\n")

    # Assessment metrics
    assessment_tok = state.get('assessment_tokens', {})
    print(f"[ASSESSMENT] (anthropic)")
    print(f"  Input tokens:  {assessment_tok.get('input', 0):,}")
    print(f"  Output tokens: {assessment_tok.get('output', 0):,}")
    print(f"  Total tokens:  {assessment_tok.get('input', 0) + assessment_tok.get('output', 0):,}")
    print(f"  Runtime:       {assessment_tok.get('runtime', 0):.2f}s\n")

    # Total metrics
    total_runtime = sum([
        discovery_tok.get('runtime', 0),
        verification_runtime,
        extraction_tok.get('runtime', 0),
        assessment_tok.get('runtime', 0)
    ])

    print(f"{'-'*60}")
    print(f"TOTAL")
    print(f"{'-'*60}")
    print(f"  Input tokens:  {total_input:,}")
    print(f"  Output tokens: {total_output:,}")
    print(f"  Total tokens:  {total_input + total_output:,}")
    print(f"  Total runtime: {total_runtime:.2f}s\n")

    # Cost breakdown
    print(f"{'-'*60}")
    print(f"COST BREAKDOWN")
    print(f"{'-'*60}")
    print(f"  Discovery:   ${perplexity_cost:.4f} USD")
    print(f"  Extraction:  ${state.get('extraction_cost', 0):.4f} USD")
    print(f"  Assessment:  ${state.get('assessment_cost', 0):.4f} USD\n")
    print(f"  {c['Y']}TOTAL COST:  ${total_cost:.4f} USD{c['R']}")
    print(f"{'='*60}\n")

    print(f"\n{'='*60}")
    print(f"✓ WORKFLOW COMPLETE")
    print(f"{'='*60}\n")
    print(f"Outputs saved to: {output_dir}")
    print(f"  - terminal_output.log (complete terminal output)")
    print(f"  - dataset.json (final dataset)")
    print(f"  - metadata.json (run metadata)")
    print(f"  - state.json (resumption state)\n")


if __name__ == "__main__":
    main()
