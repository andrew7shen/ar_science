"""Workflow Orchestrator - chains all agents together."""

import uuid
import json
import sys
import time
from pathlib import Path
from datetime import datetime

# Add project root to path for eval module imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from agents.extraction import extract_problem, print_extraction
from agents.search import search_solutions, print_solutions
from agents.assessment import assess_solutions, print_assessment
from agents.baseline import generate_baseline_solutions
import db
from config import get_config
from eval.dataset_eval.domain_matching import filter_matched_domains

# ANSI color codes (toggled by config)
def _init_colors():
    if get_config("output.use_colors", True):
        return {
            'C': '\033[96m',  # Cyan
            'B': '\033[94m',  # Blue
            'G': '\033[92m',  # Green
            'Y': '\033[93m',  # Yellow
            'R': '\033[0m'    # Reset
        }
    return {'C': '', 'B': '', 'G': '', 'Y': '', 'R': ''}


def _get_pricing(model: str) -> tuple[float, float]:
    """Get pricing (input, output) per million tokens for a given model.

    Returns:
        Tuple of (input_price, output_price) per million tokens
    """
    model_lower = model.lower()

    # Claude models
    if "haiku" in model_lower:
        return (1.00, 5.00)
    if "sonnet" in model_lower:
        return (3.00, 15.00)
    if "opus" in model_lower:
        return (15.00, 75.00)

    # GPT-5 models (via OpenAI)
    if "gpt-5-nano" in model_lower:
        return (0.10, 0.40)  # Estimated for high-throughput model
    if "gpt-5-mini" in model_lower:
        return (0.20, 0.80)  # Cost-optimized
    if "gpt-5.2-pro" in model_lower:
        return (10.00, 30.00)  # Premium reasoning model
    if "gpt-5.2" in model_lower:  # Matches gpt-5.2, gpt-5.2-chat-latest, gpt-5.2-codex
        return (5.00, 15.00)  # Estimated flagship pricing

    # GPT-4 models (backward compatibility)
    if "gpt-4o-mini" in model_lower:
        return (0.15, 0.60)
    if "gpt-4o" in model_lower:
        return (2.50, 10.00)

    # Gemini 3 models (via OpenRouter)
    if "gemini-3-pro" in model_lower:
        return (2.00, 12.00)  # $2/$12 per MTok (from OpenRouter)
    if "gemini-3-flash" in model_lower:
        return (0.50, 3.00)  # Estimated for flash variant

    # Gemini 2 models (via OpenRouter)
    if "gemini-2.0-flash" in model_lower:
        return (0.00, 0.00)  # Free tier

    # DeepSeek R1
    if "deepseek-r1" in model_lower:
        return (0.14, 0.28)

    # Default to Sonnet pricing
    return (3.00, 15.00)


class TeeOutput:
    """Write to both terminal and file simultaneously."""
    def __init__(self, file_path, original_stdout):
        self.file = open(file_path, 'w')
        self.stdout = original_stdout

    def write(self, message):
        self.stdout.write(message)
        self.file.write(message)
        self.file.flush()  # Ensure immediate write

    def flush(self):
        self.stdout.flush()
        self.file.flush()

    def close(self):
        self.file.close()


def save_workflow_log(output_dir: Path, state: dict, stage: str):
    """Save comprehensive workflow log at each stage."""
    if output_dir is None:
        return

    log_file = output_dir / "workflow_log.json"

    # Load existing log if it exists
    if log_file.exists():
        with open(log_file, 'r') as f:
            log_data = json.load(f)
    else:
        # Initialize log with comprehensive config and problem
        log_data = {
            "workflow_id": state.get("workflow_id"),
            "timestamp": state.get("timestamp"),
            "problem": state.get("problem_text"),
            "config": {
                # Model configuration
                "model": {
                    "base_model": get_config("model.name"),
                    "extraction_model": get_config("model.extraction_model"),
                    "search_model": get_config("model.search_model"),
                    "assessment_model": get_config("model.assessment_model"),
                },
                # Search configuration
                "search": {
                    "provider": get_config("search.provider"),
                    "perplexity_model": get_config("search.perplexity_model"),
                    "num_domains_to_search": get_config("search.num_domains_to_search"),
                    "num_solutions_per_domain": get_config("search.num_solutions_per_domain"),
                    "solution_sources": get_config("search.solution_sources"),
                    "max_tokens": get_config("search.max_tokens"),
                    "find_github_repos": get_config("search.find_github_repos"),
                    "repos_per_solution_workflow_a": get_config("search.repos_per_solution_workflow_a"),
                    "max_web_searches": get_config("search.max_web_searches"),
                    "use_academic_apis": get_config("search.use_academic_apis"),
                    "academic_apis_enabled": get_config("search.academic_apis.enabled"),
                    "papers_per_solution": get_config("search.academic_apis.papers_per_solution"),
                    "repos_per_paper": get_config("search.academic_apis.repos_per_paper"),
                },
                # Extraction configuration
                "extraction": {
                    "abstraction_level": state.get("abstraction_level"),
                    "max_tokens": get_config("extraction.max_tokens"),
                    "num_abstraction_levels": get_config("extraction.num_abstraction_levels"),
                    "num_key_terms": get_config("extraction.num_key_terms"),
                },
                # Assessment configuration
                "assessment": {
                    "num_solutions_to_assess": get_config("assessment.num_solutions_to_assess"),
                    "num_top_solutions": get_config("assessment.num_top_solutions"),
                    "max_tokens": get_config("assessment.max_tokens"),
                    "github_use_web_search": get_config("assessment.github.use_web_search"),
                    "github_repos_per_solution": get_config("assessment.github.repos_per_solution"),
                    "github_fetch_readmes": get_config("assessment.github.fetch_readmes"),
                    "assessment_weights": get_config("assessment.weights"),
                },
                # Agent enabled/disabled status
                "agents_enabled": {
                    "extraction": get_config("agents.extraction.enabled"),
                    "search": get_config("agents.search.enabled"),
                    "assessment": get_config("agents.assessment.enabled"),
                },
                # Output configuration
                "output": {
                    "save_runs": get_config("output.save_runs"),
                    "directory": get_config("output.directory"),
                },
                # Baseline configuration (if enabled)
                "baseline": {
                    "enabled": get_config("baseline.enabled"),
                    "mode": get_config("baseline.mode"),
                    "num_domains_to_search": get_config("baseline.num_domains_to_search"),
                    "num_solutions_per_domain": get_config("baseline.num_solutions_per_domain"),
                    "baseline_model": get_config("baseline.baseline_model"),
                    "use_assessment": get_config("baseline.use_assessment"),
                    "prompt_template": get_config("baseline.prompt_template"),
                    "max_tokens": get_config("baseline.max_tokens"),
                } if get_config("baseline.enabled", False) else None,
            },
            "stages": {}
        }

    # Add stage data
    if stage == "extraction" and "extraction" in state:
        log_data["stages"]["extraction"] = state["extraction"]

    elif stage == "search" and "solutions" in state:
        log_data["stages"]["search"] = {
            "num_solutions_found": len(state["solutions"]),
            "solutions": state["solutions"]
        }

    elif stage == "assessment" and "assessed_solutions" in state:
        log_data["stages"]["assessment"] = {
            "num_assessed": len(state.get("all_assessed_solutions", state.get("assessed_solutions", []))),
            "all_assessed_solutions": state.get("all_assessed_solutions", state.get("assessed_solutions", [])),
            "top_solutions": state["assessed_solutions"]
        }

    elif stage == "selection" and "selected_solution" in state:
        log_data["stages"]["selection"] = {
            "selected_solution": state["selected_solution"]
        }

    elif stage == "complete":
        log_data["stages"]["complete"] = {}
        log_data["status"] = "completed"

    # Save log
    with open(log_file, 'w') as f:
        json.dump(log_data, indent=2, fp=f)


def _run_ar_combined_workflow(problem_text: str, verbose: bool = False) -> tuple[dict, list, dict]:
    """
    Run AR workflow with combined extraction + search in single LLM call.

    Args:
        problem_text: The biomedical problem description
        verbose: Whether to print progress messages

    Returns:
        Tuple of (extraction, solutions, tokens)
    """
    # Load configuration
    num_domains = get_config("search.num_domains_to_search", 1)
    num_solutions_per_domain = get_config("evaluation.num_solutions_per_domain", 1)
    model = get_config("model.name", "claude-sonnet-4-5-20250929")
    combined_prompt_path = get_config("agents.combined_prompt_path", "prompts/ar_combined_extraction_search_nondiverse.txt")

    if verbose:
        print(f"  Combined AR: Extraction + search in single call...")
        print(f"  Using combined prompt: {combined_prompt_path}")

    # Load prompt template
    template_path = Path(__file__).parent.parent / combined_prompt_path
    if not template_path.exists():
        raise FileNotFoundError(f"AR combined prompt template not found: {template_path}")

    with open(template_path, 'r') as f:
        prompt_template = f.read()

    # Format prompt
    prompt = prompt_template.format(
        problem_text=problem_text,
        num_domains=num_domains,
        num_solutions_per_domain=num_solutions_per_domain
    )

    # Initialize Anthropic client
    from anthropic import Anthropic
    import os
    client = Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

    # Single LLM call
    try:
        max_tokens = get_config("extraction.max_tokens", 10000)
        temperature = get_config("extraction.temperature", 1.0)

        response = client.messages.create(
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            messages=[{"role": "user", "content": prompt}]
        )

        # Extract response text
        response_text = response.content[0].text.strip()

        # Remove markdown code fences if present
        if response_text.startswith("```"):
            lines = response_text.split('\n')
            start_idx = 0
            end_idx = len(lines)

            for i, line in enumerate(lines):
                if line.startswith("```"):
                    if start_idx == 0:
                        start_idx = i + 1
                    else:
                        end_idx = i
                        break

            response_text = '\n'.join(lines[start_idx:end_idx]).strip()

        # Parse JSON response
        result = json.loads(response_text)

        # Split into extraction and solutions
        extraction = {
            "problem_summary": result.get("problem_summary"),
            "problem_objects": result.get("problem_objects"),
            "problem_relations": result.get("problem_relations"),
            "analogies": result.get("analogies"),
            "key_terms": result.get("key_terms"),
            "target_domains": result.get("target_domains")
        }
        solutions = result.get("solutions", [])

        if verbose:
            print(f"  Discovered {len(extraction.get('target_domains', []))} domains")
            print(f"  Found {len(solutions)} total solutions")

        # Token tracking
        tokens = {
            "input": response.usage.input_tokens,
            "output": response.usage.output_tokens,
            "provider": "anthropic",
            "num_llm_calls": 1  # Single combined call
        }

        return extraction, solutions, tokens

    except json.JSONDecodeError as e:
        print(f"  Error: Failed to parse JSON response: {e}")
        tokens = {
            "input": response.usage.input_tokens if 'response' in locals() else 0,
            "output": response.usage.output_tokens if 'response' in locals() else 0,
            "provider": "anthropic",
            "note": f"JSON parse error: {str(e)}",
            "num_llm_calls": 0
        }
        return {}, [], tokens
    except Exception as e:
        print(f"  Error in AR combined workflow: {e}")
        tokens = {
            "input": 0,
            "output": 0,
            "provider": "anthropic",
            "note": f"Error: {str(e)}",
            "num_llm_calls": 0
        }
        return {}, [], tokens


def run_workflow(problem_text: str, abstraction_level: str = "conceptual", verbose: bool = True, ground_truth_domain: str = None, domain_judge_client=None, domain_judge_model: str = None) -> dict:
    """
    Run complete analogous reasoning workflow.

    Args:
        problem_text: Biomedical problem description
        abstraction_level: Abstraction level for workflow
        verbose: Print verbose output
        ground_truth_domain: Ground truth domain for evaluation filtering (optional)
        domain_judge_client: Anthropic client for LLM domain judge (optional)
        domain_judge_model: Model name for domain judge (optional)
        abstraction_level: Which abstraction level to use (concrete/conceptual/mathematical)
        verbose: Whether to print progress messages (set False for parallel execution)

    Returns:
        Dict with workflow_id, state, and output_path
    """
    # Generate workflow ID
    workflow_id = str(uuid.uuid4())
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_runs = get_config("output.save_runs", True)

    # Set up output directory and logging only if saving is enabled
    if save_runs:
        # Check if baseline mode - save to separate directory
        baseline_enabled = get_config("baseline.enabled", False)
        if baseline_enabled:
            output_base = "data/baseline_outputs"
        else:
            output_base = get_config("output.directory", "data/outputs")

        output_dir = Path(f"{output_base}/{timestamp}_{workflow_id[:8]}")
        output_dir.mkdir(parents=True, exist_ok=True)

        # Set up terminal output logging
        terminal_log_path = output_dir / "terminal_output.log"
        original_stdout = sys.stdout
        tee_output = TeeOutput(terminal_log_path, original_stdout)
        sys.stdout = tee_output
    else:
        output_dir = None
        original_stdout = sys.stdout
        tee_output = None

    try:
        # Initialize colors
        c = _init_colors()

        # Print model configuration and problem
        base_model = get_config("model.name", "claude-sonnet-4-20250514")
        if verbose:
            print(f"\n{c['C']}{'='*60}{c['R']}")
            print(f"{c['C']}WORKFLOW: {workflow_id[:8]}{c['R']}")
            print(f"{c['C']}{'='*60}{c['R']}")
            print(f"Problem: {problem_text}")
            print(f"Model: {base_model}")
            print(f"Save runs: {save_runs}")

            # Print search configuration
            search_provider = get_config("search.provider", "perplexity")
            num_solutions = get_config("search.num_solutions_per_domain", 3)
            solution_sources = get_config("search.solution_sources", "all")
            use_academic_apis = get_config("search.use_academic_apis", False)
            print(f"Search: {search_provider}, {num_solutions} solutions/domain, sources={solution_sources}")
            if use_academic_apis:
                enabled_apis = get_config("search.academic_apis.enabled", ["semantic_scholar", "arxiv"])
                papers_per = get_config("search.academic_apis.papers_per_solution", 2)
                print(f"Academic APIs: {', '.join(enabled_apis)} ({papers_per} papers/solution)")

            print(f"{'='*60}\n")

        # Initialize state and metrics
        state = {
            "workflow_id": workflow_id,
            "problem_text": problem_text,
            "timestamp": timestamp,
            "abstraction_level": abstraction_level
        }
        metrics = {"stages": {}, "total_input": 0, "total_output": 0, "total_citation": 0, "total_reasoning": 0}
        workflow_start = time.time()

        # Check if AR combined mode is enabled (single call for extraction + search)
        use_combined = get_config("agents.use_combined_prompt", False)
        baseline_enabled = get_config("baseline.enabled", False)

        if use_combined and not baseline_enabled:
            # AR combined mode - single call for extraction + search
            if verbose:
                print(f"\n{c['Y']}AR COMBINED MODE{c['R']}")
                print(f"Model: {base_model}")
                print(f"Single LLM call for extraction + search\n")

            # [1/3] COMBINED EXTRACTION + SEARCH
            if verbose:
                print(f"{c['B']}[1/3] Running combined extraction + search...{c['R']} (using {base_model})")
            t0 = time.time()
            extraction, solutions, tokens = _run_ar_combined_workflow(problem_text, verbose=verbose)
            runtime = time.time() - t0

            # Track metrics for combined stage
            metrics["stages"]["combined"] = {
                "input": tokens["input"],
                "output": tokens["output"],
                "runtime": runtime,
                "provider": tokens.get("provider", "anthropic"),
                "note": tokens.get("note"),
                "num_llm_calls": tokens.get("num_llm_calls", 1)
            }
            metrics["total_input"] += tokens["input"]
            metrics["total_output"] += tokens["output"]

            if verbose:
                print(f"\n  {c['G']}→ Tokens: {tokens['input']:,} input / {tokens['output']:,} output ({tokens.get('provider', 'anthropic')}) | Runtime: {runtime:.2f}s{c['R']}")
                # Calculate and display cost
                input_price, output_price = _get_pricing(base_model)
                combined_cost = (tokens.get("input", 0) / 1_000_000) * input_price + (tokens.get("output", 0) / 1_000_000) * output_price
                print(f"  {c['Y']}→ Cost: ${combined_cost:.4f} USD{c['R']}\n")

            # Store extraction and solutions in state
            state["extraction"] = extraction
            state["solutions"] = solutions
            state["extraction_cost"] = 0.0  # Combined cost tracked separately
            state["search_cost"] = 0.0
            state["combined_cost"] = (tokens.get("input", 0) / 1_000_000) * input_price + (tokens.get("output", 0) / 1_000_000) * output_price

            if save_runs:
                save_workflow_log(output_dir, state, "search")

            # Skip to assessment (continue below after baseline section)

        # Check if baseline mode is enabled
        elif baseline_enabled:
            baseline_mode = get_config("baseline.mode", "simple_llm")
            baseline_model = get_config("baseline.baseline_model") or base_model

            if verbose:
                print(f"\n{c['Y']}BASELINE MODE: {baseline_mode}{c['R']}")
                print(f"Model: {baseline_model}")
                print(f"Baseline will skip extraction and search stages\n")

            # Check if loading from previous run
            load_from_run = get_config("baseline.load_from_run")
            if load_from_run:
                if verbose:
                    print(f"{c['B']}[1/3] Baseline DISABLED - loading from: {load_from_run}{c['R']}")
                with open(Path(load_from_run) / "workflow_log.json", 'r') as f:
                    run_data = json.load(f)

                # Check that the problem matches the saved run
                saved_problem = run_data.get("problem", "")
                if saved_problem != problem_text:
                    if verbose:
                        print(f"  {c['Y']}⚠ WARNING: Problem mismatch!{c['R']}")
                        print(f"    Current:  {problem_text[:80]}{'...' if len(problem_text) > 80 else ''}")
                        print(f"    Saved:    {saved_problem[:80]}{'...' if len(saved_problem) > 80 else ''}")
                    raise ValueError(f"Problem mismatch: current problem does not match the saved run at {load_from_run}")

                # Load solutions from either top level (workflow) or stages.search (baseline)
                solutions = run_data.get("solutions") or run_data.get("stages", {}).get("search", {}).get("solutions", [])
                if verbose:
                    print(f"  ✓ Loaded {len(solutions)} solutions")

                # Set up metrics with zero tokens
                tokens = {
                    "input": 0,
                    "output": 0,
                    "provider": "loaded",
                    "note": "Loaded from previous run"
                }
                runtime = 0.0
            else:
                # [1/3] BASELINE GENERATION (replaces extraction + search)
                if verbose:
                    print(f"{c['B']}[1/3] Generating baseline solutions...{c['R']} (using {baseline_model})")
                t0 = time.time()
                solutions, tokens, baseline_domains = generate_baseline_solutions(problem_text, mode=baseline_mode, ground_truth_domain=ground_truth_domain, domain_judge_client=domain_judge_client, domain_judge_model=domain_judge_model)
                runtime = time.time() - t0

                # Store baseline domains in state for evaluation tracking
                state['baseline_domains'] = baseline_domains

            metrics["stages"]["baseline"] = {
                "input": tokens["input"],
                "output": tokens["output"],
                "runtime": runtime,
                "provider": tokens.get("provider", "anthropic"),
                "note": tokens.get("note"),
                "github_validation": tokens.get("github_validation"),
                "citation_tokens": tokens.get("citation_tokens", 0),
                "reasoning_tokens": tokens.get("reasoning_tokens", 0),
                "search_queries": tokens.get("search_queries", 0),
                "domain_judge_input": tokens.get("domain_judge_input", 0),
                "domain_judge_output": tokens.get("domain_judge_output", 0)
            }
            metrics["total_input"] += tokens["input"]
            metrics["total_output"] += tokens["output"]
            metrics["total_citation"] = metrics.get("total_citation", 0) + tokens.get("citation_tokens", 0)
            metrics["total_reasoning"] = metrics.get("total_reasoning", 0) + tokens.get("reasoning_tokens", 0)
            metrics["domain_judge_input"] = metrics.get("domain_judge_input", 0) + tokens.get("domain_judge_input", 0)
            metrics["domain_judge_output"] = metrics.get("domain_judge_output", 0) + tokens.get("domain_judge_output", 0)

            if verbose:
                print_solutions(solutions)
                provider = tokens.get('provider', 'anthropic')
                if tokens['input'] == 0 and tokens['output'] == 0 and 'note' in tokens:
                    print(f"  {c['G']}→ Tokens: {tokens['note']} ({provider}) | Runtime: {runtime:.2f}s{c['R']}\n")

            if tokens['input'] == 0 and tokens['output'] == 0 and 'note' in tokens:
                state["baseline_cost"] = 0.0
            else:
                if verbose:
                    # Print token usage
                    token_str = f"{tokens['input']:,} input / {tokens['output']:,} output"
                    # Add Deep Research specific tokens if present
                    if tokens.get('citation_tokens', 0) > 0 or tokens.get('reasoning_tokens', 0) > 0:
                        token_str += f" / {tokens.get('citation_tokens', 0):,} citation / {tokens.get('reasoning_tokens', 0):,} reasoning"
                    print(f"  {c['G']}→ Tokens: {token_str} ({provider}) | Runtime: {runtime:.2f}s{c['R']}")

                # Calculate cost based on provider
                baseline_cost = 0.0
                if provider == "perplexity":
                    # Deep Research cost calculation (5 components)
                    baseline_cost = (
                        (tokens.get("input", 0) / 1_000_000) * 2.0 +
                        (tokens.get("output", 0) / 1_000_000) * 8.0 +
                        (tokens.get("citation_tokens", 0) / 1_000_000) * 2.0 +
                        (tokens.get("reasoning_tokens", 0) / 1_000_000) * 3.0 +
                        (tokens.get("search_queries", 0) / 1_000) * 5.0
                    )

                    if verbose:
                        # Detailed breakdown
                        print(f"  {c['Y']}→ Cost: ${baseline_cost:.4f} USD{c['R']}")
                        print(f"    Input: {tokens.get('input', 0):,} tokens (${(tokens.get('input', 0) / 1_000_000) * 2.0:.4f})")
                        print(f"    Output: {tokens.get('output', 0):,} tokens (${(tokens.get('output', 0) / 1_000_000) * 8.0:.4f})")
                        print(f"    Citations: {tokens.get('citation_tokens', 0):,} tokens (${(tokens.get('citation_tokens', 0) / 1_000_000) * 2.0:.4f})")
                        print(f"    Reasoning: {tokens.get('reasoning_tokens', 0):,} tokens (${(tokens.get('reasoning_tokens', 0) / 1_000_000) * 3.0:.4f})")
                        print(f"    Searches: {tokens.get('search_queries', 0)} queries (${(tokens.get('search_queries', 0) / 1_000) * 5.0:.4f})")
                else:
                    # Regular Claude LLM cost
                    input_price, output_price = _get_pricing(baseline_model)
                    baseline_cost = (tokens.get("input", 0) / 1_000_000) * input_price + (tokens.get("output", 0) / 1_000_000) * output_price
                    if verbose:
                        print(f"  {c['Y']}→ Cost: ${baseline_cost:.4f} USD{c['R']}")

                state["baseline_cost"] = baseline_cost

            # Print GitHub validation stats if available
            if verbose:
                if "github_validation" in tokens:
                    val_stats = tokens["github_validation"]
                    halluc_rate = val_stats.get("hallucination_rate", 0.0)
                    print(f"  {c['Y']}→ GitHub Validation: {val_stats['valid_repos']}/{val_stats['total_repos']} repos valid (hallucination rate: {halluc_rate:.1f}%){c['R']}\n")
                else:
                    print()  # Add blank line if no validation stats

            state["extraction_cost"] = 0.0
            state["search_cost"] = 0.0
            state["solutions"] = solutions

            # Create minimal extraction for assessment compatibility
            extraction = {
                "problem_summary": problem_text,
                "target_domains": [],
                "abstraction_levels": {}
            }
            state["extraction"] = extraction

            if save_runs:
                save_workflow_log(output_dir, state, "search")

            # Skip to assessment (continue below)

        # [1/4] EXTRACTION
        if not baseline_enabled and not use_combined and get_config("agents.extraction.enabled", True):
            model = get_config("model.extraction_model") or get_config("model.name", "claude-sonnet-4-20250514")
            if verbose:
                print(f"{c['B']}[1/4] Extracting problem aspects...{c['R']} (using {model})")
            t0 = time.time()
            extraction, tokens = extract_problem(problem_text, verbose=verbose)
            runtime = time.time() - t0
            metrics["stages"]["extraction"] = {
                "input": tokens["input"],
                "output": tokens["output"],
                "runtime": runtime,
                "provider": tokens.get("provider", "anthropic"),
                "note": tokens.get("note")
            }
            metrics["total_input"] += tokens["input"]
            metrics["total_output"] += tokens["output"]
            if verbose:
                print_extraction(extraction, selected_abstraction=abstraction_level)
                provider = tokens.get('provider', 'anthropic')
                if tokens['input'] == 0 and tokens['output'] == 0 and 'note' in tokens:
                    print(f"  {c['G']}→ Tokens: {tokens['note']} ({provider}) | Runtime: {runtime:.2f}s{c['R']}\n")
                else:
                    print(f"  {c['G']}→ Tokens: {tokens['input']:,} input / {tokens['output']:,} output ({provider}) | Runtime: {runtime:.2f}s{c['R']}")
                    # Calculate and display cost for Claude API (extraction uses Anthropic API)
                    input_price, output_price = _get_pricing(base_model)
                    extraction_cost = (tokens.get("input", 0) / 1_000_000) * input_price + (tokens.get("output", 0) / 1_000_000) * output_price
                    print(f"  {c['Y']}→ Cost: ${extraction_cost:.4f} USD{c['R']}\n")

            if tokens['input'] == 0 and tokens['output'] == 0 and 'note' in tokens:
                state["extraction_cost"] = 0.0
            else:
                input_price, output_price = _get_pricing(base_model)
                extraction_cost = (tokens.get("input", 0) / 1_000_000) * input_price + (tokens.get("output", 0) / 1_000_000) * output_price
                state["extraction_cost"] = extraction_cost
            state["extraction"] = extraction
            if save_runs:
                save_workflow_log(output_dir, state, "extraction")

            # Interactive domain selection (if enabled)
            if get_config("interactive.domain_selection.enabled", False):
                from interactive import prompt_domain_selection
                num_domains = get_config("search.num_domains_to_search", 3)
                modified_domains = prompt_domain_selection(
                    extraction['target_domains'],
                    num_domains
                )
                extraction['target_domains'] = modified_domains
                state["extraction"] = extraction
                if verbose:
                    print(f"  Final domains: {', '.join(modified_domains)}\n")
        elif not use_combined:
            # Only provide minimal extraction if not in combined mode
            # (Combined mode already set extraction earlier)
            if verbose:
                print("[1/4] Extraction agent DISABLED - skipping...")
            # Provide minimal extraction data for downstream agents
            extraction = {
                "problem_summary": problem_text,
                "target_domains": [],
                "abstraction_levels": {}
            }
            state["extraction"] = extraction
            state["extraction_cost"] = 0.0

        # Conditional domain filtering (only search matched domains if enabled)
        domain_judge_tokens_ar = {'input': 0, 'output': 0}
        if get_config("agents.search.only_search_matched_domains", False):
            discovered_domains = state.get('extraction', {}).get('target_domains', [])

            if ground_truth_domain and discovered_domains:
                # Use LLM judge if client and model are provided, otherwise fall back to dictionary
                if domain_judge_client is not None and domain_judge_model is not None:
                    if verbose:
                        print(f"{c['Y']}[Domain Filtering with LLM Judge] Evaluating {len(discovered_domains)} domains...{c['R']}")

                    # Import helper function from baseline
                    from agents.baseline import _filter_domains_with_llm_judge
                    # Check if we should stop after first match (optimization)
                    stop_on_first = get_config("agents.domain_judge_stop_on_first_match", True)
                    matched_domains, domain_judge_tokens_ar = _filter_domains_with_llm_judge(
                        discovered_domains, ground_truth_domain, domain_judge_client, domain_judge_model, stop_on_first_match=stop_on_first
                    )
                    if verbose:
                        print(f"  Filtered to {len(matched_domains)} matched domains (using {domain_judge_model})")
                else:
                    # Fall back to dictionary-based matching
                    matched_domains = filter_matched_domains(discovered_domains, ground_truth_domain)
                    if verbose:
                        print(f"{c['Y']}[Domain Filtering] Filtered {len(discovered_domains)} discovered domains to {len(matched_domains)} matched domains (using CLOSE_MATCHES dictionary){c['R']}")

                # Update state to only search matched domains
                state['extraction']['target_domains'] = matched_domains

                if verbose:
                    print(f"  Ground truth: {ground_truth_domain}")
                    print(f"  Discovered: {', '.join(discovered_domains)}")
                    print(f"  Matched: {', '.join(matched_domains) if matched_domains else 'None'}\n")

        # Track AR domain judge tokens in metrics
        if domain_judge_tokens_ar['input'] > 0 or domain_judge_tokens_ar['output'] > 0:
            metrics["domain_judge_input"] = metrics.get("domain_judge_input", 0) + domain_judge_tokens_ar['input']
            metrics["domain_judge_output"] = metrics.get("domain_judge_output", 0) + domain_judge_tokens_ar['output']

        # Check if extraction testing mode is enabled (skip search and assessment)
        extraction_testing = get_config("agents.extraction.testing", False)
        if extraction_testing:
            if verbose:
                reasoning_type = get_config("extraction.reasoning_type", "hierarchical")
                print(f"\n{c['Y']}⚠ Extraction testing mode (reasoning_type: {reasoning_type}) - skipping search and assessment{c['R']}\n")
            state["solutions"] = []
            state["assessed_solutions"] = []
            state["search_cost"] = 0.0
            state["assessment_cost"] = 0.0
            metrics["total_runtime"] = time.time() - workflow_start
            state["metrics"] = metrics
            if save_runs:
                save_workflow_log(output_dir, state, "complete")

            if verbose:
                # Print metrics for extraction stage
                print(f"{c['C']}{'='*60}")
                print("METRICS")
                print(f"{'='*60}{c['R']}")
                if "extraction" in metrics["stages"]:
                    data = metrics["stages"]["extraction"]
                    print(f"\n[EXTRACTION] ({data.get('provider', 'anthropic')})")
                    print(f"  Input tokens:  {data['input']:,}")
                    print(f"  Output tokens: {data['output']:,}")
                    print(f"  Total tokens:  {data['input'] + data['output']:,}")
                    print(f"  Runtime:       {data['runtime']:.2f}s")
                print(f"\n{'-'*60}")
                print(f"TOTAL: {metrics['total_input'] + metrics['total_output']:,} tokens | {metrics['total_runtime']:.2f}s | ${state.get('extraction_cost', 0):.4f} USD")
                print("="*60)

                print(f"\n{'='*60}")
                print(f"✓ EXTRACTION TEST COMPLETE")
                print(f"{'='*60}")
                if save_runs:
                    print(f"\nOutputs saved to: {output_dir}")
                print()
            return state

        # [2/4] SEARCH
        if not baseline_enabled and not use_combined and get_config("agents.search.enabled", True):
            use_llm_fallback = get_config("search.use_llm_fallback", False)

            if use_llm_fallback:
                # HYBRID MODE: Extraction + LLM generation (without web search)
                from agents.baseline import generate_solutions_from_extraction

                # Check if there are any target domains to search
                target_domains = extraction.get('target_domains', [])
                if not target_domains:
                    if verbose:
                        print(f"{c['B']}[2/4] Generating solutions from extraction (LLM fallback){c['R']}")
                        print(f"  {c['Y']}⚠ No matched domains found - skipping solution generation{c['R']}\n")
                    solutions = []
                    tokens = {"input": 0, "output": 0}
                    runtime = 0.0
                else:
                    model = get_config("model.search_model") or get_config("model.name", "claude-sonnet-4-20250514")
                    if verbose:
                        print(f"{c['B']}[2/4] Generating solutions from extraction (LLM fallback){c['R']} (using {model}, {abstraction_level} abstraction)...")
                    t0 = time.time()
                    solutions, tokens = generate_solutions_from_extraction(extraction, abstraction_level)
                    runtime = time.time() - t0

                # Track metrics
                metrics["stages"]["search"] = {
                    "input": tokens["input"],
                    "output": tokens["output"],
                    "runtime": runtime,
                    "provider": tokens.get("provider", "anthropic"),
                    "note": "LLM fallback mode (extraction + generation)",
                    "github_validation": tokens.get("github_validation", {})
                }
                metrics["total_input"] += tokens["input"]
                metrics["total_output"] += tokens["output"]

                if verbose:
                    # Print solutions
                    print_solutions(solutions)

                    # Print token stats
                    provider = tokens.get('provider', 'anthropic')
                    print(f"  {c['G']}→ Tokens: {tokens['input']:,} input / {tokens['output']:,} output ({provider}) | Runtime: {runtime:.2f}s{c['R']}")

                # Calculate cost (Anthropic pricing)
                input_price, output_price = _get_pricing(model)
                search_cost = (tokens["input"] / 1_000_000) * input_price + (tokens["output"] / 1_000_000) * output_price

                if verbose:
                    print(f"  {c['Y']}→ Cost: ${search_cost:.4f} USD{c['R']}")

                    # Print GitHub validation stats
                    if "github_validation" in tokens:
                        valid = tokens["github_validation"]["valid_repos"]
                        invalid = tokens["github_validation"]["invalid_repos"]
                        total = tokens["github_validation"]["total_repos"]
                        hallucination_rate = tokens["github_validation"]["hallucination_rate"]
                        print(f"  {c['Y']}→ GitHub repos: {valid} valid / {total} total (hallucination rate: {hallucination_rate:.1f}%){c['R']}")

                    print()
                state["search_cost"] = search_cost
                state["solutions"] = solutions
                if save_runs:
                    save_workflow_log(output_dir, state, "search")

            else:
                # NORMAL WEB SEARCH MODE
                # Check if there are any target domains to search
                target_domains = extraction.get('target_domains', [])
                if not target_domains:
                    if verbose:
                        print(f"{c['B']}[2/4] Searching for analogous solutions{c['R']}")
                        print(f"  {c['Y']}⚠ No matched domains found - skipping solution search{c['R']}\n")
                    solutions = []
                    tokens = {"input": 0, "output": 0}
                    runtime = 0.0
                else:
                    model = get_config("model.search_model") or get_config("model.name", "claude-sonnet-4-20250514")
                    if verbose:
                        print(f"{c['B']}[2/4] Searching for analogous solutions{c['R']} (using {model}, {abstraction_level} abstraction)...")
                    t0 = time.time()
                    solutions, tokens = search_solutions(extraction, abstraction_level=abstraction_level, output_dir=output_dir)
                    runtime = time.time() - t0
                metrics["stages"]["search"] = {
                    "input": tokens["input"],
                    "output": tokens["output"],
                    "runtime": runtime,
                    "provider": tokens.get("provider", "anthropic"),
                    "note": tokens.get("note"),
                    "academic_api_calls": tokens.get("academic_api_calls", {}),
                    "academic_api_runtime": tokens.get("academic_api_runtime", 0.0),
                    "citation_tokens": tokens.get("citation_tokens", 0),
                    "reasoning_tokens": tokens.get("reasoning_tokens", 0),
                    "search_queries": tokens.get("search_queries", 0)
                }
                metrics["total_input"] += tokens["input"]
                metrics["total_output"] += tokens["output"]
                metrics["total_citation"] += tokens.get("citation_tokens", 0)
                metrics["total_reasoning"] += tokens.get("reasoning_tokens", 0)

                if verbose:
                    print_solutions(solutions)
                    provider = tokens.get('provider', 'anthropic')
                    if tokens['input'] == 0 and tokens['output'] == 0 and 'note' in tokens:
                        print(f"  {c['G']}→ Tokens: {tokens['note']} ({provider}) | Runtime: {runtime:.2f}s{c['R']}")
                    else:
                        token_str = f"{tokens['input']:,} input / {tokens['output']:,} output"
                        # Add Deep Research specific tokens if present
                        if tokens.get('citation_tokens', 0) > 0 or tokens.get('reasoning_tokens', 0) > 0:
                            token_str += f" / {tokens.get('citation_tokens', 0):,} citation / {tokens.get('reasoning_tokens', 0):,} reasoning"
                        print(f"  {c['G']}→ Tokens: {token_str} ({provider}) | Runtime: {runtime:.2f}s{c['R']}")

                # Calculate and display cost
                search_cost = 0.0
                if get_config("search.provider") == "perplexity":
                    perplexity_model = get_config("search.perplexity_model")

                    if perplexity_model == "sonar-deep-research":
                        # Calculate Deep Research cost (5 components)
                        search_cost = (
                            (tokens.get("input", 0) / 1_000_000) * 2.0 +
                            (tokens.get("output", 0) / 1_000_000) * 8.0 +
                            (tokens.get("citation_tokens", 0) / 1_000_000) * 2.0 +
                            (tokens.get("reasoning_tokens", 0) / 1_000_000) * 3.0 +
                            (tokens.get("search_queries", 0) / 1_000) * 5.0
                        )

                        if verbose:
                            # Detailed breakdown
                            print(f"  {c['Y']}→ Cost: ${search_cost:.4f} USD{c['R']}")
                            print(f"    Input: {tokens.get('input', 0):,} tokens (${(tokens.get('input', 0) / 1_000_000) * 2.0:.4f})")
                            print(f"    Output: {tokens.get('output', 0):,} tokens (${(tokens.get('output', 0) / 1_000_000) * 8.0:.4f})")
                            print(f"    Citations: {tokens.get('citation_tokens', 0):,} tokens (${(tokens.get('citation_tokens', 0) / 1_000_000) * 2.0:.4f})")
                            print(f"    Reasoning: {tokens.get('reasoning_tokens', 0):,} tokens (${(tokens.get('reasoning_tokens', 0) / 1_000_000) * 3.0:.4f})")
                            print(f"    Searches: {tokens.get('search_queries', 0)} queries (${(tokens.get('search_queries', 0) / 1_000) * 5.0:.4f})")
                    elif perplexity_model == "sonar-pro":
                        # Calculate Sonar Pro cost (Sonar Pro uses Claude Sonnet pricing)
                        search_cost = (tokens.get("input", 0) / 1_000_000) * 3.0 + (tokens.get("output", 0) / 1_000_000) * 15.0
                        if verbose:
                            print(f"  {c['Y']}→ Cost: ${search_cost:.4f} USD{c['R']}")
                else:
                    # Claude web search
                    input_price, output_price = _get_pricing(base_model)
                    search_cost = (tokens.get("input", 0) / 1_000_000) * input_price + (tokens.get("output", 0) / 1_000_000) * output_price
                    if verbose:
                        print(f"  {c['Y']}→ Cost: ${search_cost:.4f} USD{c['R']}")

                state["search_cost"] = search_cost

                if verbose:
                    # Print academic API usage if applicable
                    academic_api_calls = tokens.get('academic_api_calls', {})
                    if academic_api_calls and any(academic_api_calls.values()):
                        api_summary = ', '.join([f"{api}: {count}" for api, count in academic_api_calls.items() if count > 0])
                        api_runtime = tokens.get('academic_api_runtime', 0.0)
                        print(f"  {c['Y']}→ Academic API calls: {api_summary} | API runtime: {api_runtime:.2f}s{c['R']}")
                    print()
                state["solutions"] = solutions
                if save_runs:
                    save_workflow_log(output_dir, state, "search")

            if not solutions:
                if verbose:
                    print("No solutions found. Workflow terminated.")
                # Set final metrics before returning
                metrics["total_runtime"] = time.time() - workflow_start
                state["metrics"] = metrics
                return state
        elif not baseline_enabled and not use_combined:
            # Check if we should skip loading solutions (for domain-only evaluation)
            skip_loading = get_config("agents.search.skip_loading_solutions", False)
            if skip_loading:
                if verbose:
                    print("[2/4] Search agent DISABLED - skipping solution loading (domain-only evaluation)")
                solutions = []
            else:
                load_from_run = get_config("agents.search.load_from_run")
                if load_from_run:
                    if verbose:
                        print(f"[2/4] Search agent DISABLED - loading from: {load_from_run}")
                    with open(Path(load_from_run) / "workflow_log.json", 'r') as f:
                        run_data = json.load(f)

                    # Check that the problem matches the saved run
                    saved_problem = run_data.get("problem", "")
                    if saved_problem != problem_text:
                        if verbose:
                            print(f"  {c['Y']}⚠ WARNING: Problem mismatch!{c['R']}")
                            print(f"    Current:  {problem_text[:80]}{'...' if len(problem_text) > 80 else ''}")
                            print(f"    Saved:    {saved_problem[:80]}{'...' if len(saved_problem) > 80 else ''}")
                        raise ValueError(f"Problem mismatch: current problem does not match the saved run at {load_from_run}")

                    solutions = run_data.get("stages", {}).get("search", {}).get("solutions", [])
                    extraction = run_data.get("stages", {}).get("extraction", {})
                    if extraction:
                        state["extraction"] = extraction
                    if verbose:
                        print(f"  ✓ Loaded {len(solutions)} solutions")
                        print(f"  {c['Y']}ℹ Note: Printed search config above may not match this loaded run{c['R']}")
                else:
                    if verbose:
                        print("[2/4] Search agent DISABLED - using toy solutions")
                    toy_path = Path("data/toy_solutions.json")
                    solutions = json.load(open(toy_path)) if toy_path.exists() else []
                    if verbose:
                        print(f"  ✓ Loaded {len(solutions)} toy solutions")
            if verbose:
                print_solutions(solutions)
            state["solutions"] = solutions
            state["search_cost"] = 0.0

        # Filter solutions without GitHub repos if configured
        if get_config("assessment.require_github_repos", False) and solutions:
            before_count = len(solutions)
            solutions = [s for s in solutions if s.get("github_repos")]
            state["solutions"] = solutions
            if before_count != len(solutions) and verbose:
                print(f"  Filtered solutions: {before_count} → {len(solutions)} (dropped {before_count - len(solutions)} without repos)\n")

        # [3/4] ASSESSMENT
        # Skip assessment if baseline use_assessment is false
        skip_assessment = baseline_enabled and not get_config("baseline.use_assessment", True)
        if get_config("agents.assessment.enabled", True) and solutions and not skip_assessment:
            model = get_config("model.assessment_model") or get_config("model.name", "claude-sonnet-4-20250514")
            if verbose:
                print(f"{c['B']}[3/4] Assessing and ranking solutions...{c['R']} (using {model})")
            t0 = time.time()
            assessed, tokens = assess_solutions(extraction['problem_summary'], solutions)
            runtime = time.time() - t0
            metrics["stages"]["assessment"] = {
                "input": tokens["input"],
                "output": tokens["output"],
                "runtime": runtime,
                "provider": tokens.get("provider", "anthropic"),
                "note": tokens.get("note")
            }
            metrics["total_input"] += tokens["input"]
            metrics["total_output"] += tokens["output"]
            # Sort by corrected score (highest first) - score has been fixed to match weighted sum
            # Note: assess_solutions() already sorts by the appropriate score based on mode
            # Filter to top N for display and selection (but log all assessed)
            num_top = get_config("assessment.num_top_solutions", 3)
            top_assessed = assessed[:num_top]

            if verbose:
                print_assessment(top_assessed, all_assessed=assessed)
                provider = tokens.get('provider', 'anthropic')
                if tokens['input'] == 0 and tokens['output'] == 0 and 'note' in tokens:
                    print(f"  {c['G']}→ Tokens: {tokens['note']} ({provider}) | Runtime: {runtime:.2f}s{c['R']}\n")
                else:
                    print(f"  {c['G']}→ Tokens: {tokens['input']:,} input / {tokens['output']:,} output ({provider}) | Runtime: {runtime:.2f}s{c['R']}")
                    # Calculate and display cost for Claude API (assessment uses Anthropic API)
                    input_price, output_price = _get_pricing(base_model)
                    assessment_cost = (tokens.get("input", 0) / 1_000_000) * input_price + (tokens.get("output", 0) / 1_000_000) * output_price
                    print(f"  {c['Y']}→ Cost: ${assessment_cost:.4f} USD{c['R']}\n")

            if tokens['input'] == 0 and tokens['output'] == 0 and 'note' in tokens:
                state["assessment_cost"] = 0.0
            else:
                input_price, output_price = _get_pricing(base_model)
                assessment_cost = (tokens.get("input", 0) / 1_000_000) * input_price + (tokens.get("output", 0) / 1_000_000) * output_price
                state["assessment_cost"] = assessment_cost
            # Store all assessed solutions for logging, but also keep top for selection
            state["all_assessed_solutions"] = assessed
            state["assessed_solutions"] = top_assessed
            state["solutions"] = assessed  # Update solutions with assessed versions (includes novelty data)
            if save_runs:
                save_workflow_log(output_dir, state, "assessment")
        else:
            if verbose:
                if skip_assessment:
                    print("[3/4] Assessment SKIPPED - baseline.use_assessment = false")
                elif not get_config("agents.assessment.enabled", True):
                    print("[3/4] Assessment agent DISABLED - skipping...")
            assessed = solutions  # Use unassessed solutions
            state["assessed_solutions"] = assessed
            state["assessment_cost"] = 0.0

        if not assessed:
            if verbose:
                print("No solutions to implement. Workflow terminated.")

            # Calculate total workflow runtime and add metrics to state
            metrics["total_runtime"] = time.time() - workflow_start
            state["metrics"] = metrics

            # Print metrics summary even when no solutions
            if verbose:
                print(f"\n{c['C']}{'='*60}")
                print("METRICS")
                print(f"{'='*60}{c['R']}")
                for stage, data in metrics["stages"].items():
                    total = data["input"] + data["output"]
                    provider = data.get("provider", "anthropic")
                    print(f"\n[{stage.upper()}] ({provider})")
                    if data["input"] == 0 and data["output"] == 0 and data.get("note"):
                        print(f"  Tokens:        {data['note']}")
                    else:
                        print(f"  Input tokens:  {data['input']:,}")
                        print(f"  Output tokens: {data['output']:,}")
                        print(f"  Total tokens:  {total:,}")
                    print(f"  Runtime:       {data['runtime']:.2f}s")

                print(f"\n{'-'*60}")
                print("TOTAL")
                print("-"*60)
                print(f"  Input tokens:  {metrics['total_input']:,}")
                print(f"  Output tokens: {metrics['total_output']:,}")
                total_all = metrics['total_input'] + metrics['total_output']
                print(f"  Total tokens:  {total_all:,}")
                print(f"  Total runtime: {metrics['total_runtime']:.2f}s")

            # Calculate and display total cost
            total_cost = 0.0
            total_cost += state.get("extraction_cost", 0.0)
            total_cost += state.get("search_cost", 0.0)
            total_cost += state.get("baseline_cost", 0.0)
            total_cost += state.get("assessment_cost", 0.0)

            if verbose and total_cost > 0.0:
                print(f"\n{'-'*60}")
                print("COST BREAKDOWN")
                print("-"*60)
                if "baseline_cost" in state:
                    print(f"  Baseline:    ${state['baseline_cost']:.4f} USD")
                if "extraction_cost" in state:
                    print(f"  Extraction:  ${state['extraction_cost']:.4f} USD")
                if "search_cost" in state:
                    print(f"  Search:      ${state['search_cost']:.4f} USD")
                if "assessment_cost" in state:
                    print(f"  Assessment:  ${state['assessment_cost']:.4f} USD")
                print(f"\n  {c['Y']}TOTAL COST:  ${total_cost:.4f} USD{c['R']}")

                print("="*60 + "\n")

            return state

        # [4/4] AUTOMATIC SELECTION (highest-scoring solution)
        if verbose:
            print(f"{c['B']}[4/4] Automatic solution selection...{c['R']}")
        selected_idx = 0  # Select highest-scoring solution
        selected_solution = assessed[selected_idx]
        state["selected_solution"] = selected_solution

        if verbose:
            # Display selection message based on scoring mode
            scoring_mode = get_config("assessment.scoring_mode", "unified")
            title = selected_solution.get('title', 'Unknown')
            if scoring_mode == "split":
                overall = selected_solution.get('overall_score')
                code = selected_solution.get('code_availability_score')
                overall_str = f"{overall}/10" if overall is not None else "N/A"
                code_str = f"{code}/10" if code is not None else "N/A"
                print(f"\n{c['G']}✓ Auto-selected top solution: {title}{c['R']}")
                print(f"{c['G']}  Overall Score: {overall_str} | Code Availability Score: {code_str}{c['R']}\n")
            else:  # unified
                score = selected_solution.get('score', 'N/A')
                print(f"\n{c['G']}✓ Auto-selected top solution: {title} (Score: {score}/10){c['R']}\n")

        if save_runs:
            save_workflow_log(output_dir, state, "selection")

        # Calculate total workflow runtime
        metrics["total_runtime"] = time.time() - workflow_start
        state["metrics"] = metrics

        # Save final workflow log
        if save_runs:
            save_workflow_log(output_dir, state, "complete")

        # Print metrics summary
        if verbose:
            print(f"\n{c['C']}{'='*60}")
            print("METRICS")
            print(f"{'='*60}{c['R']}")
        if verbose:
            for stage, data in metrics["stages"].items():
                total = data["input"] + data["output"]
                provider = data.get("provider", "anthropic")
                print(f"\n[{stage.upper()}] ({provider})")
                if data["input"] == 0 and data["output"] == 0 and data.get("note"):
                    print(f"  Tokens:        {data['note']}")
                else:
                    print(f"  Input tokens:  {data['input']:,}")
                    print(f"  Output tokens: {data['output']:,}")
                    # Add Deep Research specific tokens if present
                    if data.get("citation_tokens", 0) > 0:
                        print(f"  Citation tokens: {data['citation_tokens']:,}")
                    if data.get("reasoning_tokens", 0) > 0:
                        print(f"  Reasoning tokens: {data['reasoning_tokens']:,}")
                    print(f"  Total tokens:  {total:,}")
                print(f"  Runtime:       {data['runtime']:.2f}s")

                # Display search queries if present (for Deep Research)
                if stage == "search" and data.get("search_queries", 0) > 0:
                    print(f"  Search queries: {data['search_queries']}")

                # Display academic API metrics if present (for search stage)
                if stage == "search" and "academic_api_calls" in data:
                    api_calls = data["academic_api_calls"]
                    total_calls = sum(api_calls.values())
                    if total_calls > 0:
                        print(f"\n  Academic API calls:")
                        for api, count in api_calls.items():
                            if count > 0:
                                print(f"    - {api}: {count}")
                        if data.get("academic_api_runtime"):
                            print(f"  Academic API runtime: {data['academic_api_runtime']:.2f}s")

            print(f"\n{'-'*60}")
            print("TOTAL")
            print("-"*60)
            print(f"  Input tokens:  {metrics['total_input']:,}")
            print(f"  Output tokens: {metrics['total_output']:,}")
            if metrics['total_citation'] > 0:
                print(f"  Citation tokens: {metrics['total_citation']:,}")
            if metrics['total_reasoning'] > 0:
                print(f"  Reasoning tokens: {metrics['total_reasoning']:,}")
            total_all = metrics['total_input'] + metrics['total_output'] + metrics['total_citation'] + metrics['total_reasoning']
            print(f"  Total tokens:  {total_all:,}")
            print(f"  Total runtime: {metrics['total_runtime']:.2f}s")

            # Calculate and display total cost
            total_cost = 0.0
            total_cost += state.get("extraction_cost", 0.0)
            total_cost += state.get("search_cost", 0.0)
            total_cost += state.get("baseline_cost", 0.0)
            total_cost += state.get("assessment_cost", 0.0)

            if total_cost > 0.0:
                print(f"\n{'-'*60}")
                print("COST BREAKDOWN")
                print("-"*60)
                if "baseline_cost" in state:
                    print(f"  Baseline:    ${state['baseline_cost']:.4f} USD")
                if "extraction_cost" in state:
                    print(f"  Extraction:  ${state['extraction_cost']:.4f} USD")
                if "search_cost" in state:
                    print(f"  Search:      ${state['search_cost']:.4f} USD")
                if "assessment_cost" in state:
                    print(f"  Assessment:  ${state['assessment_cost']:.4f} USD")
                print(f"\n  {c['Y']}TOTAL COST:  ${total_cost:.4f} USD{c['R']}")

            print("="*60 + "\n")

        # Save to database
        if save_runs:
            db.save_workflow(
                workflow_id,
                problem_text,
                state,
                status="completed",
                output_path=str(output_dir)
            )

            db.save_solutions(
                workflow_id,
                assessed,
                problem_domain="biomedicine",
                selected_index=selected_idx
            )

        print(f"\n{'='*60}")
        print(f"✓ WORKFLOW COMPLETE")
        print(f"{'='*60}")
        if save_runs:
            print(f"\nOutputs saved to: {output_dir}")
            print(f"  - terminal_output.log (complete terminal output)")
            print(f"  - workflow_log.json (complete workflow data)")
        else:
            print(f"\nRun not saved (output.save_runs = false)")
        print()

        return state

    finally:
        # Restore stdout and close log file
        sys.stdout = original_stdout
        if tee_output is not None:
            tee_output.close()
