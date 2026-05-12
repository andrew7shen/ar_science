"""Search Agent - finds analogous solutions across domains using web research."""

import json
import os
import re
import time
from datetime import datetime
from pathlib import Path
from anthropic import Anthropic
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import get_config
from llm_client import call_llm


def _get_analogy_for_domain(extraction: dict, domain: str) -> dict:
    """Get the analogy object for a specific domain from analogous extraction output."""
    for analogy in extraction.get('analogies', []):
        if analogy['target_domain'].lower() == domain.lower():
            return analogy
    return None


def _format_object_mappings(object_mappings: list) -> str:
    """Format object mappings as readable text for prompt."""
    lines = []
    for m in object_mappings:
        lines.append(f"  - {m['source']} → {m['target']} ({m['mapping_rationale']})")
    return "\n".join(lines)


# JSON schema for sonar-deep-research structured outputs (full version with metadata)
SOLUTION_SCHEMA_FULL = {
    "type": "array",
    "items": {
        "type": "object",
        "properties": {
            "title": {"type": "string"},
            "source_domain": {"type": "string"},
            "description": {"type": "string"},
            "key_concepts": {"type": "array", "items": {"type": "string"}, "minItems": 3, "maxItems": 5},
            "relevance": {"type": "string"},
            "sources": {"type": "array", "items": {"type": "string"}},
            "source_titles": {"type": "array", "items": {"type": "string"}},
            "software_names": {"type": "array", "items": {"type": "string"}},
            "github_repos": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "url": {"type": "string"},
                        "stars": {"type": "integer"},
                        "language": {"type": "string"},
                        "description": {"type": "string"},
                        "last_updated": {"type": "string"},
                        "maintenance_status": {"type": "string", "enum": ["active", "maintained", "stale"]}
                    },
                    "required": ["url", "stars", "language", "description", "last_updated", "maintenance_status"],
                    "additionalProperties": False
                }
            }
        },
        "required": ["title", "source_domain", "description", "key_concepts", "relevance", "sources", "source_titles", "github_repos"],
        "additionalProperties": False
    }
}

# Simplified schema - only URL and source, metadata fetched via GitHub API validation
SOLUTION_SCHEMA_SIMPLIFIED = {
    "type": "array",
    "items": {
        "type": "object",
        "properties": {
            "title": {"type": "string"},
            "source_domain": {"type": "string"},
            "description": {"type": "string"},
            "key_concepts": {"type": "array", "items": {"type": "string"}, "minItems": 3, "maxItems": 5},
            "relevance": {"type": "string"},
            "sources": {"type": "array", "items": {"type": "string"}},
            "source_titles": {"type": "array", "items": {"type": "string"}},
            "software_names": {"type": "array", "items": {"type": "string"}},
            "github_repos": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "url": {"type": "string"},
                        "source": {"type": "string", "enum": ["paper", "author", "search"]}
                    },
                    "required": ["url", "source"],
                    "additionalProperties": False
                }
            }
        },
        "required": ["title", "source_domain", "description", "key_concepts", "relevance", "sources", "source_titles", "github_repos"],
        "additionalProperties": False
    }
}


def _search_github_by_software_name(software_names: list[str]) -> list[str]:
    """
    Search for GitHub repos using software names via GitHub search API.

    Args:
        software_names: List of software/library names to search for

    Returns:
        List of GitHub URLs found
    """
    from agents.academic_apis import _get_github_headers

    import requests

    headers = _get_github_headers()
    found_urls = []

    for name in software_names:
        if not name:
            continue

        try:
            # Search GitHub repositories API
            search_url = "https://api.github.com/search/repositories"
            params = {
                "q": name,
                "sort": "stars",
                "order": "desc",
                "per_page": 5
            }

            response = requests.get(search_url, headers=headers, params=params, timeout=10)

            if response.status_code == 200:
                results = response.json()
                for item in results.get('items', []):
                    repo_url = item.get('html_url', '')
                    repo_name = item.get('name', '').lower()
                    # Check if repo name matches the software name (case-insensitive)
                    if name.lower() in repo_name or repo_name in name.lower():
                        found_urls.append(repo_url)
                        break  # Take the best match for this software name

        except Exception:
            continue

    return found_urls


def validate_github_repos(solutions: list) -> tuple[list, dict]:
    """
    Validate GitHub URLs via API and fetch real metadata.
    Removes invalid repos and enriches valid ones with accurate data.
    Falls back to fetching paper pages, then web search by software name.

    Args:
        solutions: List of solution dicts with github_repos

    Returns:
        Tuple of (updated solutions list, validation stats dict)
    """
    from agents.academic_apis import _get_github_headers, _fetch_repo_from_url, fetch_github_urls_from_paper_page

    verbose = get_config("output.verbose_validation", True)
    headers = _get_github_headers()
    stats = {"total": 0, "valid": 0, "invalid": 0, "api_calls": 0, "paper_fallback_found": 0, "search_fallback_found": 0}

    for solution in solutions:
        validated_repos = []
        had_invalid_repos = False
        seen_urls = set()

        for repo in solution.get('github_repos', []):
            stats["total"] += 1
            url = repo.get('url', '')

            # Skip obviously invalid URLs
            if not url or 'github.com' not in url:
                stats["invalid"] += 1
                had_invalid_repos = True
                continue

            # Fetch repo details from GitHub API
            stats["api_calls"] += 1
            repo_data = _fetch_repo_from_url(url)

            if repo_data:
                # Preserve the source field from original if present
                repo_data['source'] = repo.get('source', 'unknown')
                repo_data['validated'] = True
                validated_repos.append(repo_data)
                seen_urls.add(repo_data['url'])
                stats["valid"] += 1
            else:
                stats["invalid"] += 1
                had_invalid_repos = True
                if verbose:
                    print(f"      ✗ Invalid GitHub URL: {url}")

        # Fallback 1: if no valid repos, try fetching from paper pages
        if len(validated_repos) == 0:
            paper_urls = solution.get('sources', [])
            if paper_urls:
                if verbose:
                    print(f"      ↻ Fallback 1: Searching paper pages for GitHub URLs...")
                found_urls = fetch_github_urls_from_paper_page(paper_urls)

                for found_url in found_urls:
                    if found_url in seen_urls:
                        continue
                    # Validate the found URL
                    stats["api_calls"] += 1
                    repo_data = _fetch_repo_from_url(found_url)

                    if repo_data:
                        if repo_data['url'] in seen_urls:
                            continue
                        repo_data['source'] = 'paper_fallback'
                        repo_data['validated'] = True
                        validated_repos.append(repo_data)
                        seen_urls.add(repo_data['url'])
                        stats["paper_fallback_found"] += 1
                        if verbose:
                            print(f"      ✓ Found repo from paper: {found_url}")

        # Fallback 2: if still no valid repos, try searching by software name
        if len(validated_repos) == 0:
            software_names = solution.get('software_names', [])
            if software_names:
                if verbose:
                    print(f"      ↻ Fallback 2: Searching GitHub for software names: {software_names}")
                found_urls = _search_github_by_software_name(software_names)

                for found_url in found_urls:
                    if found_url in seen_urls:
                        continue
                    # Validate the found URL
                    stats["api_calls"] += 1
                    repo_data = _fetch_repo_from_url(found_url)

                    if repo_data:
                        if repo_data['url'] in seen_urls:
                            continue
                        repo_data['source'] = 'search_fallback'
                        repo_data['validated'] = True
                        validated_repos.append(repo_data)
                        seen_urls.add(repo_data['url'])
                        stats["search_fallback_found"] += 1
                        if verbose:
                            print(f"      ✓ Found repo via search: {found_url}")

        solution['github_repos'] = validated_repos

    return solutions, stats


def search_solutions(extraction: dict, abstraction_level: str = "conceptual", output_dir: Path = None) -> list:
    """
    Search for analogous solutions across target domains using configured provider.

    Args:
        extraction: Dict from extract_problem with abstraction_levels, key_terms, target_domains
        abstraction_level: Which abstraction level to use (concrete/conceptual/mathematical)
        output_dir: Output directory for the current run (for debug files)

    Returns:
        List of solutions with source_domain tracking
    """
    provider = get_config("search.provider", "web_search")

    if provider == "web_search":
        return _search_with_claude_web_search(extraction, abstraction_level)
    elif provider == "perplexity":
        return _search_with_perplexity(extraction, abstraction_level, output_dir=output_dir)
    else:
        raise ValueError(f"Unknown search provider: {provider}")


def _search_with_claude_web_search(extraction: dict, abstraction_level: str = "conceptual") -> list:
    """
    Search for analogous solutions using Claude's web search tool.

    Args:
        extraction: Dict from extract_problem with abstraction_levels, key_terms, target_domains
        abstraction_level: Which abstraction level to use (concrete/conceptual/mathematical)

    Returns:
        List of solutions with source_domain tracking
    """
    # Initialize Claude client
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise ValueError("ANTHROPIC_API_KEY environment variable not set")

    client = Anthropic(api_key=api_key)

    # Limit domains based on config
    num_domains = get_config("search.num_domains_to_search", 3)
    target_domains = extraction['target_domains'][:num_domains]
    print(f"  Searching {len(target_domains)} domain(s): {', '.join(target_domains)}")

    all_solutions = []
    total_tokens = {"input": 0, "output": 0, "provider": "anthropic"}

    # Get the selected abstraction level
    selected_abstraction = None
    for level in extraction['abstraction_levels']:
        if level['level'].lower() == abstraction_level.lower():
            selected_abstraction = level
            break

    if not selected_abstraction:
        raise ValueError(f"Abstraction level '{abstraction_level}' not found in extraction")

    abstraction_text = f"{selected_abstraction['level'].upper()}: {selected_abstraction['description']}"

    key_terms_text = ", ".join(extraction['key_terms'])

    # Print which workflow is being used
    print(f"  Using Search Workflow A: Single-stage search (solutions + repos)")

    # Check if we should find GitHub repos during search
    find_github_repos = get_config("search.find_github_repos", False)
    repos_per_solution = get_config("search.repos_per_solution_workflow_a", 3)
    num_solutions_per_domain = get_config("search.num_solutions_per_domain", 3)
    solution_sources = get_config("search.solution_sources", "all")

    # Build source types text based on config
    if solution_sources == "academic":
        source_types = "- Academic papers and research"
    else:  # "all"
        source_types = """- Classic algorithms and proven approaches
- Academic papers and research
- Industry solutions and techniques
- Theoretical frameworks"""

    # Load appropriate prompt template based on config
    use_diverse = get_config("search.use_diverse_prompts", False)

    if find_github_repos:
        prompt_filename = "search_with_github_diverse.txt" if use_diverse else "search_with_github.txt"
    else:
        prompt_filename = "search_diverse.txt" if use_diverse else "search.txt"

    prompt_path = Path("prompts") / prompt_filename
    print(f"  Using prompt: {prompt_path}")

    with open(prompt_path, "r") as f:
        prompt_template = f.read()

    # Search each domain with web research
    for domain in target_domains:
        print(f"  Researching {domain}...")

        # Fill in prompt template
        if find_github_repos:
            research_prompt = prompt_template.format(
                domain=domain,
                problem_summary=extraction['problem_summary'],
                abstraction_levels=abstraction_text,
                key_terms=key_terms_text,
                repos_per_solution=repos_per_solution,
                num_solutions=num_solutions_per_domain,
                source_types=source_types
            )
        else:
            research_prompt = prompt_template.format(
                domain=domain,
                problem_summary=extraction['problem_summary'],
                abstraction_levels=abstraction_text,
                key_terms=key_terms_text,
                num_solutions=num_solutions_per_domain,
                source_types=source_types
            )

        # Call Claude with web search enabled
        try:
            model = get_config("model.search_model") or get_config("model.name", "claude-sonnet-4-20250514")
            response = client.messages.create(
                model=model,
                max_tokens=get_config("search.max_tokens", 16000),
                tools=[{
                    "type": "web_search_20250305",
                    "name": "web_search",
                    "max_uses": get_config("search.max_web_searches", 5)
                }],
                messages=[{
                    "role": "user",
                    "content": research_prompt
                }]
            )

            # Process tool use and final response
            solutions, tokens = _process_research_response(response, client, domain, find_github_repos)
            if solutions:
                all_solutions.extend(solutions)
            total_tokens["input"] += tokens["input"]
            total_tokens["output"] += tokens["output"]

        except Exception as e:
            print(f"Warning: Research failed for {domain}: {e}")
            continue

    return all_solutions, total_tokens


def _search_with_perplexity(extraction: dict, abstraction_level: str = "conceptual", output_dir: Path = None) -> list:
    """
    Search for analogous solutions using Perplexity API.

    Args:
        extraction: Dict from extract_problem with abstraction_levels, key_terms, target_domains
        abstraction_level: Which abstraction level to use (concrete/conceptual/mathematical)
        output_dir: Output directory for the current run (for debug files)

    Returns:
        List of solutions with source_domain tracking and embedded GitHub repos
    """
    import requests

    # Check for API key
    api_key = os.getenv("PERPLEXITY_API_KEY")
    if not api_key:
        raise ValueError("PERPLEXITY_API_KEY environment variable not set (required for perplexity provider)")

    # Get model from config
    model = get_config("search.perplexity_model", "sonar-reasoning")

    # Perplexity API configuration
    url = "https://api.perplexity.ai/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    # Limit domains based on config
    num_domains = get_config("search.num_domains_to_search", 3)
    target_domains = extraction['target_domains'][:num_domains]
    print(f"  Searching {len(target_domains)} domain(s): {', '.join(target_domains)}")

    all_solutions = []
    total_tokens = {
        "input": 0,
        "output": 0,
        "provider": "perplexity",
        "academic_api_calls": {"semantic_scholar": 0, "arxiv": 0, "crossref": 0, "github_api": 0},
        "academic_api_runtime": 0.0
    }

    # Detect reasoning type
    reasoning_type = get_config("extraction.reasoning_type", "hierarchical")

    # For hierarchical: get the selected abstraction level
    # For analogous: we'll get domain-specific context per domain later
    abstraction_text = None
    if reasoning_type == "hierarchical":
        selected_abstraction = None
        for level in extraction['abstraction_levels']:
            if level['level'].lower() == abstraction_level.lower():
                selected_abstraction = level
                break

        if not selected_abstraction:
            raise ValueError(f"Abstraction level '{abstraction_level}' not found in extraction")

        abstraction_text = f"{selected_abstraction['level'].upper()}: {selected_abstraction['description']}"

    key_terms_text = ", ".join(extraction['key_terms'])

    # Check if academic API verification is enabled
    use_academic_apis = get_config("search.use_academic_apis", False)

    # Print which workflow is being used
    if use_academic_apis:
        print(f"  Using Search Workflow B: Two-stage academic search (concepts → papers → repos)")
    else:
        print(f"  Using Search Workflow A: Single-stage search (solutions + repos)")

    # Check if we should find GitHub repos during search
    find_github_repos = get_config("search.find_github_repos", False)
    repos_per_solution = get_config("search.repos_per_solution_workflow_a", 3)
    num_solutions_per_domain = get_config("search.num_solutions_per_domain", 3)
    solution_sources = get_config("search.solution_sources", "all")

    # Build source types text based on config
    if solution_sources == "academic":
        source_types = "- Academic papers and research"
    else:  # "all"
        source_types = """- Classic algorithms and proven approaches
- Academic papers and research
- Industry solutions and techniques
- Theoretical frameworks"""

    # Load appropriate prompt template based on config and model
    use_diverse = get_config("search.use_diverse_prompts", False)

    # Get GitHub repo schema setting (simplified vs full)
    github_repo_schema = get_config("search.github_repo_schema", "simplified")

    if use_academic_apis:
        # Use concept-only prompts (no sources required)
        prompt_filename = "search_concepts_diverse.txt" if use_diverse else "search_concepts.txt"
    elif reasoning_type == "analogous" and find_github_repos and model == "sonar-deep-research":
        # Use analogous-specific prompt for deep research
        if use_diverse:
            prompt_filename = "analogous_search_with_github_deep_research_diverse_validated.txt"
        else:
            prompt_filename = "analogous_search_with_github_deep_research_nondiverse_validated.txt"
    elif find_github_repos:
        # Add Deep Research prompt support
        if model == "sonar-deep-research":
            # Select prompt based on schema: v2 = simplified (URL+source only), v1 = full (with metadata)
            if github_repo_schema == "simplified":
                prompt_filename = "search_with_github_deep_research_diverse_validated.txt"
            else:
                prompt_filename = "search_with_github_deep_research_diverse.txt"
        elif use_diverse:
            prompt_filename = "search_with_github_diverse.txt"
        else:
            prompt_filename = "search_with_github.txt"
    else:
        prompt_filename = "search_diverse.txt" if use_diverse else "search.txt"

    prompt_path = Path("prompts") / prompt_filename
    print(f"  Using prompt: {prompt_path}")

    with open(prompt_path, "r") as f:
        prompt_template = f.read()

    # Search each domain with Perplexity
    for domain in target_domains:
        if find_github_repos:
            print(f"  Researching {domain} with Perplexity {model} (with GitHub discovery)...")
        else:
            print(f"  Researching {domain} with Perplexity {model}...")

        # Fill in prompt template based on reasoning type
        if reasoning_type == "analogous":
            # Get domain-specific analogy context
            analogy = _get_analogy_for_domain(extraction, domain)
            if analogy:
                analogy_title = analogy['analogy_title']
                object_mappings_text = _format_object_mappings(analogy['object_mappings'])
                shared_relations = analogy['shared_relations']
            else:
                # Fallback if analogy not found for domain
                analogy_title = f"Analogy to {domain}"
                object_mappings_text = "(No explicit mappings available)"
                shared_relations = extraction.get('problem_relations', [''])[0] if extraction.get('problem_relations') else ""

            research_prompt = prompt_template.format(
                domain=domain,
                problem_summary=extraction['problem_summary'],
                analogy_title=analogy_title,
                object_mappings=object_mappings_text,
                shared_relations=shared_relations,
                key_terms=key_terms_text,
                repos_per_solution=repos_per_solution,
                num_solutions=num_solutions_per_domain,
                source_types=source_types
            )
        elif find_github_repos:
            research_prompt = prompt_template.format(
                domain=domain,
                problem_summary=extraction['problem_summary'],
                abstraction_levels=abstraction_text,
                key_terms=key_terms_text,
                repos_per_solution=repos_per_solution,
                num_solutions=num_solutions_per_domain,
                source_types=source_types
            )
        else:
            research_prompt = prompt_template.format(
                domain=domain,
                problem_summary=extraction['problem_summary'],
                abstraction_levels=abstraction_text,
                key_terms=key_terms_text,
                num_solutions=num_solutions_per_domain,
                source_types=source_types
            )

        # Call Perplexity API
        try:
            # Use higher max_tokens for sonar-deep-research (needs space for reasoning tokens)
            if model == "sonar-deep-research":
                max_tokens = get_config("search.perplexity_deep_research_max_tokens", 16000)
            else:
                max_tokens = get_config("search.max_tokens", 4000)

            payload = {
                "model": model,
                "messages": [
                    {
                        "role": "user",
                        "content": research_prompt
                    }
                ],
                "temperature": 0.2,
                "max_tokens": max_tokens
            }

            # Add reasoning_effort for sonar-deep-research (low/medium/high)
            if model == "sonar-deep-research":
                reasoning_effort = get_config("search.perplexity_reasoning_effort", "low")
                payload["reasoning_effort"] = reasoning_effort

            # Add JSON schema for sonar-deep-research
            if model == "sonar-deep-research":
                # Select schema based on config
                schema = SOLUTION_SCHEMA_SIMPLIFIED if github_repo_schema == "simplified" else SOLUTION_SCHEMA_FULL
                payload["response_format"] = {
                    "type": "json_schema",
                    "json_schema": {
                        "name": "solutions",
                        "schema": schema
                    }
                }

            # Increase timeout for Deep Research (first schema request can take 30-60s, plus research time)
            timeout = 300 if model == "sonar-deep-research" else 30
            response = requests.post(url, json=payload, headers=headers, timeout=timeout)
            response.raise_for_status()

            result = response.json()

            # Track token usage from Perplexity response
            if "usage" in result:
                usage = result["usage"]
                total_tokens["input"] += usage.get("prompt_tokens", 0)
                total_tokens["output"] += usage.get("completion_tokens", 0)

                # Deep research metrics
                if model == "sonar-deep-research":
                    total_tokens.setdefault("citation_tokens", 0)
                    total_tokens.setdefault("reasoning_tokens", 0)
                    total_tokens.setdefault("search_queries", 0)
                    total_tokens["citation_tokens"] += usage.get("citation_tokens", 0)
                    total_tokens["reasoning_tokens"] += usage.get("reasoning_tokens", 0)
                    total_tokens["search_queries"] += usage.get("search_query_count", 0)

            # Extract response text
            if "choices" in result and len(result["choices"]) > 0:
                choice = result["choices"][0]
                response_text = choice["message"]["content"].strip()

                # Check for truncation (Deep Research)
                if model == "sonar-deep-research":
                    finish_reason = choice.get("finish_reason", "unknown")
                    output_tokens = result.get("usage", {}).get("completion_tokens", 0)
                    print(f"    Deep Research output: {output_tokens:,} tokens (finish_reason: {finish_reason})")
                    if finish_reason == "length":
                        raise RuntimeError(
                            f"Pipeline stopped: Output truncated for domain '{domain}'. "
                            f"Token limit reached ({max_tokens:,} tokens). "
                            f"Increase perplexity_deep_research_max_tokens in config.yaml to continue."
                        )

                # Parse JSON from response
                concepts = _parse_perplexity_response(response_text, domain, find_github_repos=find_github_repos, output_dir=output_dir)

                if use_academic_apis:
                    # Stage 2: Verify concepts with academic APIs
                    if concepts:
                        print(f"    Found {len(concepts)} concept(s), verifying with academic APIs...")
                        verified_count = 0
                        for concept in concepts:
                            verified, api_calls, runtime = _verify_concept_with_academic_apis(concept)

                            # Only include solutions with papers (repos optional but preferred)
                            has_papers = len(verified.get('papers', [])) > 0
                            if has_papers:
                                all_solutions.append(verified)
                                verified_count += 1
                            else:
                                print(f"      ⨯ Skipping '{concept['title']}' - no relevant papers found")

                            # Track API calls
                            for api, count in api_calls.items():
                                total_tokens['academic_api_calls'][api] += count
                            total_tokens['academic_api_runtime'] += runtime

                        print(f"    Kept {verified_count}/{len(concepts)} solutions with relevant evidence")
                    else:
                        print(f"    Warning: No valid concepts parsed from {domain}")
                else:
                    # Legacy path: concepts already have sources
                    if concepts:
                        all_solutions.extend(concepts)
                        print(f"    Found {len(concepts)} solution(s)")
                    else:
                        print(f"    Warning: No valid solutions parsed from {domain}")
            else:
                print(f"  Warning: Unexpected response format from Perplexity for {domain}")

        except requests.exceptions.RequestException as e:
            print(f"  Warning: Perplexity API request failed for {domain}: {e}")
            continue
        except Exception as e:
            print(f"  Warning: Error processing {domain}: {e}")
            continue

    # Validate GitHub repos via API if enabled (recommended for deep research)
    should_validate = get_config("search.validate_github_repos", False)
    if should_validate and find_github_repos and not use_academic_apis:
        print(f"  Validating GitHub repos via API...")
        all_solutions, validation_stats = validate_github_repos(all_solutions)
        total_tokens["github_validation"] = validation_stats
        paper_fallback = validation_stats.get('paper_fallback_found', 0)
        search_fallback = validation_stats.get('search_fallback_found', 0)
        fallback_parts = []
        if paper_fallback > 0:
            fallback_parts.append(f"{paper_fallback} from papers")
        if search_fallback > 0:
            fallback_parts.append(f"{search_fallback} from search")
        fallback_msg = f", {' + '.join(fallback_parts)} via fallback" if fallback_parts else ""
        print(f"    GitHub validation: {validation_stats['valid']}/{validation_stats['total']} repos valid ({validation_stats['invalid']} invalid{fallback_msg})")

        # Save debug file after GitHub validation
        if output_dir:
            debug_dir = output_dir / "debug"
            debug_dir.mkdir(exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            post_validation_file = debug_dir / f"solutions_post_github_validation_{timestamp}.json"
            with open(post_validation_file, 'w') as f:
                json.dump(all_solutions, f, indent=2)
            print(f"    Saved post-validation debug: {post_validation_file.name}")

    return all_solutions, total_tokens


def _verify_concept_with_academic_apis(concept: dict) -> tuple:
    """
    Verify concept by finding real papers and GitHub repos.

    Args:
        concept: Concept dict from Perplexity

    Returns:
        Tuple of (enriched_concept, api_call_counts, runtime_seconds)
    """
    from agents.academic_apis import match_papers_to_concept, discover_github_repos_for_paper

    api_start = time.time()
    api_calls = {"semantic_scholar": 0, "arxiv": 0, "crossref": 0, "github_api": 0}

    # Query academic APIs
    enabled_apis = get_config("search.academic_apis.enabled", ["semantic_scholar", "arxiv"])
    papers = match_papers_to_concept(concept, apis=enabled_apis)

    # Track API calls
    if "semantic_scholar" in enabled_apis:
        api_calls["semantic_scholar"] += 1
    if "arxiv" in enabled_apis:
        api_calls["arxiv"] += 1
    if "crossref" in enabled_apis:
        api_calls["crossref"] += 1

    if not papers:
        # No papers found or all filtered out due to low relevance
        min_score = get_config('search.academic_apis.min_paper_relevance_score', 45)
        concept['source_validation'] = 'unverified'
        concept['sources'] = []
        concept['source_titles'] = []
        concept['papers'] = []  # Empty list for consistency
        concept['github_repos'] = []
        print(f"      ⚠ No relevant papers found for: {concept['title']} (min score: {min_score}/100)")
        return concept, api_calls, time.time() - api_start

    # Enrich with paper data
    concept['sources'] = [p['url'] for p in papers]
    concept['source_titles'] = [p['title'] for p in papers]
    concept['source_validation'] = 'verified'

    # Discover GitHub repos per paper (fixed number per paper for even distribution)
    repos_per_paper = get_config("search.academic_apis.repos_per_paper", 2)

    all_repos = []  # Flat list for backward compatibility

    # Search all papers and get repos for each
    for paper in papers:
        repos, github_api_calls = discover_github_repos_for_paper(paper, concept)
        api_calls["github_api"] += github_api_calls

        # Take top N repos for this paper
        paper['github_repos'] = repos[:repos_per_paper]
        all_repos.extend(repos[:repos_per_paper])

    # Delete abstracts from all papers (not needed for assessment, saves tokens)
    for paper in papers:
        if 'abstract' in paper:
            del paper['abstract']

    concept['papers'] = papers  # Save full paper metadata with repos for logging
    concept['github_repos'] = all_repos  # Flat list for backward compatibility

    total_repos = sum(len(p.get('github_repos', [])) for p in papers)
    github_calls = api_calls.get('github_api', 0)

    # Warn if no repos found due to filtering
    if total_repos == 0:
        min_repo_score = get_config('search.academic_apis.min_repo_relevance_score', 20)
        print(f"      ⚠ {concept['title']}: {len(papers)} papers, 0 repos (all filtered, min repo score: {min_repo_score}/100)")
    else:
        print(f"      ✓ {concept['title']}: {len(papers)} papers, {total_repos} repos ({github_calls} GitHub API calls)")

    return concept, api_calls, time.time() - api_start


def _parse_perplexity_response(response_text: str, domain: str, find_github_repos: bool = False, output_dir: Path = None) -> list:
    """Parse Perplexity response and extract solutions.

    Args:
        response_text: Response text from Perplexity API
        domain: Domain being researched
        find_github_repos: Whether GitHub repos should be in the response
        output_dir: Output directory for the current run (for debug files)
    """
    original_text = response_text  # Save for debugging
    save_debug = get_config("output.save_debug_files", False)

    # Set up debug directory and timestamp
    if output_dir:
        debug_dir = output_dir / "debug"
    else:
        debug_dir = Path("data/outputs/debug")
    timestamp = time.strftime("%Y%m%d_%H%M%S")

    # Save raw response to debug file
    if save_debug:
        debug_dir.mkdir(parents=True, exist_ok=True)
        raw_file = debug_dir / f"raw_response_{domain}_{timestamp}.txt"
        with open(raw_file, 'w') as f:
            f.write(original_text)

    # Strip reasoning tokens from deep-research responses
    # Handle both closed tags and unclosed tags (strip from <think> to end if no </think>)
    if '<think>' in response_text:
        if '</think>' in response_text:
            response_text = re.sub(r'<think>.*?</think>', '', response_text, flags=re.DOTALL)
        else:
            # No closing tag - find where JSON starts and strip everything before
            json_start = response_text.find('[{')
            if json_start != -1:
                response_text = response_text[json_start:]
            else:
                # Try to find ```json block
                json_block = response_text.find('```json')
                if json_block != -1:
                    response_text = response_text[json_block:]
    if '<thinking>' in response_text:
        if '</thinking>' in response_text:
            response_text = re.sub(r'<thinking>.*?</thinking>', '', response_text, flags=re.DOTALL)
        else:
            json_start = response_text.find('[{')
            if json_start != -1:
                response_text = response_text[json_start:]
    response_text = response_text.strip()

    # Handle various JSON formats similar to existing code

    # 1. Try to extract JSON from markdown code blocks
    if "```json" in response_text:
        start = response_text.find("```json") + 7
        end = response_text.find("```", start)
        response_text = response_text[start:end].strip()
    elif "```" in response_text:
        start = response_text.find("```") + 3
        end = response_text.find("```", start)
        response_text = response_text[start:end].strip()

    # 2. Try to find JSON array in the text
    # Look for '[{' (start of JSON array of objects) to avoid matching citation refs like [9]
    if not response_text.startswith("["):
        # First try to find '[{' which is definitely JSON array of objects
        bracket_pos = response_text.find("[{")
        if bracket_pos == -1:
            # Fallback to '[' but verify it's followed by valid JSON
            bracket_pos = response_text.find("[")
        if bracket_pos != -1:
            response_text = response_text[bracket_pos:]

    # Save final JSON extraction to debug file
    if save_debug:
        json_file = debug_dir / f"json_to_parse_{domain}_{timestamp}.txt"
        with open(json_file, 'w') as f:
            f.write(response_text)

    # 3. Try to parse, handling "extra data" errors by extracting just the array
    try:
        solutions = json.loads(response_text)
        # Ensure we have a list
        if not isinstance(solutions, list):
            print(f"    Warning: Expected JSON array, got {type(solutions).__name__}")
            return []
    except json.JSONDecodeError as e:
        # If "Extra data" error, try to extract just the JSON array portion
        if "Extra data" in str(e):
            try:
                # Use JSONDecoder to get the first valid JSON object/array
                from json import JSONDecoder
                decoder = JSONDecoder()
                solutions, idx = decoder.raw_decode(response_text)
                # Ensure we have a list
                if not isinstance(solutions, list):
                    print(f"    Warning: Expected JSON array, got {type(solutions).__name__}")
                    return []
            except Exception:
                print(f"    Warning: Could not parse solutions from {domain}: {e}")
                return []
        # Try to recover from unescaped quotes in strings
        elif "Expecting ','" in str(e) or "Unterminated string" in str(e):
            try:
                fixed_text = response_text
                fixed_text = re.sub(
                    r':\s*"([^"]*)"([^"]*)"',
                    lambda m: ': "' + m.group(1) + '\\"' + m.group(2) + '"' if m.group(2) and not m.group(2).startswith(',') and not m.group(2).startswith('}') and not m.group(2).startswith(']') else m.group(0),
                    fixed_text
                )
                solutions = json.loads(fixed_text)
                if not isinstance(solutions, list):
                    print(f"    Warning: Expected JSON array, got {type(solutions).__name__}")
                    return []
                print(f"    Recovered from unescaped quotes JSON error for domain {domain}")
            except Exception as recovery_error:
                print(f"    Warning: Could not parse solutions from {domain}: {e}")
                print(f"    Quote recovery also failed: {recovery_error}")
                return []
        else:
            print(f"    Warning: Could not parse solutions from {domain}: {e}")
            return []

    # Validate solutions are dicts (filter out malformed responses like integers)
    valid_solutions = []
    for solution in solutions:
        if not isinstance(solution, dict):
            continue
        # Ensure required fields exist
        if 'title' not in solution or 'source_domain' not in solution:
            continue
        # Ensure github_repos field exists if GitHub discovery is enabled
        if find_github_repos and 'github_repos' not in solution:
            solution['github_repos'] = []
        valid_solutions.append(solution)

    return valid_solutions


def _process_research_response(response, client, domain, find_github_repos=False):
    """Process Claude's research response including tool calls.

    Args:
        response: Claude API response
        client: Anthropic client
        domain: Domain being researched
        find_github_repos: Whether GitHub repos should be in the response
    """
    messages = [{
        "role": "user",
        "content": f"Research {domain} domain"
    }]

    tokens = {"input": response.usage.input_tokens, "output": response.usage.output_tokens}

    # Handle tool use loop - web_search_20250305 is executed server-side automatically
    # No need to manually provide tool results
    while response.stop_reason == "tool_use":
        # Add assistant response to messages
        messages.append({
            "role": "assistant",
            "content": response.content
        })

        # Continue conversation - API will automatically execute web searches
        # and include results in the next response
        model = get_config("model.search_model") or get_config("model.name", "claude-sonnet-4-20250514")
        response = client.messages.create(
            model=model,
            max_tokens=get_config("search.max_tokens", 16000),
            tools=[{
                "type": "web_search_20250305",
                "name": "web_search",
                "max_uses": get_config("search.max_web_searches", 5)
            }],
            messages=messages
        )
        tokens["input"] += response.usage.input_tokens
        tokens["output"] += response.usage.output_tokens

    # Extract final JSON response
    response_text = ""
    for block in response.content:
        if block.type == "text":
            response_text = block.text.strip()
            break

    # Parse JSON from response - handle various formats
    # 1. Try to extract JSON from markdown code blocks
    if "```json" in response_text:
        start = response_text.find("```json") + 7
        end = response_text.find("```", start)
        response_text = response_text[start:end].strip()
    elif "```" in response_text:
        start = response_text.find("```") + 3
        end = response_text.find("```", start)
        response_text = response_text[start:end].strip()

    # 2. Try to find JSON array in the text
    if not response_text.startswith("["):
        # Look for first [ character
        bracket_pos = response_text.find("[")
        if bracket_pos != -1:
            response_text = response_text[bracket_pos:]

    try:
        solutions = json.loads(response_text)

        # Ensure we have a list
        if not isinstance(solutions, list):
            print(f"    Warning: Expected JSON array, got {type(solutions).__name__}")
            return [], tokens

        # Validate solutions are dicts (filter out malformed responses)
        valid_solutions = []
        for solution in solutions:
            if not isinstance(solution, dict):
                print(f"    Warning: Skipping invalid solution (got {type(solution).__name__}, expected dict)")
                continue
            if 'title' not in solution or 'source_domain' not in solution:
                print(f"    Warning: Skipping solution missing required fields")
                continue
            # Ensure github_repos field exists if GitHub discovery is enabled
            if find_github_repos and 'github_repos' not in solution:
                solution['github_repos'] = []
            valid_solutions.append(solution)

        return valid_solutions, tokens
    except json.JSONDecodeError as e:
        # Try to recover from "Extra data" error
        if "Extra data" in str(e):
            try:
                from json import JSONDecoder
                decoder = JSONDecoder()
                solutions, idx = decoder.raw_decode(response_text)
                if not isinstance(solutions, list):
                    print(f"Warning: Expected JSON array from {domain}, got {type(solutions).__name__}")
                    return [], tokens
                print(f"    Recovered from 'Extra data' JSON error for domain {domain}")
                # Continue to validation below
                valid_solutions = []
                for solution in solutions:
                    if not isinstance(solution, dict):
                        continue
                    if 'title' not in solution or 'source_domain' not in solution:
                        continue
                    if find_github_repos and 'github_repos' not in solution:
                        solution['github_repos'] = []
                    valid_solutions.append(solution)
                return valid_solutions, tokens
            except Exception:
                print(f"Warning: Could not parse solutions from {domain}: {e}")
                return [], tokens
        # Try to recover from unescaped quotes in strings
        elif "Expecting ','" in str(e) or "Unterminated string" in str(e):
            try:
                fixed_text = response_text
                fixed_text = re.sub(
                    r':\s*"([^"]*)"([^"]*)"',
                    lambda m: ': "' + m.group(1) + '\\"' + m.group(2) + '"' if m.group(2) and not m.group(2).startswith(',') and not m.group(2).startswith('}') and not m.group(2).startswith(']') else m.group(0),
                    fixed_text
                )
                solutions = json.loads(fixed_text)
                if not isinstance(solutions, list):
                    print(f"Warning: Expected JSON array from {domain}, got {type(solutions).__name__}")
                    return [], tokens
                print(f"    Recovered from unescaped quotes JSON error for domain {domain}")
                # Continue to validation below
                valid_solutions = []
                for solution in solutions:
                    if not isinstance(solution, dict):
                        continue
                    if 'title' not in solution or 'source_domain' not in solution:
                        continue
                    if find_github_repos and 'github_repos' not in solution:
                        solution['github_repos'] = []
                    valid_solutions.append(solution)
                return valid_solutions, tokens
            except Exception as recovery_error:
                print(f"Warning: Could not parse solutions from {domain}: {e}")
                print(f"    Quote recovery also failed: {recovery_error}")
                return [], tokens
        else:
            print(f"Warning: Could not parse solutions from {domain}: {e}")
            return [], tokens


def print_solutions(solutions: list):
    """Pretty print search results."""
    print("\n" + "="*60)
    print(f"FOUND {len(solutions)} SOLUTIONS")
    print("="*60)

    for idx, sol in enumerate(solutions, 1):
        print(f"\n{idx}. {sol['title']} ({sol['source_domain']})")
        print(f"   {sol['description']}")
        print(f"   Concepts: {', '.join(sol['key_concepts'])}")

        # Show papers with relevance scores (Workflow B)
        if 'papers' in sol and sol['papers']:
            total_papers = len(sol['papers'])
            total_repos = sum(len(p.get('github_repos', [])) for p in sol['papers'])
            print(f"   Papers ({total_papers}) with GitHub Repos ({total_repos}):")

            for paper in sol['papers']:
                score = paper.get('relevance_score', 'N/A')
                title = paper.get('title', 'Unknown')
                paper_url = paper.get('url', '')
                print(f"      • [{score}] {title}")
                if paper_url:
                    print(f"        {paper_url}")

                # Show repos found for this paper
                if 'github_repos' in paper and paper['github_repos']:
                    for repo in paper['github_repos']:
                        repo_score = repo.get('relevance_score', 'N/A')
                        url = repo.get('url', 'Unknown')
                        stars = repo.get('stars', 0)
                        # Extract owner/repo from URL
                        repo_name = url.split('github.com/')[-1] if 'github.com' in url else url
                        print(f"        → [{repo_score}] {repo_name} (⭐{stars}) - {url}")

        # Fallback to legacy sources format (Workflow A)
        elif 'sources' in sol and sol['sources']:
            if 'source_titles' in sol and sol['source_titles']:
                print(f"   Sources ({len(sol['sources'])}):")
                for idx, src_title in enumerate(sol['source_titles'][:3]):
                    src_url = sol['sources'][idx] if idx < len(sol['sources']) else ''
                    print(f"      • {src_title}")
                    if src_url:
                        print(f"        {src_url}")
            else:
                print(f"   Sources:")
                for src_url in sol['sources'][:3]:
                    print(f"      • {src_url}")

            # Show GitHub repos (Workflow A - not mapped to papers)
            if 'github_repos' in sol and sol['github_repos']:
                print(f"   GitHub Repos ({len(sol['github_repos'])}):")
                for repo in sol['github_repos']:
                    url = repo.get('url', 'Unknown')
                    stars = repo.get('stars', 0)
                    # Extract owner/repo from URL
                    repo_name = url.split('github.com/')[-1] if 'github.com' in url else url
                    print(f"      • {repo_name} (⭐{stars}) - {url}")

    print("="*60 + "\n")
