"""Baseline Agent - generates solutions via direct LLM query without analogical reasoning."""

import os
import json
import sys
import time
from pathlib import Path
from anthropic import Anthropic
import requests

# Add parent directories to path
sys.path.insert(0, str(Path(__file__).parent.parent))  # Add src
sys.path.insert(0, str(Path(__file__).parent.parent.parent))  # Add project root for eval
from config import get_config
from eval.dataset_eval.domain_matching import filter_matched_domains
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


def _get_abstraction_description(abstraction_levels: list, target_level: str) -> str:
    """Extract description for a specific abstraction level."""
    for level in abstraction_levels:
        if level['level'].lower() == target_level.lower():
            return level['description']
    return ""


def validate_and_filter_github_repos(solutions: list) -> tuple[list, dict]:
    """
    Validate GitHub URLs and remove broken ones.

    Args:
        solutions: List of solutions with github_repos field

    Returns:
        Tuple of (filtered_solutions, validation_stats)
    """
    total_repos = 0
    valid_repos = 0
    invalid_repos = 0

    for solution in solutions:
        original_repos = solution.get("github_repos", [])
        total_repos += len(original_repos)

        validated_repos = []
        for repo in original_repos:
            # Handle both dict and string formats
            url = repo.get("url") if isinstance(repo, dict) else repo

            if not url or not url.startswith("https://github.com"):
                invalid_repos += 1
                continue

            # Check if URL exists (simple HEAD request)
            try:
                response = requests.head(url, timeout=5, allow_redirects=True)
                if response.status_code == 200:
                    validated_repos.append(repo)
                    valid_repos += 1
                else:
                    invalid_repos += 1
            except:
                invalid_repos += 1

        solution["github_repos"] = validated_repos

    validation_stats = {
        "total_repos": total_repos,
        "valid_repos": valid_repos,
        "invalid_repos": invalid_repos,
        "hallucination_rate": (invalid_repos / total_repos * 100) if total_repos > 0 else 0.0
    }

    return solutions, validation_stats


def _filter_domains_with_llm_judge(domains: list, ground_truth_domain: str, client, judge_model: str, stop_on_first_match: bool = False) -> tuple[list, dict]:
    """
    Filter domains using LLM judge to determine matches.

    Args:
        domains: List of discovered domain names
        ground_truth_domain: Ground truth domain name
        client: Anthropic client
        judge_model: Model to use for judging
        stop_on_first_match: If True, stop evaluating after finding first match (default: False)

    Returns:
        Tuple of (matched_domains, tokens_dict)
    """
    matched_domains = []
    total_tokens = {'input': 0, 'output': 0}

    # Load prompt template
    root = Path(__file__).parent.parent.parent
    prompt_path = root / "eval" / "prompts" / "domain_match_judge.txt"
    with open(prompt_path, 'r') as f:
        template = f.read()

    for i, domain in enumerate(domains, 1):
        # Format prompt
        prompt = template.format(
            ground_truth_domain=ground_truth_domain,
            discovered_domain=domain
        )

        try:
            response = client.messages.create(
                model=judge_model,
                max_tokens=100,
                temperature=0,
                messages=[{"role": "user", "content": prompt}]
            )

            # Track tokens
            total_tokens['input'] += response["usage"]["input_tokens"]
            total_tokens['output'] += response["usage"]["output_tokens"]

            # Parse response
            if hasattr(response, 'content') and response.content:
                response_text = response.content[0].text.strip()
                is_match = response_text.lower().startswith('yes')

                if is_match:
                    matched_domains.append(domain)

                    # Early stopping if enabled
                    if stop_on_first_match:
                        remaining = len(domains) - i
                        if remaining > 0:
                            print(f"    Found match - skipping evaluation of {remaining} remaining domains")
                        break
        except Exception as e:
            print(f"  Warning: Domain match evaluation failed for {domain}: {str(e)}")
            continue

    return matched_domains, total_tokens


def generate_baseline_solutions(problem_text: str, mode: str = "simple_llm", ground_truth_domain: str = None, domain_judge_client=None, domain_judge_model: str = None) -> tuple[list, dict, list]:
    """
    Generate baseline solutions using direct LLM query without extraction or analogical reasoning.

    Args:
        problem_text: The biomedical problem description
        mode: Baseline type ("simple_llm", "no_domain_llm", etc.)
        ground_truth_domain: Ground truth domain for evaluation filtering
        domain_judge_client: Anthropic client for LLM domain judge (optional)
        domain_judge_model: Model name for domain judge (optional)

    Returns:
        Tuple of (list of solutions in standard format, token usage dict, discovered_domains)

    Solution format matches search agent output:
        {
            "title": str,
            "source_domain": str,
            "description": str,
            "key_concepts": list[str],
            "relevance": str,
            "sources": list[str],
            "source_titles": list[str],
            "github_repos": list[dict]
        }
    """
    if mode == "simple_llm":
        return _baseline_simple_llm(problem_text, ground_truth_domain=ground_truth_domain, domain_judge_client=domain_judge_client, domain_judge_model=domain_judge_model)
    elif mode == "no_domain_llm":
        return _baseline_no_domain_llm(problem_text)
    else:
        raise ValueError(f"Unknown baseline mode: {mode}. Supported modes: 'simple_llm', 'no_domain_llm'")


def _discover_domains(problem_text: str, num_domains: int, model: str) -> tuple[list[str], dict]:
    """
    First LLM call: Discover N relevant non-biomedical domains.

    Args:
        problem_text: The biomedical problem
        num_domains: How many domains to find
        model: Model to use

    Returns:
        Tuple of (list of domain names, token usage dict)
    """
    # Check for domain override
    override_domains = get_config("baseline.override_domains", None)
    if override_domains and len(override_domains) > 0:
        print(f"      Using override domains: {', '.join(override_domains)}")

        domains = override_domains[:num_domains]  # Respect num_domains limit

        tokens = {
            "input": 0,
            "output": 0,
            "provider": "anthropic",
            "note": "Baseline domains overridden via config (no LLM call)"
        }

        return domains, tokens

    # Load prompt template - configurable path
    discovery_prompt_path = get_config("baseline.domain_discovery_prompt_path", "prompts/baseline_domain_discovery.txt")
    template_path = Path(__file__).parent.parent.parent / discovery_prompt_path
    if not template_path.exists():
        raise FileNotFoundError(f"Domain discovery prompt template not found: {template_path}")

    with open(template_path, 'r') as f:
        prompt_template = f.read()

    # Format prompt
    prompt = prompt_template.format(
        problem_text=problem_text,
        num_domains=num_domains
    )

    # Retry up to 3 times if wrong number of domains returned
    max_retries = 3
    total_input_tokens = 0
    total_output_tokens = 0

    for attempt in range(max_retries):
        # Call LLM
        try:
            temperature = get_config("baseline.temperature", 1.0)
            response = call_llm(
                model=model,
                max_tokens=1000,  # Short response, just domain names
                temperature=temperature,
                messages=[{
                    "role": "user",
                    "content": prompt
                }]
            )

            # Track tokens from this attempt
            total_input_tokens += response["usage"]["input_tokens"]
            total_output_tokens += response["usage"]["output_tokens"]

            # Extract response text
            response_text = response["content"].strip()

            # Strip thinking tokens if present (for models like Gemini)
            if '<think>' in response_text:
                if '</think>' in response_text:
                    import re
                    response_text = re.sub(r'<think>.*?</think>', '', response_text, flags=re.DOTALL)
                else:
                    # If thinking token not closed, try to find JSON array
                    json_start = response_text.find('[')
                    if json_start != -1:
                        response_text = response_text[json_start:]

            response_text = response_text.strip()

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

            # If still no JSON, try to find a JSON array in the response
            if not response_text.startswith('['):
                json_start = response_text.find('[')
                if json_start != -1:
                    json_end = response_text.rfind(']')
                    if json_end != -1:
                        response_text = response_text[json_start:json_end+1]

            # Parse JSON array of domain names
            domains = json.loads(response_text)

            # Validate
            if not isinstance(domains, list):
                raise ValueError("LLM response is not a JSON array")

            # Check if we got the right number of domains
            if len(domains) != num_domains:
                if attempt < max_retries - 1:
                    print(f"Warning: Expected {num_domains} domains, got {len(domains)}. Retrying... (attempt {attempt + 1}/{max_retries})")
                    continue  # Retry
                else:
                    print(f"Warning: Expected {num_domains} domains, got {len(domains)} after {max_retries} attempts. Using first {num_domains} domains.")
                    domains = domains[:num_domains]  # Truncate as fallback after all retries

            # Success - return domains and total tokens
            tokens = {
                "input": total_input_tokens,
                "output": total_output_tokens,
                "provider": response["provider"]
            }

            return domains, tokens

        except json.JSONDecodeError as e:
            if attempt < max_retries - 1:
                print(f"Error parsing domain discovery JSON (attempt {attempt + 1}/{max_retries}): {e}")
                print(f"Response text ({len(response_text)} chars): {response_text[:1000]}...")
                print("Retrying...")
                continue
            else:
                # Final attempt failed
                print(f"Error parsing domain discovery JSON after {max_retries} attempts: {e}")
                print(f"Full response text ({len(response_text)} chars):")
                print(response_text if len(response_text) < 2000 else response_text[:2000] + "...")
                tokens = {
                    "input": total_input_tokens,
                    "output": total_output_tokens,
                    "provider": response["provider"] if 'response' in locals() else "unknown",
                    "note": f"JSON parsing error after {max_retries} attempts: {str(e)}"
                }
                return [], tokens

        except Exception as e:
            # Catch any other unexpected errors during this attempt
            if attempt < max_retries - 1:
                print(f"Unexpected error during domain discovery (attempt {attempt + 1}/{max_retries}): {e}")
                print("Retrying...")
                continue
            else:
                # Final attempt failed with unexpected error
                print(f"Error discovering domains after {max_retries} attempts: {e}")
                tokens = {
                    "input": total_input_tokens,
                    "output": total_output_tokens,
                    "provider": response["provider"] if 'response' in locals() else "unknown",
                    "note": f"Error after {max_retries} attempts: {str(e)}"
                }
                return [], tokens

    # Should never reach here, but as a fallback
    return [], {"input": total_input_tokens, "output": total_output_tokens, "provider": "unknown", "note": "Unexpected code path"}


def _find_solutions_in_domain(
    problem_text: str,
    domain: str,
    num_solutions: int,
    model: str,
    extraction_context: dict = None
) -> tuple[list, dict]:
    """
    Per-domain LLM call: Find M solutions from a specific domain.

    Args:
        problem_text: The biomedical problem
        domain: Specific domain to search (e.g., "Computer Science")
        num_solutions: How many solutions to find in this domain
        model: Model to use
        extraction_context: Optional extraction context (for hybrid mode)

    Returns:
        Tuple of (list of solutions, token usage dict)
    """
    # Load prompt template
    if extraction_context is None:
        # Standard baseline mode - use configurable prompt path
        search_prompt_path = get_config("baseline.search_prompt_path", "prompts/baseline_domain_solutions.txt")
        template_path = Path(__file__).parent.parent.parent / search_prompt_path
    else:
        # Hybrid mode with extraction context - select prompt based on reasoning type and diversity setting
        reasoning_type = extraction_context.get("reasoning_type", "analogous")
        use_diverse = get_config("search.use_diverse_prompts", False)

        if reasoning_type == "hierarchical":
            template_path = Path(__file__).parent.parent.parent / "prompts/hierarchical_extraction_llm.txt"
        else:
            # Analogous reasoning - choose based on diversity setting
            if use_diverse:
                template_path = Path(__file__).parent.parent.parent / "prompts/analogous_extraction_llm.txt"
            else:
                template_path = Path(__file__).parent.parent.parent / "prompts/analogous_extraction_llm_nondiverse.txt"

    if not template_path.exists():
        raise FileNotFoundError(f"Per-domain solution prompt template not found: {template_path}")

    with open(template_path, 'r') as f:
        prompt_template = f.read()

    # Format prompt
    if extraction_context is None:
        # Standard baseline prompt
        prompt = prompt_template.format(
            problem_text=problem_text,
            domain=domain,
            num_solutions=num_solutions
        )
    else:
        # Hybrid mode prompt with extraction context
        prompt = prompt_template.format(
            problem_summary=extraction_context.get("problem_summary", problem_text),
            domain=domain,
            num_solutions=num_solutions,
            key_terms=", ".join(extraction_context.get("key_terms", [])),
            # Hierarchical or analogous specific fields
            abstraction_description=extraction_context.get("abstraction_description", ""),
            analogy_title=extraction_context.get("analogy_title", ""),
            object_mappings=extraction_context.get("object_mappings", ""),
            shared_relations=extraction_context.get("shared_relations", "")
        )

    # Call LLM
    try:
        # Use configurable max_tokens
        max_tokens = get_config("baseline.max_tokens", 10000)
        temperature = get_config("baseline.temperature", 1.0)

        response = call_llm(
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            messages=[{
                "role": "user",
                "content": prompt
            }]
        )

        # Extract response text
        response_text = response["content"].strip()

        # Save debug file if enabled
        save_debug = get_config("output.save_debug_files", False)
        if save_debug:
            debug_dir = Path("data/outputs/debug")
            debug_dir.mkdir(parents=True, exist_ok=True)
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            raw_file = debug_dir / f"llm_fallback_{domain}_{timestamp}.txt"
            with open(raw_file, 'w') as f:
                f.write(f"Domain: {domain}\n")
                f.write(f"Model: {model}\n")
                f.write(f"Max tokens: {max_tokens}\n")
                f.write(f"Provider: {response['provider']}\n")
                f.write(f"Input tokens: {response['usage']['input_tokens']}\n")
                f.write(f"Output tokens: {response['usage']['output_tokens']}\n")
                f.write(f"\n{'='*70}\nRAW RESPONSE:\n{'='*70}\n\n")
                f.write(response_text)

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

            response_text = '\n'.join(lines[start_idx:end_idx])

        # Parse JSON response with advanced error handling
        try:
            solutions = json.loads(response_text)
            # Ensure we have a list
            if not isinstance(solutions, list):
                print(f"    Warning: Expected JSON array for domain '{domain}', got {type(solutions).__name__}")
                tokens = {
                    "input": response["usage"]["input_tokens"],
                    "output": response["usage"]["output_tokens"],
                    "provider": "anthropic",
                    "note": f"Invalid response type: {type(solutions).__name__}"
                }
                return [], tokens
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
                        print(f"    Warning: Expected JSON array for domain '{domain}', got {type(solutions).__name__}")
                        tokens = {
                            "input": response["usage"]["input_tokens"],
                            "output": response["usage"]["output_tokens"],
                            "provider": response["provider"] if 'response' in locals() else "unknown",
                            "note": f"Invalid response type after recovery: {type(solutions).__name__}"
                        }
                        return [], tokens
                    print(f"    Recovered from 'Extra data' JSON error for domain '{domain}'")
                except Exception:
                    print(f"    Error parsing solutions JSON for domain '{domain}': {e}")
                    print(f"    Response text: {response_text[:500]}...")
                    tokens = {
                        "input": response["usage"]["input_tokens"],
                        "output": response["usage"]["output_tokens"],
                        "provider": response["provider"] if 'response' in locals() else "unknown",
                        "note": f"JSON parsing error: {str(e)}"
                    }
                    return [], tokens
            else:
                print(f"    Error parsing solutions JSON for domain '{domain}': {e}")
                print(f"    Response text: {response_text[:500]}...")
                tokens = {
                    "input": response["usage"]["input_tokens"],
                    "output": response["usage"]["output_tokens"],
                    "provider": "anthropic",
                    "note": f"JSON parsing error: {str(e)}"
                }
                return [], tokens

        # Validate solutions are dicts and have required fields (filter out malformed responses)
        valid_solutions = []
        required_fields = ["title", "source_domain", "description", "key_concepts", "relevance", "sources", "source_titles"]

        for solution in solutions:
            # Ensure it's a dict
            if not isinstance(solution, dict):
                continue

            # Ensure required fields exist
            if 'title' not in solution or 'source_domain' not in solution:
                continue

            # Fill in missing fields with defaults
            for field in required_fields:
                if field not in solution:
                    solution[field] = "" if field != "key_concepts" and field != "sources" and field != "source_titles" else []

            # Ensure github_repos field exists
            if "github_repos" not in solution:
                solution["github_repos"] = []

            # Ensure source_domain matches the requested domain
            solution["source_domain"] = domain

            valid_solutions.append(solution)

        # Log if we filtered out any solutions
        if len(valid_solutions) != len(solutions):
            print(f"    Filtered {len(solutions) - len(valid_solutions)} malformed solution(s) for domain '{domain}'")

        # Track tokens
        tokens = {
            "input": response["usage"]["input_tokens"],
            "output": response["usage"]["output_tokens"],
            "provider": response["provider"]
        }

        return valid_solutions, tokens

    except json.JSONDecodeError as e:
        # This should be caught above, but keep as fallback
        print(f"    Unexpected JSON error for domain '{domain}': {e}")
        print(f"    Response text: {response_text[:500]}...")
        tokens = {
            "input": response["usage"]["input_tokens"] if 'response' in locals() else 0,
            "output": response["usage"]["output_tokens"] if 'response' in locals() else 0,
            "provider": "anthropic",
            "note": f"JSON parsing error: {str(e)}"
        }
        return [], tokens

    except Exception as e:
        print(f"Error finding solutions for domain '{domain}': {e}")
        tokens = {
            "input": 0,
            "output": 0,
            "provider": "anthropic",
            "note": f"Error for domain '{domain}': {str(e)}"
        }
        return [], tokens


def _find_solutions_in_domain_with_deep_research(
    problem_text: str,
    domain: str,
    num_solutions: int,
    reasoning_effort: str = "low"
) -> tuple[list, dict]:
    """
    Per-domain deep research call: Find M solutions using Perplexity deep research.

    Args:
        problem_text: The biomedical problem
        domain: Specific domain to search
        num_solutions: How many solutions to find in this domain
        reasoning_effort: Reasoning effort level ("low", "medium", "high")

    Returns:
        Tuple of (list of solutions, token usage dict with deep research metrics)
    """
    import re
    import time
    from datetime import datetime

    # Check for API key
    api_key = os.getenv("PERPLEXITY_API_KEY")
    if not api_key:
        raise ValueError("PERPLEXITY_API_KEY environment variable not set (required for deep research)")

    # Load prompt template
    template_path = Path(__file__).parent.parent.parent / "prompts/baseline_domain_solutions_deep_research.txt"
    if not template_path.exists():
        raise FileNotFoundError(f"Deep research prompt template not found: {template_path}")

    with open(template_path, 'r') as f:
        prompt_template = f.read()

    # Build source types text
    solution_sources = get_config("search.solution_sources", "all")
    if solution_sources == "academic":
        source_types = "- Academic papers and research"
    else:  # "all"
        source_types = """- Classic algorithms and proven approaches
- Academic papers and research
- Industry solutions and techniques
- Theoretical frameworks"""

    # Format prompt
    prompt = prompt_template.format(
        problem_text=problem_text,
        domain=domain,
        num_solutions=num_solutions,
        source_types=source_types
    )

    # Perplexity API configuration
    url = "https://api.perplexity.ai/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    # Get config settings
    max_tokens = get_config("baseline.perplexity_deep_research_max_tokens", 25000)

    # Define JSON schema (matches search.py SOLUTION_SCHEMA_SIMPLIFIED)
    schema = {
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

    payload = {
        "model": "sonar-deep-research",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.2,
        "max_tokens": max_tokens,
        "reasoning_effort": reasoning_effort,
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "solutions",
                "schema": schema
            }
        }
    }

    # Call Perplexity API
    try:
        response = requests.post(url, json=payload, headers=headers, timeout=300)
        response.raise_for_status()
        result = response.json()

        # Track token usage
        tokens = {
            "input": 0,
            "output": 0,
            "provider": "perplexity",
            "citation_tokens": 0,
            "reasoning_tokens": 0,
            "search_queries": 0
        }

        if "usage" in result:
            usage = result["usage"]
            tokens["input"] = usage.get("prompt_tokens", 0)
            tokens["output"] = usage.get("completion_tokens", 0)
            tokens["citation_tokens"] = usage.get("citation_tokens", 0)
            tokens["reasoning_tokens"] = usage.get("reasoning_tokens", 0)
            tokens["search_queries"] = usage.get("search_query_count", 0)

        # Extract response text
        if "choices" not in result or len(result["choices"]) == 0:
            print(f"  Warning: Unexpected response format from Perplexity for {domain}")
            return [], tokens

        choice = result["choices"][0]
        response_text = choice["message"]["content"].strip()

        # Save debug file if enabled
        save_debug = get_config("output.save_debug_files", False)
        if save_debug:
            debug_dir = Path("data/outputs/debug")
            debug_dir.mkdir(parents=True, exist_ok=True)
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            raw_file = debug_dir / f"perplexity_deep_research_{domain}_{timestamp}.txt"
            with open(raw_file, 'w') as f:
                f.write(f"Domain: {domain}\n")
                f.write(f"Model: sonar-deep-research\n")
                f.write(f"Max tokens: {max_tokens}\n")
                f.write(f"Reasoning effort: {reasoning_effort}\n")
                f.write(f"Finish reason: {choice.get('finish_reason', 'N/A')}\n")
                f.write(f"Input tokens: {tokens['input']}\n")
                f.write(f"Output tokens: {tokens['output']}\n")
                f.write(f"Citation tokens: {tokens['citation_tokens']}\n")
                f.write(f"Reasoning tokens: {tokens['reasoning_tokens']}\n")
                f.write(f"Search queries: {tokens['search_queries']}\n")
                f.write(f"\n{'='*70}\nRAW RESPONSE:\n{'='*70}\n\n")
                f.write(response_text)

        # Check for truncation and print token usage
        finish_reason = choice.get("finish_reason", "unknown")
        output_tokens = result.get("usage", {}).get("completion_tokens", 0)
        print(f"    Deep Research output: {output_tokens:,} tokens (finish_reason: {finish_reason})")
        if finish_reason == "length":
            raise RuntimeError(
                f"Output truncated for domain '{domain}'. "
                f"Increase baseline.perplexity_deep_research_max_tokens to continue."
            )

        # Strip reasoning tokens (handle <think> tags)
        if '<think>' in response_text:
            if '</think>' in response_text:
                response_text = re.sub(r'<think>.*?</think>', '', response_text, flags=re.DOTALL)
            else:
                json_start = response_text.find('[{')
                if json_start != -1:
                    response_text = response_text[json_start:]

        response_text = response_text.strip()

        # Extract JSON from markdown code blocks
        if "```json" in response_text:
            start = response_text.find("```json") + 7
            end = response_text.find("```", start)
            response_text = response_text[start:end].strip()
        elif "```" in response_text:
            start = response_text.find("```") + 3
            end = response_text.find("```", start)
            response_text = response_text[start:end].strip()

        # Try to parse JSON with advanced error handling
        try:
            solutions = json.loads(response_text)
            # Ensure we have a list
            if not isinstance(solutions, list):
                print(f"    Warning: Expected JSON array for domain '{domain}', got {type(solutions).__name__}")
                return [], tokens
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
                        print(f"    Warning: Expected JSON array for domain '{domain}', got {type(solutions).__name__}")
                        return [], tokens
                    print(f"    Recovered from 'Extra data' JSON error for domain '{domain}'")
                except Exception:
                    print(f"    Error parsing solutions JSON for domain '{domain}': {e}")
                    return [], tokens
            # Try to recover from unescaped quotes in strings
            elif "Expecting ','" in str(e) or "Unterminated string" in str(e):
                try:
                    import re
                    fixed_text = response_text
                    fixed_text = re.sub(
                        r':\s*"([^"]*)"([^"]*)"',
                        lambda m: ': "' + m.group(1) + '\\"' + m.group(2) + '"' if m.group(2) and not m.group(2).startswith(',') and not m.group(2).startswith('}') and not m.group(2).startswith(']') else m.group(0),
                        fixed_text
                    )
                    solutions = json.loads(fixed_text)
                    if not isinstance(solutions, list):
                        print(f"    Warning: Expected JSON array for domain '{domain}', got {type(solutions).__name__}")
                        return [], tokens
                    print(f"    Recovered from unescaped quotes JSON error for domain '{domain}'")
                except Exception as recovery_error:
                    print(f"    Error parsing solutions JSON for domain '{domain}': {e}")
                    print(f"    Quote recovery also failed: {recovery_error}")
                    return [], tokens
            else:
                print(f"    Error parsing solutions JSON for domain '{domain}': {e}")
                return [], tokens

        # Validate solutions are dicts and have required fields (filter out malformed responses)
        valid_solutions = []
        required_fields = ["title", "source_domain", "description", "key_concepts", "relevance", "sources", "source_titles"]

        for solution in solutions:
            # Ensure it's a dict
            if not isinstance(solution, dict):
                continue

            # Ensure required fields exist
            if 'title' not in solution or 'source_domain' not in solution:
                continue

            # Fill in missing fields with defaults
            for field in required_fields:
                if field not in solution:
                    solution[field] = "" if field != "key_concepts" and field != "sources" and field != "source_titles" else []

            # Ensure github_repos field exists
            if "github_repos" not in solution:
                solution["github_repos"] = []

            # Ensure source_domain matches the requested domain
            solution["source_domain"] = domain

            valid_solutions.append(solution)

        # Log if we filtered out any solutions
        if len(valid_solutions) != len(solutions):
            print(f"    Filtered {len(solutions) - len(valid_solutions)} malformed solution(s) for domain '{domain}'")

        return valid_solutions, tokens

    except requests.exceptions.RequestException as e:
        print(f"  Error: Perplexity API request failed for {domain}: {e}")
        tokens = {
            "input": 0,
            "output": 0,
            "provider": "perplexity",
            "note": f"API request failed: {str(e)}"
        }
        return [], tokens

    except Exception as e:
        print(f"  Error finding solutions for domain '{domain}': {e}")
        tokens = {
            "input": 0,
            "output": 0,
            "provider": "perplexity",
            "note": f"Error: {str(e)}"
        }
        return [], tokens


def _baseline_combined_prompt(problem_text: str, ground_truth_domain: str = None, domain_judge_client=None, domain_judge_model: str = None) -> tuple[list, dict, list]:
    """
    Combined baseline: Domain discovery + solution finding in a single LLM call.

    Args:
        problem_text: The biomedical problem description
        ground_truth_domain: Optional ground truth domain for filtering
        domain_judge_client: Optional client for domain judging
        domain_judge_model: Optional model name for domain judging

    Returns:
        Tuple of (list of solutions, token usage dict, discovered_domains)
    """
    # Load configuration
    num_domains = get_config("baseline.num_domains_to_search", 3)
    num_solutions_per_domain = get_config("baseline.num_solutions_per_domain", 3)
    model = get_config("baseline.baseline_model") or get_config("model.name", "claude-sonnet-4-5-20250929")
    combined_prompt_path = get_config("baseline.combined_prompt_path", "prompts/baseline_combined_domain_solutions_nondiverse.txt")

    print(f"  Combined baseline: Finding {num_domains} domains + {num_solutions_per_domain} solutions per domain in single call...")
    print(f"  Using combined prompt: {combined_prompt_path}")

    # Load prompt template
    template_path = Path(__file__).parent.parent.parent / combined_prompt_path
    if not template_path.exists():
        raise FileNotFoundError(f"Combined prompt template not found: {template_path}")

    with open(template_path, 'r') as f:
        prompt_template = f.read()

    # Format prompt
    prompt = prompt_template.format(
        problem_text=problem_text,
        num_domains=num_domains,
        num_solutions_per_domain=num_solutions_per_domain
    )

    # Single LLM call
    try:
        max_tokens = get_config("baseline.max_tokens", 10000)
        temperature = get_config("baseline.temperature", 1.0)

        response = call_llm(
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            messages=[{
                "role": "user",
                "content": prompt
            }]
        )

        # Extract response text
        response_text = response["content"].strip()

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

        # Extract domains and solutions
        domains = result.get("domains", [])
        solutions = result.get("solutions", [])

        print(f"  Discovered domains: {', '.join(domains)}")
        print(f"  Found {len(solutions)} total solutions")

        # Initialize domain judge tokens
        domain_judge_tokens = {'input': 0, 'output': 0}

        # Conditional domain filtering (only keep solutions from matched domains if enabled)
        if get_config("baseline.only_search_matched_domains", False):
            if ground_truth_domain and domains:
                # Use LLM judge if client and model are provided, otherwise fall back to dictionary
                if domain_judge_client is not None and domain_judge_model is not None:
                    print(f"  [Domain Filtering with LLM Judge] Evaluating {len(domains)} domains...")
                    stop_on_first = get_config("baseline.domain_judge_stop_on_first_match", True)
                    matched_domains, domain_judge_tokens = _filter_domains_with_llm_judge(
                        domains, ground_truth_domain, domain_judge_client, domain_judge_model, stop_on_first_match=stop_on_first
                    )
                    print(f"    Filtered to {len(matched_domains)} matched domains (using {domain_judge_model})")
                else:
                    # Fall back to dictionary-based matching
                    matched_domains = filter_matched_domains(domains, ground_truth_domain)
                    print(f"    Filtered to {len(matched_domains)} matched domains (using CLOSE_MATCHES dictionary)")

                print(f"    Ground truth: {ground_truth_domain}")
                print(f"    Discovered: {', '.join(domains)}")
                print(f"    Matched: {', '.join(matched_domains) if matched_domains else 'None'}")

                # Filter solutions to only keep those from matched domains
                if matched_domains:
                    matched_domains_lower = [d.lower() for d in matched_domains]
                    solutions = [s for s in solutions if s.get("source_domain", "").lower() in matched_domains_lower]
                    print(f"    Filtered to {len(solutions)} solutions from matched domains")
                else:
                    print(f"  No matched domains found - no solutions returned")
                    solutions = []

        # Validate GitHub repos
        validate_repos = get_config("baseline.validate_github_repos", True)
        if validate_repos:
            solutions, validation_stats = validate_and_filter_github_repos(solutions)
        else:
            validation_stats = {"total_repos": 0, "valid_repos": 0, "invalid_repos": 0, "hallucination_rate": 0.0}

        # Token tracking
        tokens = {
            "input": response["usage"]["input_tokens"],
            "output": response["usage"]["output_tokens"],
            "provider": response["provider"],
            "github_validation": validation_stats,
            "num_llm_calls": 1  # Single combined call
        }

        # Include domain judge tokens if used
        if domain_judge_tokens['input'] > 0 or domain_judge_tokens['output'] > 0:
            tokens['domain_judge_input'] = domain_judge_tokens['input']
            tokens['domain_judge_output'] = domain_judge_tokens['output']

        return solutions, tokens, domains

    except json.JSONDecodeError as e:
        print(f"  Error: Failed to parse JSON response: {e}")
        tokens = {
            "input": response["usage"]["input_tokens"] if 'response' in locals() else 0,
            "output": response["usage"]["output_tokens"] if 'response' in locals() else 0,
            "provider": response["provider"] if 'response' in locals() else "unknown",
            "note": f"JSON parse error: {str(e)}"
        }
        return [], tokens, []
    except Exception as e:
        print(f"  Error in combined baseline: {e}")
        tokens = {
            "input": 0,
            "output": 0,
            "provider": "unknown",
            "note": f"Error: {str(e)}"
        }
        return [], tokens, []


def _baseline_simple_llm(problem_text: str, ground_truth_domain: str = None, domain_judge_client=None, domain_judge_model: str = None) -> tuple[list, dict, list]:
    """
    Multi-call baseline: Domain discovery + per-domain solution finding.

    Args:
        problem_text: The biomedical problem description

    Returns:
        Tuple of (list of solutions, token usage dict, discovered_domains)
    """
    # Check if combined prompt mode is enabled
    use_combined = get_config("baseline.use_combined_prompt", False)
    if use_combined:
        return _baseline_combined_prompt(problem_text, ground_truth_domain, domain_judge_client, domain_judge_model)

    # Load configuration
    num_domains = get_config("baseline.num_domains_to_search", 3)
    num_solutions_per_domain = get_config("baseline.num_solutions_per_domain", 3)
    model = get_config("baseline.baseline_model") or get_config("model.name", "claude-sonnet-4-5-20250929")
    use_deep_research = get_config("baseline.use_deep_research", False)
    domain_only = get_config("baseline.domain_only", False)

    if domain_only:
        print(f"  Domain-only baseline: Discovering {num_domains} domains (skipping solution generation)...")
    elif use_deep_research:
        print(f"  Multi-call baseline with deep research: Discovering {num_domains} domains, then using Perplexity deep research to find {num_solutions_per_domain} solutions per domain...")
    else:
        search_prompt_path = get_config("baseline.search_prompt_path", "prompts/baseline_domain_solutions.txt")
        print(f"  Multi-call baseline: Discovering {num_domains} domains, then finding {num_solutions_per_domain} solutions per domain...")
        print(f"  Using search prompt: {search_prompt_path}")

    # Print domain discovery prompt path
    discovery_prompt_path = get_config("baseline.domain_discovery_prompt_path", "prompts/baseline_domain_discovery.txt")
    print(f"  Using domain discovery prompt: {discovery_prompt_path}")

    # Call 1: Discover domains
    domains, tokens_discovery = _discover_domains(problem_text, num_domains, model)

    if not domains:
        # Domain discovery failed
        print("  Error: Domain discovery failed")
        return [], tokens_discovery, []

    print(f"  Discovered domains: {', '.join(domains)}")

    # Initialize domain judge tokens
    domain_judge_tokens = {'input': 0, 'output': 0}

    # Conditional domain filtering (only search matched domains if enabled)
    if get_config("baseline.only_search_matched_domains", False):
        if ground_truth_domain and domains:
            # Use LLM judge if client and model are provided, otherwise fall back to dictionary
            if domain_judge_client is not None and domain_judge_model is not None:
                print(f"  [Domain Filtering with LLM Judge] Evaluating {len(domains)} domains...")
                # Check if we should stop after first match (optimization)
                stop_on_first = get_config("baseline.domain_judge_stop_on_first_match", True)
                matched_domains, domain_judge_tokens = _filter_domains_with_llm_judge(
                    domains, ground_truth_domain, domain_judge_client, domain_judge_model, stop_on_first_match=stop_on_first
                )
                print(f"    Filtered to {len(matched_domains)} matched domains (using {domain_judge_model})")
            else:
                # Fall back to dictionary-based matching
                matched_domains = filter_matched_domains(domains, ground_truth_domain)
                print(f"    Filtered to {len(matched_domains)} matched domains (using CLOSE_MATCHES dictionary)")

            print(f"    Ground truth: {ground_truth_domain}")
            print(f"    Discovered: {', '.join(domains)}")
            print(f"    Matched: {', '.join(matched_domains) if matched_domains else 'None'}")

            # Update domains list to only matched domains
            domains = matched_domains

            # If no matched domains, skip solution generation
            if not domains:
                print(f"  No matched domains found - skipping solution generation")
                # Include domain judge tokens even if no matches
                tokens_discovery['domain_judge_input'] = domain_judge_tokens['input']
                tokens_discovery['domain_judge_output'] = domain_judge_tokens['output']
                return [], tokens_discovery, []

    # If domain-only mode, skip solution generation
    if domain_only:
        print(f"  Skipping solution generation (domain-only mode)")
        return [], tokens_discovery, domains

    # Calls 2-N+1: Find solutions per domain
    all_solutions = []
    tokens_per_domain = []

    for i, domain in enumerate(domains, 1):
        if use_deep_research:
            # Use Perplexity deep research for solution finding
            reasoning_effort = get_config("baseline.perplexity_reasoning_effort", "low")
            print(f"  [{i}/{len(domains)}] Researching {domain} with Perplexity deep research...")
            solutions, tokens = _find_solutions_in_domain_with_deep_research(
                problem_text, domain, num_solutions_per_domain, reasoning_effort
            )
        else:
            # Use existing Claude LLM for solution finding
            print(f"  [{i}/{len(domains)}] Finding solutions in {domain}...")
            solutions, tokens = _find_solutions_in_domain(
                problem_text, domain, num_solutions_per_domain, model
            )
        all_solutions.extend(solutions)
        tokens_per_domain.append(tokens)
        print(f"      Found {len(solutions)} solutions")

    # Validate GitHub repos
    validate_repos = get_config("baseline.validate_github_repos", True) if use_deep_research else True
    if validate_repos:
        if use_deep_research:
            # Use sophisticated API validation from search.py for deep research
            from agents.search import validate_github_repos
            print(f"  Validating GitHub repos via API...")
            all_solutions, validation_stats = validate_github_repos(all_solutions)
            paper_fallback = validation_stats.get('paper_fallback_found', 0)
            search_fallback = validation_stats.get('search_fallback_found', 0)
            fallback_parts = []
            if paper_fallback > 0:
                fallback_parts.append(f"{paper_fallback} from papers")
            if search_fallback > 0:
                fallback_parts.append(f"{search_fallback} from search")
            fallback_msg = f", {' + '.join(fallback_parts)} via fallback" if fallback_parts else ""
            print(f"    GitHub validation: {validation_stats['valid']}/{validation_stats['total']} repos valid ({validation_stats['invalid']} invalid{fallback_msg})")

            # Normalize keys to match baseline format for orchestrator
            validation_stats = {
                "total_repos": validation_stats["total"],
                "valid_repos": validation_stats["valid"],
                "invalid_repos": validation_stats["invalid"],
                "hallucination_rate": (validation_stats["invalid"] / validation_stats["total"] * 100) if validation_stats["total"] > 0 else 0.0,
                "paper_fallback_found": paper_fallback,
                "search_fallback_found": search_fallback
            }
        else:
            # Use simple HEAD request validation for regular baseline
            all_solutions, validation_stats = validate_and_filter_github_repos(all_solutions)
    else:
        validation_stats = {"total_repos": 0, "valid_repos": 0, "invalid_repos": 0, "hallucination_rate": 0.0}

    # Aggregate tokens (get provider from discovery or first domain call)
    provider = tokens_discovery.get("provider", "unknown")
    if use_deep_research:
        provider = "perplexity"

    total_tokens = {
        "input": tokens_discovery["input"] + sum(t["input"] for t in tokens_per_domain),
        "output": tokens_discovery["output"] + sum(t["output"] for t in tokens_per_domain),
        "provider": provider,
        "github_validation": validation_stats,
        "num_llm_calls": 1 + len(domains),  # Track number of calls made
        "domain_judge_input": domain_judge_tokens['input'],
        "domain_judge_output": domain_judge_tokens['output']
    }

    # Add deep research metrics if using deep research
    if use_deep_research:
        total_tokens["citation_tokens"] = sum(t.get("citation_tokens", 0) for t in tokens_per_domain)
        total_tokens["reasoning_tokens"] = sum(t.get("reasoning_tokens", 0) for t in tokens_per_domain)
        total_tokens["search_queries"] = sum(t.get("search_queries", 0) for t in tokens_per_domain)

    print(f"  Total calls: {total_tokens['num_llm_calls']} ({len(all_solutions)} solutions across {len(domains)} domains)")

    return all_solutions, total_tokens, domains


def _baseline_no_domain_llm(problem_text: str) -> tuple[list, dict, list]:
    """
    Multi-call baseline: Find solutions without domain discovery.

    This is a weaker baseline that doesn't discover or constrain to specific domains.
    It makes multiple independent LLM calls to find solutions without any domain guidance.

    Args:
        problem_text: The biomedical problem description

    Returns:
        Tuple of (list of solutions, token usage dict, empty list for discovered_domains)
    """
    # Load configuration
    num_calls = get_config("baseline.num_calls", 3)
    num_solutions_per_call = get_config("baseline.num_solutions_per_call", 3)
    model = get_config("baseline.baseline_model") or get_config("model.name", "claude-sonnet-4-5-20250929")
    total_solutions = num_calls * num_solutions_per_call

    print(f"  Multi-call baseline (no domain): Making {num_calls} calls to find {num_solutions_per_call} solutions per call ({total_solutions} total) without domain constraints...")

    # Load prompt template from config or use default
    search_prompt_path = get_config("baseline.search_prompt_path", "prompts/baseline_solutions_no_domain.txt")
    print(f"  Using search prompt: {search_prompt_path}")
    template_path = Path(__file__).parent.parent.parent / search_prompt_path
    if not template_path.exists():
        raise FileNotFoundError(f"No-domain solution prompt template not found: {template_path}")

    with open(template_path, 'r') as f:
        prompt_template = f.read()

    max_tokens = get_config("baseline.max_tokens", 10000)
    temperature = get_config("baseline.temperature", 1.0)

    # Make multiple calls to generate solutions
    all_solutions = []
    tokens_per_call = []

    for i in range(num_calls):
        print(f"  [{i+1}/{num_calls}] Finding {num_solutions_per_call} solutions...")

        # Format prompt for this call
        prompt = prompt_template.format(
            problem_text=problem_text,
            num_solutions=num_solutions_per_call
        )

        # Call LLM
        try:
            response = call_llm(
                model=model,
                max_tokens=max_tokens,
                temperature=temperature,
                messages=[{
                    "role": "user",
                    "content": prompt
                }]
            )

            # Extract response text
            response_text = response["content"].strip()

            # Save debug file if enabled
            save_debug = get_config("output.save_debug_files", False)
            if save_debug:
                debug_dir = Path("data/outputs/debug")
                debug_dir.mkdir(parents=True, exist_ok=True)
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                raw_file = debug_dir / f"llm_no_domain_call{i+1}_{timestamp}.txt"
                with open(raw_file, 'w') as f:
                    f.write(f"Model: {model}\n")
                    f.write(f"Max tokens: {max_tokens}\n")
                    f.write(f"Provider: {response['provider']}\n")
                    f.write(f"Input tokens: {response['usage']['input_tokens']}\n")
                    f.write(f"Output tokens: {response['usage']['output_tokens']}\n")
                    f.write(f"\n{'='*70}\nRAW RESPONSE:\n{'='*70}\n\n")
                    f.write(response_text)

            # Remove markdown code fences if present
            if response_text.startswith("```"):
                lines = response_text.split('\n')
                start_idx = 0
                end_idx = len(lines)

                for j, line in enumerate(lines):
                    if line.startswith("```"):
                        if start_idx == 0:
                            start_idx = j + 1
                        else:
                            end_idx = j
                            break

                response_text = '\n'.join(lines[start_idx:end_idx])

            # Parse JSON response
            try:
                solutions = json.loads(response_text)
                if not isinstance(solutions, list):
                    print(f"    Warning: Expected JSON array, got {type(solutions).__name__}")
                    tokens_per_call.append({
                        "input": response["usage"]["input_tokens"],
                        "output": response["usage"]["output_tokens"],
                        "note": f"Invalid response type: {type(solutions).__name__}"
                    })
                    continue
            except json.JSONDecodeError as e:
                # Try to recover from "Extra data" error
                if "Extra data" in str(e):
                    try:
                        from json import JSONDecoder
                        decoder = JSONDecoder()
                        solutions, idx = decoder.raw_decode(response_text)
                        if not isinstance(solutions, list):
                            print(f"    Warning: Expected JSON array, got {type(solutions).__name__}")
                            tokens_per_call.append({
                                "input": response["usage"]["input_tokens"],
                                "output": response["usage"]["output_tokens"],
                                "note": f"Invalid response type after recovery: {type(solutions).__name__}"
                            })
                            continue
                        print(f"    Recovered from 'Extra data' JSON error")
                    except Exception:
                        print(f"    Error parsing solutions JSON: {e}")
                        print(f"    Response text: {response_text[:500]}...")
                        tokens_per_call.append({
                            "input": response["usage"]["input_tokens"],
                            "output": response["usage"]["output_tokens"],
                            "note": f"JSON parsing error: {str(e)}"
                        })
                        continue
                # Try to recover from unescaped quotes in strings
                elif "Expecting ','" in str(e) or "Unterminated string" in str(e):
                    try:
                        import re
                        # Fix unescaped quotes in JSON string values
                        # Pattern: finds quotes within quoted strings and escapes them
                        fixed_text = response_text
                        # Replace straight quotes within string values with escaped quotes
                        # This handles cases like: "title": "Text "quote" more text"
                        fixed_text = re.sub(
                            r':\s*"([^"]*)"([^"]*)"',
                            lambda m: ': "' + m.group(1) + '\\"' + m.group(2) + '"' if m.group(2) and not m.group(2).startswith(',') and not m.group(2).startswith('}') and not m.group(2).startswith(']') else m.group(0),
                            fixed_text
                        )
                        solutions = json.loads(fixed_text)
                        if not isinstance(solutions, list):
                            print(f"    Warning: Expected JSON array, got {type(solutions).__name__}")
                            tokens_per_call.append({
                                "input": response["usage"]["input_tokens"],
                                "output": response["usage"]["output_tokens"],
                                "note": f"Invalid response type after quote recovery: {type(solutions).__name__}"
                            })
                            continue
                        print(f"    Recovered from unescaped quotes JSON error")
                    except Exception as recovery_error:
                        print(f"    Error parsing solutions JSON: {e}")
                        print(f"    Quote recovery also failed: {recovery_error}")
                        print(f"    Response text: {response_text[:500]}...")
                        tokens_per_call.append({
                            "input": response["usage"]["input_tokens"],
                            "output": response["usage"]["output_tokens"],
                            "note": f"JSON parsing error: {str(e)}"
                        })
                        continue
                else:
                    print(f"    Error parsing solutions JSON: {e}")
                    print(f"    Response text: {response_text[:500]}...")
                    tokens_per_call.append({
                        "input": response["usage"]["input_tokens"],
                        "output": response["usage"]["output_tokens"],
                        "note": f"JSON parsing error: {str(e)}"
                    })
                    continue

            # Validate solutions from this call
            call_solutions = []
            required_fields = ["title", "source_domain", "description", "key_concepts", "relevance", "sources", "source_titles"]

            for solution in solutions:
                if not isinstance(solution, dict):
                    continue

                if 'title' not in solution or 'source_domain' not in solution:
                    continue

                # Fill in missing fields with defaults
                for field in required_fields:
                    if field not in solution:
                        solution[field] = "" if field != "key_concepts" and field != "sources" and field != "source_titles" else []

                # Ensure github_repos field exists
                if "github_repos" not in solution:
                    solution["github_repos"] = []

                call_solutions.append(solution)

            # Log if we filtered out any solutions
            if len(call_solutions) != len(solutions):
                print(f"    Filtered {len(solutions) - len(call_solutions)} malformed solution(s)")

            # Add solutions to all_solutions
            all_solutions.extend(call_solutions)
            print(f"    Found {len(call_solutions)} valid solutions")

            # Track tokens for this call
            tokens_per_call.append({
                "input": response["usage"]["input_tokens"],
                "output": response["usage"]["output_tokens"],
                "provider": response["provider"]
            })

        except Exception as e:
            print(f"    Error in call {i+1}: {e}")
            tokens_per_call.append({
                "input": 0,
                "output": 0,
                "note": f"Error: {str(e)}"
            })

    # Validate GitHub repos once at the end (on all solutions together)
    validate_repos = get_config("baseline.validate_github_repos", True)
    if validate_repos:
        all_solutions, validation_stats = validate_and_filter_github_repos(all_solutions)
    else:
        validation_stats = {"total_repos": 0, "valid_repos": 0, "invalid_repos": 0, "hallucination_rate": 0.0}

    # Aggregate tokens (use provider from first successful call)
    provider = next((t.get("provider", "unknown") for t in tokens_per_call if "provider" in t), "unknown")
    total_tokens = {
        "input": sum(t.get("input", 0) for t in tokens_per_call),
        "output": sum(t.get("output", 0) for t in tokens_per_call),
        "provider": provider,
        "github_validation": validation_stats,
        "num_llm_calls": num_calls
    }

    print(f"  Total calls: {total_tokens['num_llm_calls']} ({len(all_solutions)} solutions)")

    # Extract ALL domains from solutions (implicitly sampled by LLM when generating solutions)
    # This preserves the sampling distribution and ensures no_domain baseline tracks domains the same way as other baselines
    discovered_domains = []
    for solution in all_solutions:
        domain = solution.get('source_domain', '').strip()
        if domain:
            discovered_domains.append(domain)

    if discovered_domains:
        unique_domains = list(dict.fromkeys(discovered_domains))  # Preserve order while getting unique
        print(f"  Sampled domains: {', '.join(unique_domains)} ({len(discovered_domains)} total, {len(unique_domains)} unique)")

    return all_solutions, total_tokens, discovered_domains


def print_baseline(solutions: list):
    """
    Pretty print baseline solutions.

    Args:
        solutions: List of baseline solutions
    """
    print("\n" + "="*60)
    print("BASELINE SOLUTIONS")
    print("="*60)

    for i, sol in enumerate(solutions, 1):
        print(f"\n{i}. {sol.get('title', 'Unknown')}")
        print(f"   Domain: {sol.get('source_domain', 'Unknown')}")
        print(f"   {sol.get('description', '')[:100]}...")
        if sol.get('github_repos'):
            print(f"   GitHub: {len(sol['github_repos'])} repo(s)")

    print("\n" + "="*60 + "\n")


def generate_solutions_from_extraction(extraction: dict, abstraction_level: str) -> tuple[list, dict]:
    """
    Generate solutions using extraction context + LLM (hybrid mode).

    This mode runs extraction normally, then uses LLM to generate solutions
    based on the extracted analogies/abstractions (without web search).

    Args:
        extraction: Output from extract_problem() with analogies/abstractions
        abstraction_level: "conceptual", "concrete", or "mathematical"

    Returns:
        Tuple of (solutions list, token usage dict)
    """
    # Get config
    reasoning_type = get_config("extraction.reasoning_type", "analogous")
    num_domains = get_config("search.num_domains_to_search", 3)
    num_solutions_per_domain = get_config("search.num_solutions_per_domain", 3)
    model = get_config("model.search_model") or get_config("model.name", "claude-sonnet-4-20250514")
    use_diverse = get_config("search.use_diverse_prompts", False)

    # Print which prompt will be used
    if reasoning_type == "hierarchical":
        prompt_name = "hierarchical_extraction_llm.txt"
    else:
        prompt_name = "analogous_extraction_llm.txt" if use_diverse else "analogous_extraction_llm_nondiverse.txt"
    print(f"  Using LLM fallback with prompt: prompts/{prompt_name}")

    # Extract domains
    domains = extraction.get("target_domains", [])
    if not domains:
        # Return empty solutions if no domains found (graceful skip)
        return [], {"input": 0, "output": 0}

    # Limit to num_domains
    domains = domains[:num_domains]

    # Get problem summary
    problem_summary = extraction.get("problem_summary", "")
    key_terms = extraction.get("key_terms", [])

    # Generate solutions for each domain
    all_solutions = []
    total_input_tokens = 0
    total_output_tokens = 0
    provider = "unknown"

    for domain in domains:
        # Build extraction context based on reasoning type
        if reasoning_type == "hierarchical":
            # Get abstraction level description
            abstraction_description = _get_abstraction_description(
                extraction.get("abstraction_levels", []),
                abstraction_level
            )

            if not abstraction_description:
                raise ValueError(f"Abstraction level '{abstraction_level}' not found in extraction")

            extraction_context = {
                "reasoning_type": "hierarchical",
                "problem_summary": problem_summary,
                "key_terms": key_terms,
                "abstraction_description": abstraction_description
            }
        else:
            # Analogous reasoning - get analogy for this domain
            analogy = _get_analogy_for_domain(extraction, domain)

            if analogy:
                # Format object mappings
                object_mappings_text = _format_object_mappings(analogy.get("object_mappings", []))

                extraction_context = {
                    "reasoning_type": "analogous",
                    "problem_summary": problem_summary,
                    "key_terms": key_terms,
                    "analogy_title": analogy.get("analogy_title", f"Analogy to {domain}"),
                    "object_mappings": object_mappings_text,
                    "shared_relations": analogy.get("shared_relations", "")
                }
            else:
                # Fallback if no analogy found for this domain
                extraction_context = {
                    "reasoning_type": "analogous",
                    "problem_summary": problem_summary,
                    "key_terms": key_terms,
                    "analogy_title": f"Analogy to {domain}",
                    "object_mappings": "",
                    "shared_relations": ""
                }

        # Call LLM to find solutions in this domain
        solutions, tokens = _find_solutions_in_domain(
            problem_text=problem_summary,
            domain=domain,
            num_solutions=num_solutions_per_domain,
            model=model,
            extraction_context=extraction_context
        )

        all_solutions.extend(solutions)
        total_input_tokens += tokens["input"]
        total_output_tokens += tokens["output"]
        # Capture provider from first successful call
        if provider == "unknown" and "provider" in tokens:
            provider = tokens["provider"]

    # Validate GitHub repos (same as baseline)
    all_solutions, validation_stats = validate_and_filter_github_repos(all_solutions)

    # Aggregate token usage
    total_tokens = {
        "input": total_input_tokens,
        "output": total_output_tokens,
        "provider": provider,
        "github_validation": validation_stats
    }

    return all_solutions, total_tokens
