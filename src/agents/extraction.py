"""Problem Extraction Agent - creates multiple abstraction levels."""

import json
import os
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import get_config
from llm_client import call_llm


def extract_problem(problem_text: str, verbose: bool = True) -> dict:
    """
    Extract problem aspects and create multiple abstraction levels.

    Args:
        problem_text: The biomedical problem description
        verbose: Whether to print progress messages (set False for parallel execution)

    Returns:
        Dict with problem_summary, abstraction_levels, key_terms, target_domains
    """
    # Get domain count from config (use search.num_domains_to_search)
    num_domains = get_config("search.num_domains_to_search", 3)

    # Get key terms count from config
    num_key_terms = get_config("extraction.num_key_terms", [3, 5])
    if isinstance(num_key_terms, list):
        min_key_terms, max_key_terms = num_key_terms
    else:
        min_key_terms = max_key_terms = num_key_terms

    # Check for domain override to select appropriate prompt
    override_domains = get_config("extraction.override_domains", None)

    if override_domains and len(override_domains) > 0:
        # Use constrained prompt with pre-specified domains
        reasoning_type = get_config("extraction.reasoning_type", "hierarchical")
        if reasoning_type == "hierarchical":
            prompt_path = Path("prompts") / "hierarchical_extraction_constrained.txt"
        else:
            prompt_path = Path("prompts") / "analogous_extraction_constrained.txt"
        domains_to_use = override_domains[:num_domains]
        domain_list_str = ", ".join(domains_to_use)
        if verbose:
            print(f"  Using constrained prompt with override domains: {domain_list_str}")

        with open(prompt_path, "r") as f:
            prompt_template = f.read()

        # Format prompt with domain list
        prompt = prompt_template.format(
            problem_text=problem_text,
            domain_list=domain_list_str,
            num_domains=len(domains_to_use),
            min_key_terms=min_key_terms,
            max_key_terms=max_key_terms
        )
    else:
        # Use normal prompt selection logic
        custom_prompt = get_config("extraction.prompt_file", None)
        if custom_prompt:
            prompt_path = Path(custom_prompt)
        else:
            reasoning_type = get_config("extraction.reasoning_type", "hierarchical")
            use_diverse = get_config("extraction.use_diverse_prompts", False)

            # Build prompt filename based on reasoning_type
            if reasoning_type == "hierarchical":
                prompt_filename = "hierarchical_extraction_diverse.txt" if use_diverse else "hierarchical_extraction.txt"
            elif reasoning_type == "analogous":
                prompt_filename = "analogous_extraction_diverse.txt" if use_diverse else "analogous_extraction.txt"
            else:
                raise ValueError(f"Unknown reasoning_type: {reasoning_type}. Must be 'hierarchical' or 'analogous'.")

            prompt_path = Path("prompts") / prompt_filename

        if verbose:
            print(f"  Using prompt: {prompt_path} (reasoning_type: {get_config('extraction.reasoning_type', 'hierarchical')})")
        with open(prompt_path, "r") as f:
            prompt_template = f.read()

        # Fill in the problem text, domain count, and key terms count
        prompt = prompt_template.format(
            problem_text=problem_text,
            num_domains=num_domains,
            min_key_terms=min_key_terms,
            max_key_terms=max_key_terms
        )

    # Call LLM API
    if verbose:
        print("Calling LLM API for problem extraction...")
    model = get_config("model.extraction_model") or get_config("model.name", "claude-sonnet-4-20250514")
    temperature = get_config("extraction.temperature", 1.0)

    response = call_llm(
        model=model,
        messages=[{
            "role": "user",
            "content": prompt
        }],
        max_tokens=get_config("extraction.max_tokens", 2000),
        temperature=temperature
    )

    # Extract text and token usage from response
    response_text = response["content"].strip()
    tokens = {
        "input": response["usage"]["input_tokens"],
        "output": response["usage"]["output_tokens"],
        "provider": response["provider"]
    }

    # Parse JSON (remove code fences if present)
    if response_text.startswith("```"):
        # Remove code fence markers (```json or ```)
        lines = response_text.split("\n")
        # Skip first line (```) and last line (```)
        response_text = "\n".join(lines[1:-1])

    # Also handle inline code fence language specifiers
    if response_text.startswith("json"):
        response_text = response_text[4:].strip()

    try:
        extraction = json.loads(response_text)
    except json.JSONDecodeError as e:
        if verbose:
            print(f"Error parsing JSON response: {e}")
            print(f"Response: {response_text}")
        raise

    # Validate required fields based on reasoning type
    reasoning_type = get_config("extraction.reasoning_type", "hierarchical")
    if reasoning_type == "hierarchical":
        required = ["problem_summary", "abstraction_levels", "key_terms", "target_domains"]
    elif reasoning_type == "analogous":
        required = ["problem_summary", "problem_objects", "problem_relations", "analogies", "key_terms", "target_domains"]
    else:
        required = ["problem_summary", "key_terms", "target_domains"]

    for field in required:
        if field not in extraction:
            raise ValueError(f"Missing required field: {field}")

    # Additional validation for analogous mode
    if reasoning_type == "analogous":
        for analogy in extraction.get('analogies', []):
            if 'object_mappings' not in analogy or len(analogy['object_mappings']) == 0:
                raise ValueError(f"Analogy '{analogy.get('target_domain')}' missing object_mappings")

    return extraction, tokens


def print_extraction(extraction: dict, selected_abstraction: str = None):
    """Pretty print extraction results."""
    print("\n" + "="*60)
    print("PROBLEM EXTRACTION")
    print("="*60)

    print(f"\nSummary: {extraction['problem_summary']}")

    reasoning_type = get_config("extraction.reasoning_type", "hierarchical")

    if reasoning_type == "analogous":
        # Print problem objects
        print("\nProblem Objects:")
        for obj in extraction.get('problem_objects', []):
            print(f"  - {obj['name']}: {obj['role']}")

        # Print problem relations
        print("\nCore Relations:")
        for rel in extraction.get('problem_relations', []):
            print(f"  - {rel}")

        # Print analogies
        print("\nAnalogies:")
        for analogy in extraction.get('analogies', []):
            print(f"\n  [{analogy['target_domain'].upper()}] {analogy['analogy_title']}")
            print("    Object Mappings:")
            for m in analogy['object_mappings']:
                print(f"      {m['source']} -> {m['target']} ({m['mapping_rationale']})")
            print(f"    Shared Relations: {analogy['shared_relations']}")
    else:
        # Hierarchical mode: print abstraction levels
        if selected_abstraction:
            print(f"\nSelected Abstraction Level ({selected_abstraction}):")
            for level in extraction['abstraction_levels']:
                if level['level'].lower() == selected_abstraction.lower():
                    print(f"  {level['description']}")
                    break
        else:
            print("\nAbstraction Levels:")
            for level in extraction['abstraction_levels']:
                print(f"  [{level['level'].upper()}] {level['description']}")

    print(f"\nKey Terms: {', '.join(extraction['key_terms'])}")
    print(f"Target Domains: {', '.join(extraction['target_domains'])}")
    print("="*60 + "\n")
