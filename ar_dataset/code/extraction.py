"""
Analogy extraction module using Claude API.
"""

import os
import time
import json
from pathlib import Path
from typing import List, Dict, Tuple
from anthropic import Anthropic


def extract_analogies(papers: List[Dict], config: dict) -> Tuple[List[Dict], Dict]:
    """Extract analogy details from papers using Claude API.

    Args:
        papers: List of verified papers
        config: Configuration dictionary

    Returns:
        Tuple of (papers_with_analogies, tokens_dict)
    """
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise ValueError("ANTHROPIC_API_KEY environment variable not set")

    model = config.get("apis", {}).get("anthropic", {}).get("extraction_model", "claude-sonnet-4-5")

    # Load prompt template from config
    extraction_prompt = config.get("apis", {}).get("anthropic", {}).get("extraction_prompt", "dataset_creation/prompts/extract_analogy.txt")
    prompt_path = Path(extraction_prompt)

    print(f"  Loading extraction prompt: {extraction_prompt}")
    with open(prompt_path, 'r') as f:
        prompt_template = f.read()

    client = Anthropic(api_key=api_key)

    papers_with_analogies = []
    total_input_tokens = 0
    total_output_tokens = 0
    start_time = time.time()

    # Get rate limit delay from config (default: 0.5 seconds between calls)
    rate_limit_delay = config.get("apis", {}).get("anthropic", {}).get("rate_limit_delay", 0.5)

    total_papers = len(papers)
    print(f"  Processing {total_papers} papers with model: {model}")
    for idx, paper in enumerate(papers, 1):
        # Progress indicator
        title_preview = paper.get('title', 'Unknown')[:60]
        print(f"  [{idx}/{total_papers}] Processing: {title_preview}{'...' if len(paper.get('title', '')) > 60 else ''}")

        try:
            # Prepare prompt
            authors_str = ', '.join(paper.get('authors', []))
            prompt = prompt_template.format(
                title=paper.get('title', 'Unknown'),
                authors=authors_str,
                year=paper.get('year', 'Unknown'),
                abstract=paper.get('abstract', 'No abstract available')
            )

        except Exception as e:
            print(f"    ⚠ Error preparing prompt: {e}")
            import traceback
            traceback.print_exc()
            continue

        # Retry logic for rate limits
        max_retries = 3
        retry_count = 0

        while retry_count < max_retries:
            try:
                # Make API call
                response = client.messages.create(
                    model=model,
                    max_tokens=2000,
                    temperature=0.2,
                    messages=[
                        {"role": "user", "content": prompt}
                    ]
                )

                # Success - break out of retry loop
                break

            except Exception as e:
                error_msg = str(e)

                # Check if it's a rate limit error
                if 'rate_limit' in error_msg.lower() or '429' in error_msg:
                    retry_count += 1
                    if retry_count < max_retries:
                        wait_time = (2 ** retry_count) * rate_limit_delay  # Exponential backoff
                        print(f"    ⚠ Rate limit hit, retrying in {wait_time:.1f}s (attempt {retry_count}/{max_retries})...")
                        time.sleep(wait_time)
                        continue
                    else:
                        print(f"    ✗ Failed after {max_retries} retries: {error_msg}")
                        raise
                else:
                    # Non-rate-limit error, raise immediately
                    raise

        try:

            # Extract response
            content = response.content[0].text

            # Parse JSON response
            analogy_data = _parse_extraction_response(content)

            if analogy_data:
                # Check if paper was rejected for not using analogical reasoning
                problem = analogy_data.get('problem', '')
                if problem.lower().startswith('this paper does not use') and 'analogical reasoning' in problem.lower():
                    print(f"    → Not using analogical reasoning")
                    # Keep the paper but mark it as rejected
                    paper_with_analogy = {
                        **paper,
                        **analogy_data,
                        '_rejected_at_extraction': True
                    }
                else:
                    # Merge with paper data
                    print(f"    ✓ Extracted: {analogy_data.get('base_domain', 'unknown')} → {analogy_data.get('target_domain', 'unknown')}")
                    paper_with_analogy = {**paper, **analogy_data}

                papers_with_analogies.append(paper_with_analogy)

                # Track tokens
                total_input_tokens += response.usage.input_tokens
                total_output_tokens += response.usage.output_tokens

        except Exception as e:
            print(f"    ✗ Failed to extract analogy: {e}")
            import traceback
            traceback.print_exc()
            continue

        # Rate limiting: small delay between API calls (except for last paper)
        if idx < total_papers:
            time.sleep(rate_limit_delay)

    runtime = time.time() - start_time

    tokens = {
        "input": total_input_tokens,
        "output": total_output_tokens,
        "runtime": runtime
    }

    return papers_with_analogies, tokens


def _parse_extraction_response(content: str) -> Dict:
    """Parse extraction response from Claude.

    Args:
        content: Response content from Claude

    Returns:
        Dictionary with extracted analogy fields or empty dict
    """
    try:
        # Try to find JSON in the response
        start_idx = content.find('{')
        end_idx = content.rfind('}')

        if start_idx != -1 and end_idx != -1:
            json_str = content[start_idx:end_idx + 1]

            # Try to parse JSON
            try:
                data = json.loads(json_str)
            except json.JSONDecodeError as e:
                return {}

            # Validate required fields
            # Note: is_original_paper and original_paper_info are optional
            required_fields = [
                'problem',
                'method_name',
                'concrete_example',
                'base_domain',
                'target_domain',
                'base_domain_justification',
                'target_domain_justification',
                'analogy_justification'
            ]
            if all(field in data for field in required_fields):
                # Set defaults for new optional fields if not present
                if 'is_original_paper' not in data:
                    data['is_original_paper'] = True  # Assume original if not specified
                if 'original_paper_info' not in data:
                    data['original_paper_info'] = ""
                return data

    except Exception as e:
        pass

    return {}
