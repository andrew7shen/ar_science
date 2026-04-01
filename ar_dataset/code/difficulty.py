"""
Difficulty assessment module using Claude API.
"""

import os
import time
import json
from pathlib import Path
from typing import List, Dict, Tuple
from anthropic import Anthropic


def assess_difficulty(papers: List[Dict], config: dict) -> Tuple[List[Dict], Dict]:
    """Assess difficulty of analogical reasoning for papers.

    Args:
        papers: List of papers with extracted analogies
        config: Configuration dictionary

    Returns:
        Tuple of (papers_with_difficulty, tokens_dict)
    """
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise ValueError("ANTHROPIC_API_KEY environment variable not set")

    model = config.get("apis", {}).get("anthropic", {}).get("assessment_model", "claude-sonnet-4-5")

    # Load prompt template from config
    difficulty_prompt = config.get("apis", {}).get("anthropic", {}).get("difficulty_prompt", "dataset_creation/prompts/assess_difficulty.txt")
    prompt_path = Path(difficulty_prompt)

    print(f"  Loading difficulty prompt: {difficulty_prompt}")
    with open(prompt_path, 'r') as f:
        prompt_template = f.read()

    client = Anthropic(api_key=api_key)

    papers_with_difficulty = []
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
            # Skip assessment for papers rejected during extraction
            if paper.get('_rejected_at_extraction', False):
                print(f"    ⊘ Skipped - rejected during extraction")
                # Keep the paper with default difficulty fields
                paper_with_difficulty = {
                    **paper,
                    "difficulty": "n/a",
                    "difficulty_reasoning": "Not assessed - paper does not use cross-domain analogical reasoning"
                }
                papers_with_difficulty.append(paper_with_difficulty)
                continue

            # Prepare prompt
            prompt = prompt_template.format(
                title=paper.get('title', 'Unknown'),
                base_domain=paper.get('base_domain', 'Unknown'),
                target_domain=paper.get('target_domain', 'Unknown'),
                justification=paper.get('analogy_justification', 'No justification')
            )

            # Retry logic for rate limits
            max_retries = 3
            retry_count = 0

            while retry_count < max_retries:
                try:
                    # Make API call
                    response = client.messages.create(
                        model=model,
                        max_tokens=1000,
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

            # Extract response
            content = response.content[0].text

            # Parse JSON response
            difficulty_data = _parse_difficulty_response(content)

            if difficulty_data:
                # Merge with paper data
                diff_level = difficulty_data.get('difficulty', 'medium')
                print(f"    ✓ Difficulty: {diff_level}")
                paper_with_difficulty = {
                    **paper,
                    "difficulty": diff_level,
                    "difficulty_reasoning": difficulty_data.get('reasoning', '')
                }
                papers_with_difficulty.append(paper_with_difficulty)

                # Track tokens
                total_input_tokens += response.usage.input_tokens
                total_output_tokens += response.usage.output_tokens

        except Exception as e:
            print(f"    ✗ Failed to assess difficulty: {e}")
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

    return papers_with_difficulty, tokens


def _parse_difficulty_response(content: str) -> Dict:
    """Parse difficulty assessment response from Claude.

    Args:
        content: Response content from Claude

    Returns:
        Dictionary with difficulty and reasoning or empty dict
    """
    try:
        # Try to find JSON in the response
        start_idx = content.find('{')
        end_idx = content.rfind('}')

        if start_idx != -1 and end_idx != -1:
            json_str = content[start_idx:end_idx + 1]
            data = json.loads(json_str)

            # Validate required fields
            if 'difficulty' in data:
                # Normalize difficulty value
                difficulty = data['difficulty'].lower()
                if difficulty not in ['easy', 'medium', 'hard']:
                    difficulty = 'medium'

                return {
                    'difficulty': difficulty,
                    'reasoning': data.get('reasoning', '')
                }

    except json.JSONDecodeError:
        pass

    return {}
