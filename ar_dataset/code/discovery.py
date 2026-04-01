"""
Paper discovery module using Perplexity API.
"""

import os
import time
import json
import requests
from pathlib import Path
from typing import List, Dict
from datetime import datetime


def discover_papers(config: dict, target_count: int = 50, template_path: str = None,
                   base_domain: str = None, target_domain: str = None,
                   max_retries: int = 3) -> tuple[List[Dict], Dict]:
    """Discover papers using Perplexity API with retry logic.

    Args:
        config: Configuration dictionary
        target_count: Number of papers to discover
        template_path: Optional path to template file (default: prompts/discover_papers.txt)
        base_domain: Optional base domain for generic template formatting
        target_domain: Optional target domain for generic template formatting
        max_retries: Maximum number of retry attempts (default: 3)

    Returns:
        Tuple of (papers_list, tokens_dict)
    """
    api_key = os.getenv("PERPLEXITY_API_KEY")
    if not api_key:
        raise ValueError("PERPLEXITY_API_KEY environment variable not set")

    perplexity_config = config.get("apis", {}).get("perplexity", {})
    model = perplexity_config.get("model", "sonar-pro")
    base_max_tokens = perplexity_config.get("max_tokens", 4000)

    # Load prompt template
    if template_path is None:
        # Default template for backward compatibility
        prompt_path = Path(__file__).parent / "prompts" / "discover_papers.txt"
    else:
        # Use provided template path
        prompt_path = Path(template_path)

    with open(prompt_path, 'r') as f:
        query_template = f.read()

    # Format query with parameters
    format_params = {"target_count": target_count}
    if base_domain:
        format_params["base_domain"] = base_domain
    if target_domain:
        format_params["target_domain"] = target_domain

    query = query_template.format(**format_params)

    # Retry loop
    last_error = None
    total_runtime = 0.0

    for attempt in range(max_retries):
        try:
            # Increase max_tokens on retry if previous attempt was truncated
            max_tokens = base_max_tokens
            if attempt > 0:
                max_tokens = int(base_max_tokens * (1.5 ** attempt))
                print(f"  → Retry {attempt + 1}/{max_retries} with increased max_tokens: {max_tokens}")

            # Make API request
            papers, tokens, runtime = _attempt_discovery(
                api_key, model, max_tokens, query, prompt_path,
                base_domain, target_domain
            )

            total_runtime += runtime
            tokens['runtime'] = total_runtime

            # Success - return papers
            if papers:
                if attempt > 0:
                    print(f"  ✓ Retry successful - found {len(papers)} papers")
                return papers, tokens

            # No papers found - retry
            print(f"  ⚠ No papers found on attempt {attempt + 1}/{max_retries}")
            last_error = "No papers found in response"

        except requests.exceptions.RequestException as e:
            # Network/API error - retry with exponential backoff
            total_runtime += time.time()
            last_error = f"API request failed: {e}"
            print(f"  ⚠ Attempt {attempt + 1}/{max_retries} failed: {last_error}")

            if attempt < max_retries - 1:
                backoff_time = 2 ** attempt  # 1s, 2s, 4s
                print(f"  → Waiting {backoff_time}s before retry...")
                time.sleep(backoff_time)

        except Exception as e:
            # Parsing or other error - retry
            total_runtime += time.time()
            last_error = f"Discovery error: {e}"
            print(f"  ⚠ Attempt {attempt + 1}/{max_retries} failed: {last_error}")

    # All retries exhausted
    error_msg = f"Failed to discover papers after {max_retries} attempts. Last error: {last_error}"
    raise RuntimeError(error_msg)


def _attempt_discovery(api_key: str, model: str, max_tokens: int, query: str,
                      prompt_path: Path, base_domain: str = None,
                      target_domain: str = None) -> tuple[List[Dict], Dict, float]:
    """Single discovery attempt.

    Args:
        api_key: API key
        model: Model name
        max_tokens: Max tokens for response
        query: Query text
        prompt_path: Path to prompt file
        base_domain: Optional base domain
        target_domain: Optional target domain

    Returns:
        Tuple of (papers_list, tokens_dict, runtime)
    """
    url = "https://api.perplexity.ai/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": "You are a research assistant specializing in identifying papers that use analogical reasoning for creative problem-solving."
            },
            {
                "role": "user",
                "content": query
            }
        ],
        "max_tokens": max_tokens
    }

    start_time = time.time()

    response = requests.post(url, json=payload, headers=headers)
    response.raise_for_status()
    result = response.json()

    runtime = time.time() - start_time

    # Extract response
    content = result['choices'][0]['message']['content']

    # Check for token limit issues
    finish_reason = result['choices'][0].get('finish_reason', 'unknown')
    usage = result.get('usage', {})
    output_tokens = usage.get('completion_tokens', 0)

    # Log token limit warnings
    if finish_reason == 'length':
        print(f"  ⚠ WARNING: Response was truncated due to token limit!")
        print(f"  Output tokens: {output_tokens}/{max_tokens} (hit max_tokens limit)")
    elif output_tokens >= max_tokens * 0.95:
        print(f"  ⚠ WARNING: Response is near token limit!")
        print(f"  Output tokens: {output_tokens}/{max_tokens} ({output_tokens/max_tokens*100:.1f}%)")

    # Log finish reason if not normal
    if finish_reason not in ['stop', 'end_turn']:
        print(f"  ⚠ Finish reason: {finish_reason}")

    # Log response length
    print(f"  Response length: {len(content):,} characters")

    # Try to parse JSON from response
    papers = _parse_papers_from_response(content)

    if not papers:
        error_msg = f"Failed to parse any papers from response"
        print(f"\n  ⚠ ERROR: {error_msg}")
        print(f"  First 500 chars: {content[:500]}...")
        print(f"  Last 500 chars: ...{content[-500:]}")
        raise RuntimeError(error_msg)

    # Check if response was truncated and array might be incomplete
    if finish_reason == 'length':
        # Estimate if we got significantly fewer papers than requested
        # This suggests the response was cut off before completion
        print(f"  ⚠ Response truncated - consider retrying with higher max_tokens")
        raise RuntimeError(f"Response truncated (finish_reason: length), retry needed")

    # Extract token usage
    tokens = {
        "input": usage.get('prompt_tokens', 0),
        "output": usage.get('completion_tokens', 0),
        "runtime": runtime,
        "finish_reason": finish_reason
    }

    # Add discovered_at timestamp and template name to each paper
    # Use domain pair name if both domains provided, otherwise use template file name
    if base_domain and target_domain:
        template_name = f"{base_domain}_to_{target_domain}"
    else:
        template_name = prompt_path.stem

    for paper in papers:
        paper['discovered_at'] = datetime.now().isoformat()
        paper['discovered_by_template'] = template_name
        if base_domain:
            paper['base_domain'] = base_domain
        if target_domain:
            paper['target_domain'] = target_domain

    return papers, tokens, runtime


def _parse_papers_from_response(content: str) -> List[Dict]:
    """Parse paper information from Perplexity response.

    Args:
        content: Response content from Perplexity

    Returns:
        List of paper dictionaries
    """
    papers = []

    # Try to find JSON array in the response
    try:
        # Look for JSON array in the content
        start_idx = content.find('[')
        end_idx = content.rfind(']')

        if start_idx != -1 and end_idx != -1:
            json_str = content[start_idx:end_idx + 1]
            papers_data = json.loads(json_str)

            # Normalize field names
            for item in papers_data:
                paper = {}
                # Handle various possible field names
                paper['title'] = item.get('title') or item.get('paper_title') or item.get('name', 'Unknown')
                paper['url'] = item.get('url') or item.get('doi') or item.get('source') or item.get('link', '')
                paper['analogy_description'] = item.get('analogy_description') or item.get('description') or item.get('analogy', '')

                papers.append(paper)

        return papers

    except json.JSONDecodeError as e:
        # If JSON parsing fails, try to extract structured information manually
        print(f"  ⚠ JSON parsing error: {str(e)}")
        print(f"  Error at position {e.pos}: ...{content[max(0,e.pos-50):e.pos+50]}...")

        # Try to parse individual paper objects from the malformed JSON
        papers = _parse_individual_papers(content)
        if papers:
            print(f"  → Recovered {len(papers)} papers via individual parsing")
            return papers

        # Last resort: fallback parser for non-JSON format
        papers = _fallback_parse_papers(content)
        if papers:
            print(f"  → Recovered {len(papers)} papers via fallback parsing")
            return papers

        # If all parsing methods fail, raise an error
        raise RuntimeError(f"All parsing methods failed. JSON error: {str(e)}")


def _parse_individual_papers(content: str) -> List[Dict]:
    """Try to parse individual paper objects from malformed JSON.

    Args:
        content: Response content with potentially malformed JSON

    Returns:
        List of successfully parsed paper dictionaries
    """
    import re
    papers = []

    # Find all occurrences of paper-like objects: {"title": ..., "url": ..., "analogy_description": ...}
    # Use regex to find individual paper objects
    pattern = r'\{\s*"title"\s*:\s*"[^"]*"[^}]*\}'

    matches = re.finditer(pattern, content, re.DOTALL)

    for match in matches:
        try:
            paper_str = match.group(0)
            paper_obj = json.loads(paper_str)

            # Normalize fields
            paper = {}
            paper['title'] = paper_obj.get('title', 'Unknown')
            paper['url'] = paper_obj.get('url') or paper_obj.get('doi') or paper_obj.get('link', '')
            paper['analogy_description'] = paper_obj.get('analogy_description') or paper_obj.get('description', '')

            if paper['title'] and paper['url']:
                papers.append(paper)
        except:
            # Skip this malformed paper, continue to next
            continue

    return papers


def _fallback_parse_papers(content: str) -> List[Dict]:
    """Fallback parser for unstructured content.

    Args:
        content: Response content from Perplexity

    Returns:
        List of paper dictionaries
    """
    papers = []
    lines = content.split('\n')

    current_paper = {}
    for line in lines:
        line = line.strip()

        if not line:
            if current_paper.get('title') and current_paper.get('url'):
                papers.append(current_paper)
                current_paper = {}
            continue

        # Look for title patterns
        if line.startswith(('1.', '2.', '3.', '4.', '5.', '6.', '7.', '8.', '9.', '-', '*')) or 'Title:' in line:
            if 'Title:' in line:
                title = line.split('Title:', 1)[1].strip()
            else:
                title = line.split('.', 1)[1].strip() if '.' in line else line.strip('- *')
            current_paper['title'] = title

        # Look for URL/DOI
        elif ('doi.org' in line or 'http' in line or 'DOI:' in line or 'URL:' in line):
            url = line.split(':', 1)[1].strip() if ':' in line else line.strip()
            current_paper['url'] = url

        # Look for description
        elif ('Analogy:' in line or 'Description:' in line):
            desc = line.split(':', 1)[1].strip()
            current_paper['analogy_description'] = desc

    # Add last paper if exists
    if current_paper.get('title') and current_paper.get('url'):
        papers.append(current_paper)

    return papers
