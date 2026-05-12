"""Assessment Agent - ranks and scores cross-domain solutions."""

import json
import os
import time
from pathlib import Path
from anthropic import Anthropic
import requests
from datetime import datetime, timedelta
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import get_config
from agents.academic_apis import (
    query_semantic_scholar,
    query_semantic_scholar_snippet,
    simplify_search_terms,
    get_specter_embedding,
    fetch_paper_embeddings_batch,
    cosine_similarity
)
import re


def extract_github_urls(text: str) -> list[str]:
    """Extract GitHub repository URLs from text.

    Args:
        text: Text containing GitHub URLs

    Returns:
        List of GitHub repo URLs (e.g., https://github.com/owner/repo)
    """
    # Match github.com URLs, extract base repo URL
    pattern = r'https?://github\.com/([a-zA-Z0-9_-]+/[a-zA-Z0-9_.-]+?)(?:/|(?:\s|$|[^\w/-]))'
    matches = re.findall(pattern, text)

    # Deduplicate and return full URLs
    unique_repos = list(set(matches))
    return [f"https://github.com/{repo}" for repo in unique_repos]


def find_github_repos_with_claude(solution: dict, client, model: str) -> tuple[list[str], dict]:
    """Use Claude with web search to find GitHub repos for a solution.

    Args:
        solution: Solution dict with title and other metadata
        client: Anthropic client
        model: Model name to use

    Returns:
        Tuple of (List of GitHub repository URLs, token usage dict)
    """
    title = solution.get('title', '')
    domain = solution.get('source_domain', '')

    # Craft focused prompt for Claude
    prompt = f"""Find 3 high-quality GitHub repositories that implement "{title}" from {domain}.

Requirements:
- Python implementations preferred
- Well-maintained (active, good documentation)
- Return ONLY the GitHub repository URLs, one per line
- Example format: https://github.com/owner/repo

Focus on finding actual implementations, not tutorials or papers."""

    tokens = {"input": 0, "output": 0}

    try:
        response = client.messages.create(
            model=model,
            max_tokens=1000,
            tools=[{
                "type": "web_search_20250305",
                "name": "web_search",
                "max_uses": 1
            }],
            messages=[{
                "role": "user",
                "content": prompt
            }]
        )

        tokens["input"] += response.usage.input_tokens
        tokens["output"] += response.usage.output_tokens

        # Handle tool use loop - remove tools so Claude can only respond with text
        messages = [{"role": "user", "content": prompt}]
        if response.stop_reason == "tool_use":
            messages.append({"role": "assistant", "content": response.content})
            # Don't provide tools again - force Claude to give final text response
            response = client.messages.create(
                model=model,
                max_tokens=1000,
                messages=messages
            )
            tokens["input"] += response.usage.input_tokens
            tokens["output"] += response.usage.output_tokens

        # Extract text response
        response_text = ""
        for block in response.content:
            if block.type == "text":
                response_text = block.text
                break

        # Extract GitHub URLs from response
        urls = extract_github_urls(response_text)
        return urls[:3], tokens  # Limit to top 3

    except Exception as e:
        print(f"      Warning: Claude web search failed: {e}")
        return [], tokens


def fetch_repo_details(repo_url: str, headers: dict) -> dict:
    """Fetch detailed information for a specific GitHub repository.

    Args:
        repo_url: GitHub repository URL (e.g., https://github.com/owner/repo)
        headers: HTTP headers for GitHub API

    Returns:
        Dict with repo metadata, or empty dict on failure
    """
    try:
        # Extract owner/repo from URL
        match = re.search(r'github\.com/([^/]+)/([^/]+)', repo_url)
        if not match:
            return {}

        owner, repo_name = match.groups()
        full_name = f"{owner}/{repo_name}"

        # Fetch repo details
        url = f"https://api.github.com/repos/{full_name}"
        response = requests.get(url, headers=headers, timeout=10)

        if response.status_code == 200:
            repo = response.json()
            return {
                "url": repo['html_url'],
                "full_name": repo['full_name'],
                "stars": repo['stargazers_count'],
                "description": repo.get('description', ''),
                "language": repo.get('language', ''),
                "updated_at": repo['updated_at'],
                "topics": repo.get('topics', []),
                "license": repo.get('license', {}).get('key', 'none') if repo.get('license') else 'none'
            }
        else:
            return {}

    except Exception:
        return {}


def fetch_readme_content(repo_full_name: str, headers: dict) -> str:
    """Fetch README content from a GitHub repository.

    Args:
        repo_full_name: Repository name in format "owner/repo"
        headers: HTTP headers for GitHub API

    Returns:
        README content as string, or empty string if not found
    """
    try:
        url = f"https://api.github.com/repos/{repo_full_name}/readme"
        response = requests.get(url, headers=headers, timeout=10)

        if response.status_code == 200:
            readme_data = response.json()
            # GitHub returns base64 encoded content
            import base64
            content = base64.b64decode(readme_data['content']).decode('utf-8', errors='ignore')
            # Truncate to first 1500 characters to avoid token overload
            return content[:1500] if len(content) > 1500 else content
        else:
            return ""
    except Exception:
        return ""


def extract_keywords_from_solution(solution: dict) -> str:
    """Extract key search terms from a solution using title.

    Args:
        solution: Solution dict with title

    Returns:
        Space-separated keywords for GitHub search (2-3 core terms)
    """
    title = solution.get('title', '')

    # Check for abbreviation in parentheses like (SVM), (RLDA)
    abbrev_match = re.search(r'\(([A-Z]{2,})\)', title)

    # Common filler words to skip
    skip_words = {'with', 'for', 'using', 'based', 'on', 'and', 'the', 'a', 'an',
                  'in', 'of', 'to', 'from', 'by', 'as', 'at'}

    keywords = []

    if abbrev_match:
        # Use abbreviation
        keywords.append(abbrev_match.group(1))
        # Add 1-2 more words from title
        title_words = [w for w in title.split() if w.lower() not in skip_words and not w.startswith('(')][:2]
        keywords.extend(title_words)
    else:
        # Take first 2-3 significant words from title only
        title_words = [w for w in title.split() if w.lower() not in skip_words][:3]
        keywords.extend(title_words)

    # Return just 2-3 keywords to keep search broad
    return ' '.join(keywords[:3])


def fetch_github_repos_for_solutions(solutions: list, client, model: str) -> tuple[dict, int, dict]:
    """Fetch GitHub repos using configurable approach (web search or GitHub API).

    Args:
        solutions: List of solution dicts with 'title' and optionally 'key_concepts'
        client: Anthropic client for web search (if enabled)
        model: Model name to use for web search (if enabled)

    Returns:
        Tuple of (Dict mapping solution index to list of repo dicts, GitHub API requests, token usage)
    """
    use_web_search = get_config("assessment.github.use_web_search", False)
    fetch_readmes = get_config("assessment.github.fetch_readmes", True)

    github_token = os.getenv("GITHUB_TOKEN")  # Optional, for higher rate limits
    headers = {"Accept": "application/vnd.github+json"}
    if github_token:
        headers["Authorization"] = f"Bearer {github_token}"

    repos_by_solution = {}
    github_api_requests = 0  # Track GitHub API calls only
    web_search_tokens = {"input": 0, "output": 0}  # Track web search tokens
    max_repos = get_config("assessment.github.repos_per_solution", 3)

    for idx, solution in enumerate(solutions):
        try:
            title = solution.get('title', '')

            if use_web_search:
                # Method 1: Use Claude web search to find repos
                print(f"    Using Claude web search to find repos for: '{title[:60]}...'")
                repo_urls, search_tokens = find_github_repos_with_claude(solution, client, model)
                web_search_tokens["input"] += search_tokens["input"]
                web_search_tokens["output"] += search_tokens["output"]
                print(f"      Found {len(repo_urls)} repo URLs via web search")

                # Fetch details for each URL
                repos = []
                for url in repo_urls:
                    repo_data = fetch_repo_details(url, headers)
                    if repo_data:
                        github_api_requests += 1
                        repos.append(repo_data)
                        print(f"        ✓ {repo_data['full_name']} ({repo_data['stars']:,} stars)")
            else:
                # Method 2: Use GitHub API search directly with keywords
                keywords = extract_keywords_from_solution(solution)
                query = f"{keywords} language:python stars:>5"
                print(f"    GitHub API search for: '{keywords}'")

                url = "https://api.github.com/search/repositories"
                params = {"q": query, "per_page": max_repos, "sort": "stars", "order": "desc"}
                response = requests.get(url, headers=headers, params=params, timeout=10)
                github_api_requests += 1

                if response.status_code == 200:
                    results = response.json()
                    total_count = results.get('total_count', 0)
                    print(f"      Found {total_count} total repos, fetching top {max_repos}")

                    repos = []
                    for repo in results.get('items', [])[:max_repos]:
                        repo_data = {
                            "url": repo['html_url'],
                            "full_name": repo['full_name'],
                            "stars": repo['stargazers_count'],
                            "description": repo.get('description', ''),
                            "language": repo.get('language', ''),
                            "updated_at": repo['updated_at'],
                            "topics": repo.get('topics', []),
                            "license": repo.get('license', {}).get('key', 'none') if repo.get('license') else 'none'
                        }

                        # Fetch README if enabled
                        if fetch_readmes:
                            readme = fetch_readme_content(repo_data['full_name'], headers)
                            github_api_requests += 1
                            if readme:
                                repo_data['readme'] = readme

                        repos.append(repo_data)
                        print(f"        ✓ {repo_data['full_name']} ({repo_data['stars']:,} stars)")
                elif response.status_code in [403, 429]:
                    reset_time = response.headers.get('X-RateLimit-Reset', 'unknown')
                    remaining = response.headers.get('X-RateLimit-Remaining', 'unknown')
                    print(f"  ⚠️  RATE LIMIT: GitHub API rate limit exceeded for '{title}'")
                    print(f"      Status: {response.status_code} | Remaining: {remaining} | Reset time: {reset_time}")
                    if not github_token:
                        print(f"      💡 Tip: Set GITHUB_TOKEN env variable for 30 req/min (currently 10 req/min)")
                    repos = []
                else:
                    print(f"  Warning: GitHub search failed for '{title}' (status {response.status_code})")
                    repos = []

            repos_by_solution[idx] = repos

        except Exception as e:
            print(f"  Warning: Error fetching repos for '{solution.get('title', 'unknown')}': {e}")
            repos_by_solution[idx] = []

    return repos_by_solution, github_api_requests, web_search_tokens


def _format_embedded_github_repos(solutions: list) -> str:
    """Format GitHub repos that are already embedded in solutions (from Perplexity).

    Args:
        solutions: List of solution dicts with embedded github_repos field

    Returns:
        Formatted string with GitHub repo information
    """
    github_text = "\n**GitHub Repositories Found:**\n\n"

    for idx, solution in enumerate(solutions):
        repos = solution.get('github_repos', [])
        github_text += f"Solution {idx + 1}: {solution.get('title', 'Unknown')}\n"

        if repos:
            for i, repo in enumerate(repos, 1):
                github_text += f"  Repo {i}:\n"
                github_text += f"    - URL: {repo.get('url', 'N/A')}\n"
                github_text += f"    - Stars: {repo.get('stars', 'N/A')}\n"
                github_text += f"    - Language: {repo.get('language', 'N/A')}\n"
                if 'description' in repo and repo['description']:
                    github_text += f"    - Description: {repo['description']}\n"
                if 'maintenance_status' in repo:
                    github_text += f"    - Status: {repo['maintenance_status']}\n"
                if 'last_updated' in repo:
                    github_text += f"    - Last Updated: {repo['last_updated']}\n"
                github_text += "\n"
        else:
            github_text += "  No GitHub repositories found.\n\n"

    return github_text


def format_github_repos_for_prompt(solutions: list, repos_by_solution: dict) -> str:
    """Format GitHub repo data as structured text for Claude.

    Args:
        solutions: List of solution dicts
        repos_by_solution: Dict mapping solution index to list of repo dicts

    Returns:
        Formatted string with GitHub repo information
    """
    github_text = "\n**GitHub Repositories Found:**\n\n"

    for idx, solution in enumerate(solutions):
        repos = repos_by_solution.get(idx, [])
        github_text += f"Solution {idx + 1}: {solution.get('title', 'Unknown')}\n"

        if repos:
            for i, repo in enumerate(repos, 1):
                # Get raw star count and recency info - let Claude judge quality
                stars = repo.get('stars', 0)

                # Handle both 'updated_at' (legacy) and 'last_updated' (workflow B) fields
                age_days = None
                if repo.get('updated_at'):
                    try:
                        updated = datetime.fromisoformat(repo['updated_at'].replace('Z', '+00:00'))
                        age_days = (datetime.now(updated.tzinfo) - updated).days
                    except (ValueError, TypeError):
                        pass
                elif repo.get('last_updated'):
                    # last_updated might already be formatted as "X days ago"
                    age_days = repo['last_updated']

                github_text += f"  Repo {i}:\n"
                github_text += f"    - URL: {repo.get('url', 'N/A')}\n"
                github_text += f"    - Stars: {stars:,}\n"
                github_text += f"    - Language: {repo.get('language', 'N/A')}\n"
                if age_days is not None:
                    if isinstance(age_days, int):
                        github_text += f"    - Last Updated: {age_days} days ago\n"
                    else:
                        github_text += f"    - Last Updated: {age_days}\n"
                github_text += f"    - License: {repo.get('license', 'N/A')}\n"
                if repo.get('description'):
                    github_text += f"    - Description: {repo['description']}\n"

                # Include README content if available (for conceptual understanding)
                if 'readme' in repo and repo.get('readme'):
                    github_text += f"    - README excerpt:\n"
                    readme_lines = repo['readme'].split('\n')[:20]  # First 20 lines
                    for line in readme_lines:
                        github_text += f"      {line}\n"

                github_text += "\n"
        else:
            github_text += "  No GitHub repositories found.\n\n"

    return github_text


def _filter_solution_for_assessment(solution: dict) -> dict:
    """Filter solution data to keep only fields needed for assessment.

    Removes large fields like paper abstracts and truncates READMEs to reduce token count.

    Args:
        solution: Full solution dict with papers and github_repos

    Returns:
        Filtered solution dict with only essential fields
    """
    # Convert sources/source_titles to papers format if not already present
    # This handles baseline and Workflow A solutions
    if 'papers' not in solution and 'sources' in solution and 'source_titles' in solution:
        sources = solution.get('sources', [])
        titles = solution.get('source_titles', [])

        # Create papers array from sources and titles
        papers = []
        for i in range(max(len(sources), len(titles))):
            paper = {
                'title': titles[i] if i < len(titles) else 'Unknown',
                'url': sources[i] if i < len(sources) else ''
            }
            papers.append(paper)

        solution['papers'] = papers

    filtered = {
        'title': solution.get('title', ''),
        'source_domain': solution.get('source_domain', ''),
        'description': solution.get('description', ''),
        'key_concepts': solution.get('key_concepts', [])
    }

    # Filter papers: keep title, url, year, abstract (truncated) for Claude's relevance assessment
    # Remove pre-computed relevance_score - let Claude evaluate from content
    if 'papers' in solution and isinstance(solution.get('papers'), list):
        # New format: papers is list of dicts
        filtered['papers'] = []
        for paper in solution['papers']:
            if not isinstance(paper, dict):
                continue
            filtered_paper = {
                'title': paper.get('title', ''),
                'url': paper.get('url', ''),
                'year': paper.get('year')
            }

            # Include truncated abstract for relevance assessment (max 500 chars)
            if 'abstract' in paper and paper.get('abstract'):
                filtered_paper['abstract'] = paper['abstract'][:500]

            # Filter repos within each paper
            if 'github_repos' in paper and isinstance(paper.get('github_repos'), list):
                filtered_paper['github_repos'] = []
                for repo in paper['github_repos']:
                    if not isinstance(repo, dict):
                        continue
                    filtered_repo = {
                        'url': repo.get('url', ''),
                        'stars': repo.get('stars', 0),
                        'language': repo.get('language', ''),
                        'description': repo.get('description', ''),
                        'last_updated': repo.get('last_updated', ''),
                        'license': repo.get('license', '')
                    }
                    # Truncate README to max 500 chars for relevance assessment
                    if 'readme' in repo and repo.get('readme'):
                        filtered_repo['readme'] = repo['readme'][:500]
                    filtered_paper['github_repos'].append(filtered_repo)

            filtered['papers'].append(filtered_paper)

    # Filter top-level github_repos (for backwards compatibility with Workflow A)
    if 'github_repos' in solution and isinstance(solution.get('github_repos'), list):
        filtered['github_repos'] = []
        for repo in solution['github_repos']:
            if not isinstance(repo, dict):
                continue
            filtered_repo = {
                'url': repo.get('url', ''),
                'stars': repo.get('stars', 0),
                'language': repo.get('language', ''),
                'description': repo.get('description', ''),
                'last_updated': repo.get('last_updated', ''),
                'license': repo.get('license', ''),
                'maintenance_status': repo.get('maintenance_status', '')
            }
            # Truncate README to max 500 chars
            if 'readme' in repo and repo.get('readme'):
                filtered_repo['readme'] = repo['readme'][:500]
            filtered['github_repos'].append(filtered_repo)

    return filtered


def generate_novelty_queries(solution: dict, problem_summary: str, client, num_queries: int = 1) -> tuple[list[str], dict]:
    """Use Haiku to generate Semantic Scholar search queries.

    Args:
        solution: Solution dict with title, key_concepts, description
        problem_summary: The problem being solved
        client: Anthropic client
        num_queries: Number of queries to generate (1 or 3)

    Returns:
        Tuple of (list of query strings, tokens dict)
    """
    # Load prompt template
    prompt_file = "novelty_query_comprehensive.txt" if num_queries > 1 else "novelty_query.txt"
    prompt_path = Path("prompts") / prompt_file
    with open(prompt_path, "r") as f:
        prompt_template = f.read()

    # Format key concepts
    key_concepts = solution.get('key_concepts', [])
    if isinstance(key_concepts, list):
        key_concepts_str = ', '.join(key_concepts)
    else:
        key_concepts_str = str(key_concepts)

    # Fill in template
    prompt = prompt_template.format(
        key_concepts=key_concepts_str,
        problem_summary=problem_summary
    )

    response = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=100 if num_queries > 1 else 50,
        temperature=0,
        messages=[{"role": "user", "content": prompt}]
    )

    text = response.content[0].text.strip()
    tokens = {
        "input": response.usage.input_tokens,
        "output": response.usage.output_tokens
    }

    # Parse response into list of queries
    if num_queries > 1:
        queries = [q.strip() for q in text.split('\n') if q.strip()][:num_queries]
    else:
        # Single query mode - take first line only
        queries = [text.split('\n')[0].strip()]

    return queries, tokens


def retrieve_papers_comprehensive(solution: dict, problem_summary: str, client, papers_per_query: int = 10, top_k_papers: int = 10) -> tuple[list[dict], dict, dict]:
    """Multi-strategy paper retrieval with embedding reranking.

    Args:
        solution: Solution dict with title, description, key_concepts
        problem_summary: The biomedical problem being solved
        client: Anthropic client for query generation
        papers_per_query: Number of papers to fetch from Semantic Scholar per query
        top_k_papers: Maximum papers to return after reranking

    Returns:
        Tuple of (list of top papers with similarity scores, tokens dict, search metadata dict)
    """
    tokens = {"input": 0, "output": 0}
    all_papers = {}  # paper_id -> paper dict (for deduplication)
    search_metadata = {"mode": "comprehensive"}

    title = solution.get('title', '')
    description = solution.get('description', '')

    # Strategy 1: Generate 3 keyword queries
    queries, query_tokens = generate_novelty_queries(solution, problem_summary, client, num_queries=3)
    tokens["input"] += query_tokens["input"]
    tokens["output"] += query_tokens["output"]
    search_metadata["queries"] = queries

    snippet_enabled = get_config("assessment.novelty_check.snippet_search_enabled", True)
    snippet_status = "+ snippet search" if snippet_enabled else "(snippet search disabled)"
    print(f"      Comprehensive: {len(queries)} queries {snippet_status}")

    # Execute keyword searches
    for i, query in enumerate(queries):
        if i > 0:
            time.sleep(1.0)  # Rate limit
        result = query_semantic_scholar(query, max_results=papers_per_query)
        for paper in result.get("papers", []):
            paper_id = _get_paper_id(paper)
            if paper_id not in all_papers:
                all_papers[paper_id] = paper
        print(f"        Query {i+1}: '{query}' -> {len(result.get('papers', []))} papers")

    # Strategy 2: Snippet search with title + description + target domain (optional)
    if snippet_enabled:
        # Semantic Scholar snippet API has 300 char limit
        time.sleep(1.0)
        snippet_query = f"{title[:80]}. {description[:120]} Applied to: {problem_summary[:80]}"
        snippet_query = snippet_query[:300]  # Enforce limit
        search_metadata["snippet_query"] = snippet_query
        snippet_result = query_semantic_scholar_snippet(snippet_query, max_results=papers_per_query)
        for paper in snippet_result.get("papers", []):
            paper_id = _get_paper_id(paper)
            if paper_id not in all_papers:
                all_papers[paper_id] = paper
        print(f"        Snippet search -> {len(snippet_result.get('papers', []))} papers")
    else:
        print(f"        Snippet search DISABLED")

    papers_list = list(all_papers.values())
    search_metadata["total_candidates"] = len(papers_list)
    print(f"        Total unique: {len(papers_list)} papers")

    if not papers_list:
        return [], tokens, search_metadata

    # Strategy 3: Embedding reranking
    # Get paper IDs that have Semantic Scholar IDs
    paper_ids_to_fetch = []
    for paper in papers_list:
        # Try to get Semantic Scholar paper ID
        if paper.get('url'):
            # Extract from URL: https://www.semanticscholar.org/paper/xxx
            url = paper['url']
            if 'semanticscholar.org/paper/' in url:
                parts = url.split('/')
                if parts:
                    paper_ids_to_fetch.append(parts[-1])

    # Fetch embeddings for papers
    print(f"        Fetching embeddings...")
    paper_embeddings = fetch_paper_embeddings_batch(paper_ids_to_fetch) if paper_ids_to_fetch else {}

    # Use Haiku to rewrite query emphasizing application to target domain
    prompt_path = Path("prompts") / "rewrite_embedding_query.txt"
    with open(prompt_path, "r") as f:
        rewrite_prompt_template = f.read()

    key_concepts = solution.get('key_concepts', [])
    key_concepts_str = ', '.join(key_concepts) if isinstance(key_concepts, list) else str(key_concepts)

    rewrite_prompt = rewrite_prompt_template.format(
        title=title,
        description=description,
        key_concepts=key_concepts_str,
        problem_summary=problem_summary
    )

    try:
        response = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=1000,  # Allow Haiku to generate full title + abstract without truncation
            temperature=0,
            messages=[{"role": "user", "content": rewrite_prompt}]
        )
        text = response.content[0].text.strip()

        # Parse JSON (remove code fences if present) - same as extraction.py
        if text.startswith("```"):
            # Remove code fence markers (```json or ```)
            lines = text.split("\n")
            # Skip first line (```) and last line (```)
            text = "\n".join(lines[1:-1])

        # Also handle inline code fence language specifiers
        if text.startswith("json"):
            text = text[4:].strip()

        rewritten = json.loads(text)
        tokens["input"] += response.usage.input_tokens
        tokens["output"] += response.usage.output_tokens
        embedding_title = rewritten.get('title', title)
        embedding_abstract = rewritten.get('abstract', f"{description[:500]} Application to: {problem_summary}")
        print(f"        Rewritten title: {embedding_title}")
    except Exception as e:
        # Fallback to original format on any error
        print(f"        Warning: Failed to rewrite embedding query: {e}")
        try:
            # Show full raw response for debugging
            raw_response = response.content[0].text if 'response' in locals() else 'N/A'
            print(f"        Raw Haiku response:\n{raw_response}")
            if 'text' in locals() and text != raw_response:
                print(f"        After preprocessing:\n{text}")
        except:
            pass
        embedding_title = title
        embedding_abstract = f"{description[:500]} Application to: {problem_summary}"

    search_metadata["embedding_query"] = {"title": embedding_title, "abstract": embedding_abstract}
    idea_embedding = get_specter_embedding(embedding_title, embedding_abstract)

    if not idea_embedding:
        print(f"        Warning: Could not generate idea embedding, returning papers in order")
        return papers_list[:top_k_papers], tokens, search_metadata

    # Score papers by similarity
    scored_papers = []
    for paper in papers_list:
        # Try to get embedding from batch fetch
        paper_embedding = None
        if paper.get('url') and 'semanticscholar.org/paper/' in paper['url']:
            paper_id = paper['url'].split('/')[-1]
            paper_embedding = paper_embeddings.get(paper_id)

        if paper_embedding:
            score = cosine_similarity(idea_embedding, paper_embedding)
        else:
            score = 0.0  # Papers without embeddings get low priority

        paper['similarity_score'] = round(score, 4)
        scored_papers.append((paper, score))

    # Sort by similarity
    scored_papers.sort(key=lambda x: x[1], reverse=True)
    top_papers = [p for p, _ in scored_papers[:top_k_papers]]

    if top_papers:
        top_score = scored_papers[0][1] if scored_papers else 0
        bottom_score = scored_papers[min(len(scored_papers)-1, top_k_papers-1)][1] if scored_papers else 0
        print(f"        Reranked: top similarity {top_score:.3f}, bottom {bottom_score:.3f}")

    return top_papers, tokens, search_metadata


def _get_paper_id(paper: dict) -> str:
    """Generate unique ID for paper deduplication."""
    if paper.get('doi'):
        return f"doi:{paper['doi']}"
    elif paper.get('arxiv_id'):
        return f"arxiv:{paper['arxiv_id']}"
    elif paper.get('url'):
        return f"url:{paper['url']}"
    else:
        return f"title:{paper.get('title', '').lower()[:100]}"


def check_novelty(solutions: list, problem_summary: str, client, model: str) -> tuple[list, dict]:
    """Check novelty for each solution via Semantic Scholar + Claude.

    Args:
        solutions: List of solution dicts with 'title' and 'description'
        problem_summary: The biomedical problem being solved
        client: Anthropic client
        model: Model name to use

    Returns:
        Tuple of (novelty_results list, tokens dict)
    """
    tokens = {"input": 0, "output": 0}
    results = []
    papers_per_query = get_config("assessment.novelty_check.papers_per_query", 10)
    top_k_papers = get_config("assessment.novelty_check.top_k_papers", 10)
    comprehensive = get_config("assessment.novelty_check.comprehensive", False)

    # Load prompt template(s) for novelty assessment
    scoring_type = get_config("assessment.novelty_check.scoring_type", "stratified")
    prompt_variant = get_config("assessment.novelty_check.prompt_variant", "default")

    # Determine prompt directory and filename pattern based on variant
    if prompt_variant == "nlp":
        prompt_dir = Path("eval/prompts")
        prompt_prefix = "novelty_check_nlp_"
    else:
        prompt_dir = Path("prompts")
        prompt_prefix = "novelty_check_"

    prompt_files = {
        "simple": f"{prompt_prefix}simple.txt",
        "stratified": f"{prompt_prefix}stratified.txt",
        "stratified_anchored": f"{prompt_prefix}stratified_anchored.txt",
        "binary": f"{prompt_prefix}binary.txt"
    }

    # Load prompt templates
    prompt_templates = {}
    if scoring_type == "all":
        # Load all templates
        for method, filename in prompt_files.items():
            with open(prompt_dir / filename, "r") as f:
                prompt_templates[method] = f.read()
        scoring_methods = ["simple", "stratified", "stratified_anchored", "binary"]
        print(f"  Novelty check prompt: Using ALL methods ({prompt_variant} variant)")
    else:
        # Load single template
        prompt_file = prompt_files.get(scoring_type, f"{prompt_prefix}stratified.txt")
        with open(prompt_dir / prompt_file, "r") as f:
            prompt_templates[scoring_type] = f.read()
        scoring_methods = [scoring_type]
        print(f"  Novelty check prompt: {prompt_dir}/{prompt_file}")

    mode_str = "comprehensive" if comprehensive else "standard"
    print(f"  Checking novelty for {len(solutions)} solutions ({mode_str} mode, {scoring_type} scoring)...")

    for idx, solution in enumerate(solutions):
        title = solution.get('title', '')
        description = solution.get('description', '')
        key_concepts = solution.get('key_concepts', [])

        # Add delay between solutions to avoid rate limiting
        if idx > 0:
            time.sleep(2.0)

        # Print full title (no truncation)
        print(f"    [{idx + 1}/{len(solutions)}] \"{title}\"")

        if comprehensive:
            # Comprehensive mode: 3 queries + snippet search + embedding rerank
            papers, comp_tokens, search_metadata = retrieve_papers_comprehensive(solution, problem_summary, client, papers_per_query, top_k_papers)
            tokens["input"] += comp_tokens["input"]
            tokens["output"] += comp_tokens["output"]
            rate_limited = False
        else:
            # Standard mode: single query
            queries, query_tokens = generate_novelty_queries(solution, problem_summary, client, num_queries=1)
            tokens["input"] += query_tokens["input"]
            tokens["output"] += query_tokens["output"]
            query = queries[0] if queries else ""
            search_metadata = {"mode": "standard", "query": query}

            result = query_semantic_scholar(query, max_results=top_k_papers)
            papers = result.get("papers", [])
            rate_limited = result.get("rate_limited", False)

            papers_status = f"{len(papers)} papers found"
            if not papers and rate_limited:
                papers_status = "0 papers (rate limited - retries exhausted)"
            print(f"      Query: '{query}' - {papers_status}")

        # Interactive paper addition (if enabled)
        if get_config("interactive.paper_addition.enabled", False):
            from interactive import prompt_paper_addition, lookup_paper_by_title
            additional_titles = prompt_paper_addition(papers, title)
            for add_title in additional_titles:
                print(f"      Looking up: {add_title[:40]}...")
                extra_paper = lookup_paper_by_title(add_title)
                if extra_paper:
                    papers.append(extra_paper)
                    print(f"      + Added: {extra_paper.get('title', 'Unknown')[:40]}...")
                else:
                    print(f"      x Not found: {add_title[:40]}...")

        if not papers:
            assessment_msg = "Rate limit exhausted - unable to assess novelty" if rate_limited else "No papers found - unable to assess novelty"
            if scoring_type == "all":
                # For "all" mode, create structure with all methods marked as unable to assess
                results.append({
                    "scoring_methods": {
                        "simple": {"novelty_score": None, "assessment": assessment_msg},
                        "stratified": {"novelty_score": None, "assessment": assessment_msg},
                        "stratified_anchored": {"novelty_score": None, "assessment": assessment_msg},
                        "binary": {"novelty_score": None, "assessment": assessment_msg}
                    },
                    "papers_found": 0,
                    "papers": [],
                    "rate_limited": rate_limited
                })
            else:
                results.append({
                    "novelty_score": None,
                    "assessment": assessment_msg,
                    "papers_found": 0,
                    "papers": [],
                    "rate_limited": rate_limited
                })
            continue

        # Format papers for Claude
        papers_text = ""
        for i, paper in enumerate(papers, 1):
            papers_text += f"{i}. \"{paper.get('title', 'Unknown')}\" ({paper.get('year', 'N/A')})\n"
            if paper.get('abstract'):
                papers_text += f"   Abstract: {paper['abstract'][:300]}...\n"

        # Format key concepts for the prompt
        key_concepts_str = ', '.join(key_concepts) if key_concepts else 'Not specified'

        # Run novelty assessment with each scoring method
        if scoring_type == "all":
            # Run all three methods and store separately
            all_method_results = {}
            for method in scoring_methods:
                prompt_template = prompt_templates[method]
                prompt = prompt_template.format(
                    title=title,
                    description=description,
                    key_concepts=key_concepts_str,
                    problem_summary=problem_summary,
                    papers_text=papers_text
                )

                try:
                    response = client.messages.create(
                        model=model,
                        max_tokens=500,
                        messages=[{"role": "user", "content": prompt}]
                    )
                    tokens["input"] += response.usage.input_tokens
                    tokens["output"] += response.usage.output_tokens

                    response_text = ""
                    for block in response.content:
                        if block.type == "text":
                            response_text = block.text.strip()
                            break

                    # Extract JSON
                    if "```json" in response_text:
                        start = response_text.find("```json") + 7
                        end = response_text.find("```", start)
                        response_text = response_text[start:end].strip()
                    elif "```" in response_text:
                        start = response_text.find("```") + 3
                        end = response_text.find("```", start)
                        response_text = response_text[start:end].strip()

                    method_result = json.loads(response_text)

                    # Handle binary scoring format (convert is_novel to novelty_score)
                    if method == "binary" and "is_novel" in method_result:
                        method_result["novelty_score"] = 10 if method_result["is_novel"] else 0
                        method_result["methodology_overlap"] = 0 if method_result["is_novel"] else 10

                    all_method_results[method] = method_result

                except Exception as e:
                    print(f"      Warning: {method} scoring failed: {e}")
                    all_method_results[method] = {
                        "novelty_score": 5,
                        "assessment": f"Could not assess novelty ({method})"
                    }

            # Combine results from all methods
            result = {
                "scoring_methods": all_method_results,
                "papers_found": len(papers),
                "papers": [{
                    "title": p.get("title", ""),
                    "year": p.get("year"),
                    "url": p.get("url", ""),
                    "authors": p.get("authors", []),
                    "citations": p.get("citations"),
                    "github_urls": p.get("github_urls", []),
                    "similarity_score": p.get("similarity_score")
                } for p in papers],
                "search_metadata": search_metadata
            }

            # Print comparison
            print(f"      Scores: simple={all_method_results.get('simple', {}).get('novelty_score', 'N/A')}, "
                  f"stratified={all_method_results.get('stratified', {}).get('novelty_score', 'N/A')}, "
                  f"stratified_anchored={all_method_results.get('stratified_anchored', {}).get('novelty_score', 'N/A')}, "
                  f"binary={all_method_results.get('binary', {}).get('novelty_score', 'N/A')}")

            results.append(result)

        else:
            # Single method
            prompt_template = prompt_templates[scoring_type]
            prompt = prompt_template.format(
                title=title,
                description=description,
                key_concepts=key_concepts_str,
                problem_summary=problem_summary,
                papers_text=papers_text
            )

            try:
                response = client.messages.create(
                    model=model,
                    max_tokens=500,
                    messages=[{"role": "user", "content": prompt}]
                )
                tokens["input"] += response.usage.input_tokens
                tokens["output"] += response.usage.output_tokens

                response_text = ""
                for block in response.content:
                    if block.type == "text":
                        response_text = block.text.strip()
                        break

                # Extract JSON
                if "```json" in response_text:
                    start = response_text.find("```json") + 7
                    end = response_text.find("```", start)
                    response_text = response_text[start:end].strip()
                elif "```" in response_text:
                    start = response_text.find("```") + 3
                    end = response_text.find("```", start)
                    response_text = response_text[start:end].strip()

                result = json.loads(response_text)

                # Handle binary scoring format (convert is_novel to novelty_score)
                if scoring_type == "binary" and "is_novel" in result:
                    result["novelty_score"] = 10 if result["is_novel"] else 0
                    result["methodology_overlap"] = 0 if result["is_novel"] else 10

                result["papers_found"] = len(papers)
                result["papers"] = [{
                    "title": p.get("title", ""),
                    "year": p.get("year"),
                    "url": p.get("url", ""),
                    "authors": p.get("authors", []),
                    "citations": p.get("citations"),
                    "github_urls": p.get("github_urls", []),
                    "similarity_score": p.get("similarity_score")
                } for p in papers]
                result["search_metadata"] = search_metadata
                results.append(result)

            except Exception as e:
                print(f"      Warning: Novelty check failed for '{title[:30]}...': {e}")
                results.append({
                    "novelty_score": 5,
                    "assessment": "Could not assess novelty",
                    "papers_found": len(papers),
                    "papers": [],
                    "search_metadata": search_metadata
                })

    return results, tokens


def assess_solutions(problem_summary: str, solutions: list) -> list:
    """
    Assess and rank solutions by relevance and feasibility.

    Args:
        problem_summary: Original problem description
        solutions: List of solutions from search agent

    Returns:
        List of top 3 assessed solutions, sorted by score (highest first)
    """
    # Initialize Claude client
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise ValueError("ANTHROPIC_API_KEY environment variable not set")

    client = Anthropic(api_key=api_key)

    # Limit solutions to assess based on config
    num_to_assess = get_config("assessment.num_solutions_to_assess")
    if num_to_assess is not None:
        solutions = solutions[:num_to_assess]

    # Check scoring mode (with backward compatibility)
    old_scoring_enabled = get_config("assessment.scoring_enabled")
    if old_scoring_enabled is not None:
        # Backward compatibility: convert old boolean to new mode
        scoring_mode = "unified" if old_scoring_enabled else "none"
        print(f"  Warning: Using deprecated 'scoring_enabled' config. Please use 'scoring_mode' instead.")
    else:
        scoring_mode = get_config("assessment.scoring_mode", "unified")

    # Handle scoring mode
    if scoring_mode == "none":
        print(f"  Scoring DISABLED - skipping assessment, running novelty check only...")
    elif scoring_mode == "split":
        print(f"  Assessing {len(solutions)} solutions using SPLIT scoring (overall + code)...")
    else:  # unified
        print(f"  Assessing {len(solutions)} solutions using UNIFIED scoring...")

    total_tokens = {"input": 0, "output": 0}
    web_search_tokens_total = {"input": 0, "output": 0}

    # Load prompt template based on scoring mode
    if scoring_mode == "split":
        # Check if domain-neutral scoring is requested
        domain_neutral = get_config("assessment.domain_neutral", False)
        if domain_neutral:
            prompt_file = "prompts/assessment_explainable_split_domain_neutral.txt"
        else:
            prompt_file = "prompts/assessment_explainable_split.txt"
    else:  # unified or none (prompt not used for none, but load for consistency)
        prompt_file = get_config("assessment.prompt_file", "prompts/assessment_explainable.txt")

    prompt_path = Path(prompt_file)
    if scoring_mode != "none":
        print(f"  Using prompt: {prompt_path}")
    with open(prompt_path, "r") as f:
        prompt_template = f.read()

    if scoring_mode != "none":
        # Check if solutions already have GitHub repos embedded (from Perplexity search)
        if solutions and 'github_repos' in solutions[0]:
            # Solutions already have GitHub data - extract and format
            # Create repos_by_solution mapping from embedded data
            repos_by_solution = {idx: sol.get('github_repos', []) for idx, sol in enumerate(solutions)}
            github_data = _format_embedded_github_repos(solutions)
            web_search_tokens = {"input": 0, "output": 0}
        else:
            # Legacy path: fetch GitHub repos as before (for web_search provider)
            use_web_search = get_config("assessment.github.use_web_search", False)
            model = get_config("model.assessment_model") or get_config("model.name", "claude-sonnet-4-20250514")
            repos_by_solution, github_api_requests, web_search_tokens = fetch_github_repos_for_solutions(solutions, client, model)
            github_data = format_github_repos_for_prompt(solutions, repos_by_solution)

        # Track web search tokens
        web_search_tokens_total["input"] += web_search_tokens["input"]
        web_search_tokens_total["output"] += web_search_tokens["output"]

        # Filter solutions to remove unnecessary fields (abstracts, excessive metadata)
        filtered_solutions = [_filter_solution_for_assessment(sol) for sol in solutions]

        # Format solutions for prompt
        solutions_text = json.dumps(filtered_solutions, indent=2)

        # Get weights from config based on scoring mode
        if scoring_mode == "split":
            # Split mode: load two independent weight configs
            overall_weights = get_config("assessment.weights_split.overall_score", {
                "conceptual_match": 100
            })
            code_weights = get_config("assessment.weights_split.code_availability_score", {
                "code_availability": 25,
                "code_quality": 25,
                "transfer_effort": 25,
                "repo_relevance": 25
            })

            # Fill in prompt template for split mode
            prompt = prompt_template.format(
                problem_summary=problem_summary,
                solutions=solutions_text,
                github_repos=github_data,
                overall_conceptual_pct=int(overall_weights.get("conceptual_match", 100)),
                code_availability_pct=int(code_weights.get("code_availability", 25)),
                code_quality_pct=int(code_weights.get("code_quality", 25)),
                code_transfer_pct=int(code_weights.get("transfer_effort", 25)),
                code_repo_pct=int(code_weights.get("repo_relevance", 25))
            )
        else:  # unified mode
            # Unified mode: load single weight config (current behavior)
            weights = get_config("assessment.weights_unified", {
                "conceptual_match": 50,
                "repo_relevance": 20,
                "code_availability": 10,
                "transfer_effort": 10,
                "code_quality": 10
            })
            weight_conceptual = weights.get("conceptual_match", 50) / 100
            weight_repo = weights.get("repo_relevance", 20) / 100
            weight_availability = weights.get("code_availability", 10) / 100
            weight_transfer = weights.get("transfer_effort", 10) / 100
            weight_quality = weights.get("code_quality", 10) / 100

            # Fill in prompt template for unified mode
            prompt = prompt_template.format(
                problem_summary=problem_summary,
                solutions=solutions_text,
                github_repos=github_data,
                weight_conceptual=weight_conceptual,
                weight_repo=weight_repo,
                weight_availability=weight_availability,
                weight_transfer=weight_transfer,
                weight_quality=weight_quality,
                weight_conceptual_pct=int(weights.get("conceptual_match", 50)),
                weight_repo_pct=int(weights.get("repo_relevance", 20)),
                weight_availability_pct=int(weights.get("code_availability", 10)),
                weight_transfer_pct=int(weights.get("transfer_effort", 10)),
                weight_quality_pct=int(weights.get("code_quality", 10))
            )
    
        # Score each solution individually to eliminate batch effects
        model = get_config("model.assessment_model") or get_config("model.name", "claude-sonnet-4-20250514")
        assessed_solutions = []

        for idx, filtered_solution in enumerate(filtered_solutions):
            original_solution = solutions[idx]
            print(f"    [{idx + 1}/{len(filtered_solutions)}] Scoring: {original_solution.get('title', 'Unknown')[:50]}...")

            # Format GitHub data for this solution only
            repos = repos_by_solution.get(idx, [])
            if solutions and 'github_repos' in solutions[0]:
                single_github_data = _format_embedded_github_repos([original_solution])
            else:
                single_github_data = format_github_repos_for_prompt([original_solution], {0: repos})

            # Format prompt for single solution (wrap in list to reuse template)
            single_solution_json = json.dumps([filtered_solution], indent=2)

            if scoring_mode == "split":
                single_prompt = prompt_template.format(
                    problem_summary=problem_summary,
                    solutions=single_solution_json,
                    github_repos=single_github_data,
                    overall_conceptual_pct=int(overall_weights.get("conceptual_match", 100)),
                    code_availability_pct=int(code_weights.get("code_availability", 25)),
                    code_quality_pct=int(code_weights.get("code_quality", 25)),
                    code_transfer_pct=int(code_weights.get("transfer_effort", 25)),
                    code_repo_pct=int(code_weights.get("repo_relevance", 25))
                )
            else:  # unified
                single_prompt = prompt_template.format(
                    problem_summary=problem_summary,
                    solutions=single_solution_json,
                    github_repos=single_github_data,
                    weight_conceptual=weight_conceptual,
                    weight_repo=weight_repo,
                    weight_availability=weight_availability,
                    weight_transfer=weight_transfer,
                    weight_quality=weight_quality,
                    weight_conceptual_pct=int(weights.get("conceptual_match", 50)),
                    weight_repo_pct=int(weights.get("repo_relevance", 20)),
                    weight_availability_pct=int(weights.get("code_availability", 10)),
                    weight_transfer_pct=int(weights.get("transfer_effort", 10)),
                    weight_quality_pct=int(weights.get("code_quality", 10))
                )

            # Call API for this single solution
            response = client.messages.create(
                model=model,
                max_tokens=get_config("assessment.max_tokens", 2500),
                messages=[{
                    "role": "user",
                    "content": single_prompt
                }]
            )

            # Track tokens
            total_tokens["input"] += response.usage.input_tokens
            total_tokens["output"] += response.usage.output_tokens

            # Extract text from response
            response_text = ""
            for block in response.content:
                if block.type == "text":
                    response_text = block.text.strip()
                    break

            # Parse JSON from response - handle various formats
            original_text = response_text  # Keep for debugging

            # 1. Try to extract JSON from markdown code blocks
            if "```json" in response_text:
                start = response_text.find("```json") + 7
                end = response_text.find("```", start)
                if end != -1:
                    response_text = response_text[start:end].strip()
            elif "```" in response_text:
                start = response_text.find("```") + 3
                end = response_text.find("```", start)
                if end != -1:
                    response_text = response_text[start:end].strip()

            # 2. Try to find JSON array in the text
            if not response_text.startswith("["):
                bracket_pos = response_text.find("[")
                if bracket_pos != -1:
                    response_text = response_text[bracket_pos:]
                    last_bracket = response_text.rfind("]")
                    if last_bracket != -1:
                        response_text = response_text[:last_bracket + 1]

            # 3. Remove any trailing text after JSON
            if response_text.startswith("["):
                bracket_count = 0
                for i, char in enumerate(response_text):
                    if char == '[':
                        bracket_count += 1
                    elif char == ']':
                        bracket_count -= 1
                        if bracket_count == 0:
                            response_text = response_text[:i+1]
                            break

            # Parse as array and extract first element
            try:
                single_assessed = json.loads(response_text)
                if isinstance(single_assessed, list) and len(single_assessed) > 0:
                    assessed_solutions.append(single_assessed[0])
                else:
                    # Fallback if LLM returned object instead of array
                    assessed_solutions.append(single_assessed)
            except json.JSONDecodeError as e:
                print(f"      ✗ JSON parsing failed for solution {idx + 1}: {e}")
                # Add solution with minimal fields to avoid breaking the pipeline
                assessed_solutions.append({
                    "title": original_solution.get("title", "Unknown"),
                    "source_domain": original_solution.get("source_domain", "unknown"),
                    "score": 0.0 if scoring_mode == "unified" else None,
                    "overall_score": 0.0 if scoring_mode == "split" else None,
                    "code_availability_score": None if scoring_mode == "split" else None,
                    "rationale": f"JSON parsing error: {str(e)}"
                })

        # Print total prompt info
        print(f"    Assessed {len(assessed_solutions)} solutions individually")
        print(f"    Total tokens: {total_tokens['input']:,} input, {total_tokens['output']:,} output")

        try:
            pass  # Continue to validation logic below
    
            # Validate score breakdown math and fix negative scores
            verbose = get_config("output.verbose_validation", True)
            if verbose:
                print(f"    Validating score calculations...")

            if scoring_mode == "split":
                # Validate both overall_score and code_availability_score
                for idx, solution in enumerate(assessed_solutions):
                    # Validate overall_score (just conceptual_match at 100% weight)
                    if 'overall_score_breakdown' in solution:
                        breakdown = solution['overall_score_breakdown']
                        if 'conceptual_match' in breakdown:
                            # Fix negative scores
                            if breakdown['conceptual_match'].get('score', 0) < 0:
                                breakdown['conceptual_match']['score'] = 0

                            # Since conceptual_match is 100% weight, overall_score = conceptual_match score
                            conceptual_score = breakdown['conceptual_match']['score']
                            breakdown['conceptual_match']['weighted_value'] = conceptual_score
                            solution['overall_score'] = round(conceptual_score, 2)

                    # Validate code_availability_score
                    if 'code_availability_score_breakdown' in solution:
                        breakdown = solution['code_availability_score_breakdown']
                        for criterion, data in breakdown.items():
                            if data.get('score', 0) < 0:
                                data['score'] = 0
                            # Recalculate weighted_value (weights are percentages)
                            weight = code_weights.get(criterion, 25) / 100
                            data['weighted_value'] = data['score'] * weight

                        weighted_sum = sum(
                            breakdown[c].get('weighted_value', 0)
                            for c in ['code_availability', 'code_quality', 'transfer_effort', 'repo_relevance']
                            if c in breakdown
                        )
                        if 'code_availability_score' in solution:
                            if abs(weighted_sum - solution['code_availability_score']) > 0.01:
                                if verbose:
                                    print(f"      ✗ Code score mismatch for solution {idx + 1}: correcting to {weighted_sum:.2f}")
                                solution['code_availability_score'] = round(weighted_sum, 2)
            else:  # unified mode
                # Validate single score (current behavior)
                for idx, solution in enumerate(assessed_solutions):
                    if 'score_breakdown' in solution and 'score' in solution:
                        breakdown = solution['score_breakdown']

                        # Fix any negative scores
                        for criterion, data in breakdown.items():
                            if data.get('score', 0) < 0:
                                data['score'] = 0
                                # Recalculate weighted value
                                weight = {
                                    'conceptual_match': weight_conceptual,
                                    'repo_relevance': weight_repo,
                                    'code_availability': weight_availability,
                                    'transfer_effort': weight_transfer,
                                    'code_quality': weight_quality
                                }.get(criterion, 0)
                                data['weighted_value'] = data['score'] * weight

                        weighted_sum = sum(
                            criterion.get('weighted_value', 0)
                            for criterion in breakdown.values()
                        )
                        final_score = solution['score']

                        # Check if they match (allow small floating point tolerance)
                        if abs(weighted_sum - final_score) > 0.01:
                            if verbose:
                                print(f"      ✗ Score mismatch for solution {idx + 1}: LLM returned {final_score}, weighted sum is {weighted_sum:.2f}")
                                print(f"      ✓ Corrected score: {weighted_sum:.2f}")
                            solution['score'] = round(weighted_sum, 2)
    
            print(f"  ✓ Assessed {len(assessed_solutions)} solutions")
    
        except json.JSONDecodeError as e:
            print(f"\n{'='*60}")
            print(f"ERROR: JSON parsing failed")
            print(f"{'='*60}")
            print(f"Error: {e}")
            print(f"Response length: {len(original_text)}")
            print(f"\nFull response text:")
            print(f"{'-'*60}")
            print(original_text)
            print(f"{'-'*60}")
            print(f"\nExtracted JSON text (after preprocessing):")
            print(f"{'-'*60}")
            print(response_text)
            print(f"{'-'*60}\n")
            raise
    
        # Merge original solution fields (sources, source_titles, papers) into assessed solutions
        # The LLM only outputs assessment fields, so we need to preserve the original metadata
        solution_lookup = {sol['title']: sol for sol in solutions}
        for assessed in assessed_solutions:
            original = solution_lookup.get(assessed.get('title'))
            if original:
                # Preserve source/paper fields from original solution
                for field in ['sources', 'source_titles', 'papers', 'description', 'key_concepts', 'relevance', 'github_repos']:
                    if field in original and field not in assessed:
                        assessed[field] = original[field]

        # Handle solutions without GitHub repos in split mode
        if scoring_mode == "split":
            for assessed in assessed_solutions:
                # Check if solution has no GitHub repos
                repos = assessed.get('github_repos', [])
                if not repos or (isinstance(repos, list) and len(repos) == 0):
                    # Set code availability score to None (N/A) - no code to evaluate
                    assessed['code_availability_score'] = None
                    if 'code_availability_score_breakdown' in assessed:
                        assessed['code_availability_score_breakdown'] = None

        # Sort by score (mode-dependent)
        if scoring_mode == "split":
            sort_by = get_config("assessment.weights_split.default_sort_by", "overall_score")
            # Handle None values in sorting (treat as -1 so they go to the end)
            assessed_solutions.sort(key=lambda x: x.get(sort_by) if x.get(sort_by) is not None else -1, reverse=True)
        else:  # unified
            assessed_solutions.sort(key=lambda x: x.get('score', 0), reverse=True)

    else:
        # Scoring disabled - use original solutions without scores
        assessed_solutions = solutions
        model = get_config("model.assessment_model") or get_config("model.name", "claude-sonnet-4-20250514")

    # Novelty check (after scoring to ensure it doesn't affect scores)
    novelty_tokens = {"input": 0, "output": 0}
    if get_config("assessment.novelty_check.enabled", False):
        novelty_results, novelty_tokens = check_novelty(assessed_solutions, problem_summary, client, model)
        # Attach novelty data to each solution
        for idx, sol in enumerate(assessed_solutions):
            if idx < len(novelty_results):
                sol["novelty_data"] = novelty_results[idx]

    # Return all assessed solutions with combined token counts
    tokens = {
        "input": total_tokens["input"] + web_search_tokens_total["input"] + novelty_tokens["input"],
        "output": total_tokens["output"] + web_search_tokens_total["output"] + novelty_tokens["output"],
        "provider": "anthropic"
    }

    return assessed_solutions, tokens


def print_assessment(assessed: list, all_assessed: list = None):
    """Pretty print assessment results.

    Args:
        assessed: Top solutions to show in detailed view
        all_assessed: All assessed solutions for summary table (defaults to assessed)
    """
    if all_assessed is None:
        all_assessed = assessed

    # Detect scoring mode
    scoring_mode = get_config("assessment.scoring_mode", "unified")

    print("\n" + "="*60)
    print("TOP RANKED SOLUTIONS")
    print("="*60)

    for idx, sol in enumerate(assessed, 1):
        # Display scores based on mode
        if scoring_mode == "split":
            overall = sol.get('overall_score')
            code = sol.get('code_availability_score')
            print(f"\n{idx}. {sol['title']}")
            if overall is not None:
                overall_str = f"{overall}/10"
            else:
                overall_str = "N/A"
            if code is not None:
                code_str = f"{code}/10"
            else:
                code_str = "N/A"
            print(f"   Overall Score: {overall_str} | Code Availability Score: {code_str}")
        else:  # unified
            score = sol.get('score')
            if score is not None:
                print(f"\n{idx}. {sol['title']} (Score: {score}/10)")
            else:
                print(f"\n{idx}. {sol['title']}")
        print(f"   Domain: {sol['source_domain']}")

        # Display score breakdown based on mode
        if scoring_mode == "split":
            # Display overall score breakdown
            if 'overall_score_breakdown' in sol and sol['overall_score_breakdown'] is not None:
                print(f"   Overall Score Breakdown:")
                breakdown = sol['overall_score_breakdown']
                for criterion, data in breakdown.items():
                    criterion_name = criterion.replace('_', ' ').title()
                    print(f"     • {criterion_name}: {data['score']}/10 (weighted: {data['weighted_value']:.2f})")
                    print(f"       → {data['explanation']}")

            # Display code availability score breakdown (skip if N/A)
            if 'code_availability_score_breakdown' in sol and sol['code_availability_score_breakdown'] is not None:
                print(f"   Code Availability Score Breakdown:")
                breakdown = sol['code_availability_score_breakdown']
                for criterion, data in breakdown.items():
                    criterion_name = criterion.replace('_', ' ').title()
                    print(f"     • {criterion_name}: {data['score']}/10 (weighted: {data['weighted_value']:.2f})")
                    print(f"       → {data['explanation']}")
            elif sol.get('code_availability_score') is None:
                print(f"   Code Availability Score: N/A (no GitHub repositories found)")
        else:  # unified
            # Display single score breakdown (current behavior)
            if 'score_breakdown' in sol:
                print(f"   Score Breakdown:")
                breakdown = sol['score_breakdown']
                for criterion, data in breakdown.items():
                    criterion_name = criterion.replace('_', ' ').title()
                    print(f"     • {criterion_name}: {data['score']}/10 (weighted: {data['weighted_value']:.2f})")
                    print(f"       → {data['explanation']}")

        # Display code quality if available
        if 'code_quality_rating' in sol:
            print(f"   Code Quality: {sol['code_quality_rating']}")
        elif 'code_quality' in sol:
            print(f"   Code Quality: {sol['code_quality']}")

        # Display novelty if available
        if 'novelty_data' in sol:
            novelty = sol['novelty_data']
            # Handle "all" scoring type (multiple methods)
            if 'scoring_methods' in novelty:
                methods = novelty['scoring_methods']
                scores = []
                for method in ['simple', 'stratified', 'stratified_anchored', 'binary']:
                    if method in methods:
                        score = methods[method].get('novelty_score', 'N/A')
                        scores.append(f"{method}={score}")
                print(f"   Novelty: {', '.join(scores)}")
            else:
                # Single method
                novelty_score = novelty.get('novelty_score', 'N/A')
                novelty_score_str = f"{novelty_score:.1f}" if isinstance(novelty_score, (int, float)) else str(novelty_score)
                print(f"   Novelty: {novelty_score_str} - {novelty.get('assessment', '')}")

        # Display papers/sources if available
        if 'papers' in sol and sol['papers']:
            print(f"   Papers ({len(sol['papers'])}):")
            for paper in sol['papers'][:3]:  # Show up to 3 papers
                title = paper.get('title', 'Unknown')
                url = paper.get('url', '')
                print(f"     • {title}")
                if url:
                    print(f"       {url}")
        elif 'sources' in sol and sol['sources']:
            print(f"   Sources ({len(sol['sources'])}):")
            for i, url in enumerate(sol['sources'][:3]):  # Show up to 3 sources
                title = sol.get('source_titles', [])[i] if i < len(sol.get('source_titles', [])) else None
                if title:
                    print(f"     • {title}")
                    print(f"       {url}")
                else:
                    print(f"     • {url}")

        # Display Github repos if available
        if 'github_repos' in sol and sol['github_repos']:
            # github_repos is a list of dicts with 'url' field
            repo_urls = [repo.get('url', repo) if isinstance(repo, dict) else repo for repo in sol['github_repos'][:2]]
            print(f"   Github Repos: {', '.join(repo_urls)}")

        # Display assessment fields if available (only present when scoring is enabled)
        if 'strengths' in sol:
            print(f"   Strengths: {', '.join(sol['strengths'])}")
        if 'challenges' in sol:
            print(f"   Challenges: {', '.join(sol['challenges'])}")
        if 'rationale' in sol:
            print(f"   Rationale: {sol['rationale']}")

    print("="*60 + "\n")

    # Print summary table of all solutions
    print("\n" + "="*70)
    print("ALL SOLUTIONS SUMMARY")
    print("="*70)

    # Check if using "all" scoring mode
    using_all_mode = any('scoring_methods' in sol.get('novelty_data', {}) for sol in all_assessed)

    if using_all_mode:
        # Check if using split scoring mode for header
        if scoring_mode == "split":
            print(f"{'Rank':<6}{'Overall':<9}{'Code':<9}{'Simp':<6}{'Strat':<6}{'StratA':<7}{'Bin':<6}{'Domain':<12}{'Title'}")
            print("-"*85)
            for idx, sol in enumerate(all_assessed, 1):
                title = sol.get('title', 'Unknown')
                if len(title) > 28:
                    title = title[:25] + "..."
                domain = sol.get('source_domain', 'unknown')
                if len(domain) > 11:
                    domain = domain[:8] + "..."

                overall = sol.get('overall_score')
                code = sol.get('code_availability_score')

                # Handle None values
                overall_str = "N/A" if overall is None else f"{overall:.2f}" if isinstance(overall, (int, float)) else str(overall)
                code_str = "N/A" if code is None else f"{code:.2f}" if isinstance(code, (int, float)) else str(code)

                novelty_data = sol.get('novelty_data', {})
                methods = novelty_data.get('scoring_methods', {})
                simple = methods.get('simple', {}).get('novelty_score', '-')
                stratified = methods.get('stratified', {}).get('novelty_score', '-')
                stratified_anchored = methods.get('stratified_anchored', {}).get('novelty_score', '-')
                binary = methods.get('binary', {}).get('novelty_score', '-')

                simple_str = str(simple) if simple == '-' else f"{simple}"
                strat_str = str(stratified) if stratified == '-' else f"{stratified}"
                strat_a_str = str(stratified_anchored) if stratified_anchored == '-' else f"{stratified_anchored}"
                binary_str = str(binary) if binary == '-' else f"{binary}"

                print(f"{idx:<6}{overall_str:<9}{code_str:<9}{simple_str:<6}{strat_str:<6}{strat_a_str:<7}{binary_str:<6}{domain:<12}{title}")
        else:  # unified mode
            print(f"{'Rank':<6}{'Score':<8}{'Simp':<6}{'Strat':<6}{'StratA':<7}{'Bin':<6}{'Domain':<14}{'Title'}")
            print("-"*80)
            for idx, sol in enumerate(all_assessed, 1):
                title = sol.get('title', 'Unknown')
                if len(title) > 30:
                    title = title[:27] + "..."
                domain = sol.get('source_domain', 'unknown')
                if len(domain) > 13:
                    domain = domain[:10] + "..."
                score = sol.get('score', 0)

                novelty_data = sol.get('novelty_data', {})
                methods = novelty_data.get('scoring_methods', {})
                simple = methods.get('simple', {}).get('novelty_score', '-')
                stratified = methods.get('stratified', {}).get('novelty_score', '-')
                stratified_anchored = methods.get('stratified_anchored', {}).get('novelty_score', '-')
                binary = methods.get('binary', {}).get('novelty_score', '-')

                simple_str = str(simple) if simple == '-' else f"{simple}"
                strat_str = str(stratified) if stratified == '-' else f"{stratified}"
                strat_a_str = str(stratified_anchored) if stratified_anchored == '-' else f"{stratified_anchored}"
                binary_str = str(binary) if binary == '-' else f"{binary}"

                print(f"{idx:<6}{score:<8.2f}{simple_str:<6}{strat_str:<6}{strat_a_str:<7}{binary_str:<6}{domain:<14}{title}")
    else:
        # Check if using split scoring mode
        if scoring_mode == "split":
            print(f"{'Rank':<6}{'Overall':<9}{'Code':<9}{'Novelty':<10}{'Domain':<18}{'Title'}")
            print("-"*75)
            for idx, sol in enumerate(all_assessed, 1):
                title = sol.get('title', 'Unknown')
                if len(title) > 30:
                    title = title[:27] + "..."
                domain = sol.get('source_domain', 'unknown')
                if len(domain) > 17:
                    domain = domain[:14] + "..."

                overall = sol.get('overall_score')
                code = sol.get('code_availability_score')

                # Handle None (N/A) values
                if overall is None:
                    overall_str = "N/A"
                elif isinstance(overall, (int, float)):
                    overall_str = f"{overall:.2f}"
                else:
                    overall_str = str(overall)

                if code is None:
                    code_str = "N/A"
                elif isinstance(code, (int, float)):
                    code_str = f"{code:.2f}"
                else:
                    code_str = str(code)

                novelty_data = sol.get('novelty_data', {})
                novelty = novelty_data.get('novelty_score', '-')
                novelty_str = f"{novelty:.1f}" if isinstance(novelty, (int, float)) else str(novelty)

                print(f"{idx:<6}{overall_str:<9}{code_str:<9}{novelty_str:<10}{domain:<18}{title}")
        else:  # unified mode
            print(f"{'Rank':<6}{'Score':<8}{'Novelty':<10}{'Domain':<22}{'Title'}")
            print("-"*70)
            for idx, sol in enumerate(all_assessed, 1):
                title = sol.get('title', 'Unknown')
                if len(title) > 35:
                    title = title[:32] + "..."
                domain = sol.get('source_domain', 'unknown')
                if len(domain) > 20:
                    domain = domain[:17] + "..."
                score = sol.get('score', 0)

                novelty_data = sol.get('novelty_data', {})
                novelty = novelty_data.get('novelty_score', '-')
                novelty_str = f"{novelty:.1f}" if isinstance(novelty, (int, float)) else str(novelty)

                print(f"{idx:<6}{score:<8.2f}{novelty_str:<10}{domain:<22}{title}")

    print("="*70 + "\n")
