"""Academic API integration for source verification and GitHub discovery."""

import os
import time
import re
import requests
import math
from datetime import datetime
from typing import Any
from concurrent.futures import ThreadPoolExecutor, as_completed
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import get_config


def _get_semantic_scholar_headers() -> dict:
    """Get headers for Semantic Scholar API requests.

    With API key: 100 requests/sec rate limit
    Without API key: 1 request/sec rate limit
    """
    headers = {}
    api_key = os.getenv('SEMANTIC_SCHOLAR_API_KEY')
    if api_key:
        headers['x-api-key'] = api_key
    return headers


def query_semantic_scholar(keywords: str, max_results: int = 5) -> dict:
    """Query Semantic Scholar API for papers.

    Args:
        keywords: Search keywords
        max_results: Maximum number of results to return

    Returns:
        Dict with keys:
            - papers: List of normalized paper dicts
            - rate_limited: Boolean indicating if request was rate limited and retries exhausted
    """
    url = "https://api.semanticscholar.org/graph/v1/paper/search"
    params = {
        "query": keywords,
        "limit": max_results,
        "fields": "title,authors,year,abstract,url,citationCount,externalIds,openAccessPdf"
    }
    headers = _get_semantic_scholar_headers()

    max_retries = get_config("search.academic_apis.max_retries", 5)
    timeout = get_config("search.academic_apis.timeout_seconds", 10)
    # Use shorter delay with API key (100 req/sec allowed)
    delay = get_config("search.academic_apis.semantic_scholar_delay", 0.5)
    if headers.get('x-api-key'):
        delay = min(delay, 0.05)  # 50ms delay with API key

    for attempt in range(max_retries + 1):
        try:
            time.sleep(delay)  # Rate limiting
            response = requests.get(url, params=params, headers=headers, timeout=timeout)

            if response.status_code == 200:
                data = response.json()
                papers = []
                for paper in data.get('data', []):
                    papers.append(_normalize_semantic_scholar_paper(paper))
                return {"papers": papers, "rate_limited": False}
            elif response.status_code == 429:  # Rate limited
                wait_time = 2 ** attempt
                print(f"      Semantic Scholar rate limited, waiting {wait_time}s...")
                time.sleep(wait_time)
                continue
            else:
                return {"papers": [], "rate_limited": False}

        except requests.Timeout:
            if attempt < max_retries:
                continue
            return {"papers": [], "rate_limited": False}
        except Exception:
            return {"papers": [], "rate_limited": False}

    # All retries exhausted due to rate limiting
    return {"papers": [], "rate_limited": True}


def query_semantic_scholar_snippet(query: str, max_results: int = 10) -> dict:
    """Query Semantic Scholar snippet/search endpoint.

    Args:
        query: Full text to search for similar snippets
        max_results: Maximum number of results to return

    Returns:
        Dict with keys:
            - papers: List of normalized paper dicts
            - rate_limited: Boolean indicating if request was rate limited
    """
    url = "https://api.semanticscholar.org/graph/v1/snippet/search"
    params = {
        "query": query,
        "limit": max_results,
        "fields": "paper.paperId,paper.title,paper.authors,paper.year,paper.abstract,paper.url,paper.citationCount,paper.externalIds"
    }
    headers = _get_semantic_scholar_headers()

    max_retries = get_config("search.academic_apis.max_retries", 5)
    timeout = get_config("search.academic_apis.timeout_seconds", 10)
    delay = get_config("search.academic_apis.semantic_scholar_delay", 0.5)
    if headers.get('x-api-key'):
        delay = min(delay, 0.05)

    for attempt in range(max_retries + 1):
        try:
            time.sleep(delay)
            response = requests.get(url, params=params, headers=headers, timeout=timeout)

            if response.status_code == 200:
                data = response.json()
                papers = []
                for item in data.get('data', []):
                    paper = item.get('paper', {})
                    if paper:
                        papers.append(_normalize_semantic_scholar_paper(paper))
                return {"papers": papers, "rate_limited": False}
            elif response.status_code == 429:
                wait_time = 2 ** attempt
                print(f"      Semantic Scholar snippet search rate limited, waiting {wait_time}s...")
                time.sleep(wait_time)
                continue
            else:
                print(f"      Semantic Scholar snippet search failed: status {response.status_code}")
                return {"papers": [], "rate_limited": False}

        except requests.Timeout:
            if attempt < max_retries:
                continue
            return {"papers": [], "rate_limited": False}
        except Exception:
            return {"papers": [], "rate_limited": False}

    return {"papers": [], "rate_limited": True}


def get_specter_embedding(title: str, abstract: str) -> list[float]:
    """Generate SPECTER embedding for a paper/idea.

    Args:
        title: Paper or idea title
        abstract: Paper abstract or idea description

    Returns:
        768-dim embedding vector, or empty list on failure
    """
    url = "https://model-apis.semanticscholar.org/specter/v1/invoke"
    payload = [{
        "paper_id": "query",
        "title": title,
        "abstract": abstract[:1000]  # Truncate long abstracts
    }]

    try:
        response = requests.post(url, json=payload, timeout=30)
        if response.status_code == 200:
            data = response.json()
            preds = data.get('preds', [])
            if preds and 'embedding' in preds[0]:
                return preds[0]['embedding']
        return []
    except Exception:
        return []


def fetch_paper_embeddings_batch(paper_ids: list[str]) -> dict[str, list[float]]:
    """Fetch embeddings for papers via batch endpoint.

    Args:
        paper_ids: List of Semantic Scholar paper IDs

    Returns:
        Dict mapping paper_id to embedding vector
    """
    if not paper_ids:
        return {}

    url = "https://api.semanticscholar.org/graph/v1/paper/batch"
    headers = _get_semantic_scholar_headers()
    embeddings = {}

    # Batch into groups of 500 (API limit)
    batch_size = 500
    delay = get_config("search.academic_apis.semantic_scholar_delay", 0.5)
    if headers.get('x-api-key'):
        delay = min(delay, 0.05)

    for i in range(0, len(paper_ids), batch_size):
        batch = paper_ids[i:i+batch_size]
        batch_num = i//batch_size + 1
        max_retries = 3

        for attempt in range(max_retries):
            try:
                # Wait before request (initial delay or backoff delay)
                if attempt == 0:
                    time.sleep(delay)
                else:
                    # Exponential backoff: 1s, 2s, 4s
                    backoff_delay = 2 ** (attempt - 1)
                    print(f"        Semantic Scholar rate limited, waiting {backoff_delay}s... (batch {batch_num}, attempt {attempt + 1}/{max_retries})")
                    time.sleep(backoff_delay)

                response = requests.post(
                    url,
                    params={"fields": "embedding"},
                    json={"ids": batch},
                    headers=headers,
                    timeout=30
                )

                if response.status_code == 200:
                    data = response.json()
                    for j, paper in enumerate(data):
                        if paper and paper.get('embedding'):
                            embeddings[batch[j]] = paper['embedding'].get('vector', [])
                    break  # Success, exit retry loop
                elif response.status_code == 429:
                    # Rate limited - retry with backoff
                    if attempt == max_retries - 1:
                        print(f"        Warning: Semantic Scholar rate limited during embedding fetch (batch {batch_num}) - max retries exceeded")
                    # Otherwise continue to next retry attempt
                    continue
                elif response.status_code == 404:
                    print(f"        Warning: Some paper IDs not found in embedding fetch (batch {batch_num})")
                    break  # Don't retry 404s
                else:
                    print(f"        Warning: Embedding fetch failed with status {response.status_code} (batch {batch_num})")
                    break  # Don't retry other errors
            except requests.Timeout:
                print(f"        Warning: Embedding fetch timed out (batch {batch_num})")
                break  # Don't retry timeouts
            except Exception as e:
                print(f"        Warning: Embedding fetch error (batch {batch_num}): {type(e).__name__}")
                break  # Don't retry other exceptions

    # Log success rate
    success_rate = len(embeddings) / len(paper_ids) * 100 if paper_ids else 0
    if success_rate < 100:
        print(f"        Warning: Only fetched {len(embeddings)}/{len(paper_ids)} embeddings ({success_rate:.1f}%)")

    return embeddings


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Compute cosine similarity between two vectors."""
    if not a or not b or len(a) != len(b):
        return 0.0

    dot_product = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))

    if norm_a == 0 or norm_b == 0:
        return 0.0

    return dot_product / (norm_a * norm_b)


def query_arxiv(keywords: str, max_results: int = 5) -> list[dict]:
    """Query arXiv API for papers.

    Args:
        keywords: Search keywords
        max_results: Maximum number of results to return

    Returns:
        List of normalized paper dicts
    """
    url = "http://export.arxiv.org/api/query"
    params = {
        "search_query": f"all:{keywords}",
        "start": 0,
        "max_results": max_results,
        "sortBy": "relevance",
        "sortOrder": "descending"
    }

    timeout = get_config("search.academic_apis.timeout_seconds", 10)
    delay = get_config("search.academic_apis.arxiv_delay", 3.0)

    try:
        time.sleep(delay)  # arXiv recommends 3s delay
        response = requests.get(url, params=params, timeout=timeout)

        if response.status_code == 200:
            return _parse_arxiv_response(response.text)
        else:
            return []

    except Exception:
        return []


def query_crossref(keywords: str, max_results: int = 5) -> list[dict]:
    """Query CrossRef API for papers.

    Args:
        keywords: Search keywords
        max_results: Maximum number of results to return

    Returns:
        List of normalized paper dicts
    """
    url = "https://api.crossref.org/works"
    params = {
        "query": keywords,
        "rows": max_results,
        "sort": "relevance"
    }
    headers = {
        "User-Agent": "AnalogousReasoningAgent/1.0 (mailto:research@example.com)"
    }

    timeout = get_config("search.academic_apis.timeout_seconds", 10)
    delay = get_config("search.academic_apis.crossref_delay", 0.1)

    try:
        time.sleep(delay)
        response = requests.get(url, params=params, headers=headers, timeout=timeout)

        if response.status_code == 200:
            data = response.json()
            papers = []
            for item in data.get('message', {}).get('items', []):
                papers.append(_normalize_crossref_paper(item))
            return papers
        else:
            return []

    except Exception:
        return []


def _normalize_semantic_scholar_paper(paper: dict) -> dict:
    """Normalize Semantic Scholar paper to common schema."""
    authors = [author.get('name', '') for author in paper.get('authors', [])]

    # Extract GitHub URLs from externalIds if present
    external_ids = paper.get('externalIds', {})
    arxiv_id = external_ids.get('ArXiv', '')
    doi = external_ids.get('DOI', '')

    # Check for GitHub in externalIds (some papers have this!)
    github_urls = []
    if 'GitHub' in external_ids:
        github_urls.append(external_ids['GitHub'])

    return {
        'title': paper.get('title', ''),
        'authors': authors,
        'year': paper.get('year'),
        'abstract': paper.get('abstract', ''),
        'url': paper.get('url', ''),
        'doi': doi if doi else None,
        'arxiv_id': arxiv_id if arxiv_id else None,
        'citations': paper.get('citationCount'),
        'source_api': 'semantic_scholar',
        'github_urls': github_urls
    }


def _parse_arxiv_response(xml_text: str) -> list[dict]:
    """Parse arXiv API XML response."""
    import xml.etree.ElementTree as ET

    papers = []
    try:
        root = ET.fromstring(xml_text)
        ns = {'atom': 'http://www.w3.org/2005/Atom', 'arxiv': 'http://arxiv.org/schemas/atom'}

        for entry in root.findall('atom:entry', ns):
            title = entry.find('atom:title', ns)
            summary = entry.find('atom:summary', ns)
            published = entry.find('atom:published', ns)
            id_elem = entry.find('atom:id', ns)

            # Extract arXiv ID
            arxiv_id = ''
            if id_elem is not None:
                match = re.search(r'arxiv.org/abs/(\S+)', id_elem.text)
                if match:
                    arxiv_id = match.group(1)

            # Extract authors
            authors = []
            for author in entry.findall('atom:author', ns):
                name = author.find('atom:name', ns)
                if name is not None:
                    authors.append(name.text)

            # Extract year
            year = None
            if published is not None:
                try:
                    year = int(published.text[:4])
                except (ValueError, TypeError):
                    pass

            # Extract GitHub URLs from abstract
            abstract_text = summary.text if summary is not None else ''
            github_urls = extract_github_urls_from_text(abstract_text)

            papers.append({
                'title': title.text.strip() if title is not None else '',
                'authors': authors,
                'year': year,
                'abstract': abstract_text.strip(),
                'url': f'https://arxiv.org/abs/{arxiv_id}' if arxiv_id else '',
                'doi': None,
                'arxiv_id': arxiv_id,
                'citations': None,
                'source_api': 'arxiv',
                'github_urls': github_urls
            })
    except Exception:
        pass

    return papers


def _normalize_crossref_paper(item: dict) -> dict:
    """Normalize CrossRef paper to common schema."""
    # Extract authors
    authors = []
    for author in item.get('author', []):
        given = author.get('given', '')
        family = author.get('family', '')
        name = f"{given} {family}".strip()
        if name:
            authors.append(name)

    # Extract year
    year = None
    published = item.get('published-print') or item.get('published-online')
    if published and 'date-parts' in published:
        try:
            year = published['date-parts'][0][0]
        except (IndexError, TypeError):
            pass

    # Extract title
    titles = item.get('title', [])
    title = titles[0] if titles else ''

    # Extract abstract
    abstract = item.get('abstract', '')

    return {
        'title': title,
        'authors': authors,
        'year': year,
        'abstract': abstract,
        'url': item.get('URL', ''),
        'doi': item.get('DOI'),
        'arxiv_id': None,
        'citations': item.get('is-referenced-by-count'),
        'source_api': 'crossref',
        'github_urls': []
    }


def match_papers_to_concept(concept: dict, apis: list[str]) -> list[dict]:
    """Query multiple academic APIs and return ranked papers.

    Args:
        concept: Concept dict with search_keywords field
        apis: List of API names to query

    Returns:
        Ranked list of papers
    """
    keywords = concept.get('search_keywords', ' '.join(concept.get('key_concepts', [])))
    if isinstance(keywords, list):
        keywords = ' '.join(keywords)
    papers = []

    # Query APIs in parallel
    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = {}

        if 'semantic_scholar' in apis:
            max_results = get_config('search.academic_apis.papers_per_solution', 2) + 3
            futures[executor.submit(query_semantic_scholar, keywords, max_results)] = 'semantic_scholar'

        if 'arxiv' in apis:
            max_results = get_config('search.academic_apis.papers_per_solution', 2) + 3
            futures[executor.submit(query_arxiv, keywords, max_results)] = 'arxiv'

        if 'crossref' in apis:
            max_results = get_config('search.academic_apis.papers_per_solution', 2) + 3
            futures[executor.submit(query_crossref, keywords, max_results)] = 'crossref'

        # Collect results
        for future in as_completed(futures):
            try:
                papers.extend(future.result())
            except Exception:
                continue

    # Deduplicate
    papers = deduplicate_papers(papers)

    # Rank by relevance
    min_score = get_config('search.academic_apis.min_paper_relevance_score', 45)
    scored_papers = []
    for paper in papers:
        score = compute_relevance_score(paper, concept)
        if score >= min_score:
            paper['relevance_score'] = score
            scored_papers.append((paper, score))

    scored_papers.sort(key=lambda x: x[1], reverse=True)

    papers_per_solution = get_config('search.academic_apis.papers_per_solution', 2)
    return [p for p, score in scored_papers[:papers_per_solution]]


def compute_relevance_score(paper: dict, concept: dict) -> float:
    """Score paper relevance to concept (0-100).

    Scoring factors:
    - Keyword overlap (title + abstract vs concept keywords): 40 points
    - Citation count (normalized): 20 points
    - Recency (papers from last 5 years): 20 points
    - Abstract similarity to concept description: 20 points
    """
    score = 0.0

    # Keyword overlap
    keywords_str = concept.get('search_keywords', ' '.join(concept.get('key_concepts', [])))
    if isinstance(keywords_str, list):
        keywords_str = ' '.join(keywords_str)
    keywords = set(keywords_str.lower().split())
    paper_text = f"{paper['title']} {paper['abstract']}".lower()
    matches = sum(1 for kw in keywords if kw in paper_text)
    if keywords:
        score += (matches / len(keywords)) * 40

    # Citation count (log scale)
    if paper['citations']:
        score += min(20, math.log10(paper['citations'] + 1) * 5)

    # Recency
    if paper['year']:
        current_year = datetime.now().year
        if paper['year'] >= current_year - 5:
            score += 20
        elif paper['year'] >= current_year - 10:
            score += 10

    # Abstract similarity
    if paper['abstract'] and concept.get('description'):
        concept_words = set(concept['description'].lower().split())
        abstract_words = set(paper['abstract'].lower().split())
        overlap = len(concept_words & abstract_words) / max(len(concept_words), 1)
        score += overlap * 20

    return round(score, 2)


def deduplicate_papers(papers: list[dict]) -> list[dict]:
    """Remove duplicate papers by DOI or arXiv ID."""
    seen = set()
    unique_papers = []

    for paper in papers:
        # Create identifier
        identifier = None
        if paper['doi']:
            identifier = f"doi:{paper['doi']}"
        elif paper['arxiv_id']:
            identifier = f"arxiv:{paper['arxiv_id']}"
        else:
            identifier = f"title:{paper['title'].lower()}"

        if identifier not in seen:
            seen.add(identifier)
            unique_papers.append(paper)

    return unique_papers


def discover_github_repos_for_paper(paper: dict, concept: dict) -> tuple[list[dict], int]:
    """Discover GitHub repos for a paper using multiple strategies.

    Args:
        paper: Paper dict with metadata
        concept: Concept dict for additional context

    Returns:
        Tuple of (list of GitHub repo dicts, count of API calls made)
    """
    repos = []
    api_calls = 0

    # Strategy 1: Semantic Scholar metadata
    if paper.get('github_urls'):
        for url in paper['github_urls']:
            repo_data = _fetch_repo_from_url(url)
            api_calls += 1  # API call to fetch repo details
            if repo_data:
                repos.append(repo_data)

    # Strategy 2: Parse abstract for GitHub URLs
    if paper['abstract']:
        github_urls = extract_github_urls_from_text(paper['abstract'])
        for url in github_urls:
            if not any(r['url'] == url for r in repos):  # Avoid duplicates
                repo_data = _fetch_repo_from_url(url)
                api_calls += 1  # API call to fetch repo details
                if repo_data:
                    repos.append(repo_data)

    # Strategy 3: GitHub API search
    max_repos = get_config('search.academic_apis.repos_per_paper', 2)
    if len(repos) < max_repos:
        search_repos, search_api_calls = search_github_for_paper(paper, concept)
        api_calls += search_api_calls
        for repo in search_repos:
            if not any(r['url'] == repo['url'] for r in repos):
                repos.append(repo)
            if len(repos) >= max_repos:
                break

    # Rank and return top N
    ranked_repos = rank_repos_by_relevance(repos, paper, concept)
    return ranked_repos[:max_repos], api_calls


def extract_github_urls_from_text(text: str) -> list[str]:
    """Extract GitHub repository URLs from text."""
    # Match github.com/owner/repo patterns
    pattern = r'https?://github\.com/([a-zA-Z0-9_-]+)/([a-zA-Z0-9_.-]+)'
    matches = re.findall(pattern, text)

    urls = []
    for owner, repo in matches:
        # Clean repo name (remove .git, trailing punctuation)
        repo = repo.rstrip('.git').rstrip('.,;:)')
        urls.append(f'https://github.com/{owner}/{repo}')

    return list(set(urls))  # Deduplicate


def simplify_search_terms(terms: str | list, max_terms: int = 3) -> list[str]:
    """
    Simplify academic/technical terms to GitHub-friendly search keywords.

    Args:
        terms: String or list of search terms
        max_terms: Maximum number of simplified terms to return

    Returns:
        List of simplified search terms
    """
    # Academic filler words to remove
    academic_fluff = {
        'algorithm', 'method', 'approach', 'framework', 'technique', 'strategy',
        'novel', 'efficient', 'effective', 'robust', 'improved', 'enhanced',
        'based', 'using', 'via', 'for', 'with', 'and', 'or', 'the', 'a', 'an',
        'system', 'model', 'analysis', 'study', 'evaluation', 'application'
    }

    if isinstance(terms, list):
        terms = ' '.join(terms)

    # Split into words and filter
    words = terms.lower().split()
    core_terms = []

    for word in words:
        # Remove punctuation
        word = word.strip('.,;:()[]{}"\'-')
        # Keep if not in fluff list and has substance (length > 2)
        if word not in academic_fluff and len(word) > 2:
            core_terms.append(word)

    # Remove duplicates while preserving order
    seen = set()
    unique_terms = []
    for term in core_terms:
        if term not in seen:
            seen.add(term)
            unique_terms.append(term)

    return unique_terms[:max_terms]


def search_github_for_paper(paper: dict, concept: dict) -> tuple[list[dict], int]:
    """Search GitHub API for repos related to a paper.

    Returns:
        Tuple of (list of repo dicts, count of GitHub API calls made)
    """
    # Import from assessment.py to reuse
    from agents.assessment import fetch_repo_details

    # Get config settings
    min_stars = get_config('search.academic_apis.github_search.min_stars', 1)
    languages = get_config('search.academic_apis.github_search.languages', [])
    per_page = get_config('search.academic_apis.github_search.per_page', 15)
    max_queries = get_config('search.academic_apis.github_search.max_queries', 4)
    include_archived = get_config('search.academic_apis.github_search.include_archived', False)

    # Build search queries (in priority order - PAPER-SPECIFIC first, then concept fallbacks)
    queries = []

    # Strategy: Prioritize paper-specific queries to find repos for THIS paper, not just the concept
    # Priority: paper identifiers > paper title keywords > concept keywords

    # Query 1: ArXiv ID (most specific - repos often cite papers by arXiv ID)
    if paper.get('arxiv_id') and len(queries) < max_queries:
        queries.append(paper['arxiv_id'])

    # Query 2: Simplified paper title (2-3 core terms from paper title)
    if paper.get('title') and len(queries) < max_queries:
        simplified_title = simplify_search_terms(paper['title'], max_terms=3)
        if simplified_title and len(simplified_title) >= 2:
            queries.append(' '.join(simplified_title[:2]))

    # Query 3: Individual paper title keywords (try top 2 keywords separately)
    if paper.get('title') and len(queries) < max_queries:
        simplified_title = simplify_search_terms(paper['title'], max_terms=5)
        for term in simplified_title[:2]:  # Try top 2 individually
            if len(queries) < max_queries:
                queries.append(term)

    # Query 4-5: Paper authors + key term (if we have author info)
    if paper.get('authors') and len(paper['authors']) > 0 and len(queries) < max_queries:
        # Use last name of first author + simplified paper title keyword
        first_author = paper['authors'][0]
        last_name = first_author.split()[-1] if ' ' in first_author else first_author
        if paper.get('title'):
            simplified_title = simplify_search_terms(paper['title'], max_terms=1)
            if simplified_title:
                queries.append(f"{last_name} {simplified_title[0]}")

    # Fallback to concept-based queries if we need more
    # Query 6-7: Individual key concepts (broadest - each concept separately)
    if concept.get('key_concepts') and len(queries) < max_queries:
        simplified_concepts = simplify_search_terms(concept['key_concepts'], max_terms=5)
        for term in simplified_concepts[:2]:  # Try top 2 individually
            if len(queries) < max_queries:
                queries.append(term)

    # Query 8: Simplified key concepts combined (2 core terms)
    if concept.get('key_concepts') and len(queries) < max_queries:
        simplified = simplify_search_terms(concept['key_concepts'], max_terms=2)
        if simplified:
            queries.append(' '.join(simplified))

    # Query 9: Simplified search keywords combined
    if concept.get('search_keywords') and len(queries) < max_queries:
        simplified = simplify_search_terms(concept['search_keywords'], max_terms=2)
        if simplified:
            queries.append(' '.join(simplified))

    # Query 10: Domain + first core concept (broadest fallback)
    if concept.get('source_domain') and concept.get('key_concepts') and len(queries) < max_queries:
        domain_term = concept['source_domain'].lower().split()[0]  # First word of domain
        simplified = simplify_search_terms(concept['key_concepts'], max_terms=1)
        if simplified:
            queries.append(f"{domain_term} {simplified[0]}")

    # Build query filters
    filters = []
    if min_stars > 0:
        filters.append(f"stars:>{min_stars}")
    if languages:
        lang_query = ' OR '.join([f"language:{lang}" for lang in languages])
        filters.append(f"({lang_query})")
    if not include_archived:
        filters.append("archived:false")

    filter_str = ' '.join(filters)

    # Search GitHub
    repos = []
    api_calls = 0  # Track API calls
    headers = _get_github_headers()
    max_repos_target = get_config('search.academic_apis.repos_per_paper', 2) * 3  # Fetch extra for ranking

    print(f"      GitHub: Trying {len(queries)} query variation(s)...")
    paper_specific_query_count = 5  # Queries 1-5 are paper-specific, 6-10 are concept-level
    repos_per_paper_target = get_config('search.academic_apis.repos_per_paper', 2)

    for query_idx, query in enumerate(queries, 1):
        # Early stopping: if we have enough repos from paper-specific queries, skip concept queries
        if query_idx > paper_specific_query_count and len(repos) >= repos_per_paper_target:
            print(f"        Skipping concept-level queries (found {len(repos)} repos from paper-specific queries)")
            break

        try:
            url = "https://api.github.com/search/repositories"

            # Build full query string
            full_query = f"{query} {filter_str}".strip()

            params = {
                "q": full_query,
                "sort": "stars",
                "order": "desc",
                "per_page": per_page
            }

            response = requests.get(url, params=params, headers=headers, timeout=10)
            api_calls += 1  # Count search API call

            if response.status_code == 200:
                data = response.json()
                total_count = data.get('total_count', 0)
                items_count = len(data.get('items', []))
                print(f"        Query {query_idx}: '{query}' → {items_count} repos (total: {total_count})")
                for item in data.get('items', []):
                    # Stop if we have enough repos
                    if len(repos) >= max_repos_target:
                        break

                    repo_url = item['html_url']

                    # Skip if already found
                    if any(r['url'] == repo_url for r in repos):
                        continue

                    repo_data = fetch_repo_details(repo_url, headers)
                    api_calls += 1  # Count fetch_repo_details API call
                    if repo_data:
                        # Fetch README for relevance scoring
                        readme = _fetch_readme(item['full_name'], headers)
                        api_calls += 1  # Count README fetch API call
                        if readme:
                            repo_data['readme'] = readme
                        repos.append(repo_data)
            else:
                # Non-200 status code
                print(f"        Query {query_idx}: GitHub API returned status {response.status_code}")
                if response.status_code == 403:
                    print(f"        Rate limit may be exceeded. Headers: {response.headers.get('X-RateLimit-Remaining')}")

            # Stop if we have enough repos
            if len(repos) >= max_repos_target:
                break

        except Exception as e:
            # Log error but continue with other queries
            print(f"        Warning: GitHub search error for query '{query}': {e}")
            continue

    return repos, api_calls


def rank_repos_by_relevance(repos: list[dict], paper: dict, concept: dict) -> list[dict]:
    """Rank GitHub repos by relevance to paper and concept."""
    scored_repos = []

    for repo in repos:
        score = 0
        readme = repo.get('readme', '').lower()

        # Paper mention in README
        if paper['title'].lower() in readme:
            score += 50
        if paper.get('arxiv_id') and paper['arxiv_id'].lower() in readme:
            score += 40
        if paper.get('doi') and paper['doi'].lower() in readme:
            score += 40

        # Concept keywords in README
        for keyword in concept.get('key_concepts', []):
            if keyword.lower() in readme:
                score += 5

        # Stars (log scale)
        if repo.get('stars'):
            score += min(20, math.log10(repo['stars'] + 1) * 5)

        # Recency
        if repo.get('updated_at'):
            try:
                updated = datetime.fromisoformat(repo['updated_at'].replace('Z', '+00:00'))
                days_old = (datetime.now(updated.tzinfo) - updated).days
                if days_old <= 180:
                    score += 10
                elif days_old <= 365:
                    score += 5
            except (ValueError, TypeError):
                pass

        repo['relevance_score'] = score
        scored_repos.append((repo, score))

    scored_repos.sort(key=lambda x: x[1], reverse=True)

    # Filter by minimum relevance score
    min_repo_score = get_config('search.academic_apis.min_repo_relevance_score', 20)
    ranked_repos = [repo for repo, score in scored_repos if score >= min_repo_score]

    # Delete READMEs after scoring (not needed for assessment, saves tokens)
    for repo in ranked_repos:
        if 'readme' in repo:
            del repo['readme']

    return ranked_repos


def _fetch_repo_from_url(url: str) -> dict:
    """Fetch repo details from GitHub URL."""
    from agents.assessment import fetch_repo_details

    headers = _get_github_headers()
    return fetch_repo_details(url, headers)


def _fetch_readme(repo_full_name: str, headers: dict) -> str:
    """Fetch README content."""
    from agents.assessment import fetch_readme_content
    return fetch_readme_content(repo_full_name, headers)


def _get_github_headers() -> dict:
    """Get headers for GitHub API requests."""
    import os

    headers = {
        'Accept': 'application/vnd.github.v3+json'
    }

    # Add token if available
    github_token = os.getenv('GITHUB_TOKEN')
    if github_token:
        headers['Authorization'] = f'token {github_token}'

    return headers


def fetch_github_urls_from_paper_page(paper_urls: list[str]) -> list[str]:
    """
    Fetch paper pages and extract GitHub repository URLs from Code Availability sections.

    Only extracts URLs that appear to be the paper's own implementation,
    not cited repos or dependencies.

    Args:
        paper_urls: List of paper URLs (arxiv, nature, pmc, etc.)

    Returns:
        List of GitHub URLs found in Code Availability sections
    """
    import requests

    found_urls = []

    for url in paper_urls:
        try:
            # Convert arxiv abstract URLs to HTML version (has Code Availability section)
            fetch_url = url
            if 'arxiv.org/abs/' in url:
                fetch_url = url.replace('arxiv.org/abs/', 'arxiv.org/html/')
            elif 'arxiv.org/pdf/' in url:
                fetch_url = url.replace('arxiv.org/pdf/', 'arxiv.org/html/').replace('.pdf', '')

            # Fetch the page
            headers = {
                'User-Agent': 'Mozilla/5.0 (compatible; research bot)'
            }
            response = requests.get(fetch_url, headers=headers, timeout=15, allow_redirects=True)

            if response.status_code == 200:
                # Extract GitHub URLs specifically from Code Availability sections
                urls = _extract_code_availability_urls(response.text)

                # Filter out common non-repo URLs
                repo_urls = [u for u in urls if _is_likely_repo_url(u)]
                found_urls.extend(repo_urls)

        except Exception:
            continue

    # Deduplicate while preserving order
    seen = set()
    unique_urls = []
    for url in found_urls:
        if url not in seen:
            seen.add(url)
            unique_urls.append(url)

    return unique_urls


def _extract_code_availability_urls(html_content: str) -> list[str]:
    """
    Extract GitHub URLs specifically from Code Availability sections or
    near phrases indicating the paper's own implementation.

    Args:
        html_content: HTML content of the paper page

    Returns:
        List of GitHub URLs likely to be the paper's own code
    """
    import re

    found_urls = []

    # Patterns that indicate the paper's own code (case-insensitive)
    code_indicators = [
        r'code\s+(?:is\s+)?(?:available|accessible|released|published)\s+(?:at|on|via|from)',
        r'(?:source\s+)?code\s+(?:can\s+be\s+)?(?:found|accessed)\s+(?:at|on)',
        r'implementation\s+(?:is\s+)?(?:available|provided)\s+(?:at|on)',
        r'our\s+(?:source\s+)?code',
        r'code\s+availability',
        r'data\s+(?:and\s+)?code\s+availability',
        r'software\s+availability',
        r'open[- ]source\s+implementation',
    ]

    # GitHub URL pattern
    github_pattern = r'https?://github\.com/([a-zA-Z0-9_-]+)/([a-zA-Z0-9_.-]+)'

    # Strategy 1: Look for Code Availability section headers and extract URLs nearby
    # Common section patterns in HTML
    section_patterns = [
        r'(?:<h[1-6][^>]*>.*?(?:code|software|data)\s*(?:and\s*(?:code|data))?\s*availability.*?</h[1-6]>)(.*?)(?=<h[1-6]|$)',
        r'(?:<strong>.*?(?:code|software)\s*availability.*?</strong>)(.*?)(?=<strong>|<h[1-6]|$)',
        r'(?:<b>.*?(?:code|software)\s*availability.*?</b>)(.*?)(?=<b>|<h[1-6]|$)',
    ]

    for pattern in section_patterns:
        matches = re.findall(pattern, html_content, re.IGNORECASE | re.DOTALL)
        for section_content in matches:
            # Extract GitHub URLs from this section
            urls = re.findall(github_pattern, section_content)
            for owner, repo in urls:
                repo = repo.rstrip('.git').rstrip('.,;:)')
                found_urls.append(f'https://github.com/{owner}/{repo}')

    # Strategy 2: Look for code indicator phrases with nearby GitHub URLs
    # Search in a window around each indicator phrase
    for indicator in code_indicators:
        for match in re.finditer(indicator, html_content, re.IGNORECASE):
            # Get text window around the match (500 chars after)
            start = match.start()
            end = min(match.end() + 500, len(html_content))
            window = html_content[start:end]

            # Extract GitHub URLs from this window
            urls = re.findall(github_pattern, window)
            for owner, repo in urls:
                repo = repo.rstrip('.git').rstrip('.,;:)')
                found_urls.append(f'https://github.com/{owner}/{repo}')

    # Deduplicate while preserving order
    seen = set()
    unique_urls = []
    for url in found_urls:
        if url not in seen:
            seen.add(url)
            unique_urls.append(url)

    return unique_urls


def _is_likely_repo_url(url: str) -> bool:
    """Check if a GitHub URL is likely a repository (not a profile, topic, etc.)."""
    import re
    match = re.match(r'https://github\.com/([^/]+)/([^/]+)/?', url)
    if not match:
        return False

    owner, repo = match.groups()

    # Exclude known non-repo paths
    non_repo_owners = {'topics', 'features', 'explore', 'trending', 'collections',
                       'sponsors', 'settings', 'notifications', 'issues', 'pulls',
                       'marketplace', 'pricing', 'enterprise', 'team', 'about'}

    if owner.lower() in non_repo_owners:
        return False

    non_repo_paths = {'issues', 'pulls', 'actions', 'projects', 'wiki', 'security',
                      'pulse', 'graphs', 'network', 'settings', 'releases', 'tags'}

    if repo.lower() in non_repo_paths:
        return False

    return True
