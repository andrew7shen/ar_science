"""
Paper verification module using Semantic Scholar and arXiv APIs.
"""

import os
import time
import re
import requests
import xml.etree.ElementTree as ET
from typing import List, Dict, Optional, Tuple
from datetime import datetime


def verify_papers(papers: List[Dict], config: dict) -> Tuple[List[Dict], float]:
    """Verify papers and enrich with metadata.

    Args:
        papers: List of papers from discovery
        config: Configuration dictionary

    Returns:
        Tuple of (verified_papers_list, runtime_seconds)
    """
    start_time = time.time()
    verified_papers = []

    academic_config = config.get("apis", {}).get("academic", {})
    semantic_scholar_config = academic_config.get("semantic_scholar", {})
    arxiv_config = academic_config.get("arxiv", {})

    for paper in papers:
        verified = verify_single_paper(
            paper,
            semantic_scholar_config,
            arxiv_config
        )
        if verified:
            verified_papers.append(verified)

    runtime = time.time() - start_time
    return verified_papers, runtime


def _is_editorial_content(title: str) -> bool:
    """Check if a paper title indicates editorial/non-research content.

    Args:
        title: Paper title to check

    Returns:
        True if title appears to be editorial content
    """
    editorial_patterns = [
        'in this issue',
        'editorial',
        'correction',
        'erratum',
        'retraction',
        'corrigendum',
        'addendum',
        'publisher note',
        'table of contents',
        'front matter',
        'back matter',
        'author index',
        'subject index',
        'preface',
        'introduction to special issue',
        'introduction to the special issue',
    ]

    title_lower = title.lower().strip()
    return any(pattern in title_lower for pattern in editorial_patterns)


def _is_review_or_survey(title: str, abstract: str) -> bool:
    """Check if a paper is a review/survey/tutorial rather than original research.

    Args:
        title: Paper title
        abstract: Paper abstract (can be None)

    Returns:
        True if paper appears to be a review/survey
    """
    review_keywords = [
        'review',
        'survey',
        'tutorial',
        'overview',
        'state of the art',
        'state-of-the-art',
        'literature review',
        'systematic review',
        'meta-analysis',
        'perspectives on',
        'trends in',
        'recent advances',
        'recent developments',
    ]

    # Check title
    title_lower = title.lower().strip()
    if any(keyword in title_lower for keyword in review_keywords):
        return True

    # Check abstract if available
    if abstract:
        abstract_lower = abstract.lower().strip()
        # Look for review indicators in abstract
        review_phrases = [
            'this review',
            'this survey',
            'we review',
            'we survey',
            'comprehensive review',
            'comprehensive survey',
            'systematic review',
        ]
        if any(phrase in abstract_lower for phrase in review_phrases):
            return True

    return False


def verify_single_paper(
    paper: Dict,
    semantic_scholar_config: dict,
    arxiv_config: dict
) -> Optional[Dict]:
    """Verify a single paper using academic APIs.

    Args:
        paper: Paper dictionary with title and url
        semantic_scholar_config: Semantic Scholar configuration
        arxiv_config: arXiv configuration

    Returns:
        Verified paper dictionary or None if verification fails
    """
    title = paper.get('title', '')
    url = paper.get('url', '')

    # Try Semantic Scholar first
    if semantic_scholar_config.get('enabled', True):
        verified = _verify_with_semantic_scholar(
            title, url, semantic_scholar_config
        )
        if verified:
            verified['verified'] = True
            verified['verified_at'] = datetime.now().isoformat()
            verified['analogy_description'] = paper.get('analogy_description', '')
            verified['discovered_by_template'] = paper.get('discovered_by_template')
            verified['discovered_at'] = paper.get('discovered_at')
            return verified

    # Fallback to arXiv
    if arxiv_config.get('enabled', True):
        verified = _verify_with_arxiv(
            title, url, arxiv_config
        )
        if verified:
            verified['verified'] = True
            verified['verified_at'] = datetime.now().isoformat()
            verified['analogy_description'] = paper.get('analogy_description', '')
            verified['discovered_by_template'] = paper.get('discovered_by_template')
            verified['discovered_at'] = paper.get('discovered_at')
            return verified

    return None


def _fetch_by_semantic_scholar_id(
    paper_id: str,
    headers: dict,
    rate_limit_delay: float
) -> Optional[Dict]:
    """Fetch paper directly by Semantic Scholar ID.

    Args:
        paper_id: Semantic Scholar paper ID (40-character hex string)
        headers: API headers with optional API key
        rate_limit_delay: Rate limit delay in seconds

    Returns:
        Normalized paper dictionary or None
    """
    max_retries = 3
    for attempt in range(max_retries):
        try:
            api_url = f"https://api.semanticscholar.org/graph/v1/paper/{paper_id}"
            params = {
                "fields": "title,authors,year,abstract,url,citationCount,externalIds,paperId"
            }

            response = requests.get(api_url, params=params, headers=headers, timeout=10)

            if response.status_code == 429:
                if attempt < max_retries - 1:
                    wait_time = (2 ** attempt) * rate_limit_delay
                    time.sleep(wait_time)
                    continue
                else:
                    return None

            if response.status_code == 200:
                paper = response.json()
                if paper:
                    # Skip editorial content
                    if _is_editorial_content(paper.get('title', '')):
                        return None

                    # Check if it's a review/survey
                    if _is_review_or_survey(paper.get('title', ''), paper.get('abstract', '')):
                        print(f"    Note: Paper is a review/survey, but using it since it was from loaded run")

                    return _normalize_semantic_scholar_paper(paper)

            # If non-429, non-200, break
            if response.status_code not in [429, 200]:
                break

        except (requests.exceptions.Timeout, requests.exceptions.ConnectionError):
            if attempt < max_retries - 1:
                wait_time = (2 ** attempt) * rate_limit_delay
                time.sleep(wait_time)
                continue
            else:
                return None
        except Exception:
            return None

    return None


def _verify_with_semantic_scholar(
    title: str,
    url: str,
    config: dict
) -> Optional[Dict]:
    """Verify paper with Semantic Scholar API.

    Args:
        title: Paper title
        url: Paper URL or DOI
        config: Semantic Scholar configuration

    Returns:
        Verified paper dictionary or None
    """
    api_key = os.getenv('SEMANTIC_SCHOLAR_API_KEY')
    headers = {}
    if api_key:
        headers['x-api-key'] = api_key

    rate_limit_delay = config.get('rate_limit_delay', 0.5)
    if api_key:
        rate_limit_delay = min(rate_limit_delay, 0.05)

    time.sleep(rate_limit_delay)

    # Try fetching by Semantic Scholar ID if URL contains one
    ss_id_match = re.search(r'semanticscholar\.org/paper/([a-f0-9]{40})', url)
    if ss_id_match:
        paper_id = ss_id_match.group(1)
        direct_fetch = _fetch_by_semantic_scholar_id(paper_id, headers, rate_limit_delay)
        if direct_fetch:
            return direct_fetch

    # Try searching by title with retry logic
    max_retries = 3
    for attempt in range(max_retries):
        try:
            api_url = "https://api.semanticscholar.org/graph/v1/paper/search"
            params = {
                "query": title,
                "limit": 5,
                "fields": "title,authors,year,abstract,url,citationCount,externalIds,paperId"
            }

            response = requests.get(api_url, params=params, headers=headers, timeout=10)

            # Handle rate limiting
            if response.status_code == 429:
                if attempt < max_retries - 1:
                    wait_time = (2 ** attempt) * rate_limit_delay  # Exponential backoff
                    time.sleep(wait_time)
                    continue
                else:
                    return None

            if response.status_code == 200:
                data = response.json()
                papers = data.get('data', [])

                if papers:
                    # Filter out editorial content and reviews
                    candidate_papers = []
                    filtered_reviews = []
                    for paper in papers:
                        paper_title = paper.get('title', '')
                        paper_abstract = paper.get('abstract', '')

                        # Skip editorial content
                        if _is_editorial_content(paper_title):
                            continue

                        # Skip reviews/surveys
                        if _is_review_or_survey(paper_title, paper_abstract):
                            filtered_reviews.append(f"{paper_title} ({paper.get('year', 'N/A')})")
                            continue

                        candidate_papers.append(paper)

                    # Log filtered reviews if any
                    if filtered_reviews:
                        print(f"    Filtered {len(filtered_reviews)} review/survey paper(s)")

                    # If no non-review papers found, fall back to all papers (excluding editorial)
                    if not candidate_papers:
                        print(f"    Warning: Only reviews found, using best review paper")
                        candidate_papers = [p for p in papers if not _is_editorial_content(p.get('title', ''))]

                    if candidate_papers:
                        # Take first non-review, non-editorial paper (no sorting to avoid bias)
                        best_paper = candidate_papers[0]

                        # Log selected paper
                        if len(candidate_papers) > 1:
                            print(f"    Selected first of {len(candidate_papers)} candidates: {best_paper.get('title', 'Unknown')} ({best_paper.get('year', 'N/A')}, {best_paper.get('citationCount', 0):,} citations)")
                        else:
                            print(f"    Selected: {best_paper.get('title', 'Unknown')} ({best_paper.get('year', 'N/A')}, {best_paper.get('citationCount', 0):,} citations)")

                        return _normalize_semantic_scholar_paper(best_paper)

            # If we got here with a non-429, non-200 status, break out of retry loop
            if response.status_code not in [429, 200]:
                break

        except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as e:
            # Retry on network errors
            if attempt < max_retries - 1:
                wait_time = (2 ** attempt) * rate_limit_delay
                time.sleep(wait_time)
                continue
            else:
                return None
        except Exception:
            # For other exceptions, don't retry
            return None

    return None


def _verify_with_arxiv(
    title: str,
    url: str,
    config: dict
) -> Optional[Dict]:
    """Verify paper with arXiv API.

    Args:
        title: Paper title
        url: Paper URL or arXiv ID
        config: arXiv configuration

    Returns:
        Verified paper dictionary or None
    """
    rate_limit_delay = config.get('rate_limit_delay', 3.0)
    time.sleep(rate_limit_delay)

    # Extract arXiv ID from URL if present
    arxiv_id = None
    if 'arxiv.org' in url:
        match = re.search(r'(\d+\.\d+)', url)
        if match:
            arxiv_id = match.group(1)

    try:
        api_url = "http://export.arxiv.org/api/query"

        if arxiv_id:
            params = {
                "id_list": arxiv_id,
                "max_results": 1
            }
        else:
            params = {
                "search_query": f"ti:{title}",
                "max_results": 1,
                "sortBy": "relevance"
            }

        response = requests.get(api_url, params=params, timeout=10)

        if response.status_code == 200:
            papers = _parse_arxiv_response(response.text)
            if papers:
                return papers[0]

    except Exception:
        pass

    return None


def _normalize_semantic_scholar_paper(paper: dict) -> dict:
    """Normalize Semantic Scholar paper to common schema."""
    authors = [author.get('name', '') for author in paper.get('authors', [])]

    external_ids = paper.get('externalIds', {})
    arxiv_id = external_ids.get('ArXiv', '')
    doi = external_ids.get('DOI', '')
    openalex_id = external_ids.get('OpenAlex', '')

    return {
        'title': paper.get('title', ''),
        'authors': authors,
        'year': paper.get('year'),
        'abstract': paper.get('abstract', ''),
        'url': paper.get('url', ''),
        'doi': doi if doi else '',
        'arxiv_id': arxiv_id if arxiv_id else '',
        'citation_count': paper.get('citationCount', 0),
        'source_api': 'semantic_scholar',
        's2_paper_id': paper.get('paperId', ''),
        'openalex_id': openalex_id if openalex_id else ''
    }


def _parse_arxiv_response(xml_text: str) -> List[Dict]:
    """Parse arXiv API XML response."""
    papers = []

    try:
        root = ET.fromstring(xml_text)
        ns = {'atom': 'http://www.w3.org/2005/Atom'}

        for entry in root.findall('atom:entry', ns):
            title_elem = entry.find('atom:title', ns)
            summary_elem = entry.find('atom:summary', ns)
            published_elem = entry.find('atom:published', ns)
            id_elem = entry.find('atom:id', ns)

            # Extract arXiv ID
            arxiv_id = ''
            if id_elem is not None and id_elem.text:
                match = re.search(r'(\d+\.\d+)', id_elem.text)
                if match:
                    arxiv_id = match.group(1)

            # Extract authors
            authors = []
            for author in entry.findall('atom:author', ns):
                name_elem = author.find('atom:name', ns)
                if name_elem is not None and name_elem.text:
                    authors.append(name_elem.text.strip())

            # Extract year
            year = None
            if published_elem is not None and published_elem.text:
                try:
                    year = int(published_elem.text[:4])
                except:
                    pass

            title = title_elem.text.strip() if title_elem is not None else ''
            abstract = summary_elem.text.strip() if summary_elem is not None else ''

            papers.append({
                'title': title,
                'authors': authors,
                'year': year,
                'abstract': abstract,
                'url': f'https://arxiv.org/abs/{arxiv_id}' if arxiv_id else '',
                'doi': '',
                'arxiv_id': arxiv_id,
                'citation_count': None,
                'source_api': 'arxiv',
                's2_paper_id': '',
                'openalex_id': ''
            })

    except Exception:
        pass

    return papers
