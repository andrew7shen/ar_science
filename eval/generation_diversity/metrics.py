"""Diversity metrics for extraction agent analysis."""

from collections import defaultdict
from statistics import mean, stdev


def jaccard_similarity(set1: set, set2: set) -> float:
    """Compute Jaccard similarity between two sets."""
    if not set1 and not set2:
        return 1.0
    intersection = len(set1 & set2)
    union = len(set1 | set2)
    return intersection / union if union > 0 else 0.0


def compute_basic_metrics(runs: list) -> dict:
    """Compute basic frequency metrics across all runs."""
    all_domains = []
    domain_counts = defaultdict(int)
    domains_per_run = []

    for run in runs:
        domains = run["target_domains"]
        domains_per_run.append(len(domains))
        all_domains.extend(domains)
        for d in domains:
            domain_counts[d] += 1

    unique_domains = sorted(set(all_domains))
    return {
        "unique_domains": unique_domains,
        "domain_frequency": dict(sorted(domain_counts.items(), key=lambda x: -x[1])),
        "domains_per_run": {
            "mean": mean(domains_per_run) if domains_per_run else 0,
            "std": stdev(domains_per_run) if len(domains_per_run) > 1 else 0,
            "min": min(domains_per_run) if domains_per_run else 0,
            "max": max(domains_per_run) if domains_per_run else 0,
        },
    }


def compute_per_question_metrics(runs: list) -> dict:
    """Compute metrics per question including Jaccard similarity."""
    by_question = defaultdict(list)
    for run in runs:
        by_question[run["question"]].append(run)

    per_question = {}
    for question, q_runs in by_question.items():
        domain_sets = [set(r["target_domains"]) for r in q_runs]
        all_domains = set().union(*domain_sets) if domain_sets else set()
        domain_counts = defaultdict(int)
        total_domain_occurrences = 0
        for r in q_runs:
            for d in r["target_domains"]:
                domain_counts[d] += 1
                total_domain_occurrences += 1

        # Jaccard similarity matrix
        n = len(domain_sets)
        jaccard_values = []
        for i in range(n):
            for j in range(i + 1, n):
                jaccard_values.append(jaccard_similarity(domain_sets[i], domain_sets[j]))

        avg_domains_per_trial = total_domain_occurrences / len(q_runs) if q_runs else 0

        per_question[question] = {
            "trials": len(q_runs),
            "unique_domains": sorted(all_domains),
            "domain_counts": dict(sorted(domain_counts.items(), key=lambda x: -x[1])),
            "mean_jaccard": mean(jaccard_values) if jaccard_values else 1.0,
            "avg_domains_per_trial": avg_domains_per_trial,
        }

    return per_question


def compute_cross_question_metrics(runs: list, per_question: dict) -> dict:
    """Compute cross-question analysis: universal vs question-specific domains."""
    questions = list(per_question.keys())
    if not questions:
        return {"universal_domains": [], "question_specific": {}}

    domain_to_questions = defaultdict(set)
    for q, data in per_question.items():
        for d in data["unique_domains"]:
            domain_to_questions[d].add(q)

    universal = [d for d, qs in domain_to_questions.items() if len(qs) == len(questions)]
    question_specific = {}
    for q in questions:
        q_only = [d for d, qs in domain_to_questions.items() if qs == {q}]
        if q_only:
            question_specific[q] = sorted(q_only)

    return {
        "universal_domains": sorted(universal),
        "question_specific": question_specific,
    }


def compute_all_metrics(runs: list) -> dict:
    """Compute all diversity metrics."""
    basic = compute_basic_metrics(runs)
    per_question = compute_per_question_metrics(runs)
    cross_question = compute_cross_question_metrics(runs, per_question)

    return {
        **basic,
        "per_question": per_question,
        "cross_question": cross_question,
    }
