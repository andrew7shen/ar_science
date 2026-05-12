# AR Dataset

`data/dataset.json` contains 266 examples of analogical reasoning (AR) in
science: scientific papers that solve a problem in one domain by adapting
concepts, structures, or methods from a distant source domain. For each
example, the dataset includes paper metadata, the extracted analogy (problem,
base/target domains, justifications, a concrete instance), and classification
labels with rationales (domain distance, usage depth, difficulty, etc.).

Papers were pulled from Semantic Scholar and arXiv between **2026-01-13 and
2026-01-14**. `citation_count` reflects values as of the pull date.

Built by a four-stage pipeline in `code/`: **discovery** (cross-domain templates
surface candidates) → **verification** (Semantic Scholar / arXiv lookup) →
**extraction** (Claude pulls out the analogy's structural components) →
**difficulty assessment** (Claude rates the creative leap).

## Field reference

### Paper metadata
- `title` — Paper title.
- `authors` — Author names.
- `year` — Publication year.
- `abstract` — Paper abstract, or `null`.
- `url` — Link to the paper.
- `doi` — DOI, or empty.
- `arxiv_id` — arXiv ID, or empty.
- `citation_count` — Citations at time of pull.
- `source_api` — `semantic_scholar` or `arxiv`.
- `s2_paper_id` — Semantic Scholar ID, or empty.
- `is_original_paper` — `true` if this paper introduces the analogy, `false` for reviews/applications.
- `original_paper_info` — Author/year of the originating paper if `is_original_paper` is `false`.

### Analogy core
- `problem` — Problem the paper solves with analogical reasoning.
- `rewritten_problem` — `problem` stripped of any hints from the base (analogous) domain; used as the shared input across all three settings (no-domain, cross-domain, AR) for evaluation tasks.
- `method_name` — Short label for the method.
- `base_domain` — Domain the analogous solution comes from.
- `target_domain` — Domain the problem originates in.
- `base_domain_justification` — Description of the source domain.
- `target_domain_justification` — Description of the target domain.
- `analogy_description` — One-line summary of the cross-domain mapping.
- `analogy_justification` — Why the analogy holds.
- `concrete_example` — Specific worked example from the paper.

### Classification
- `domain_distance` — Distance between `base_domain` (analogous) and `target_domain` (problem). One of `moderate` or `distant`.
- `domain_distance_justification` — Reason for the distance rating.
- `analogy_usage_depth` — `methodological_adaptation` (borrows techniques) or `deep_structural_transfer` (transfers core framework).
- `analogy_usage_justification` — Reason for the depth rating.
- `requires_structural_reasoning` — `true` if recognizing the analogy needs mechanistic insight, not just surface similarity between the domains.
- `structural_reasoning_justification` — Reason for the structural-reasoning flag.
- `likely_well_known_example` — `true` if the analogy is a famous, widely-cited cross-domain application that appears frequently in academic literature or textbooks.
- `well_known_justification` — Reason for the well-known flag.
- `difficulty` — Difficulty of the cross-domain analogical leap. One of `easy`, `medium`, or `hard`.
- `difficulty_reasoning` — Reason for the difficulty rating.
