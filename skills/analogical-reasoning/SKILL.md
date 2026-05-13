---
name: analogical-reasoning
description: Use analogical reasoning to find real, existing solutions to a problem by mapping it to structurally analogous problems in other research domains.
---

# Analogical Reasoning

AR is two prompts. Run them in order and return the union of solutions.

Defaults: `num_domains=3`, `num_solutions=3` per domain, `min_key_terms=3`, `max_key_terms=5`. Override only if the user asks.

## Getting the problem

If the user invoked the skill with the problem on the same line (e.g. `/analogical-reasoning How can we make machine learning models more sample-efficient?`), use everything after the slash command as `{{problem_text}}`. If they typed `/analogical-reasoning` alone with no problem, ask once: "What problem would you like AR to analyze?"

## Step 1 — extraction

Send this prompt to the LLM, substituting `{{problem_text}}` with the user's problem (and the other placeholders with the defaults above). Parse the response as JSON.

````
You are analyzing a scientific research problem to find analogous solutions in other domains through explicit object-to-object mappings.

**Problem:**
{{problem_text}}

**Your task:**
Extract the problem's key objects and relations, then generate analogies to {{num_domains}} other domains with explicit object mappings.

**Return a JSON object with:**

1. "problem_summary": A concise 1-2 sentence summary
2. "problem_objects": Array of key objects/entities with their functional roles
3. "problem_relations": Array of core relational structures between objects
4. "analogies": Array of {{num_domains}} analogies, each with:
   - "target_domain": The domain name (e.g., "computer_science", "logistics")
   - "analogy_title": A descriptive title for this analogy
   - "object_mappings": Array of source-to-target mappings with rationale
   - "shared_relations": The relational structure preserved across domains
5. "key_terms": Array of {{min_key_terms}}-{{max_key_terms}} important terms/concepts
6. "target_domains": Array of the {{num_domains}} domain names (from analogies)

**Map objects by FUNCTION, not surface similarity.** "Delivers payload" is a good mapping basis; "is liquid" is not.

**Example format:**
```json
{
  "problem_summary": "Brief description of the scientific problem",
  "problem_objects": [
    {"name": "object_A", "role": "functional role of A"},
    {"name": "object_B", "role": "functional role of B"}
  ],
  "problem_relations": [
    "key relation between objects"
  ],
  "analogies": [
    {
      "target_domain": "domain_name",
      "analogy_title": "Title describing the analogy",
      "object_mappings": [
        {"source": "object_A", "target": "analogous_A", "mapping_rationale": "why these map functionally"},
        {"source": "object_B", "target": "analogous_B", "mapping_rationale": "why these map functionally"}
      ],
      "shared_relations": "the relational structure preserved in this domain"
    }
  ],
  "key_terms": ["term1", "term2", "term3"],
  "target_domains": ["domain_name"]
}
```

Return ONLY the JSON object, no other text.
````

## Step 2 — search (run once per analogy)

For each `analogy` in `extraction["analogies"]`, send the prompt below to the LLM with these substitutions:

- `{{domain}}` ← `analogy["target_domain"]`
- `{{problem_summary}}` ← `extraction["problem_summary"]`
- `{{analogy_title}}` ← `analogy["analogy_title"]`
- `{{object_mappings}}` ← `analogy["object_mappings"]`, joined as one line per mapping: `- source → target (mapping_rationale)`
- `{{shared_relations}}` ← `analogy["shared_relations"]`
- `{{key_terms}}` ← `extraction["key_terms"]` joined with `, `
- `{{num_solutions}}` ← `3` (override only if the user asks)

Parse each response as a JSON array.

````
You are finding solutions from {{domain}} that could be applied to a scientific problem through analogical reasoning.

**Problem:**
{{problem_summary}}

**Analogy to {{domain}}:**
{{analogy_title}}

**Object Mappings (source → {{domain}}):**
{{object_mappings}}

**Shared Relational Structure:**
{{shared_relations}}

**Key Concepts:**
{{key_terms}}

**IMPORTANT:** Use the object mappings above to guide your search. Find solutions in {{domain}} that work with the TARGET objects (right side of mappings) and preserve the shared relational structure. Do NOT search for direct solutions in the source problem's own domain — the whole point is to find them elsewhere.

**OUTPUT CONCISENESS:**
- Keep descriptions to 2-3 sentences, focus on core algorithm/method only
- Avoid verbose background - provide only technical essentials for analogical reasoning

**Your task:**
Find {{num_solutions}} real, existing solutions, algorithms, or methods in {{domain}} that address the shared relational structure. Look for published research, documented algorithms, or established methods.

**CRITICAL - Citation Accuracy Requirements:**
- Each solution MUST be based on a SPECIFIC paper, article, or documented method
- The cited source MUST directly describe the specific solution/algorithm/method you're reporting
- DO NOT cite general surveys or overview papers unless they specifically describe the method in detail
- DO NOT cite a source that only mentions the domain or related concepts - the source must describe THIS SPECIFIC SOLUTION
- Each solution should have DIFFERENT primary sources - if you're citing the same paper for multiple solutions, you're doing it wrong
- When in doubt, find MORE SPECIFIC papers that directly describe the method
- **CRITICAL:** You MUST provide the COMPLETE, EXACT TITLE of each paper/source you cite
- The paper title must match exactly what appears on the paper's webpage - do not truncate, abbreviate, or approximate titles

**Code Implementation Discovery:**
For each solution, try to find GitHub repositories with code implementations:
- Search for "[solution name] github implementation"
- Search for "[paper title] code"
- Check paper pages for "Code Availability" sections
- Look for official implementations or well-maintained research codebases
- If you cannot find verified GitHub repos, leave the github_repos array empty

For each solution found, return:
1. "title": Descriptive title of the solution/algorithm
2. "source_domain": "{{domain}}" (MUST be this exact domain name)
3. "description": 2-3 sentence explanation with specifics
4. "key_concepts": 3-5 core techniques/concepts
5. "relevance": How this solution addresses the shared relational structure and could transfer back to the source problem
6. "sources": URLs or citations you found
7. "source_titles": EXACT titles of papers/articles at each source URL (must match order of sources array)
8. "github_repos": Array of GitHub repositories found (can be empty if none found)

**CRITICAL:** After completing your research, return ONLY the JSON array with NO additional text, explanation, or commentary. Do not write "Based on my research" or any other introduction. Start your response directly with the JSON array.

Format:
```json
[
  {
    "title": "Solution name",
    "source_domain": "{{domain}}",
    "description": "Detailed explanation...",
    "key_concepts": ["concept1", "concept2", "concept3"],
    "relevance": "How this addresses the shared relational structure...",
    "sources": ["url1", "url2"],
    "source_titles": ["Exact Paper Title 1", "Exact Paper Title 2"],
    "github_repos": [
      {
        "url": "https://github.com/owner/repo",
        "source": "paper"
      }
    ]
  }
]
```

Return ONLY the JSON array, nothing else.
````

## Output

Concatenate the JSON arrays returned by every search call — the union of solutions across all analogies is the AR output.

For presentation, render the results as readable prose grouped by `source_domain`: show each solution's title, 2-3 sentence description, key concepts, and citation (with URL if available). Do not dump the raw JSON to the user — the JSON is for parsing; the chat surface should be human-friendly.

**Citation note for the user:** AR draws citations from the LLM's training data and may occasionally generate plausible-but-inaccurate references. Encourage the user to verify any source before relying on it.

## Saving the raw JSON (optional)

If the user invokes the skill with `--save` (optionally followed by a path), or otherwise explicitly asks to save the JSON, write the full run to disk using the Write tool. Otherwise, do not save anything.

**What to write** — a single JSON object with these top-level keys:

- `"problem"`: the user's problem text (string).
- `"extraction"`: the complete Step 1 JSON object, verbatim — includes `problem_summary`, `problem_objects`, `problem_relations`, `analogies` (each with `target_domain`, `analogy_title`, `object_mappings` containing `mapping_rationale`, and `shared_relations`), `key_terms`, and `target_domains`.
- `"solutions"`: the concatenated array of all Step 2 results across analogies, verbatim — each item has `title`, `source_domain`, `description`, `key_concepts`, `relevance`, `sources`, `source_titles`, and `github_repos`.

This is a strict superset of what is rendered in chat — every analogy, mapping rationale, solution description, relevance justification, citation, and GitHub repo discovered during the run is preserved.

**Path resolution:**

- If the user provided a path with `--save <path>` or named one when asking to save, use it.
- Otherwise, default to `ar_output/<YYYYMMDD-HHMMSS>_<problem>.json` in the current working directory, where `<problem>` is the first 5 words of the problem, lowercased, with non-alphanumeric runs replaced by `-`. Create `ar_output/` if it doesn't exist.

After writing, tell the user the absolute path on a single line, then continue with the normal prose rendering described in "Output" above.

---

From "Unlocking LLM Creativity in Science through Analogical Reasoning" — see the repo's top-level README for the paper link.
