"""Minimal runner for the Analogical Reasoning prompts.

Reads extraction.txt and search.txt from this directory, sends them to an LLM,
and returns the union of solutions across all analogies.

API keys are loaded exclusively from a `.env` file at the project root. Copy
`.env.example` to `.env` and fill in the keys for whichever provider(s) you
plan to use. The `.env` file is gitignored — do NOT commit it.

Defaults to Anthropic (Claude). To switch LLM providers, change the
`PROVIDER` constant in the "Choose LLM provider" block below to "openai"
or "gemini".
"""

import json
import os
from pathlib import Path

# === Load API keys from .env =================================================
# TODO: Before running this script, copy `.env.example` to `.env` and fill in
#       the API key for the provider you plan to use.
# Reads `<repo-root>/.env` and populates os.environ with its key=value pairs.
# `.env` is gitignored and must NOT be committed — it holds your private keys.

def _load_dotenv(path: Path) -> None:
    if not path.exists():
        return
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        key = key.strip()
        val = val.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = val

_load_dotenv(Path(__file__).parent.parent / ".env")
# =============================================================================

# === Choose LLM provider =====================================================
# Set PROVIDER to one of: "anthropic", "openai", "gemini".
# Only the SDK for the chosen provider needs to be installed.
#
# Models tested in the paper: claude-sonnet-4-5, gpt-5.2, gemini-3-flash-preview.
# Other models may work but have NOT been tested.

PROVIDER = "anthropic"  # TODO: set to "anthropic", "openai", or "gemini"

if PROVIDER == "anthropic":
    # `pip install anthropic`
    from anthropic import Anthropic

    client = Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

    def _ask(prompt: str) -> str:
        msg = client.messages.create(
            model="claude-sonnet-4-5",
            max_tokens=16000,
            messages=[{"role": "user", "content": prompt}],
        )
        if not msg.content:
            raise RuntimeError(
                f"Empty response from Claude (stop_reason={msg.stop_reason}, usage={msg.usage})"
            )
        for block in msg.content:
            if hasattr(block, "text"):
                return block.text
        raise RuntimeError(f"No text block in Claude response: {msg.content!r}")

elif PROVIDER == "openai":
    # `pip install openai`
    from openai import OpenAI

    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

    def _ask(prompt: str) -> str:
        response = client.chat.completions.create(
            model="gpt-5.2",
            max_completion_tokens=16000,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.choices[0].message.content

elif PROVIDER == "gemini":
    # `pip install google-genai`
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=os.environ.get("GOOGLE_API_KEY"))

    def _ask(prompt: str) -> str:
        response = client.models.generate_content(
            model="gemini-3-flash-preview",
            contents=prompt,
            config=types.GenerateContentConfig(max_output_tokens=16000),
        )
        return response.text

else:
    raise ValueError(
        f"Unknown PROVIDER: {PROVIDER!r}. Use 'anthropic', 'openai', or 'gemini'."
    )
# =============================================================================

_HERE = Path(__file__).parent
EXTRACTION_TMPL = (_HERE / "extraction.txt").read_text()
SEARCH_TMPL = (_HERE / "search.txt").read_text()


def _fill(tmpl: str, **kwargs) -> str:
    for k, v in kwargs.items():
        tmpl = tmpl.replace("{{" + k + "}}", str(v))
    return tmpl


def _parse_json(text: str):
    """Parse JSON from an LLM response, stripping ```json ... ``` fences if present."""
    text = text.strip()
    if text.startswith("```"):
        first_newline = text.find("\n")
        if first_newline != -1:
            text = text[first_newline + 1:]
        if text.endswith("```"):
            text = text[:-3]
        text = text.strip()
    return json.loads(text)


def _ask_json(prompt: str, max_retries: int = 2):
    """Call the LLM and parse the response as JSON, retrying on parse or API failures."""
    last_err = None
    for attempt in range(max_retries + 1):
        try:
            return _parse_json(_ask(prompt))
        except Exception as e:
            last_err = e
            if attempt < max_retries:
                print(f"          retry {attempt + 1}/{max_retries} ({e.__class__.__name__}: {str(e)[:80]})")
    raise last_err


def analogical_reasoning(problem: str) -> dict:
    print(f"[1/2] Extracting analogies...")
    extraction = _ask_json(_fill(
        EXTRACTION_TMPL,
        problem_text=problem,
        num_domains=3,
        min_key_terms=3,
        max_key_terms=5,
    ))
    domains = [a["target_domain"] for a in extraction["analogies"]]
    print(f"      domains: {', '.join(domains)}")

    print(f"[2/2] Searching each domain for solutions...")
    solutions = []
    for i, analogy in enumerate(extraction["analogies"], 1):
        print(f"      [{i}/{len(extraction['analogies'])}] {analogy['target_domain']}...")
        mappings = "\n".join(
            f"- {m['source']} → {m['target']} ({m['mapping_rationale']})"
            for m in analogy["object_mappings"]
        )
        try:
            solutions.extend(_ask_json(_fill(
                SEARCH_TMPL,
                domain=analogy["target_domain"],
                problem_summary=extraction["problem_summary"],
                analogy_title=analogy["analogy_title"],
                object_mappings=mappings,
                shared_relations=analogy["shared_relations"],
                key_terms=", ".join(extraction["key_terms"]),
                num_solutions=3,
            )))
        except Exception as e:
            print(f"          skipped ({e.__class__.__name__}: {e})")
            continue
    return {"problem": problem, "extraction": extraction, "solutions": solutions}


def print_solutions(solutions: list) -> None:
    by_domain: dict = {}
    for s in solutions:
        by_domain.setdefault(s["source_domain"], []).append(s)
    for domain, sols in by_domain.items():
        print(f"\n=== {domain} ===\n")
        for i, s in enumerate(sols, 1):
            print(f"{i}. {s['title']}")
            print(f"   {s['description']}")
            if s.get("key_concepts"):
                print(f"   Key concepts: {', '.join(s['key_concepts'])}")
            titles = s.get("source_titles") or []
            urls = s.get("sources") or []
            # Some model outputs return a single string instead of a list — normalize.
            if isinstance(titles, str):
                titles = [titles]
            if isinstance(urls, str):
                urls = [urls]
            for j, title in enumerate(titles):
                url = urls[j] if j < len(urls) else ""
                print(f"   Source: {title}" + (f" — {url}" if url else ""))
            print()


if __name__ == "__main__":
    import argparse
    import re
    from datetime import datetime

    parser = argparse.ArgumentParser(
        description="Run analogical reasoning on a scientific problem.",
        usage='python run_ar.py "<your problem>" [--save [PATH]]',
        epilog='Example: python run_ar.py "How can we make machine learning models more sample-efficient?" --save',
    )
    parser.add_argument("problem", nargs="+", help="The scientific problem text.")
    parser.add_argument(
        "--save",
        nargs="?",
        const=True,
        default=None,
        metavar="PATH",
        help=(
            "Save the full run (problem + extraction + solutions) as JSON. "
            "If PATH is omitted, defaults to ar_output/<YYYYMMDD-HHMMSS>_<problem>.json. "
            "Place this flag AFTER the problem text to avoid argument ambiguity."
        ),
    )
    args = parser.parse_args()

    problem = " ".join(args.problem)
    result = analogical_reasoning(problem)
    print_solutions(result["solutions"])

    if args.save is not None:
        if args.save is True:
            slug_words = re.findall(r"\w+", problem.lower())[:5]
            slug = "-".join(slug_words) or "ar-run"
            ts = datetime.now().strftime("%Y%m%d-%H%M%S")
            out_path = Path("ar_output") / f"{ts}_{slug}.json"
        else:
            out_path = Path(args.save)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(result, indent=2))
        print(f"\nSaved JSON → {out_path.resolve()}")
