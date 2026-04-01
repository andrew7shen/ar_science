#!/usr/bin/env python3
"""
Analogous Reasoning Multi-Agent System
Ultra-minimal MVP for cross-domain solution transfer

Run with: python src/main.py "How can we optimize drug delivery to tumors?"
"""

import sys
import argparse
from pathlib import Path
from dotenv import load_dotenv

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

# Load environment variables
load_dotenv()

from orchestrator import run_workflow
import db
from config import config


def main():
    parser = argparse.ArgumentParser(
        description="Analogous Reasoning System - Find cross-domain solutions"
    )
    parser.add_argument(
        "problem",
        nargs="?",
        help="Biomedical problem to solve"
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List past workflows"
    )
    parser.add_argument(
        "--config",
        type=str,
        help="Path to custom config file (default: config.yaml in project root)"
    )
    parser.add_argument(
        "--abstraction",
        choices=["concrete", "conceptual", "mathematical"],
        help="Abstraction level to use for search (overrides config default)"
    )

    args = parser.parse_args()

    # Load config
    if args.config:
        config.load(args.config)
    else:
        config.load()

    # Get abstraction level from args or config
    if args.abstraction:
        abstraction_level = args.abstraction
    else:
        abstraction_level = config.get("search.default_abstraction_level", "conceptual")

    # Initialize database
    db.init_database()

    if args.list:
        # List past workflows
        workflows = db.list_workflows()
        if not workflows:
            print("No workflows found.")
            return

        print("\nPast Workflows:")
        print("-" * 80)
        for wf in workflows:
            print(f"{wf['id'][:8]}  {wf['created_at'][:10]}  {wf['problem_text'][:50]}  {wf['status']}")
        print("-" * 80)
        return

    if not args.problem:
        parser.print_help()
        return

    # Run complete workflow
    try:
        state = run_workflow(args.problem, abstraction_level=abstraction_level)

        if state.get("selected_solution"):
            print("✓ Workflow completed successfully!")
        else:
            print("Workflow cancelled or incomplete.")

    except Exception as e:
        print(f"\n✗ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
