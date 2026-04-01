"""
Data schema for analogical reasoning paper dataset.
"""

from dataclasses import dataclass, asdict
from typing import List, Dict, Optional
from enum import Enum


class DifficultyLevel(Enum):
    """Difficulty levels for analogical reasoning."""
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"


@dataclass
class AnalogicalReasoningExample:
    """Single example of a paper using analogical reasoning."""

    # Paper metadata
    paper_title: str
    paper_source: str  # DOI or URL
    authors: List[str]
    year: int
    citation_count: int

    # Analogy components
    problem: str  # The problem being solved
    base_domain: str  # Source domain of the analogy
    target_domain: str  # Target domain where analogy is applied
    analogy_justification: str  # Why/how the analogy works

    # Difficulty assessment
    difficulty: DifficultyLevel
    difficulty_reasoning: str  # Explanation for difficulty assignment

    # Optional metadata
    abstract: Optional[str] = None

    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        data = asdict(self)
        data['difficulty'] = self.difficulty.value
        return data

    @classmethod
    def from_dict(cls, data: Dict) -> 'AnalogicalReasoningExample':
        """Create instance from dictionary."""
        if isinstance(data['difficulty'], str):
            data['difficulty'] = DifficultyLevel(data['difficulty'])
        return cls(**data)


# Difficulty assessment criteria
DIFFICULTY_CRITERIA = """
Assess the difficulty of the analogical reasoning based on:

EASY:
- Domains are closely related or within similar fields
- Analogy is relatively straightforward and intuitive
- Conceptual leap is small
- Examples: applying physics from one mechanical system to another similar system

MEDIUM:
- Domains are moderately distant (e.g., different scientific fields)
- Requires some creative insight to see the connection
- Conceptual structures have notable similarities but aren't obvious
- Examples: biology concepts applied to computer science, physics to economics

HARD:
- Domains are very distant (e.g., arts to hard science, nature to technology)
- Requires significant creative leap and deep insight
- Non-obvious structural similarities that most wouldn't see
- Revolutionary or paradigm-shifting analogies
- Examples: natural selection to algorithm design, ant colony behavior to optimization algorithms
"""
