# models.py
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

class SectionType(Enum):
    """Types of sections in research papers."""
    ABSTRACT = "abstract"
    INTRODUCTION = "introduction"
    RELATED_WORK = "related_work"
    METHODOLOGY = "methodology"
    RESULTS = "results"
    DISCUSSION = "discussion"
    CONCLUSION = "conclusion"
    APPENDIX = "appendix"
    OTHER = "other"

@dataclass
class Chunk:
    """A chunk of text from a research paper."""
    id: str
    text: str
    section: str                    # Human-readable section name
    section_id: str                 # Unique section identifier
    section_type: SectionType
    doc_position: int               # Order in document
    page: int
    
# --- New Models for Narrative-Driven Podcast Generation ---

@dataclass
class NarrativeHook:
    """Something interesting/surprising discovered in the paper."""
    type: str  # "surprising_finding", "contradiction", "novel_approach", "mystery", etc.
    description: str
    evidence_chunk_ids: list[str]
    quote: Optional[str] = None

@dataclass
class StoryBeat:
    """A segment of the podcast narrative."""
    title: str
    focus: str  # What this beat is about
    narrative_role: str  # "establish_stakes", "create_tension", "reveal_solution", etc.
    retrieve_from_sections: list[str]  # Which sections to pull content from
    content_query: str  # Query to retrieve relevant chunks
    retrieved_chunks: list[Chunk] = field(default_factory=list)
    contextual_summaries: list[str] = field(default_factory=list)

@dataclass
class StoryStructure:
    """Overall narrative arc for the podcast."""
    narrative_type: str  # "detective_story", "breakthrough", "comparison", etc.
    hook: str  # The opening hook that grabs attention
    tension_points: list[str]  # Key moments of surprise/conflict
    beats: list[StoryBeat]  # Ordered sequence of story segments

@dataclass
class DialogueTurn:
    """A single turn in the podcast dialogue."""
    speaker: str  # "host1" or "host2"
    text: str
    
@dataclass
class PodcastScript:
    """Complete podcast script."""
    paper_title: str
    story_structure: StoryStructure
    dialogue: list[DialogueTurn]