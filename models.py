# models.py
from dataclasses import dataclass
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

class RelationshipType(Enum):
    """Types of relationships between chunks."""
    PREREQUISITE = "prerequisite"  # A depends on B (must read B first)
    SUPPORTS = "supports"           # A provides evidence for B
    REFERENCES = "references"       # A explicitly mentions B

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
    
@dataclass
class Edge:
    """A relationship between two chunks."""
    source: str              # chunk_id
    target: str              # chunk_id
    relation_type: RelationshipType
    confidence: float        # 0.0 - 1.0
    evidence_chunk_id: str   # chunk containing evidence text

@dataclass
class DependencyGraph:
    """Graph of relationships between chunks in a paper."""
    chunks: dict[str, Chunk]
    edges: dict[str, list[Edge]]           # source_id -> list of outgoing edges
    reverse_edges: dict[str, list[Edge]]   # target_id -> list of incoming edges
    paper_title: str
    authors: list[str]