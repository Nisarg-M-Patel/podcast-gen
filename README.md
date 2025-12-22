# Research Paper to Podcast Pipeline

## Phase 1: Ingest
Parse PDF with Docling. Research papers at subsection level - author's intended atomic unit of information.

**Output:** Structured document with section hierarchy, titles, page numbers

## Phase 2: Chunk
Hierarchical chunking based on subsection.
- If subsection < MAX_TOKENS: single chunk
- If subsection > MAX_TOKENS: split by paragraph with OVERLAP tokens

**Chunk Metadata:**
```python
{
   "chunk_id": "paper_001_chunk_017",
   "section": "3.2 Experimental Setup",
   "section_id": "3.2",
   "page": 5,
   "doc_position": 17  # global ordering
}
```

## Phase 3: Build Dependency Graph (One-Time, Expensive)

Extract relationships per SECTION (not subsection - diminishing returns).

**For key sections:** Abstract, Introduction, Methodology, Results, Discussion, Conclusion

**LLM Prompt:**
```
Paper: {title} by {authors}
Section: {section_title} ({section_id})
Content: {section_text}
Document Structure: {outline}

Identify relationships to other sections:
- Which sections does this DEPEND_ON?
- Which sections does this IMPLEMENT?
- Which sections does this VALIDATE?
- Which sections does this CONTEXTUALIZE?
- Which sections does this EXTEND?
- Which sections does this COMPARE with?
- Which sections does this REFERENCE?

Return JSON:
[
  {
    "target": "section_id",
    "type": "DEPENDS_ON",
    "confidence": 0.9,
    "evidence_chunk_id": "chunk_id_containing_evidence"
  },
  ...
]
```

**Graph Schema:**
```python
enum RelationshipType:
    DEPENDS_ON
    IMPLEMENTS
    VALIDATES
    CONTEXTUALIZES
    EXTENDS
    COMPARES
    REFERENCES

enum SectionType:
    ABSTRACT
    INTRODUCTION
    RELATED_WORK
    METHODOLOGY
    RESULTS
    DISCUSSION
    CONCLUSION
    APPENDIX

class Chunk:
    id: string
    text: string
    section: string
    section_id: string
    section_type: SectionType
    doc_position: int
    page: int

class Edge:
    source: string
    target: string
    relation_type: RelationshipType
    confidence: float
    evidence_chunk_id: string

class DependencyGraph:
    chunks: dict[string, Chunk]
    edges: dict[string, list[Edge]]
    reverse_edges: dict[string, list[Edge]]
    paper_title: string
    authors: list[string]
```

**Store:** `DependencyGraph` serialized with document

## Phase 4: Index

Simple embedding + keyword index:
```python
for chunk in chunks:
    embed(chunk.text) → vector_db
    index(chunk.text) → bm25_index
```

**Store:** Vector index + BM25 index

## Phase 5: RAG Retrieval + Graph Expansion

### 5.1: Initial Retrieval
```python
vector_results = vector_search(query, k=20)
bm25_results = bm25_search(query, k=20)
initial_chunks = reciprocal_rank_fusion(vector_results, bm25_results, k=5)
```

### 5.2: Graph-Based Expansion
```python
expanded = set(initial_chunks)

for chunk in initial_chunks:
    related = graph.get_related_chunks(
        chunk.id,
        relation_types=[all RelationshipTypes],  # or subset
        direction="both",
        min_confidence=0.7
    )
    expanded.update(related)

# Order by document position
ordered = sorted(expanded, key=lambda c: c.doc_position)
```

**Output:** Ordered list of contextually relevant chunks

## Phase 6: Podcast Script Generation

### 6.1: Multi-Query Structure
Generate script segments for engaging narrative:
```python
queries = [
    "What's the main contribution/result?",  # Hook
    "What problem does this solve?",          # Motivation
    "How did they do it?",                    # Methodology
    "What are the results?",                  # Validation
    "What are the tradeoffs?",                # Critique
    "What are the implications?"              # Future
]

for query in queries:
    chunks = retrieve_and_expand(query, graph)
    segment = llm.generate_segment(query, chunks)
    script.append(segment)
```

### 6.2: Script Refinement
```python
# Add disfluencies, natural transitions
refined_script = llm.refine(script, style="conversational podcast")
```

**Output:** Script with speaker labels

## Phase 7: Audio Synthesis
```python
for segment in script:
    audio = tts.generate(
        segment.text,
        voice=segment.speaker
    )
    audio_segments.append(audio)

final_audio = merge_with_crossfade(audio_segments)
```

**TTS Options:**
- ElevenLabs v3 (cloud, high quality)
- Kokoro (local, open source)

**Output:** Final podcast audio file

---

## Open Questions / Tunable Parameters

- MAX_TOKENS per chunk: 500? 1000?
- OVERLAP tokens: 50? 100?
- Confidence threshold for edge following: 0.5? 0.7? 0.9?
- Which relationship types to follow at query time: all? subset?
- Depth of graph traversal: 1 hop? 2 hops? Full connected component? 