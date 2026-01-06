# graph.py
import json
import re
import os
import yaml
from typing import List, Dict
from pathlib import Path
from dotenv import load_dotenv
from google import genai
from models import Chunk, Edge, RelationshipType, DependencyGraph, SectionType

# Load environment variables
load_dotenv()

class GraphBuilder:
    """Build dependency graph with deterministic + validated LLM."""
    
    def __init__(self, config_path: str = "config.yaml"):
        # Load config
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Setup LLM
        llm_config = self.config['graph']['llm']
        self.provider = llm_config['provider']
        self.model_name = llm_config['model']
        self.temperature = llm_config.get('temperature', 0.0)
        self.max_tokens = llm_config.get('max_tokens', 2000)
        
        # Initialize LLM client
        if self.provider == "google":
            api_key = os.getenv("GOOGLE_API_KEY")
            if not api_key:
                raise ValueError("GOOGLE_API_KEY not found in .env file")
            
            # New API
            self.client = genai.Client(api_key=api_key)
            self.model = self.model_name  # Just store the name
        else:
            raise ValueError(f"Unsupported LLM provider: {self.provider}")
        
        # Setup validator
        validation_config = self.config['graph']['validation']
        self.validator = EdgeValidator(
            confidence_threshold=validation_config['confidence_threshold']
        )
        
        print(f"Using {self.provider} - {self.model_name}")
    
    def build_graph(self, chunks: List[Chunk], paper_title: str) -> DependencyGraph:
        """
        Build complete dependency graph:
        1. Extract REFERENCES deterministically (regex)
        2. Extract PREREQUISITE/SUPPORTS with LLM (validated)
        """
        print("Building dependency graph...")
        
        # Step 1: Deterministic REFERENCES
        print("  [1/3] Extracting explicit references...")
        reference_edges = self._extract_references(chunks)
        print(f"    Found {len(reference_edges)} reference edges")
        
        # Step 2: LLM for semantic relationships
        print(f"  [2/3] Extracting semantic relationships ({self.model_name})...")
        semantic_edges = self._extract_semantic_edges(chunks)
        print(f"    Found {len(semantic_edges)} semantic edges (before validation)")
        
        # Step 3: Validate all LLM edges
        print("  [3/3] Validating edges...")
        valid_edges = [e for e in semantic_edges if self.validator.validate(e, chunks)]
        print(f"    Kept {len(valid_edges)} valid edges")
        
        # Combine
        all_edges = reference_edges + valid_edges
        
        # Build graph structure
        graph = self._build_graph_structure(chunks, all_edges, paper_title)
        
        return graph
    
    def _extract_references(self, chunks: List[Chunk]) -> List[Edge]:
        """Deterministic extraction of explicit references."""
        edges = []
        
        for source in chunks:
            # Pattern 1: "Section X", "Sec. X"
            for match in re.finditer(
                r'(?:Section|Sec\.)\s+([IVX]+|\d+)',
                source.text,
                re.IGNORECASE
            ):
                target = self._find_section_by_number(match.group(1), chunks)
                if target and target.id != source.id:
                    edges.append(Edge(
                        source=source.id,
                        target=target.id,
                        relation_type=RelationshipType.REFERENCES,
                        confidence=1.0,
                        evidence_chunk_id=source.id
                    ))
            
            # Pattern 2: "Algorithm X", "Figure X", "Table X"
            for match in re.finditer(
                r'(?:Algorithm|Figure|Fig\.|Table)\s+(\d+)',
                source.text,
                re.IGNORECASE
            ):
                target = self._find_chunk_containing(match.group(0), chunks)
                if target and target.id != source.id:
                    edges.append(Edge(
                        source=source.id,
                        target=target.id,
                        relation_type=RelationshipType.REFERENCES,
                        confidence=1.0,
                        evidence_chunk_id=source.id
                    ))
        
        return self._deduplicate_edges(edges)
    
    def _extract_semantic_edges(self, chunks: List[Chunk]) -> List[Edge]:
        """Use LLM to extract PREREQUISITE and SUPPORTS relationships."""
        edges = []
        max_relationships = self.config['graph']['validation']['max_relationships_per_section']
        
        # Only analyze key sections
        key_chunks = [c for c in chunks if c.section_type in [
            SectionType.ABSTRACT,
            SectionType.INTRODUCTION,
            SectionType.RELATED_WORK,
            SectionType.METHODOLOGY,
            SectionType.RESULTS,
            SectionType.DISCUSSION,
            SectionType.CONCLUSION
        ]]
        
        print(f"    Analyzing {len(key_chunks)} key sections...")
        
        for i, source in enumerate(key_chunks):
            print(f"    [{i+1}/{len(key_chunks)}] {source.section[:50]}...")
            
            # Build context of other sections
            other_sections = [
                f"{j}. {c.section} ({c.section_type.value})"
                for j, c in enumerate(key_chunks)
                if c.id != source.id
            ]
            
            # Ask LLM
            prompt = self._build_relationship_prompt(source, other_sections, max_relationships)
            response = self._call_llm(prompt)
            
            # Parse and create edges
            edges.extend(self._parse_llm_response(response, source, key_chunks))
        
        return edges
    
    def _build_relationship_prompt(self, source: Chunk, other_sections: List[str], max_rel: int) -> str:
        return f"""You are analyzing relationships between sections of a research paper.

SOURCE SECTION:
Title: {source.section}
Type: {source.section_type.value}
Content (first 800 chars): {source.text[:800]}...

OTHER SECTIONS:
{chr(10).join(other_sections)}

Identify relationships from the SOURCE to OTHER sections:

1. PREREQUISITE: Which sections must be read BEFORE this one to understand it?
   - Introduction depends on Abstract
   - Results depend on Methodology
   - Discussion depends on Results

2. SUPPORTS: Which sections does this one provide evidence/validation for?
   - Methodology supports Results
   - Results support Discussion/Conclusion

Rules:
- Be conservative - only return HIGH-CONFIDENCE relationships
- Provide specific evidence from the text
- A section can have 0-{max_rel} relationships

Return JSON array:
[
  {{
    "target_index": <index from OTHER SECTIONS list>,
    "relation_type": "PREREQUISITE" or "SUPPORTS",
    "confidence": <0.7-1.0>,
    "evidence": "<specific quote from source text showing this relationship>"
  }}
]

Return ONLY the JSON array, no other text."""

    def _call_llm(self, prompt: str) -> str:
        """Call LLM API with retry logic."""
        import time
        
        max_retries = 3
        for attempt in range(max_retries):
            try:
                if self.provider == "google":
                    response = self.client.models.generate_content(
                        model=self.model,
                        contents=prompt,
                        config={
                            "temperature": self.temperature,
                            "max_output_tokens": self.max_tokens,
                        }
                    )
                    return response.text
            except Exception as e:
                if "429" in str(e) and attempt < max_retries - 1:
                    wait_time = (attempt + 1) * 10
                    print(f"      Rate limited, waiting {wait_time}s...")
                    time.sleep(wait_time)
                else:
                    print(f"      Warning: LLM API error: {e}")
                    return "[]"
        return "[]"
    
    def _parse_llm_response(
        self,
        response: str,
        source: Chunk,
        chunks: List[Chunk]
    ) -> List[Edge]:
        """Parse LLM JSON response into edges."""
        edges = []
        
        try:
            # Extract JSON from response
            json_match = re.search(r'\[.*\]', response, re.DOTALL)
            if not json_match:
                return edges
            
            relationships = json.loads(json_match.group(0))
            
            for rel in relationships:
                target_idx = rel.get('target_index')
                if target_idx is None or target_idx >= len(chunks):
                    continue
                
                target = chunks[target_idx]
                
                edges.append(Edge(
                    source=source.id,
                    target=target.id,
                    relation_type=RelationshipType(rel['relation_type'].lower()),
                    confidence=float(rel['confidence']),
                    evidence_chunk_id=source.id
                ))
        
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            print(f"      Warning: Failed to parse LLM response: {e}")
        
        return edges
    
    def _find_section_by_number(self, num: str, chunks: List[Chunk]) -> Chunk:
        """Find section by Roman numeral or number."""
        for chunk in chunks:
            if chunk.section.startswith(f"{num}.") or chunk.section.startswith(f"{num} "):
                return chunk
        return None
    
    def _find_chunk_containing(self, text: str, chunks: List[Chunk]) -> Chunk:
        """Find chunk containing specific text."""
        for chunk in chunks:
            if text in chunk.section or text in chunk.text[:200]:
                return chunk
        return None
    
    def _deduplicate_edges(self, edges: List[Edge]) -> List[Edge]:
        """Remove duplicate edges."""
        seen = set()
        unique = []
        
        for edge in edges:
            key = (edge.source, edge.target, edge.relation_type.value)
            if key not in seen:
                seen.add(key)
                unique.append(edge)
        
        return unique
    
    def _build_graph_structure(
        self,
        chunks: List[Chunk],
        edges: List[Edge],
        paper_title: str
    ) -> DependencyGraph:
        """Build final graph structure with forward and reverse edges."""
        chunks_dict = {c.id: c for c in chunks}
        edges_dict = {}
        reverse_edges_dict = {}
        
        for edge in edges:
            # Forward edges
            if edge.source not in edges_dict:
                edges_dict[edge.source] = []
            edges_dict[edge.source].append(edge)
            
            # Reverse edges
            if edge.target not in reverse_edges_dict:
                reverse_edges_dict[edge.target] = []
            reverse_edges_dict[edge.target].append(edge)
        
        return DependencyGraph(
            chunks=chunks_dict,
            edges=edges_dict,
            reverse_edges=reverse_edges_dict,
            paper_title=paper_title,
            authors=[]
        )
    
    def save_graph(self, graph: DependencyGraph, output_dir: Path):
        """Save graph as JSON and DOT."""
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 1. JSON (programmatic access)
        json_path = output_dir / "graph.json"
        self._save_json(graph, json_path)
        
        # 2. DOT (Graphviz visualization)
        dot_path = output_dir / "graph.dot"
        self._save_dot(graph, dot_path)
        
        print(f"\n✓ Graph saved to {output_dir}/")
        print(f"  - graph.json (programmatic access)")
        print(f"  - graph.dot (install Graphviz Preview extension in VSCode)")
        print(f"\nStatistics:")
        print(f"  Total edges: {sum(len(e) for e in graph.edges.values())}")
        print(f"  PREREQUISITE: {sum(1 for edges in graph.edges.values() for e in edges if e.relation_type == RelationshipType.PREREQUISITE)}")
        print(f"  SUPPORTS: {sum(1 for edges in graph.edges.values() for e in edges if e.relation_type == RelationshipType.SUPPORTS)}")
        print(f"  REFERENCES: {sum(1 for edges in graph.edges.values() for e in edges if e.relation_type == RelationshipType.REFERENCES)}")
    
    def _save_json(self, graph: DependencyGraph, path: Path):
        """Save as JSON."""
        data = {
            'title': graph.paper_title,
            'authors': graph.authors,
            'chunks': [
                {
                    'id': c.id,
                    'section': c.section,
                    'section_type': c.section_type.value,
                    'doc_position': c.doc_position,
                }
                for c in graph.chunks.values()
            ],
            'edges': [
                {
                    'source': edge.source,
                    'target': edge.target,
                    'relation_type': edge.relation_type.value,
                    'confidence': edge.confidence,
                }
                for edges_list in graph.edges.values()
                for edge in edges_list
            ]
        }
        
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)
    
    def _save_dot(self, graph: DependencyGraph, path: Path):
        """Save as Graphviz DOT format."""
        lines = [
            'digraph paper {',
            '    rankdir=TB;',
            '    node [shape=box, style="rounded,filled"];',
            '    graph [splines=ortho, nodesep=1.0, ranksep=1.5];',
            ''
        ]
        
        # Add nodes with colors
        for chunk in graph.chunks.values():
            node_id = self._sanitize_id(chunk.id)
            label = chunk.section[:50].replace('"', '\\"')
            
            # Color by section type
            colors = {
                SectionType.ABSTRACT: '#e8f5e9',
                SectionType.INTRODUCTION: '#e8f5e9',
                SectionType.RELATED_WORK: '#fff3e0',
                SectionType.METHODOLOGY: '#fff3e0',
                SectionType.RESULTS: '#e3f2fd',
                SectionType.DISCUSSION: '#f3e5f5',
                SectionType.CONCLUSION: '#f3e5f5',
            }
            color = colors.get(chunk.section_type, '#f5f5f5')
            
            lines.append(f'    {node_id} [label="{label}", fillcolor="{color}"];')
        
        lines.append('')
        
        # Add edges with styling
        for source_id, edges in graph.edges.items():
            source_node = self._sanitize_id(source_id)
            for edge in edges:
                target_node = self._sanitize_id(edge.target)
                
                # Style by relationship type
                if edge.relation_type == RelationshipType.PREREQUISITE:
                    color = '#ff6b6b'
                    style = 'solid'
                    penwidth = '2.0'
                elif edge.relation_type == RelationshipType.SUPPORTS:
                    color = '#4ecdc4'
                    style = 'solid'
                    penwidth = '2.0'
                else:  # REFERENCES
                    color = '#95a5a6'
                    style = 'dashed'
                    penwidth = '1.0'
                
                label = edge.relation_type.value[:3].upper()
                lines.append(
                    f'    {source_node} -> {target_node} '
                    f'[label="{label}", color="{color}", style={style}, penwidth={penwidth}];'
                )
        
        lines.extend([
            '',
            '    // Legend',
            '    subgraph cluster_legend {',
            '        label="Legend";',
            '        style=filled;',
            '        color=lightgrey;',
            '        node [shape=plaintext];',
            '        legend [label=<',
            '            <table border="0" cellborder="0" cellspacing="0">',
            '                <tr><td><font color="#ff6b6b">PRE</font></td><td>PREREQUISITE</td></tr>',
            '                <tr><td><font color="#4ecdc4">SUP</font></td><td>SUPPORTS</td></tr>',
            '                <tr><td><font color="#95a5a6">REF</font></td><td>REFERENCES</td></tr>',
            '            </table>',
            '        >];',
            '    }',
            '}'
        ])
        
        with open(path, 'w') as f:
            f.write('\n'.join(lines))
    
    def _sanitize_id(self, node_id: str) -> str:
        """Convert chunk ID to valid DOT identifier."""
        return node_id.replace('-', '_').replace('.', '_')


class EdgeValidator:
    """Validate LLM-generated edges with domain rules."""
    
    def __init__(self, confidence_threshold: float = 0.7):
        self.confidence_threshold = confidence_threshold
    
    def validate(self, edge: Edge, chunks: List[Chunk]) -> bool:
        """Check if edge passes all validation rules."""
        chunks_dict = {c.id: c for c in chunks}
        source = chunks_dict.get(edge.source)
        target = chunks_dict.get(edge.target)
        
        if not source or not target:
            return False
        
        # Rule 1: No self-loops
        if source.id == target.id:
            return False
        
        # Rule 2: Confidence threshold
        if edge.confidence < self.confidence_threshold:
            return False
        
        # Rule 3: PREREQUISITE flows backward in doc
        if edge.relation_type == RelationshipType.PREREQUISITE:
            if source.doc_position <= target.doc_position:
                return False
        
        # Rule 4: SUPPORTS flows forward in doc
        if edge.relation_type == RelationshipType.SUPPORTS:
            if source.doc_position >= target.doc_position:
                return False
        
        # Rule 5: Type compatibility
        if not self._compatible_types(source, target, edge.relation_type):
            return False
        
        return True
    
    def _compatible_types(
        self,
        source: Chunk,
        target: Chunk,
        rel_type: RelationshipType
    ) -> bool:
        """Check if section types make sense for relationship."""
        # Results can't be PREREQUISITE for Introduction
        if (source.section_type == SectionType.RESULTS and
            target.section_type == SectionType.INTRODUCTION and
            rel_type == RelationshipType.PREREQUISITE):
            return False
        
        # Introduction can't SUPPORT Results
        if (source.section_type == SectionType.INTRODUCTION and
            target.section_type == SectionType.RESULTS and
            rel_type == RelationshipType.SUPPORTS):
            return False
        
        return True


if __name__ == "__main__":
    from parser import PDFParser, Chunker
    
    # Setup
    arxiv_id = "2503.10918"
    paper_dir = Path(f"data/papers/{arxiv_id}")
    chunks_path = paper_dir / "chunks.json"
    
    # Load chunks
    chunker = Chunker()
    chunks = chunker.load_chunks(chunks_path)
    print(f"Loaded {len(chunks)} chunks")
    
    # Build graph (reads config from config.yaml)
    builder = GraphBuilder()
    graph = builder.build_graph(chunks, "GAVEL Paper")
    
    # Save
    builder.save_graph(graph, paper_dir)