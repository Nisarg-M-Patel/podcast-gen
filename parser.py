# parser.py
from pathlib import Path
from typing import List, Dict
import logging
import re
import json
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from models import Chunk, SectionType

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Chunking parameters
MAX_TOKENS = 800
OVERLAP_TOKENS = 100


class PDFParser:
    """Parse PDFs using Docling to extract structured content."""
    
    def __init__(self):
        # Configure pipeline options
        pipeline_options = PdfPipelineOptions()
        pipeline_options.do_ocr = False  # Most papers are text-based PDFs, not scans
        pipeline_options.do_table_structure = True  # Extract tables properly
        pipeline_options.table_structure_options.do_cell_matching = True
        
        # Configure format options
        format_options = {
            InputFormat.PDF: PdfFormatOption(
                pipeline_options=pipeline_options
            )
        }
        
        self.converter = DocumentConverter(
            format_options=format_options
        )
    
    def parse(self, pdf_path: Path) -> dict:
        """
        Parse a PDF and extract sections, text, and metadata.
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            Dict with keys: title, authors, sections (list of dicts with name, text, level)
        """
        logger.info(f"Parsing {pdf_path.name} with Docling...")
        result = self.converter.convert(str(pdf_path))
        
        # Extract document-level metadata
        doc = result.document
        title = doc.name or pdf_path.stem
        
        # Get markdown
        markdown = doc.export_to_markdown()
        logger.info(f"Extracted markdown ({len(markdown)} chars)")
        
        # Parse markdown into sections
        sections = self._parse_markdown_sections(markdown)
        logger.info(f"Parsed {len(sections)} sections")
        
        return {
            'title': title,
            'authors': [],
            'sections': sections
        }
    
    def _parse_markdown_sections(self, markdown: str) -> List[Dict]:
        """
        Parse markdown into sections based on headers.
        
        Returns list of dicts with: name, text, level
        """
        sections = []
        lines = markdown.split('\n')
        
        current_section = None
        current_text = []
        
        for line in lines:
            # Check if line is a header
            header_match = re.match(r'^(#{1,6})\s+(.+)$', line)
            
            if header_match:
                # Save previous section
                if current_section is not None:
                    sections.append({
                        'name': current_section['name'],
                        'text': '\n'.join(current_text).strip(),
                        'level': current_section['level']
                    })
                
                # Start new section
                level = len(header_match.group(1))  # Number of # symbols
                name = header_match.group(2).strip()
                current_section = {'name': name, 'level': level}
                current_text = []
            else:
                # Accumulate text under current section
                if line.strip():  # Skip empty lines
                    current_text.append(line)
        
        # Save final section
        if current_section is not None and current_text:
            sections.append({
                'name': current_section['name'],
                'text': '\n'.join(current_text).strip(),
                'level': current_section['level']
            })
        
        return sections


class Chunker:
    """Chunk parsed document into smaller pieces for retrieval."""
    
    def __init__(self, max_tokens: int = MAX_TOKENS, overlap_tokens: int = OVERLAP_TOKENS):
        self.max_tokens = max_tokens
        self.overlap_tokens = overlap_tokens
    
    def _estimate_tokens(self, text: str) -> int:
        """Rough token estimate: ~0.75 tokens per word."""
        return int(len(text.split()) * 0.75)
    
    def _infer_section_type(self, section_name: str) -> SectionType:
        """Map section names to SectionType enum."""
        section_lower = section_name.lower()
        
        if 'abstract' in section_lower:
            return SectionType.ABSTRACT
        elif 'introduction' in section_lower or section_lower.startswith('intro'):
            return SectionType.INTRODUCTION
        elif 'related' in section_lower or 'background' in section_lower or 'prior' in section_lower:
            return SectionType.RELATED_WORK
        elif 'method' in section_lower or 'approach' in section_lower or 'model' in section_lower or 'design' in section_lower:
            return SectionType.METHODOLOGY
        elif 'result' in section_lower or 'experiment' in section_lower or 'evaluation' in section_lower:
            return SectionType.RESULTS
        elif 'discussion' in section_lower or 'analysis' in section_lower:
            return SectionType.DISCUSSION
        elif 'conclusion' in section_lower or 'summary' in section_lower:
            return SectionType.CONCLUSION
        elif 'appendix' in section_lower or 'supplement' in section_lower:
            return SectionType.APPENDIX
        else:
            return SectionType.OTHER
    
    def chunk(self, parsed_doc: dict) -> List[Chunk]:
        """
        Chunk a parsed document into retrieval units.
        
        Strategy:
        - If section < MAX_TOKENS: keep as one chunk
        - If section > MAX_TOKENS: split by sentences/paragraphs with overlap
        
        Args:
            parsed_doc: Output from PDFParser.parse()
            
        Returns:
            List of Chunks
        """
        chunks = []
        
        for idx, section in enumerate(parsed_doc['sections']):
            section_chunks = self._create_chunks_from_section(
                section_name=section['name'],
                section_text=section['text'],
                section_id=f"sec_{idx}",
                section_idx=idx
            )
            chunks.extend(section_chunks)
        
        logger.info(f"Created {len(chunks)} chunks from {len(parsed_doc['sections'])} sections")
        return chunks
    
    def _create_chunks_from_section(
        self, 
        section_name: str, 
        section_text: str,
        section_id: str,
        section_idx: int
    ) -> List[Chunk]:
        """Create one or more chunks from a section."""
        section_type = self._infer_section_type(section_name)
        
        # If section is small enough, return single chunk
        if self._estimate_tokens(section_text) <= self.max_tokens:
            return [Chunk(
                id=f"{section_id}_c0",
                text=section_text,
                section=section_name,
                section_id=section_id,
                section_type=section_type,
                doc_position=section_idx,
                page=0  # We don't have page info from markdown
            )]
        
        # Otherwise split by paragraphs with overlap
        paragraphs = [p.strip() for p in section_text.split('\n\n') if p.strip()]
        
        chunks = []
        current_chunk_text = []
        current_tokens = 0
        chunk_idx = 0
        
        for para in paragraphs:
            para_tokens = self._estimate_tokens(para)
            
            if current_tokens + para_tokens > self.max_tokens and current_chunk_text:
                # Flush current chunk
                chunks.append(Chunk(
                    id=f"{section_id}_c{chunk_idx}",
                    text='\n\n'.join(current_chunk_text),
                    section=section_name,
                    section_id=section_id,
                    section_type=section_type,
                    doc_position=section_idx * 100 + chunk_idx,  # Preserve ordering
                    page=0
                ))
                chunk_idx += 1
                
                # Start new chunk with overlap (keep last paragraph)
                if len(current_chunk_text) > 0:
                    current_chunk_text = [current_chunk_text[-1]]
                    current_tokens = self._estimate_tokens(current_chunk_text[0])
                else:
                    current_chunk_text = []
                    current_tokens = 0
            
            current_chunk_text.append(para)
            current_tokens += para_tokens
        
        # Flush final chunk
        if current_chunk_text:
            chunks.append(Chunk(
                id=f"{section_id}_c{chunk_idx}",
                text='\n\n'.join(current_chunk_text),
                section=section_name,
                section_id=section_id,
                section_type=section_type,
                doc_position=section_idx * 100 + chunk_idx,
                page=0
            ))
        
        return chunks
    
    def save_chunks(self, chunks: List[Chunk], output_path: Path):
        """Save chunks to JSON file."""
        chunks_data = [
            {
                'id': c.id,
                'text': c.text,
                'section': c.section,
                'section_id': c.section_id,
                'section_type': c.section_type.value,
                'doc_position': c.doc_position,
                'page': c.page
            }
            for c in chunks
        ]
        
        with open(output_path, 'w') as f:
            json.dump(chunks_data, f, indent=2)
        
        logger.info(f"Saved {len(chunks)} chunks to {output_path}")
    
    def load_chunks(self, chunks_path: Path) -> List[Chunk]:
        """Load chunks from JSON file."""
        with open(chunks_path, 'r') as f:
            chunks_data = json.load(f)
        
        chunks = [
            Chunk(
                id=c['id'],
                text=c['text'],
                section=c['section'],
                section_id=c['section_id'],
                section_type=SectionType(c['section_type']),
                doc_position=c['doc_position'],
                page=c['page']
            )
            for c in chunks_data
        ]
        
        logger.info(f"Loaded {len(chunks)} chunks from {chunks_path}")
        return chunks


if __name__ == "__main__":
    # Test with already downloaded paper
    from pathlib import Path
    
    arxiv_id = "2503.10918"
    paper_dir = Path(f"data/papers/{arxiv_id}")
    pdf_path = paper_dir / "paper.pdf"
    chunks_path = paper_dir / "chunks.json"
    
    if not pdf_path.exists():
        print(f"Please download the paper first with: python ingest.py")
        exit(1)
    
    parser = PDFParser()
    chunker = Chunker()
    
    # Check if chunks already exist
    if chunks_path.exists():
        print(f"Loading cached chunks from {chunks_path}")
        chunks = chunker.load_chunks(chunks_path)
    else:
        # Parse and chunk
        parsed = parser.parse(pdf_path)
        print(f"✓ Parsed: {parsed['title']}")
        print(f"  Sections: {len(parsed['sections'])}")
        
        chunks = chunker.chunk(parsed)
        
        # Save chunks
        chunker.save_chunks(chunks, chunks_path)
    
    print(f"\n✓ Loaded {len(chunks)} chunks")
    print(f"\nFirst chunk:")
    print(f"  ID: {chunks[0].id}")
    print(f"  Section: {chunks[0].section}")
    print(f"  Type: {chunks[0].section_type}")
    print(f"  Text ({len(chunks[0].text)} chars):\n{chunks[0].text[:300]}...")