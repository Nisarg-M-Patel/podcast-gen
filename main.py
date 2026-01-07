# main.py
"""
Main entry point for single-shot podcast generation.
"""
import asyncio
from pathlib import Path
from generate import PodcastGenerator


async def main():
    """Run single-shot podcast generation."""
    
    # Configuration
    arxiv_id = "2503.10918"
    paper_dir = Path(f"data/papers/{arxiv_id}")
    chunks_path = paper_dir / "chunks.json"
    output_dir = paper_dir / "podcast"
    
    if not chunks_path.exists():
        print(f"❌ Chunks not found at {chunks_path}")
        print(f"Please run parser.py first to create chunks")
        return
    
    # Load chunks
    print(f"Loading chunks from {chunks_path}...")
    from parser import Chunker
    chunker = Chunker()
    chunks = chunker.load_chunks(chunks_path)
    
    print(f"Loaded {len(chunks)} chunks\n")
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate podcast with SINGLE LLM call
    generator = PodcastGenerator(model="gemini-2.5-flash")
    
    try:
        script = await generator.generate_podcast(
            chunks=chunks,
            paper_title="GAVEL: Generating Games Via Evolution and Language Models",
            output_dir=output_dir
        )
        
        print(f"✅ Success! Check outputs at: {output_dir}")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("\nIf rate limited, wait a few minutes and try again.")
        print("Or upgrade at: https://aistudio.google.com/")


if __name__ == "__main__":
    asyncio.run(main())