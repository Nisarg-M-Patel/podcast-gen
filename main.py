# main.py
"""
Main entry point for podcast generation.
"""
import asyncio
from pathlib import Path
from generate import PodcastGenerator


async def main():
    """Run podcast generation."""
    
    # Configuration
    arxiv_id = "2503.10918"
    paper_dir = Path(f"data/papers/{arxiv_id}")
    chunks_path = paper_dir / "chunks.json"
    output_dir = paper_dir / "podcast"
    
    if not chunks_path.exists():
        print(f"❌ Chunks not found at {chunks_path}")
        print(f"Run parser.py first to create chunks")
        return
    
    # Load chunks
    print(f"Loading chunks from {chunks_path}...")
    from parser import Chunker
    chunker = Chunker()
    chunks = chunker.load_chunks(chunks_path)
    print(f"Loaded {len(chunks)} chunks")
    
    # Generate podcast (2 LLM calls: truth pack + script)
    generator = PodcastGenerator(model="gemini-2.0-flash")
    
    try:
        script = await generator.generate_podcast(
            chunks=chunks,
            paper_title="Hadar: Heterogeneity-Aware Optimization-Based Online Scheduling for Deep Learning Clusters",
            output_dir=output_dir
        )
        
        print(f"\n✅ Done! Outputs at: {output_dir}")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("\nIf rate limited, wait a few minutes and try again.")


if __name__ == "__main__":
    asyncio.run(main())