# ingest.py
import time
import requests
from pathlib import Path
from typing import Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ArXivFetcher:
    """
    Download papers from ArXiv using the export API.
    
    IMPORTANT: Always use export.arxiv.org, NOT arxiv.org
    The export subdomain is for programmatic access and respects rate limits.
    """
    
    BASE_URL = "https://export.arxiv.org/pdf/"  # DO NOT change to arxiv.org
    RATE_LIMIT_SECONDS = 3
    
    def __init__(self, save_dir: str = "data/papers"):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.last_request_time = 0
    
    def _enforce_rate_limit(self):
        """Ensure at least 3 seconds between requests."""
        elapsed = time.time() - self.last_request_time
        if elapsed < self.RATE_LIMIT_SECONDS:
            sleep_time = self.RATE_LIMIT_SECONDS - elapsed
            logger.info(f"Rate limiting: sleeping {sleep_time:.2f}s")
            time.sleep(sleep_time)
        self.last_request_time = time.time()
    
    def fetch(self, arxiv_id: str) -> Optional[Path]:
        """
        Download a paper from ArXiv.
        
        Creates directory structure: data/papers/{arxiv_id}/paper.pdf
        
        Args:
            arxiv_id: ArXiv ID (e.g., "2503.10918")
        
        Returns:
            Path to downloaded PDF, or None if failed
        """
        clean_id = arxiv_id.strip().replace("arXiv:", "")
        
        # Create paper directory
        paper_dir = self.save_dir / clean_id
        paper_dir.mkdir(parents=True, exist_ok=True)
        
        # PDF saved as paper.pdf inside directory
        filepath = paper_dir / "paper.pdf"
        
        if filepath.exists():
            logger.info(f"Paper {clean_id} already exists at {filepath}")
            return filepath
        
        # Enforce rate limit
        self._enforce_rate_limit()
        
        # Download
        url = f"{self.BASE_URL}{clean_id}"
        logger.info(f"Downloading {clean_id} from {url}")
        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            
            filepath.write_bytes(response.content)
            logger.info(f"Saved to {filepath}")
            return filepath
            
        except requests.RequestException as e:
            logger.error(f"Failed to download {clean_id}: {e}")
            return None


if __name__ == "__main__":
    fetcher = ArXivFetcher()
    paper = fetcher.fetch("2503.10918")
    if paper:
        print(f"✓ Downloaded: {paper}")