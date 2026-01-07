# generate.py
"""
Single-shot podcast generation - everything in ONE LLM call.
"""
import os
import json
import re
from typing import List
from pathlib import Path
from dotenv import load_dotenv
from google import genai

from models import Chunk, StoryStructure, StoryBeat, DialogueTurn, PodcastScript, SectionType
from prompts import SINGLE_SHOT_PODCAST_PROMPT

load_dotenv()


class PodcastGenerator:
    """Generate complete podcast in a single LLM call."""
    
    def __init__(self, model: str = "gemini-2.5-flash"):
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY not found in .env file")
        
        self.client = genai.Client(api_key=api_key)
        self.model = model
    
    async def generate_podcast(
        self,
        chunks: List[Chunk],
        paper_title: str,
        output_dir: Path
    ) -> PodcastScript:
        """
        Generate complete podcast in ONE LLM call.
        
        Args:
            chunks: Parsed chunks from paper
            paper_title: Title of the paper
            output_dir: Directory to save outputs
            
        Returns:
            Complete PodcastScript
        """
        print(f"\n🎙️ Generating podcast for: {paper_title}")
        print(f"Using single-shot approach (1 LLM call)\n")
        
        # Prepare paper content
        abstract = self._get_abstract(chunks)
        all_sections = self._format_all_sections(chunks)
        
        # Build mega-prompt
        prompt = SINGLE_SHOT_PODCAST_PROMPT.format(
            title=paper_title,
            abstract=abstract,
            all_sections=all_sections
        )
        
        print("📝 Generating complete podcast script...")
        print(f"   (This may take 30-60 seconds)\n")
        
        # ONE LLM call for everything
        response = self._call_llm(prompt)
        
        print("✓ Received response, parsing...\n")
        
        # Parse JSON response
        script = self._parse_response(response, paper_title)
        
        # Save outputs
        output_dir.mkdir(parents=True, exist_ok=True)
        self._save_script(script, output_dir)
        
        print("=" * 60)
        print("✅ PODCAST GENERATION COMPLETE")
        print("=" * 60)
        print(f"Output directory: {output_dir}")
        print(f"Narrative type: {script.story_structure.narrative_type}")
        print(f"Story beats: {len(script.story_structure.beats)}")
        print(f"Total dialogue turns: {len(script.dialogue)}")
        print()
        
        return script
    
    def _get_abstract(self, chunks: List[Chunk]) -> str:
        """Extract abstract or first chunk."""
        for chunk in chunks:
            if chunk.section_type == SectionType.ABSTRACT:
                return chunk.text
        
        return chunks[0].text[:1000] if chunks else ""
    
    def _format_all_sections(self, chunks: List[Chunk]) -> str:
        """Format all sections with content."""
        sections_dict = {}
        
        # Group chunks by section
        for chunk in chunks:
            if chunk.section not in sections_dict:
                sections_dict[chunk.section] = []
            sections_dict[chunk.section].append(chunk.text)
        
        # Format for prompt
        lines = []
        for section, texts in sections_dict.items():
            combined_text = "\n\n".join(texts)
            # Limit each section to 1500 chars to keep prompt reasonable
            if len(combined_text) > 1500:
                combined_text = combined_text[:1500] + "..."
            lines.append(f"## {section}\n{combined_text}\n")
        
        return "\n".join(lines)
    
    def _call_llm(self, prompt: str) -> str:
        """Call LLM with retry logic."""
        import time
        
        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = self.client.models.generate_content(
                    model=self.model,
                    contents=prompt,
                    config={
                        "temperature": 0.5,
                        "max_output_tokens": 8000,  # Large output for complete script
                    }
                )
                return response.text
            except Exception as e:
                error_str = str(e)
                if "429" in error_str or "RESOURCE_EXHAUSTED" in error_str:
                    if attempt < max_retries - 1:
                        # Extract retry delay from error if available
                        retry_match = re.search(r'retry.*?(\d+)s', error_str, re.IGNORECASE)
                        wait_time = int(retry_match.group(1)) if retry_match else (attempt + 1) * 30
                        print(f"⏳ Rate limited, waiting {wait_time}s...")
                        time.sleep(wait_time)
                    else:
                        raise Exception(f"Rate limit exceeded after {max_retries} retries. Please wait a few minutes and try again.")
                else:
                    raise Exception(f"LLM API error: {e}")
        
        return ""
    
    def _parse_response(self, response: str, paper_title: str) -> PodcastScript:
        """Parse JSON response into PodcastScript."""
        try:
            # Extract JSON from response
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if not json_match:
                raise ValueError("No JSON found in response")
            
            data = json.loads(json_match.group(0))
            
            # Parse beats
            beats = []
            all_dialogue = []
            
            for beat_data in data.get('beats', []):
                # Parse dialogue for this beat
                beat_dialogue = []
                for turn_data in beat_data.get('dialogue', []):
                    turn = DialogueTurn(
                        speaker=turn_data.get('speaker', 'host1'),
                        text=turn_data.get('text', '')
                    )
                    beat_dialogue.append(turn)
                    all_dialogue.append(turn)
                
                # Create beat
                beat = StoryBeat(
                    title=beat_data.get('title', ''),
                    focus=beat_data.get('focus', ''),
                    narrative_role=beat_data.get('narrative_role', ''),
                    retrieve_from_sections=[],
                    content_query='',
                )
                beats.append(beat)
            
            # Create story structure
            story = StoryStructure(
                narrative_type=data.get('narrative_type', 'unknown'),
                hook=data.get('hook', ''),
                tension_points=data.get('tension_points', []),
                beats=beats
            )
            
            # Create script
            script = PodcastScript(
                paper_title=paper_title,
                story_structure=story,
                dialogue=all_dialogue
            )
            
            return script
            
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            print(f"⚠️  Warning: Failed to parse response: {e}")
            print(f"Raw response preview: {response[:500]}...")
            
            # Return minimal script as fallback
            return PodcastScript(
                paper_title=paper_title,
                story_structure=StoryStructure(
                    narrative_type="unknown",
                    hook="Failed to generate",
                    tension_points=[],
                    beats=[]
                ),
                dialogue=[]
            )
    
    def _save_script(self, script: PodcastScript, output_dir: Path):
        """Save script to files."""
        # Save story structure
        story_data = {
            "narrative_type": script.story_structure.narrative_type,
            "hook": script.story_structure.hook,
            "tension_points": script.story_structure.tension_points,
            "beats": [
                {
                    "title": b.title,
                    "focus": b.focus,
                    "narrative_role": b.narrative_role
                }
                for b in script.story_structure.beats
            ]
        }
        
        with open(output_dir / "story.json", 'w') as f:
            json.dump(story_data, f, indent=2)
        
        # Save dialogue
        with open(output_dir / "dialogue.txt", 'w') as f:
            for turn in script.dialogue:
                speaker_label = "Host 1" if turn.speaker == "host1" else "Host 2"
                f.write(f"{speaker_label}: {turn.text}\n\n")
        
        print(f"💾 Saved:")
        print(f"   - {output_dir}/story.json")
        print(f"   - {output_dir}/dialogue.txt")