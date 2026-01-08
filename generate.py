# generate.py
"""
Two-pass grounded podcast generation.

Pass 1: Extract "truth pack" - all claims with chunk ID citations
Pass 2: Generate script using ONLY claims from truth pack

This prevents hallucination and forces depth because the model must
find specific evidence before making claims.
"""
import os
import json
import re
import time
from typing import List
from pathlib import Path
from dotenv import load_dotenv
from google import genai

from models import Chunk, StoryStructure, StoryBeat, DialogueTurn, PodcastScript, SectionType

load_dotenv()


TRUTH_PACK_PROMPT = """You are extracting factual content from a research paper. Be thorough and precise.

Paper Title: {title}

=== FULL PAPER CONTENT ===
{full_paper}
=== END PAPER CONTENT ===

## YOUR TASK: Create a "Truth Pack" - a structured extraction of ALL claims and evidence.

This truth pack will be used to generate a podcast. The podcast can ONLY mention things that appear in this truth pack, so be comprehensive.

### EXTRACTION REQUIREMENTS:

1. **Core Thesis** (1-2 sentences): What is this paper's main contribution?

2. **Key Contributions** (bullet list): What does this paper claim to offer?

3. **Method Summary** (detailed bullets): 
   - HOW does the approach work? Be specific about mechanisms.
   - What algorithms/techniques are used?
   - What makes it different from prior work?

4. **Results** (each with specifics):
   - Metric name
   - Exact numbers (X% improvement, Y speedup, etc.)
   - Comparison baseline
   - Conditions/dataset
   - Chunk ID where this appears

5. **Limitations** (bullets): What doesn't work? What's out of scope?

6. **Key Terms** (definitions): Technical terms a listener would need explained

7. **Quotable Moments** (short excerpts ≤2 sentences, with chunk IDs):
   - Surprising claims
   - Strong statements
   - Memorable phrases

8. **Claims Table**: For EVERY significant claim, provide:
   - claim: The statement
   - evidence: What supports it
   - chunk_ids: Which chunks contain the evidence
   - confidence: high/medium/low

### OUTPUT FORMAT (JSON):
{{
  "core_thesis": "...",
  "contributions": ["...", "..."],
  "method_summary": [
    {{"mechanism": "...", "details": "...", "chunk_ids": ["sec_X_c0"]}},
    ...
  ],
  "results": [
    {{
      "metric": "training speedup",
      "value": "1.2x",
      "baseline": "Gavel",
      "conditions": "trace-driven simulation",
      "chunk_ids": ["sec_5_c2"]
    }},
    ...
  ],
  "limitations": ["...", "..."],
  "key_terms": [
    {{"term": "...", "definition": "...", "chunk_ids": ["..."]}},
    ...
  ],
  "quotable_moments": [
    {{"quote": "...", "context": "...", "chunk_ids": ["..."]}},
    ...
  ],
  "claims_table": [
    {{
      "claim": "...",
      "evidence": "...",
      "chunk_ids": ["..."],
      "confidence": "high|medium|low"
    }},
    ...
  ]
}}

BE THOROUGH. If it's not in this truth pack, it won't be in the podcast.
Extract now:
"""


SCRIPT_PROMPT = """You are writing a 15-minute podcast script. You may ONLY include information from the truth pack below.

Paper Title: {title}

=== TRUTH PACK (your only source of facts) ===
{truth_pack}
=== END TRUTH PACK ===

## TARGET FORMAT

Length: 1,900-2,400 spoken words (~15 minutes at 140 wpm)

Structure with timestamps:
- Hook (0:00-0:45): Grab attention with the most surprising finding
- Problem + Stakes (0:45-2:30): Why does this matter? What was broken?
- Key Idea / Method (2:30-7:00): HOW does it work? This is the meat.
- Results + Surprises (7:00-10:30): What did they find? What's unexpected?
- Limitations + Failure Modes (10:30-12:30): What doesn't work?
- What's Next (12:30-14:00): Implications, future work
- Recap + Close (14:00-15:00): Key takeaways

## HOST PERSONAS

**Host 1 (The Explainer)**: 
- Confident, teaches concepts step-by-step
- Uses concrete analogies (everyday objects, not abstract)
- References specific numbers from the truth pack
- No filler words

**Host 2 (The Skeptic)**:
- Asks "how does that actually work?" and "why should I believe that?"
- Pushes back on claims, asks for evidence
- Represents a smart but non-expert listener
- Reactions: "Wait—", "Hang on—", "That's a big claim..."

## DIALOGUE RULES

1. Each section needs 15-25 exchanges (we need depth!)
2. Every factual claim must come from the truth pack
3. When citing a result, include the actual number
4. Host 2 must challenge at least 2 claims per section with REAL skepticism (not just "that sounds complicated")
5. Include at least 1 DEVELOPED analogy per major concept (explain the analogy, don't just name it)
6. After each section, include a [SOURCES: chunk_id1, chunk_id2] line
7. Short sentences (podcast-readable)
8. Occasional recap lines: "So the key insight is..."
9. NEVER have Host 1 just confirm with "Exactly" or "Precisely" - add new information
10. When explaining mechanisms, use the pattern: "The intuition is..." then concrete example
11. If a number seems counterintuitive (like "124% reduction"), EXPLAIN what it actually means
12. Host 2 should ask "but WHY does that work?" at least twice per technical section

## WHAT MAKES IT GOOD

- DEPTH over breadth: Explain mechanisms, not just outcomes
- TENSION: Hosts don't always agree; skepticism is good
- SPECIFICITY: "1.2x faster" not "significantly faster"
- GROUNDING: Every claim traces to the truth pack

## OUTPUT FORMAT

```
=== HOOK (0:00-0:45) ===

Host 1: [dialogue]

Host 2: [dialogue]

[SOURCES: sec_1_c0, sec_3_c2]

=== PROBLEM + STAKES (0:45-2:30) ===

Host 1: [dialogue]

...
```

## ANTI-PATTERNS TO AVOID

❌ "This is really impressive" (vague praise)
❌ "They achieved great results" (no specifics)
❌ "It's a novel approach" (what makes it novel?)
❌ Constant agreement between hosts
❌ Skipping over HOW things work
❌ Host 2: "So X?" Host 1: "Exactly." (empty confirmation)
❌ Host 2: "That sounds complicated" (weak pushback)
❌ Naming an analogy without developing it: "It's like a kitchen" (then what?)

✅ "They got a 1.2x speedup over Gavel by profiling each task individually"
✅ "Wait, how is that different from just using the fastest GPU for everything?"
✅ Developed analogy: "Think of it like a restaurant kitchen. You wouldn't have your best sushi chef making salads - that's wasted talent. Hadar figures out which 'chef' (GPU) is best at which 'dish' (task) and assigns accordingly. But here's the key - it does this at the level of individual cooking steps, not whole meals."
✅ Real skepticism: "But if you're training copies of the same model on different machines, won't they diverge? How do you reconcile different weight updates?"
✅ When Host 2 asks "So X?", Host 1 should say "Yes, AND..." with additional insight

Write the complete podcast script now. Remember: if it's not in the truth pack, don't say it.
"""


class PodcastGenerator:
    """
    Two-pass generation:
    1. Extract truth pack (grounding)
    2. Generate script from truth pack (constrained generation)
    """
    
    def __init__(self, model: str = "gemini-2.5-flash"):
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY not found in .env")
        
        self.client = genai.Client(api_key=api_key)
        self.model = model
        self.call_count = 0
    
    async def generate_podcast(
        self,
        chunks: List[Chunk],
        paper_title: str,
        output_dir: Path
    ) -> PodcastScript:
        """Two-pass grounded generation."""
        print(f"\n🎙️ Podcast Generation: {paper_title}")
        print(f"Strategy: Truth Pack → Grounded Script (2 LLM calls)\n")
        
        # Format full paper (no truncation - Gemini handles it)
        full_paper = self._format_full_paper(chunks)
        print(f"📄 Paper content: {len(full_paper):,} chars (~{len(full_paper)//4:,} tokens)")
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # === PASS 1: Truth Pack ===
        print("\n" + "="*60)
        print("PASS 1: Extracting Truth Pack")
        print("="*60)
        
        truth_pack = await self._extract_truth_pack(paper_title, full_paper)
        
        with open(output_dir / "truth_pack.json", 'w') as f:
            json.dump(truth_pack, f, indent=2)
        
        self._print_truth_pack_stats(truth_pack)
        
        # Rate limit pause
        print("\n⏳ Waiting 60s before next call (rate limit)...")
        time.sleep(60)
        
        # === PASS 2: Script Generation ===
        print("\n" + "="*60)
        print("PASS 2: Generating Script")
        print("="*60)
        
        script_text = await self._generate_script(paper_title, truth_pack)
        
        with open(output_dir / "script_raw.txt", 'w') as f:
            f.write(script_text)
        
        script = self._parse_script(script_text, paper_title, truth_pack)
        self._save_outputs(script, output_dir)
        
        # Stats
        print("\n" + "="*60)
        print("✅ COMPLETE")
        print("="*60)
        print(f"LLM calls: {self.call_count}")
        print(f"Dialogue turns: {len(script.dialogue)}")
        word_count = sum(len(t.text.split()) for t in script.dialogue)
        print(f"Words: ~{word_count} ({word_count // 140} min @ 140 wpm)")
        print(f"Output: {output_dir}")
        
        return script
    
    def _format_full_paper(self, chunks: List[Chunk]) -> str:
        """Format complete paper with chunk IDs for citation."""
        lines = []
        current_section = None
        
        for chunk in chunks:
            if chunk.section != current_section:
                current_section = chunk.section
                lines.append(f"\n### {current_section}\n")
            lines.append(f"[{chunk.id}]\n{chunk.text}\n")
        
        return "\n".join(lines)
    
    async def _extract_truth_pack(self, title: str, full_paper: str) -> dict:
        """Pass 1: Extract structured claims with evidence."""
        prompt = TRUTH_PACK_PROMPT.format(title=title, full_paper=full_paper)
        
        print("📝 Extracting claims and evidence...")
        response = self._call_llm(prompt, max_tokens=8000)
        
        try:
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                return json.loads(json_match.group(0))
        except json.JSONDecodeError as e:
            print(f"⚠️ JSON parse error: {e}")
        
        return {"core_thesis": "Parse failed", "contributions": [], "method_summary": [],
                "results": [], "limitations": [], "key_terms": [], 
                "quotable_moments": [], "claims_table": []}
    
    async def _generate_script(self, title: str, truth_pack: dict) -> str:
        """Pass 2: Generate script constrained to truth pack."""
        prompt = SCRIPT_PROMPT.format(title=title, truth_pack=json.dumps(truth_pack, indent=2))
        
        print("📝 Generating script...")
        return self._call_llm(prompt, max_tokens=12000)
    
    def _call_llm(self, prompt: str, max_tokens: int = 8000) -> str:
        """Call LLM with retry and fallback."""
        self.call_count += 1
        
        # Models to try in order
        models_to_try = [self.model]
        if self.model == "gemini-2.5-flash":
            models_to_try.append("gemini-2.0-flash")  # Fallback
        
        for model in models_to_try:
            for attempt in range(3):
                try:
                    response = self.client.models.generate_content(
                        model=model,
                        contents=prompt,
                        config={"temperature": 0.5, "max_output_tokens": max_tokens}
                    )
                    return response.text
                except Exception as e:
                    error_str = str(e)
                    
                    # Rate limit - wait and retry
                    if "429" in error_str or "RESOURCE_EXHAUSTED" in error_str:
                        wait = (attempt + 1) * 90
                        print(f"⏳ Rate limited. Waiting {wait}s...")
                        time.sleep(wait)
                    
                    # Model overloaded - wait and retry, then try fallback
                    elif "503" in error_str or "UNAVAILABLE" in error_str:
                        wait = (attempt + 1) * 30
                        print(f"⏳ Model overloaded ({model}). Waiting {wait}s... (attempt {attempt + 1}/3)")
                        time.sleep(wait)
                    
                    else:
                        raise
            
            # If we exhausted retries for this model, try next one
            if model != models_to_try[-1]:
                print(f"⚠️ {model} unavailable, trying fallback...")
        
        raise Exception("All models failed after retries")
    
    def _print_truth_pack_stats(self, tp: dict):
        """Show extraction stats."""
        print(f"\n📊 Truth Pack:")
        print(f"   Contributions: {len(tp.get('contributions', []))}")
        print(f"   Method details: {len(tp.get('method_summary', []))}")
        print(f"   Results: {len(tp.get('results', []))}")
        print(f"   Claims: {len(tp.get('claims_table', []))}")
    
    def _parse_script(self, script_text: str, title: str, truth_pack: dict) -> PodcastScript:
        """Parse raw script into structured format."""
        dialogue = []
        
        for line in script_text.split('\n'):
            line = line.strip()
            if line.startswith('Host 1:'):
                dialogue.append(DialogueTurn(speaker="host1", text=line[7:].strip()))
            elif line.startswith('Host 2:'):
                dialogue.append(DialogueTurn(speaker="host2", text=line[7:].strip()))
        
        story = StoryStructure(
            narrative_type="grounded",
            hook=truth_pack.get('core_thesis', ''),
            tension_points=[r.get('claim', '') for r in truth_pack.get('claims_table', [])[:5]],
            beats=[]
        )
        
        return PodcastScript(paper_title=title, story_structure=story, dialogue=dialogue)
    
    def _save_outputs(self, script: PodcastScript, output_dir: Path):
        """Save outputs."""
        with open(output_dir / "dialogue.txt", 'w') as f:
            for turn in script.dialogue:
                speaker = "Host 1" if turn.speaker == "host1" else "Host 2"
                f.write(f"{speaker}: {turn.text}\n\n")
        
        with open(output_dir / "story.json", 'w') as f:
            json.dump({
                "core_thesis": script.story_structure.hook,
                "key_claims": script.story_structure.tension_points,
                "dialogue_turns": len(script.dialogue)
            }, f, indent=2)
        
        print(f"💾 Saved: truth_pack.json, script_raw.txt, dialogue.txt, story.json")