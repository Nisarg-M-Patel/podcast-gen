# prompts.py
"""
Prompt templates for narrative-driven podcast generation.
"""

# ==============================================================================
# SINGLE-SHOT: ALL STAGES IN ONE CALL
# ==============================================================================

SINGLE_SHOT_PODCAST_PROMPT = """You are generating a complete podcast script from a research paper.

Paper Title: {title}

Abstract:
{abstract}

All Sections:
{all_sections}

Your task: Generate a complete podcast script in ONE structured output.

## Step 1: Discover Narrative Hooks (what's interesting?)
Identify 2-4 hooks that create TENSION or SURPRISE:
- Surprising findings (contradicts expectations)
- Novel approaches (differs from standard practice)
- Contradictions (disagrees with prior work)
- Mysteries solved

## Step 2: Choose Narrative Type & Structure
Based on hooks, pick ONE narrative type:
- "detective_story" - Mystery that gets solved
- "breakthrough" - Paradigm shift
- "comparison" - Systematic evaluation
- "simplification" - Complex → simple
- "warning" - Cautionary tale

Create 4-6 story beats with:
- Title (scene name)
- Focus (what this beat covers)
- Narrative role (establish_stakes, create_tension, reveal_solution, etc.)

## Step 3: Generate Dialogue
For each beat, write 6-10 dialogue turns between two hosts:

**Host 1 (The Explainer)**: Confident, teaches, uses analogies, no fillers
**Host 2 (The Curious One)**: Asks questions, reacts ("Oh!", "Wait—"), represents listener

Rules:
- Each turn < 100 words
- Ground claims in the paper content
- Include ONE moment of disagreement
- Build on previous beats

## Step 4: Add Natural Speech (sparse!)
Add to dialogue (sparingly, max 1 per 3 turns):
- Fillers: "Um", "Well", "So"
- Reactions: "Oh!", "Wow", "Huh" (Host 2)
- Self-repairs: "I mean—" (Host 1)
- [laughs], [chuckles]

## Output Format (JSON):
{{
  "narrative_type": "detective_story|breakthrough|comparison|simplification|warning",
  "hook": "One-sentence attention grabber",
  "beats": [
    {{
      "title": "Beat Title",
      "focus": "What this beat covers",
      "narrative_role": "establish_stakes|create_tension|reveal_solution|validate_claims|show_limitations|broader_impact",
      "dialogue": [
        {{"speaker": "host1", "text": "..."}},
        {{"speaker": "host2", "text": "..."}},
        ...
      ]
    }}
  ]
}}

Remember: Each paper tells a unique story. Don't force templates - let the interesting findings drive the structure.

Generate the complete podcast now:
"""

# ==============================================================================
# ORIGINAL MULTI-STAGE PROMPTS (kept for reference/fallback)
# ==============================================================================

DISCOVERY_PROMPT = """You are analyzing a research paper to find what makes it narratively INTERESTING for a podcast.

Paper Title: {title}

Abstract:
{abstract}

Available Sections:
{section_summaries}

Your task: Identify narrative hooks that create TENSION, SURPRISE, or INTRIGUE.

Look for:
1. **Surprising findings**: Results that contradict common assumptions or expectations
2. **Novel approaches**: Methods that differ significantly from standard practices
3. **Contradictions**: Where this paper disagrees with prior work
4. **Mysteries solved**: Problems that were previously unsolved
5. **Unexpected applications**: Uses of techniques in surprising domains
6. **Controversial claims**: Statements likely to spark debate

For each hook you find:
- Quote specific text (if available)
- Identify which sections contain evidence
- Explain WHY it's narratively interesting

Return JSON:
{{
  "hooks": [
    {{
      "type": "surprising_finding|contradiction|novel_approach|mystery|unexpected_application|controversial_claim",
      "description": "Clear description of what's interesting",
      "evidence_sections": ["section_id_1", "section_id_2"],
      "quote": "Exact quote from paper (if applicable)",
      "narrative_potential": "Why this creates tension/surprise for listeners"
    }}
  ]
}}

Focus on hooks that would make someone say "Wait, really?" or "How is that possible?"
"""

# ==============================================================================
# STAGE 2: STORY STRUCTURE BUILDING
# ==============================================================================

STORY_STRUCTURE_PROMPT = """You are a podcast producer creating a narrative arc from research paper discoveries.

Paper Title: {title}

Discovered Narrative Hooks:
{hooks}

Available Sections:
{section_list}

Your task: Design a story structure that transforms this paper into an engaging audio narrative.

Guidelines:
1. **Choose a narrative type** based on the hooks:
   - "detective_story": Mystery that gets solved
   - "breakthrough": Paradigm-shifting discovery
   - "comparison": Systematic evaluation of approaches
   - "simplification": Complex → simple solution
   - "warning": Cautionary tale about limitations

2. **Create an opening hook** (1-2 sentences) that immediately grabs attention

3. **Identify tension points**: 2-4 moments that create surprise or conflict

4. **Design story beats**: 4-7 segments that build a coherent narrative
   - Each beat should have a clear FOCUS (what it's about)
   - Each beat should have a NARRATIVE ROLE (why it's in this position)
   - Specify which sections to retrieve content from

Return JSON:
{{
  "narrative_type": "detective_story|breakthrough|comparison|simplification|warning",
  "hook": "Opening line that grabs attention",
  "tension_points": [
    "First surprising moment",
    "Second contradictory finding",
    "Unexpected limitation"
  ],
  "beats": [
    {{
      "title": "Scene-setting title",
      "focus": "Specific question or topic this beat addresses",
      "narrative_role": "establish_stakes|create_tension|reveal_solution|validate_claims|show_limitations|broader_impact",
      "retrieve_from_sections": ["section_id_1", "section_id_2"],
      "content_query": "Specific query to retrieve relevant chunks"
    }}
  ]
}}

Important: 
- Don't force every paper into the same structure
- Let the paper's unique findings shape the narrative
- Build toward the most interesting discoveries
- End with implications, not just "and that's the paper"
"""

# ==============================================================================
# STAGE 3: CONTEXTUAL SUMMARIZATION (Per Beat)
# ==============================================================================

CONTEXTUAL_SUMMARY_PROMPT = """You are summarizing a text chunk for a specific podcast segment.

Podcast Segment: {beat_title}
Segment Focus: {beat_focus}

Text Chunk:
{chunk_text}

Your task: Extract ONLY the information relevant to this segment's focus.

Guidelines:
- Be concise (2-3 sentences max)
- Focus on facts, not interpretation
- Include specific numbers, results, or claims if present
- If this chunk contains no relevant information, return empty string

Summary:
"""

# ==============================================================================
# STAGE 4: DIALOGUE GENERATION
# ==============================================================================

DIALOGUE_PROMPT = """You are writing dialogue for a two-host research podcast.

**Host Personas:**
- **Host 1 (The Explainer)**: Confident, articulate, teaches concepts. Uses analogies. No filler words.
- **Host 2 (The Curious One)**: Enthusiastic, asks questions, represents the listener. Reacts genuinely.

**Story Context:**
Narrative Type: {narrative_type}
Opening Hook: {hook}
Previous Beats: {previous_beat_summaries}

**Current Beat:**
Title: {beat_title}
Focus: {beat_focus}
Role: {narrative_role}

**Content Available:**
{contextual_summaries}

**Your Task:** Write 8-12 dialogue turns for this beat.

**Rules:**
1. Each turn < 100 words (natural conversation length)
2. Host 2 asks clarifying questions, reacts with "Oh!", "Wait—", "That's fascinating!"
3. Host 1 uses analogies to explain complex concepts
4. Include ONE moment of tension/disagreement (don't always agree)
5. Build on what was discussed in previous beats
6. Ground ALL claims in the provided content
7. NO XML tags, just natural dialogue

**Output Format:**
Host 1: [dialogue]
Host 2: [dialogue]
Host 1: [dialogue]
...

Begin:
"""

# ==============================================================================
# STAGE 5: DISFLUENCY INJECTION
# ==============================================================================

DISFLUENCY_PROMPT = """You are adding natural speech patterns to make podcast dialogue sound spontaneous.

Original Dialogue:
{dialogue}

Add SPARSE disfluencies:
- "Um", "Uh", "Well", "So" (use sparingly, max 1 per every 3 turns)
- "Mm-hmm", "Right", "Yeah" (Host 2 backchanneling)
- Self-repairs: "I mean—", "Well, actually—" (Host 1 only)
- Reactions: "Oh!", "Wow", "Huh" (Host 2)
- [laughs], [chuckles] (both hosts)

Important: 
- Too many fillers sounds robotic
- Use them to mark transitions or surprises
- Don't add disfluencies to technical explanations

Return the dialogue with disfluencies added, same format as input.
"""

# ==============================================================================
# RERANKING PROMPT (for retrieval)
# ==============================================================================

RERANK_PROMPT = """Score how relevant this text chunk is for answering the query.

Query: {query}

Text Chunk:
{chunk_text}

Score from 0.0 (completely irrelevant) to 1.0 (highly relevant).
Consider:
- Does it directly address the query?
- Does it provide supporting evidence?
- Is it background information that helps understanding?

Return ONLY a number between 0.0 and 1.0:
"""