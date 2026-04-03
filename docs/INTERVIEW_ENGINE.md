# Interview Engine — Architecture & Design

> **Version**: 2.0 (Plan-Driven Architecture)
> **Last updated**: 2026-03-30
> **Status**: Active — update this file whenever the orchestrator, plan generator, depth engine, or frontend interview UI changes.

---

## Table of Contents

1. [Overview](#overview)
2. [High-Level Flow](#high-level-flow)
3. [LangGraph State Machine](#langgraph-state-machine)
4. [Plan Generation](#plan-generation)
5. [Depth Engine](#depth-engine)
6. [Intent Detection](#intent-detection)
7. [Action Nodes](#action-nodes)
8. [State Schema](#state-schema)
9. [Constants & Configuration](#constants--configuration)
10. [Frontend Integration](#frontend-integration)
11. [Data Persistence](#data-persistence)
12. [File Reference Map](#file-reference-map)
13. [Design Decisions & Trade-offs](#design-decisions--trade-offs)
14. [Known Limitations & Future Work](#known-limitations--future-work)

---

## Overview

The Menntr engine is a **plan-driven, seniority-aware** AI interviewer built on LangGraph.

### What changed from v1 (reactive) to v2 (plan-driven)

| Aspect | v1 (Reactive) | v2 (Plan-Driven) |
|---|---|---|
| Question selection | LLM picks freely each turn | Pre-generated plan with ordered topics |
| Depth control | None — moves on after 1 answer | Depth Engine: 1–3 follow-ups per topic based on quality |
| Seniority awareness | None | Seniority estimated upfront; calibrates all rules |
| Topic coverage | Random, often repetitive | Guaranteed coverage of planned topics |
| Code editor visibility | Always shown | Hidden by default; shown only when agent decides |
| Candidate questions | Ignored or mis-routed | Dedicated `candidate_question` intent + node |
| Plan persistence | N/A | Plan survives reconnects via `resume_context` |

---

## High-Level Flow

```
User starts interview
       │
       ▼
  ingest_input  ──── turn_count == 0 ──────► plan_interview ──► greeting
       │                                           │
       │ (subsequent turns)                        │ (plan stored in state + DB)
       ▼                                           │
  detect_intent ◄──────────────────────────────────
       │
       ▼
 decide_next_action  (Depth Engine runs here)
       │
       ├──► answer_candidate_question
       ├──► question          (advance to next topic)
       ├──► followup          (probe deeper on same topic)
       ├──► sandbox_guidance  (present coding exercise)
       ├──► code_review       (review submitted code)
       ├──► evaluation        (generate feedback)
       ├──► closing           (end interview)
       └──► termination       (immediate end for inappropriate behavior)
       │
       ▼
  finalize_turn  (single writer: saves to DB + updates plan tracking)
       │
       ▼
      END
```

Each turn is one complete pass through the graph. The graph is run via LangGraph's `StateGraph` with `MemorySaver` checkpointing so state is never lost across HTTP requests or reconnects.

---

## LangGraph State Machine

### Graph definition

**File**: `src/services/orchestrator/graph.py`

```
Nodes registered:
  initialize        — idempotent state setup
  ingest_input      — increments turn count
  plan_interview    — generates interview plan (once)
  detect_intent     — detects user intent
  decide_next_action — depth engine + routing (includes termination fast-path)
  greeting          — personalized opening
  question          — next planned topic question
  followup          — depth probe on current topic
  answer_candidate_question — answers "which company are you from?" etc.
  sandbox_guidance  — guides to code editor
  code_review       — reviews submitted code
  evaluation        — generates final feedback
  closing           — warm goodbye
  termination       — hard stop for inappropriate behavior (no LLM call)
  finalize_turn     — single-writer: persists state
```

### Routing logic

```
route_from_ingest():
  if turn_count == 0 AND no greeting in history  →  plan_interview
  elif code was just submitted                   →  code_review
  else                                           →  detect_intent

route_action_node():
  Maps NextActionDecision.action string to node name
  e.g. "question"     → question_node
       "followup"     → followup_node
       "termination"  → termination_node
```

### Checkpointing

Uses LangGraph's `MemorySaver`. Thread ID = `f"interview-{interview_id}"`. Each turn resumes from the same thread, preserving full state history in-memory. State is also persisted to the database after every turn via `state_to_interview()`.

---

## Plan Generation

**File**: `src/services/orchestrator/plan_generator.py`
**Triggered**: Once, before greeting (when `turn_count == 0`)

### What it generates

```python
InterviewPlan = {
    "topics": [TopicPlan, ...],      # 6–10 ordered topics
    "seniority_level": str,          # junior | mid | senior | staff_principal
    "expected_depth": str,           # foundational | intermediate | expert | principal
    "requires_coding": bool,         # should coding exercise be included?
    "coding_language": str | None,   # e.g. "python", "javascript"
    "target_turns": int,             # estimated interview length (8–25)
    "interview_style": str,          # TECHNICAL_HEAVY | BEHAVIORAL_HEAVY | BALANCED
}
```

### Each topic (TopicPlan)

```python
TopicPlan = {
    # Static (set at plan time)
    "id": str,                       # unique slug, e.g. "t1"
    "topic": str,                    # "System Design — Microservices"
    "category": str,                 # technical | behavioral | background | coding | ...
    "priority": int,                 # 1 = must-ask, 2 = important, 3 = optional
    "source": str,                   # "resume" | "job_description" | "general"
    "initial_question": str,         # LLM-generated anchor question
    "max_iterations": int,           # max follow-ups before moving on
    "min_quality_to_advance": float, # 0.0–1.0 quality threshold
    "requires_code": bool,

    # Runtime (updated each turn by finalize_turn_node)
    "coverage_status": str,          # pending | in_progress | covered | skipped
    "iterations_done": int,
    "last_quality_score": float,
}
```

### Seniority estimation

`_estimate_seniority()` asks the LLM:
- Years of experience from resume
- Seniority signals from job description (e.g. "Staff Engineer", "Principal")
- Returns one of: `junior`, `mid`, `senior`, `staff_principal`

### Topic generation

`_generate_topics()` instructs the LLM to:
- Extract key themes from resume (projects, skills, technologies)
- Extract requirements from job description
- Order topics from warm-up → depth
- Set `max_iterations` and `min_quality_to_advance` per topic based on seniority
- Flag topics that require code demonstration

### Fallback

If LLM fails: `_fallback_topics()` returns 3 generic topics (background, technical, problem-solving) with conservative thresholds.

---

## Depth Engine

**File**: `src/services/orchestrator/control_nodes.py` — `decide_next_action_node()` and `_depth_engine_decide()`

The Depth Engine is the core of v2. It decides whether to:
- **Stay on the current topic** and probe deeper (followup)
- **Advance to the next topic** (question)

### Decision tree

```
decide_next_action_node():

1. Check override intents (immediate routing, skip depth engine):
   - candidate_question  →  answer_candidate_question
   - write_code          →  sandbox_guidance
   - review_code         →  sandbox_guidance
   - stop                →  evaluation
   - clarify             →  followup (clarification mode)
   - technical_assessment → sandbox_guidance

2. Safety valve: if turn_count >= MAX_TURNS_BEFORE_EVALUATION (30)  →  evaluation

3. If interview_plan exists  →  run Depth Engine:
   a. Get current topic (by current_topic_id) or next pending topic
   b. Get answer_quality from state (set by previous node)
   c. Get seniority rules from DEPTH_RULES[seniority_level]
   d. Get topic-specific max_iterations and min_quality_to_advance
   e. Decision:
      - iterations_done < max_iterations
        AND answer_quality < min_quality_to_advance
        → "followup"  (need deeper answer)
      - else
        → "question"  (advance to next topic)
   f. If no more pending topics AND priority-1 topics all covered → "evaluation"

4. Fallback: LLM-driven NextActionDecision (reactive, no plan)
```

### Depth rules per seniority

Defined in `constants.py` under `DEPTH_RULES`:

| Seniority | Technical max_iter | Behavioral max_iter | min_quality | Expected depth |
|---|---|---|---|---|
| junior | 2 | 1 | 0.40 | foundational |
| mid | 2 | 2 | 0.55 | intermediate |
| senior | 3 | 2 | 0.65 | expert |
| staff_principal | 3 | 2 | 0.75 | principal |

Higher seniority = higher bar to advance + more follow-ups allowed before moving on.

### Topic iteration tracking

`finalize_turn_node()` updates the plan after every turn:
- Increments `iterations_done` for the current topic
- Updates `last_quality_score`
- Sets `coverage_status` = `"covered"` when depth engine decides to advance

---

## Intent Detection

**File**: `src/services/orchestrator/intent_detection.py`

Runs every turn (except first). Identifies what the candidate *wants to do*, not just what they said.

### Intent types

| Intent | Meaning | Action triggered |
|---|---|---|
| `write_code` | Wants to write/demo code | sandbox_guidance |
| `review_code` | Wants to share existing code | sandbox_guidance |
| `change_topic` | Wants to redirect conversation | (LLM decides) |
| `clarify` | Confused, needs help | followup (clarification) |
| `technical_assessment` | Wants different format | sandbox_guidance |
| `candidate_question` | Asking about company/role/process | answer_candidate_question |
| `stop` | Wants to end interview | evaluation |
| `continue` | Normal affirmation | (no override) |
| `rude_or_inappropriate` | Sexual, abusive, or threatening content | **termination** |
| `no_intent` | Just answering | (no override) |

### Confidence thresholds

- `>= 0.9` — Very clear intent → override routing
- `0.7–0.89` — Clear intent → set as `active_user_request`
- `< 0.7` — Ambiguous → treat as `no_intent`

### Decision framework used in the LLM prompt

1. What is the user's **goal**? (DO something / SAY something / CHANGE something)
2. What happens if we **ignore** this? (Breaks flow = request | Fine = answer)
3. Does it require **action**? (Yes = specific intent | No = `no_intent`)

---

## Termination Protocol

When a candidate uses sexual, abusive, or threatening language, the interview is **immediately and hard-terminated** — no LLM call, no softening, no warning.

### Detection — two layers

**Layer 1 — Fast keyword pre-check** (synchronous, no LLM):

`_is_inappropriate_content(text)` in `control_nodes.py` scans the last response against `INAPPROPRIATE_PATTERNS` from `constants.py`. Catches explicit content before the LLM even runs.

```python
# constants.py
INAPPROPRIATE_PATTERNS = [
    "have sex", "want sex", "sex with you", "fuck you", ...
]
```

**Layer 2 — LLM intent detection**:

`UserIntentDetection` now includes `rude_or_inappropriate` as an intent type. The detection prompt instructs the LLM that this intent takes the **highest priority** and must be flagged for any sexual, abusive, or threatening content.

### Routing

```
decide_next_action_node():

1. if phase == "terminated"  →  termination  (subsequent turns, no reprocessing)
2. if _is_inappropriate_content(last_response)  →  termination  (fast path)
3. if active_request.type == "rude_or_inappropriate"  →  termination  (LLM path)
```

### termination_node

No LLM call. Outputs `TERMINATION_MESSAGE` verbatim:

> "Your behavior, attitude, and manner are inappropriate and intolerable in a professional setting. This interview is now terminated."

Sets:
- `phase = "terminated"` — blocks any further normal processing
- `feedback.terminated = True` — persisted to DB for audit
- `feedback.terminated_at_turn = N`

### After termination — persistence across reconnects

`feedback.terminated = True` is saved to the DB by `state_to_interview()`. On reconnect, `interview_to_state()` checks this flag:

```python
"phase": "terminated" if (interview.feedback or {}).get("terminated") else "intro",
```

`decide_next_action_node` also checks `feedback.terminated` as a secondary guard (in case phase wasn't in-memory):

```python
already_terminated = (
    state.get("phase") == "terminated"
    or (state.get("feedback") or {}).get("terminated")
)
```

Any subsequent message routes back to `termination_node`, which outputs the same message again.

---

## Action Nodes

**File**: `src/services/orchestrator/action_nodes.py`

All nodes follow the same pattern:
1. Read state
2. Build LLM prompt
3. Call `llm_helper.generate()` or structured model
4. Return partial state update (only changed fields)

### Conversation style rules (all nodes)

- **Acknowledge first** — every response starts with a 1-sentence acknowledgment of the candidate's previous answer (varied: "Great point.", "That makes sense.", "Interesting approach.", etc.)
- **English only** — enforced via `COMMON_SYSTEM_PROMPT`
- **No bullet points in speech** — interviewer responses are written for TTS (spoken audio)
- **Short sentences** — suitable for voice delivery

### greeting_node

- Extracts interviewer persona (name, company, role) from job description
- Mentions the interview structure briefly
- Notes code editor availability if `requires_coding=True`
- Idempotent: skips if greeting already in conversation history

### question_node

Two paths:

**Plan-driven** (normal):
```
1. get_next_pending_topic(interview_plan)  →  next topic with status="pending"
2. Build prompt anchored to topic.initial_question
3. LLM crafts natural conversational version
4. Set current_topic_id, mark topic as "in_progress"
5. Record in questions_asked
```

**Reactive fallback** (no plan):
```
LLM generates question freely from resume + conversation history
```

Duplicate prevention: checks questions_asked before confirming.

### followup_node

- Reads `seniority_level` → looks up `DEPTH_RULES[seniority]` for `probe_style`
- Looks up `PROBE_DEPTH_DESCRIPTORS[seniority]` for example probe styles
- Generates a natural probe: "Can you walk me through how you'd handle X at scale?"
- Handles `clarify` intent: softer, supportive tone

### answer_candidate_question_node

- Detects interviewer persona from greeting message
- Answers the candidate's question honestly (makes up plausible company details if needed)
- Bridges back to interview: "Great question. Now, continuing from where we were..."

### sandbox_guidance_node

- Calls `_should_provide_exercise()`:
  - If plan has `requires_coding=True` and no exercise given yet → provide exercise
  - Else: invite candidate to use sandbox freely
- Calls `_generate_coding_exercise()` if exercise needed:
  - Uses seniority + primary language from plan
  - Returns: description, starter_code, hints, difficulty
- Sets `show_code_editor = True` in state → frontend shows code editor

### code_review_node

- Executes code via `SandboxService`
- Analyzes: correctness, efficiency, readability, best practices
- Detects exercise mismatch (prevents off-topic submissions)
- Generates feedback + follow-up question on the code
- Appends to `sandbox.submissions`

### evaluation_node

- Calls `feedback_generator.generate_feedback()`
- Includes: overall_score, summary, per-topic analysis, strengths, improvements
- Stores in `state["feedback"]`

### termination_node

- **No LLM call** — outputs `TERMINATION_MESSAGE` from `constants.py` verbatim
- Sets `phase = "terminated"` (blocks further normal processing)
- Sets `feedback.terminated = True` for DB audit trail
- Triggered by: keyword pre-check, `rude_or_inappropriate` intent, or re-entry when already terminated

---

## State Schema

**File**: `src/services/orchestrator/types.py`

### Key fields in InterviewState

```python
class InterviewState(TypedDict):
    # Identity
    interview_id: int
    user_id: int
    candidate_name: str | None

    # Conversation (APPEND-ONLY via operator.add reducer)
    conversation_history: list[dict]
    turn_count: int

    # Plan (set once by plan_interview_node, updated each turn)
    interview_plan: dict | None         # full InterviewPlan
    current_topic_id: str | None        # which topic we're on
    topic_iterations: dict[str, int]    # {topic_id: iterations_done}
    seniority_level: str | None         # junior | mid | senior | staff_principal
    show_code_editor: bool              # signal to frontend

    # Intent (APPEND-ONLY)
    detected_intents: list[UserIntent]
    active_user_request: UserIntent | None

    # Quality signal (set by finalize_turn)
    answer_quality: float               # 0.0–1.0

    # Code
    sandbox: SandboxState
    current_code: str | None
    code_execution_result: dict | None

    # Output
    next_message: str | None            # the message sent to candidate
    feedback: dict | None
```

### Reducer fields (append-only)

LangGraph reducers prevent overwrites on concurrent updates:
- `conversation_history` — `operator.add`
- `detected_intents` — `operator.add`
- `questions_asked` — `operator.add`
- `checkpoints` — `operator.add`

---

## Constants & Configuration

**File**: `src/services/orchestrator/constants.py`

All magic numbers, thresholds, and strings live here. Never hardcode values in nodes.

### Key constants

```python
# LLM
DEFAULT_LLM_MODEL = "gpt-4o-mini"
TEMP_CREATIVE = 0.8       # greetings, questions
TEMP_BALANCED = 0.7       # follow-ups
TEMP_ANALYTICAL = 0.3     # intent detection, evaluation
TEMP_QUESTION = 0.85      # question generation

# Seniority labels
SENIORITY_JUNIOR = "junior"
SENIORITY_MID = "mid"
SENIORITY_SENIOR = "senior"
SENIORITY_STAFF = "staff_principal"

# Interview flow
MIN_TURNS_BEFORE_CLOSING = 6
MAX_TURNS_BEFORE_EVALUATION = 30
SUMMARY_UPDATE_INTERVAL = 5        # update conversation summary every N turns

# Sandbox
SANDBOX_POLL_INTERVAL_SECONDS = 10.0
SANDBOX_STUCK_THRESHOLD_SECONDS = 30.0

# Behavior
TERMINATION_MESSAGE = "Your behavior, attitude, and manner are inappropriate..."

# Topic coverage statuses
COVERAGE_PENDING = "pending"          # Not yet discussed
COVERAGE_IN_PROGRESS = "in_progress"  # Currently being probed
COVERAGE_ADEQUATE = "adequate"        # Sufficient information gathered
COVERAGE_SKIPPED = "skipped"          # Skipped (time / not relevant)

# Priority levels
PRIORITY_MUST_ASK = 1       # Core to this role — must be covered
PRIORITY_SHOULD_ASK = 2     # Important — ask unless time is short
PRIORITY_NICE_TO_HAVE = 3   # Interesting but optional
```

---

## Frontend Integration

### show_code_editor signal

The backend controls when the code editor appears:

```
Backend state: show_code_editor = True
      ↓
state_to_interview() persists to resume_context["_show_code_editor"]
      ↓
_interview_to_response() reads it → InterviewResponse.show_code_editor = True
      ↓
API JSON: { "show_code_editor": true }
      ↓
Frontend Interview interface: show_code_editor: boolean
      ↓
page.tsx: {interview.show_code_editor && <CodeSandbox />}
```

**Files**:
- `src/schemas/interview.py` — `show_code_editor: bool = Field(False, ...)`
- `src/api/v1/endpoints/interviews.py` — `_interview_to_response()`
- `frontend/lib/api/interviews.ts` — `Interview` interface
- `frontend/app/dashboard/interviews/[id]/page.tsx` — conditional render

### Layout adaptation

When code editor is hidden (`show_code_editor=false`), the left panel (video + transcription) expands to full width:

```tsx
// frontend/app/dashboard/interviews/[id]/page.tsx
<div className={`${interview.show_code_editor ? 'w-1/3 border-r border-border' : 'w-full'} flex flex-col`}>
  {/* Video + Transcription */}
</div>

{interview.show_code_editor && (
  <div className="w-2/3 min-w-0 p-4">
    <CodeSandbox interviewId={interviewId} />
  </div>
)}
```

### Mic / Camera controls

`RoomControls` (mic mute + camera toggle) renders as soon as `roomInstance` exists, regardless of connection state. The component itself disables buttons when `room.state !== 'connected'`. This avoids the controls being hidden due to state timing issues on reconnect.

```tsx
{showVoiceVideo && roomInstance && (
  <div className="px-4 pb-2">
    <RoomControls room={roomInstance} />
  </div>
)}
```

### API polling

The frontend polls the interview endpoint every 2 seconds while `status === 'in_progress'`:

```typescript
refetchInterval: (query) => query.state.data?.status === 'in_progress' ? 2000 : false
```

This ensures `show_code_editor` and `current_message` stay in sync after each agent turn.

---

## Data Persistence

**File**: `src/services/data/state_manager.py`

### interview_to_state()

Converts `Interview` DB model → `InterviewState` for LangGraph. Called at the start of each turn.

Key operations:
- Sanitizes `conversation_history`: filters messages with wrong `interview_id` or timestamps before `interview.created_at`
- Restores plan fields from `resume_context` (keys prefixed with `_`):
  - `_interview_plan` → `interview_plan`
  - `_topic_iterations` → `topic_iterations`
  - `_seniority_level` → `seniority_level`
  - `_current_topic_id` → `current_topic_id`
  - `_show_code_editor` → `show_code_editor`

### state_to_interview()

Converts `InterviewState` → updates `Interview` DB model. Called after every turn via `finalize_turn_node`.

Key operations:
- Validates `state.interview_id == interview.id` (prevents cross-contamination)
- Persists plan fields back to `resume_context` with `_` prefix
- Updates `conversation_history`, `turn_count`, `feedback`

### Why resume_context?

The `Interview` model has a JSONB `resume_context` field. Plan state is stored there (with `_` prefix keys) to avoid schema migrations when adding new plan fields. This is a pragmatic design choice for early development — a dedicated `interview_plan` column should be considered for production.

---

## File Reference Map

```
src/
└── services/
    └── orchestrator/
        ├── constants.py          All config, thresholds, depth rules, messages
        ├── types.py              TypedDicts: InterviewState, TopicPlan, UserIntent, etc.
        ├── graph.py              LangGraph StateGraph definition + routing
        ├── plan_generator.py     Generates InterviewPlan + seniority estimation
        ├── control_nodes.py      Depth Engine, intent routing, finalize_turn
        ├── action_nodes.py       All content-generating nodes (question, followup, etc.)
        ├── intent_detection.py   User intent classification
        └── llm_helper.py         LLM wrapper (OpenAI, structured output)
    └── data/
        └── state_manager.py      DB ↔ LangGraph state conversion + plan persistence
└── schemas/
    └── interview.py              Pydantic API schemas (includes show_code_editor)
└── api/v1/endpoints/
    └── interviews.py             REST endpoints + _interview_to_response()
└── agents/
    └── resources.py              LiveKit agent resources (STT language="en" enforced)

frontend/
└── lib/api/
    └── interviews.ts             Interview TypeScript interface (show_code_editor: boolean)
└── app/dashboard/interviews/[id]/
    └── page.tsx                  Interview UI — conditional code editor, full-width layout
```

---

## Design Decisions & Trade-offs

### 1. Plan generated once, not per-turn

**Why**: Generating a new plan each turn wastes tokens and creates inconsistency. The plan is the interview blueprint — it should be stable.

**Trade-off**: If a candidate's answers reveal unexpected skill gaps, the plan doesn't adapt mid-interview. The Depth Engine partially compensates by allowing more follow-ups, but the topic list itself is fixed.

### 2. Plan stored in resume_context (not a dedicated column)

**Why**: Avoids DB migration for a field that is still evolving. Using `_`-prefixed keys in `resume_context` JSONB is a quick way to persist structured state.

**Trade-off**: Coupling plan state to resume context is semantically wrong and makes the code less readable. Should be migrated to a dedicated `interview_plan` JSONB column.

### 3. Depth Engine in control_nodes, not a separate service

**Why**: The depth engine reads directly from state and returns a routing decision. Keeping it inline in `decide_next_action_node` avoids unnecessary abstraction for now.

**Trade-off**: `decide_next_action_node` is complex. If the engine grows further (e.g., multi-topic backtracking), extract to `depth_engine.py`.

### 4. show_code_editor as a state signal (not a frontend decision)

**Why**: The backend has all the context (seniority, job role, plan.requires_coding). Letting the frontend decide when to show the editor would require duplicating that logic.

**Trade-off**: The frontend polls every 2 seconds to detect `show_code_editor` changes. This introduces ~2s latency between the agent deciding to show the editor and the user seeing it.

### 5. STT forced to English (language="en")

**Why**: Without language lock, Whisper auto-detects the spoken language. For candidates who speak with accent or mix languages, this caused Urdu/Hindi script in transcriptions.

**Trade-off**: Candidates who genuinely want to speak in another language cannot. This is acceptable for an English-only professional interview platform.

---

## Known Limitations & Future Work

### Not yet implemented

| Feature | Description | Priority |
|---|---|---|
| Web search for evaluation | When agent confidence < 0.6 on factual questions, call Tavily API to verify | High |
| Feedback per topic | `feedback_generator.py` still operates on whole conversation; should use `QuestionRecord.planned_topic_id` for per-topic scoring | High |
| Phase discipline | Phase transitions (warm_up → technical → behavioral → closing) are loosely managed; not strictly tied to plan progress | Medium |
| Plan adaptation | Plan topics are fixed at generation time; no mid-interview replanning based on revealed skill gaps | Medium |
| Dedicated DB column | `interview_plan` stored in `resume_context` JSONB with `_` prefix; should move to own column | Low |

### Known issues

- **2s polling delay** for `show_code_editor` signal: the code editor appears ~2 seconds after the agent signals it, due to frontend polling interval.
- **Duplicate greeting guard** is heuristic (checks for greeting text in history); a proper `phase` field would be more reliable.
- **answer_quality** is currently a placeholder float (0.0); the actual quality assessment LLM call needs to be wired into `finalize_turn_node` for the Depth Engine to function correctly at runtime.

---

*This document should be updated whenever any of the files in the File Reference Map are changed. Keep the flow diagrams, decision trees, and trade-off notes in sync with the code.*
