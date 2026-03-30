# InterviewLab — Technical Architecture & Design Document

> A comprehensive breakdown of every technology, strategy, and technique used in the platform — what it is, where it is used, and exactly why it was chosen.

---

## Table of Contents

1. [System Overview](#1-system-overview)
2. [High-Level Architecture](#2-high-level-architecture)
3. [Backend Stack](#3-backend-stack)
4. [LangGraph — Interview Orchestration Engine](#4-langgraph--interview-orchestration-engine)
5. [LiveKit — Real-Time Voice Communication](#5-livekit--real-time-voice-communication)
6. [OpenAI — Language, Speech & Structured Output](#6-openai--language-speech--structured-output)
7. [ElevenLabs](#7-elevenlabs)
8. [Redis — Caching & Async Infrastructure](#8-redis--caching--async-infrastructure)
9. [PostgreSQL & SQLAlchemy — Persistence Layer](#9-postgresql--sqlalchemy--persistence-layer)
10. [Authentication — JWT + bcrypt](#10-authentication--jwt--bcrypt)
11. [Code Sandbox — Docker Execution](#11-code-sandbox--docker-execution)
12. [Resume Processing — PDF + LLM Analysis](#12-resume-processing--pdf--llm-analysis)
13. [RAG vs In-Context Strategy](#13-rag-vs-in-context-strategy)
14. [Frontend Stack](#14-frontend-stack)
15. [Docker & Deployment Architecture](#15-docker--deployment-architecture)
16. [End-to-End Interview Flow](#16-end-to-end-interview-flow)
17. [Key Design Decisions & Trade-offs](#17-key-design-decisions--trade-offs)

---

## 1. System Overview

InterviewLab is an AI-powered mock interview platform that conducts fully autonomous, real-time voice interviews with candidates. A candidate uploads their resume, describes the target job, and joins a voice room where an AI interviewer:

- Greets them by name
- Asks relevant technical and behavioural questions based on their resume
- Responds intelligently to answers, asking follow-ups or probing deeper
- Presents live coding challenges
- Reviews submitted code with detailed feedback
- Closes the interview with a structured performance report

The platform is built around three pillars:
1. **Intelligence** — LangGraph-driven state machine that decides what to ask and when
2. **Voice** — LiveKit WebRTC + OpenAI TTS/STT for natural spoken conversation
3. **Persistence** — PostgreSQL stores the full interview history, scores, and resume data

---

## 2. High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                          FRONTEND (Next.js)                         │
│  Browser ─── LiveKit Client SDK ─────────────────────────────────┐  │
│  (React)      (WebRTC audio)                                      │  │
└──────────────────────────────────────────────────────────────────┼──┘
                         REST API (axios)   WebRTC audio tracks     │
                              │                                     │
┌─────────────────────────────▼───────────────────────────────────────┐
│                       BACKEND (FastAPI)                             │
│                                                                     │
│   /api/v1/auth         /api/v1/interviews    /api/v1/voice          │
│   /api/v1/resumes      /api/v1/sandbox                              │
└──────────────────────────┬──────────────────────────────────────────┘
                           │
         ┌─────────────────┼─────────────────────┐
         │                 │                     │
┌────────▼────────┐ ┌──────▼──────┐  ┌──────────▼─────────┐
│   PostgreSQL     │ │    Redis    │  │   LiveKit Server   │
│  (interviews,   │ │  (sessions, │  │  (WebRTC rooms,    │
│   users, resume)│ │   cache)    │  │   audio relay)     │
└─────────────────┘ └─────────────┘  └──────────┬─────────┘
                                                 │ agent connects
                                      ┌──────────▼─────────┐
                                      │   LiveKit Agent    │
                                      │ (interview_agent)  │
                                      │                    │
                                      │  STT ──► LangGraph │
                                      │  LangGraph ──► TTS │
                                      └────────────────────┘
```

---

## 3. Backend Stack

### FastAPI (`src/main.py`)

**What**: Python async web framework.

**Why**: FastAPI is chosen over Flask or Django for three key reasons:
- **Native async/await**: All database, Redis, and LLM calls are I/O-bound. FastAPI with `asyncio` allows thousands of concurrent requests without threads.
- **Automatic OpenAPI docs**: Pydantic schemas generate `/docs` and `/redoc` automatically — essential for API-first development.
- **Type safety**: Request/response types are validated at the framework boundary via Pydantic, eliminating an entire category of bugs.

**Where**:
- `src/main.py` — App factory, CORS middleware, lifespan hooks (DB pool init, Redis init)
- `src/api/v1/router.py` — Route registration
- `src/api/v1/endpoints/` — Auth, interviews, voice, sandbox, resumes

### Pydantic v2 + Pydantic Settings

**What**: Data validation library and settings management.

**Why**: Pydantic v2 is significantly faster than v1 (Rust core). Settings management via `BaseSettings` lets us load from `.env` files with type coercion and validation on startup — fail fast if `DATABASE_URL` is missing rather than failing at first DB query.

**Where**: `src/core/config.py`, `src/schemas/`

### Uvicorn

**What**: ASGI server for FastAPI.

**Why**: Only ASGI-compliant server that supports async Python natively. Alternatives (gunicorn, waitress) are WSGI-only and would block the event loop.

**Where**: Process entrypoint — `uvicorn src.main:app --reload --port 8000`

---

## 4. LangGraph — Interview Orchestration Engine

### What is LangGraph?

LangGraph is a library for building stateful, graph-based LLM applications. Instead of a simple chain (A → B → C), it lets you define a **directed graph** where nodes are processing steps and edges can be conditional. This is fundamentally different from LangChain's sequential chains because:
- State is explicitly typed and can be accumulated (not just passed through)
- Routing is dynamic (the graph decides the next node at runtime)
- State can be checkpointed and resumed mid-conversation

### Why LangGraph for Interviews?

An interview is inherently **non-linear and stateful**:
- The interviewer must remember all previous questions asked
- The next action depends on what the candidate just said (follow-up? new topic? code challenge?)
- State must persist across many voice turns
- Different "modes" exist: greeting, exploration, technical, coding, closing

A simple LLM prompt chain would have no memory or routing intelligence. LangGraph handles all of this explicitly.

### State Definition (`src/services/orchestrator/types.py`)

```python
class InterviewState(TypedDict):
    # Identity
    interview_id: str
    user_id: str
    candidate_name: str

    # Conversation (append-only via operator.add reducer)
    conversation_history: Annotated[list, operator.add]
    turn_count: int

    # Questions (append-only — never lose track of what was asked)
    questions_asked: Annotated[list, operator.add]
    current_question: str

    # Resume context
    resume_structured: dict       # parsed resume sections
    topics_covered: list

    # Job context
    job_description: str

    # Intent detection
    detected_intents: Annotated[list, operator.add]
    active_user_request: str

    # Sandbox / Code
    sandbox: SandboxState         # code submissions, signals, exercises

    # Flow control
    phase: str                    # intro | exploration | technical | closing
    last_node: str
    next_node: str

    # Analysis
    answer_quality: dict
    code_quality: dict
    feedback: dict

    # Runtime (single-turn outputs)
    next_message: str             # what the AI will say next
    current_code: str             # code being reviewed this turn
    code_execution_result: dict
```

**Key design choice — `Annotated[list, operator.add]`**: Fields like `conversation_history`, `questions_asked`, and `detected_intents` use LangGraph's **reducer pattern**. When a node updates this field, the new value is *appended* to the existing list rather than replacing it. This makes state updates safe for concurrent nodes and prevents accidental data loss.

### Graph Structure (`src/services/orchestrator/graph.py`)

```
START
  │
  ▼
ingest_input_node         ← Process the user's raw text/code input
  │
  ▼
[First turn?] ──yes──► greeting_node ──► finalize_turn_node ──► END
  │
  no
  ▼
[Code submitted?] ──yes──► code_review_node ──► finalize_turn_node ──► END
  │
  no
  ▼
detect_intent_node        ← LLM classifies user intent (8 categories)
  │
  ▼
decide_next_action_node   ← LLM decides which action is most appropriate
  │
  ▼ (conditional routing)
┌─────────────────────────────────────────────────────────┐
│  question_node       — Ask a new interview question      │
│  followup_node       — Follow up on previous answer      │
│  sandbox_guidance_node — Guide through coding exercise   │
│  code_review_node    — Review submitted code             │
│  evaluation_node     — Assess answer quality             │
│  closing_node        — End the interview                 │
└──────────────────────────────┬──────────────────────────┘
                               │
                               ▼
                      finalize_turn_node   ← Commit turn to history
                               │
                               ▼
                              END
```

### Nodes (`src/services/orchestrator/nodes.py`, `action_nodes.py`, `control_nodes.py`)

| Node | Purpose | LLM call? |
|------|---------|-----------|
| `ingest_input_node` | Parse raw input, extract code vs text | No |
| `detect_intent_node` | Classify intent (write_code, technical_answer, etc.) | Yes |
| `decide_next_action_node` | Choose action given phase + intent | Yes |
| `greeting_node` | Personalized opening message | Yes |
| `question_node` | Generate next question from resume context | Yes |
| `followup_node` | Probe deeper on previous answer | Yes |
| `sandbox_guidance_node` | Hint or guide a coding challenge | Yes |
| `code_review_node` | Detailed code review with CodeAnalyzer | Yes |
| `evaluation_node` | Score the candidate's answer | Yes |
| `closing_node` | Summarize and close the interview | Yes |
| `finalize_turn_node` | Append to conversation_history, increment turn_count | No |

### Checkpointing — `MemorySaver`

**What**: LangGraph's built-in in-process checkpointer that serializes graph state after each node execution.

**Why**: Interviews span many turns (20-50+). The graph state must be preserved between turns — the LangGraph process cannot re-execute from scratch each time. `MemorySaver` stores the serialized state in memory, keyed by `thread_id` (which equals `interview_id`).

**Concurrency isolation**: Each interview has its own `thread_id`, so 100 concurrent interviews have 100 isolated state machines with zero crosstalk.

**Where**: `src/services/orchestrator/langgraph_orchestrator.py`
```python
self.graph = build_interview_graph().compile(checkpointer=MemorySaver())
config = {"configurable": {"thread_id": interview_id}}
result = self.graph.invoke(state, config)
```

### Instructor Library — Structured LLM Outputs

**What**: Patches the OpenAI client to return validated Pydantic model instances instead of raw text.

**Why**: LLM outputs are non-deterministic strings. When we need structured data (intent classification, resume parsing, code scores), we need guaranteed JSON that matches a schema. Instructor + Pydantic eliminates JSON parsing errors and schema mismatches entirely.

**Where**: `src/services/orchestrator/intent_detection.py`, `src/services/analysis/feedback_generator.py`, `src/services/data/resume_parser.py`

```python
client = instructor.patch(openai.AsyncOpenAI())
intent = await client.chat.completions.create(
    model="gpt-4o-mini",
    response_model=IntentDetectionResult,   # Pydantic model
    messages=[...]
)
# intent is now a typed IntentDetectionResult, not a string
```

---

## 5. LiveKit — Real-Time Voice Communication

### What is LiveKit?

LiveKit is an open-source WebRTC infrastructure platform. It provides:
- A **server** that manages rooms, participants, and audio/video tracks
- A **client SDK** (`livekit-client`) for browsers/mobile to join rooms
- An **agents SDK** (`livekit-agents`) for building bots that join rooms as participants

### Why LiveKit over alternatives?

| Alternative | Why not chosen |
|------------|---------------|
| Twilio | Expensive, closed-source, vendor lock-in |
| Agora | Closed-source, complex pricing |
| Daily.co | SaaS-only, no self-hosting |
| Raw WebRTC | Requires building signaling server, TURN/STUN infra from scratch |
| **LiveKit** | Open-source, self-hostable, first-class Python agent SDK, purpose-built for AI voice agents |

LiveKit is specifically designed for the "AI voice agent" use case — it has built-in abstractions for STT→LLM→TTS pipelines.

### Architecture

```
Browser (candidate)
    │
    │  WebRTC audio track (mic input)
    ▼
LiveKit Server (port 7880)
    │
    │  WebRTC audio track (relayed)
    ▼
LiveKit Agent (Python process)
    │
    ├── VAD (Silero) ──► detects speech activity
    ├── STT (OpenAI) ──► converts audio → text
    ├── LLM (OrchestratorLLM) ──► text → LangGraph → response text
    └── TTS (OpenAI) ──► response text → audio
    │
    │  Audio track sent back to room
    ▼
Browser (candidate hears AI response)
```

### Token-Based Room Access (`src/services/voice/livekit_service.py`)

**Why tokens**: Direct room access without authentication would be a security hole. LiveKit uses short-lived JWT tokens signed with the API secret. The backend generates a token when the user starts an interview — the frontend uses that token to join the room. The token encodes:
- Which room to join (`interview-{interview_id}`)
- What the participant can do (publish audio, subscribe to tracks)
- Expiry time

**Where**: `POST /api/v1/voice/token` calls `livekit_service.create_token()` which uses `livekit.api.AccessToken`.

### Agent Entrypoint (`src/agents/interview_agent.py`)

The agent runs as a **separate process** that listens for new rooms to join:

```python
@server.rtc_session()
async def entrypoint(ctx: JobContext):
    # Phase 1: Before connecting (bootstrap resources)
    interview_id = extract_interview_id(ctx.room.name)  # "interview-{id}"
    resources = await bootstrap_resources(interview_id)  # DB, LLM, TTS, STT, VAD

    # Phase 2: Connect to room
    await ctx.connect(auto_subscribe=AutoSubscribe.AUDIO_ONLY)

    # Phase 3: Start agent session
    session = AgentSession(
        vad=resources.vad,
        stt=resources.stt,
        llm=resources.orchestrator_llm,  # Our custom LLM bridge
        tts=resources.tts
    )
    await session.start(ctx.room)

    # Phase 4: Initial greeting
    greeting = await resources.orchestrator_llm.get_greeting(...)
    await session.say(greeting)

    # Phase 5: Monitor until interview ends
    await monitor_interview_completion(ctx, resources)
```

**Why two-phase init (before/after connect)**: Database and LLM initialization takes time. By doing it before connecting to the room, the agent is fully ready the moment it joins — no latency gap where the candidate hears silence.

### OrchestratorLLM Bridge (`src/agents/orchestrator_llm.py`)

LiveKit's `AgentSession` expects a `llm.LLM` interface. Our LangGraph orchestrator is not natively a LiveKit LLM. `OrchestratorLLM` is a bridge class that:
1. Implements LiveKit's `LLM` interface
2. Translates LiveKit's chat messages into `InterviewState` updates
3. Calls `LangGraphInterviewOrchestrator.execute_step()`
4. Returns the response as a LiveKit `LLMStream`

This adapter pattern decouples the LangGraph logic from LiveKit's specific API.

### VAD — Voice Activity Detection

**What**: Silero VAD (Voice Activity Detection) model that detects when a person starts/stops speaking.

**Why**: Without VAD, the STT service would receive a continuous audio stream and not know when to transcribe. VAD detects silence boundaries, segments audio into speech chunks, and triggers STT on complete utterances. This prevents mid-sentence transcriptions and reduces STT API costs (only transcribe real speech).

**Where**: `src/agents/resources.py` — Silero VAD loaded once per process and cached.

### TTS Text Normalization (`src/agents/tts_utils.py`)

**What**: Pre-processing pipeline that transforms LLM text output before sending to TTS.

**Why**: LLMs write for reading, not speaking. Raw LLM output contains formatting that sounds unnatural when spoken:
- `"Here are the topics: Python, SQL"` → TTS reads "colon" aloud
- `"Good answer — let's move on"` → em-dash sounds like a pause glitch

The normalizer applies rules: colon → period, em-dash → comma, etc., to improve speech prosody.

---

## 6. OpenAI — Language, Speech & Structured Output

### GPT-4o-mini (Primary LLM)

**Why `gpt-4o-mini` over `gpt-4o`**: Cost/quality trade-off. For interview conversations, structured intent detection, and question generation, `gpt-4o-mini` is sufficiently capable at a fraction of the cost. The system makes many LLM calls per turn (detect_intent, decide_action, generate_response) — using `gpt-4o` would make each interview prohibitively expensive.

**Where it is used**:
| Task | File |
|------|------|
| Intent detection | `src/services/orchestrator/intent_detection.py` |
| Action decision | `src/services/orchestrator/nodes.py` |
| Question generation | `src/services/orchestrator/action_nodes.py` |
| Follow-up generation | `src/services/orchestrator/action_nodes.py` |
| Code review | `src/services/analysis/code_analyzer.py` |
| Resume parsing | `src/services/data/resume_parser.py` |
| Feedback synthesis | `src/services/analysis/feedback_generator.py` |
| Answer evaluation | `src/services/analysis/response_analyzer.py` |

### OpenAI TTS (Text-to-Speech)

**Model**: `tts-1-hd` (high-definition) or `tts-1` (faster, lower latency)

**Why OpenAI TTS over ElevenLabs**: OpenAI TTS is directly integrated with the LiveKit agents SDK via `openai.tts.TTS()` — it is the path of least resistance for the LiveKit + OpenAI stack. ElevenLabs produces higher quality voice but requires a custom integration and introduces an additional API dependency.

**Voices available**: `alloy`, `echo`, `fable`, `onyx`, `nova`, `shimmer` (configurable via `OPENAI_TTS_VOICE` in `.env`)

**Where**: `src/services/voice/tts_service.py`, `src/agents/resources.py`

### OpenAI STT (Speech-to-Text) — Whisper

**Model**: OpenAI Whisper via LiveKit agents STT plugin

**Why**: Highest accuracy English STT available via API, works seamlessly with the LiveKit agents framework, handles accents and technical terminology (Python, SQL, REST API) well.

**Where**: `src/services/voice/stt_service.py`, `src/agents/resources.py`

---

## 7. ElevenLabs

**Status**: Configured in `.env` (`ELEVENLABS_API_KEY`) but **not currently implemented** in the active code path.

**Intended use**: ElevenLabs produces more natural, emotionally expressive voices than OpenAI TTS. It was planned as an upgrade to the TTS layer for more engaging interview delivery.

**Current status**: The system uses OpenAI TTS (`tts-1-hd`) for all speech synthesis. ElevenLabs integration would replace `src/services/voice/tts_service.py` and the TTS constructor in `src/agents/resources.py` via a LiveKit-compatible ElevenLabs TTS plugin.

---

## 8. Redis — Caching & Async Infrastructure

### What Redis Provides

Redis is an in-memory data store that supports key-value caching, pub/sub messaging, sorted sets, streams, and distributed locking — all with sub-millisecond latency.

### Where Redis is Configured

**Client**: `src/core/redis.py`
```python
class RedisClient:
    _client: Optional[aioredis.Redis] = None

    @classmethod
    async def get_client(cls) -> aioredis.Redis:
        if cls._client is None:
            cls._client = aioredis.from_url(
                settings.REDIS_URL,
                encoding="utf-8",
                decode_responses=True,
                socket_connect_timeout=5,
                socket_keepalive=True,
            )
        return cls._client
```

**Connection**: `redis://localhost:6381/0` (local dev), Docker service on port 6379 internally.

### Why Redis is in the Architecture

Even though Redis is not heavily used in the current interview flow, it is a critical infrastructure component for:

1. **Session caching**: Interview state (beyond what LangGraph's MemorySaver holds) can be cached in Redis for sub-millisecond lookups instead of hitting PostgreSQL on every turn.

2. **Rate limiting**: Per-user request throttling (e.g., max 5 concurrent interviews) — Redis sorted sets with TTL are the standard approach.

3. **LiveKit token cache**: Generated tokens can be cached briefly to avoid re-generating on page refreshes.

4. **Pub/Sub for real-time events**: When an interview completes, a Redis publish event can notify the frontend (via SSE or WebSocket) without polling.

5. **Distributed locks**: When multiple agent processes are running, Redis distributed locks prevent two agents from processing the same interview simultaneously.

6. **Future scaling**: The moment the system scales beyond a single API process, Redis becomes the shared state store that all instances can read from, eliminating race conditions.

### Design Philosophy

Redis is provisioned as infrastructure from day one because retrofitting it into an existing distributed system is painful. Having it available means any of the above use cases can be implemented without architectural changes.

---

## 9. PostgreSQL & SQLAlchemy — Persistence Layer

### Why PostgreSQL over SQLite or MongoDB?

- **ACID transactions**: Interview state updates (turn_count, conversation_history, feedback) must be atomic. Partial writes would corrupt interview data.
- **JSON columns**: PostgreSQL's `JSONB` type stores structured data (conversation history, resume data, feedback scores) with indexing support — no need for a separate document store.
- **Relational integrity**: Foreign keys between `users`, `interviews`, and `resumes` enforce data consistency at the database level.
- **Async driver (asyncpg)**: Native async PostgreSQL driver that works with SQLAlchemy's async session — critical for non-blocking I/O in FastAPI.

### Schema Design

#### `users` table
Stores authentication credentials and profile data. Passwords are stored as bcrypt hashes (never plaintext). `is_verified` supports future email verification flow.

#### `interviews` table
The core entity. Key columns:
- `conversation_history` (JSON): Array of `{role, content, timestamp}` objects — the full interview transcript
- `resume_context` (JSON): Parsed resume data used during the interview
- `feedback` (JSON): Final scores — communication, technical, problem-solving, code_quality
- `status` (enum): `pending → in_progress → completed`
- `turn_count`: How many exchanges have occurred

#### `resumes` table
Stores uploaded PDF metadata and extracted structured data:
- `file_path`: Path on disk (or cloud storage)
- `extracted_data` (JSON): `{profile, experience, education, projects, hobbies}` — parsed by GPT-4o-mini

### SQLAlchemy Async (`src/core/database.py`)

```python
engine = create_async_engine(DATABASE_URL, echo=False)
AsyncSessionLocal = async_sessionmaker(engine, class_=AsyncSession)
```

**Why async sessions**: Database queries in interview processing (load interview, update turn, save history) must not block the event loop. Using async sessions means 50 concurrent interviews can all be awaiting DB operations simultaneously without thread exhaustion.

### Alembic — Database Migrations (`alembic/`)

**Why**: Schema changes are inevitable. Alembic tracks migration history, allows rollbacks, and ensures all environments (dev, staging, prod) have the same schema. Migrations are idempotent (check for column existence before adding) to handle partial deployments safely.

---

## 10. Authentication — JWT + bcrypt

### Flow

```
POST /api/v1/auth/register
  → validate email/password (Pydantic)
  → hash password: bcrypt.hashpw(password, bcrypt.gensalt(rounds=12))
  → store user in DB

POST /api/v1/auth/login
  → load user by email
  → verify: bcrypt.checkpw(input_password, stored_hash)
  → generate JWT: jose.jwt.encode({"sub": user_id, "exp": now+30min}, SECRET_KEY, "HS256")
  → return access_token

GET /api/v1/... (protected)
  → extract "Authorization: Bearer <token>" header
  → decode JWT: jose.jwt.decode(token, SECRET_KEY, algorithms=["HS256"])
  → validate exp claim
  → load user from DB
  → inject as Depends(get_current_user)
```

### Why JWT over sessions?

The backend is stateless (no server-side session storage needed for auth). JWTs are self-contained — the token itself encodes user identity and expiry. This works perfectly with the async FastAPI model and scales horizontally without a shared session store.

### Why bcrypt for passwords?

bcrypt is intentionally slow (12 rounds = ~250ms per hash). This makes brute-force attacks computationally expensive. Unlike MD5/SHA hashing, bcrypt has a built-in salt and work factor that can be increased as hardware improves.

### Frontend Auth State (`frontend/lib/store/auth-store.ts`)

Zustand store manages `user`, `isAuthenticated`, `isLoading`. The JWT is stored in `localStorage` and injected into every Axios request via an interceptor in `frontend/lib/api/client.ts`. Next.js middleware redirects unauthenticated routes to `/login`.

---

## 11. Code Sandbox — Docker Execution

### What

When a candidate submits code during an interview, it must be executed in a secure, isolated environment. The sandbox service (`src/services/execution/sandbox_service.py`) runs submitted code inside a Docker container.

### Why Docker for Code Execution?

Running arbitrary user code directly on the server is a catastrophic security risk — an attacker could read files, make network requests, or crash the process. Docker provides:
- **Filesystem isolation**: Container cannot access the host filesystem
- **Network isolation**: Container has no network access by default
- **Resource limits**: Memory cap (e.g., 256MB), CPU quota, execution timeout (e.g., 10s)
- **Process isolation**: `kill` inside the container cannot affect the host

### Supported Languages

Code execution supports multiple languages via Docker images:
- Python (`python:3.11-slim`)
- JavaScript/Node.js (`node:18-slim`)
- Java, Go (extensible)

### Flow

```
POST /api/v1/sandbox/execute
  {code: "...", language: "python", timeout: 10}
  │
  ▼
SandboxService.execute()
  → Write code to temp file
  → docker run --rm --network=none --memory=256m --cpus=0.5 \
               -v /tmp/code:/code python:3.11-slim python /code/solution.py
  → Capture stdout, stderr, exit_code
  → Return within timeout (kill container if exceeded)
  │
  ▼
POST /api/v1/sandbox/submit
  → Execute code as above
  → Send code + output to CodeAnalyzer (GPT-4o-mini)
  → Return structured feedback: correctness, efficiency, style, suggestions
```

### Docker Socket Mount

The API container mounts `/var/run/docker.sock` to spawn sibling containers (not nested Docker). This is a common pattern for CI/CD and sandbox systems.

---

## 12. Resume Processing — PDF + LLM Analysis

### PDF Extraction (`pdfplumber`)

**Why `pdfplumber` over `PyPDF2` or `pdfminer`**: `pdfplumber` handles complex PDF layouts (columns, tables) better and provides cleaner text extraction with configurable page margins.

**Where**: `src/services/data/resume_parser.py`

```python
with pdfplumber.open(file_path) as pdf:
    text = "\n".join(page.extract_text() for page in pdf.pages)
```

### Structured Extraction (GPT-4o-mini + Instructor)

Raw PDF text is unstructured. GPT-4o-mini with Instructor extracts it into a typed Pydantic model:

```python
class ResumeStructured(BaseModel):
    profile: ProfileSection       # name, email, phone, summary
    experience: list[ExperienceItem]  # company, role, dates, achievements
    education: list[EducationItem]    # degree, institution, GPA
    projects: list[ProjectItem]       # name, tech stack, description
    skills: list[str]                 # extracted skill keywords
    hobbies: list[str]
```

This structured data is then stored in `resumes.extracted_data` (JSON column) and loaded into `InterviewState.resume_structured` at interview start. LangGraph nodes use it to:
- Generate questions specific to the candidate's experience
- Identify technical topics to probe
- Personalize the greeting with the candidate's name

---

## 13. RAG vs In-Context Strategy

### What is RAG?

Retrieval-Augmented Generation (RAG) embeds documents into a vector database and retrieves relevant chunks at query time using semantic similarity search.

### Why InterviewLab Does NOT Use RAG

| Criteria | RAG | InterviewLab Approach |
|---------|-----|----------------------|
| Data source | Large external corpus | Single candidate resume (~1-2 pages) |
| Data size | Too large for context window | Fits entirely in context window |
| Retrieval needed? | Yes — can't include all docs | No — include entire resume |
| Latency | Extra round-trip to vector DB | Zero — data already in state |
| Complexity | Vector DB + embedding model | Just a JSON field |

**The resume is small enough to include entirely in the LLM context window**. RAG adds latency, infrastructure complexity (vector DB, embeddings), and retrieval noise for no benefit when the source document is a single resume.

### What is Used Instead?

**In-Context Learning**: The full parsed resume (`resume_structured`) is included in the `InterviewState` and injected into every LLM prompt. The LLM has complete resume knowledge at all times without retrieval.

**State Accumulation**: LangGraph's append-only reducers build a growing context of the conversation — the LLM always knows every question asked and answer given, enabling coherent multi-turn reasoning.

---

## 14. Frontend Stack

### Next.js 16.1 + React 19.2 + TypeScript

**Why Next.js**: App Router provides file-based routing, server-side rendering for the landing page (SEO), and layout nesting for the dashboard. React Server Components reduce client-side JavaScript bundle.

**Why TypeScript**: Type safety across API calls prevents entire classes of runtime errors (undefined properties, wrong field names). Pydantic models on the backend translate directly to TypeScript interfaces.

### State Management — Zustand

**Why Zustand over Redux or Context API**:
- Redux is overkill for this app's state complexity (just auth + interview data)
- Context API re-renders the entire tree on state changes
- Zustand is minimal, has hooks-based API, and only re-renders components that subscribe to changed slices

**Used for**: `useAuthStore` — user identity, login/logout, loading states

### Data Fetching — TanStack React Query

**Why React Query over SWR or plain `useEffect`**:
- Automatic caching with configurable staleness
- Background refetching keeps interview lists fresh
- Optimistic updates for instant UI feedback
- Loading/error states built-in

**Used for**: Interview lists, resume lists, analytics data

### LiveKit React Components + Client SDK

**Why**: LiveKit provides official React components (`@livekit/components-react`) that handle WebRTC complexity — mic permissions, audio track subscription, speaking indicators, connection state. Without this, implementing real-time audio in a browser requires significant WebRTC expertise.

**Where**: `frontend/app/dashboard/interviews/[id]/page.tsx` — the live interview room

### Monaco Editor

**Why Monaco over CodeMirror or plain `<textarea>`**:
- VSCode's editor engine — candidates get a familiar, professional IDE experience
- Syntax highlighting for all major languages
- IntelliSense, bracket matching, line numbers
- Dark/light theme support

**Where**: `frontend/app/dashboard/sandbox/page.tsx` — code submission during interviews

### Form Validation — React Hook Form + Zod

**Why this combination**:
- React Hook Form avoids re-rendering on every keystroke (uncontrolled inputs)
- Zod schemas define validation rules in TypeScript (shared with backend schemas conceptually)
- Together: performant, type-safe forms with minimal boilerplate

**Where**: Login, register, interview creation forms

### Framer Motion

**Why**: Declarative animation API for React. Used for page transitions, badge animations, and UI entrance effects. The alternative (CSS transitions + JavaScript event listeners) is much more verbose for complex animations.

### Recharts

**Where**: `frontend/app/dashboard/analytics/page.tsx`

Used to render interview performance analytics: skill score radar charts, improvement over time line charts. Recharts is composable (built on D3) with a React-friendly declarative API.

---

## 15. Docker & Deployment Architecture

### Services

```yaml
db (postgres:16-alpine):
  - External port 5434 → internal 5432
  - Data volume for persistence
  - Health check with pg_isready

redis (redis:7-alpine):
  - External port 6381 → internal 6379
  - Data volume for persistence
  - Health check with redis-cli ping

api (custom Dockerfile):
  - Python 3.11-slim base
  - External port 8003 → internal 8000
  - Depends on db + redis (waits for health checks)
  - Mounts docker.sock for sandbox execution

agent (custom Dockerfile):
  - Same Python 3.11-slim base
  - No external port (internal service)
  - Connects to LiveKit server
  - restart: unless-stopped (auto-restart on crash)
```

### Why Non-Standard Ports?

`5434` instead of `5432`, `6381` instead of `6379` — avoids conflicts with system-installed PostgreSQL or Redis instances on developer machines.

### Why Separate API and Agent Containers?

The LiveKit agent is a long-running process that handles real-time audio. The API server handles HTTP requests. They have different scaling characteristics:
- API: scale horizontally with load balancer
- Agent: scale based on number of concurrent interviews (each agent handles one interview)

Separating them allows independent scaling, independent restarts, and different resource limits.

### Code Sandbox via Docker Socket

The API container mounts `/var/run/docker.sock`, giving it the ability to launch sibling containers for code execution. This is the "Docker-out-of-Docker" pattern — safer than Docker-in-Docker (which requires privileged mode).

---

## 16. End-to-End Interview Flow

```
User Action                    System Response
───────────────────────────────────────────────────────────────────

1. Upload Resume
   POST /api/v1/resumes         pdfplumber extracts text
   (PDF file)                   GPT-4o-mini + Instructor → ResumeStructured
                                Stored in resumes.extracted_data (JSON)

2. Create Interview
   POST /api/v1/interviews      New Interview row: status=pending
   {title, job_description,     resume_context populated from resume
    resume_id}

3. Start Interview
   POST /api/v1/interviews/     Interview status → in_progress
   start                        Frontend calls POST /voice/token
                                LiveKit AccessToken generated for room
                                "interview-{id}"

4. Join Voice Room
   Frontend connects to         LiveKit Server creates room
   LiveKit room via SDK         Agent process detects new room participant
   (livekit-client)             Agent bootstraps: DB session, OrchestratorLLM,
                                TTS, STT, Silero VAD

5. Agent Connects + Greets
   Room connection established  LangGraph: ingest_input → greeting_node
                                GPT-4o-mini generates personalized greeting
                                OpenAI TTS converts text → MP3 audio
                                Audio streamed to candidate via WebRTC

6. Candidate Speaks
   Microphone audio → LiveKit   Silero VAD detects speech start/end
   Server → Agent               OpenAI Whisper transcribes audio → text
                                Text passed to OrchestratorLLM.chat()

7. LangGraph Processes Turn
   (per turn, 5-8 LLM calls)
                                ingest_input_node: parse text/code
                                detect_intent_node: classify (technical_answer,
                                  asking_question, write_code, etc.)
                                decide_next_action_node: choose response type
                                [question|followup|code_review|closing]_node:
                                  generate AI response
                                finalize_turn_node: append to history,
                                  increment turn_count
                                Save state to DB (conversation_history,
                                  turn_count, resume_context)

8. AI Responds
                                next_message extracted from state
                                tts_utils normalizes text for speech
                                OpenAI TTS synthesizes audio
                                Audio streamed to candidate via WebRTC

9. Code Challenge
   Candidate types in Monaco    POST /sandbox/execute
   Editor and submits           Docker container runs code (isolated,
                                  network=none, memory=256MB, timeout=10s)
                                stdout/stderr/exit_code returned to frontend
                                POST /sandbox/submit triggers code_review_node
                                CodeAnalyzer + GPT-4o-mini reviews code
                                Feedback spoken by agent

10. Interview Completion
    Candidate says "I'm done"   detect_intent → closing intent
    or N turns reached          closing_node generates farewell + summary
                                FeedbackGenerator synthesizes full interview
                                Scores computed: communication, technical,
                                  problem_solving, code_quality (0-100)
                                Interview status → completed
                                completed_at timestamp set
                                Agent disconnects from room

11. View Results
    GET /interviews/{id}        Feedback JSON returned with scores,
                                  strengths, improvement areas
    Analytics dashboard         GET /interviews/analytics/skills/averages
                                Recharts renders radar/line charts
```

---

## 17. Key Design Decisions & Trade-offs

### Decision 1: LangGraph over prompt chaining

**Chosen**: LangGraph state machine with explicit nodes and conditional routing.

**Alternative**: Single mega-prompt that handles all interview logic.

**Why LangGraph**: A single prompt cannot maintain growing state (conversation history, questions asked, phases) reliably across 30-50 turns. LangGraph provides explicit state management, deterministic routing, and checkpointing. The interview graph is inspectable and debuggable in a way that a massive prompt is not.

**Trade-off**: More complex setup, more code. Worth it for production reliability.

---

### Decision 2: OpenAI TTS over ElevenLabs

**Chosen**: OpenAI TTS (`tts-1-hd`)

**Alternative**: ElevenLabs (higher quality voice)

**Why OpenAI TTS**: Native LiveKit agents integration, single API key for all AI calls, lower latency. ElevenLabs' voice quality is superior but requires custom LiveKit plugin work.

**Trade-off**: Slightly less natural voice. Acceptable for an MVP/portfolio system.

---

### Decision 3: In-Context resume (no RAG)

**Chosen**: Full resume JSON in LangGraph state

**Alternative**: Embed resume into vector DB, retrieve chunks per turn

**Why no RAG**: Resume is small (< 2KB structured JSON), fits trivially in context window. RAG adds latency, a vector DB service, and embedding costs for zero benefit on single-document contexts.

**Trade-off**: For future multi-document knowledge bases (company policies, role descriptions), RAG would become necessary.

---

### Decision 4: Docker sandbox for code execution

**Chosen**: Spawn isolated Docker containers per submission

**Alternative**: Subprocess isolation (restricted Python exec), third-party sandboxes (Judge0, Piston)

**Why Docker**: Full OS-level isolation without trusting application-level sandboxes. Docker is already a project dependency for infrastructure, so no new tools needed.

**Trade-off**: Container startup time (~200-500ms). Acceptable for non-competitive code evaluation.

---

### Decision 5: JWT stateless auth over session cookies

**Chosen**: JWT in localStorage, injected via Axios interceptor

**Alternative**: Server-side sessions stored in Redis

**Why JWT**: Stateless backend simplifies horizontal scaling. No shared session store needed for authentication. Works naturally with the single-page app architecture.

**Trade-off**: Tokens cannot be invalidated before expiry (logout is client-side only). For a portfolio system, this is acceptable. Production systems would use short-lived JWTs + refresh tokens stored in Redis.

---

*Document generated from codebase analysis — reflects actual implementation as of project state.*
