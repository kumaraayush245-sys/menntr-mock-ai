"""Control nodes for interview orchestrator.

Nodes: initialize, plan_interview, ingest_input, detect_intent,
       decide_next_action (with depth engine), finalize_turn.

The core design change vs the old system:
  OLD: decide_next_action_node asks an LLM "what should I do next?" — fully reactive,
       no memory of what was planned, no concept of "is this topic done?".

  NEW: decide_next_action_node runs a DEPTH ENGINE first:
    1. Look up the current plan topic and its iteration count + quality score.
    2. Apply seniority-calibrated rules to decide: push deeper OR advance to next topic.
    3. Only fall back to LLM for edge cases (candidate question, code request, etc.).
"""

import logging
import json
from typing import TYPE_CHECKING
from datetime import datetime
from openai import AsyncOpenAI

from src.services.orchestrator.types import InterviewState, NextActionDecision
from src.services.orchestrator.context_builders import (
    build_decision_context, build_conversation_context, build_resume_context
)
from src.services.orchestrator.constants import (
    COMMON_SYSTEM_PROMPT,
    SUMMARY_UPDATE_INTERVAL, MAX_CONVERSATION_LENGTH_FOR_SUMMARY,
    TEMPERATURE_ANALYTICAL, TEMPERATURE_BALANCED, DEFAULT_MODEL,
    DEPTH_RULES, SENIORITY_MID,
    COVERAGE_ADEQUATE, COVERAGE_IN_PROGRESS, COVERAGE_PENDING,
    PRIORITY_MUST_ASK, PRIORITY_SHOULD_ASK,
    MIN_TURNS_BEFORE_CLOSING, MAX_TURNS_BEFORE_EVALUATION,
    TERMINATION_MESSAGE, INAPPROPRIATE_PATTERNS,
    TOPIC_CODING,
)
from src.services.orchestrator.intent_detection import detect_user_intent
from src.services.orchestrator.plan_generator import (
    generate_interview_plan,
    get_next_pending_topic,
    get_topic_by_id,
    update_topic_in_plan,
)

logger = logging.getLogger(__name__)

# Intents that require special routing regardless of plan state
_OVERRIDE_INTENTS = {
    "write_code", "review_code", "candidate_question", "stop", "clarify",
    "technical_assessment", "rude_or_inappropriate",
}


def _is_inappropriate_content(text: str) -> bool:
    """Fast rule-based check for explicit/abusive content.

    Acts as a safety net BEFORE the LLM intent detection runs, catching
    obvious cases the LLM might hallucinate away or soften.
    """
    if not text:
        return False
    lowered = text.lower()
    return any(pattern in lowered for pattern in INAPPROPRIATE_PATTERNS)


class ControlNodeMixin:
    """Mixin containing all control node methods."""

    async def initialize_node(self, state: InterviewState) -> InterviewState:
        """Initialize interview state with required fields. Idempotent."""
        defaults = {
            "conversation_history": [],
            "questions_asked": [],
            "detected_intents": [],
            "checkpoints": [],
            "sandbox": {
                "is_active": False,
                "last_activity_ts": 0.0,
                "submissions": [],
                "signals": [],
                "hints_provided": [],
                "initial_code": "",
                "exercise_description": "",
                "exercise_difficulty": "medium",
                "exercise_hints": [],
                "last_code_snapshot": "",
                "last_poll_time": 0.0,
            },
            "turn_count": 0,
            "phase": "intro",
            "code_submissions": [],
            "conversation_summary": "No conversation yet.",
            "candidate_name": None,
            "current_question": None,
            "active_user_request": None,
            "answer_quality": 0.0,
            "last_node": "initialize",
            # New plan fields
            "interview_plan": None,
            "current_topic_id": None,
            "topic_iterations": {},
            "seniority_level": None,
            "show_code_editor": False,
        }

        updates = {}
        for key, default_value in defaults.items():
            if key not in state or state.get(key) is None:
                updates[key] = default_value
            elif key == "sandbox" and not isinstance(state.get(key), dict):
                updates[key] = default_value

        if "topics_covered" not in state:
            updates["topics_covered"] = []

        return updates

    # ─── NEW: Plan Interview Node ──────────────────────────────────────────────

    async def plan_interview_node(self, state: InterviewState) -> InterviewState:
        """Generate the interview blueprint ONCE before the greeting.

        Why: Without a plan, the LLM invents questions randomly each turn —
        inconsistent coverage, no depth calibration, no awareness of what's left.
        With a plan, every subsequent node knows what topic to cover and how deep to go.
        """
        # Skip if plan already exists (reconnect / state restore scenario)
        if state.get("interview_plan"):
            return {"last_node": "plan_interview"}

        try:
            plan = await generate_interview_plan(state, self.llm_helper)
            seniority = plan.get("seniority_level", SENIORITY_MID)

            # Initialize topic_iterations tracker from plan
            topic_iterations = {
                topic["id"]: {
                    "iterations_done": 0,
                    "last_quality_score": None,
                }
                for topic in plan.get("topics", [])
            }

            logger.info(
                f"Interview plan ready: {len(plan.get('topics', []))} topics, "
                f"seniority={seniority}, coding={plan.get('requires_coding')}"
            )

            return {
                "last_node": "plan_interview",
                "interview_plan": plan,
                "seniority_level": seniority,
                "topic_iterations": topic_iterations,
                # Code editor starts HIDDEN — shown only when sandbox_guidance_node runs
                "show_code_editor": False,
            }

        except Exception as e:
            logger.error(f"Plan generation failed: {e}", exc_info=True)
            # Graceful fallback — interview continues without a plan (reactive mode)
            return {
                "last_node": "plan_interview",
                "show_code_editor": False,
            }

    # ─── Existing Nodes ───────────────────────────────────────────────────────

    async def ingest_input_node(self, state: InterviewState) -> InterviewState:
        """Ingest user input and code submission into state."""
        updates: dict = {"last_node": "ingest_input"}

        if state.get("last_response"):
            updates["turn_count"] = state.get("turn_count", 0) + 1

        updates["next_message"] = None
        return updates

    async def detect_intent_node(self, state: InterviewState) -> InterviewState:
        """Detect user intent from their last response."""
        if not state.get("last_response"):
            return {
                "active_user_request": None,
                "last_node": "detect_intent",
            }

        updates = await detect_user_intent(
            state,
            self.openai_client,
            self.interview_logger
        )

        return {**updates, "last_node": "detect_intent"}

    async def decide_next_action_node(self, state: InterviewState) -> InterviewState:
        """Decide the next action using the Depth Engine first, then LLM fallback.

        DEPTH ENGINE LOGIC:
        ┌───────────────────────────────────────────────────────────────┐
        │ 1. Check active_user_request for override intents             │
        │    (candidate_question, write_code, stop, clarify) → route   │
        │    directly without consulting the plan.                      │
        │                                                               │
        │ 2. Find current plan topic.                                   │
        │    - If topic iterations < max_iterations AND                 │
        │      last_quality_score < min_quality_to_advance:             │
        │      → followup  (push deeper on this topic)                 │
        │    - Else:                                                    │
        │      → question  (move to next topic in plan)                │
        │                                                               │
        │ 3. If no plan, fall back to LLM decision (old behavior).     │
        └───────────────────────────────────────────────────────────────┘
        """
        active_request = state.get("active_user_request")
        turn_count = state.get("turn_count", 0)

        # ── Guard: already terminated — don't process further ───────────────
        already_terminated = (
            state.get("phase") == "terminated"
            or (state.get("feedback") or {}).get("terminated")
        )
        if already_terminated:
            return {
                "next_node": "termination",
                "last_node": "decide_next_action",
                "answer_quality": 0.0,
            }

        # ── Fast pre-check: explicit/abusive content (before LLM intent) ───
        last_response = state.get("last_response", "")
        if _is_inappropriate_content(last_response):
            logger.warning(
                f"Inappropriate content detected via keyword filter at turn {turn_count}"
            )
            return {
                "next_node": "termination",
                "last_node": "decide_next_action",
                "answer_quality": 0.0,
            }

        # ── Override: special intents always take priority ──────────────────
        if active_request and active_request.get("type") in _OVERRIDE_INTENTS:
            intent_type = active_request["type"]
            action = {
                "write_code": "sandbox_guidance",
                "review_code": "code_review",
                "candidate_question": "answer_candidate_question",
                "stop": "evaluation",
                "clarify": "followup",
                "technical_assessment": "sandbox_guidance",
                "rude_or_inappropriate": "termination",
            }.get(intent_type, "followup")

            return {
                "next_node": action,
                "last_node": "decide_next_action",
                "answer_quality": 0.0,
            }

        # ── Force evaluation if max turns reached ───────────────────────────
        if turn_count >= MAX_TURNS_BEFORE_EVALUATION:
            return {
                "next_node": "evaluation",
                "last_node": "decide_next_action",
                "answer_quality": 0.0,
            }

        # ── Depth engine: plan-driven routing ───────────────────────────────
        plan = state.get("interview_plan")
        if plan:
            action, answer_quality = await self._depth_engine_decide(state, plan)
            return {
                "next_node": action,
                "last_node": "decide_next_action",
                "answer_quality": answer_quality,
            }

        # ── Fallback: LLM-driven routing (no plan available) ────────────────
        return await self._llm_driven_decide(state)

    async def _depth_engine_decide(
        self, state: InterviewState, plan: dict
    ) -> tuple[str, float]:
        """Apply depth engine rules to decide followup vs question vs evaluation.

        Returns (action_name, answer_quality_score).
        """
        seniority = state.get("seniority_level") or plan.get("seniority_level", SENIORITY_MID)
        depth_rules = DEPTH_RULES.get(seniority, DEPTH_RULES[SENIORITY_MID])

        # Analyze current answer quality if there's a response
        answer_quality = 0.0
        if state.get("last_response") and state.get("current_question"):
            try:
                analysis = await self.response_analyzer.analyze_answer(
                    state["current_question"],
                    state["last_response"],
                    {"resume_context": state.get("resume_structured", {})},
                )
                answer_quality = analysis.quality_score
            except Exception:
                pass

        # Get current topic from plan
        current_topic_id = state.get("current_topic_id")
        current_topic = None
        if current_topic_id:
            current_topic = get_topic_by_id(plan, current_topic_id)

        if current_topic:
            topic_category = current_topic.get("category", "technical")
            max_iterations = current_topic.get("max_iterations", 2)
            min_quality = current_topic.get("min_quality_to_advance", depth_rules["min_quality_to_advance"])

            topic_iter = state.get("topic_iterations", {}).get(current_topic_id, {})
            iterations_done = topic_iter.get("iterations_done", 0)

            # DEPTH ENGINE DECISION TREE:
            # If we have a pending answer to probe further:
            if state.get("last_response"):
                insufficient = answer_quality < min_quality
                can_probe_more = iterations_done < max_iterations

                if insufficient and can_probe_more:
                    logger.debug(
                        f"Depth engine: stay on '{current_topic['topic']}' — "
                        f"quality={answer_quality:.2f} < threshold={min_quality}, "
                        f"iterations={iterations_done}/{max_iterations}"
                    )
                    return "followup", answer_quality

                # Topic is adequately covered or exhausted → advance
                logger.debug(
                    f"Depth engine: advance from '{current_topic['topic']}' — "
                    f"quality={answer_quality:.2f}, iterations={iterations_done}/{max_iterations}"
                )

        # Check if all must-ask topics are covered → evaluate
        topics = plan.get("topics", [])
        pending_topics = [
            t for t in topics
            if t.get("coverage_status") in (COVERAGE_PENDING, COVERAGE_IN_PROGRESS)
            and t.get("priority") <= PRIORITY_SHOULD_ASK
        ]

        if not pending_topics and state.get("turn_count", 0) >= MIN_TURNS_BEFORE_CLOSING:
            return "evaluation", answer_quality

        # Advance to next topic (question node will pick from plan)
        return "question", answer_quality

    async def _llm_driven_decide(self, state: InterviewState) -> InterviewState:
        """Fallback LLM-driven routing when no plan is available."""
        decision_ctx = build_decision_context(state, self.interview_logger)
        conversation_context = build_conversation_context(state, self.interview_logger)

        answer_quality = 0.0
        if state.get("last_response") and state.get("current_question"):
            try:
                analysis = await self.response_analyzer.analyze_answer(
                    state["current_question"],
                    state["last_response"],
                    {"resume_context": state.get("resume_structured", {})},
                )
                answer_quality = analysis.quality_score
            except Exception:
                pass

        conversation_text = "\n".join([
            f"{msg.get('role', 'unknown').upper()}: {msg.get('content', '')}"
            for msg in state.get("conversation_history", [])[-20:]
        ])

        active_request = state.get("active_user_request")
        intent_info = ""
        if active_request:
            intent_info = (
                f"\nDetected Intent: {active_request.get('type')} "
                f"(confidence: {active_request.get('confidence', 0):.2f})\n"
            )

        prompt = f"""You are an experienced interviewer deciding the next action.

CONVERSATION:
{conversation_text}

STATE:
- Turn: {decision_ctx['turn']}, Phase: {decision_ctx['phase']}
- Questions: {decision_ctx['questions_count']}, Quality: {answer_quality:.2f}
- Sandbox: {'Active' if state.get('sandbox', {}).get('is_active') else 'Inactive'}
{intent_info}

AVAILABLE ACTIONS:
- greeting: First interaction only
- answer_candidate_question: PRIORITY if candidate asked a question about company/role/process
- question: New question (move to new topic)
- followup: Dig deeper into last answer
- sandbox_guidance: Guide to code sandbox
- code_review: Review submitted code
- evaluation: Final interview evaluation
- closing: End interview

Choose the most natural next action."""

        try:
            decision = await self.llm_helper.call_llm_with_instructor(
                system_prompt="You are an experienced interviewer. Make decisions based on natural conversation flow.",
                user_prompt=prompt,
                response_model=NextActionDecision,
                temperature=TEMPERATURE_BALANCED,
            )
            return {
                "next_node": decision.action,
                "last_node": "decide_next_action",
                "answer_quality": answer_quality,
            }
        except Exception as e:
            logger.warning(f"LLM decision failed: {e}")
            conversation_history = state.get("conversation_history", [])
            has_assistant = any(m.get("role") == "assistant" for m in conversation_history)
            return {
                "next_node": "greeting" if not has_assistant else "question",
                "last_node": "decide_next_action",
                "answer_quality": answer_quality,
            }

    async def finalize_turn_node(self, state: InterviewState) -> InterviewState:
        """Finalize the turn: write conversation history, update plan topic state.

        This is the SINGLE writer for conversation_history.
        Also updates the topic_iterations tracker and plan coverage status.
        """
        updates: dict = {"last_node": "finalize_turn"}

        # ── Write conversation messages ──────────────────────────────────────
        user_messages = []
        if state.get("last_response"):
            user_messages.append({
                "role": "user",
                "content": state["last_response"],
                "timestamp": datetime.utcnow().isoformat(),
            })

        assistant_messages = []
        if state.get("next_message"):
            assistant_messages.append({
                "role": "assistant",
                "content": state["next_message"],
                "timestamp": datetime.utcnow().isoformat(),
            })

        existing_history = state.get("conversation_history", [])
        messages_to_add = []

        def _is_duplicate(msg: dict, existing: list) -> bool:
            return any(
                e.get("role") == msg.get("role") and e.get("content") == msg.get("content")
                for e in existing
            )

        for msg in user_messages + assistant_messages:
            if not _is_duplicate(msg, existing_history):
                messages_to_add.append(msg)

        if messages_to_add:
            updates["conversation_history"] = messages_to_add

        # ── Update plan topic tracking ────────────────────────────────────────
        plan = state.get("interview_plan")
        current_topic_id = state.get("current_topic_id")
        answer_quality = state.get("answer_quality", 0.0)

        if plan and current_topic_id and state.get("last_response"):
            topic_iterations = dict(state.get("topic_iterations", {}))
            topic_entry = dict(topic_iterations.get(current_topic_id, {}))

            # Increment iterations and record quality
            new_iterations = topic_entry.get("iterations_done", 0) + 1
            topic_entry["iterations_done"] = new_iterations
            topic_entry["last_quality_score"] = answer_quality
            topic_iterations[current_topic_id] = topic_entry

            # Update coverage status in plan
            current_topic = get_topic_by_id(plan, current_topic_id)
            if current_topic:
                min_quality = current_topic.get("min_quality_to_advance", 0.5)
                max_iterations = current_topic.get("max_iterations", 2)

                if answer_quality >= min_quality or new_iterations >= max_iterations:
                    new_status = COVERAGE_ADEQUATE
                else:
                    new_status = COVERAGE_IN_PROGRESS

                updated_plan = update_topic_in_plan(plan, current_topic_id, {
                    "coverage_status": new_status,
                    "iterations_done": new_iterations,
                    "last_quality_score": answer_quality,
                })
                updates["interview_plan"] = updated_plan
                updates["topic_iterations"] = topic_iterations

        updates["last_response"] = None
        updates["next_node"] = None
        updates["current_code"] = None

        # ── Update conversation summary periodically ─────────────────────────
        conversation_history = state.get("conversation_history", [])
        current_summary = state.get("conversation_summary", "")
        turn_count = state.get("turn_count", 0)

        should_update = (
            turn_count % SUMMARY_UPDATE_INTERVAL == 0 or
            not current_summary or
            len(conversation_history) > MAX_CONVERSATION_LENGTH_FOR_SUMMARY
        )

        if should_update:
            recent_messages = conversation_history[-10:] if len(conversation_history) > 10 else conversation_history
            recent_context = "\n".join([
                f"{msg.get('role', 'unknown').upper()}: {msg.get('content', '')[:200]}"
                for msg in recent_messages
            ])
            prompt = f"""Summarize this interview conversation in 2-3 sentences. Focus on:
- Key topics discussed
- Candidate's main strengths or gaps observed
- Current phase of the interview

CURRENT SUMMARY: {current_summary or "None yet."}
RECENT MESSAGES: {recent_context}

Return only the summary text."""

            try:
                new_summary = await self.llm_helper.call_llm_analytical(
                    system_prompt="You are a conversation summarizer. Be concise and factual.",
                    user_prompt=prompt,
                )
                updates["conversation_summary"] = new_summary
            except Exception as e:
                logger.error(f"Failed to update summary: {e}", exc_info=True)

        return updates
