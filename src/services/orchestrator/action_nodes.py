"""Action nodes for interview orchestrator.

This module contains all action nodes: greeting, question, followup,
sandbox_guidance, code_review, evaluation, and closing.
"""

import logging
import json
import re
import uuid
import time
from typing import TYPE_CHECKING
from datetime import datetime
from openai import AsyncOpenAI

from src.services.execution.sandbox_service import SandboxService, Language as SandboxLanguage
from src.services.analysis.code_metrics import get_code_metrics
from src.services.orchestrator.types import InterviewState, QuestionRecord, QuestionGeneration
from src.services.orchestrator.context_builders import (
    build_resume_context, build_conversation_context, build_job_context
)
from src.services.orchestrator.constants import (
    COMMON_SYSTEM_PROMPT, TERMINATION_MESSAGE,
    SANDBOX_POLL_INTERVAL_SECONDS, SANDBOX_STUCK_THRESHOLD_SECONDS,
    TEMPERATURE_CREATIVE, TEMPERATURE_BALANCED, TEMPERATURE_ANALYTICAL, TEMPERATURE_QUESTION,
    DEFAULT_MODEL,
    DEPTH_RULES, SENIORITY_MID, COVERAGE_IN_PROGRESS, COVERAGE_ADEQUATE,
    PROBE_DEPTH_DESCRIPTORS,
)
from src.services.orchestrator.plan_generator import (
    get_next_pending_topic, get_topic_by_id, update_topic_in_plan
)
logger = logging.getLogger(__name__)


class ActionNodeMixin:
    """Mixin containing all action node methods."""

    def _is_duplicate_question(
        self,
        question_text: str,
        state: InterviewState
    ) -> bool:
        """Check if question is duplicate using exact text match."""
        questions_asked = state.get("questions_asked", [])
        normalized = question_text.lower().strip()

        for q in questions_asked:
            if q["text"].lower().strip() == normalized:
                return True

        return False

    async def greeting_node(self, state: InterviewState) -> InterviewState:
        """Generate personalized greeting.

        Returns partial state update. Prevents duplicate greetings on reconnects.
        """
        # Check if greeting was already shown (state-based check)
        last_node = state.get("last_node", "")
        conv_history = state.get("conversation_history", [])

        # If greeting node was already executed, return existing greeting from history
        if last_node == "greeting":
            for msg in reversed(conv_history):
                if msg.get("role") == "assistant" and msg.get("content"):
                    return {
                        "last_node": "greeting",
                        "phase": "intro",
                        "next_message": msg.get("content"),
                    }

        resume_context = build_resume_context(state)
        job_context = build_job_context(state)
        candidate_name = state.get("candidate_name")

        # Generate interviewer persona from job description
        persona_prompt = f"""Based on this job description, create a realistic interviewer persona.

        Job Description:
        {job_context if job_context else "No specific job description provided."}

        Generate:
        1. A female name for the interviewer (e.g., Sarah, Emma, Maya, Priya, etc.)
        2. The company name (extract from job description or use a realistic tech company name)
        3. The interviewer's role/title (e.g., Senior Engineering Manager, Tech Lead, etc.)

        Return ONLY a JSON object with: "name", "company", "role"
        Example: {{"name": "Sarah", "company": "TechCorp", "role": "Senior Engineering Manager"}}"""

        try:
            persona_json = await self.llm_helper.call_llm_json(
                system_prompt="You are generating a realistic interviewer persona. Return only valid JSON.",
                user_prompt=persona_prompt,
                temperature=TEMPERATURE_BALANCED,
            )
            persona = json.loads(persona_json)
            interviewer_name = persona.get("name", "Sarah")
            company_name = persona.get("company", "our company")
            interviewer_role = persona.get("role", "Engineering Manager")
        except Exception as e:
            logger.warning(f"Failed to generate persona, using defaults: {e}")
            interviewer_name = "Sarah"
            company_name = "our company"
            interviewer_role = "Engineering Manager"

        name_context = ""
        if candidate_name:
            first_name = candidate_name.split()[0] if candidate_name else None
            if first_name:
                name_context = f"\nCandidate's name: {first_name}\nYou can refer to them by their first name ({first_name}) to make it more personal and friendly."

        prompt = f"""Generate a personalized greeting for the interview. You are {interviewer_name}, a {interviewer_role} at {company_name}.
        {name_context}
        Resume Context:
        {resume_context}

        Your greeting will be spoken aloud.
        - Introduce yourself: "Hi, I'm {interviewer_name}. I'm a {interviewer_role} at {company_name}."
        - Welcome them warmly and personally
        - Reference something brief from their resume if relevant
        - Keep it conversational and natural, not formal."""

        try:
            greeting = await self.llm_helper.call_llm_creative(
                system_prompt=COMMON_SYSTEM_PROMPT +
                f" You are {interviewer_name}, a {interviewer_role} at {company_name}. Welcome the candidate genuinely and personally. Be warm and conversational.",
                user_prompt=prompt,
            )
            return {
                "last_node": "greeting",
                "phase": "intro",
                "next_message": greeting,
            }
        except Exception:
            default_greeting = "Hello! Welcome to your interview. I'm looking forward to learning more about your background."
            return {
                "last_node": "greeting",
                "phase": "intro",
                "next_message": default_greeting,
            }

    async def question_node(self, state: InterviewState) -> InterviewState:
        """Generate the next question, driven by the interview plan when available.

        Plan-driven path:
          1. Get next pending topic from plan.
          2. Use topic's initial_question as the anchor.
          3. LLM crafts a natural, conversational version (not verbatim).
          4. Mark topic as in_progress and set current_topic_id.

        Reactive fallback (no plan):
          LLM generates freely based on resume + conversation (old behaviour).
        """
        conversation_context = build_conversation_context(state, self.interview_logger)
        resume_context = build_resume_context(state)
        job_context = build_job_context(state)
        candidate_name = state.get("candidate_name")
        name_note = (
            f"\nCandidate's first name: {candidate_name.split()[0]}"
            if candidate_name else ""
        )

        # ── Plan-driven path ──────────────────────────────────────────────────
        plan = state.get("interview_plan")
        next_topic = get_next_pending_topic(plan) if plan else None

        if next_topic:
            seniority = state.get("seniority_level") or plan.get("seniority_level", SENIORITY_MID)
            depth_rules = DEPTH_RULES.get(seniority, DEPTH_RULES[SENIORITY_MID])
            expected_depth = depth_rules["expected_depth"]

            last_user_response = ""
            for msg in reversed(state.get("conversation_history", [])):
                if msg.get("role") == "user":
                    last_user_response = msg.get("content", "")
                    break

            prompt = f"""You are conducting an interview. Move to the next planned topic.
{name_note}

NEXT TOPIC TO COVER: {next_topic['topic']}
CATEGORY: {next_topic['category']}
SUGGESTED OPENING: "{next_topic['initial_question']}"
CANDIDATE SENIORITY: {seniority} (expected depth: {expected_depth})

{job_context}
Resume Context:
{resume_context}

Recent Conversation:
{conversation_context[-2000:]}

CANDIDATE'S LAST ANSWER (if any): {last_user_response or "None yet (first question)"}

Your response will be spoken aloud.
STRUCTURE:
1. If there was a last answer: start with a brief, genuine acknowledgment (1 sentence, vary phrasing — e.g. "That's really interesting.", "I appreciate you sharing that.", "Good point.", "Nice approach.").
2. Then bridge naturally into the new topic. Don't say "Moving on to..." — weave it conversationally.
3. Ask the opening question for this topic. Use the suggested opening as inspiration but make it feel natural, not scripted. Adjust complexity for {seniority} level.

One question only. Keep it conversational."""

            try:
                response = await self.llm_helper.call_llm_with_instructor(
                    system_prompt=(
                        COMMON_SYSTEM_PROMPT +
                        f" Always respond in English. You are calibrating questions for a {seniority}-level candidate. "
                        f"Expected depth: {expected_depth}. Acknowledge briefly then transition naturally to the next topic."
                    ),
                    user_prompt=prompt,
                    response_model=QuestionGeneration,
                    temperature=TEMPERATURE_QUESTION,
                )

                question_text = response.question.strip()
                if self._is_duplicate_question(question_text, state):
                    question_text = next_topic["initial_question"]

                question_record: QuestionRecord = {
                    "id": str(uuid.uuid4()),
                    "text": question_text,
                    "source": "plan",
                    "resume_anchor": response.resume_anchor,
                    "aspect": next_topic["category"],
                    "asked_at_turn": state["turn_count"],
                    "planned_topic_id": next_topic["id"],
                }

                # Update plan: mark topic as in_progress
                updated_plan = update_topic_in_plan(plan, next_topic["id"], {
                    "coverage_status": COVERAGE_IN_PROGRESS,
                })

                # Update topics_covered list (simple string tracking, keep for analytics)
                topics_covered = list(state.get("topics_covered", []))
                if next_topic["topic"] not in topics_covered:
                    topics_covered.append(next_topic["topic"])

                return {
                    "last_node": "question",
                    "phase": "exploration",
                    "current_question": question_text,
                    "next_message": question_text,
                    "questions_asked": [question_record],
                    "current_topic_id": next_topic["id"],
                    "interview_plan": updated_plan,
                    "topics_covered": topics_covered,
                }

            except Exception as e:
                logger.error(f"Plan-driven question generation failed: {e}", exc_info=True)
                # Fall through to reactive path

        # ── Reactive fallback (no plan or plan exhausted) ─────────────────────
        topics_covered = state.get("topics_covered", [])
        questions_asked = [q["text"] for q in state.get("questions_asked", [])[-10:]]
        topics_info = (
            f"\nTopics Already Covered: {', '.join(topics_covered)}\nExplore new topics or go deeper."
            if topics_covered else ""
        )

        prompt = f"""You are conducting an interview. Generate the next response.
{name_note}
{job_context}
Resume Context: {resume_context}
{topics_info}

Full Conversation:
{conversation_context}

Questions Already Asked:
{chr(10).join(f"- {q}" for q in questions_asked) if questions_asked else "None yet"}

Your response will be spoken aloud.
STRUCTURE:
1. If there was a previous answer: acknowledge it briefly (1 sentence, vary phrasing).
2. Ask a natural question relevant to their background.
Skip acknowledgment if this is the very first question."""

        try:
            response = await self.llm_helper.call_llm_with_instructor(
                system_prompt=COMMON_SYSTEM_PROMPT + " Always respond in English. Acknowledge then ask. Be genuine.",
                user_prompt=prompt,
                response_model=QuestionGeneration,
                temperature=TEMPERATURE_QUESTION,
            )

            question_text = response.question.strip()
            if self._is_duplicate_question(question_text, state):
                question_text = "Can you tell me about a challenging project you've worked on?"

            question_record: QuestionRecord = {
                "id": str(uuid.uuid4()),
                "text": question_text,
                "source": "resume",
                "resume_anchor": response.resume_anchor,
                "aspect": response.aspect,
                "asked_at_turn": state["turn_count"],
                "planned_topic_id": None,
            }

            existing_questions = state.get("questions_asked", [])
            already_asked = any(q.get("text") == question_text for q in existing_questions)

            updates: dict = {
                "last_node": "question",
                "phase": "exploration",
                "current_question": question_text,
                "next_message": question_text,
            }
            if not already_asked:
                updates["questions_asked"] = [question_record]

            if response.resume_anchor and response.resume_anchor not in topics_covered:
                current_topics = list(topics_covered)
                current_topics.append(response.resume_anchor)
                updates["topics_covered"] = current_topics

            return updates

        except Exception as e:
            logger.error(f"Reactive question generation failed: {e}", exc_info=True)
            fallback = "Can you tell me about a challenging project you've worked on?"
            return {
                "last_node": "question",
                "phase": "exploration",
                "current_question": fallback,
                "next_message": fallback,
            }

    async def followup_node(self, state: InterviewState) -> InterviewState:
        """Generate follow-up question or clarification.

        Returns partial state update. conversation_history is written by finalize_turn_node.
        """
        last_question = state.get("current_question", "")
        last_answer = state.get("last_response", "")

        active_request = state.get("active_user_request")
        needs_clarification = (
            active_request and
            active_request.get("type") == "clarify"
        )

        # Get seniority-calibrated probe style
        seniority = state.get("seniority_level", SENIORITY_MID)
        depth_rules = DEPTH_RULES.get(seniority, DEPTH_RULES[SENIORITY_MID])
        expected_depth = depth_rules["expected_depth"]
        probe_style = depth_rules["probe_style"]
        probe_examples = PROBE_DEPTH_DESCRIPTORS.get(expected_depth, [])
        probe_hint = (
            f"Probe examples for {seniority} level: {', '.join(probe_examples[:2])}"
            if probe_examples else ""
        )

        # Get current topic context from plan
        plan = state.get("interview_plan")
        current_topic_id = state.get("current_topic_id")
        current_topic = get_topic_by_id(plan, current_topic_id) if plan and current_topic_id else None
        topic_context = (
            f"\nCurrent topic: {current_topic['topic']} (category: {current_topic['category']})"
            if current_topic else ""
        )

        if needs_clarification:
            prompt = f"""The candidate asked for clarification on this question: "{last_question}"

        Your response will be spoken aloud.
        - Briefly acknowledge their confusion (e.g. "Of course, let me rephrase that.")
        - Rephrase clearly and simply. Break it down if needed."""
        else:
            prompt = f"""Generate a depth-probe follow-up. The interviewer is pushing deeper on the current topic.

        CANDIDATE SENIORITY: {seniority} — probe style: {probe_style}
        {probe_hint}
        {topic_context}

        Previous Question: "{last_question}"
        Candidate's Answer: "{last_answer}"

        Full Conversation:
        {build_conversation_context(state, self.interview_logger)}

        Your response will be spoken aloud.
        STRUCTURE:
        1. Acknowledge the answer briefly (1 sentence, genuine, varied — not "Great!" every time).
        2. Probe DEEPER with a follow-up calibrated to {seniority} level. For {expected_depth} depth, ask about: {', '.join(probe_examples) if probe_examples else 'specifics, trade-offs, and real examples'}.
        Keep it conversational. One question only."""

        try:
            followup = await self.llm_helper.call_llm_creative(
                system_prompt=(
                    COMMON_SYSTEM_PROMPT +
                    f" Always respond in English. You are probing a {seniority}-level candidate. "
                    f"Probe style: {probe_style}. Acknowledge briefly then push deeper with a targeted follow-up."
                ),
                user_prompt=prompt,
            )
            question_record: QuestionRecord = {
                "id": str(uuid.uuid4()),
                "text": followup,
                "source": "followup",
                "resume_anchor": None,
                "aspect": "deep_dive",
                "asked_at_turn": state["turn_count"],
            }
            existing_questions = state.get("questions_asked", [])
            question_already_asked = any(
                q.get("text") == followup
                for q in existing_questions
            )

            updates = {
                "last_node": "followup",
                "current_question": followup,
                "next_message": followup,
            }
            if not question_already_asked:
                updates["questions_asked"] = [question_record]

            return updates
        except Exception:
            fallback = "Can you provide more details about that?"
            return {
                "last_node": "followup",
                "current_question": fallback,
                "next_message": fallback,
            }

    async def answer_candidate_question_node(self, state: InterviewState) -> InterviewState:
        """Answer a question the candidate asked about the company, role, or process.

        Returns partial state update. Answers directly then transitions back to interview.
        """
        candidate_question = state.get("last_response", "")
        last_interviewer_question = state.get("current_question", "")

        # Extract interviewer persona from conversation history
        interviewer_name = "the interviewer"
        company_name = "our company"
        interviewer_role = "Engineering Manager"
        conv_history = state.get("conversation_history", [])
        for msg in conv_history:
            if msg.get("role") == "assistant":
                content = msg.get("content", "")
                # Attempt to extract persona from greeting (e.g. "I'm Sarah, a ... at TechCorp")
                name_match = re.search(r"I'm ([A-Z][a-z]+)", content)
                company_match = re.search(r"at ([A-Z][A-Za-z\s]+?)[\.,]", content)
                role_match = re.search(r"(?:a |an )([A-Za-z\s]+?) at", content)
                if name_match:
                    interviewer_name = name_match.group(1)
                if company_match:
                    company_name = company_match.group(1).strip()
                if role_match:
                    interviewer_role = role_match.group(1).strip()
                break

        job_context = build_job_context(state)

        prompt = f"""The candidate has asked you a question during the interview. Answer it directly and professionally, then transition back to the interview.

Interviewer persona: {interviewer_name}, {interviewer_role} at {company_name}
{job_context}

Candidate's question: "{candidate_question}"
Last interview question you asked (to return to): "{last_interviewer_question}"

Your response will be spoken aloud.
- Answer the candidate's question clearly and concisely
- Be honest. If you don't have specific info (e.g. salary), say so briefly
- After answering, use a natural transition back to the interview (e.g. "Does that help? Now, going back to what I was asking...")
- Keep the answer brief — 2-4 sentences max before transitioning back"""

        try:
            answer = await self.llm_helper.call_llm_creative(
                system_prompt=COMMON_SYSTEM_PROMPT +
                f" You are {interviewer_name}, a {interviewer_role} at {company_name}. Always respond in English. Answer the candidate's question honestly and transition back to the interview naturally.",
                user_prompt=prompt,
            )
            return {
                "last_node": "answer_candidate_question",
                "next_message": answer,
            }
        except Exception:
            fallback = f"That's a good question. {company_name} is a great place to work and I'd be happy to tell you more after the interview. For now, let's continue — {last_interviewer_question}"
            return {
                "last_node": "answer_candidate_question",
                "next_message": fallback,
            }

    async def evaluation_node(self, state: InterviewState) -> InterviewState:
        """Generate comprehensive interview evaluation and feedback.

        Returns partial state update. conversation_history is written by finalize_turn_node.
        """

        try:
            topics_covered = state.get("topics_covered", [])
            comprehensive_feedback = await self.feedback_generator.generate_feedback(
                conversation_history=state.get("conversation_history", []),
                resume_context=state.get("resume_structured", {}),
                code_submissions=state.get("code_submissions", []),
                topics_covered=topics_covered,
                job_description=state.get("job_description"),
            )

            feedback_dict = comprehensive_feedback.model_dump()
            return {
                "last_node": "evaluation",
                "phase": "closing",
                "feedback": feedback_dict,
            }
        except Exception as e:
            logger.error(
                f"Failed to generate comprehensive feedback: {e}", exc_info=True)
            topics_covered = state.get("topics_covered", [])
            return {
                "last_node": "evaluation",
                "phase": "closing",
                "feedback": {
                    "summary": "Interview completed successfully.",
                    "topics_covered": topics_covered,
                    "turn_count": state.get("turn_count", 0),
                    "overall_score": 0.5,
                },
            }

    async def termination_node(self, state: InterviewState) -> InterviewState:
        """Immediately terminate the interview due to inappropriate behavior.

        Outputs TERMINATION_MESSAGE verbatim — no LLM call, no variation.
        Sets phase='terminated' so subsequent turns are short-circuited.
        """
        logger.warning(
            f"Interview {state.get('interview_id')} terminated for inappropriate behavior "
            f"at turn {state.get('turn_count', 0)}"
        )
        return {
            "last_node": "termination",
            "phase": "terminated",
            "next_message": TERMINATION_MESSAGE,
            "feedback": {
                **(state.get("feedback") or {}),
                "terminated": True,
                "terminated_reason": "inappropriate_behavior",
                "terminated_at_turn": state.get("turn_count", 0),
            },
        }

    async def closing_node(self, state: InterviewState) -> InterviewState:
        """Generate closing message.

        Returns partial state update. conversation_history is written by finalize_turn_node.
        """
        conversation_summary = build_conversation_context(
            state, self.interview_logger)

        prompt = f"""Generate a closing message for the interview.

        Conversation Summary:
        {conversation_summary}

        Your closing will be spoken aloud. Thank them genuinely. Reference something specific from the conversation if relevant. Be warm and authentic."""

        try:
            closing = await self.llm_helper.call_llm_creative(
                system_prompt=COMMON_SYSTEM_PROMPT +
                " You are closing an interview. Be warm and appreciative. Reference the conversation naturally.",
                user_prompt=prompt,
            )
            return {
                "last_node": "closing",
                "phase": "closing",
                "next_message": closing,
            }
        except Exception:
            return {
                "last_node": "closing",
                "phase": "closing",
                "next_message": "Thank you for your time today. It was great learning more about your background!",
            }

    async def sandbox_guidance_node(self, state: InterviewState) -> InterviewState:
        """Guide user to use sandbox for writing code, optionally providing an exercise.

        Returns partial state update.
        """
        if self.interview_logger:
            self.interview_logger.log_state("sandbox_guidance", state)

        sandbox = state.get("sandbox", {})
        should_provide_exercise = await self._should_provide_exercise(state)
        exercise = None
        if should_provide_exercise:
            exercise = await self._generate_coding_exercise(state)

        resume_context = build_resume_context(state)
        job_context = build_job_context(state)
        conversation_context = build_conversation_context(
            state, self.interview_logger)

        if should_provide_exercise:
            prompt = f"""Generate a message introducing a coding exercise to the candidate.

        {job_context}

        Resume Context:
        {resume_context}

        Exercise Description:
        {exercise.get('description', '')}

        Your message will be spoken aloud. 
        - Introduce the coding exercise naturally and conversationally
        - Tell them the code sandbox is already set up with starter code
        - Guide them to look at the sandbox on the right side of their screen
        - Be clear and supportive. Keep it natural."""
        else:
            prompt = f"""Generate a message guiding the candidate to use the code sandbox.

        Conversation Context:
        {conversation_context}

        Resume Context:
        {resume_context}

        Your message will be spoken aloud. 
        - Acknowledge their request to write code
        - Guide them to the code sandbox on the right side of their screen
        - Let them know you'll review it when they submit
        - Be natural and helpful. Keep it conversational."""

        try:
            guidance_message = await self.llm_helper.call_llm_creative(
                system_prompt=COMMON_SYSTEM_PROMPT +
                " You are guiding a candidate to use the code sandbox. Be clear and helpful.",
                user_prompt=prompt,
            )

            if self.interview_logger:
                self.interview_logger.log_llm_call(
                    "sandbox_guidance", prompt, guidance_message, "gpt-4o-mini"
                )

            sandbox = state.get("sandbox", {})
            sandbox_update = {
                **sandbox,
                "is_active": True,
                "last_activity_ts": time.time(),
            }
            if should_provide_exercise and exercise:
                sandbox_update.update({
                    "initial_code": exercise.get("starter_code", ""),
                    "exercise_description": exercise.get("description", ""),
                    "exercise_difficulty": exercise.get("difficulty", "medium"),
                    "exercise_hints": exercise.get("hints", []),
                })
                if "hints_provided" not in sandbox_update:
                    sandbox_update["hints_provided"] = []

            return {
                "last_node": "sandbox_guidance",
                "next_message": guidance_message,
                "sandbox": sandbox_update,
                # Signal to frontend: show the code editor panel NOW
                "show_code_editor": True,
            }
        except Exception as e:
            logger.error(
                f"Error generating sandbox guidance: {e}", exc_info=True)
            if self.interview_logger:
                self.interview_logger.log_error("sandbox_guidance", e)
            fallback = "Great! I'd love to see your code. Please use the code sandbox on the right side of your screen. Write your code there and submit it when you're ready, and I'll review it for you."
            return {
                "last_node": "sandbox_guidance",
                "next_message": fallback,
                "show_code_editor": True,
            }

    async def _should_provide_exercise(self, state: InterviewState) -> bool:
        """Determine if agent should provide a coding exercise.

        Plan takes precedence: if we generated a plan and it says no coding,
        we respect that (prevents code editor from popping up for non-tech roles).
        """
        active_request = state.get("active_user_request")
        if active_request and active_request.get("type") == "write_code":
            return True

        # Respect the interview plan's coding decision
        plan = state.get("interview_plan")
        if plan is not None:
            return plan.get("requires_coding", False)

        # Fallback heuristic (no plan available)
        job_desc = state.get("job_description", "").lower() if state.get("job_description") else ""
        coding_keywords = ["python", "javascript", "code", "programming", "developer", "engineer", "software"]
        if job_desc and any(keyword in job_desc for keyword in coding_keywords):
            return True

        conversation = build_conversation_context(state, self.interview_logger)
        return "technical" in conversation.lower() or "coding" in conversation.lower()

    async def _generate_coding_exercise(self, state: InterviewState) -> dict:
        """Generate a coding exercise based on job description and resume."""
        job_context = build_job_context(state)
        resume_context = build_resume_context(state)
        conversation_context = build_conversation_context(
            state, self.interview_logger)

        prompt = f"""Generate a coding exercise for an interview candidate.

        {job_context}

        Resume Context:
        {resume_context}

        Recent Conversation:
        {conversation_context}

        Create a coding exercise that:
        1. Is relevant to the job requirements
        2. Matches the candidate's experience level
        3. Can be completed in 15-30 minutes
        4. Tests practical programming skills

        Return a JSON object with:
        - "description": Clear problem description (2-3 sentences)
        - "starter_code": Python code with function signatures and docstrings, comments explaining the problem
        - "language": "python" or "javascript"
        - "difficulty": "easy", "medium", or "hard"
        - "hints": List of 2-3 hints if candidate gets stuck

        Example starter_code format:
        ```python
    def solve_problem(input_data):
        \"\"\"
        Problem: [description]

        Args:
        input_data: [description]

        Returns:
        [description]
        \"\"\"
        # TODO: Implement your solution here
        pass
        ```

        Return ONLY valid JSON, no markdown formatting."""

        try:
            exercise_json = await self.llm_helper.call_llm_json(
                system_prompt=COMMON_SYSTEM_PROMPT +
                " You are creating coding interview exercises. Generate practical, relevant problems.",
                user_prompt=prompt,
                temperature=TEMPERATURE_CREATIVE,
            )
            exercise = json.loads(exercise_json)
            return exercise
        except Exception as e:
            logger.error(f"Error generating exercise: {e}", exc_info=True)
            return {
                "description": "Implement a function that finds the maximum value in a list.",
                "starter_code": """def find_max(numbers):
        \"\"\"
        Find the maximum value in a list of numbers.

        Args:
        numbers: List of integers

        Returns:
        Maximum integer in the list
        \"\"\"
        # TODO: Implement your solution here
        pass""",
                "language": "python",
                "difficulty": "easy",
                "hints": ["Think about iterating through the list", "Keep track of the maximum value seen so far"]
            }

    async def check_sandbox_code_changes(self, state: InterviewState) -> dict:
        """Poll sandbox for code changes and provide real-time guidance if needed.

        Returns state updates (no mutations). Used by polling endpoint.
        """
        updates: dict = {}

        if not state.get("sandbox", {}).get("is_active"):
            return updates

        sandbox = state.get("sandbox", {})
        last_poll = sandbox.get("last_poll_time", 0.0)
        current_time = time.time()

        if current_time - last_poll < SANDBOX_POLL_INTERVAL_SECONDS:
            return updates

        current_code = state.get("current_code", "")
        last_snapshot = sandbox.get("last_code_snapshot", "")
        initial_code = sandbox.get("initial_code", "")
        last_activity_ts = sandbox.get("last_activity_ts", 0.0)

        # Simplified: Check if stuck based on time since last activity
        time_since_activity = current_time - \
            last_activity_ts if last_activity_ts > 0 else 0
        is_stuck = time_since_activity > SANDBOX_STUCK_THRESHOLD_SECONDS and current_code != initial_code

        sandbox_updates = {
            **sandbox,
            "last_poll_time": current_time,
        }

        # Track code changes
        if current_code and current_code != last_snapshot:
            if current_code != initial_code:
                sandbox_updates["last_code_snapshot"] = current_code
                sandbox_updates["last_activity_ts"] = current_time

        # Provide hints if stuck
        if is_stuck:
            exercise_hints = sandbox.get("exercise_hints", [])
            hints_provided = list(sandbox.get("hints_provided", []))

            if exercise_hints and len(hints_provided) < len(exercise_hints):
                next_hint_index = len(hints_provided)
                next_hint = exercise_hints[next_hint_index]
                hint_message = f"You seem to be stuck. Here's a hint to help you: {next_hint}"

                updates["next_message"] = hint_message
                hints_provided.append(next_hint)
                sandbox_updates["hints_provided"] = hints_provided

                signals = list(sandbox.get("signals", []))
                if "needs_help" not in signals:
                    signals.append("needs_help")
                    sandbox_updates["signals"] = signals

        if sandbox_updates != sandbox:
            updates["sandbox"] = sandbox_updates

        return updates

    async def code_review_node(self, state: InterviewState) -> InterviewState:
        """Execute code, analyze it, and generate feedback.

        Returns partial state update. conversation_history is written by finalize_turn_node.
        """
        if self.interview_logger:
            self.interview_logger.log_state("code_review_start", state)

        code = state.get("current_code")
        if not code:
            return {
                "last_node": "code_review",
                "next_message": "I don't see any code to review. Please submit your code.",
            }

        sandbox = state.get("sandbox", {})
        exercise_description = sandbox.get("exercise_description", "")
        initial_code = sandbox.get("initial_code", "")

        exercise_mismatch_note = ""
        if exercise_description and initial_code:
            try:
                check_prompt = f"""You are reviewing code submitted by a candidate. 

        EXERCISE PROVIDED:
        {exercise_description}

        STARTER CODE PROVIDED:
        ```python
        {initial_code[:500]}
        ```

        CODE SUBMITTED BY CANDIDATE:
        ```python
        {code[:1000]}
        ```

        Determine if the submitted code is an attempt to solve the exercise provided, or if it's completely different code.

        IMPORTANT: The candidate must solve the EXACT exercise provided. If the exercise asks for "task management API" but code implements "book management API", that's a MISMATCH. Only consider it a match if the code addresses the specific domain and requirements of the exercise.

        Return a JSON object with:
        - "matches_exercise": true/false
        - "reason": Brief explanation (1 sentence)

        If the code doesn't match, the candidate may have submitted unrelated code instead of working on the exercise."""

                check_result_json = await self.llm_helper.call_llm_json(
                    system_prompt=COMMON_SYSTEM_PROMPT +
                    " You are a code reviewer. Analyze if submitted code matches the exercise.",
                    user_prompt=check_prompt,
                    temperature=TEMPERATURE_ANALYTICAL,
                )
                check_result = json.loads(check_result_json)

                if not check_result.get("matches_exercise", True):
                    reason = check_result.get(
                        "reason", "The code doesn't appear to match the exercise.")
                    exercise_mismatch_note = f"\n\nNote: I notice you submitted code that doesn't match the exercise I provided ({exercise_description[:100]}...). I'll review what you submitted, but I'd also like to see your solution to the original exercise when you're ready."
            except Exception:
                pass

        sandbox = state.get("sandbox", {})
        sandbox_update = {
            **sandbox,
            "is_active": True,
            "last_activity_ts": datetime.utcnow().timestamp(),
            "signals": list(set(sandbox.get("signals", []) + ["code_submitted"])),
        }

        language_str = state.get("current_language", "python").lower()
        try:
            sandbox_language = SandboxLanguage(language_str)
        except ValueError:
            sandbox_language = SandboxLanguage.PYTHON

        try:
            execution_result = await self.sandbox_service.execute_code(
                code=code,
                language=sandbox_language,
            )

            exec_result_dict = execution_result.to_dict()
            # Will be included in return statement below

            conversation_summary = build_conversation_context(
                state, self.interview_logger)
            job_context = build_job_context(state)
            code_quality = await self.code_analyzer.analyze_code(
                code=code,
                language=language_str,
                execution_result=exec_result_dict,
                context={
                    "question": state.get("current_question", ""),
                    "conversation_summary": conversation_summary,
                    "job_description": job_context,
                },
            )

            code_quality_dict = {
                "quality_score": code_quality.quality_score,
                "correctness_score": code_quality.correctness_score,
                "efficiency_score": code_quality.efficiency_score,
                "readability_score": code_quality.readability_score,
                "best_practices_score": code_quality.best_practices_score,
                "strengths": code_quality.strengths,
                "weaknesses": code_quality.weaknesses,
                "feedback": code_quality.feedback,
                "suggestions": code_quality.suggestions,
            }

            feedback_message = await self.code_analyzer.generate_code_feedback_message(
                code_quality=code_quality,
                execution_result=exec_result_dict,
            )

            followup_question = await self.code_analyzer.generate_adaptive_question(
                code_quality=code_quality,
                execution_result=exec_result_dict,
                conversation_context=conversation_summary,
            )

            combined_message = f"{feedback_message}{exercise_mismatch_note}\n\n{followup_question}"

            submission = {
                "code": code,
                "language": language_str,
                "execution_result": exec_result_dict,
                "code_quality": code_quality_dict,
                "timestamp": datetime.utcnow().isoformat(),
            }

            sandbox_update = {
                **sandbox,
                "is_active": True,
                "last_activity_ts": datetime.utcnow().timestamp(),
                "submissions": sandbox.get("submissions", []) + [submission],
                "signals": list(set(sandbox.get("signals", []) + ["code_submitted"])),
            }

            try:
                metrics = get_code_metrics()
                metrics.record_execution(
                    user_id=state["user_id"],
                    interview_id=state["interview_id"],
                    code=code,
                    language=language_str,
                    execution_result=exec_result_dict,
                    code_quality=code_quality_dict,
                )
            except Exception:
                pass

            existing_submissions = state.get("code_submissions", [])
            code_already_submitted = any(
                sub.get("code") == submission.get("code")
                for sub in existing_submissions
            )

            updates = {
                "last_node": "code_review",
                "next_message": combined_message,
                "current_question": followup_question,
                "code_execution_result": exec_result_dict,
                "code_quality": code_quality_dict,
                "sandbox": sandbox_update,
            }
            if not code_already_submitted:
                updates["code_submissions"] = [submission]

            return updates

        except Exception as e:
            logger.error(f"Error in code review: {e}", exc_info=True)
            sandbox = state.get("sandbox", {})
            sandbox_update = {
                **sandbox,
                "signals": list(set(sandbox.get("signals", []) + ["execution_error"])),
            }
            return {
                "last_node": "code_review",
                "next_message": "I encountered an issue reviewing your code. Please try submitting it again.",
                "code_execution_result": {"error": str(e)},
                "sandbox": sandbox_update,
            }
