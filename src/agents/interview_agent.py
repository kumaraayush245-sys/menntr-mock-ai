"""LiveKit agent for interview voice conversations.

This agent connects to LiveKit rooms and handles voice conversations
by integrating with the interview orchestrator.

CRITICAL: This module must import in <100ms for LiveKit handshake.
All heavy imports are lazy-loaded after handshake completes.
"""

import asyncio
import logging
from pathlib import Path
from typing import TYPE_CHECKING

# Load .env into os.environ before LiveKit SDK reads LIVEKIT_URL / credentials
from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent.parent / ".env")

# Module-level imports: only standard library and LiveKit core
# Heavy imports are deferred until after handshake to meet <100ms requirement
from livekit.agents import (
    Agent,
    AgentServer,
    AgentSession,
    AutoSubscribe,
    JobContext,
    cli,
    room_io,
)

# Plugins MUST be imported at module level (main thread) before any job subprocess runs.
# LiveKit Agents v1.x enforces: Plugin.register_plugin() raises RuntimeError if called
# from a non-main thread, which happens when imports are deferred into async entrypoints.
from livekit.plugins import openai as _openai_plugin  # noqa: F401
from livekit.plugins import silero as _silero_plugin  # noqa: F401

# Type hints are conditionally imported to avoid runtime overhead
if TYPE_CHECKING:
    from src.agents.resources import AgentResources

# Ensure src directory is in Python path for imports
# Note: In production, this should be handled via PYTHONPATH environment variable
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

logger = logging.getLogger(__name__)

# AgentServer is lightweight and safe to instantiate at module level
server = AgentServer()


@server.rtc_session()
async def entrypoint(ctx: JobContext):
    """Entry point for the LiveKit agent job.

    Production-ready pattern (per LiveKit best practices):
    1. Extract metadata BEFORE connection (room.name available without connecting)
    2. Bootstrap resources BEFORE ctx.connect() (agent ready before handshake)
    3. Connect to room (handshake completes, agent is already initialized)
    4. Start session with all resources ready
    5. Clean up on exit

    This ensures frontend doesn't show agent participant before it's ready to listen.
    """
    # Extract interview ID from room name before establishing connection
    # Room name format: "interview-{id}" and is available without connecting
    try:
        interview_id = int(ctx.room.name.replace("interview-", ""))
    except ValueError:
        logger.error(
            f"Could not extract interview_id from room name: {ctx.room.name}")
        return

    # Bootstrap all resources before connection to ensure agent is ready
    # This follows LiveKit best practices: agent should be initialized before handshake
    # so the frontend doesn't display the agent participant until it's ready to listen

    # Defer heavy imports until after metadata extraction but before connection
    from src.agents.resources import bootstrap_resources

    resources: "AgentResources | None" = None
    try:
        resources = await bootstrap_resources(ctx, interview_id)
    except Exception as e:
        logger.error(f"Bootstrap failed before connection: {e}", exc_info=True)
        return

    # Connect to room after bootstrap completes
    # Agent is now fully initialized, so frontend won't display it until ready
    await ctx.connect(auto_subscribe=AutoSubscribe.AUDIO_ONLY)

    # Handshake complete: safe to perform heavy operations and start session

    try:
        resources.session = AgentSession(
            vad=resources.vad,  # Required for OpenAI non-streaming STT
            stt=resources.stt,
            llm=resources.orchestrator_llm,
            tts=resources.tts,
        )

        agent = Agent(
            instructions=(
                "You are a professional interviewer conducting a technical interview. "
                "All your responses will be SPOKEN ALOUD. "
                "Use short, clear sentences and natural, conversational language. "
                "Ensure your questions are focused and easy to understand when spoken."
            ),
        )

        await resources.session.start(
            agent=agent,
            room=ctx.room,
            room_options=room_io.RoomOptions(
                audio_output=True,
                text_output=True,
            )
        )

        # Brief delay to ensure TTS audio output is initialized
        await asyncio.sleep(0.5)

        def handle_data_message(data_packet):
            """Handle data messages, including test audio requests."""
            try:
                import json
                if data_packet.user and data_packet.user.payload:
                    data = data_packet.user.payload
                    message = json.loads(data.decode('utf-8'))
                    if message.get('type') == 'test_audio' and resources.session:
                        test_message = prepare_text_for_tts(
                            "Hello! This is an audio test. Can you hear me clearly?"
                        )
                        asyncio.create_task(
                            resources.session.say(test_message))
            except Exception:
                pass

        ctx.room.on("data_received", handle_data_message)

        # The OrchestratorLLM handles the initial greeting automatically via
        # AgentSession's first LLM call (greeting_node fires when conversation_history
        # is empty). No manual session.say() needed — doing so causes a duplicate.

        # Monitor interview status and room connection state
        try:

            while True:
                await asyncio.sleep(5)

                try:
                    async with AsyncSessionLocal() as check_db:
                        result = await check_db.execute(
                            select(Interview).where(
                                Interview.id == interview_id)
                        )
                        interview = result.scalar_one_or_none()

                        if interview and interview.status == "completed":
                            # Clean up orchestrator state before exiting
                            if resources and resources.orchestrator_llm and resources.orchestrator_llm.orchestrator:
                                try:
                                    await resources.orchestrator_llm.orchestrator.cleanup_interview(interview_id)
                                except Exception as cleanup_error:
                                    logger.error(
                                        f"Failed to cleanup orchestrator: {cleanup_error}", exc_info=True)
                            break
                except Exception:
                    pass

                if not ctx.room or not ctx.room.isconnected():
                    break

        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error(f"Error in agent monitoring loop: {e}", exc_info=True)

    except Exception as e:
        logger.error(f"Agent entrypoint error: {e}", exc_info=True)
    finally:
        if resources:
            await resources.aclose()


if __name__ == "__main__":
    cli.run_app(server)
