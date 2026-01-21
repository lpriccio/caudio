"""MCP server for audio generation (TTS) and transcription (STT)."""

import logging
import os
import re
import sys
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP
from openai import OpenAI

# Load .env file from the project directory
load_dotenv(Path(__file__).parent.parent.parent / ".env")

# =============================================================================
# CONFIGURATION
# =============================================================================
# TTS backend: "openai" or "elevenlabs" (when added)
TTS_BACKEND = "openai"

# OpenAI TTS model: "tts-1" (fast) or "tts-1-hd" (higher quality)
OPENAI_TTS_MODEL = "tts-1"

# OpenAI voices: alloy, echo, fable, onyx, nova, shimmer
OPENAI_TTS_VOICE = "nova"
# =============================================================================

# Setup logging to stderr (not stdout - important for MCP stdio transport)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    stream=sys.stderr,
)
logger = logging.getLogger(__name__)

# Initialize the MCP server
mcp = FastMCP("caudio")


def text_to_slug(text: str, max_length: int = 40) -> str:
    """Convert text to a filename-safe slug."""
    slug = text.lower().strip()
    slug = re.sub(r"[\s_]+", "-", slug)
    slug = re.sub(r"[^a-z0-9-]", "", slug)
    slug = re.sub(r"-+", "-", slug)
    slug = slug.strip("-")
    if len(slug) > max_length:
        slug = slug[:max_length].rsplit("-", 1)[0]
    return slug or "audio"


def get_openai_client() -> OpenAI:
    """Get an OpenAI client, checking for API key."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY not set in environment")
    return OpenAI(api_key=api_key)


@mcp.tool()
def speak(
    text: str,
    output_dir: str,
    voice: str = "",
    model: str = "",
) -> str:
    """
    Convert text to speech and save as an audio file.

    Args:
        text: The text to convert to speech.
        output_dir: Directory to save the audio. Use "{cwd}/audio" where cwd is Claude's working directory.
        voice: Voice to use. OpenAI options: alloy, echo, fable, onyx, nova, shimmer.
               Leave empty to use default (nova).
        model: TTS model. OpenAI options: "tts-1" (fast) or "tts-1-hd" (quality).
               Leave empty to use default (tts-1).

    Returns:
        Path to the saved audio file.
    """
    voice = voice or OPENAI_TTS_VOICE
    model = model or OPENAI_TTS_MODEL

    logger.info(f"Generating speech with {TTS_BACKEND} ({model}, {voice}): {text[:50]}...")

    client = get_openai_client()

    response = client.audio.speech.create(
        model=model,
        voice=voice,
        input=text,
    )

    # Create output directory if needed
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Generate filename from text slug and timestamp
    slug = text_to_slug(text)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{slug}-{timestamp}.mp3"
    filepath = output_path / filename

    # Save audio
    response.stream_to_file(filepath)

    logger.info(f"Audio saved to: {filepath}")
    return f"Audio saved to: {filepath}"


@mcp.tool()
def transcribe(
    audio_file: str,
    language: str = "",
) -> str:
    """
    Transcribe an audio file to text using OpenAI Whisper.

    Args:
        audio_file: Path to the audio file to transcribe.
        language: Optional language code (e.g., "en", "es", "fr").
                  Leave empty for auto-detection.

    Returns:
        The transcribed text.
    """
    audio_path = Path(audio_file)
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_file}")

    logger.info(f"Transcribing: {audio_file}")

    client = get_openai_client()

    with open(audio_path, "rb") as f:
        kwargs = {"model": "whisper-1", "file": f}
        if language:
            kwargs["language"] = language

        response = client.audio.transcriptions.create(**kwargs)

    logger.info(f"Transcription complete: {len(response.text)} characters")
    return response.text


@mcp.tool()
def list_voices() -> str:
    """
    List available TTS voices for the current backend.

    Returns:
        A formatted list of available voices.
    """
    if TTS_BACKEND == "openai":
        voices = [
            ("alloy", "Neutral, balanced"),
            ("echo", "Warm, conversational"),
            ("fable", "Expressive, storytelling"),
            ("onyx", "Deep, authoritative"),
            ("nova", "Friendly, upbeat"),
            ("shimmer", "Clear, gentle"),
        ]
        result = "OpenAI TTS Voices:\n"
        for name, desc in voices:
            marker = " (default)" if name == OPENAI_TTS_VOICE else ""
            result += f"  - {name}: {desc}{marker}\n"
        return result
    else:
        return f"Unknown backend: {TTS_BACKEND}"


def main():
    """Run the MCP server."""
    logger.info(f"Starting caudio MCP server (tts={TTS_BACKEND})")
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
