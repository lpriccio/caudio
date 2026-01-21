"""MCP server for audio generation (TTS) and transcription (STT)."""

import logging
import os
import re
import sys
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
from elevenlabs import ElevenLabs
from mcp.server.fastmcp import FastMCP
from openai import OpenAI

# Load .env file from the project directory
load_dotenv(Path(__file__).parent.parent.parent / ".env")

# =============================================================================
# CONFIGURATION - Change TTS_BACKEND to switch between providers
# =============================================================================
# TTS backend: "openai" or "elevenlabs"
TTS_BACKEND = "elevenlabs"

# OpenAI config
OPENAI_TTS_MODEL = "tts-1"      # "tts-1" (fast) or "tts-1-hd" (quality)
OPENAI_TTS_VOICE = "nova"       # alloy, echo, fable, onyx, nova, shimmer

# ElevenLabs config
ELEVENLABS_MODEL = "eleven_multilingual_v2"  # or "eleven_turbo_v2_5" (faster)
ELEVENLABS_VOICE = "Rachel"     # See list_voices() for options
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

# ElevenLabs voice name -> ID mapping (popular voices)
ELEVENLABS_VOICES = {
    "Rachel": "21m00Tcm4TlvDq8ikWAM",      # Warm, clear, American female
    "Domi": "AZnzlk1XvdvUeBnXmlld",        # Strong, confident female
    "Bella": "EXAVITQu4vr4xnSDxMaL",       # Soft, gentle female
    "Antoni": "ErXwobaYiN019PkySvjV",      # Well-rounded male
    "Elli": "MF3mGyEYCl7XYWbV9V6O",        # Young, friendly female
    "Josh": "TxGEqnHWrfWFTfGW9XjX",        # Deep, narrative male
    "Arnold": "VR6AewLTigWG4xSOukaG",      # Crisp, announcer male
    "Adam": "pNInz6obpgDQGcFmaJgB",        # Deep, narrative male
    "Sam": "yoZ06aMxZJJ28mfd3POQ",         # Raspy, dynamic male
}


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


def get_elevenlabs_client() -> ElevenLabs:
    """Get an ElevenLabs client, checking for API key."""
    api_key = os.getenv("ELEVENLABS_API_KEY")
    if not api_key:
        raise ValueError("ELEVENLABS_API_KEY not set in environment")
    return ElevenLabs(api_key=api_key)


def speak_openai(text: str, voice: str, model: str) -> bytes:
    """Generate speech using OpenAI TTS."""
    client = get_openai_client()
    response = client.audio.speech.create(
        model=model,
        voice=voice,
        input=text,
    )
    return response.content


def speak_elevenlabs(text: str, voice: str, model: str) -> bytes:
    """Generate speech using ElevenLabs."""
    client = get_elevenlabs_client()

    # Resolve voice name to ID
    voice_id = ELEVENLABS_VOICES.get(voice, voice)

    response = client.text_to_speech.convert(
        voice_id=voice_id,
        text=text,
        model_id=model,
    )

    # Response is a generator, collect all chunks
    audio_bytes = b"".join(response)
    return audio_bytes


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
        voice: Voice to use. Leave empty for default.
               OpenAI: alloy, echo, fable, onyx, nova, shimmer
               ElevenLabs: Rachel, Domi, Bella, Antoni, Elli, Josh, Arnold, Adam, Sam
        model: TTS model. Leave empty for default.
               OpenAI: "tts-1" or "tts-1-hd"
               ElevenLabs: "eleven_multilingual_v2" or "eleven_turbo_v2_5"

    Returns:
        Path to the saved audio file.
    """
    # Set defaults based on backend
    if TTS_BACKEND == "elevenlabs":
        voice = voice or ELEVENLABS_VOICE
        model = model or ELEVENLABS_MODEL
        audio_bytes = speak_elevenlabs(text, voice, model)
    else:
        voice = voice or OPENAI_TTS_VOICE
        model = model or OPENAI_TTS_MODEL
        audio_bytes = speak_openai(text, voice, model)

    logger.info(f"Generated speech with {TTS_BACKEND} ({model}, {voice}): {text[:50]}...")

    # Create output directory if needed
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Generate filename from text slug and timestamp
    slug = text_to_slug(text)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{slug}-{timestamp}.mp3"
    filepath = output_path / filename

    # Save audio
    with open(filepath, "wb") as f:
        f.write(audio_bytes)

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

    elif TTS_BACKEND == "elevenlabs":
        voices = [
            ("Rachel", "Warm, clear, American female"),
            ("Domi", "Strong, confident female"),
            ("Bella", "Soft, gentle female"),
            ("Antoni", "Well-rounded male"),
            ("Elli", "Young, friendly female"),
            ("Josh", "Deep, narrative male"),
            ("Arnold", "Crisp, announcer male"),
            ("Adam", "Deep, narrative male"),
            ("Sam", "Raspy, dynamic male"),
        ]
        result = "ElevenLabs Voices:\n"
        for name, desc in voices:
            marker = " (default)" if name == ELEVENLABS_VOICE else ""
            result += f"  - {name}: {desc}{marker}\n"
        result += "\nModels:\n"
        result += "  - eleven_multilingual_v2: Best quality, 29 languages\n"
        result += "  - eleven_turbo_v2_5: Faster, English-optimized\n"
        return result

    else:
        return f"Unknown backend: {TTS_BACKEND}"


def main():
    """Run the MCP server."""
    logger.info(f"Starting caudio MCP server (tts={TTS_BACKEND})")
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
