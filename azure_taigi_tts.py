#!/usr/bin/env python3
"""
Taiwanese Hokkien (Minnan) speech synthesis via Microsoft Azure AI Speech
(Text-to-Speech), as an alternate voice to the free 意傳科技 pipeline
(ithuan_tts.py).

Azure's Mandarin "Xiaoxiao" voice has a multi-dialect variant,
zh-CN-XiaoxiaoDialectsNeural (female), that can speak Han-character text in
the Minnan (Southern Min / Hokkien) dialect by wrapping it in
<lang xml:lang="nan-CN"> inside the SSML <voice> element - Azure's standard
mechanism for selecting a secondary locale/dialect on a multi-dialect voice.

Note: this is Mainland China's Minnan dialect tag (nan-CN, roughly the
Xiamen/Amoy accent), not a dedicated Taiwan-specific "nan-TW" voice - Azure
does not appear to publish one. It's linguistically very close to Taiwanese
Hokkien but may differ in accent/vocabulary from 意傳科技's Taiwan-standard
台灣台語 output.

Requires an Azure AI Speech resource, configured via:
    AZURE_SPEECH_KEY     - resource subscription key
    AZURE_SPEECH_REGION  - resource region, e.g. "eastus"

Docs: https://learn.microsoft.com/azure/ai-services/speech-service/
"""

import hashlib
import os
from pathlib import Path
from xml.sax.saxutils import escape

import requests

VOICE_NAME = os.environ.get("AZURE_TAIGI_VOICE", "zh-CN-XiaoxiaoDialectsNeural")
DIALECT_LANG = os.environ.get("AZURE_TAIGI_DIALECT", "nan-CN")
UA = {"User-Agent": "taiwan-audio-dict/2.0"}

AUDIO_DIR = Path(__file__).parent / "static" / "audio" / "generated_azure"


def _endpoint(region: str) -> str:
    return f"https://{region}.tts.speech.microsoft.com/cognitiveservices/v1"


def _build_ssml(hanji: str) -> str:
    text = escape(hanji)
    return (
        '<speak version="1.0" xmlns="http://www.w3.org/2001/10/synthesis" '
        'xmlns:mstts="https://www.w3.org/2001/mstts" xml:lang="zh-CN">'
        f'<voice name="{VOICE_NAME}">'
        f'<lang xml:lang="{DIALECT_LANG}">{text}</lang>'
        "</voice></speak>"
    )


def synthesize_mp3(hanji: str) -> bytes:
    """Han-character text -> MP3 bytes, spoken in Minnan by Azure's Xiaoxiao dialect voice."""
    key = os.environ.get("AZURE_SPEECH_KEY")
    region = os.environ.get("AZURE_SPEECH_REGION")
    if not key or not region:
        raise RuntimeError(
            "AZURE_SPEECH_KEY and AZURE_SPEECH_REGION are not set. Create a Speech "
            "resource in the Azure portal and export both before running the server."
        )

    resp = requests.post(
        _endpoint(region),
        headers={
            **UA,
            "Ocp-Apim-Subscription-Key": key,
            "Content-Type": "application/ssml+xml",
            "X-Microsoft-OutputFormat": "audio-24khz-48kbitrate-mono-mp3",
        },
        data=_build_ssml(hanji).encode("utf-8"),
        timeout=30,
    )
    if not resp.ok:
        raise RuntimeError(f"Azure Speech API error {resp.status_code}: {resp.text[:300]}")
    audio = resp.content
    if not audio:
        raise RuntimeError("Empty audio response from Azure Speech API")
    return audio


def get_audio_url(hanji: str) -> str:
    """Synthesize (or reuse cached) Azure audio for Taigi hanji text and return its static URL."""
    AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    if not hanji:
        raise ValueError("No text to synthesize")

    digest = hashlib.sha1(hanji.encode("utf-8")).hexdigest()[:20]
    path = AUDIO_DIR / f"{digest}.mp3"
    if not path.exists():
        audio = synthesize_mp3(hanji)
        path.write_bytes(audio)

    return f"/static/audio/generated_azure/{digest}.mp3"


def main() -> None:
    import sys

    if len(sys.argv) < 2:
        print('Usage: python3 azure_taigi_tts.py "我叫做賴清德"')
        sys.exit(1)
    try:
        url = get_audio_url(sys.argv[1])
    except RuntimeError as e:
        raise SystemExit(str(e))
    print(f"wrote {(Path(__file__).parent / url.lstrip('/'))}")


if __name__ == "__main__":
    main()
