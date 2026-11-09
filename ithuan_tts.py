#!/usr/bin/env python3
"""
Taiwanese Hokkien speech synthesis via 意傳科技 (ithuan.tw) - 媠聲/鬥拍字 API.

Two hops:
  1. POST https://hokbu.ithuan.tw/tau        Han characters -> KIP (Tai-lo), tone sandhi applied
  2. GET  https://hapsing.ithuan.tw/bangtsam  KIP text -> MP3

Callers (claude_taigi.py, gemini_taigi.py) already produce a Tai-lo/KIP
romanization alongside the Han characters, so get_audio_url() synthesizes
directly from that romanization (synthesize_tailo()) whenever it's given -
the audio should say what the displayed romanization says, not a second,
independently re-derived reading. Step 1 (hanji -> KIP via /tau) is only
used as a fallback, for hanji-only callers with no romanization on hand.

Notes (found by probing the live service; the deployed API had drifted
from what the 鬥拍字 web app's own source implies):
  - /tau requires POST with a form-encoded body ("taibun" field) - a GET,
    or a POST with the same field as a query string, returns 500.
  - /tau's JSON response has a top-level "KIP" key (plus a "kiatko" list
    of per-clause breakdowns) - not "綜合標音".
  - /bangtsam returns an MP3 (audio/octet-stream), not a WAV.
  - For multi-clause input, /tau's aggregate "KIP" field glues sentence
    punctuation directly onto the next word with no space (e.g.
    "kuè.ū kuá-jîn", "bat;ū kuá-jîn"). Sending that straight to /bangtsam
    mangles pronunciation at every clause boundary - the engine expects
    one clause at a time (matching the 鬥拍字 UI's own per-sentence design
    and rate limit). So multi-clause text is split on sentence-ending
    punctuation, synthesized clause by clause, and the resulting clips are
    concatenated.

Free service, rate-limited (~3 sentences/minute/IP per the 鬥拍字 UI), so
callers doing several lookups in a row should space them out.
"""

import hashlib
import time
from pathlib import Path

import requests

TAU_URL = "https://hokbu.ithuan.tw/tau"
BANGTSAM_URL = "https://hapsing.ithuan.tw/bangtsam"
UA = {"User-Agent": "taiwan-audio-dict/2.0"}

AUDIO_DIR = Path(__file__).parent / "static" / "audio" / "generated"

_SENTENCE_END_CHARS = "。！？；!?;"


def _split_clauses(hanji: str) -> list[str]:
    """
    Split into clauses at sentence-ending punctuation (Chinese and Latin
    forms), keeping the punctuation attached to the preceding clause. A
    "." only counts as a boundary when it isn't a decimal point (i.e. not
    flanked by digits on both sides), so numbers like "0.01" stay intact.
    """
    clauses = []
    start = 0
    for i, ch in enumerate(hanji):
        boundary = ch in _SENTENCE_END_CHARS
        if ch == ".":
            prev_digit = i > 0 and hanji[i - 1].isdigit()
            next_digit = i + 1 < len(hanji) and hanji[i + 1].isdigit()
            boundary = not (prev_digit and next_digit)
        if boundary:
            clauses.append(hanji[start:i + 1])
            start = i + 1
    if start < len(hanji):
        clauses.append(hanji[start:])
    return [c.strip() for c in clauses if c.strip()]


def _strip_id3(mp3_bytes: bytes) -> bytes:
    """Drop a leading ID3v2 tag, if present, so clips can be concatenated cleanly."""
    if mp3_bytes[:3] != b"ID3":
        return mp3_bytes
    size = (
        (mp3_bytes[6] & 0x7F) << 21
        | (mp3_bytes[7] & 0x7F) << 14
        | (mp3_bytes[8] & 0x7F) << 7
        | (mp3_bytes[9] & 0x7F)
    )
    return mp3_bytes[10 + size:]


def hanji_to_kip(hanji: str) -> str | None:
    """Han characters (single clause) -> canonical KIP romanization, or None on failure."""
    if not hanji:
        return None
    try:
        resp = requests.post(TAU_URL, headers=UA, data={"taibun": hanji}, timeout=15)
        resp.raise_for_status()
        data = resp.json()
    except (requests.RequestException, ValueError):
        return None
    kip = (data.get("KIP") or "").strip()
    return kip or None


def synthesize_mp3(taibun: str) -> bytes:
    """KIP romanization (or Han characters) -> MP3 bytes."""
    resp = requests.get(BANGTSAM_URL, headers=UA, params={"taibun": taibun}, timeout=30)
    resp.raise_for_status()
    audio = resp.content
    if not audio:
        raise RuntimeError("Empty audio response from hapsing.ithuan.tw")
    return audio


def synthesize_hanji(hanji: str, delay: float = 0.4) -> bytes:
    """
    Han-character text -> MP3 bytes. Splits multi-clause text into individual
    sentences, synthesizes each separately (with its own /tau-derived KIP),
    and concatenates the clips - see module docstring for why.

    This re-derives romanization from hanji via /tau, independently of
    whatever romanization the caller may already have (e.g. from Claude/
    Gemini's own translation). Only used when no such romanization is
    available - see get_audio_url(), which prefers synthesize_tailo() when
    it has one, since two independently-computed romanizations can disagree
    and the displayed one should be what's actually heard.
    """
    clauses = _split_clauses(hanji)
    if not clauses:
        raise ValueError("No text to synthesize")

    chunks = []
    for i, clause in enumerate(clauses):
        kip = hanji_to_kip(clause) or clause
        audio = synthesize_mp3(kip)
        chunks.append(audio if i == 0 else _strip_id3(audio))
        if i + 1 < len(clauses):
            time.sleep(delay)
    return b"".join(chunks)


def synthesize_tailo(tailo: str, delay: float = 0.4) -> bytes:
    """
    Tai-lo/KIP romanization text -> MP3 bytes. Splits multi-clause text into
    individual clauses (same rule as synthesize_hanji, since /bangtsam wants
    one clause at a time) and synthesizes each directly - no /tau round trip,
    so the audio matches this exact romanization rather than one re-derived
    from hanji.
    """
    clauses = _split_clauses(tailo)
    if not clauses:
        raise ValueError("No text to synthesize")

    chunks = []
    for i, clause in enumerate(clauses):
        audio = synthesize_mp3(clause)
        chunks.append(audio if i == 0 else _strip_id3(audio))
        if i + 1 < len(clauses):
            time.sleep(delay)
    return b"".join(chunks)


def get_audio_url(hanji: str, tailo: str) -> str:
    """
    Synthesize (or reuse cached) audio for a Taigi phrase and return its
    static URL path.

    When a romanization (`tailo`) is available - e.g. from Claude/Gemini's
    translation - it drives the audio directly via synthesize_tailo(),
    rather than re-deriving a (possibly different) romanization from `hanji`
    via ithuan.tw's own /tau. The displayed romanization and the spoken
    audio should always be the same reading; falling back to /tau only when
    there's no romanization to work with (hanji-only callers) keeps that
    true instead of letting two independent conversions drift apart.
    """
    AUDIO_DIR.mkdir(parents=True, exist_ok=True)

    if not hanji and not tailo:
        raise ValueError("No text to synthesize")

    digest = hashlib.sha1((tailo or hanji).encode("utf-8")).hexdigest()[:20]
    path = AUDIO_DIR / f"{digest}.mp3"
    if not path.exists():
        audio = synthesize_tailo(tailo) if tailo else synthesize_hanji(hanji)
        path.write_bytes(audio)

    return f"/static/audio/generated/{digest}.mp3"


def main() -> None:
    import sys

    if len(sys.argv) < 2:
        print('Usage: python3 ithuan_tts.py "我叫做賴清德" [--kip "Guá kiò-tsò Luā Tshing-tik"]')
        sys.exit(1)
    hanji = sys.argv[1]
    tailo = sys.argv[3] if len(sys.argv) > 3 and sys.argv[2] == "--kip" else ""
    url = get_audio_url(hanji, tailo)
    print(f"wrote {(Path(__file__).parent / url.lstrip('/'))}")


if __name__ == "__main__":
    main()
