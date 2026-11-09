#!/usr/bin/env python3
"""
Helpers for the Mandarin | Taiwanese | English results layout:
- Mandarin pinyin (local, via pypinyin - no API call)
- Character-level diff between the Mandarin sentence and a Taigi hanji
  translation, so the UI can bold the words/characters that are specific
  to Taigi (i.e. not shared with the Mandarin source) and dim the rest.
"""

import difflib
import re
import unicodedata

from pypinyin import Style, pinyin


def mandarin_pinyin(text: str) -> str:
    """Mandarin text -> space-separated pinyin with tone marks."""
    if not text:
        return ""
    return " ".join(p[0] for p in pinyin(text, style=Style.TONE))


def _strip_tailo_tone_marks(tailo: str) -> str:
    """Drop Tai-lo tone-mark diacritics (acute/grave/circumflex/macron/vertical-line-above),
    keeping the base letters, by decomposing to NFD and dropping combining marks."""
    decomposed = unicodedata.normalize("NFD", tailo)
    return "".join(ch for ch in decomposed if unicodedata.category(ch) != "Mn")


def tailo_to_romanization(tailo: str) -> str:
    """Deterministically derive the casual, tone-mark-free romanization from a Tai-lo
    string - stripped diacritics, lowercased, hyphenated, sentence punctuation dropped -
    so it's always byte-for-byte consistent with the tailo it's shown next to instead of
    being a second value the translator could generate inconsistently."""
    if not tailo:
        return ""
    plain = _strip_tailo_tone_marks(tailo).lower()
    plain = re.sub(r"[?!.,，。！？]", "", plain)
    return re.sub(r"\s+", "-", plain.strip())


def tailo_skeleton(tailo: str) -> str:
    """Letters-only, tone-mark-free, lowercase skeleton of a Tai-lo string, for comparing
    whether two Tai-lo strings represent the same underlying words (ignoring tone marks,
    hyphenation, spacing, and punctuation)."""
    return re.sub(r"[^a-z]", "", _strip_tailo_tone_marks(tailo).lower())


def finalize_tailo_and_romanization(tailo: str, segments: list[dict]) -> tuple[str, str]:
    """
    The translator gives a whole-phrase "tailo" independently of the per-segment
    "tailo" readings that build up "hanji" - so nothing guarantees the two agree,
    and models occasionally drift (e.g. garbling one word's whole-phrase reading
    to something that doesn't match its own segment breakdown at all). Tai-lo
    tone sandhi only ever changes tone *marks*, never the underlying letters, so
    a correct whole-phrase tailo's letter skeleton should closely match the
    concatenated segment tailos' skeleton; a bigger divergence than that means
    the model substituted a different/garbled word, not legitimate sandhi.

    Returns (tailo, romanization): `tailo` is the given whole-phrase reading, or
    the segment-tailo fallback if it diverged too much; `romanization` is always
    deterministically derived from that final `tailo`, so it can never itself be
    a second, independently-drifting value.
    """
    tailo = (tailo or "").strip()
    segment_tailos = [s.get("tailo", "").strip() for s in segments if s.get("tailo", "").strip()]
    fallback = " ".join(segment_tailos)

    if not tailo:
        tailo = fallback
    elif fallback:
        skeleton = tailo_skeleton(tailo)
        fallback_skeleton = tailo_skeleton(fallback)
        if fallback_skeleton:
            ratio = difflib.SequenceMatcher(None, skeleton, fallback_skeleton).ratio()
            if ratio < 0.82:
                tailo = fallback

    return tailo, tailo_to_romanization(tailo)


def diff_segments(mandarin: str, taigi_hanji: str) -> list[dict]:
    """
    Character-level diff of `taigi_hanji` against `mandarin`. Returns a list of
    {"text": str, "specific": bool} segments covering taigi_hanji in order,
    where "specific" marks characters not shared with the Mandarin source
    (i.e. Taigi-specific words/characters, to be bolded in the UI).
    """
    if not taigi_hanji:
        return []
    if not mandarin:
        return [{"text": taigi_hanji, "specific": True}]

    matcher = difflib.SequenceMatcher(None, mandarin, taigi_hanji)
    segments = []
    for tag, _a1, _a2, b1, b2 in matcher.get_opcodes():
        if b1 == b2:
            continue
        segments.append({"text": taigi_hanji[b1:b2], "specific": tag != "equal"})
    return segments


def annotate_word_segments(mandarin: str, taigi_hanji: str, segments: list[dict]) -> list[dict]:
    """
    Given word/phrase-level `segments` from the translator (each
    {"mandarin", "hanji", "tailo", "english"} - see
    claude_taigi.py/gemini_taigi.py), add:
    - "specific": bool, True when its hanji differs from its mandarin
      (genuinely Taigi-specific wording, bolded in the UI), False when they
      match (shared with Mandarin, dimmed)
    - "pinyin": Mandarin pinyin for just this segment's "mandarin" text,
      computed locally (no LLM round-trip needed, unlike "tailo" which the
      translator already supplies per-segment)
    These segments carry everything the hover-to-translate tooltip needs:
    per-word Mandarin (+ pinyin), Taigi hanji (+ tailo), and English gloss.

    Falls back to a plain character-level diff (losing the per-word tailo
    and English gloss, but still usable for the bold/dim styling and hanji
    display) if the translator didn't return any segments.
    """
    if not segments:
        return [
            {
                "mandarin": seg["text"],
                "hanji": seg["text"],
                "tailo": "",
                "english": "",
                "pinyin": mandarin_pinyin(seg["text"]),
                "specific": seg["specific"],
            }
            for seg in diff_segments(mandarin, taigi_hanji)
        ]
    return [
        {**seg, "specific": seg["hanji"] != seg["mandarin"], "pinyin": mandarin_pinyin(seg["mandarin"])}
        for seg in segments
    ]
