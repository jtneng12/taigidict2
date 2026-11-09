#!/usr/bin/env python3
"""
Local lookup against Taiwan's Ministry of Education Taiwanese Hokkien
dictionary (教育部臺灣台語常用詞辭典), via g0v's open-data mirror of it
(moedict-data-twblg, https://github.com/g0v/moedict-data-twblg - government
open data, CC0-equivalent per iTaigi's description of the same source).

Used to ground claude_taigi.py/gemini_taigi.py's translation prompts with
real, authoritative hanji + Tai-lo romanization for concepts related to the
input text, so the AI prefers idiomatic native Taigi vocabulary over
inventing a Mandarin-derived character-for-character reading (e.g. 歡喜
huann-hi for "happy", not the literal 開心 khai-sim).

Deliberately a *retrieval* aid, not a replacement for the AI call: the
~14,500 entries are indexed by TAIGI headword with Mandarin definitions,
not the reverse, so a query like "開心" won't necessarily match its single
best synonym by keyword overlap - but related entries usually surface
(開心 itself appears in 心適's and 爽快's definitions, for instance), and
handing the AI a handful of real candidates to choose or compose from beats
letting it invent hanji unconstrained.
"""

import json
import re
from pathlib import Path

DICT_PATH = Path(__file__).parent / "moe_dict_twblg.json"

_entries: list[dict] | None = None

# Same CJK-ideograph range used elsewhere in this app to detect Mandarin text.
_CJK_RE = re.compile("[㐀-䶿一-鿿豈-﫿]")


def _load() -> list[dict]:
    global _entries
    if _entries is not None:
        return _entries

    try:
        raw = json.loads(DICT_PATH.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        raw = []

    entries = []
    for e in raw:
        title = e.get("title", "")
        for h in e.get("heteronyms", []) or []:
            trs = (h.get("trs") or "").strip()
            if not title or not trs:
                continue
            glosses = [d.get("def", "") for d in (h.get("definitions") or []) if d.get("def")]
            blob = " ".join([title, h.get("synonyms", ""), h.get("antonyms", ""), *glosses])
            entries.append({
                "hanji": title,
                "tailo": trs,
                "gloss": glosses[0] if glosses else "",
                "blob": blob,
            })
    _entries = entries
    return _entries


def _bigrams(s: str) -> set[str]:
    chars = [c for c in s if _CJK_RE.match(c)]
    if len(chars) < 2:
        return set(chars)
    return {"".join(chars[i:i + 2]) for i in range(len(chars) - 1)}


def search(query: str, limit: int = 6) -> list[dict]:
    """
    Rank dictionary entries by keyword overlap with `query` (a Mandarin
    word/phrase). Returns up to `limit` {"hanji", "tailo", "gloss"} dicts,
    best matches first, or [] if query isn't Mandarin or nothing scores.
    """
    query = query.strip()
    if not query or not _CJK_RE.search(query):
        return []

    q_bigrams = _bigrams(query)
    if not q_bigrams:
        return []

    scored = []
    for e in _load():
        score = 0
        if query == e["hanji"]:
            score += 1000
        elif query in e["blob"]:
            score += 50
        score += len(q_bigrams & _bigrams(e["blob"]))
        if score > 0:
            scored.append((score, e))

    scored.sort(key=lambda pair: -pair[0])

    seen_hanji: set[str] = set()
    results = []
    for _, e in scored:
        if e["hanji"] in seen_hanji:
            continue
        seen_hanji.add(e["hanji"])
        results.append({"hanji": e["hanji"], "tailo": e["tailo"], "gloss": e["gloss"]})
        if len(results) >= limit:
            break
    return results


def build_context(query: str) -> str:
    """Format search() results as a prompt-ready reference block, or "" if none."""
    hits = search(query)
    if not hits:
        return ""
    lines = "\n".join(f'- {h["hanji"]} ({h["tailo"]}) - {h["gloss"]}' for h in hits)
    return (
        "Reference - real entries from Taiwan's Ministry of Education "
        "Taiwanese Hokkien dictionary (教育部臺灣台語常用詞辭典) possibly "
        "relevant to the text below. Prefer their hanji/romanization over "
        "inventing your own when one genuinely fits the meaning - idiomatic "
        "Taigi is often NOT a literal Mandarin-derived reading:\n"
        f"{lines}\n\n"
    )


def main() -> None:
    import sys
    if len(sys.argv) < 2:
        print('Usage: python3 moe_dict.py "開心"')
        raise SystemExit(1)
    for hit in search(sys.argv[1]):
        print(f'{hit["hanji"]}  ({hit["tailo"]})  - {hit["gloss"]}')


if __name__ == "__main__":
    main()
