#!/usr/bin/env python3
"""
New Taiwan Audio Dictionary - Gemini + 意傳科技 / Azure edition.

Same app as claude_app.py, same visual interface, but translation goes
through Google's Gemini API (gemini_taigi.py) instead of Claude. Gemini has
a real free tier (no credit card needed), unlike Anthropic's or OpenAI's
pay-from-the-first-token APIs - see gemini_taigi.py for details and setup.

Runs on port 5003 so it can sit side by side with claude_app.py (port 5002)
for comparing translation quality between the two.
"""

import os
from pathlib import Path

from flask import Flask, render_template, request, jsonify


def _load_dotenv() -> None:
    """Load KEY=VALUE lines from .env in this directory into os.environ, for
    vars not already set - see claude_app.py for why this exists."""
    env_path = Path(__file__).parent / ".env"
    if not env_path.exists():
        return
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)


_load_dotenv()

import aten_taigi_tts
import azure_taigi_tts
import ithuan_tts
import search_counter
from gemini_taigi import translate_to_taigi
from text_compare import annotate_word_segments, mandarin_pinyin

app = Flask(__name__)

VOICE_ENGINES = {"ithuan", "azure", "aten"}


@app.route("/")
def index():
    return render_template("gemini_index.html")


@app.route("/api/search", methods=["POST"])
def search():
    data = request.get_json() or {}
    sentence = data.get("sentence", "").strip()

    if not sentence:
        return jsonify({"error": "Please enter a sentence"}), 400

    try:
        translation = translate_to_taigi(sentence)
    except RuntimeError as e:
        return jsonify({"error": str(e)}), 500
    except Exception as e:
        return jsonify({"error": f"Translation failed: {e}"}), 502

    variations = translation["variations"]
    if not variations:
        return jsonify({"error": "No Taigi translation found for that sentence"}), 404

    # "mandarin"/"english" are always in their own language regardless of
    # which one the visitor actually typed - see gemini_taigi.py.
    mandarin_text = translation["mandarin"]
    english_text = translation["english"]

    results = []
    for v in variations:
        segments = annotate_word_segments(mandarin_text, v["hanji"], v["segments"])
        variation_mandarin = "".join(s["mandarin"] for s in segments)
        results.append({
            "hanji": v["hanji"],
            "tailo": v["tailo"],
            "romanization": v["romanization"],
            "segments": segments,
            "pinyin": mandarin_pinyin(variation_mandarin),
        })

    return jsonify({
        "sentence": mandarin_text,
        "pinyin": mandarin_pinyin(mandarin_text),
        "english": english_text,
        "variations": results,
        "search_count": search_counter.increment_and_get(),
    })


@app.route("/api/search-count")
def search_count():
    return jsonify({"count": search_counter.get_count()})


@app.route("/api/audio", methods=["POST"])
def audio():
    data = request.get_json() or {}
    hanji = (data.get("hanji") or "").strip()
    tailo = (data.get("tailo") or "").strip()
    engine = data.get("engine") or "ithuan"

    if engine not in VOICE_ENGINES:
        return jsonify({"error": f"Unknown voice engine: {engine}"}), 400
    if not hanji:
        return jsonify({"error": "No text to synthesize"}), 400

    try:
        if engine == "azure":
            url = azure_taigi_tts.get_audio_url(hanji)
        elif engine == "aten":
            url = aten_taigi_tts.get_audio_url(hanji, tailo)
        else:
            url = ithuan_tts.get_audio_url(hanji, tailo)
    except Exception as e:
        return jsonify({"error": str(e)}), 502

    return jsonify({"audio_url": url})


if __name__ == "__main__":
    app.run(debug=True, port=5003, threaded=True)
