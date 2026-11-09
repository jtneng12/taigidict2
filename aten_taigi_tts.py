#!/usr/bin/env python3
"""
Taiwanese Hokkien (Taigi) speech synthesis via ATEN AI Voice (宏正優聲學),
as a third voice engine alongside the free 意傳科技 pipeline
(ithuan_tts.py) and Azure (azure_taigi_tts.py).

Why ATEN: in MediaTek Research's Breeze Taigi TTS benchmark
(arXiv:2603.19259) ATEN scored the highest Taiwanese pronunciation accuracy
of the systems tested (89.8%, MOS 3.8) - ahead of Taigi AI Labs (78.8%,
MOS 2.0) and BreezyVoice-Taigi (59.2%, which code-switched into Mandarin
roughly 40% of the time). It also takes 台羅 romanization natively, which
matters here: the rest of this app drives audio from the romanization shown
on screen so the two can never disagree (see ithuan_tts.get_audio_url).

Protocol per ATEN's "AIVoice API 使用說明" v1.2.111. Three steps:

    POST {base}/syntheses/api_token        {"ssml": ..., "audio_format": ...}
        -> {"synthesis_id", "synthesis_path", "srt_path"}
    GET  {base}/syntheses/{id}/api_token   -> {"status": "Success"|...}
    GET  {synthesis_path}                  -> audio bytes

    Auth: a BARE token in the Authorization header - `Authorization: <token>`,
    with no "Bearer " prefix. Rate limit 120/min.

Two traps worth recording, both of which cost real debugging time:
  - The endpoints take an "/api_token" suffix. The un-suffixed /syntheses is
    the web console's own session-authenticated route; calling it with an
    API key returns 422/42203 ("you do not have this permission"), which
    reads like a billing problem but is just the wrong endpoint.
  - ATEN's published SDK (github.com/ATEN-International/ai-voice-sdk-python
    v0.0.2) is stale on every detail: wrong base path (/api/v1.0/syn/*),
    wrong auth (Authorization: Bearer), wrong SSML version (1.0.demo), and a
    hardcoded six-voice Mandarin list that omits all 8 Taigi voices the live
    API serves. Don't trust it; trust the PDF.

lang_type="TB" selects Taiwanese and accepts EITHER Tai-lo romanization or
Taigi Han characters - verified by synthesizing both "Ke-āu" and "家後" and
getting distinct, valid audio from each. ATEN also documents
<phoneme alphabet="tailo"> for per-word pronunciation overrides.

Configured via .env:
    ATEN_API_KEY       - API token from the console's User Settings
    ATEN_SERVER_URL    - API base. Defaults to the /atzone endpoint; the
                         enterprise path is .../business/enterprise/api/v1,
                         and on-prem installs use http://<ip>/api/v1.
    ATEN_ORATOR_NAME   - voice id, default Raina_narrative (台語女聲-思羽).
                         Run `python3 aten_taigi_tts.py --voices` to list.
    ATEN_INPUT_MODE    - "tailo" (default) sends the romanization,
                         "hanji" sends the Han characters instead.
"""

import hashlib
import os
import time
from pathlib import Path
from xml.sax.saxutils import escape, quoteattr

import requests

UA = {"User-Agent": "taiwan-audio-dict/2.0"}
AUDIO_DIR = Path(__file__).parent / "static" / "audio" / "generated_aten"

DEFAULT_SERVER = "https://www.aivoice.com.tw/atzone/api/v1"
DEFAULT_ORATOR = "Raina_narrative"            # 台語女聲-思羽 (Siyu)
AUDIO_FORMAT = "Audio24Khz96KBitRateMonoMp3"  # MP3, matching the other engines
TAIGI_LANG_TYPE = "TB"

TEXT_LIMIT = 2000
POLL_INTERVAL = 0.8
POLL_TIMEOUT = 90.0


def _config() -> tuple[str, str, str]:
    token = os.environ.get("ATEN_API_KEY", "").strip()
    # An empty value in .env must fall back too - os.environ.get()'s default
    # only applies when the key is absent, not when it is present but blank.
    server = (os.environ.get("ATEN_SERVER_URL") or DEFAULT_SERVER).strip().rstrip("/") or DEFAULT_SERVER
    orator = (os.environ.get("ATEN_ORATOR_NAME") or DEFAULT_ORATOR).strip() or DEFAULT_ORATOR

    if not token:
        raise RuntimeError(
            "ATEN_API_KEY is not set. Get the token from the ATEN console under "
            "User Settings, then add it to .env."
        )
    if not server.startswith("http"):
        raise RuntimeError(f"ATEN_SERVER_URL must start with http(s)://, got {server!r}")
    return token, server, orator


def _headers(token: str) -> dict:
    # Bare token, NOT "Bearer <token>" - see module docstring.
    return {"Authorization": token, "Content-Type": "application/json", **UA}


def _diagnose(resp: requests.Response) -> str:
    body = resp.text[:200]
    if resp.status_code in (401, 403) or "4401" in body:
        return (
            "ATEN rejected the token. Check ATEN_API_KEY matches the token in the "
            f"console's User Settings. Server said: {body}"
        )
    if "42203" in body or "4203" in body:
        return (
            "ATEN returned a permission error. This usually means the request went to "
            "the console's session route instead of the API one - every endpoint needs "
            f"the '/api_token' suffix. Server said: {body}"
        )
    if "4207" in body:
        return "Text exceeds ATEN's per-synthesis character limit."
    if "4208" in body:
        return "ATEN monthly character quota exhausted."
    if "4210" in body:
        return "Too many ATEN syntheses queued at once; retry shortly."
    if "4212" in body:
        return f"ATEN rejected the SSML as malformed: {body}"
    return f"ATEN API error ({resp.status_code}): {body}"


def _build_ssml(orator: str, text: str) -> str:
    return (
        '<speak xmlns="http://www.w3.org/2001/10/synthesis" version="1.5" '
        'xml:lang="zh-TW">'
        f"<voice name={quoteattr(orator)}>"
        f'<lang lang_type="{TAIGI_LANG_TYPE}">{escape(text)}</lang>'
        "</voice></speak>"
    )


def list_voices(taigi_only: bool = True) -> list[dict]:
    """Enumerate the voices this account can use (live, not the SDK's stale list)."""
    token, server, _ = _config()
    resp = requests.get(f"{server}/models/api_token", headers=_headers(token), timeout=30)
    if not resp.ok:
        raise RuntimeError(_diagnose(resp))
    models = resp.json()
    if isinstance(models, dict):
        models = models.get("models", [])
    if taigi_only:
        models = [m for m in models if "台" in "".join(m.get("languages") or [])]
    return [{"model_id": m.get("model_id"), "name": m.get("name")} for m in models]


def synthesize(text: str) -> bytes:
    """Text (Tai-lo romanization or Taigi Han characters) -> MP3 bytes."""
    if not text.strip():
        raise ValueError("No text to synthesize")
    if len(text) > TEXT_LIMIT:
        raise ValueError(f"Text exceeds ATEN's {TEXT_LIMIT}-character limit ({len(text)}).")

    token, server, orator = _config()
    headers = _headers(token)

    resp = requests.post(
        f"{server}/syntheses/api_token",
        json={"ssml": _build_ssml(orator, text), "audio_format": AUDIO_FORMAT},
        headers=headers,
        timeout=60,
    )
    if not resp.ok:
        raise RuntimeError(_diagnose(resp))
    job = resp.json()
    synthesis_id = job.get("synthesis_id")
    synthesis_path = job.get("synthesis_path")
    if not synthesis_id or not synthesis_path:
        raise RuntimeError(f"ATEN returned no synthesis job: {str(job)[:200]}")

    deadline = time.time() + POLL_TIMEOUT
    status = None
    while time.time() < deadline:
        poll = requests.get(
            f"{server}/syntheses/{synthesis_id}/api_token", headers=headers, timeout=30
        )
        if not poll.ok:
            raise RuntimeError(_diagnose(poll))
        status = poll.json().get("status")
        if status == "Success":
            break
        if status in ("Fail", "Error", "Cancel"):
            raise RuntimeError(f"ATEN synthesis ended with status {status!r}.")
        time.sleep(POLL_INTERVAL)
    else:
        raise RuntimeError(
            f"ATEN synthesis {synthesis_id} still {status!r} after {POLL_TIMEOUT:.0f}s."
        )

    audio = requests.get(synthesis_path, headers=headers, timeout=60)
    if not audio.ok:
        raise RuntimeError(_diagnose(audio))
    if "audio" not in audio.headers.get("Content-Type", ""):
        raise RuntimeError(f"ATEN returned no audio for {synthesis_id}: {audio.text[:200]}")
    if not audio.content:
        raise RuntimeError(f"ATEN returned an empty audio file for {synthesis_id}.")
    return audio.content


def get_audio_url(hanji: str, tailo: str = "") -> str:
    """
    Synthesize (or reuse cached) audio and return its static URL path.

    Mirrors ithuan_tts.get_audio_url: the romanization drives the audio when
    there is one, so what the page displays is what the voice says. Set
    ATEN_INPUT_MODE=hanji to send Han characters instead - ATEN accepts both.
    """
    AUDIO_DIR.mkdir(parents=True, exist_ok=True)

    prefer_hanji = os.environ.get("ATEN_INPUT_MODE", "tailo").strip().lower() == "hanji"
    text = (hanji or tailo) if prefer_hanji else (tailo or hanji)
    if not text:
        raise ValueError("No text to synthesize")

    orator = (os.environ.get("ATEN_ORATOR_NAME") or DEFAULT_ORATOR).strip() or DEFAULT_ORATOR
    digest = hashlib.sha1(f"{orator}|{text}".encode("utf-8")).hexdigest()[:20]
    path = AUDIO_DIR / f"{digest}.mp3"
    if not path.exists():
        path.write_bytes(synthesize(text))

    return f"/static/audio/generated_aten/{digest}.mp3"


def main() -> None:
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "--voices":
        for v in list_voices():
            print(f"  {v['model_id']:<24} {v['name']}")
        return
    if len(sys.argv) < 2:
        print('Usage: python3 aten_taigi_tts.py "Guá ê miâ sī"   |   --voices')
        raise SystemExit(1)
    try:
        url = get_audio_url("", sys.argv[1])
    except (RuntimeError, ValueError) as e:
        raise SystemExit(f"ERROR: {e}")
    print(f"wrote {Path(__file__).parent / url.lstrip('/')}")


if __name__ == "__main__":
    main()
