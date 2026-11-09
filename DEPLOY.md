# Deploying the Gemini Taigi dictionary to PythonAnywhere + taigidict.com

This is `gemini_app.py` (port 5003 locally) and exactly what it imports,
trimmed and packaged to run as a standalone PythonAnywhere web app. No code
was changed — see "Why no code changes" below for why that's safe, not an
oversight.

## 0. Two things to check before anything else

**Outbound internet access.** PythonAnywhere's free tier restricts outbound
HTTP(S) requests to a domain allowlist, to prevent abuse. This app calls
`generativelanguage.googleapis.com` (Gemini), `hokbu.ithuan.tw` and
`hapsing.ithuan.tw` (意傳科技 TTS), and optionally Azure/ATEN endpoints — none
of those are likely to be on a general-purpose free-tier allowlist. **A paid
plan is required for this app to make any of its real network calls**, not
just for the custom domain. Confirm this on PythonAnywhere's own pricing page
before paying for anything, since I can't verify current plan details or
pricing myself.

**Custom domain + HTTPS** also requires a paid plan — the free tier only gives
you `<username>.pythonanywhere.com`.

So budget for a paid plan either way; check their current pricing page for
the exact tier and cost.

## 1. Create the account and a Flask web app

1. Sign up at pythonanywhere.com, pick a plan that includes custom domains and
   unrestricted internet access (see above).
2. **Web** tab → **Add a new web app** → choose **Flask**, the Python version
   matching what you run locally, and accept the default project path (or
   note whatever path it gives you — you'll need it in step 3).
3. This step pre-creates a WSGI config file and a starter Flask project.
   You're about to replace the starter project's contents with this bundle,
   not the WSGI file itself.

## 2. Get this bundle onto the server

This now lives at **github.com/jtneng12/taigidict2** (public), so a plain
clone works — PythonAnywhere's Bash console doesn't have your local SSH key,
so use the HTTPS URL, not `git@github.com:...`:

```bash
git clone https://github.com/jtneng12/taigidict2.git taigidict
```

That creates `~/taigidict` (or wherever your home directory lands you) with
everything from the repo. Future updates are a `git pull` in that same
directory, followed by the **Web** tab's **Reload** button — no more
re-zipping and re-uploading.

(If you ever need the original zip-upload path instead — no git required —
it's: **Files** tab → upload `taigidict-pythonanywhere.zip` → Bash console →
`unzip -o taigidict-pythonanywhere.zip`.)

## 3. Virtualenv and dependencies

In the same Bash console:
```bash
mkvirtualenv taigidict-venv --python=python3.11   # match your installed version
pip install -r requirements.txt
```
(`mkvirtualenv` is PythonAnywhere's own helper; if it's not available in your
plan, use `python3.11 -m venv ~/.virtualenvs/taigidict-venv` and
`source ~/.virtualenvs/taigidict-venv/bin/activate` first.)

Then on the **Web** tab, under **Virtualenv**, point it at
`/home/<username>/.virtualenvs/taigidict-venv`.

## 4. Secrets

Create a real `.env` file (copy `.env.example`'s names, fill in real values)
in the same directory as `gemini_app.py` on PythonAnywhere — either through
the Files tab's editor, or `nano .env` in the Bash console. This is the exact
same mechanism the app already uses locally (`gemini_app.py`'s own
`_load_dotenv()`), so nothing PythonAnywhere-specific is needed here.

At minimum: `GEMINI_API_KEY`.

## 5. Wire the WSGI file

**Web** tab → click the WSGI configuration file link (something like
`/var/www/<username>_pythonanywhere_com_wsgi.py`) → edit it, using
`wsgi_snippet.py` in this bundle as the template: set `project_home` to the
real path from step 2, and make sure the file ends by importing
`app` from `gemini_app` as `application`.

Click the big green **Reload** button on the Web tab.

## 6. Verify before touching DNS

Your app is now live at `https://<username>.pythonanywhere.com`. Test it
there first — don't wire the domain until this works:

```bash
curl -s -X POST https://<username>.pythonanywhere.com/api/search \
  -H 'Content-Type: application/json' -d '{"sentence":"開心"}'
```

Should return a translation with `"tailo"` containing `huann-hí`, not
`hua-hí` — that's the MOE-grounding check. Also open the page in a browser and
try the search box and the listen button end to end.

Check the **Web** tab's error log if anything 500s — the most common first
failures here are the virtualenv not being set, `.env` missing or
misplaced, or `project_home` in the WSGI file not matching where the files
actually live.

## 7. Point taigidict.com at it

1. Register `taigidict.com` if you haven't (it isn't registered yet, confirmed
   by DNS lookup — nobody owns it). Cloudflare Registrar sells at cost.
2. On PythonAnywhere's **Web** tab, add `taigidict.com` as a custom domain for
   this web app (paid-plan feature). PythonAnywhere will show you the exact
   DNS record(s) to create — copy those values exactly; I'm deliberately not
   guessing a specific IP or CNAME target here since I can't see their current
   dashboard and getting that wrong would just waste your time.
3. In Cloudflare's **DNS** tab for `taigidict.com`, add exactly the record(s)
   PythonAnywhere showed you.
4. Set that record's proxy status to **grey cloud ("DNS only")** at first —
   PythonAnywhere's own domain verification and certificate issuance needs to
   reach the server directly. You can switch to orange (proxied) afterward if
   you want Cloudflare's CDN/caching layer, but then also set Cloudflare's
   SSL/TLS mode to **Full (strict)** or visitors will get a certificate error.
5. Wait for DNS to propagate (minutes to a few hours), then PythonAnywhere's
   dashboard should show the domain verified with HTTPS issued.
6. Re-run the step 6 check against `https://taigidict.com` directly.

## Why no code changes were needed

`gemini_app.py`'s `if __name__ == "__main__": app.run(debug=True, port=5003,
threaded=True)` line only executes when the file is run directly
(`python gemini_app.py`). PythonAnywhere's WSGI server imports the module and
calls the `app` object directly through `wsgi_snippet.py` — that `__main__`
block never runs, so neither the hardcoded port nor the debug server are ever
reached in this deployment. Every module this app imports
(`aten_taigi_tts`, `azure_taigi_tts`, `claude_taigi`) reads its own
configuration lazily (inside functions, not at import time), so missing
optional secrets degrade to a clear per-request error instead of crashing the
whole app at startup.

## What was deliberately left out of this bundle

- `openpyxl`, `wordfreq`, `jieba`, `zhconv`, `claude-agent-sdk` from
  `requirements.txt` — verified by grep that nothing in this app's actual
  import chain uses them; they belong to other scripts in the source project.
- The existing `static/audio/generated*` caches (local project has ~400
  cached MP3s, ~3 MB) — not shipped. They'll regenerate on first request per
  phrase/voice, same as a fresh local checkout. If you'd rather not have
  first visitors wait on a cold cache, say so and I'll pre-warm a set before
  the next bundle.
- The legacy `static/audio` directory with ~21,000 numbered files from the
  older `app.py`/`mkdict_audio.py` pipeline — unrelated to this app, not
  shipped.
- `static/img/backgrounds/` (the 4 Taiwan-scene photos) and
  `static/img/taigi-char.png` / `taigi-logo.png` — not referenced by
  `gemini_index.html` (checked directly), so not shipped. That rotating-photo
  treatment belongs to the Replit/desk version, not this one.

## Known limitations

- **No fluent-speaker review of output** — same caveat as the Replit handoff;
  this ships the same translation logic, not newly verified.
- **Single server, no autoscaling** — fine for a side project's traffic, but
  means ithuan.tw's per-IP rate limit (~3 sentences/min) applies to this one
  PythonAnywhere egress IP shared across however many visitors are using the
  site at once, not per-visitor like it was when everyone ran it locally.
- **Cross-model verification (Claude) is inert** without `ANTHROPIC_API_KEY` —
  same as the Replit deployment. Translations are cached unverified.
