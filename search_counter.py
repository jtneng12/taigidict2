#!/usr/bin/env python3
"""
Shared search counter across every app in this project (ports 5001-5004).

Each app runs as its own Flask process, so the count can't just live in a
Python global - it's persisted to a single JSON file at the project root
that all of them read/write. fcntl.flock guards each read-modify-write so
concurrent increments from different processes/threads don't clobber each
other and lose counts.
"""

import fcntl
import json
from pathlib import Path

COUNTER_PATH = Path(__file__).parent / "search_count.json"


def increment_and_get() -> int:
    """Atomically increment the shared counter and return the new total."""
    with open(COUNTER_PATH, "a+") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        try:
            f.seek(0)
            try:
                count = json.loads(f.read() or "{}").get("count", 0)
            except json.JSONDecodeError:
                count = 0
            count += 1
            f.seek(0)
            f.truncate()
            json.dump({"count": count}, f)
            return count
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)


def get_count() -> int:
    """Read the shared counter without incrementing it."""
    try:
        return json.loads(COUNTER_PATH.read_text()).get("count", 0)
    except (FileNotFoundError, json.JSONDecodeError):
        return 0
