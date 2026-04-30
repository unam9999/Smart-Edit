"""
VisionSort — In-Memory Session Store
======================================
Holds raw image bytes + classification results between
the classify step and the apply-presets step.

Sessions auto-expire after SESSION_TTL_SECONDS (default 10 min).
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Optional

SESSION_TTL_SECONDS = 600  # 10 minutes


@dataclass
class ImageEntry:
    filename: str
    raw_bytes: bytes
    category: str


@dataclass
class Session:
    session_id: str
    images: list[ImageEntry] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)

    def is_expired(self) -> bool:
        return (time.time() - self.created_at) > SESSION_TTL_SECONDS


# ── Global store ──────────────────────────────────────────────
_sessions: dict[str, Session] = {}


def create_session() -> str:
    """Create a new session and return its ID."""
    _purge_expired()
    sid = str(uuid.uuid4())
    _sessions[sid] = Session(session_id=sid)
    return sid


def add_image(session_id: str, filename: str, raw_bytes: bytes, category: str) -> None:
    """Add a classified image to an existing session."""
    session = _sessions.get(session_id)
    if session is None:
        raise KeyError(f"Session {session_id!r} not found")
    session.images.append(ImageEntry(filename=filename, raw_bytes=raw_bytes, category=category))


def get_session(session_id: str) -> Optional[Session]:
    """Retrieve a session by ID. Returns None if not found or expired."""
    session = _sessions.get(session_id)
    if session is None:
        return None
    if session.is_expired():
        del _sessions[session_id]
        return None
    return session


def delete_session(session_id: str) -> None:
    """Delete a session after it has been used."""
    _sessions.pop(session_id, None)


def _purge_expired() -> None:
    """Remove all expired sessions to prevent unbounded memory growth."""
    expired = [sid for sid, s in _sessions.items() if s.is_expired()]
    for sid in expired:
        del _sessions[sid]
