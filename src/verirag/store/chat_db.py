"""Chat history persistence (SQLite).

Answers are only auditable if the *evidence* is stored with them, so this schema
persists more than message text: every turn keeps its citations (document, page,
line range, quote, retrieval score), every per-sentence groundedness verdict,
the provider/model that produced it, latency, and optional user feedback.

Design notes worth mentioning in an interview:

* ``PRAGMA foreign_keys=ON`` with ``ON DELETE CASCADE`` so deleting a session
  cannot orphan citations.
* WAL journalling so the Streamlit UI can read while a write is in flight.
* Every statement is parameterised — no string-built SQL anywhere.
* Full-text search over history via FTS5 when the SQLite build provides it,
  with a LIKE-based fallback so the app never hard-depends on it.
"""

from __future__ import annotations

import json
import sqlite3
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Sequence

from ..models import Answer, Citation, ClaimVerdict, EvidenceSpan

SCHEMA_VERSION = 1

_SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS sessions (
    id         TEXT PRIMARY KEY,
    title      TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    doc_filter TEXT NOT NULL DEFAULT ''
);

CREATE TABLE IF NOT EXISTS messages (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id      TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    role            TEXT NOT NULL CHECK (role IN ('user', 'assistant')),
    content         TEXT NOT NULL,
    created_at      TEXT NOT NULL,
    provider        TEXT NOT NULL DEFAULT '',
    model           TEXT NOT NULL DEFAULT '',
    groundedness    REAL NOT NULL DEFAULT 0.0,
    confidence      TEXT NOT NULL DEFAULT '',
    latency_ms      INTEGER NOT NULL DEFAULT 0,
    refused         INTEGER NOT NULL DEFAULT 0,
    weak_evidence   INTEGER NOT NULL DEFAULT 0,
    retrieval_score REAL NOT NULL DEFAULT 0.0,
    retrieval_trace TEXT NOT NULL DEFAULT '[]'
);

CREATE TABLE IF NOT EXISTS citations (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    message_id      INTEGER NOT NULL REFERENCES messages(id) ON DELETE CASCADE,
    marker          TEXT NOT NULL,
    doc_id          TEXT NOT NULL,
    doc_name        TEXT NOT NULL,
    page_no         INTEGER NOT NULL,
    line_start      INTEGER NOT NULL,
    line_end        INTEGER NOT NULL,
    quote           TEXT NOT NULL,
    retrieval_score REAL NOT NULL DEFAULT 0.0,
    used_in_answer  INTEGER NOT NULL DEFAULT 0,
    bboxes          TEXT NOT NULL DEFAULT '[]'
);

CREATE TABLE IF NOT EXISTS verdicts (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    message_id INTEGER NOT NULL REFERENCES messages(id) ON DELETE CASCADE,
    sentence   TEXT NOT NULL,
    supported  INTEGER NOT NULL DEFAULT 0,
    score      REAL NOT NULL DEFAULT 0.0,
    markers    TEXT NOT NULL DEFAULT '[]',
    evidence   TEXT
);

CREATE TABLE IF NOT EXISTS feedback (
    message_id INTEGER PRIMARY KEY REFERENCES messages(id) ON DELETE CASCADE,
    rating     INTEGER NOT NULL,
    note       TEXT NOT NULL DEFAULT '',
    created_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_messages_session ON messages(session_id, id);
CREATE INDEX IF NOT EXISTS idx_citations_message ON citations(message_id);
CREATE INDEX IF NOT EXISTS idx_citations_doc ON citations(doc_id, page_no);
CREATE INDEX IF NOT EXISTS idx_verdicts_message ON verdicts(message_id);
"""

_FTS_SCHEMA = """
CREATE VIRTUAL TABLE IF NOT EXISTS messages_fts USING fts5(
    content,
    content='messages',
    content_rowid='id',
    tokenize='porter'
);
CREATE TRIGGER IF NOT EXISTS messages_ai AFTER INSERT ON messages BEGIN
    INSERT INTO messages_fts(rowid, content) VALUES (new.id, new.content);
END;
CREATE TRIGGER IF NOT EXISTS messages_ad AFTER DELETE ON messages BEGIN
    INSERT INTO messages_fts(messages_fts, rowid, content) VALUES('delete', old.id, old.content);
END;
CREATE TRIGGER IF NOT EXISTS messages_au AFTER UPDATE ON messages BEGIN
    INSERT INTO messages_fts(messages_fts, rowid, content) VALUES('delete', old.id, old.content);
    INSERT INTO messages_fts(rowid, content) VALUES (new.id, new.content);
END;
"""


def _now() -> str:
    """UTC timestamp with microsecond precision.

    Second granularity is too coarse: two sessions touched within the same
    second tie on ``updated_at`` and then sort arbitrarily in the sidebar, so the
    "most recent chat" can appear to jump around.
    """
    return datetime.now(timezone.utc).isoformat(timespec="microseconds")


@dataclass(slots=True)
class SessionInfo:
    id: str
    title: str
    created_at: str
    updated_at: str
    n_messages: int = 0
    doc_filter: str = ""


@dataclass(slots=True)
class StoredMessage:
    id: int
    session_id: str
    role: str
    content: str
    created_at: str
    provider: str = ""
    model: str = ""
    groundedness: float = 0.0
    confidence: str = ""
    latency_ms: int = 0
    refused: bool = False
    weak_evidence: bool = False
    retrieval_score: float = 0.0
    citations: list[Citation] = None  # type: ignore[assignment]
    verdicts: list[ClaimVerdict] = None  # type: ignore[assignment]
    retrieval_trace: list[dict[str, Any]] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        self.citations = self.citations or []
        self.verdicts = self.verdicts or []
        self.retrieval_trace = self.retrieval_trace or []


class ChatDB:
    """Durable, queryable chat history with full evidence attached."""

    def __init__(self, db_path: str | Path) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.has_fts = False
        self._init_schema()

    # ------------------------------------------------------------ connections
    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(self.db_path, timeout=30.0)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        try:
            yield connection
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def _init_schema(self) -> None:
        with self._connect() as connection:
            connection.execute("PRAGMA journal_mode = WAL")
            connection.executescript(_SCHEMA)
            connection.execute(
                "INSERT OR REPLACE INTO meta(key, value) VALUES ('schema_version', ?)",
                (str(SCHEMA_VERSION),),
            )
            try:
                connection.executescript(_FTS_SCHEMA)
                self.has_fts = True
            except sqlite3.OperationalError:
                self.has_fts = False  # SQLite built without FTS5

    # --------------------------------------------------------------- sessions
    def create_session(self, title: str = "New chat", doc_filter: Sequence[str] = ()) -> str:
        session_id = uuid.uuid4().hex[:16]
        timestamp = _now()
        with self._connect() as connection:
            connection.execute(
                "INSERT INTO sessions(id, title, created_at, updated_at, doc_filter) VALUES (?,?,?,?,?)",
                (session_id, title.strip() or "New chat", timestamp, timestamp, json.dumps(list(doc_filter))),
            )
        return session_id

    def list_sessions(self, limit: int = 100) -> list[SessionInfo]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT s.id, s.title, s.created_at, s.updated_at, s.doc_filter,
                       COUNT(m.id) AS n_messages
                FROM sessions s
                LEFT JOIN messages m ON m.session_id = s.id
                GROUP BY s.id
                ORDER BY s.updated_at DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
        return [
            SessionInfo(
                id=row["id"],
                title=row["title"],
                created_at=row["created_at"],
                updated_at=row["updated_at"],
                n_messages=int(row["n_messages"]),
                doc_filter=row["doc_filter"],
            )
            for row in rows
        ]

    def rename_session(self, session_id: str, title: str) -> None:
        with self._connect() as connection:
            connection.execute(
                "UPDATE sessions SET title = ?, updated_at = ? WHERE id = ?",
                (title.strip() or "New chat", _now(), session_id),
            )

    def delete_session(self, session_id: str) -> bool:
        with self._connect() as connection:
            cursor = connection.execute("DELETE FROM sessions WHERE id = ?", (session_id,))
            return cursor.rowcount > 0

    def touch_session(self, session_id: str) -> None:
        with self._connect() as connection:
            connection.execute("UPDATE sessions SET updated_at = ? WHERE id = ?", (_now(), session_id))

    # --------------------------------------------------------------- messages
    def add_user_message(self, session_id: str, content: str) -> int:
        with self._connect() as connection:
            cursor = connection.execute(
                "INSERT INTO messages(session_id, role, content, created_at) VALUES (?,?,?,?)",
                (session_id, "user", content, _now()),
            )
            connection.execute("UPDATE sessions SET updated_at = ? WHERE id = ?", (_now(), session_id))
            message_id = int(cursor.lastrowid)

        # First user message becomes the session title (like real chat apps).
        with self._connect() as connection:
            row = connection.execute(
                "SELECT title, (SELECT COUNT(*) FROM messages WHERE session_id = ? AND role='user') AS n"
                " FROM sessions WHERE id = ?",
                (session_id, session_id),
            ).fetchone()
            if row and int(row["n"]) == 1 and row["title"] in {"New chat", ""}:
                snippet = content.strip().replace("\n", " ")
                connection.execute(
                    "UPDATE sessions SET title = ? WHERE id = ?",
                    (snippet[:60] + ("…" if len(snippet) > 60 else ""), session_id),
                )
        return message_id

    def add_answer(self, session_id: str, answer: Answer) -> int:
        """Persist an assistant turn together with all of its evidence."""
        with self._connect() as connection:
            cursor = connection.execute(
                """
                INSERT INTO messages(
                    session_id, role, content, created_at, provider, model,
                    groundedness, confidence, latency_ms, refused, weak_evidence,
                    retrieval_score, retrieval_trace
                ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)
                """,
                (
                    session_id,
                    "assistant",
                    answer.text,
                    _now(),
                    answer.provider,
                    answer.model,
                    float(answer.groundedness),
                    answer.confidence_band(),
                    int(answer.latency_ms),
                    int(answer.refused),
                    int(answer.weak_evidence),
                    float(answer.retrieval_score),
                    json.dumps(answer.retrieval_trace, ensure_ascii=False),
                ),
            )
            message_id = int(cursor.lastrowid)

            connection.executemany(
                """
                INSERT INTO citations(
                    message_id, marker, doc_id, doc_name, page_no, line_start,
                    line_end, quote, retrieval_score, used_in_answer, bboxes
                ) VALUES (?,?,?,?,?,?,?,?,?,?,?)
                """,
                [
                    (
                        message_id,
                        c.marker,
                        c.doc_id,
                        c.doc_name,
                        c.page_no,
                        c.line_start,
                        c.line_end,
                        c.quote,
                        float(c.retrieval_score),
                        int(c.used_in_answer),
                        json.dumps([list(b) for b in c.bboxes]),
                    )
                    for c in answer.citations
                ],
            )

            connection.executemany(
                "INSERT INTO verdicts(message_id, sentence, supported, score, markers, evidence)"
                " VALUES (?,?,?,?,?,?)",
                [
                    (
                        message_id,
                        v.sentence,
                        int(v.supported),
                        float(v.score),
                        json.dumps(v.markers),
                        json.dumps(v.evidence.to_dict(), ensure_ascii=False) if v.evidence else None,
                    )
                    for v in answer.verdicts
                ],
            )

            connection.execute("UPDATE sessions SET updated_at = ? WHERE id = ?", (_now(), session_id))
        return message_id

    def get_messages(self, session_id: str, *, with_evidence: bool = True) -> list[StoredMessage]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT * FROM messages WHERE session_id = ? ORDER BY id ASC", (session_id,)
            ).fetchall()
            messages = [self._row_to_message(row) for row in rows]
            if not with_evidence or not messages:
                return messages

            ids = [m.id for m in messages]
            placeholders = ",".join("?" for _ in ids)
            citation_rows = connection.execute(
                f"SELECT * FROM citations WHERE message_id IN ({placeholders}) ORDER BY id ASC", ids
            ).fetchall()
            verdict_rows = connection.execute(
                f"SELECT * FROM verdicts WHERE message_id IN ({placeholders}) ORDER BY id ASC", ids
            ).fetchall()

        by_id = {m.id: m for m in messages}
        for row in citation_rows:
            by_id[int(row["message_id"])].citations.append(_row_to_citation(row))
        for row in verdict_rows:
            by_id[int(row["message_id"])].verdicts.append(_row_to_verdict(row))
        return messages

    def history_pairs(self, session_id: str, limit: int = 3) -> list[tuple[str, str]]:
        """Recent (question, answer) pairs used for follow-up context."""
        messages = self.get_messages(session_id, with_evidence=False)
        pairs: list[tuple[str, str]] = []
        pending: str | None = None
        for message in messages:
            if message.role == "user":
                pending = message.content
            elif pending is not None:
                pairs.append((pending, message.content))
                pending = None
        return pairs[-limit:]

    def delete_message(self, message_id: int) -> bool:
        with self._connect() as connection:
            return connection.execute("DELETE FROM messages WHERE id = ?", (message_id,)).rowcount > 0

    # --------------------------------------------------------------- feedback
    def set_feedback(self, message_id: int, rating: int, note: str = "") -> None:
        with self._connect() as connection:
            connection.execute(
                "INSERT INTO feedback(message_id, rating, note, created_at) VALUES (?,?,?,?)"
                " ON CONFLICT(message_id) DO UPDATE SET rating=excluded.rating,"
                " note=excluded.note, created_at=excluded.created_at",
                (message_id, int(rating), note, _now()),
            )

    def get_feedback(self, message_id: int) -> dict[str, Any] | None:
        with self._connect() as connection:
            row = connection.execute("SELECT * FROM feedback WHERE message_id = ?", (message_id,)).fetchone()
        return dict(row) if row else None

    # ----------------------------------------------------------------- search
    def search(self, query: str, limit: int = 25) -> list[dict[str, Any]]:
        """Search every stored turn; FTS5 when available, LIKE otherwise."""
        term = (query or "").strip()
        if not term:
            return []
        with self._connect() as connection:
            if self.has_fts:
                try:
                    rows = connection.execute(
                        """
                        SELECT m.id, m.session_id, m.role, m.content, m.created_at, s.title
                        FROM messages_fts f
                        JOIN messages m ON m.id = f.rowid
                        JOIN sessions s ON s.id = m.session_id
                        WHERE messages_fts MATCH ?
                        ORDER BY bm25(messages_fts)
                        LIMIT ?
                        """,
                        (_fts_query(term), limit),
                    ).fetchall()
                    return [dict(row) for row in rows]
                except sqlite3.OperationalError:
                    pass
            rows = connection.execute(
                """
                SELECT m.id, m.session_id, m.role, m.content, m.created_at, s.title
                FROM messages m JOIN sessions s ON s.id = m.session_id
                WHERE lower(m.content) LIKE ?
                ORDER BY m.id DESC LIMIT ?
                """,
                (f"%{term.lower()}%", limit),
            ).fetchall()
        return [dict(row) for row in rows]

    # ------------------------------------------------------------------ stats
    def stats(self) -> dict[str, Any]:
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT
                    (SELECT COUNT(*) FROM sessions) AS sessions,
                    (SELECT COUNT(*) FROM messages) AS messages,
                    (SELECT COUNT(*) FROM citations) AS citations,
                    (SELECT COUNT(*) FROM messages WHERE role='assistant' AND refused=1) AS refusals,
                    (SELECT AVG(groundedness) FROM messages WHERE role='assistant' AND refused=0)
                        AS avg_groundedness,
                    (SELECT AVG(latency_ms) FROM messages WHERE role='assistant') AS avg_latency_ms
                """
            ).fetchone()
        result = {key: row[key] for key in row.keys()}
        result["avg_groundedness"] = round(result["avg_groundedness"] or 0.0, 4)
        result["avg_latency_ms"] = int(result["avg_latency_ms"] or 0)
        result["fts5"] = self.has_fts
        result["db_path"] = str(self.db_path)
        return result

    # ----------------------------------------------------------------- export
    def export_session(self, session_id: str) -> dict[str, Any]:
        with self._connect() as connection:
            row = connection.execute("SELECT * FROM sessions WHERE id = ?", (session_id,)).fetchone()
        if row is None:
            raise KeyError(f"unknown session: {session_id}")
        messages = self.get_messages(session_id)
        return {
            "session": dict(row),
            "messages": [
                {
                    "id": m.id,
                    "role": m.role,
                    "content": m.content,
                    "created_at": m.created_at,
                    "provider": m.provider,
                    "model": m.model,
                    "groundedness": m.groundedness,
                    "confidence": m.confidence,
                    "latency_ms": m.latency_ms,
                    "refused": m.refused,
                    "citations": [c.to_dict() for c in m.citations],
                    "verdicts": [v.to_dict() for v in m.verdicts],
                }
                for m in messages
            ],
        }

    def export_markdown(self, session_id: str) -> str:
        payload = self.export_session(session_id)
        lines = [f"# {payload['session']['title']}", "", f"_Session `{session_id}`_", ""]
        for message in payload["messages"]:
            if message["role"] == "user":
                lines += [f"## Q: {message['content']}", ""]
                continue
            lines += [message["content"], ""]
            if message["citations"]:
                lines.append("**Evidence**")
                for citation in message["citations"]:
                    if not citation["used_in_answer"]:
                        continue
                    locator = (
                        f"p.{citation['page_no']} L{citation['line_start']}-{citation['line_end']}"
                    )
                    quote = citation["quote"].strip().replace("\n", " ")
                    lines.append(
                        f"- `[{citation['marker']}]` **{citation['doc_name']}** — {locator}: "
                        f"\"{quote[:220]}\""
                    )
                lines.append("")
            band = message["confidence"]
            lines += [f"_groundedness: {message['groundedness']:.2f} ({band}) · "
                      f"{message['provider']} · {message['latency_ms']} ms_", ""]
        return "\n".join(lines)

    # ---------------------------------------------------------------- helpers
    @staticmethod
    def _row_to_message(row: sqlite3.Row) -> StoredMessage:
        return StoredMessage(
            id=int(row["id"]),
            session_id=row["session_id"],
            role=row["role"],
            content=row["content"],
            created_at=row["created_at"],
            provider=row["provider"],
            model=row["model"],
            groundedness=float(row["groundedness"]),
            confidence=row["confidence"],
            latency_ms=int(row["latency_ms"]),
            refused=bool(row["refused"]),
            weak_evidence=bool(row["weak_evidence"]),
            retrieval_score=float(row["retrieval_score"]),
            retrieval_trace=json.loads(row["retrieval_trace"] or "[]"),
        )


def _row_to_citation(row: sqlite3.Row) -> Citation:
    return Citation(
        marker=row["marker"],
        doc_id=row["doc_id"],
        doc_name=row["doc_name"],
        page_no=int(row["page_no"]),
        line_start=int(row["line_start"]),
        line_end=int(row["line_end"]),
        quote=row["quote"],
        retrieval_score=float(row["retrieval_score"]),
        used_in_answer=bool(row["used_in_answer"]),
        bboxes=[tuple(b) for b in json.loads(row["bboxes"] or "[]")],
    )


def _row_to_verdict(row: sqlite3.Row) -> ClaimVerdict:
    raw_evidence = row["evidence"]
    evidence: EvidenceSpan | None = None
    if raw_evidence:
        payload = json.loads(raw_evidence)
        payload["bboxes"] = [tuple(b) for b in payload.get("bboxes", [])]
        evidence = EvidenceSpan(**payload)
    return ClaimVerdict(
        sentence=row["sentence"],
        supported=bool(row["supported"]),
        score=float(row["score"]),
        markers=json.loads(row["markers"] or "[]"),
        evidence=evidence,
    )


def _fts_query(term: str) -> str:
    """Quote each token so user punctuation cannot break FTS5 syntax."""
    tokens = [t for t in "".join(ch if ch.isalnum() else " " for ch in term).split() if t]
    return " OR ".join(f'"{t}"' for t in tokens) if tokens else '""'
