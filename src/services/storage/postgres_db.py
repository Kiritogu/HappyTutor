from __future__ import annotations

"""
PostgreSQL-backed storage for structured user data.

Implements the same API surface as the SQLite storage in `src/services/storage/user_db.py`.
"""

import json
import logging
from pathlib import Path
import threading
import time
from typing import Any
import uuid
from contextlib import contextmanager

logger = logging.getLogger(__name__)

try:
    import psycopg2  # type: ignore
    from psycopg2.pool import ThreadedConnectionPool  # type: ignore
except Exception:  # pragma: no cover
    psycopg2 = None  # type: ignore
    ThreadedConnectionPool = None  # type: ignore


def _safe_json_dumps(value: Any) -> str:
    try:
        return json.dumps(value, ensure_ascii=False)
    except TypeError:
        return json.dumps(str(value), ensure_ascii=False)


def _load_json_file(path: Path) -> Any | None:
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


class PostgresUserDB:
    def __init__(
        self,
        dsn: str,
        *,
        auto_migrate: bool = True,
        project_root: Path,
        minconn: int = 1,
        maxconn: int = 10,
    ) -> None:
        if not dsn:
            raise ValueError("PostgreSQL DSN is required (DEEPTUTOR_POSTGRES_DSN).")
        if psycopg2 is None or ThreadedConnectionPool is None:
            raise RuntimeError(
                "PostgreSQL backend requires `psycopg2-binary`. "
                "Install it and restart the server."
            )

        self.dsn = dsn
        self._project_root = project_root
        self._lock = threading.RLock()

        self._pool = ThreadedConnectionPool(minconn=minconn, maxconn=maxconn, dsn=dsn)

        self._init_schema()
        if auto_migrate:
            self._migrate_from_files()

    def close(self) -> None:
        with self._lock:
            self._pool.closeall()

    @contextmanager
    def _conn(self):
        conn = self._pool.getconn()
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            self._pool.putconn(conn)

    def _init_schema(self) -> None:
        statements = [
            """
            CREATE TABLE IF NOT EXISTS storage_meta (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS notebooks (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                description TEXT NOT NULL DEFAULT '',
                created_at DOUBLE PRECISION NOT NULL,
                updated_at DOUBLE PRECISION NOT NULL,
                color TEXT NOT NULL DEFAULT '#3B82F6',
                icon TEXT NOT NULL DEFAULT 'book'
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS notebook_records (
                notebook_id TEXT NOT NULL,
                record_id TEXT NOT NULL,
                type TEXT NOT NULL,
                title TEXT NOT NULL,
                user_query TEXT NOT NULL,
                output TEXT NOT NULL,
                metadata TEXT NOT NULL DEFAULT '{}',
                created_at DOUBLE PRECISION NOT NULL,
                kb_name TEXT,
                PRIMARY KEY (notebook_id, record_id),
                FOREIGN KEY (notebook_id) REFERENCES notebooks(id) ON DELETE CASCADE
            )
            """,
            "CREATE INDEX IF NOT EXISTS idx_notebook_records_notebook_id ON notebook_records(notebook_id)",
            "CREATE INDEX IF NOT EXISTS idx_notebook_records_type ON notebook_records(type)",
            """
            CREATE TABLE IF NOT EXISTS history_entries (
                id TEXT PRIMARY KEY,
                timestamp DOUBLE PRECISION NOT NULL,
                type TEXT NOT NULL,
                title TEXT NOT NULL,
                summary TEXT NOT NULL DEFAULT '',
                content TEXT NOT NULL DEFAULT '{}'
            )
            """,
            "CREATE INDEX IF NOT EXISTS idx_history_timestamp ON history_entries(timestamp)",
            "CREATE INDEX IF NOT EXISTS idx_history_type_timestamp ON history_entries(type, timestamp)",
            """
            CREATE TABLE IF NOT EXISTS chat_sessions (
                session_id TEXT PRIMARY KEY,
                title TEXT NOT NULL,
                settings TEXT NOT NULL DEFAULT '{}',
                created_at DOUBLE PRECISION NOT NULL,
                updated_at DOUBLE PRECISION NOT NULL
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS chat_messages (
                id BIGSERIAL PRIMARY KEY,
                session_id TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                sources TEXT,
                timestamp DOUBLE PRECISION NOT NULL,
                FOREIGN KEY (session_id) REFERENCES chat_sessions(session_id) ON DELETE CASCADE
            )
            """,
            "CREATE INDEX IF NOT EXISTS idx_chat_messages_session_ts ON chat_messages(session_id, timestamp)",
            """
            CREATE TABLE IF NOT EXISTS ui_settings (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL,
                updated_at DOUBLE PRECISION NOT NULL
            )
            """,
            "CREATE EXTENSION IF NOT EXISTS citext",
            """
            CREATE TABLE IF NOT EXISTS users (
                id UUID PRIMARY KEY,
                email CITEXT NOT NULL UNIQUE,
                password_hash TEXT NOT NULL,
                is_email_verified BOOLEAN NOT NULL DEFAULT FALSE,
                status TEXT NOT NULL DEFAULT 'active',
                created_at DOUBLE PRECISION NOT NULL,
                updated_at DOUBLE PRECISION NOT NULL,
                last_login_at DOUBLE PRECISION
            )
            """,
            "CREATE INDEX IF NOT EXISTS idx_users_email ON users(email)",
            """
            CREATE TABLE IF NOT EXISTS refresh_tokens (
                id UUID PRIMARY KEY,
                user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                token_hash TEXT NOT NULL UNIQUE,
                expires_at DOUBLE PRECISION NOT NULL,
                revoked_at DOUBLE PRECISION,
                created_at DOUBLE PRECISION NOT NULL,
                created_ip TEXT,
                user_agent TEXT
            )
            """,
            "CREATE INDEX IF NOT EXISTS idx_refresh_tokens_user_id ON refresh_tokens(user_id)",
            "CREATE INDEX IF NOT EXISTS idx_refresh_tokens_expires_at ON refresh_tokens(expires_at)",
            """
            CREATE TABLE IF NOT EXISTS email_verification_tokens (
                id UUID PRIMARY KEY,
                user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                token_hash TEXT NOT NULL UNIQUE,
                expires_at DOUBLE PRECISION NOT NULL,
                used_at DOUBLE PRECISION,
                created_at DOUBLE PRECISION NOT NULL
            )
            """,
            "CREATE INDEX IF NOT EXISTS idx_email_verification_tokens_user_id ON email_verification_tokens(user_id)",
            """
            CREATE TABLE IF NOT EXISTS password_reset_tokens (
                id UUID PRIMARY KEY,
                user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                token_hash TEXT NOT NULL UNIQUE,
                expires_at DOUBLE PRECISION NOT NULL,
                used_at DOUBLE PRECISION,
                created_at DOUBLE PRECISION NOT NULL
            )
            """,
            "CREATE INDEX IF NOT EXISTS idx_password_reset_tokens_user_id ON password_reset_tokens(user_id)",
        ]

        with self._conn() as conn:
            with conn.cursor() as cur:
                for stmt in statements:
                    cur.execute(stmt)

    # -------------------------------------------------------------------------
    # Meta
    # -------------------------------------------------------------------------

    def _get_meta(self, key: str) -> str | None:
        with self._conn() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT value FROM storage_meta WHERE key = %s", (key,))
                row = cur.fetchone()
                return str(row[0]) if row else None

    def _set_meta(self, key: str, value: str) -> None:
        with self._conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "INSERT INTO storage_meta(key, value) VALUES (%s, %s) "
                    "ON CONFLICT(key) DO UPDATE SET value = EXCLUDED.value",
                    (key, value),
                )

    # -------------------------------------------------------------------------
    # Migration (JSON files -> PostgreSQL)
    # -------------------------------------------------------------------------

    def _migrate_from_files(self) -> None:
        migrated_key = "migrated_from_files_v1"
        if self._get_meta(migrated_key) == "1":
            return

        user_dir = self._project_root / "data" / "user"
        self._migrate_notebooks_from_files(user_dir / "notebook")
        self._migrate_history_from_file(user_dir / "user_history.json")
        self._migrate_chat_from_file(user_dir / "chat_sessions.json")
        self._migrate_ui_settings_from_file(user_dir / "settings" / "interface.json")

        self._set_meta(migrated_key, "1")

    def _migrate_notebooks_from_files(self, notebook_dir: Path) -> None:
        if not notebook_dir.exists():
            return

        notebook_files = [p for p in notebook_dir.glob("*.json") if p.name != "notebooks_index.json"]
        if not notebook_files:
            return

        with self._conn() as conn:
            with conn.cursor() as cur:
                for notebook_file in notebook_files:
                    notebook = _load_json_file(notebook_file)
                    if not isinstance(notebook, dict):
                        continue

                    notebook_id = str(notebook.get("id") or "").strip()
                    if not notebook_id:
                        continue

                    name = str(notebook.get("name") or "Untitled")
                    description = str(notebook.get("description") or "")
                    created_at = float(notebook.get("created_at") or time.time())
                    updated_at = float(notebook.get("updated_at") or created_at)
                    color = str(notebook.get("color") or "#3B82F6")
                    icon = str(notebook.get("icon") or "book")

                    cur.execute(
                        "INSERT INTO notebooks(id, name, description, created_at, updated_at, color, icon) "
                        "VALUES (%s, %s, %s, %s, %s, %s, %s) "
                        "ON CONFLICT(id) DO UPDATE SET "
                        "name=EXCLUDED.name, description=EXCLUDED.description, updated_at=EXCLUDED.updated_at, "
                        "color=EXCLUDED.color, icon=EXCLUDED.icon",
                        (notebook_id, name, description, created_at, updated_at, color, icon),
                    )

                    for record in notebook.get("records", []) or []:
                        if not isinstance(record, dict):
                            continue
                        record_id = str(record.get("id") or "").strip()
                        if not record_id:
                            continue

                        record_type = str(record.get("type") or "")
                        title = str(record.get("title") or "")
                        user_query = str(record.get("user_query") or "")
                        output = str(record.get("output") or "")
                        metadata = (
                            record.get("metadata") if isinstance(record.get("metadata"), dict) else {}
                        )
                        created_at_r = float(record.get("created_at") or updated_at)
                        kb_name = record.get("kb_name")

                        cur.execute(
                            "INSERT INTO notebook_records("
                            "notebook_id, record_id, type, title, user_query, output, metadata, created_at, kb_name"
                            ") VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s) "
                            "ON CONFLICT(notebook_id, record_id) DO UPDATE SET "
                            "type=EXCLUDED.type, title=EXCLUDED.title, user_query=EXCLUDED.user_query, "
                            "output=EXCLUDED.output, metadata=EXCLUDED.metadata, created_at=EXCLUDED.created_at, "
                            "kb_name=EXCLUDED.kb_name",
                            (
                                notebook_id,
                                record_id,
                                record_type,
                                title,
                                user_query,
                                output,
                                _safe_json_dumps(metadata),
                                created_at_r,
                                kb_name,
                            ),
                        )

    def _migrate_history_from_file(self, history_file: Path) -> None:
        data = _load_json_file(history_file)
        sessions: list[dict[str, Any]] = []
        if isinstance(data, dict):
            raw = data.get("sessions", [])
            if isinstance(raw, list):
                sessions = [s for s in raw if isinstance(s, dict)]
        elif isinstance(data, list):
            sessions = [s for s in data if isinstance(s, dict)]

        if not sessions:
            return

        with self._conn() as conn:
            with conn.cursor() as cur:
                for entry in sessions:
                    entry_id = str(entry.get("id") or "").strip()
                    if not entry_id:
                        continue
                    ts = float(entry.get("timestamp") or time.time())
                    entry_type = str(entry.get("type") or "")
                    title = str(entry.get("title") or "")
                    summary = str(entry.get("summary") or "")
                    content = entry.get("content") if isinstance(entry.get("content"), dict) else {}

                    cur.execute(
                        "INSERT INTO history_entries(id, timestamp, type, title, summary, content) "
                        "VALUES (%s, %s, %s, %s, %s, %s) "
                        "ON CONFLICT(id) DO UPDATE SET "
                        "timestamp=EXCLUDED.timestamp, type=EXCLUDED.type, title=EXCLUDED.title, "
                        "summary=EXCLUDED.summary, content=EXCLUDED.content",
                        (entry_id, ts, entry_type, title, summary, _safe_json_dumps(content)),
                    )

    def _migrate_chat_from_file(self, chat_file: Path) -> None:
        data = _load_json_file(chat_file)
        if not isinstance(data, dict):
            return
        sessions = data.get("sessions", [])
        if not isinstance(sessions, list):
            return

        with self._conn() as conn:
            with conn.cursor() as cur:
                for session in sessions:
                    if not isinstance(session, dict):
                        continue
                    session_id = str(session.get("session_id") or "").strip()
                    if not session_id:
                        continue

                    title = str(session.get("title") or "New Chat")
                    settings = session.get("settings") if isinstance(session.get("settings"), dict) else {}
                    created_at = float(session.get("created_at") or time.time())
                    updated_at = float(session.get("updated_at") or created_at)

                    cur.execute(
                        "INSERT INTO chat_sessions(session_id, title, settings, created_at, updated_at) "
                        "VALUES (%s, %s, %s, %s, %s) "
                        "ON CONFLICT(session_id) DO UPDATE SET "
                        "title=EXCLUDED.title, settings=EXCLUDED.settings, updated_at=EXCLUDED.updated_at",
                        (session_id, title, _safe_json_dumps(settings), created_at, updated_at),
                    )

                    cur.execute(
                        "SELECT 1 FROM chat_messages WHERE session_id = %s LIMIT 1",
                        (session_id,),
                    )
                    if cur.fetchone():
                        continue

                    for msg in session.get("messages", []) or []:
                        if not isinstance(msg, dict):
                            continue
                        role = str(msg.get("role") or "")
                        content = str(msg.get("content") or "")
                        ts = float(msg.get("timestamp") or updated_at)
                        sources = msg.get("sources")
                        sources_json = _safe_json_dumps(sources) if isinstance(sources, dict) else None

                        cur.execute(
                            "INSERT INTO chat_messages(session_id, role, content, sources, timestamp) "
                            "VALUES (%s, %s, %s, %s, %s)",
                            (session_id, role, content, sources_json, ts),
                        )

    def _migrate_ui_settings_from_file(self, interface_file: Path) -> None:
        settings = _load_json_file(interface_file)
        if not isinstance(settings, dict):
            return
        now = time.time()
        with self._conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "INSERT INTO ui_settings(key, value, updated_at) VALUES (%s, %s, %s) "
                    "ON CONFLICT(key) DO UPDATE SET value=EXCLUDED.value, updated_at=EXCLUDED.updated_at",
                    ("interface", _safe_json_dumps(settings), now),
                )

    # -------------------------------------------------------------------------
    # Notebooks
    # -------------------------------------------------------------------------

    def notebook_create(
        self,
        *,
        name: str,
        description: str = "",
        color: str = "#3B82F6",
        icon: str = "book",
    ) -> dict[str, Any]:
        now = time.time()
        for _ in range(3):
            notebook_id = uuid.uuid4().hex[:8]
            try:
                with self._conn() as conn:
                    with conn.cursor() as cur:
                        cur.execute(
                            "INSERT INTO notebooks(id, name, description, created_at, updated_at, color, icon) "
                            "VALUES (%s, %s, %s, %s, %s, %s, %s)",
                            (notebook_id, name, description, now, now, color, icon),
                        )
                return {
                    "id": notebook_id,
                    "name": name,
                    "description": description,
                    "created_at": now,
                    "updated_at": now,
                    "records": [],
                    "color": color,
                    "icon": icon,
                }
            except Exception as e:
                if psycopg2 is not None and isinstance(e, psycopg2.IntegrityError):
                    continue
                raise

        raise RuntimeError("Failed to create notebook (ID collision)")

    def notebook_list(self, *, limit: int | None = None) -> list[dict[str, Any]]:
        sql = """
        SELECT
            n.id,
            n.name,
            n.description,
            n.created_at,
            n.updated_at,
            n.color,
            n.icon,
            COALESCE(r.record_count, 0) AS record_count
        FROM notebooks n
        LEFT JOIN (
            SELECT notebook_id, COUNT(1) AS record_count
            FROM notebook_records
            GROUP BY notebook_id
        ) r ON r.notebook_id = n.id
        ORDER BY n.updated_at DESC
        """
        params: tuple[Any, ...] = ()
        if isinstance(limit, int) and limit > 0:
            sql += " LIMIT %s"
            params = (limit,)

        with self._conn() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, params)
                rows = cur.fetchall()

        return [
            {
                "id": r[0],
                "name": r[1],
                "description": r[2],
                "created_at": float(r[3]),
                "updated_at": float(r[4]),
                "record_count": int(r[7] or 0),
                "color": r[5],
                "icon": r[6],
            }
            for r in rows
        ]

    def notebook_get(self, notebook_id: str) -> dict[str, Any] | None:
        with self._conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT id, name, description, created_at, updated_at, color, icon "
                    "FROM notebooks WHERE id = %s",
                    (notebook_id,),
                )
                nb = cur.fetchone()
                if not nb:
                    return None

                cur.execute(
                    "SELECT record_id, type, title, user_query, output, metadata, created_at, kb_name "
                    "FROM notebook_records WHERE notebook_id = %s "
                    "ORDER BY created_at ASC",
                    (notebook_id,),
                )
                record_rows = cur.fetchall()

        records: list[dict[str, Any]] = []
        for r in record_rows:
            metadata: dict[str, Any] = {}
            try:
                metadata = json.loads(r[5]) if r[5] else {}
                if not isinstance(metadata, dict):
                    metadata = {}
            except Exception:
                metadata = {}
            records.append(
                {
                    "id": r[0],
                    "type": r[1],
                    "title": r[2],
                    "user_query": r[3],
                    "output": r[4],
                    "metadata": metadata,
                    "created_at": float(r[6]),
                    "kb_name": r[7],
                }
            )

        return {
            "id": nb[0],
            "name": nb[1],
            "description": nb[2],
            "created_at": float(nb[3]),
            "updated_at": float(nb[4]),
            "records": records,
            "color": nb[5],
            "icon": nb[6],
        }

    def notebook_update(
        self,
        notebook_id: str,
        *,
        name: str | None = None,
        description: str | None = None,
        color: str | None = None,
        icon: str | None = None,
    ) -> dict[str, Any] | None:
        now = time.time()
        fields: list[str] = []
        values: list[Any] = []
        if name is not None:
            fields.append("name = %s")
            values.append(name)
        if description is not None:
            fields.append("description = %s")
            values.append(description)
        if color is not None:
            fields.append("color = %s")
            values.append(color)
        if icon is not None:
            fields.append("icon = %s")
            values.append(icon)

        fields.append("updated_at = %s")
        values.append(now)
        values.append(notebook_id)

        with self._conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    f"UPDATE notebooks SET {', '.join(fields)} WHERE id = %s",
                    tuple(values),
                )
                if cur.rowcount == 0:
                    return None

        return self.notebook_get(notebook_id)

    def notebook_delete(self, notebook_id: str) -> bool:
        with self._conn() as conn:
            with conn.cursor() as cur:
                cur.execute("DELETE FROM notebooks WHERE id = %s", (notebook_id,))
                return cur.rowcount > 0

    def notebook_add_record(
        self,
        *,
        notebook_ids: list[str],
        record_type: Any,
        title: str,
        user_query: str,
        output: str,
        metadata: dict | None = None,
        kb_name: str | None = None,
    ) -> dict[str, Any]:
        record_id = uuid.uuid4().hex[:8]
        now = time.time()

        record = {
            "id": record_id,
            "type": record_type,
            "title": title,
            "user_query": user_query,
            "output": output,
            "metadata": metadata or {},
            "created_at": now,
            "kb_name": kb_name,
        }

        added_to: list[str] = []
        with self._conn() as conn:
            with conn.cursor() as cur:
                for notebook_id in notebook_ids:
                    cur.execute("SELECT 1 FROM notebooks WHERE id = %s", (notebook_id,))
                    if not cur.fetchone():
                        continue

                    cur.execute(
                        "INSERT INTO notebook_records("
                        "notebook_id, record_id, type, title, user_query, output, metadata, created_at, kb_name"
                        ") VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s) "
                        "ON CONFLICT(notebook_id, record_id) DO UPDATE SET "
                        "type=EXCLUDED.type, title=EXCLUDED.title, user_query=EXCLUDED.user_query, "
                        "output=EXCLUDED.output, metadata=EXCLUDED.metadata, created_at=EXCLUDED.created_at, "
                        "kb_name=EXCLUDED.kb_name",
                        (
                            notebook_id,
                            record_id,
                            str(record_type),
                            title,
                            user_query,
                            output,
                            _safe_json_dumps(metadata or {}),
                            now,
                            kb_name,
                        ),
                    )
                    cur.execute(
                        "UPDATE notebooks SET updated_at = %s WHERE id = %s",
                        (now, notebook_id),
                    )
                    added_to.append(notebook_id)

        return {"record": record, "added_to_notebooks": added_to}

    def notebook_remove_record(self, *, notebook_id: str, record_id: str) -> bool:
        now = time.time()
        with self._conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "DELETE FROM notebook_records WHERE notebook_id = %s AND record_id = %s",
                    (notebook_id, record_id),
                )
                if cur.rowcount == 0:
                    return False
                cur.execute(
                    "UPDATE notebooks SET updated_at = %s WHERE id = %s",
                    (now, notebook_id),
                )
                return True

    def notebook_statistics(self) -> dict[str, Any]:
        with self._conn() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT COUNT(1) FROM notebooks")
                total_notebooks = int(cur.fetchone()[0])
                cur.execute("SELECT COUNT(1) FROM notebook_records")
                total_records = int(cur.fetchone()[0])
                cur.execute("SELECT type, COUNT(1) FROM notebook_records GROUP BY type")
                type_rows = cur.fetchall()

        type_counts: dict[str, int] = {str(r[0]): int(r[1]) for r in type_rows}
        recent = self.notebook_list(limit=5)
        return {
            "total_notebooks": total_notebooks,
            "total_records": total_records,
            "records_by_type": {
                "solve": type_counts.get("solve", 0),
                "question": type_counts.get("question", 0),
                "research": type_counts.get("research", 0),
                "co_writer": type_counts.get("co_writer", 0),
            },
            "recent_notebooks": recent,
        }

    # -------------------------------------------------------------------------
    # History
    # -------------------------------------------------------------------------

    def history_add_entry(
        self,
        *,
        activity_type: Any,
        title: str,
        content: dict,
        summary: str = "",
        limit: int = 100,
    ) -> dict[str, Any]:
        base_id = str(int(time.time() * 1000))
        entry_id = base_id
        ts = time.time()
        entry = {
            "id": entry_id,
            "timestamp": ts,
            "type": activity_type,
            "title": title,
            "summary": summary,
            "content": content,
        }

        with self._conn() as conn:
            with conn.cursor() as cur:
                try:
                    cur.execute(
                        "INSERT INTO history_entries(id, timestamp, type, title, summary, content) "
                        "VALUES (%s, %s, %s, %s, %s, %s)",
                        (entry_id, ts, str(activity_type), title, summary, _safe_json_dumps(content)),
                    )
                except Exception as e:
                    if psycopg2 is not None and isinstance(e, psycopg2.IntegrityError):
                        entry_id = f"{base_id}_{uuid.uuid4().hex[:4]}"
                        entry["id"] = entry_id
                        cur.execute(
                            "INSERT INTO history_entries(id, timestamp, type, title, summary, content) "
                            "VALUES (%s, %s, %s, %s, %s, %s)",
                            (
                                entry_id,
                                ts,
                                str(activity_type),
                                title,
                                summary,
                                _safe_json_dumps(content),
                            ),
                        )
                    else:
                        raise

                cur.execute("SELECT COUNT(1) FROM history_entries")
                count = int(cur.fetchone()[0])
                extra = count - int(limit)
                if extra > 0:
                    cur.execute(
                        "DELETE FROM history_entries WHERE id IN ("
                        "  SELECT id FROM history_entries ORDER BY timestamp ASC LIMIT %s"
                        ")",
                        (extra,),
                    )

        return entry

    def history_get_recent(self, *, limit: int = 10, type_filter: str | None = None) -> list[dict]:
        if type_filter:
            sql = (
                "SELECT id, timestamp, type, title, summary, content "
                "FROM history_entries WHERE type = %s ORDER BY timestamp DESC LIMIT %s"
            )
            params = (type_filter, limit)
        else:
            sql = (
                "SELECT id, timestamp, type, title, summary, content "
                "FROM history_entries ORDER BY timestamp DESC LIMIT %s"
            )
            params = (limit,)

        with self._conn() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, params)
                rows = cur.fetchall()

        results: list[dict[str, Any]] = []
        for r in rows:
            parsed_content: dict[str, Any] = {}
            try:
                parsed_content = json.loads(r[5]) if r[5] else {}
                if not isinstance(parsed_content, dict):
                    parsed_content = {}
            except Exception:
                parsed_content = {}
            results.append(
                {
                    "id": r[0],
                    "timestamp": float(r[1]),
                    "type": r[2],
                    "title": r[3],
                    "summary": r[4],
                    "content": parsed_content,
                }
            )
        return results

    def history_get_entry(self, entry_id: str) -> dict[str, Any] | None:
        with self._conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT id, timestamp, type, title, summary, content "
                    "FROM history_entries WHERE id = %s",
                    (entry_id,),
                )
                r = cur.fetchone()
                if not r:
                    return None

        parsed_content: dict[str, Any] = {}
        try:
            parsed_content = json.loads(r[5]) if r[5] else {}
            if not isinstance(parsed_content, dict):
                parsed_content = {}
        except Exception:
            parsed_content = {}
        return {
            "id": r[0],
            "timestamp": float(r[1]),
            "type": r[2],
            "title": r[3],
            "summary": r[4],
            "content": parsed_content,
        }

    # -------------------------------------------------------------------------
    # Chat sessions
    # -------------------------------------------------------------------------

    def chat_create_session(
        self,
        *,
        title: str = "New Chat",
        settings: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        session_id = f"chat_{int(time.time() * 1000)}_{uuid.uuid4().hex[:8]}"
        now = time.time()
        session = {
            "session_id": session_id,
            "title": title[:100],
            "messages": [],
            "settings": settings or {},
            "created_at": now,
            "updated_at": now,
        }

        with self._conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "INSERT INTO chat_sessions(session_id, title, settings, created_at, updated_at) "
                    "VALUES (%s, %s, %s, %s, %s)",
                    (session_id, session["title"], _safe_json_dumps(session["settings"]), now, now),
                )

        return session

    def chat_get_session(self, session_id: str) -> dict[str, Any] | None:
        with self._conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT session_id, title, settings, created_at, updated_at "
                    "FROM chat_sessions WHERE session_id = %s",
                    (session_id,),
                )
                s = cur.fetchone()
                if not s:
                    return None

                cur.execute(
                    "SELECT role, content, sources, timestamp "
                    "FROM chat_messages WHERE session_id = %s "
                    "ORDER BY timestamp ASC, id ASC",
                    (session_id,),
                )
                msg_rows = cur.fetchall()

        settings: dict[str, Any] = {}
        try:
            settings = json.loads(s[2]) if s[2] else {}
            if not isinstance(settings, dict):
                settings = {}
        except Exception:
            settings = {}

        messages: list[dict[str, Any]] = []
        for m in msg_rows:
            msg: dict[str, Any] = {
                "role": m[0],
                "content": m[1],
                "timestamp": float(m[3]),
            }
            if m[2]:
                try:
                    msg["sources"] = json.loads(m[2])
                except Exception:
                    pass
            messages.append(msg)

        return {
            "session_id": s[0],
            "title": s[1],
            "messages": messages,
            "settings": settings,
            "created_at": float(s[3]),
            "updated_at": float(s[4]),
        }

    def chat_update_session(
        self,
        session_id: str,
        *,
        messages: list[dict[str, Any]] | None = None,
        title: str | None = None,
        settings: dict[str, Any] | None = None,
    ) -> dict[str, Any] | None:
        now = time.time()
        with self._conn() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT 1 FROM chat_sessions WHERE session_id = %s", (session_id,))
                if not cur.fetchone():
                    return None

                if title is not None:
                    cur.execute(
                        "UPDATE chat_sessions SET title = %s, updated_at = %s WHERE session_id = %s",
                        (title[:100], now, session_id),
                    )
                if settings is not None:
                    cur.execute(
                        "UPDATE chat_sessions SET settings = %s, updated_at = %s WHERE session_id = %s",
                        (_safe_json_dumps(settings), now, session_id),
                    )
                if title is None and settings is None:
                    cur.execute(
                        "UPDATE chat_sessions SET updated_at = %s WHERE session_id = %s",
                        (now, session_id),
                    )

                if messages is not None:
                    cur.execute("DELETE FROM chat_messages WHERE session_id = %s", (session_id,))
                    for msg in messages:
                        if not isinstance(msg, dict):
                            continue
                        role = str(msg.get("role") or "")
                        content = str(msg.get("content") or "")
                        ts = float(msg.get("timestamp") or time.time())
                        sources = msg.get("sources")
                        sources_json = _safe_json_dumps(sources) if isinstance(sources, dict) else None
                        cur.execute(
                            "INSERT INTO chat_messages(session_id, role, content, sources, timestamp) "
                            "VALUES (%s, %s, %s, %s, %s)",
                            (session_id, role, content, sources_json, ts),
                        )

        return self.chat_get_session(session_id)

    def chat_add_message(
        self,
        *,
        session_id: str,
        role: str,
        content: str,
        sources: dict[str, Any] | None = None,
    ) -> dict[str, Any] | None:
        with self._conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT title FROM chat_sessions WHERE session_id = %s",
                    (session_id,),
                )
                row = cur.fetchone()
                if not row:
                    return None

                ts = time.time()
                sources_json = _safe_json_dumps(sources) if isinstance(sources, dict) else None
                cur.execute(
                    "INSERT INTO chat_messages(session_id, role, content, sources, timestamp) "
                    "VALUES (%s, %s, %s, %s, %s)",
                    (session_id, role, content, sources_json, ts),
                )

                new_title = None
                if row[0] == "New Chat" and role == "user":
                    new_title = content[:50] + ("..." if len(content) > 50 else "")

                if new_title is not None:
                    cur.execute(
                        "UPDATE chat_sessions SET title = %s, updated_at = %s WHERE session_id = %s",
                        (new_title[:100], ts, session_id),
                    )
                else:
                    cur.execute(
                        "UPDATE chat_sessions SET updated_at = %s WHERE session_id = %s",
                        (ts, session_id),
                    )

        return self.chat_get_session(session_id)

    def chat_list_sessions(self, *, limit: int = 20, include_messages: bool = False) -> list[dict]:
        if include_messages:
            with self._conn() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        "SELECT session_id FROM chat_sessions ORDER BY updated_at DESC LIMIT %s",
                        (limit,),
                    )
                    session_rows = cur.fetchall()
            session_ids = [r[0] for r in session_rows]
            return [self.chat_get_session(sid) for sid in session_ids if sid]

        sql = """
        SELECT
            s.session_id,
            s.title,
            s.settings,
            s.created_at,
            s.updated_at,
            COALESCE(mc.message_count, 0) AS message_count,
            COALESCE(lm.last_message, '') AS last_message
        FROM chat_sessions s
        LEFT JOIN (
            SELECT session_id, COUNT(1) AS message_count
            FROM chat_messages
            GROUP BY session_id
        ) mc ON mc.session_id = s.session_id
        LEFT JOIN LATERAL (
            SELECT SUBSTRING(m.content FROM 1 FOR 100) AS last_message
            FROM chat_messages m
            WHERE m.session_id = s.session_id
            ORDER BY m.timestamp DESC, m.id DESC
            LIMIT 1
        ) lm ON TRUE
        ORDER BY s.updated_at DESC
        LIMIT %s
        """
        with self._conn() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, (limit,))
                rows = cur.fetchall()

        summaries: list[dict[str, Any]] = []
        for r in rows:
            parsed_settings: dict[str, Any] = {}
            try:
                parsed_settings = json.loads(r[2]) if r[2] else {}
                if not isinstance(parsed_settings, dict):
                    parsed_settings = {}
            except Exception:
                parsed_settings = {}
            summaries.append(
                {
                    "session_id": r[0],
                    "title": r[1],
                    "message_count": int(r[5] or 0),
                    "settings": parsed_settings,
                    "created_at": float(r[3]),
                    "updated_at": float(r[4]),
                    "last_message": r[6] or "",
                }
            )

        return summaries

    def chat_delete_session(self, session_id: str) -> bool:
        with self._conn() as conn:
            with conn.cursor() as cur:
                cur.execute("DELETE FROM chat_sessions WHERE session_id = %s", (session_id,))
                return cur.rowcount > 0

    def chat_clear_all_sessions(self) -> int:
        with self._conn() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT COUNT(1) FROM chat_sessions")
                count = int(cur.fetchone()[0])
                cur.execute("DELETE FROM chat_sessions")
                return count

    # -------------------------------------------------------------------------
    # UI settings
    # -------------------------------------------------------------------------

    def ui_get(self, *, key: str = "interface") -> dict[str, Any] | None:
        with self._conn() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT value FROM ui_settings WHERE key = %s", (key,))
                row = cur.fetchone()
                if not row:
                    return None

        try:
            data = json.loads(row[0]) if row[0] else {}
            return data if isinstance(data, dict) else None
        except Exception:
            return None

    def ui_set(self, *, key: str = "interface", value: dict[str, Any]) -> None:
        now = time.time()
        with self._conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "INSERT INTO ui_settings(key, value, updated_at) VALUES (%s, %s, %s) "
                    "ON CONFLICT(key) DO UPDATE SET value=EXCLUDED.value, updated_at=EXCLUDED.updated_at",
                    (key, _safe_json_dumps(value), now),
                )

    # -------------------------------------------------------------------------
    # Authentication
    # -------------------------------------------------------------------------

    def auth_create_user(self, *, email: str, password_hash: str) -> dict[str, Any]:
        now = time.time()
        user_id = str(uuid.uuid4())
        with self._conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "INSERT INTO users(id, email, password_hash, is_email_verified, status, created_at, updated_at) "
                    "VALUES (%s, %s, %s, %s, %s, %s, %s)",
                    (user_id, email, password_hash, False, "active", now, now),
                )
        return {
            "id": user_id,
            "email": email,
            "password_hash": password_hash,
            "is_email_verified": False,
            "status": "active",
            "created_at": now,
            "updated_at": now,
            "last_login_at": None,
        }

    def auth_get_user_by_email(self, email: str) -> dict[str, Any] | None:
        with self._conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT id, email, password_hash, is_email_verified, status, created_at, updated_at, last_login_at "
                    "FROM users WHERE email = %s",
                    (email,),
                )
                row = cur.fetchone()
        if not row:
            return None
        return {
            "id": str(row[0]),
            "email": str(row[1]),
            "password_hash": str(row[2]),
            "is_email_verified": bool(row[3]),
            "status": str(row[4]),
            "created_at": float(row[5]),
            "updated_at": float(row[6]),
            "last_login_at": float(row[7]) if row[7] is not None else None,
        }

    def auth_get_user_by_id(self, user_id: str) -> dict[str, Any] | None:
        with self._conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT id, email, password_hash, is_email_verified, status, created_at, updated_at, last_login_at "
                    "FROM users WHERE id = %s",
                    (user_id,),
                )
                row = cur.fetchone()
        if not row:
            return None
        return {
            "id": str(row[0]),
            "email": str(row[1]),
            "password_hash": str(row[2]),
            "is_email_verified": bool(row[3]),
            "status": str(row[4]),
            "created_at": float(row[5]),
            "updated_at": float(row[6]),
            "last_login_at": float(row[7]) if row[7] is not None else None,
        }

    def auth_update_password_hash(self, *, user_id: str, password_hash: str) -> bool:
        now = time.time()
        with self._conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "UPDATE users SET password_hash = %s, updated_at = %s WHERE id = %s",
                    (password_hash, now, user_id),
                )
                return cur.rowcount > 0

    def auth_set_email_verified(self, *, user_id: str) -> bool:
        now = time.time()
        with self._conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "UPDATE users SET is_email_verified = TRUE, updated_at = %s WHERE id = %s",
                    (now, user_id),
                )
                return cur.rowcount > 0

    def auth_update_last_login_at(self, *, user_id: str) -> bool:
        now = time.time()
        with self._conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "UPDATE users SET last_login_at = %s, updated_at = %s WHERE id = %s",
                    (now, now, user_id),
                )
                return cur.rowcount > 0

    def auth_create_refresh_token(
        self,
        *,
        user_id: str,
        token_hash: str,
        expires_at: float,
        token_id: str | None = None,
        created_ip: str | None = None,
        user_agent: str | None = None,
    ) -> dict[str, Any]:
        token_id = token_id or str(uuid.uuid4())
        now = time.time()
        with self._conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "INSERT INTO refresh_tokens(id, user_id, token_hash, expires_at, revoked_at, created_at, created_ip, user_agent) "
                    "VALUES (%s, %s, %s, %s, %s, %s, %s, %s)",
                    (token_id, user_id, token_hash, expires_at, None, now, created_ip, user_agent),
                )
        return {
            "id": token_id,
            "user_id": user_id,
            "token_hash": token_hash,
            "expires_at": expires_at,
            "revoked_at": None,
            "created_at": now,
            "created_ip": created_ip,
            "user_agent": user_agent,
        }

    def auth_get_refresh_token(self, token_id: str) -> dict[str, Any] | None:
        with self._conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT id, user_id, token_hash, expires_at, revoked_at, created_at, created_ip, user_agent "
                    "FROM refresh_tokens WHERE id = %s",
                    (token_id,),
                )
                row = cur.fetchone()
        if not row:
            return None
        return {
            "id": str(row[0]),
            "user_id": str(row[1]),
            "token_hash": str(row[2]),
            "expires_at": float(row[3]),
            "revoked_at": float(row[4]) if row[4] is not None else None,
            "created_at": float(row[5]),
            "created_ip": row[6],
            "user_agent": row[7],
        }

    def auth_get_refresh_token_by_hash(self, token_hash: str) -> dict[str, Any] | None:
        with self._conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT id, user_id, token_hash, expires_at, revoked_at, created_at, created_ip, user_agent "
                    "FROM refresh_tokens WHERE token_hash = %s",
                    (token_hash,),
                )
                row = cur.fetchone()
        if not row:
            return None
        return {
            "id": str(row[0]),
            "user_id": str(row[1]),
            "token_hash": str(row[2]),
            "expires_at": float(row[3]),
            "revoked_at": float(row[4]) if row[4] is not None else None,
            "created_at": float(row[5]),
            "created_ip": row[6],
            "user_agent": row[7],
        }

    def auth_is_refresh_token_active(self, token_id: str) -> bool:
        now = time.time()
        with self._conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT 1 FROM refresh_tokens WHERE id = %s AND revoked_at IS NULL AND expires_at > %s",
                    (token_id, now),
                )
                row = cur.fetchone()
                return row is not None

    def auth_revoke_refresh_token(self, token_id: str) -> bool:
        now = time.time()
        with self._conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "UPDATE refresh_tokens SET revoked_at = %s WHERE id = %s AND revoked_at IS NULL",
                    (now, token_id),
                )
                return cur.rowcount > 0

    def auth_revoke_all_refresh_tokens(self, *, user_id: str) -> int:
        now = time.time()
        with self._conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "UPDATE refresh_tokens SET revoked_at = %s WHERE user_id = %s AND revoked_at IS NULL",
                    (now, user_id),
                )
                return int(cur.rowcount)

    def auth_create_email_verification_token(
        self,
        *,
        user_id: str,
        token_hash: str,
        expires_at: float,
    ) -> dict[str, Any]:
        token_id = str(uuid.uuid4())
        now = time.time()
        with self._conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "INSERT INTO email_verification_tokens(id, user_id, token_hash, expires_at, used_at, created_at) "
                    "VALUES (%s, %s, %s, %s, %s, %s)",
                    (token_id, user_id, token_hash, expires_at, None, now),
                )
        return {
            "id": token_id,
            "user_id": user_id,
            "token_hash": token_hash,
            "expires_at": expires_at,
            "used_at": None,
            "created_at": now,
        }

    def auth_consume_email_verification_token(self, *, token_hash: str) -> dict[str, Any] | None:
        now = time.time()
        with self._conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT id, user_id, token_hash, expires_at, used_at, created_at "
                    "FROM email_verification_tokens WHERE token_hash = %s",
                    (token_hash,),
                )
                row = cur.fetchone()
                if not row:
                    return None

                expires_at = float(row[3])
                used_at = row[4]
                if used_at is not None or expires_at <= now:
                    return None

                cur.execute(
                    "UPDATE email_verification_tokens SET used_at = %s WHERE id = %s",
                    (now, row[0]),
                )

        return {
            "id": str(row[0]),
            "user_id": str(row[1]),
            "token_hash": str(row[2]),
            "expires_at": float(row[3]),
            "used_at": now,
            "created_at": float(row[5]),
        }

    def auth_create_password_reset_token(
        self,
        *,
        user_id: str,
        token_hash: str,
        expires_at: float,
    ) -> dict[str, Any]:
        token_id = str(uuid.uuid4())
        now = time.time()
        with self._conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "INSERT INTO password_reset_tokens(id, user_id, token_hash, expires_at, used_at, created_at) "
                    "VALUES (%s, %s, %s, %s, %s, %s)",
                    (token_id, user_id, token_hash, expires_at, None, now),
                )
        return {
            "id": token_id,
            "user_id": user_id,
            "token_hash": token_hash,
            "expires_at": expires_at,
            "used_at": None,
            "created_at": now,
        }

    def auth_consume_password_reset_token(self, *, token_hash: str) -> dict[str, Any] | None:
        now = time.time()
        with self._conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT id, user_id, token_hash, expires_at, used_at, created_at "
                    "FROM password_reset_tokens WHERE token_hash = %s",
                    (token_hash,),
                )
                row = cur.fetchone()
                if not row:
                    return None

                expires_at = float(row[3])
                used_at = row[4]
                if used_at is not None or expires_at <= now:
                    return None

                cur.execute(
                    "UPDATE password_reset_tokens SET used_at = %s WHERE id = %s",
                    (now, row[0]),
                )

        return {
            "id": str(row[0]),
            "user_id": str(row[1]),
            "token_hash": str(row[2]),
            "expires_at": float(row[3]),
            "used_at": now,
            "created_at": float(row[5]),
        }
