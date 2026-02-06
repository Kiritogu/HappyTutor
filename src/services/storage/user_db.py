from __future__ import annotations

"""
Database-backed storage for structured user data (SQLite / PostgreSQL).

This module provides a lightweight replacement for several JSON files under `data/user/`,
including:
- Activity history (`user_history.json`)
- Notebooks (`data/user/notebook/*.json`)
- Chat sessions (`chat_sessions.json`)
- UI settings (`data/user/settings/interface.json`)

Large artifacts (markdown reports, images, etc.) remain file-based and are served from
`data/user/` via `/api/outputs`.
"""

from dataclasses import dataclass
import json
import os
from pathlib import Path
import sqlite3
import threading
import time
from typing import Any
import uuid

from dotenv import load_dotenv

from src.services.config import load_config_with_main

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
load_dotenv(PROJECT_ROOT / "DeepTutor.env", override=False)
load_dotenv(PROJECT_ROOT / ".env", override=False)


def _parse_bool(value: str | None, default: bool) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _normalize_backend(value: str) -> str:
    backend = (value or "").strip().lower()
    if backend in {"sqlite", "sqlite3", "db", "database"}:
        return "sqlite"
    if backend in {"postgres", "postgresql", "pg"}:
        return "postgres"
    if backend in {"file", "files", "json", "local"}:
        return "file"
    return "file"


@dataclass(frozen=True)
class StorageSettings:
    backend: str
    sqlite_path: Path
    postgres_dsn: str
    auto_migrate: bool


def get_storage_settings(project_root: Path | None = None) -> StorageSettings:
    """
    Get storage settings for user structured data.

    Priority:
    1) Environment variables
    2) config/main.yaml ("storage" section)
    3) Built-in defaults
    """
    if project_root is None:
        project_root = PROJECT_ROOT

    cfg: dict[str, Any] = {}
    try:
        cfg = load_config_with_main("solve_config.yaml", project_root)
    except Exception:
        cfg = {}

    storage_cfg = cfg.get("storage", {}) if isinstance(cfg, dict) else {}

    default_backend = _normalize_backend(str(storage_cfg.get("backend", "file")))
    default_sqlite_path = str(storage_cfg.get("sqlite_path", "./data/db/deeptutor.sqlite"))
    default_postgres_dsn = str(storage_cfg.get("postgres_dsn", "") or "")
    default_auto_migrate = bool(storage_cfg.get("auto_migrate", True))

    backend = _normalize_backend(os.getenv("DEEPTUTOR_STORAGE_BACKEND") or default_backend)
    sqlite_path_str = os.getenv("DEEPTUTOR_SQLITE_PATH") or default_sqlite_path
    postgres_dsn = os.getenv("DEEPTUTOR_POSTGRES_DSN") or default_postgres_dsn
    auto_migrate = _parse_bool(os.getenv("DEEPTUTOR_STORAGE_AUTO_MIGRATE"), default_auto_migrate)

    sqlite_path = Path(sqlite_path_str)
    if not sqlite_path.is_absolute():
        sqlite_path = (project_root / sqlite_path).resolve()

    return StorageSettings(
        backend=backend,
        sqlite_path=sqlite_path,
        postgres_dsn=postgres_dsn,
        auto_migrate=auto_migrate,
    )


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


class UserDB:
    def __init__(
        self,
        db_path: Path,
        *,
        auto_migrate: bool = True,
        project_root: Path | None = None,
    ) -> None:
        self.db_path = db_path
        self._lock = threading.RLock()
        self._project_root = project_root or PROJECT_ROOT

        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row

        self._apply_pragmas()
        self._init_schema()

        if auto_migrate:
            self._migrate_from_files()

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    def _apply_pragmas(self) -> None:
        with self._lock:
            try:
                self._conn.execute("PRAGMA journal_mode=WAL;")
                self._conn.execute("PRAGMA synchronous=NORMAL;")
            except Exception:
                # Non-fatal (e.g., some networked filesystems)
                pass
            self._conn.execute("PRAGMA foreign_keys=ON;")

    def _init_schema(self) -> None:
        schema = """
        CREATE TABLE IF NOT EXISTS storage_meta (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS notebooks (
            id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            description TEXT NOT NULL DEFAULT '',
            created_at REAL NOT NULL,
            updated_at REAL NOT NULL,
            color TEXT NOT NULL DEFAULT '#3B82F6',
            icon TEXT NOT NULL DEFAULT 'book'
        );

        CREATE TABLE IF NOT EXISTS notebook_records (
            notebook_id TEXT NOT NULL,
            record_id TEXT NOT NULL,
            type TEXT NOT NULL,
            title TEXT NOT NULL,
            user_query TEXT NOT NULL,
            output TEXT NOT NULL,
            metadata TEXT NOT NULL DEFAULT '{}',
            created_at REAL NOT NULL,
            kb_name TEXT,
            PRIMARY KEY (notebook_id, record_id),
            FOREIGN KEY (notebook_id) REFERENCES notebooks(id) ON DELETE CASCADE
        );

        CREATE INDEX IF NOT EXISTS idx_notebook_records_notebook_id
            ON notebook_records(notebook_id);
        CREATE INDEX IF NOT EXISTS idx_notebook_records_type
            ON notebook_records(type);

        CREATE TABLE IF NOT EXISTS history_entries (
            id TEXT PRIMARY KEY,
            timestamp REAL NOT NULL,
            type TEXT NOT NULL,
            title TEXT NOT NULL,
            summary TEXT NOT NULL DEFAULT '',
            content TEXT NOT NULL DEFAULT '{}'
        );

        CREATE INDEX IF NOT EXISTS idx_history_timestamp
            ON history_entries(timestamp);
        CREATE INDEX IF NOT EXISTS idx_history_type_timestamp
            ON history_entries(type, timestamp);

        CREATE TABLE IF NOT EXISTS chat_sessions (
            session_id TEXT PRIMARY KEY,
            title TEXT NOT NULL,
            settings TEXT NOT NULL DEFAULT '{}',
            created_at REAL NOT NULL,
            updated_at REAL NOT NULL
        );

        CREATE TABLE IF NOT EXISTS chat_messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL,
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            sources TEXT,
            timestamp REAL NOT NULL,
            FOREIGN KEY (session_id) REFERENCES chat_sessions(session_id) ON DELETE CASCADE
        );

        CREATE INDEX IF NOT EXISTS idx_chat_messages_session_ts
            ON chat_messages(session_id, timestamp);

        CREATE TABLE IF NOT EXISTS ui_settings (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL,
            updated_at REAL NOT NULL
        );
        """
        with self._lock:
            self._conn.executescript(schema)
            self._conn.commit()

    # -------------------------------------------------------------------------
    # Meta
    # -------------------------------------------------------------------------

    def _get_meta(self, key: str) -> str | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT value FROM storage_meta WHERE key = ?",
                (key,),
            ).fetchone()
            return str(row["value"]) if row else None

    def _set_meta(self, key: str, value: str) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT INTO storage_meta(key, value) VALUES (?, ?) "
                "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                (key, value),
            )
            self._conn.commit()

    # -------------------------------------------------------------------------
    # Migration (JSON files -> SQLite)
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

        for notebook_file in notebook_dir.glob("*.json"):
            if notebook_file.name == "notebooks_index.json":
                continue
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

            with self._lock:
                self._conn.execute(
                    "INSERT INTO notebooks(id, name, description, created_at, updated_at, color, icon) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?) "
                    "ON CONFLICT(id) DO UPDATE SET "
                    "name=excluded.name, description=excluded.description, updated_at=excluded.updated_at, "
                    "color=excluded.color, icon=excluded.icon",
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
                    metadata = record.get("metadata") if isinstance(record.get("metadata"), dict) else {}
                    created_at_r = float(record.get("created_at") or updated_at)
                    kb_name = record.get("kb_name")

                    self._conn.execute(
                        "INSERT OR REPLACE INTO notebook_records("
                        "notebook_id, record_id, type, title, user_query, output, metadata, created_at, kb_name"
                        ") VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
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

                self._conn.commit()

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

        with self._lock:
            for entry in sessions:
                entry_id = str(entry.get("id") or "").strip()
                if not entry_id:
                    continue
                ts = float(entry.get("timestamp") or time.time())
                entry_type = str(entry.get("type") or "")
                title = str(entry.get("title") or "")
                summary = str(entry.get("summary") or "")
                content = entry.get("content") if isinstance(entry.get("content"), dict) else {}

                self._conn.execute(
                    "INSERT OR REPLACE INTO history_entries(id, timestamp, type, title, summary, content) "
                    "VALUES (?, ?, ?, ?, ?, ?)",
                    (entry_id, ts, entry_type, title, summary, _safe_json_dumps(content)),
                )

            self._conn.commit()

    def _migrate_chat_from_file(self, chat_file: Path) -> None:
        data = _load_json_file(chat_file)
        if not isinstance(data, dict):
            return
        sessions = data.get("sessions", [])
        if not isinstance(sessions, list):
            return

        with self._lock:
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

                self._conn.execute(
                    "INSERT INTO chat_sessions(session_id, title, settings, created_at, updated_at) "
                    "VALUES (?, ?, ?, ?, ?) "
                    "ON CONFLICT(session_id) DO UPDATE SET "
                    "title=excluded.title, settings=excluded.settings, updated_at=excluded.updated_at",
                    (session_id, title, _safe_json_dumps(settings), created_at, updated_at),
                )

                existing = self._conn.execute(
                    "SELECT 1 FROM chat_messages WHERE session_id = ? LIMIT 1",
                    (session_id,),
                ).fetchone()
                if existing:
                    continue

                for msg in session.get("messages", []) or []:
                    if not isinstance(msg, dict):
                        continue
                    role = str(msg.get("role") or "")
                    content = str(msg.get("content") or "")
                    ts = float(msg.get("timestamp") or updated_at)
                    sources = msg.get("sources")
                    sources_json = _safe_json_dumps(sources) if isinstance(sources, dict) else None

                    self._conn.execute(
                        "INSERT INTO chat_messages(session_id, role, content, sources, timestamp) "
                        "VALUES (?, ?, ?, ?, ?)",
                        (session_id, role, content, sources_json, ts),
                    )

            self._conn.commit()

    def _migrate_ui_settings_from_file(self, interface_file: Path) -> None:
        settings = _load_json_file(interface_file)
        if not isinstance(settings, dict):
            return
        now = time.time()
        with self._lock:
            self._conn.execute(
                "INSERT INTO ui_settings(key, value, updated_at) VALUES (?, ?, ?) "
                "ON CONFLICT(key) DO UPDATE SET value=excluded.value, updated_at=excluded.updated_at",
                ("interface", _safe_json_dumps(settings), now),
            )
            self._conn.commit()

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
                with self._lock:
                    self._conn.execute(
                        "INSERT INTO notebooks(id, name, description, created_at, updated_at, color, icon) "
                        "VALUES (?, ?, ?, ?, ?, ?, ?)",
                        (notebook_id, name, description, now, now, color, icon),
                    )
                    self._conn.commit()
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
            except sqlite3.IntegrityError:
                continue

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
            COUNT(r.record_id) AS record_count
        FROM notebooks n
        LEFT JOIN notebook_records r ON r.notebook_id = n.id
        GROUP BY n.id
        ORDER BY n.updated_at DESC
        """
        params: tuple[Any, ...] = ()
        if isinstance(limit, int) and limit > 0:
            sql += " LIMIT ?"
            params = (limit,)

        with self._lock:
            rows = self._conn.execute(sql, params).fetchall()
            return [
                {
                    "id": r["id"],
                    "name": r["name"],
                    "description": r["description"],
                    "created_at": r["created_at"],
                    "updated_at": r["updated_at"],
                    "record_count": int(r["record_count"] or 0),
                    "color": r["color"],
                    "icon": r["icon"],
                }
                for r in rows
            ]

    def notebook_get(self, notebook_id: str) -> dict[str, Any] | None:
        with self._lock:
            nb = self._conn.execute("SELECT * FROM notebooks WHERE id = ?", (notebook_id,)).fetchone()
            if not nb:
                return None

            record_rows = self._conn.execute(
                "SELECT record_id, type, title, user_query, output, metadata, created_at, kb_name "
                "FROM notebook_records WHERE notebook_id = ? "
                "ORDER BY created_at ASC",
                (notebook_id,),
            ).fetchall()

        records: list[dict[str, Any]] = []
        for r in record_rows:
            metadata = {}
            try:
                metadata = json.loads(r["metadata"]) if r["metadata"] else {}
            except Exception:
                metadata = {}
            records.append(
                {
                    "id": r["record_id"],
                    "type": r["type"],
                    "title": r["title"],
                    "user_query": r["user_query"],
                    "output": r["output"],
                    "metadata": metadata,
                    "created_at": r["created_at"],
                    "kb_name": r["kb_name"],
                }
            )

        return {
            "id": nb["id"],
            "name": nb["name"],
            "description": nb["description"],
            "created_at": nb["created_at"],
            "updated_at": nb["updated_at"],
            "records": records,
            "color": nb["color"],
            "icon": nb["icon"],
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
            fields.append("name = ?")
            values.append(name)
        if description is not None:
            fields.append("description = ?")
            values.append(description)
        if color is not None:
            fields.append("color = ?")
            values.append(color)
        if icon is not None:
            fields.append("icon = ?")
            values.append(icon)

        fields.append("updated_at = ?")
        values.append(now)
        values.append(notebook_id)

        with self._lock:
            cur = self._conn.execute(
                f"UPDATE notebooks SET {', '.join(fields)} WHERE id = ?",
                tuple(values),
            )
            self._conn.commit()
            if cur.rowcount == 0:
                return None

        return self.notebook_get(notebook_id)

    def notebook_delete(self, notebook_id: str) -> bool:
        with self._lock:
            cur = self._conn.execute("DELETE FROM notebooks WHERE id = ?", (notebook_id,))
            self._conn.commit()
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
        with self._lock:
            for notebook_id in notebook_ids:
                exists = self._conn.execute(
                    "SELECT 1 FROM notebooks WHERE id = ?",
                    (notebook_id,),
                ).fetchone()
                if not exists:
                    continue

                self._conn.execute(
                    "INSERT OR REPLACE INTO notebook_records("
                    "notebook_id, record_id, type, title, user_query, output, metadata, created_at, kb_name"
                    ") VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
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
                self._conn.execute(
                    "UPDATE notebooks SET updated_at = ? WHERE id = ?",
                    (now, notebook_id),
                )
                added_to.append(notebook_id)

            self._conn.commit()

        return {"record": record, "added_to_notebooks": added_to}

    def notebook_remove_record(self, *, notebook_id: str, record_id: str) -> bool:
        now = time.time()
        with self._lock:
            cur = self._conn.execute(
                "DELETE FROM notebook_records WHERE notebook_id = ? AND record_id = ?",
                (notebook_id, record_id),
            )
            if cur.rowcount == 0:
                self._conn.commit()
                return False

            self._conn.execute(
                "UPDATE notebooks SET updated_at = ? WHERE id = ?",
                (now, notebook_id),
            )
            self._conn.commit()
            return True

    def notebook_statistics(self) -> dict[str, Any]:
        with self._lock:
            total_notebooks = int(
                self._conn.execute("SELECT COUNT(1) AS c FROM notebooks").fetchone()["c"]
            )
            total_records = int(
                self._conn.execute("SELECT COUNT(1) AS c FROM notebook_records").fetchone()["c"]
            )
            type_rows = self._conn.execute(
                "SELECT type, COUNT(1) AS c FROM notebook_records GROUP BY type"
            ).fetchall()
            type_counts: dict[str, int] = {str(r["type"]): int(r["c"]) for r in type_rows}

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

        with self._lock:
            try:
                self._conn.execute(
                    "INSERT INTO history_entries(id, timestamp, type, title, summary, content) "
                    "VALUES (?, ?, ?, ?, ?, ?)",
                    (entry_id, ts, str(activity_type), title, summary, _safe_json_dumps(content)),
                )
            except sqlite3.IntegrityError:
                entry_id = f"{base_id}_{uuid.uuid4().hex[:4]}"
                entry["id"] = entry_id
                self._conn.execute(
                    "INSERT INTO history_entries(id, timestamp, type, title, summary, content) "
                    "VALUES (?, ?, ?, ?, ?, ?)",
                    (entry_id, ts, str(activity_type), title, summary, _safe_json_dumps(content)),
                )

            count = int(
                self._conn.execute("SELECT COUNT(1) AS c FROM history_entries").fetchone()["c"]
            )
            extra = count - int(limit)
            if extra > 0:
                self._conn.execute(
                    "DELETE FROM history_entries WHERE id IN ("
                    "  SELECT id FROM history_entries ORDER BY timestamp ASC LIMIT ?"
                    ")",
                    (extra,),
                )

            self._conn.commit()

        return entry

    def history_get_recent(self, *, limit: int = 10, type_filter: str | None = None) -> list[dict]:
        with self._lock:
            if type_filter:
                rows = self._conn.execute(
                    "SELECT * FROM history_entries WHERE type = ? "
                    "ORDER BY timestamp DESC LIMIT ?",
                    (type_filter, limit),
                ).fetchall()
            else:
                rows = self._conn.execute(
                    "SELECT * FROM history_entries ORDER BY timestamp DESC LIMIT ?",
                    (limit,),
                ).fetchall()

        results: list[dict[str, Any]] = []
        for r in rows:
            content = {}
            try:
                content = json.loads(r["content"]) if r["content"] else {}
            except Exception:
                content = {}
            results.append(
                {
                    "id": r["id"],
                    "timestamp": r["timestamp"],
                    "type": r["type"],
                    "title": r["title"],
                    "summary": r["summary"],
                    "content": content,
                }
            )
        return results

    def history_get_entry(self, entry_id: str) -> dict[str, Any] | None:
        with self._lock:
            r = self._conn.execute(
                "SELECT * FROM history_entries WHERE id = ?",
                (entry_id,),
            ).fetchone()
        if not r:
            return None
        content = {}
        try:
            content = json.loads(r["content"]) if r["content"] else {}
        except Exception:
            content = {}
        return {
            "id": r["id"],
            "timestamp": r["timestamp"],
            "type": r["type"],
            "title": r["title"],
            "summary": r["summary"],
            "content": content,
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

        with self._lock:
            self._conn.execute(
                "INSERT INTO chat_sessions(session_id, title, settings, created_at, updated_at) "
                "VALUES (?, ?, ?, ?, ?)",
                (session_id, session["title"], _safe_json_dumps(session["settings"]), now, now),
            )
            self._conn.commit()

        return session

    def chat_get_session(self, session_id: str) -> dict[str, Any] | None:
        with self._lock:
            s = self._conn.execute(
                "SELECT * FROM chat_sessions WHERE session_id = ?",
                (session_id,),
            ).fetchone()
            if not s:
                return None

            msg_rows = self._conn.execute(
                "SELECT role, content, sources, timestamp FROM chat_messages "
                "WHERE session_id = ? ORDER BY timestamp ASC, id ASC",
                (session_id,),
            ).fetchall()

        settings: dict[str, Any] = {}
        try:
            settings = json.loads(s["settings"]) if s["settings"] else {}
        except Exception:
            settings = {}

        messages: list[dict[str, Any]] = []
        for m in msg_rows:
            msg: dict[str, Any] = {
                "role": m["role"],
                "content": m["content"],
                "timestamp": m["timestamp"],
            }
            if m["sources"]:
                try:
                    msg["sources"] = json.loads(m["sources"])
                except Exception:
                    pass
            messages.append(msg)

        return {
            "session_id": s["session_id"],
            "title": s["title"],
            "messages": messages,
            "settings": settings,
            "created_at": s["created_at"],
            "updated_at": s["updated_at"],
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
        with self._lock:
            exists = self._conn.execute(
                "SELECT 1 FROM chat_sessions WHERE session_id = ?",
                (session_id,),
            ).fetchone()
            if not exists:
                return None

            if title is not None:
                self._conn.execute(
                    "UPDATE chat_sessions SET title = ?, updated_at = ? WHERE session_id = ?",
                    (title[:100], now, session_id),
                )
            if settings is not None:
                self._conn.execute(
                    "UPDATE chat_sessions SET settings = ?, updated_at = ? WHERE session_id = ?",
                    (_safe_json_dumps(settings), now, session_id),
                )
            if title is None and settings is None:
                self._conn.execute(
                    "UPDATE chat_sessions SET updated_at = ? WHERE session_id = ?",
                    (now, session_id),
                )

            if messages is not None:
                self._conn.execute("DELETE FROM chat_messages WHERE session_id = ?", (session_id,))
                for msg in messages:
                    if not isinstance(msg, dict):
                        continue
                    role = str(msg.get("role") or "")
                    content = str(msg.get("content") or "")
                    ts = float(msg.get("timestamp") or time.time())
                    sources = msg.get("sources")
                    sources_json = _safe_json_dumps(sources) if isinstance(sources, dict) else None
                    self._conn.execute(
                        "INSERT INTO chat_messages(session_id, role, content, sources, timestamp) "
                        "VALUES (?, ?, ?, ?, ?)",
                        (session_id, role, content, sources_json, ts),
                    )

            self._conn.commit()

        return self.chat_get_session(session_id)

    def chat_add_message(
        self,
        *,
        session_id: str,
        role: str,
        content: str,
        sources: dict[str, Any] | None = None,
    ) -> dict[str, Any] | None:
        with self._lock:
            s = self._conn.execute(
                "SELECT title FROM chat_sessions WHERE session_id = ?",
                (session_id,),
            ).fetchone()
            if not s:
                return None

            ts = time.time()
            sources_json = _safe_json_dumps(sources) if isinstance(sources, dict) else None
            self._conn.execute(
                "INSERT INTO chat_messages(session_id, role, content, sources, timestamp) "
                "VALUES (?, ?, ?, ?, ?)",
                (session_id, role, content, sources_json, ts),
            )

            new_title = None
            if s["title"] == "New Chat" and role == "user":
                new_title = content[:50] + ("..." if len(content) > 50 else "")

            if new_title is not None:
                self._conn.execute(
                    "UPDATE chat_sessions SET title = ?, updated_at = ? WHERE session_id = ?",
                    (new_title[:100], ts, session_id),
                )
            else:
                self._conn.execute(
                    "UPDATE chat_sessions SET updated_at = ? WHERE session_id = ?",
                    (ts, session_id),
                )

            self._conn.commit()

        return self.chat_get_session(session_id)

    def chat_list_sessions(self, *, limit: int = 20, include_messages: bool = False) -> list[dict]:
        with self._lock:
            session_rows = self._conn.execute(
                "SELECT session_id FROM chat_sessions ORDER BY updated_at DESC LIMIT ?",
                (limit,),
            ).fetchall()

        session_ids = [r["session_id"] for r in session_rows]
        if include_messages:
            return [self.chat_get_session(sid) for sid in session_ids if sid]

        sql = """
        SELECT
            s.session_id,
            s.title,
            s.settings,
            s.created_at,
            s.updated_at,
            (SELECT COUNT(1) FROM chat_messages m WHERE m.session_id = s.session_id) AS message_count,
            (SELECT substr(m.content, 1, 100)
               FROM chat_messages m
              WHERE m.session_id = s.session_id
              ORDER BY m.timestamp DESC, m.id DESC
              LIMIT 1
            ) AS last_message
        FROM chat_sessions s
        ORDER BY s.updated_at DESC
        LIMIT ?
        """
        with self._lock:
            rows = self._conn.execute(sql, (limit,)).fetchall()

        summaries: list[dict[str, Any]] = []
        for r in rows:
            settings: dict[str, Any] = {}
            try:
                settings = json.loads(r["settings"]) if r["settings"] else {}
            except Exception:
                settings = {}
            summaries.append(
                {
                    "session_id": r["session_id"],
                    "title": r["title"],
                    "message_count": int(r["message_count"] or 0),
                    "settings": settings,
                    "created_at": r["created_at"],
                    "updated_at": r["updated_at"],
                    "last_message": r["last_message"] or "",
                }
            )
        return summaries

    def chat_delete_session(self, session_id: str) -> bool:
        with self._lock:
            cur = self._conn.execute("DELETE FROM chat_sessions WHERE session_id = ?", (session_id,))
            self._conn.commit()
            return cur.rowcount > 0

    def chat_clear_all_sessions(self) -> int:
        with self._lock:
            count = int(
                self._conn.execute("SELECT COUNT(1) AS c FROM chat_sessions").fetchone()["c"]
            )
            self._conn.execute("DELETE FROM chat_sessions")
            self._conn.commit()
            return count

    # -------------------------------------------------------------------------
    # UI settings
    # -------------------------------------------------------------------------

    def ui_get(self, *, key: str = "interface") -> dict[str, Any] | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT value FROM ui_settings WHERE key = ?",
                (key,),
            ).fetchone()
        if not row:
            return None
        try:
            data = json.loads(row["value"]) if row["value"] else {}
            return data if isinstance(data, dict) else None
        except Exception:
            return None

    def ui_set(self, *, key: str = "interface", value: dict[str, Any]) -> None:
        now = time.time()
        with self._lock:
            self._conn.execute(
                "INSERT INTO ui_settings(key, value, updated_at) VALUES (?, ?, ?) "
                "ON CONFLICT(key) DO UPDATE SET value=excluded.value, updated_at=excluded.updated_at",
                (key, _safe_json_dumps(value), now),
            )
            self._conn.commit()


_user_db_lock = threading.Lock()
_user_db_instance: Any | None = None
_user_db_key: tuple[str, str] | None = None


def get_user_db(project_root: Path | None = None) -> Any | None:
    settings = get_storage_settings(project_root=project_root)
    if settings.backend == "file":
        return None

    if settings.backend == "sqlite":
        key = ("sqlite", str(settings.sqlite_path))
    elif settings.backend == "postgres":
        if not settings.postgres_dsn:
            raise ValueError(
                "DEEPTUTOR_POSTGRES_DSN is required when DEEPTUTOR_STORAGE_BACKEND=postgres "
                "(or storage.backend=postgres)."
            )
        key = ("postgres", settings.postgres_dsn)
    else:
        return None

    global _user_db_instance, _user_db_key
    with _user_db_lock:
        if _user_db_instance is None or _user_db_key != key:
            if _user_db_instance is not None:
                try:
                    _user_db_instance.close()
                except Exception:
                    pass

            if settings.backend == "sqlite":
                _user_db_instance = UserDB(
                    settings.sqlite_path,
                    auto_migrate=settings.auto_migrate,
                    project_root=project_root or PROJECT_ROOT,
                )
            else:
                from .postgres_db import PostgresUserDB

                _user_db_instance = PostgresUserDB(
                    settings.postgres_dsn,
                    auto_migrate=settings.auto_migrate,
                    project_root=project_root or PROJECT_ROOT,
                )

            _user_db_key = key

    return _user_db_instance
