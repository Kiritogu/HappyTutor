from enum import Enum
from pathlib import Path


class ActivityType(str, Enum):
    SOLVE = "solve"
    QUESTION = "question"
    RESEARCH = "research"
    CHAT = "chat"


class HistoryManager:
    def __init__(self, base_dir: str | None = None):
        """
        History record manager

        Args:
            base_dir: History record directory. Default fixed to "project root/user",
                      at the same level as user/question, user/solve, user/research,
                      does not depend on current working directory, avoids path misalignment
                      when uvicorn / IDE start differently.
        """
        # Always resolve project root for DB backend
        project_root = Path(__file__).resolve().parents[3]

        if base_dir is None:
            # Current file: DeepTutor/src/api/utils/history.py
            # Project root should be three levels up: DeepTutor/
            base_dir_path = project_root / "data" / "user"
        else:
            base_dir_path = Path(base_dir)

        self.base_dir = base_dir_path
        self.base_dir.mkdir(parents=True, exist_ok=True)

        from src.services.storage import get_user_db

        self._db = get_user_db(project_root=project_root)

    def add_entry(
        self,
        *,
        user_id: str,
        activity_type: ActivityType,
        title: str,
        content: dict,
        summary: str = "",
    ):
        """
        Add a new history entry.

        Args:
            activity_type: The type of activity (solve, question, research)
            title: A short title (e.g. the question asked, or topic)
            content: The full result/payload
            summary: A short summary if applicable
        """
        return self._db.history_add_entry(
            user_id=user_id,
            activity_type=activity_type,
            title=title,
            content=content,
            summary=summary,
            limit=100,
        )

    def get_recent(self, *, user_id: str, limit: int = 10, type_filter: str | None = None) -> list[dict]:
        return self._db.history_get_recent(user_id=user_id, limit=limit, type_filter=type_filter)

    def get_entry(self, *, user_id: str, entry_id: str) -> dict | None:
        return self._db.history_get_entry(user_id=user_id, entry_id=entry_id)


# Global instance
history_manager = HistoryManager()
