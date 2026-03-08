"""
Notebook Manager - Manages user notebooks and records
All notebook data is stored via the database backend.
"""

from enum import Enum
from pathlib import Path

from pydantic import BaseModel


class RecordType(str, Enum):
    """Record type"""

    SOLVE = "solve"
    QUESTION = "question"
    RESEARCH = "research"
    CO_WRITER = "co_writer"


class NotebookRecord(BaseModel):
    """Single record in notebook"""

    id: str
    type: RecordType
    title: str
    user_query: str
    output: str
    metadata: dict = {}
    created_at: float
    kb_name: str | None = None


class Notebook(BaseModel):
    """Notebook model"""

    id: str
    name: str
    description: str = ""
    created_at: float
    updated_at: float
    records: list[NotebookRecord] = []
    color: str = "#3B82F6"  # Default blue
    icon: str = "book"  # Default icon


class NotebookManager:
    """Notebook manager"""

    def __init__(self, base_dir: str | None = None):
        """
        Initialize notebook manager

        Args:
            base_dir: Notebook storage directory, defaults to project root/user/notebook
        """
        project_root = Path(__file__).resolve().parents[3]

        if base_dir is None:
            # Current file: DeepTutor/src/api/utils/notebook_manager.py
            # Project root should be three levels up: DeepTutor/
            base_dir_path = project_root / "data" / "user" / "notebook"
        else:
            base_dir_path = Path(base_dir)

        self.base_dir = base_dir_path
        self.base_dir.mkdir(parents=True, exist_ok=True)

        from src.services.storage import get_user_db

        self._db = get_user_db(project_root=project_root)

    # === Notebook Operations ===

    def create_notebook(
        self,
        *,
        user_id: str,
        name: str,
        description: str = "",
        color: str = "#3B82F6",
        icon: str = "book",
    ) -> dict:
        """
        Create new notebook

        Args:
            name: Notebook name
            description: Notebook description
            color: Color code
            icon: Icon name

        Returns:
            Created notebook information
        """
        return self._db.notebook_create(
            user_id=user_id,
            name=name,
            description=description,
            color=color,
            icon=icon,
        )

    def list_notebooks(self, *, user_id: str) -> list[dict]:
        """
        List all notebooks (summary information)

        Returns:
            Notebook list
        """
        return self._db.notebook_list(user_id=user_id)

    def get_notebook(self, *, user_id: str, notebook_id: str) -> dict | None:
        """
        Get notebook details (includes all records)

        Args:
            notebook_id: Notebook ID

        Returns:
            Notebook details
        """
        return self._db.notebook_get(user_id=user_id, notebook_id=notebook_id)

    def update_notebook(
        self,
        *,
        user_id: str,
        notebook_id: str,
        name: str | None = None,
        description: str | None = None,
        color: str | None = None,
        icon: str | None = None,
    ) -> dict | None:
        """
        Update notebook information

        Args:
            notebook_id: Notebook ID
            name: New name
            description: New description
            color: New color
            icon: New icon

        Returns:
            Updated notebook information
        """
        return self._db.notebook_update(
            notebook_id,
            user_id=user_id,
            name=name,
            description=description,
            color=color,
            icon=icon,
        )

    def delete_notebook(self, *, user_id: str, notebook_id: str) -> bool:
        """
        Delete notebook

        Args:
            notebook_id: Notebook ID

        Returns:
            Whether deletion was successful
        """
        return self._db.notebook_delete(user_id=user_id, notebook_id=notebook_id)

    # === Record Operations ===

    def add_record(
        self,
        *,
        user_id: str,
        notebook_ids: list[str],
        record_type: RecordType,
        title: str,
        user_query: str,
        output: str,
        metadata: dict = None,
        kb_name: str = None,
    ) -> dict:
        """
        Add record to one or more notebooks

        Args:
            notebook_ids: Target notebook ID list
            record_type: Record type
            title: Title
            user_query: User input
            output: Output result
            metadata: Additional metadata
            kb_name: Knowledge base name

        Returns:
            Added record information
        """
        return self._db.notebook_add_record(
            user_id=user_id,
            notebook_ids=notebook_ids,
            record_type=record_type,
            title=title,
            user_query=user_query,
            output=output,
            metadata=metadata,
            kb_name=kb_name,
        )

    def remove_record(self, *, user_id: str, notebook_id: str, record_id: str) -> bool:
        """
        Remove record from notebook

        Args:
            notebook_id: Notebook ID
            record_id: Record ID

        Returns:
            Whether deletion was successful
        """
        return self._db.notebook_remove_record(
            user_id=user_id,
            notebook_id=notebook_id,
            record_id=record_id,
        )

    def get_statistics(self, *, user_id: str) -> dict:
        """
        Get notebook statistics

        Returns:
            Statistics information
        """
        return self._db.notebook_statistics(user_id=user_id)


# Global instance
notebook_manager = NotebookManager()
