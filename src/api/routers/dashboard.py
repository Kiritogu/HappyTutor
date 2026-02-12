from fastapi import APIRouter, Depends, HTTPException

from src.api.dependencies.auth import get_current_user_from_header
from src.api.utils.history import history_manager

router = APIRouter()


@router.get("/recent")
async def get_recent_history(
    limit: int = 10,
    type: str | None = None,
    current_user: dict = Depends(get_current_user_from_header),
):
    return history_manager.get_recent(user_id=current_user["id"], limit=limit, type_filter=type)


@router.get("/{entry_id}")
async def get_history_entry(
    entry_id: str,
    current_user: dict = Depends(get_current_user_from_header),
):
    entry = history_manager.get_entry(user_id=current_user["id"], entry_id=entry_id)
    if not entry:
        raise HTTPException(status_code=404, detail="Entry not found")
    return entry
