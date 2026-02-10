from __future__ import annotations

import inspect

from src.agents.chat.session_manager import SessionManager
from src.api.utils.history import HistoryManager
from src.api.utils.notebook_manager import NotebookManager
from src.services.settings import interface_settings
from src.services.storage.postgres_db import PostgresUserDB


def _assert_has_parameter(callable_obj, parameter_name: str) -> None:
    signature = inspect.signature(callable_obj)
    assert parameter_name in signature.parameters, f"missing parameter: {parameter_name}"


def test_postgres_db_user_scoped_method_signatures():
    user_scoped_methods = [
        PostgresUserDB.notebook_create,
        PostgresUserDB.notebook_list,
        PostgresUserDB.notebook_get,
        PostgresUserDB.notebook_update,
        PostgresUserDB.notebook_delete,
        PostgresUserDB.notebook_add_record,
        PostgresUserDB.notebook_remove_record,
        PostgresUserDB.notebook_statistics,
        PostgresUserDB.history_add_entry,
        PostgresUserDB.history_get_recent,
        PostgresUserDB.history_get_entry,
        PostgresUserDB.chat_create_session,
        PostgresUserDB.chat_get_session,
        PostgresUserDB.chat_update_session,
        PostgresUserDB.chat_add_message,
        PostgresUserDB.chat_list_sessions,
        PostgresUserDB.chat_delete_session,
        PostgresUserDB.chat_clear_all_sessions,
        PostgresUserDB.ui_get,
        PostgresUserDB.ui_set,
    ]

    for method in user_scoped_methods:
        _assert_has_parameter(method, "user_id")


def test_manager_and_settings_user_scoped_signatures():
    user_scoped_targets = [
        NotebookManager.create_notebook,
        NotebookManager.list_notebooks,
        NotebookManager.get_notebook,
        NotebookManager.update_notebook,
        NotebookManager.delete_notebook,
        NotebookManager.add_record,
        NotebookManager.remove_record,
        NotebookManager.get_statistics,
        HistoryManager.add_entry,
        HistoryManager.get_recent,
        HistoryManager.get_entry,
        SessionManager.create_session,
        SessionManager.get_session,
        SessionManager.update_session,
        SessionManager.add_message,
        SessionManager.list_sessions,
        SessionManager.delete_session,
        SessionManager.clear_all_sessions,
        interface_settings.get_ui_settings,
        interface_settings.get_ui_language,
    ]

    for target in user_scoped_targets:
        _assert_has_parameter(target, "user_id")

