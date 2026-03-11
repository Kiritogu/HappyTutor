import shutil
from pathlib import Path

from src.knowledge.manager import KnowledgeBaseManager


def test_rag_initialized_uses_status_not_local_directory():
    base_dir = Path("data") / "user" / "tmp_test_kb_manager"
    if base_dir.exists():
        shutil.rmtree(base_dir)
    base_dir.mkdir(parents=True, exist_ok=True)

    manager = KnowledgeBaseManager(base_dir=str(base_dir))
    kb_dir = base_dir / "kb1"
    kb_dir.mkdir(parents=True, exist_ok=True)
    (kb_dir / "metadata.json").write_text("{}", encoding="utf-8")
    manager.register_knowledge_base("kb1", "test kb")
    manager.update_kb_status("kb1", "ready")

    info = manager.get_info("kb1")
    assert info["status"] == "ready"
    assert info["statistics"]["rag_initialized"] is True

    shutil.rmtree(base_dir, ignore_errors=True)
