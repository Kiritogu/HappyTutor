from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from src.api.routers import (
    agent_config,
    chat,
    config,
    dashboard,
    guide,
    knowledge,
    notebook,
    question,
    research,
    settings,
    system,
)
from src.logging import get_logger

# Note: Don't set service_prefix here - start_web.py already adds [Backend] prefix
logger = get_logger("API")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifecycle management
    Gracefully handle startup and shutdown events, avoid CancelledError
    """
    # Execute on startup
    logger.info("Application startup")

    # Initialize structured-data storage backend (SQLite / PostgreSQL) early.
    # If configured, fail fast on misconfiguration to avoid silently falling back to JSON files.
    try:
        from src.services.storage import get_storage_settings, get_user_db

        project_root = Path(__file__).parent.parent.parent
        storage_settings = get_storage_settings(project_root=project_root)
        if storage_settings.backend != "file":
            get_user_db(project_root=project_root)
            logger.info(f"Storage backend initialized: {storage_settings.backend}")
    except Exception as e:
        logger.error(f"Storage initialization failed: {e}")
        raise

    # Initialize LLM client early to set environment variables for LightRAG
    # LightRAG reads OPENAI_API_KEY from os.environ internally, so we must
    # set it before any RAG operations can happen
    try:
        from src.services.llm import get_llm_client

        llm_client = get_llm_client()
        logger.info(f"LLM client initialized: model={llm_client.config.model}")
    except Exception as e:
        logger.warning(f"Failed to initialize LLM client at startup: {e}")

    # Run warmup to reduce first-request latency
    try:
        from src.services.warmup import warmup_all

        # Run warmup in background to not block startup
        # Set skip_llm_call=False to fully warm up the connection pool
        await warmup_all(skip_llm_call=False, llm_timeout=30.0)
    except Exception as e:
        logger.warning(f"Warmup failed: {e}")

    yield
    # Execute on shutdown
    logger.info("Application shutdown")


app = FastAPI(
    title="DeepTutor API",
    version="1.0.0",
    lifespan=lifespan,
    # Disable automatic trailing slash redirects to prevent protocol downgrade issues
    # when deployed behind HTTPS reverse proxies (e.g., nginx).
    # Without this, FastAPI's 307 redirects may change HTTPS to HTTP.
    # See: https://github.com/HKUDS/DeepTutor/issues/112
    redirect_slashes=False,
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific frontend origin
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount user directory as static root for generated artifacts
# This allows frontend to access generated artifacts (images, PDFs, etc.)
# URL: /api/outputs/solve/solve_xxx/artifacts/image.png
# Physical Path: DeepTutor/data/user/solve/solve_xxx/artifacts/image.png
project_root = Path(__file__).parent.parent.parent
user_dir = project_root / "data" / "user"

# Initialize user directories on startup
try:
    from src.services.setup import init_user_directories

    init_user_directories(project_root)
except Exception:
    # Fallback: just create the main directory if it doesn't exist
    if not user_dir.exists():
        user_dir.mkdir(parents=True)

app.mount("/api/outputs", StaticFiles(directory=str(user_dir)), name="outputs")

# Include routers
app.include_router(chat.router, prefix="/api/v1", tags=["chat"])
app.include_router(question.router, prefix="/api/v1/question", tags=["question"])
app.include_router(research.router, prefix="/api/v1/research", tags=["research"])
app.include_router(knowledge.router, prefix="/api/v1/knowledge", tags=["knowledge"])
app.include_router(dashboard.router, prefix="/api/v1/dashboard", tags=["dashboard"])
app.include_router(notebook.router, prefix="/api/v1/notebook", tags=["notebook"])
app.include_router(guide.router, prefix="/api/v1/guide", tags=["guide"])
app.include_router(settings.router, prefix="/api/v1/settings", tags=["settings"])
app.include_router(system.router, prefix="/api/v1/system", tags=["system"])
app.include_router(config.router, prefix="/api/v1/config", tags=["config"])
app.include_router(agent_config.router, prefix="/api/v1/agent-config", tags=["agent-config"])


@app.get("/")
async def root():
    return {"message": "Welcome to DeepTutor API"}


if __name__ == "__main__":
    from pathlib import Path

    import uvicorn

    # Get project root directory
    project_root = Path(__file__).parent.parent.parent

    # Ensure project root is in Python path
    import sys

    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    # Get port from configuration
    from src.services.setup import get_backend_port

    backend_port = get_backend_port(project_root)

    # Configure reload_excludes with absolute paths to properly exclude directories
    venv_dir = project_root / "venv"
    data_dir = project_root / "data"
    reload_excludes = [
        str(d)
        for d in [
            venv_dir,
            project_root / ".venv",
            data_dir,
            project_root / "web" / "node_modules",
            project_root / "web" / ".next",
            project_root / ".git",
        ]
        if d.exists()
    ]

    uvicorn.run(
        "api.main:app",
        host="0.0.0.0",
        port=backend_port,
        reload=True,
        reload_excludes=reload_excludes,
    )
