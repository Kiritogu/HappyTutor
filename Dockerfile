# ==============================================================
# DeepTutor Backend - Python FastAPI
# ==============================================================
FROM python:3.10-slim

WORKDIR /app

# Switch apt sources to Alibaba Cloud mirror (China acceleration)
# Replace both main repo and security repo in all possible source file formats
RUN { sed -i 's|deb.debian.org|mirrors.aliyun.com|g; s|security.debian.org/debian-security|mirrors.aliyun.com/debian-security|g' /etc/apt/sources.list 2>/dev/null || true; } && \
    { sed -i 's|deb.debian.org|mirrors.aliyun.com|g; s|security.debian.org/debian-security|mirrors.aliyun.com/debian-security|g' /etc/apt/sources.list.d/debian.sources 2>/dev/null || true; }

# Install system dependencies needed by some Python packages
RUN apt-get update && \
    apt-get install -y --no-install-recommends build-essential && \
    rm -rf /var/lib/apt/lists/*

# Install Python dependencies with Alibaba Cloud PyPI mirror
COPY requirements.txt .
RUN pip install --no-cache-dir -i https://mirrors.aliyun.com/pypi/simple/ --trusted-host mirrors.aliyun.com -r requirements.txt

# Copy application code
COPY src/ src/
COPY config/ config/

# Create data directories (will be overridden by volume mount)
RUN mkdir -p data/user data/knowledge_bases data/db data/user/logs

EXPOSE 8001

CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8001"]
