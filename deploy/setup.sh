#!/bin/bash
# ==============================================================
# DeepTutor - Ubuntu Server One-Click Setup Script
# ==============================================================
# 使用方法：
#   1. 上传此脚本到服务器
#   2. chmod +x setup.sh
#   3. ./setup.sh
# ==============================================================

set -e

REPO_URL="https://github.com/YOUR_USER/HappyTutor.git"  # TODO: 替换为你的 Git 仓库地址
APP_DIR="/opt/deeptutor"
BRANCH="master"

echo "======================================================"
echo "  DeepTutor 云服务器部署脚本"
echo "======================================================"

# ---------- 1. 安装 Docker ----------
if ! command -v docker &> /dev/null; then
    echo "[1/5] 安装 Docker..."
    curl -fsSL https://get.docker.com | sh
    sudo usermod -aG docker $USER
    echo "Docker 安装完成，请重新登录服务器后再运行此脚本（newgrp docker 或重新登录）"
    echo "或者运行: sudo newgrp docker"
else
    echo "[1/5] Docker 已安装，跳过..."
fi

# ---------- 2. 安装 Docker Compose ----------
if ! command -v docker compose &> /dev/null; then
    echo "[2/5] 安装 Docker Compose..."
    sudo apt-get update -y
    sudo apt-get install -y docker-compose-plugin
else
    echo "[2/5] Docker Compose 已安装，跳过..."
fi

# ---------- 3. 克隆项目 ----------
echo "[3/5] 克隆项目到 ${APP_DIR}..."
if [ -d "$APP_DIR" ]; then
    echo "目录已存在，拉取最新代码..."
    cd "$APP_DIR"
    git pull origin "$BRANCH"
else
    sudo git clone --branch "$BRANCH" "$REPO_URL" "$APP_DIR"
    sudo chown -R $USER:$USER "$APP_DIR"
    cd "$APP_DIR"
fi

# ---------- 4. 配置 .env ----------
echo "[4/5] 配置环境变量..."
if [ ! -f "$APP_DIR/.env" ]; then
    echo ""
    echo ">>> 未找到 .env 文件，请手动创建："
    echo "    cp $APP_DIR/.env.example_CN $APP_DIR/.env"
    echo "    nano $APP_DIR/.env"
    echo ""
    echo "需要填写的关键配置："
    echo "  DEEPTUTOR_POSTGRES_DSN=postgresql://postgres:[密码]@db.[项目ID].supabase.co:5432/postgres?sslmode=require"
    echo "  NEXT_PUBLIC_API_BASE=http://$(curl -s ifconfig.me)"
    echo "  NEXT_PUBLIC_API_BASE_EXTERNAL=http://$(curl -s ifconfig.me)"
    echo "  LLM_API_KEY=你的API密钥"
    echo ""
    exit 1
else
    echo ".env 文件已存在..."
fi

# ---------- 5. 启动服务 ----------
echo "[5/5] 构建并启动 Docker 服务..."
cd "$APP_DIR"
docker compose down --remove-orphans 2>/dev/null || true
docker compose up -d --build

echo ""
echo "======================================================"
echo "  部署完成！"
echo "======================================================"
echo ""
echo "服务状态："
docker compose ps
echo ""
SERVER_IP=$(curl -s ifconfig.me 2>/dev/null || echo "你的服务器IP")
echo "访问地址: http://${SERVER_IP}"
echo "API文档: http://${SERVER_IP}/docs"
echo ""
echo "常用命令："
echo "  查看日志: docker compose logs -f"
echo "  查看后端日志: docker compose logs -f backend"
echo "  重启服务: docker compose restart"
echo "  停止服务: docker compose down"
echo "  更新部署: git pull && docker compose up -d --build"
