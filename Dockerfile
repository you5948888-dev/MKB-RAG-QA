# 基础镜像作为构建参数，必要时可通过 --build-arg 覆盖。
ARG PYTHON_IMAGE=python:3.12-slim-bookworm

# builder 阶段只负责创建虚拟环境并安装 Python 依赖。
FROM ${PYTHON_IMAGE} AS builder
# 关闭 pip 的版本检查，减少无关提示和联网检查。
ENV PIP_DISABLE_PIP_VERSION_CHECK=1
# 后续构建命令都在 /build 中执行。
WORKDIR /build
# 把依赖集中安装到可整体复制的虚拟环境。
RUN python -m venv /opt/venv
# 让后续 python 和 pip 默认来自 /opt/venv。
ENV PATH="/opt/venv/bin:$PATH"
# 先复制依赖清单，使源码变化时仍能复用依赖安装层。
COPY requirements.runtime.txt ./requirements.txt
# 不保留 pip 下载缓存，并兼容未启用 BuildKit 的旧构建器。
RUN python -m pip install --no-cache-dir -r requirements.txt

# runtime 阶段从干净的基础镜像开始，只保留运行所需内容。
FROM ${PYTHON_IMAGE} AS runtime
# 使用虚拟环境；不生成 pyc；日志立即输出；为非 root 用户提供可写 HOME。
ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HOME=/tmp
# 应用代码、配置和静态资源统一放在 /app。
WORKDIR /app
# 创建固定 UID/GID 的无登录用户，避免应用以 root 身份运行。
RUN groupadd --gid 10001 app \
    && useradd --uid 10001 --gid 10001 --no-create-home --shell /usr/sbin/nologin app
# 从 builder 复制已经安装好依赖的完整虚拟环境。
COPY --from=builder /opt/venv /opt/venv
# 复制应用模块和 Pipeline 配置，并设置为应用用户所有。
COPY --chown=10001:10001 *.py pipeline_config.json /app/
# 复制 FastAPI 路由。
COPY --chown=10001:10001 routes /app/routes
# 复制浏览器访问的静态页面。
COPY --chown=10001:10001 frontend /app/frontend
# 同时设置父目录和上传目录的所有权，确保 SQLite 能在 /app/data 创建数据库。
RUN install -d -o 10001 -g 10001 /app/data /app/data/uploads
# 从这里开始，默认命令以非 root 用户运行。
USER 10001:10001
# 声明容器内应用端口；宿主机端口映射由 Compose 完成。
EXPOSE 8013
# 默认启动 FastAPI；Worker 会在 compose.yaml 中覆盖这条命令。
CMD ["python", "-m", "uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8013"]
