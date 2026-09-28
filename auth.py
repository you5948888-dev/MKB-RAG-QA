"""
用户认证模块：密码工具、用户 CRUD、KB 归属、FastAPI 依赖、认证路由。

核心概念：
  - 认证（Authentication）：你是谁？（登录校验）
  - 授权（Authorization）：你能做什么？（权限检查）
  - 签名 Cookie：服务端签发带过期时间的 Cookie 标识用户身份
  - 依赖注入（Depends）：FastAPI 自动把当前用户注入到路由函数
"""
import hmac
import hashlib
import time
from datetime import datetime
from uuid import uuid4

import bcrypt as _bcrypt

from config import CHAT_DB_PATH, AUTH_SECRET_KEY


# ── 密码工具 ────────────────────────────────────────────────────


def hash_password(password: str) -> str:
    """用 bcrypt 加盐哈希密码。"""
    return _bcrypt.hashpw(password.encode("utf-8"), _bcrypt.gensalt()).decode("utf-8")


def verify_password(password: str, password_hash: str) -> bool:
    """验证密码是否匹配哈希值。"""
    return _bcrypt.checkpw(password.encode("utf-8"), password_hash.encode("utf-8"))


# ── 数据库操作 ──────────────────────────────────────────────────


def init_auth_tables() -> None:
    """创建 users + user_kbs 表。"""
    import sqlite3
    with sqlite3.connect(CHAT_DB_PATH) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS users (
                user_id       TEXT PRIMARY KEY,
                username      TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                display_name  TEXT,
                role          TEXT DEFAULT 'user',
                api_key       TEXT UNIQUE,
                created_at    TEXT NOT NULL
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS user_kbs (
                user_id    TEXT NOT NULL,
                kb_id      TEXT NOT NULL,
                permission TEXT DEFAULT 'owner',
                PRIMARY KEY (user_id, kb_id)
            )
        """)


def ensure_default_admin() -> None:
    """创建内置 admin 账户（幂等，已存在则跳过）。默认密码 123456。"""
    import sqlite3
    with sqlite3.connect(CHAT_DB_PATH) as conn:
        row = conn.execute("SELECT 1 FROM users WHERE username = 'admin'").fetchone()
        if row:
            return
        now = datetime.now().isoformat(timespec="seconds")
        api_key = f"ak_{uuid4().hex}"
        conn.execute(
            "INSERT INTO users (user_id, username, password_hash, display_name, role, api_key, created_at) VALUES (?, ?, ?, '管理员', 'admin', ?, ?)",
            (f"u_{uuid4().hex[:8]}", "admin", hash_password("123456"), api_key, now),
        )


def create_user(username: str, password: str, display_name: str | None = None) -> dict:
    """注册新用户。"""
    import sqlite3
    username = username.strip()
    if not username or len(username) > 32:
        raise ValueError("用户名长度需在 1-32 之间")
    if len(password) < 4:
        raise ValueError("密码长度不能少于4位")

    with sqlite3.connect(CHAT_DB_PATH) as conn:
        existing = conn.execute("SELECT 1 FROM users WHERE username = ?", (username,)).fetchone()
        if existing:
            raise ValueError(f"用户名 '{username}' 已存在")

        user_id = f"u_{uuid4().hex[:8]}"
        api_key = f"ak_{uuid4().hex}"
        now = datetime.now().isoformat(timespec="seconds")
        conn.execute(
            "INSERT INTO users (user_id, username, password_hash, display_name, role, api_key, created_at) VALUES (?, ?, ?, ?, 'user', ?, ?)",
            (user_id, username, hash_password(password), display_name or username, api_key, now),
        )

    return {
        "user_id": user_id,
        "username": username,
        "display_name": display_name or username,
        "role": "user",
        "api_key": api_key,
    }


def authenticate_user(username: str, password: str) -> dict | None:
    """验证登录凭证，成功返回用户信息，失败返回 None。"""
    import sqlite3
    with sqlite3.connect(CHAT_DB_PATH) as conn:
        row = conn.execute(
            "SELECT user_id, username, display_name, role, api_key, password_hash FROM users WHERE username = ?",
            (username,),
        ).fetchone()
    if not row:
        return None
    if not verify_password(password, row[5]):
        return None
    return {
        "user_id": row[0],
        "username": row[1],
        "display_name": row[2],
        "role": row[3],
        "api_key": row[4],
    }


def get_user_by_id(user_id: str) -> dict | None:
    """通过 user_id 查找用户。"""
    import sqlite3
    with sqlite3.connect(CHAT_DB_PATH) as conn:
        row = conn.execute(
            "SELECT user_id, username, display_name, role, api_key FROM users WHERE user_id = ?",
            (user_id,),
        ).fetchone()
    if not row:
        return None
    return {
        "user_id": row[0],
        "username": row[1],
        "display_name": row[2],
        "role": row[3],
        "api_key": row[4],
    }


def get_user_by_api_key(api_key: str) -> dict | None:
    """通过 API Key 查找用户。"""
    import sqlite3
    with sqlite3.connect(CHAT_DB_PATH) as conn:
        row = conn.execute(
            "SELECT user_id, username, display_name, role, api_key FROM users WHERE api_key = ?",
            (api_key,),
        ).fetchone()
    if not row:
        return None
    return {
        "user_id": row[0],
        "username": row[1],
        "display_name": row[2],
        "role": row[3],
        "api_key": row[4],
    }


def regenerate_api_key(user_id: str) -> str:
    """重新生成 API Key。"""
    import sqlite3
    new_key = f"ak_{uuid4().hex}"
    with sqlite3.connect(CHAT_DB_PATH) as conn:
        conn.execute("UPDATE users SET api_key = ? WHERE user_id = ?", (new_key, user_id))
    return new_key


def change_password(user_id: str, old_password: str, new_password: str) -> bool:
    """修改密码，成功返回 True。"""
    import sqlite3
    with sqlite3.connect(CHAT_DB_PATH) as conn:
        row = conn.execute("SELECT password_hash FROM users WHERE user_id = ?", (user_id,)).fetchone()
        if not row or not verify_password(old_password, row[0]):
            return False
        conn.execute(
            "UPDATE users SET password_hash = ? WHERE user_id = ?",
            (hash_password(new_password), user_id),
        )
    return True


# ── KB 归属操作 ─────────────────────────────────────────────────


def bind_kb_to_user(user_id: str, kb_id: str, permission: str = "owner") -> None:
    """绑定知识库到用户。"""
    import sqlite3
    with sqlite3.connect(CHAT_DB_PATH) as conn:
        conn.execute(
            "INSERT OR IGNORE INTO user_kbs (user_id, kb_id, permission) VALUES (?, ?, ?)",
            (user_id, kb_id, permission),
        )


def get_user_owned_kb_ids(user_id: str) -> list[str]:
    """获取用户拥有的知识库 ID 列表。"""
    import sqlite3
    with sqlite3.connect(CHAT_DB_PATH) as conn:
        rows = conn.execute("SELECT kb_id FROM user_kbs WHERE user_id = ?", (user_id,)).fetchall()
    return [row[0] for row in rows]


def check_kb_permission(user_id: str, kb_id: str) -> str | None:
    """检查用户对 KB 的权限，返回 permission 或 None。"""
    import sqlite3
    with sqlite3.connect(CHAT_DB_PATH) as conn:
        row = conn.execute(
            "SELECT permission FROM user_kbs WHERE user_id = ? AND kb_id = ?",
            (user_id, kb_id),
        ).fetchone()
    return row[0] if row else None


def remove_kb_bindings(kb_id: str) -> None:
    """删除 KB 的所有用户绑定（KB 删除时调用）。"""
    import sqlite3
    with sqlite3.connect(CHAT_DB_PATH) as conn:
        conn.execute("DELETE FROM user_kbs WHERE kb_id = ?", (kb_id,))


# ── FastAPI 认证依赖 ───────────────────────────────────────────

from fastapi import APIRouter, HTTPException, Request, Response, Depends
from models import UserRegisterRequest, UserLoginRequest, ChangePasswordRequest

AUTH_COOKIE_NAME = "rag_session"
COOKIE_MAX_AGE = 86400 * 7  # 7 天
# Ch11 改造：签名密钥统一从 config 读取（config 从 .env 加载，禁止占位符/内置默认值）
_SECRET_KEY = AUTH_SECRET_KEY


def _sign_session_payload(payload: str) -> str:
    """用 HMAC-SHA256 对 Cookie 载荷签名，防止内容被伪造。"""
    return hmac.new(_SECRET_KEY.encode(), payload.encode(), hashlib.sha256).hexdigest()


def _create_session_cookie(user_id: str) -> str:
    """创建签名 Cookie：user_id.过期时间.签名。"""
    expires_at = int(time.time()) + COOKIE_MAX_AGE
    payload = f"{user_id}.{expires_at}"
    return f"{payload}.{_sign_session_payload(payload)}"


def _verify_session_cookie(cookie_value: str) -> str | None:
    """验证 Cookie 的格式、签名和服务端过期时间。"""
    if not cookie_value or cookie_value.count(".") != 2:
        return None
    user_id, expires_text, signature = cookie_value.split(".", 2)
    try:
        expires_at = int(expires_text)
    except ValueError:
        return None
    payload = f"{user_id}.{expires_at}"
    expected = _sign_session_payload(payload)
    if not hmac.compare_digest(expected, signature):
        return None
    if expires_at <= int(time.time()):
        return None
    return user_id


async def get_current_user(request: Request) -> dict:
    """从 Cookie 或 X-API-Key Header 获取当前用户。"""
    # 优先从 Cookie 获取
    user = None
    cookie_val = request.cookies.get(AUTH_COOKIE_NAME)
    if cookie_val:
        user_id = _verify_session_cookie(cookie_val)
        if user_id:
            user = get_user_by_id(user_id)
    # 其次从 API Key Header 获取
    if not user:
        api_key = request.headers.get("X-API-Key")
        if api_key:
            user = get_user_by_api_key(api_key)
    if not user:
        raise HTTPException(status_code=401, detail={"code": 40101, "message": "未登录或认证失败"})
    return user


def require_role(role: str):
    """角色权限检查工厂函数。用法：Depends(require_role("admin"))。"""
    async def _check(user: dict = Depends(get_current_user)):
        if user["role"] != role and user["role"] != "admin":
            raise HTTPException(status_code=403, detail={"code": 40302, "message": "权限不足"})
        return user
    return _check


# ── 认证路由 ────────────────────────────────────────────────────

router = APIRouter(prefix="/api/auth", tags=["auth"])


@router.post("/register")
def register(payload: UserRegisterRequest, response: Response):
    """注册新用户，成功后自动登录（设置 Cookie）。"""
    try:
        user = create_user(payload.username, payload.password, payload.display_name)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    cookie_val = _create_session_cookie(user["user_id"])
    response.set_cookie(AUTH_COOKIE_NAME, cookie_val, max_age=COOKIE_MAX_AGE, httponly=True, samesite="lax")
    return {"code": 0, "message": "ok", "data": user}


@router.post("/login")
def login(payload: UserLoginRequest, response: Response):
    """用户登录。"""
    user = authenticate_user(payload.username, payload.password)
    if not user:
        raise HTTPException(status_code=401, detail="用户名或密码错误")
    cookie_val = _create_session_cookie(user["user_id"])
    response.set_cookie(AUTH_COOKIE_NAME, cookie_val, max_age=COOKIE_MAX_AGE, httponly=True, samesite="lax")
    return {"code": 0, "message": "ok", "data": user}


@router.post("/logout")
def logout(response: Response):
    """退出登录（清除 Cookie）。"""
    response.delete_cookie(AUTH_COOKIE_NAME)
    return {"code": 0, "message": "ok"}


@router.get("/me")
def get_me(user: dict = Depends(get_current_user)):
    """获取当前登录用户信息。"""
    return {"code": 0, "message": "ok", "data": user}


@router.put("/password")
def change_my_password(payload: ChangePasswordRequest, user: dict = Depends(get_current_user)):
    """修改密码。"""
    if len(payload.new_password) < 4:
        raise HTTPException(status_code=400, detail="新密码长度不能少于4位")
    ok = change_password(user["user_id"], payload.old_password, payload.new_password)
    if not ok:
        raise HTTPException(status_code=400, detail="旧密码不正确")
    return {"code": 0, "message": "ok"}


@router.post("/api-key")
def refresh_api_key(user: dict = Depends(get_current_user)):
    """刷新 API Key。"""
    new_key = regenerate_api_key(user["user_id"])
    return {"code": 0, "message": "ok", "data": {"api_key": new_key}}


def ensure_kb_access(kb_id: str, user: dict) -> dict:
    """校验资源所属知识库；admin 可访问所有库，其他用户必须有绑定。"""
    import database
    kb = database.get_kb(kb_id)
    if kb is None or (user["role"] != "admin" and not check_kb_permission(user["user_id"], kb_id)):
        raise HTTPException(
            status_code=404,
            detail={"code": 40401, "message": "知识库不存在或无权访问"},
        )
    return kb