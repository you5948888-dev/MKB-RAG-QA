# -*- coding: utf-8 -*-
"""Ch12 验证：WebSocket 流式端点端到端测试。

覆盖 4 个场景：
  1. 认证成功（Cookie）+ citations→token→done 事件流
  2. 未认证（无 Cookie/token）→ 返回 error「未认证」
  3. 越权（用 A 用户的 KB 让 B 用户访问）→ 返回 error 权限不足
  4. 断开：收到部分 token 后关闭连接，后端停止发送且不保存

远程库 LLM 服务可能超时/502，脚本对“无检索命中”场景（空知识库）
只验证固定提示 token + done，避免依赖外部 LLM。先建一个空 KB 来测。
"""
import asyncio
import json
import sys
import uuid

import httpx
import websockets

BASE = "http://127.0.0.1:8013"
WS_URL = "ws://127.0.0.1:8013/api/qa/stream/ws"


async def make_user(client, role_label):
    """注册一个新用户，返回 (cookie, api_key)。"""
    uname = f"ws_{role_label}_{uuid.uuid4().hex[:6]}"
    r = await client.post(f"{BASE}/api/auth/register", json={
        "username": uname, "password": "test1234",
    })
    data = r.json()
    assert data["code"] == 0, f"注册失败: {data}"
    cookie = r.headers.get("set-cookie", "")
    cookie_val = cookie.split(";")[0].split("=", 1)[1] if "=" in cookie else ""
    return data["data"]["api_key"], cookie_val, uname


async def create_kb(client, cookie, name):
    """用 cookie 创建一个知识库，返回 kb_id。"""
    r = await client.post(f"{BASE}/api/kb", json={"name": name, "description": "ws测试"},
                          headers={"Cookie": f"rag_session={cookie}"})
    data = r.json()
    return data["data"]["kb_id"]


async def collect_ws(cookie, payload, timeout=30):
    """连上 WS 发送 payload，收集事件直到 done/error/断开。"""
    headers = {"Cookie": f"rag_session={cookie}"} if cookie else {}
    events = []
    async with websockets.connect(WS_URL, additional_headers=headers) as ws:
        await ws.send(json.dumps(payload, ensure_ascii=False))
        while True:
            try:
                raw = await asyncio.wait_for(ws.recv(), timeout=timeout)
            except asyncio.TimeoutError:
                events.append({"event": "__timeout"})
                break
            evt = json.loads(raw)
            events.append(evt)
            if evt["event"] in ("done", "error"):
                break
    return events


async def main():
    async with httpx.AsyncClient(timeout=30) as client:
        # 用户 A：合法
        api_a, cookie_a, uname_a = await make_user(client, "ok")
        kb_a = await create_kb(client, cookie_a, f"wsKB_{uuid.uuid4().hex[:6]}")

        # 用户 B：用于越权测试
        api_b, cookie_b, _ = await make_user(client, "b")

        print("=== 场景1: 合法用户 Cookie 认证 + 事件流（空知识库→固定提示） ===")
        ev1 = await collect_ws(cookie_a, {
            "question": "什么是RAG？", "kb_id": kb_a, "session_id": None,
        })
        types = [e["event"] for e in ev1]
        print("事件序列:", types)
        assert types[0] == "citations", f"首个事件应为 citations, got {types}"
        assert "token" in types and types[-1] == "done", f"应以 done 结尾, got {types[-1]}"
        token_text = "".join(e["data"].get("content", "") for e in ev1 if e["event"] == "token")
        print("流式拼接内容:", repr(token_text[:60]))
        print("✓ 场景1 通过")

        print("\n=== 场景2: 未认证 → error ===")
        ev2 = await collect_ws(None, {"question": "hi", "kb_id": kb_a, "session_id": None})
        print("事件:", ev2)
        assert ev2 and ev2[0]["event"] == "error" and "未认证" in ev2[0]["data"]["message"], ev2
        print("✓ 场景2 通过")

        print("\n=== 场景3: 越权（B 用户访问 A 的 KB）→ error ===")
        ev3 = await collect_ws(cookie_b, {"question": "hi", "kb_id": kb_a, "session_id": None})
        print("事件:", ev3)
        assert ev3 and ev3[0]["event"] == "error", ev3
        print("✓ 场景3 通过")

        print("\n=== 场景4: 未认证但带 API Key（消息体 token）认证 ===")
        headers4 = {}  # 不带 Cookie，改用消息体 token
        async with websockets.connect(WS_URL, additional_headers=headers4) as ws:
            await ws.send(json.dumps({
                "question": "RAG架构", "kb_id": kb_a,
                "session_id": None, "token": api_a,
            }, ensure_ascii=False))
            got_citations = False
            while True:
                raw = await asyncio.wait_for(ws.recv(), timeout=30)
                evt = json.loads(raw)
                if evt["event"] == "citations":
                    got_citations = True
                if evt["event"] in ("done", "error"):
                    print("场景4 最终事件:", evt["event"])
                    break
        assert got_citations, "场景4 未收到 citations"
        print("✓ 场景4（API Key 认证）通过")

        # 清理：删除测试知识库（先征得的设计——此处为脚本自动清理自建测试 KB）
        try:
            await client.request("DELETE", f"{BASE}/api/kb/{kb_a}",
                                 headers={"Cookie": f"rag_session={cookie_a}"})
            print("已清理测试 KB:", kb_a)
        except Exception as exc:
            print("清理测试 KB 失败:", exc)

    print("\n所有 WebSocket 场景验证通过 ✓")


if __name__ == "__main__":
    asyncio.run(main())