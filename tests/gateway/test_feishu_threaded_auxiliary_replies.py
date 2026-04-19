import asyncio
import sys
import threading
import time
import types
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

import gateway.run as gateway_run
from gateway.config import Platform
from gateway.session import SessionSource, build_session_key


class _CapturingFeishuAdapter:
    name = "feishu"

    def __init__(self):
        self.send_calls = []
        self.edit_calls = []
        self.typing_calls = []
        self._pending_messages = {}

    async def send(self, chat_id, content, reply_to=None, metadata=None):
        self.send_calls.append(
            {
                "chat_id": chat_id,
                "content": content,
                "reply_to": reply_to,
                "metadata": metadata,
            }
        )
        return SimpleNamespace(success=True, message_id=f"sent-{len(self.send_calls)}")

    async def edit_message(self, chat_id, message_id, content):
        self.edit_calls.append(
            {"chat_id": chat_id, "message_id": message_id, "content": content}
        )
        return SimpleNamespace(success=True, message_id=message_id)

    async def send_typing(self, chat_id, metadata=None):
        self.typing_calls.append({"chat_id": chat_id, "metadata": metadata})
        return None

    def pause_typing_for_chat(self, _chat_id):
        return None

    def get_pending_message(self, _session_key):
        return None

    def has_pending_interrupt(self, _session_key):
        return False


class _ThreadedStatusAgent:
    def __init__(self, *args, **kwargs):
        self.tools = []
        self.status_callback = None
        self.tool_progress_callback = None
        self.interim_assistant_callback = None
        self.stream_delta_callback = None
        self.step_callback = None
        self.reasoning_config = None
        self.service_tier = None
        self.request_overrides = None

    def run_conversation(self, user_message, conversation_history=None, task_id=None):
        if self.status_callback:
            self.status_callback("status", "⌛ Retrying in 2.0s (attempt 1/3)...")
        if self.tool_progress_callback:
            self.tool_progress_callback(
                "tool.started",
                tool_name="process",
                preview='poll proc_123',
                args={"action": "poll", "session_id": "proc_123"},
            )
        time.sleep(0.6)
        return {
            "final_response": "ok",
            "messages": [],
            "api_calls": 1,
            "completed": True,
        }

    def get_activity_summary(self):
        return {
            "api_call_count": 1,
            "max_iterations": 90,
            "current_tool": "process",
            "last_activity_desc": "testing",
            "seconds_since_activity": 0.0,
        }


class _FakeRegistry:
    def __init__(self, sessions):
        self._sessions = list(sessions)

    def get(self, _session_id):
        if self._sessions:
            return self._sessions.pop(0)
        return None

    def is_completion_consumed(self, _session_id):
        return False



def _install_fake_agent(monkeypatch):
    fake_run_agent = types.ModuleType("run_agent")
    fake_run_agent.AIAgent = _ThreadedStatusAgent
    monkeypatch.setitem(sys.modules, "run_agent", fake_run_agent)



def _make_runner(adapter=None):
    runner = object.__new__(gateway_run.GatewayRunner)
    runner.adapters = {}
    runner._ephemeral_system_prompt = ""
    runner._prefill_messages = []
    runner._reasoning_config = None
    runner._service_tier = None
    runner._provider_routing = {}
    runner._fallback_model = None
    runner._smart_model_routing = {}
    runner._running_agents = {}
    runner._running_agents_ts = {}
    runner._running_agent_message_ids = {}
    runner._pending_model_notes = {}
    runner._pending_messages = {}
    runner._pending_approvals = {}
    runner._session_db = None
    runner._agent_cache = {}
    runner._agent_cache_lock = threading.Lock()
    runner._session_model_overrides = {}
    runner._busy_ack_ts = {}
    runner._background_tasks = set()
    runner.hooks = SimpleNamespace(loaded_hooks=False)
    runner.config = SimpleNamespace(streaming=None)
    runner._get_or_create_gateway_honcho = lambda session_key: (None, None)
    runner._load_reasoning_config = lambda: None
    runner._load_service_tier = lambda: None
    runner._get_proxy_url = lambda: None
    runner._resolve_session_agent_runtime = lambda source, session_key, user_config: (
        "gpt-5.4",
        {
            "provider": "openrouter",
            "api_mode": "chat_completions",
            "base_url": "https://openrouter.ai/api/v1",
            "api_key": "***",
        },
    )
    runner._resolve_turn_agent_config = lambda message, model, runtime: {
        "model": model,
        "runtime": runtime,
        "request_overrides": None,
    }
    runner._run_in_executor_with_context = gateway_run.GatewayRunner._run_in_executor_with_context.__get__(
        runner, gateway_run.GatewayRunner
    )
    runner.session_store = SimpleNamespace(_entries={})
    if adapter is not None:
        runner.adapters[Platform.FEISHU] = adapter
    return runner



def _make_feishu_source():
    return SessionSource(
        platform=Platform.FEISHU,
        chat_id="oc_chat",
        chat_type="group",
        thread_id="omt_thread",
        user_id="ou_user",
        user_name="Link",
    )


@pytest.mark.asyncio
async def test_run_agent_feishu_status_and_tool_progress_reply_to_thread_message(monkeypatch, tmp_path):
    _install_fake_agent(monkeypatch)
    adapter = _CapturingFeishuAdapter()
    runner = _make_runner(adapter)

    (tmp_path / "config.yaml").write_text(
        "display:\n  platforms:\n    feishu:\n      tool_progress: new\n      interim_assistant_messages: false\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)
    monkeypatch.setattr(gateway_run, "_env_path", tmp_path / ".env")
    monkeypatch.setattr(gateway_run, "load_dotenv", lambda *args, **kwargs: None)
    monkeypatch.setattr(gateway_run, "_load_gateway_config", lambda: {})

    import hermes_cli.tools_config as tools_config

    monkeypatch.setattr(tools_config, "_get_platform_tools", lambda user_config, platform_key: {"core"})

    result = await runner._run_agent(
        message="hi",
        context_prompt="",
        history=[],
        source=_make_feishu_source(),
        session_id="session-1",
        session_key="agent:main:feishu:group:oc_chat:omt_thread:ou_user",
        event_message_id="om_123",
    )

    assert result["final_response"] == "ok"

    send_calls = adapter.send_calls
    retry_call = next(call for call in send_calls if "Retrying in 2.0s" in call["content"])
    progress_call = next(call for call in send_calls if "proc_123" in call["content"])

    assert retry_call["reply_to"] == "om_123"
    assert retry_call["metadata"] == {"thread_id": "omt_thread"}
    assert progress_call["reply_to"] == "om_123"
    assert progress_call["metadata"] == {"thread_id": "omt_thread"}


@pytest.mark.asyncio
async def test_shutdown_notification_replies_to_active_feishu_thread_message():
    adapter = _CapturingFeishuAdapter()
    runner = _make_runner(adapter)
    source = _make_feishu_source()
    session_key = build_session_key(source)
    runner._running_agents[session_key] = MagicMock()
    runner._running_agent_message_ids[session_key] = "om_shutdown"
    runner.session_store._entries[session_key] = SimpleNamespace(origin=source)
    runner._restart_requested = True

    await runner._notify_active_sessions_of_shutdown()

    call = adapter.send_calls[0]
    assert call["chat_id"] == "oc_chat"
    assert "restarting" in call["content"]
    assert call["reply_to"] == "om_shutdown"
    assert call["metadata"] == {"thread_id": "omt_thread"}


@pytest.mark.asyncio
async def test_process_watcher_feishu_notification_replies_to_origin_message(monkeypatch, tmp_path):
    import tools.process_registry as pr_module

    async def _instant_sleep(*_args, **_kwargs):
        return None

    monkeypatch.setattr(asyncio, "sleep", _instant_sleep)
    monkeypatch.setattr(
        pr_module,
        "process_registry",
        _FakeRegistry([SimpleNamespace(output_buffer="done\n", exited=True, exit_code=0, command="sleep 1")]),
    )

    adapter = _CapturingFeishuAdapter()
    runner = _make_runner(adapter)
    monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)
    (tmp_path / "config.yaml").write_text(
        "display:\n  background_process_notifications: all\n",
        encoding="utf-8",
    )

    await runner._run_process_watcher(
        {
            "session_id": "proc_test",
            "check_interval": 0,
            "platform": "feishu",
            "chat_id": "oc_chat",
            "thread_id": "omt_thread",
            "message_id": "om_proc",
        }
    )

    call = adapter.send_calls[0]
    assert call["chat_id"] == "oc_chat"
    assert "finished with exit code 0" in call["content"]
    assert call["reply_to"] == "om_proc"
    assert call["metadata"] == {"thread_id": "omt_thread"}
