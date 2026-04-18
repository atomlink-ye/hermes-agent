"""Tests for session auto-reset notifications.

Verifies that:
- _should_reset() returns a reason string ("idle" or "daily") instead of bool
- SessionEntry captures auto_reset_reason
- SessionResetPolicy.notify controls whether notifications are sent
- notify_exclude_platforms skips notifications for excluded platforms
- Feishu threaded reset notices stay anchored to the triggering message
"""

from datetime import datetime, timedelta
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.platforms.base import MessageEvent, MessageType, SendResult
from gateway.config import (
    GatewayConfig,
    Platform,
    PlatformConfig,
    SessionResetPolicy,
)
from gateway.run import GatewayRunner
from gateway.session import SessionEntry, SessionSource, SessionStore


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_source(platform=Platform.TELEGRAM, chat_id="123", user_id="u1"):
    return SessionSource(
        platform=platform,
        chat_id=chat_id,
        user_id=user_id,
    )


def _make_store(policy=None, tmp_path=None):
    config = GatewayConfig()
    if policy:
        config.default_reset_policy = policy
    store = SessionStore(sessions_dir=tmp_path or "/tmp/test-sessions", config=config)
    return store


# ---------------------------------------------------------------------------
# _should_reset returns reason string
# ---------------------------------------------------------------------------

class TestShouldResetReason:
    def test_returns_none_when_not_expired(self, tmp_path):
        store = _make_store(
            SessionResetPolicy(mode="both", idle_minutes=60, at_hour=4),
            tmp_path,
        )
        entry = SessionEntry(
            session_key="test",
            session_id="s1",
            created_at=datetime.now(),
            updated_at=datetime.now(),  # just updated
        )
        source = _make_source()
        assert store._should_reset(entry, source) is None

    def test_returns_idle_when_idle_expired(self, tmp_path):
        store = _make_store(
            SessionResetPolicy(mode="idle", idle_minutes=30),
            tmp_path,
        )
        entry = SessionEntry(
            session_key="test",
            session_id="s1",
            created_at=datetime.now() - timedelta(hours=2),
            updated_at=datetime.now() - timedelta(hours=1),  # 60min ago > 30min threshold
        )
        source = _make_source()
        assert store._should_reset(entry, source) == "idle"

    def test_returns_daily_when_daily_boundary_crossed(self, tmp_path):
        now = datetime.now()
        store = _make_store(
            SessionResetPolicy(mode="daily", at_hour=now.hour),
            tmp_path,
        )
        entry = SessionEntry(
            session_key="test",
            session_id="s1",
            created_at=now - timedelta(days=2),
            updated_at=now - timedelta(days=1),  # last active yesterday
        )
        source = _make_source()
        assert store._should_reset(entry, source) == "daily"

    def test_returns_none_when_mode_is_none(self, tmp_path):
        store = _make_store(
            SessionResetPolicy(mode="none"),
            tmp_path,
        )
        entry = SessionEntry(
            session_key="test",
            session_id="s1",
            created_at=datetime.now() - timedelta(days=30),
            updated_at=datetime.now() - timedelta(days=30),
        )
        source = _make_source()
        assert store._should_reset(entry, source) is None


# ---------------------------------------------------------------------------
# SessionEntry captures reason
# ---------------------------------------------------------------------------

class TestSessionEntryReason:
    def test_auto_reset_reason_stored(self, tmp_path):
        store = _make_store(
            SessionResetPolicy(mode="idle", idle_minutes=1),
            tmp_path,
        )
        source = _make_source()

        # Create initial session
        entry1 = store.get_or_create_session(source)
        assert not entry1.was_auto_reset

        # Age it past the idle threshold
        entry1.updated_at = datetime.now() - timedelta(minutes=5)
        store._save()

        # Next call should create a new session with reason
        entry2 = store.get_or_create_session(source)
        assert entry2.was_auto_reset is True
        assert entry2.auto_reset_reason == "idle"
        assert entry2.session_id != entry1.session_id

    def test_reset_had_activity_false_when_no_tokens(self, tmp_path):
        """Expired session with no tokens → reset_had_activity=False."""
        store = _make_store(
            SessionResetPolicy(mode="idle", idle_minutes=1),
            tmp_path,
        )
        source = _make_source()

        entry1 = store.get_or_create_session(source)
        # No tokens used — session was idle with no conversation
        entry1.updated_at = datetime.now() - timedelta(minutes=5)
        store._save()

        entry2 = store.get_or_create_session(source)
        assert entry2.was_auto_reset is True
        assert entry2.reset_had_activity is False

    def test_reset_had_activity_true_when_tokens_used(self, tmp_path):
        """Expired session with tokens → reset_had_activity=True."""
        store = _make_store(
            SessionResetPolicy(mode="idle", idle_minutes=1),
            tmp_path,
        )
        source = _make_source()

        entry1 = store.get_or_create_session(source)
        # Simulate some conversation happened
        entry1.total_tokens = 5000
        entry1.updated_at = datetime.now() - timedelta(minutes=5)
        store._save()

        entry2 = store.get_or_create_session(source)
        assert entry2.was_auto_reset is True
        assert entry2.reset_had_activity is True


# ---------------------------------------------------------------------------
# SessionResetPolicy notify config
# ---------------------------------------------------------------------------

class TestResetPolicyNotify:
    def test_notify_defaults_true(self):
        policy = SessionResetPolicy()
        assert policy.notify is True

    def test_notify_exclude_defaults(self):
        policy = SessionResetPolicy()
        assert "api_server" in policy.notify_exclude_platforms
        assert "webhook" in policy.notify_exclude_platforms

    def test_from_dict_with_notify_false(self):
        policy = SessionResetPolicy.from_dict({"notify": False})
        assert policy.notify is False

    def test_from_dict_with_custom_excludes(self):
        policy = SessionResetPolicy.from_dict({
            "notify_exclude_platforms": ["api_server", "webhook", "homeassistant"],
        })
        assert "homeassistant" in policy.notify_exclude_platforms

    def test_from_dict_preserves_defaults_on_missing_keys(self):
        policy = SessionResetPolicy.from_dict({})
        assert policy.notify is True
        assert "api_server" in policy.notify_exclude_platforms

    def test_to_dict_roundtrip(self):
        original = SessionResetPolicy(
            mode="idle",
            notify=False,
            notify_exclude_platforms=("api_server",),
        )
        restored = SessionResetPolicy.from_dict(original.to_dict())
        assert restored.notify == original.notify
        assert restored.notify_exclude_platforms == original.notify_exclude_platforms
        assert restored.mode == original.mode


@pytest.mark.asyncio
async def test_feishu_threaded_auto_reset_notice_replies_to_trigger_message(monkeypatch):
    adapter = MagicMock()
    adapter.send = AsyncMock(return_value=SendResult(success=True, message_id="reset-1"))

    runner = object.__new__(GatewayRunner)
    runner.adapters = {Platform.FEISHU: adapter}
    runner.config = GatewayConfig(
        platforms={Platform.FEISHU: PlatformConfig(enabled=True, token="***")}
    )
    runner._voice_mode = {}
    runner._session_model_overrides = {}
    runner._pending_model_notes = {}
    runner._background_tasks = set()
    runner._pending_messages = {}
    runner._pending_approvals = {}
    runner._running_agents = {}
    runner._running_agents_ts = {}
    runner._session_db = None
    runner.hooks = SimpleNamespace(emit=AsyncMock(), loaded_hooks=False)
    runner._format_session_info = lambda: ""
    runner._set_session_env = lambda _context: []
    runner._clear_session_env = lambda _tokens: None
    runner._prepare_inbound_message_text = AsyncMock(return_value="hello")
    runner._run_agent = AsyncMock(return_value={"final_response": "done", "messages": [], "api_calls": 0})
    runner._clear_restart_failure_count = MagicMock()
    runner._should_send_voice_reply = lambda *_args, **_kwargs: False
    runner._deliver_media_from_response = AsyncMock()

    session_key = "agent:main:feishu:group:oc_chat:omt_thread"
    source = SessionSource(
        platform=Platform.FEISHU,
        chat_id="oc_chat",
        chat_type="group",
        thread_id="omt_thread",
        user_id="u1",
    )
    session_entry = SessionEntry(
        session_key=session_key,
        session_id="sess-reset",
        created_at=datetime.now(),
        updated_at=datetime.now(),
        origin=source,
        platform=Platform.FEISHU,
        chat_type="group",
        was_auto_reset=True,
        auto_reset_reason="suspended",
        reset_had_activity=True,
    )

    runner.session_store = MagicMock()
    runner.session_store.config = GatewayConfig(
        default_reset_policy=SessionResetPolicy(mode="idle", idle_minutes=60, notify=True),
    )
    runner.session_store.get_or_create_session.return_value = session_entry
    runner.session_store.load_transcript.return_value = []
    runner.session_store.has_any_sessions.return_value = True
    runner.session_store.append_to_transcript = MagicMock()
    runner.session_store.update_session = MagicMock()

    monkeypatch.setenv("FEISHU_HOME_CHANNEL", "oc_home")

    event = MessageEvent(
        text="resume",
        message_type=MessageType.TEXT,
        source=source,
        message_id="om_origin",
    )

    response = await runner._handle_message_with_agent(event, source, "quick-reset")

    assert response == "done"
    notice_call = adapter.send.await_args_list[0]
    assert notice_call.args[0] == "oc_chat"
    assert "Session automatically reset" in notice_call.args[1]
    assert notice_call.kwargs["reply_to"] == "om_origin"
    assert notice_call.kwargs["metadata"] == {"thread_id": "omt_thread"}
