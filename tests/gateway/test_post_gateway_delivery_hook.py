"""Behavior contract for the gateway post-delivery plugin boundary."""

import asyncio
from unittest.mock import AsyncMock, patch

import pytest

from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import BasePlatformAdapter, MessageEvent, SendResult
from gateway.run import GatewayRunner
from gateway.session import SessionSource, build_session_key


class DeliveryAdapter(BasePlatformAdapter):
    def __init__(self, results=None):
        super().__init__(
            PlatformConfig(enabled=True, token="test", typing_indicator=False),
            Platform.TELEGRAM,
        )
        self.results = list(results or [SendResult(success=True, message_id="ok")])
        self.send_calls = []

    async def connect(self, *, is_reconnect=False):
        return True

    async def disconnect(self):
        return None

    async def send(self, chat_id, content, reply_to=None, metadata=None):
        self.send_calls.append(content)
        if self.results:
            return self.results.pop(0)
        return SendResult(success=True, message_id="ok")

    async def send_typing(self, chat_id, metadata=None):
        return None

    async def get_chat_info(self, chat_id):
        return {"id": chat_id}


def _event():
    return MessageEvent(
        text="hello",
        message_id="in-1",
        source=SessionSource(
            platform=Platform.TELEGRAM,
            chat_id="chat-1",
            chat_type="direct",
            user_id="user-1",
        ),
    )


async def _run(adapter, monkeypatch, response="answer", *, stale=False):
    event = _event()
    session_key = build_session_key(event.source)
    calls = []

    async def capture(name, **payload):
        calls.append((name, payload))

    async def handler(_event):
        guard = adapter._active_sessions[session_key]
        guard._hermes_run_generation = 7
        if stale:
            replacement = asyncio.Event()
            replacement._hermes_run_generation = 8
            adapter._active_sessions[session_key] = replacement
        return response

    monkeypatch.setattr("hermes_cli.plugins.invoke_hook_async", capture)
    adapter.set_message_handler(handler)
    await adapter._process_message_background(event, session_key)
    return calls


@pytest.mark.asyncio
async def test_success_emits_once_after_delivery(monkeypatch):
    calls = await _run(DeliveryAdapter(), monkeypatch)

    assert len(calls) == 1
    name, payload = calls[0]
    assert name == "post_gateway_delivery"
    assert payload["event"].message_id == "in-1"
    assert payload["session_key"] == build_session_key(_event().source)
    assert payload["generation"] == 7
    assert payload["turn_id"] is None
    assert payload["outcome"] == "success"
    assert payload["delivery_attempted"] is True
    assert payload["delivery_succeeded"] is True
    assert payload["outbound_message_id"] == "ok"
    assert payload["outbound_message_ids"] == ("ok",)


@pytest.mark.asyncio
async def test_final_delivery_failure_emits_once(monkeypatch):
    timeout = SendResult(success=False, error="ReadTimeout: uncertain delivery")
    calls = await _run(DeliveryAdapter([timeout]), monkeypatch)

    assert len(calls) == 1
    assert calls[0][1]["outcome"] == "failure"
    assert calls[0][1]["delivery_attempted"] is True
    assert calls[0][1]["delivery_succeeded"] is False


@pytest.mark.asyncio
async def test_chunked_adapter_emits_only_after_all_chunks(monkeypatch):
    adapter = DeliveryAdapter()

    async def chunked_send(chat_id, content, reply_to=None, metadata=None):
        adapter.send_calls.extend([content[:3], content[3:]])
        return SendResult(success=True, message_id="chunk-2")

    adapter.send = chunked_send
    calls = await _run(adapter, monkeypatch, response="chunked")

    assert adapter.send_calls == ["chu", "nked"]
    assert len(calls) == 1
    assert calls[0][1]["delivery_succeeded"] is True
    assert calls[0][1]["outbound_message_id"] == "chunk-2"
    assert calls[0][1]["outbound_message_ids"] == ("chunk-2",)


@pytest.mark.asyncio
async def test_split_delivery_exposes_all_outbound_message_ids(monkeypatch):
    split = SendResult(
        success=True,
        message_id="part-3",
        continuation_message_ids=("part-1", "part-2", "part-3"),
    )
    calls = await _run(DeliveryAdapter([split]), monkeypatch)

    payload = calls[0][1]
    assert payload["outbound_message_id"] == "part-3"
    assert payload["outbound_message_ids"] == ("part-1", "part-2", "part-3")


@pytest.mark.asyncio
async def test_retry_emits_once_for_final_success(monkeypatch):
    adapter = DeliveryAdapter(
        [
            SendResult(success=False, error="ConnectError", retryable=True),
            SendResult(success=True, message_id="retry-ok"),
        ]
    )
    with patch("asyncio.sleep", new_callable=AsyncMock):
        calls = await _run(adapter, monkeypatch)

    assert len(adapter.send_calls) == 2
    assert len(calls) == 1
    assert calls[0][1]["outcome"] == "success"
    assert calls[0][1]["outbound_message_id"] == "retry-ok"
    assert calls[0][1]["outbound_message_ids"] == ("retry-ok",)


@pytest.mark.asyncio
async def test_no_delivery_still_emits_once(monkeypatch):
    calls = await _run(DeliveryAdapter(), monkeypatch, response=None)

    assert len(calls) == 1
    assert calls[0][1]["outcome"] == "success"
    assert calls[0][1]["delivery_attempted"] is False
    assert calls[0][1]["delivery_succeeded"] is False
    assert calls[0][1]["outbound_message_id"] is None
    assert calls[0][1]["outbound_message_ids"] == ()


@pytest.mark.asyncio
async def test_turn_id_is_exposed_when_bound_to_gateway_turn(monkeypatch):
    adapter = DeliveryAdapter()
    event = _event()
    session_key = build_session_key(event.source)
    calls = []

    async def capture(name, **payload):
        calls.append((name, payload))

    async def handler(_event):
        guard = adapter._active_sessions[session_key]
        guard._hermes_run_generation = 7
        guard._hermes_turn_id = "turn-123"
        return "answer"

    monkeypatch.setattr("hermes_cli.plugins.invoke_hook_async", capture)
    adapter.set_message_handler(handler)
    await adapter._process_message_background(event, session_key)

    assert calls[0][1]["turn_id"] == "turn-123"


def test_runner_binds_turn_id_only_to_matching_delivery_generation():
    runner = object.__new__(GatewayRunner)
    adapter = DeliveryAdapter()
    session_key = "agent:cigre:zulip:stream:5:topic"
    guard = asyncio.Event()
    guard._hermes_run_generation = 7
    adapter._active_sessions[session_key] = guard

    runner._bind_adapter_turn_id(adapter, session_key, "turn-7", 7)
    assert guard._hermes_turn_id == "turn-7"

    replacement = asyncio.Event()
    replacement._hermes_run_generation = 8
    adapter._active_sessions[session_key] = replacement
    runner._bind_adapter_turn_id(adapter, session_key, "stale-turn-7", 7)
    assert not hasattr(replacement, "_hermes_turn_id")


@pytest.mark.asyncio
async def test_secondary_profile_key_survives_through_delivery_hook(
    monkeypatch, tmp_path
):
    runner = object.__new__(GatewayRunner)
    runner.session_store = None
    runner._busy_text_mode = "queue"
    monkeypatch.setattr(
        "hermes_cli.profiles.get_profile_dir", lambda _name: tmp_path
    )
    monkeypatch.setattr(
        runner, "_make_adapter_auth_check", lambda *_args, **_kwargs: None
    )

    adapter = DeliveryAdapter()
    runner._configure_profile_adapter(adapter, "cigre", Platform.TELEGRAM)

    event = _event()
    expected_key = build_session_key(event.source, profile="cigre")
    calls = []
    delivered = asyncio.Event()

    async def capture(name, **payload):
        calls.append((name, payload))
        delivered.set()

    async def handler(_event):
        guard = adapter._active_sessions[expected_key]
        guard._hermes_run_generation = 7
        guard._hermes_turn_id = "turn-cigre"
        return "answer"

    monkeypatch.setattr("hermes_cli.plugins.invoke_hook_async", capture)
    adapter.set_message_handler(handler)

    await adapter.handle_message(event)
    await asyncio.wait_for(delivered.wait(), timeout=2)

    assert event.source.profile == "cigre"
    assert calls[0][0] == "post_gateway_delivery"
    assert calls[0][1]["session_key"] == expected_key
    assert calls[0][1]["turn_id"] == "turn-cigre"


@pytest.mark.asyncio
async def test_stale_generation_cannot_report_success(monkeypatch):
    calls = await _run(DeliveryAdapter(), monkeypatch, stale=True)

    assert len(calls) == 1
    assert calls[0][1]["generation"] == 7
    assert calls[0][1]["outcome"] == "cancelled"
