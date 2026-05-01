"""Tests for DashScope ASR client polling."""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from app.infra import dashscope_asr as asr_client
from app.infra.dashscope_asr import wait_transcription
from app.config import Settings


@dataclass(slots=True)
class FakeResponse:
    """Small DashScope response fixture."""

    output: dict | None
    status_code: int = 200


def test_wait_transcription_polls_until_success(monkeypatch: pytest.MonkeyPatch) -> None:
    """wait_transcription should poll with fetch and emit status callbacks."""
    responses = [
        FakeResponse({"task_status": "RUNNING"}),
        FakeResponse(
            {
                "task_status": "SUCCEEDED",
                "results": [{"subtask_status": "SUCCEEDED", "transcription_url": "https://example.invalid/asr.json"}],
            }
        ),
    ]
    sleeps: list[float] = []
    events = []

    monkeypatch.setattr(asr_client.Transcription, "fetch", lambda task: responses.pop(0))
    monkeypatch.setattr(asr_client.time, "sleep", sleeps.append)

    response = wait_transcription(settings=_settings(), task="task-demo", poll_callback=events.append)

    assert response.output["task_status"] == "SUCCEEDED"
    assert [event.status for event in events] == ["RUNNING", "SUCCEEDED"]
    assert sleeps == [1.0]
    assert responses == []


def test_wait_transcription_fails_on_terminal_status(monkeypatch: pytest.MonkeyPatch) -> None:
    """Terminal failed statuses should raise instead of polling forever."""
    monkeypatch.setattr(asr_client.Transcription, "fetch", lambda task: FakeResponse({"task_status": "FAILED"}))
    monkeypatch.setattr(asr_client.time, "sleep", lambda seconds: None)

    with pytest.raises(RuntimeError, match="task failed"):
        wait_transcription(settings=_settings(), task="task-demo")


def _settings() -> Settings:
    """Build minimal runtime settings for client tests."""
    return Settings(dashscope_api_key="test-key", dashscope_base_url="https://dashscope.aliyuncs.com/api/v1")
