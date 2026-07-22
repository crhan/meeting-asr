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


def test_wait_transcription_polls_until_success(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """wait_transcription should poll with fetch and emit status callbacks."""
    responses = [
        FakeResponse({"task_status": "RUNNING"}),
        FakeResponse(
            {
                "task_status": "SUCCEEDED",
                "results": [
                    {
                        "subtask_status": "SUCCEEDED",
                        "transcription_url": "https://example.invalid/asr.json",
                    }
                ],
            }
        ),
    ]
    sleeps: list[float] = []
    events = []

    monkeypatch.setattr(
        asr_client.Transcription, "fetch", lambda task: responses.pop(0)
    )
    monkeypatch.setattr(asr_client.time, "sleep", sleeps.append)

    response = wait_transcription(
        settings=_settings(), task="task-demo", poll_callback=events.append
    )

    assert response.output["task_status"] == "SUCCEEDED"
    assert [event.status for event in events] == ["RUNNING", "SUCCEEDED"]
    assert sleeps == [1.0]
    assert responses == []


def test_wait_transcription_fails_on_terminal_status(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Terminal failed statuses should raise instead of polling forever."""
    monkeypatch.setattr(
        asr_client.Transcription,
        "fetch",
        lambda task: FakeResponse({"task_status": "FAILED"}),
    )
    monkeypatch.setattr(asr_client.time, "sleep", lambda seconds: None)

    with pytest.raises(RuntimeError, match="task failed"):
        wait_transcription(settings=_settings(), task="task-demo")


def test_wait_transcription_times_out_on_wall_clock_budget(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A task stuck in RUNNING must stop polling once the budget elapses."""
    monkeypatch.setattr(
        asr_client.Transcription,
        "fetch",
        lambda task: FakeResponse({"task_status": "RUNNING"}),
    )
    monkeypatch.setattr(asr_client.time, "sleep", lambda seconds: None)

    with pytest.raises(asr_client.TranscriptionWaitTimeout):
        wait_transcription(
            settings=_settings(), task="task-demo", max_wait_seconds=0.0
        )


def test_probe_transcription_reattaches_running_task(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A pending or running task should be reported as reattachable."""
    monkeypatch.setattr(
        asr_client.Transcription,
        "fetch",
        lambda task: FakeResponse({"task_status": "RUNNING"}),
    )

    response = asr_client.probe_transcription(settings=_settings(), task="task-demo")

    assert response is not None
    assert response.output["task_status"] == "RUNNING"


def test_probe_transcription_rejects_failed_or_unreachable_task(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Failed and unfetchable tasks must fall back to a fresh submission."""
    monkeypatch.setattr(
        asr_client.Transcription,
        "fetch",
        lambda task: FakeResponse({"task_status": "FAILED"}),
    )
    assert asr_client.probe_transcription(settings=_settings(), task="t") is None

    def _boom(task):
        raise RuntimeError("network down")

    monkeypatch.setattr(asr_client.Transcription, "fetch", _boom)
    monkeypatch.setattr(asr_client.time, "sleep", lambda seconds: None)
    assert asr_client.probe_transcription(settings=_settings(), task="t") is None


def test_submit_transcription_passes_vocabulary_id(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """ASR hotword vocabulary ids should be sent with the task request."""
    captured = {}

    def fake_async_call(**kwargs):
        captured.update(kwargs)
        return FakeResponse({"task_id": "task-demo"})

    monkeypatch.setattr(asr_client.Transcription, "async_call", fake_async_call)

    asr_client.submit_transcription(
        settings=_settings(),
        file_url="https://example.invalid/audio.wav",
        model="fun-asr",
        language_hints=["zh", "en"],
        speaker_count=2,
        vocabulary_id="vocab-demo",
        timestamp_alignment_enabled=True,
        disfluency_removal_enabled=False,
    )

    assert captured["vocabulary_id"] == "vocab-demo"
    assert captured["model"] == "fun-asr"
    assert captured["file_urls"] == ["https://example.invalid/audio.wav"]


def _settings() -> Settings:
    """Build minimal runtime settings for client tests."""
    return Settings(
        dashscope_api_key="test-key",
        dashscope_base_url="https://dashscope.aliyuncs.com/api/v1",
    )
