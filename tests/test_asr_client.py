"""Tests for DashScope ASR client polling."""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from app.infra import dashscope_asr as asr_client
from app.infra.dashscope_asr import SubmittedTranscriptionTask, download_transcription_json, wait_transcription
from app.config import Settings


@dataclass(slots=True)
class FakeResponse:
    """Small DashScope response fixture."""

    output: dict | None
    status_code: int = 200


@dataclass(slots=True)
class FakeHttpResponse:
    """Small requests response fixture."""

    payload: dict

    def raise_for_status(self) -> None:
        """Pretend the HTTP response succeeded."""

    def json(self) -> dict:
        """Return the configured JSON body."""
        return self.payload


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


def test_submit_transcription_passes_vocabulary_id(monkeypatch: pytest.MonkeyPatch) -> None:
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


def test_submit_transcription_uses_qwen_filetrans_rest_api(monkeypatch: pytest.MonkeyPatch) -> None:
    """Qwen-ASR long-audio models should use the REST async transcription API."""
    captured = {}

    def fake_post(url, *, headers, json, timeout):
        captured["url"] = url
        captured["headers"] = headers
        captured["json"] = json
        captured["timeout"] = timeout
        return FakeHttpResponse({"output": {"task_id": "qwen-task"}})

    monkeypatch.setattr(asr_client.requests, "post", fake_post)

    task = asr_client.submit_transcription(
        settings=_settings(),
        file_url="https://example.invalid/audio.wav",
        model="qwen3-asr-flash-filetrans",
        language_hints=["zh", "en"],
        speaker_count=2,
        vocabulary_id="vocab-ignored",
        timestamp_alignment_enabled=True,
        disfluency_removal_enabled=True,
    )

    assert isinstance(task, SubmittedTranscriptionTask)
    assert task.task_id == "qwen-task"
    assert captured["url"] == "https://dashscope.aliyuncs.com/api/v1/services/audio/asr/transcription"
    assert captured["headers"]["X-DashScope-Async"] == "enable"
    assert captured["json"]["model"] == "qwen3-asr-flash-filetrans"
    assert captured["json"]["input"]["file_url"] == "https://example.invalid/audio.wav"
    assert captured["json"]["parameters"] == {
        "channel_id": [0],
        "enable_itn": False,
        "enable_words": True,
    }
    assert captured["timeout"] == 30


def test_wait_transcription_polls_qwen_filetrans_task(monkeypatch: pytest.MonkeyPatch) -> None:
    """Qwen-ASR task polling should query the REST task endpoint."""
    responses = [
        {"output": {"task_status": "RUNNING", "task_id": "qwen-task"}},
        {
            "output": {
                "task_status": "SUCCEEDED",
                "task_id": "qwen-task",
                "result": {"transcription_url": "https://example.invalid/qwen.json"},
            },
            "usage": {"seconds": 3},
        },
    ]
    urls: list[str] = []
    events = []

    def fake_get(url, *, headers, timeout):
        urls.append(url)
        assert headers["Authorization"] == "Bearer test-key"
        assert timeout == 30
        return FakeHttpResponse(responses.pop(0))

    monkeypatch.setattr(asr_client.requests, "get", fake_get)
    monkeypatch.setattr(asr_client.time, "sleep", lambda seconds: None)

    response = wait_transcription(
        settings=_settings(),
        task=SubmittedTranscriptionTask("qwen_filetrans", "qwen-task", {}),
        poll_callback=events.append,
    )

    assert response["output"]["task_status"] == "SUCCEEDED"
    assert urls == [
        "https://dashscope.aliyuncs.com/api/v1/tasks/qwen-task",
        "https://dashscope.aliyuncs.com/api/v1/tasks/qwen-task",
    ]
    assert [event.status for event in events] == ["RUNNING", "SUCCEEDED"]


def test_download_transcription_json_accepts_qwen_result_url(monkeypatch: pytest.MonkeyPatch) -> None:
    """Qwen-ASR completion responses expose the result URL under output.result."""
    captured = {}

    def fake_get(url, *, timeout):
        captured["url"] = url
        captured["timeout"] = timeout
        return FakeHttpResponse({"transcripts": [{"sentences": [{"text": "hello"}]}]})

    monkeypatch.setattr(asr_client.requests, "get", fake_get)

    payload = download_transcription_json(
        {
            "output": {
                "task_status": "SUCCEEDED",
                "result": {"transcription_url": "https://example.invalid/qwen.json"},
            }
        }
    )

    assert payload["transcripts"][0]["sentences"][0]["text"] == "hello"
    assert captured == {"url": "https://example.invalid/qwen.json", "timeout": 30}


def _settings() -> Settings:
    """Build minimal runtime settings for client tests."""
    return Settings(dashscope_api_key="test-key", dashscope_base_url="https://dashscope.aliyuncs.com/api/v1")
