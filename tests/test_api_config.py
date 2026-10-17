import json

import httpx
import pytest
from openai import OpenAI
from streamlit.proto.TextInput_pb2 import TextInput
from streamlit.testing.v1 import AppTest

from campus_health import cli, llm
from campus_health.api_config import DEEPSEEK_URL, APIConfig, page_api
from campus_health.api_ui import cost_rows
from campus_health.storage import config


@pytest.fixture
def fake_api(monkeypatch):
    requests = []
    state = {"status": 200, "content": '{"ok":true}'}

    def handler(request):
        requests.append(request)
        if request.url.path.endswith("/models"):
            if state["status"] != 200:
                return httpx.Response(state["status"], json={"error": {"message": "echo-secret"}})
            return httpx.Response(
                200,
                json={
                    "object": "list",
                    "data": [
                        {"id": "my-model", "object": "model", "created": 0, "owned_by": "test"}
                    ],
                },
            )
        if state["status"] != 200:
            return httpx.Response(state["status"], json={"error": {"message": "echo-secret"}})
        return httpx.Response(
            200,
            json={
                "id": "test",
                "object": "chat.completion",
                "created": 0,
                "model": "my-model",
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": "stop",
                        "message": {"role": "assistant", "content": state["content"]},
                    }
                ],
                "usage": {"prompt_tokens": 5, "completion_tokens": 3, "total_tokens": 8},
            },
        )

    def factory(**kwargs):
        kwargs.pop("http_client").close()
        return OpenAI(**kwargs, http_client=httpx.Client(transport=httpx.MockTransport(handler)))

    monkeypatch.setattr(llm, "OpenAI", factory)
    return requests, state


def test_page_key_precedence_and_no_cross_provider_fallback(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "local-secret")
    monkeypatch.delenv("DEEPSEEK_BASE_URL", raising=False)
    assert page_api("deepseek", "", "deepseek-flash", "page-secret").api_key == "page-secret"
    assert page_api("deepseek", "", "deepseek-flash", "").api_key == "local-secret"
    assert not page_api("custom", "https://example.org/v1", "m", "").api_key
    monkeypatch.setenv("DEEPSEEK_BASE_URL", "https://other.example/v1")
    assert not page_api("deepseek", "", "deepseek-flash", "").api_key
    assert "page-secret" not in repr(page_api("deepseek", "", "m", "page-secret"))


@pytest.mark.parametrize(
    "url",
    [
        "http://example.org",
        "https://key@example.org",
        "https://example.org?key=secret",
        "https://example.org/#key",
    ],
)
def test_reject_unsafe_endpoint(url):
    with pytest.raises(ValueError):
        APIConfig("custom", url, "m", "secret")


def test_custom_payload_and_echo_redaction(fake_api, tmp_path):
    requests, state = fake_api
    state["content"] = '{"ok":true,"echo":"echo-secret"}'
    api = APIConfig("custom", "https://custom.example/v1", "my-model", "echo-secret")
    value, info = llm.call_json([], config(), tmp_path / "calls.jsonl", "test", api=api)
    payload = json.loads(requests[0].content)
    assert "thinking" not in payload and "extra_body" not in payload
    assert payload["response_format"] == {"type": "json_object"}
    assert requests[0].headers["authorization"] == "Bearer echo-secret"
    assert requests[0].url.host == "custom.example"
    assert info["cost"]["cny_min"] is None
    assert value["echo"] == "[REDACTED]"
    assert "echo-secret" not in (tmp_path / "calls.jsonl").read_text(encoding="utf-8")


@pytest.mark.parametrize("status, expected", [(401, "鉴权失败"), (400, "请求不兼容")])
def test_error_does_not_leak_provider_body(fake_api, tmp_path, status, expected):
    requests, state = fake_api
    state["status"] = status
    api = APIConfig("custom", "https://custom.example/v1", "my-model", "echo-secret")
    with pytest.raises(RuntimeError, match=expected) as exc:
        llm.call_json([], config(), tmp_path / "calls.jsonl", "test", api=api)
    assert len(requests) == 1
    assert "echo-secret" not in str(exc.value)
    assert "echo-secret" not in (tmp_path / "calls.jsonl").read_text(encoding="utf-8")


def test_models_and_connection(fake_api, tmp_path):
    api = APIConfig("custom", "https://custom.example/v1", "my-model", "echo-secret")
    assert llm.list_models(api, config()) == ["my-model"]
    info = llm.test_connection(api, config(), tmp_path / "calls.jsonl")
    assert info["response"]["model"] == "my-model"
    assert info["usage"]["prompt_tokens"] == 5
    fake_api[1]["status"] = 404
    with pytest.raises(RuntimeError, match="可继续手填"):
        llm.list_models(api, config())


def test_official_only_parameters_and_price(fake_api, tmp_path):
    fake_api[1]["content"] = '{"ok":true}'
    api = APIConfig("deepseek", DEEPSEEK_URL, "my-model", "secret")
    _, info = llm.call_json(
        [], {**config(), "price_model": "my-model"}, tmp_path / "calls.jsonl", "test", api=api
    )
    assert json.loads(fake_api[0][0].content)["thinking"] == {"type": "disabled"}
    assert info["cost"]["cny_min"] > 0


def test_b0_needs_no_api(monkeypatch, tmp_path):
    monkeypatch.setattr(cli, "new_run", lambda *args: tmp_path)
    monkeypatch.setenv("DEEPSEEK_BASE_URL", "invalid")

    class Retriever:
        def search(self, question, method):
            assert method == "B0"
            return [{"text": "真实检索原文", "title": "原文"}]

    result, _ = cli.interactive("问题", "B0", config(), Retriever())
    assert result["generation_api"] is None
    assert result["usage"] is None
    assert "真实检索原文" in result["answer"]


UI_SCRIPT = """
from campus_health.api_ui import api_controls, apply_pending_model
from campus_health.storage import config
apply_pending_model()
api_controls(config())
"""


def test_ui_session_isolation_clear_and_manual_model(monkeypatch):
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    first = AppTest.from_string(UI_SCRIPT).run()
    second = AppTest.from_string(UI_SCRIPT).run()
    first.text_input(key="api_key_deepseek").set_value("private-session-key").run()
    assert second.text_input(key="api_key_deepseek").value == ""
    assert first.text_input(key="api_key_deepseek").proto.type == TextInput.PASSWORD
    next(b for b in first.button if b.label == "清除会话配置").click().run()
    assert first.text_input(key="api_key_deepseek").value == ""
    assert not first.checkbox(key="api_use_local").value
    first.selectbox(key="api_provider").select("自定义兼容接口").run()
    first.text_input(key="api_base_url").set_value("https://custom.example/v1").run()
    first.text_input(key="api_model_custom").set_value("manual-model").run()
    assert not first.exception
    assert first.text_input(key="api_model_custom").value == "manual-model"


def test_ui_models_selection_and_wrong_key(fake_api, monkeypatch, tmp_path):
    import campus_health.api_ui as ui

    monkeypatch.setattr(ui, "new_run", lambda *args: tmp_path)
    first = AppTest.from_string(UI_SCRIPT).run()
    first.selectbox(key="api_provider").select("自定义兼容接口").run()
    first.text_input(key="api_base_url").set_value("https://custom.example/v1").run()
    first.text_input(key="api_key_custom").set_value("echo-secret").run()
    next(b for b in first.button if b.label == "获取可用模型").click().run()
    assert first.selectbox(key="api_selected_model").value == "my-model"
    next(b for b in first.button if b.label == "使用所选模型").click().run()
    assert first.text_input(key="api_model_custom").value == "my-model"
    next(b for b in first.button if b.label.startswith("测试连接")).click().run()
    assert first.success
    fake_api[1]["status"] = 401
    next(b for b in first.button if b.label.startswith("测试连接")).click().run()
    assert "鉴权失败" in first.error[0].value
    assert not first.success
    assert not first.exception
    assert "echo-secret" not in (tmp_path / "calls.jsonl").read_text(encoding="utf-8")


def test_cost_display_has_readable_precision_and_unknown():
    rows = cost_rows({"answer": {"cny_min": 0.02362772, "cny_max": 0.04725544}, "ai_judge": {}})
    assert rows[0]["费用估算区间"] == "¥0.023628—¥0.047255"
    assert rows[1]["费用估算区间"] == "未配置价格"
