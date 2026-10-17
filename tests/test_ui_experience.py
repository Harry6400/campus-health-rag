from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from campus_health.storage import config, write_json
from campus_health.ui_views import EXAMPLES, latency_rows, score_cell, summary_rows

APP = Path(__file__).resolve().parents[1] / "src/campus_health/app.py"
FIXTURE_SHA = "fixture-corpus-sha256"


@pytest.fixture
def demo(monkeypatch, tmp_path):
    """All corpus contents, answers and API checks here are synthetic test fixtures."""
    import streamlit as st

    from campus_health import api_ui, cli, data, retrieval, storage

    settings = config()
    docs = [
        {
            "id": "sleep01",
            "theme": "sleep",
            "title": "睡眠原文",
            "url": "https://example.org/sleep",
            "text": "示例测试资料",
        },
        {
            "id": "diet01",
            "theme": "diet",
            "title": "饮食原文",
            "url": "https://example.org/diet",
            "text": "示例测试资料",
        },
    ]
    monkeypatch.setattr(storage, "ROOT", tmp_path)
    monkeypatch.setattr(storage, "config", lambda: settings.copy())
    monkeypatch.setattr(data, "load_corpus", lambda: docs)
    for name in (
        "DEEPSEEK_API_KEY",
        "DEEPSEEK_BASE_URL",
        "DEEPSEEK_MODEL",
        "DEMO_NAME",
        "DEMO_STUDENT_ID",
    ):
        monkeypatch.delenv(name, raising=False)
    write_json(
        tmp_path / "data/corpus-manifest.json",
        {"version": "test-fixture", "sha256": FIXTURE_SHA},
    )
    calls = []
    api_checks = []

    class Retriever:
        def __init__(self, *args, **kwargs):
            pass

    def interactive(question, method, settings, retriever, api):
        calls.append((question, method))
        return {
            "question": question,
            "method": method,
            "generation_api": None,
            "structured": {"abstained": False, "claims": [{"text": "原文", "citations": [1]}]},
            "answer": "原文",
            "retrieval_seconds": 0.01,
            "generation_seconds": 0,
            "total_seconds": 0.01,
            "setup_seconds": 0.02,
            "usage": None,
            "cost": None,
            "evidence": [
                {
                    "title": "睡眠原文",
                    "text": "原文",
                    "source_metadata": "测试夹具",
                    "url": "https://example.org/sleep",
                    "unit_ids": ["sleep01:u0001"],
                }
            ],
        }, tmp_path / "ui-fixture"

    def unexpected_api_check(*args, **kwargs):
        api_checks.append(True)
        raise AssertionError("UI must not run model listing or connection checks automatically")

    monkeypatch.setattr(retrieval, "Retriever", Retriever)
    monkeypatch.setattr(cli, "interactive", interactive)
    monkeypatch.setattr(api_ui, "list_models", unexpected_api_check)
    monkeypatch.setattr(api_ui, "test_connection", unexpected_api_check)
    st.cache_resource.clear()
    yield tmp_path, calls, api_checks
    st.cache_resource.clear()


def submit(app):
    next(b for b in app.button if b.label in {"检索原文", "查找证据并回答"}).click().run()
    return app


def displayed_text(app):
    return "\n".join(
        item.value
        for kind in ("markdown", "caption", "info", "success", "warning", "error")
        for item in getattr(app, kind)
    )


def write_record(root, *, fingerprint=FIXTURE_SHA, exploratory=True):
    run = root / "runs/20261016T133932572718Z_exploratory-dev"
    row = {
        "method": "R3",
        "completed": 12,
        "attempted": 12,
        "evidence_recall": 1,
        "evidence_recall_n": 9,
        "keypoint_coverage": None,
        "keypoint_coverage_n": 0,
        "citation_support": 0.8,
        "citation_support_n": 8,
        "total_seconds": {"mean": 1, "median": 0.8, "p95": 2, "n": 12},
    }
    report = {
        "rows": 12,
        "expected_rows": 12,
        "errors": 0,
        "judge_errors": 2,
        "summary": [row],
        "costs": {
            "answer": {
                "responses": 12,
                "prompt_tokens": 50,
                "completion_tokens": 10,
                "cny_min": None,
                "cny_max": None,
            }
        },
    }
    write_json(run / "summary.json", report)
    spec = {"exploratory": exploratory}
    if fingerprint is not None:
        spec["corpus_sha256"] = fingerprint
    write_json(run / "evaluation.json", spec)
    return run, report


def test_initial_question_is_ready_without_api_or_automatic_run(demo):
    app = AppTest.from_file(str(APP)).run()
    assert not app.exception
    assert app.text_area(key="draft_question").value == EXAMPLES["运动"]
    assert app.selectbox(key="question_method").value == "B0"
    assert not demo[1] and not demo[2]
    submit(app)
    assert demo[1] == [(EXAMPLES["运动"], "B0")]
    assert "不生成概括" in displayed_text(app)
    assert not app.exception


def test_local_key_defaults_to_r3_without_running_or_claiming_verified(demo, monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "fixture-key-not-real")
    app = AppTest.from_file(str(APP)).run()
    assert not app.exception
    assert app.selectbox(key="question_method").value == "R3"
    assert app.text_area(key="draft_question").value == EXAMPLES["运动"]
    assert "密钥已配置" in displayed_text(app)
    assert not app.success
    assert not demo[1] and not demo[2]


def test_example_fills_without_running_and_survives_settings_change(demo):
    app = AppTest.from_file(str(APP)).run()
    app.button(key="example_睡眠").click().run()
    question = app.text_area(key="draft_question").value
    assert "睡足多少小时" in question
    assert not demo[1]
    app.text_input(key="api_model_deepseek").set_value("other-model").run()
    assert app.text_area(key="draft_question").value == question
    assert not demo[1] and not demo[2]
    submit(app)
    assert demo[1] == [(question, "B0")]
    assert not app.exception
    assert any(e.label.startswith("[1]") and e.proto.expanded for e in app.expander)


@pytest.mark.parametrize("change", ["question", "method", "model", "key", "provider"])
def test_previous_result_is_marked_when_inputs_or_api_change(demo, change):
    app = submit(AppTest.from_file(str(APP)).run())
    assert "latest_signature" in app.session_state
    if change == "question":
        app.text_area(key="draft_question").set_value("修改后的问题").run()
    elif change == "method":
        app.selectbox(key="question_method").select("R1").run()
    elif change == "model":
        app.text_input(key="api_model_deepseek").set_value("other-model").run()
    elif change == "key":
        app.text_input(key="api_key_deepseek").set_value("fixture-session-key").run()
        assert "fixture-session-key" not in repr(app.session_state["latest_signature"])
    else:
        app.selectbox(key="api_provider").select("自定义兼容接口").run()
    assert not app.exception
    assert "latest" in app.session_state
    assert len(demo[1]) == 1 and not demo[2]
    assert any("上次结果" in w.value and "尚未运行" in w.value for w in app.warning)



def test_new_submission_replaces_snapshot_and_clears_stale_warning(demo):
    app = submit(AppTest.from_file(str(APP)).run())
    old_signature = app.session_state["latest_signature"]
    app.text_area(key="draft_question").set_value("修改后的问题").run()
    assert any("上次结果" in w.value for w in app.warning)
    submit(app)
    assert not app.exception
    assert app.session_state["latest"][0]["question"] == "修改后的问题"
    assert app.session_state["latest_signature"] != old_signature
    assert not any("上次结果" in w.value for w in app.warning)
    assert demo[1] == [(EXAMPLES["运动"], "B0"), ("修改后的问题", "B0")]



def test_corpus_change_marks_old_result_and_rebuilds_only_after_submission(demo, monkeypatch):
    from campus_health import retrieval
    from campus_health.storage import read_json

    manifest_path = demo[0] / "data/corpus-manifest.json"
    builds = []

    class TrackingRetriever:
        def __init__(self, *args, **kwargs):
            builds.append(read_json(manifest_path)["sha256"])

    monkeypatch.setattr(retrieval, "Retriever", TrackingRetriever)
    app = submit(AppTest.from_file(str(APP)).run())
    assert builds == [FIXTURE_SHA]
    assert len(demo[1]) == 1
    write_json(manifest_path, {"version": "updated-test-fixture", "sha256": "new-fixture-sha256"})
    app.run()
    assert not app.exception
    assert any("上次结果" in item.value and "尚未运行" in item.value for item in app.warning)
    assert builds == [FIXTURE_SHA]
    assert len(demo[1]) == 1 and not demo[2]
    submit(app)
    assert not app.exception
    assert builds == [FIXTURE_SHA, "new-fixture-sha256"]
    assert len(demo[1]) == 2
    assert not any("上次结果" in item.value for item in app.warning)


def test_generated_answer_opens_cited_evidence_first(demo, monkeypatch):
    from campus_health import cli

    monkeypatch.setenv("DEEPSEEK_API_KEY", "fixture-key-not-real")
    original = cli.interactive

    def with_second_citation(*args, **kwargs):
        result, run = original(*args, **kwargs)
        result["structured"]["claims"][0]["citations"] = [2]
        result["evidence"].append({**result["evidence"][0], "title": "第二份测试原文"})
        return result, run

    monkeypatch.setattr(cli, "interactive", with_second_citation)
    app = submit(AppTest.from_file(str(APP)).run())
    assert not app.exception
    evidence = [item for item in app.expander if item.label.startswith(("[1]", "[2]"))]
    assert len(evidence) == 2
    assert not evidence[0].proto.expanded
    assert evidence[1].proto.expanded
    assert "回答已引用" in evidence[1].label


def test_failed_new_request_clears_old_answer_and_can_switch_to_b0(demo):
    app = submit(AppTest.from_file(str(APP)).run())
    question = app.text_area(key="draft_question").value
    assert "latest" in app.session_state
    app.selectbox(key="question_method").select("R1").run()
    submit(app)
    assert any("未运行" in e.value for e in app.error)
    assert "latest" not in app.session_state
    assert "ui_failure" in app.session_state
    assert app.text_area(key="draft_question").value == question
    assert len(demo[1]) == 1
    assert app.button(key="retry_question")
    app.button(key="switch_to_b0").click().run()
    assert app.selectbox(key="question_method").value == "B0"
    assert len(demo[1]) == 1
    assert app.text_area(key="draft_question").value == question
    submit(app)
    assert demo[1] == [(question, "B0"), (question, "B0")]
    assert "latest" in app.session_state
    assert not app.exception


def test_api_failure_can_retry_retained_question(demo, monkeypatch):
    from campus_health import cli

    monkeypatch.setenv("DEEPSEEK_API_KEY", "fixture-key-not-real")
    original = cli.interactive
    failures = []

    def fail_once(*args, **kwargs):
        if not failures:
            failures.append(True)
            raise RuntimeError("fixture request failed fixture-key-not-real")
        return original(*args, **kwargs)

    app = submit(AppTest.from_file(str(APP)).run())
    assert "latest" in app.session_state
    monkeypatch.setattr(cli, "interactive", fail_once)
    app.text_area(key="draft_question").set_value("失败后保留的问题").run()
    submit(app)
    assert "latest" not in app.session_state
    assert "ui_failure" in app.session_state
    assert any("本次未完成" in e.value for e in app.error)
    assert "fixture-key-not-real" not in displayed_text(app)
    assert app.text_area(key="draft_question").value == "失败后保留的问题"
    app.button(key="retry_question").click().run()
    assert not app.exception
    assert demo[1] == [(EXAMPLES["运动"], "R3"), ("失败后保留的问题", "R3")]
    assert "latest" in app.session_state
    assert "ui_failure" not in app.session_state



def test_refusal_without_evidence_remains_a_completed_result(demo, monkeypatch):
    from campus_health import cli

    monkeypatch.setenv("DEEPSEEK_API_KEY", "fixture-key-not-real")
    original = cli.interactive

    def refusal(*args, **kwargs):
        result, run = original(*args, **kwargs)
        result["structured"] = {"abstained": True, "claims": []}
        result["answer"] = "测试资料没有覆盖这个问题。"
        result["evidence"] = []
        return result, run

    monkeypatch.setattr(cli, "interactive", refusal)
    app = submit(AppTest.from_file(str(APP)).run())
    assert not app.exception and not app.error
    assert "latest" in app.session_state and "ui_failure" not in app.session_state
    assert any("未给出事实答案" in item.value for item in app.info)
    assert any("没有检索到原文证据" in item.value for item in app.warning)
    assert not any(button.key == "retry_question" for button in app.button)


def test_clear_session_keeps_question_and_returns_to_b0(demo):
    app = AppTest.from_file(str(APP)).run()
    app.button(key="example_睡眠").click().run()
    question = app.text_area(key="draft_question").value
    app.text_input(key="api_key_deepseek").set_value("fixture-session-key").run()
    app.selectbox(key="question_method").select("R1").run()
    submit(app)
    next(b for b in app.button if b.label == "清除会话配置").click().run()
    assert not app.exception
    assert app.selectbox(key="question_method").value == "B0"
    assert app.text_area(key="draft_question").value == question
    assert app.text_input(key="api_key_deepseek").value == ""
    assert "latest" not in app.session_state


def test_sources_filter_and_empty_state_can_reset(demo):
    app = AppTest.from_file(str(APP)).run()
    app.selectbox(key="source_theme").select("饮食").run()
    assert app.dataframe[-1].value["主题"].tolist() == ["饮食"]
    app.text_input(key="source_search").set_value("没有这篇文章").run()
    assert any("没有匹配" in e.value for e in app.info)
    app.button(key="clear_source_filters").click().run()
    assert not app.exception
    assert app.selectbox(key="source_theme").value == "全部"
    assert app.text_input(key="source_search").value == ""
    assert len(app.dataframe[-1].value) == 2
    assert not demo[1] and not demo[2]


def test_records_missing_scores_and_scoring_error_are_not_hidden(demo):
    write_record(demo[0])
    app = AppTest.from_file(str(APP)).run()
    assert not app.exception
    assert any(m.label == "辅助评分错误" and m.value == "2" for m in app.metric)
    assert app.dataframe[0].value["答案要点覆盖"].iloc[0] == "不适用 / 未评分"
    assert app.dataframe[0].value["证据召回"].iloc[0] == "100.0%（9题）"
    assert any("未配置价格" in m.value for m in app.markdown)
    assert any("不能当作正常拒答" in w.value for w in app.warning)
    assert "开发集探索" in displayed_text(app)
    assert "历史运行" in displayed_text(app)


@pytest.mark.parametrize(
    "fingerprint, expected",
    [(FIXTURE_SHA, "一致"), ("older-fixture-sha256", "资料版本不同"), (None, "无法确认")],
)
def test_record_corpus_fingerprint_status_preserves_saved_scores(demo, fingerprint, expected):
    from campus_health.storage import read_json

    run, original = write_record(demo[0], fingerprint=fingerprint)
    app = AppTest.from_file(str(APP)).run()
    assert not app.exception
    assert expected in displayed_text(app)
    if fingerprint != FIXTURE_SHA:
        assert any(expected in w.value for w in app.warning)
    assert read_json(run / "summary.json") == original
    assert app.dataframe[0].value["证据召回"].iloc[0] == "100.0%（9题）"


def test_record_corpus_fingerprint_unavailable_without_current_manifest(demo):
    write_record(demo[0])
    (demo[0] / "data/corpus-manifest.json").unlink()
    app = AppTest.from_file(str(APP)).run()
    assert not app.exception
    assert any("无法确认" in w.value for w in app.warning)
    assert any(m.label == "辅助评分错误" for m in app.metric)


def test_formal_record_is_labeled_as_saved_formal_test(demo):
    write_record(demo[0], exploratory=False)
    app = AppTest.from_file(str(APP)).run()
    assert not app.exception
    assert "正式测试记录" in displayed_text(app)
    assert "历史运行" in displayed_text(app)


def test_record_corruption_is_local_error(demo):
    run = demo[0] / "runs/broken"
    run.mkdir(parents=True)
    (run / "summary.json").write_text("not json")
    app = AppTest.from_file(str(APP)).run()
    assert not app.exception
    assert any("此记录不完整" in e.value for e in app.error)


def test_missing_corpus_preserves_records_and_shows_recovery(demo, monkeypatch):
    from campus_health import data

    write_record(demo[0])
    original_loader = data.load_corpus

    def unavailable():
        raise RuntimeError("fixture corpus unavailable")

    monkeypatch.setattr(data, "load_corpus", unavailable)
    app = AppTest.from_file(str(APP)).run()
    assert not app.exception
    assert any("资料尚未准备好" in e.value for e in app.error)
    assert any("campus-health prepare" in block.value for block in app.code)
    assert [tab.label for tab in app.tabs] == ["证据问答", "运行记录", "资料来源"]
    assert any(m.label == "辅助评分错误" for m in app.metric)
    assert not demo[1] and not demo[2]
    monkeypatch.setattr(data, "load_corpus", original_loader)
    app.run()
    assert not app.exception
    assert not any("资料尚未准备好" in e.value for e in app.error)
    assert app.text_area(key="draft_question").value == EXAMPLES["运动"]


def test_presentation_preserves_zero_and_denominators():
    assert score_cell({"faithfulness": 0, "faithfulness_n": 3}, "faithfulness") == "0.0%（3题）"
    assert score_cell({}, "faithfulness") == "不适用 / 未评分"
    row = {
        "method": "B0",
        "completed": 1,
        "attempted": 1,
        "total_seconds": {"mean": 0.1, "median": 0.1, "p95": 0.1, "n": 1},
    }
    assert summary_rows([row])[0]["方法"] == "B0 原文检索"
    assert latency_rows([row])[0]["样本数"] == 1
