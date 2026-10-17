from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from campus_health.storage import config, write_json
from campus_health.ui_views import latency_rows, score_cell, summary_rows

APP = Path(__file__).resolve().parents[1] / "src/campus_health/app.py"


@pytest.fixture
def demo(monkeypatch, tmp_path):
    import streamlit as st

    from campus_health import cli, data, retrieval, storage

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
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    monkeypatch.delenv("DEMO_NAME", raising=False)
    monkeypatch.delenv("DEMO_STUDENT_ID", raising=False)
    write_json(tmp_path / "data/corpus-manifest.json", {"version": "test-fixture"})
    calls = []

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

    monkeypatch.setattr(retrieval, "Retriever", Retriever)
    monkeypatch.setattr(cli, "interactive", interactive)
    st.cache_resource.clear()
    yield tmp_path, calls
    st.cache_resource.clear()


def test_example_fills_without_running_and_survives_settings_change(demo):
    app = AppTest.from_file(str(APP)).run()
    assert not app.exception
    assert app.selectbox(key="question_method").value == "B0"
    app.button(key="example_睡眠").click().run()
    question = app.text_area(key="draft_question").value
    assert "睡足多少小时" in question
    assert not demo[1]
    app.text_input(key="api_model_deepseek").set_value("other-model").run()
    assert app.text_area(key="draft_question").value == question
    assert not demo[1]
    next(b for b in app.button if b.label == "检索原文").click().run()
    assert demo[1] == [(question, "B0")]
    assert not app.exception
    assert any(e.label.startswith("[1]") and e.proto.expanded for e in app.expander)


def test_failed_new_request_clears_old_answer(demo):
    app = AppTest.from_file(str(APP)).run()
    app.button(key="example_睡眠").click().run()
    next(b for b in app.button if b.label == "检索原文").click().run()
    assert "latest" in app.session_state
    app.selectbox(key="question_method").select("R1").run()
    next(b for b in app.button if b.label == "查找证据并回答").click().run()
    assert any("未运行" in e.value for e in app.error)
    assert "latest" not in app.session_state
    assert len(demo[1]) == 1
    next(b for b in app.button if b.label == "清除会话配置").click().run()
    assert app.selectbox(key="question_method").value == "B0"
    assert app.text_area(key="draft_question").value


def test_sources_filter_and_empty_state(demo):
    app = AppTest.from_file(str(APP)).run()
    app.selectbox(key="source_theme").select("饮食").run()
    assert app.dataframe[-1].value["主题"].tolist() == ["饮食"]
    app.text_input(key="source_search").set_value("没有这篇文章").run()
    assert any("没有匹配" in e.value for e in app.info)
    assert not app.exception


def test_records_missing_scores_and_scoring_error_are_not_hidden(demo):
    run = demo[0] / "runs/20261016T133932572718Z_exploratory-dev"
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
    write_json(
        run / "summary.json",
        {
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
        },
    )
    write_json(run / "evaluation.json", {"exploratory": True})
    app = AppTest.from_file(str(APP)).run()
    assert not app.exception
    assert any(m.label == "辅助评分错误" and m.value == "2" for m in app.metric)
    assert app.dataframe[0].value["答案要点覆盖"].iloc[0] == "不适用 / 未评分"
    assert app.dataframe[0].value["证据召回"].iloc[0] == "100.0%（9题）"
    assert any("未配置价格" in m.value for m in app.markdown)
    assert any("不能当作正常拒答" in w.value for w in app.warning)


def test_record_corruption_is_local_error(demo):
    run = demo[0] / "runs/broken"
    run.mkdir(parents=True)
    (run / "summary.json").write_text("not json")
    app = AppTest.from_file(str(APP)).run()
    assert not app.exception
    assert any("此记录不完整" in e.value for e in app.error)


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
