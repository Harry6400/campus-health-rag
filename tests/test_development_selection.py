"""模拟选参流程；夹具不是实际实验数据，不调用API。"""

import pytest

from campus_health import experiments
from campus_health.storage import read_json, write_json


def test_choose_uses_priority_then_tie_break_and_rejects_missing():
    rows = [
        {"value": 6, "summary": {"a": 1, "a_n": 6, "b": 0.5, "b_n": 24}},
        {"value": 2, "summary": {"a": 1, "a_n": 6, "b": 0.5, "b_n": 24}},
        {"value": 4, "summary": {"a": 0.5, "a_n": 6, "b": 1, "b_n": 24}},
    ]
    assert experiments.choose(rows, ["a", "b"], lambda v: -v)["value"] == 2
    rows[0]["summary"]["b"] = None
    with pytest.raises(ValueError, match="不可比较"):
        experiments.choose(rows, ["a", "b"], lambda v: -v)


@pytest.mark.parametrize("fail", [False, True])
def test_sweep_propagates_selected_tokenizer_and_stops_on_error(tmp_path, monkeypatch, fail):
    monkeypatch.setattr(experiments, "new_run", lambda *a: tmp_path)
    calls = []

    def fake_run(settings, questions, split, limit, methods, exploratory, judge, **kwargs):
        assert split == "dev" and exploratory and kwargs["stop_on_error"]
        calls.append((settings, methods, judge))
        path = tmp_path / str(len(calls))
        summary = {}
        for metric in ["evidence_recall", "evidence_precision", "correct_refusal",
                       "keypoint_coverage", "faithfulness", "citation_support"]:
            summary[metric] = 1 if settings["tokenizer"] == "char2" else 0.5
            summary[metric + "_n"] = 24
        write_json(path / "summary.json", {
            "complete": True, "errors": int(fail), "judge_errors": 0, "summary": [summary],
        })
        return path

    monkeypatch.setattr(experiments, "run_evaluation", fake_run)
    if fail:
        with pytest.raises(ValueError, match="不完整"):
            experiments.sweep({"top_k": 4}, tmp_path / "questions", None, True)
        assert len(calls) == 1
        assert (tmp_path / "stopped.json").exists()
    else:
        experiments.sweep({"top_k": 4}, tmp_path / "questions", None, True)
        assert len(calls) == 5
        assert all(c[0]["tokenizer"] == "char2" for c in calls[2:])
        assert [c[0]["top_k"] for c in calls[2:]] == [2, 4, 6]
        assert all(not c[2] for c in calls[:2])
        assert read_json(tmp_path / "selection.json")["top_k"] == 2
