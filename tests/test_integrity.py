"""边界/计分/协议检查；所有夹具仅用于测试，不充当实验数据。"""

import os
import subprocess
import sys

import pytest

from campus_health.data import canonical_units, normalize, parse_article
from campus_health.evaluation import run_evaluation, validate_questions
from campus_health.llm import ModelOutputError, estimate_cost, validate_answer
from campus_health.metrics import retrieval_metrics, summarize
from campus_health.retrieval import build_chunks, tokenize


def test_script_normalization_keeps_numbers_and_negation():
    assert normalize("不應 每週150分鐘、7.5小時") == "不应 每周150分钟、7.5小时"


def test_char2_does_not_join_across_punctuation_or_change_numeric_unit():
    assert tokenize("运动，睡眠150", "char2") == ["运动", "睡眠", "150"]


def test_recall_requires_all_groups_but_accepts_alternative_and_cross_chunk_union():
    q = {"answerable": True, "evidence_groups": [[["a"], ["b"]], [["c", "d"]]]}
    r = retrieval_metrics(q, [{"unit_ids": ["b", "c"]}, {"unit_ids": ["x"]}])
    assert r == {"evidence_recall": 0.5, "evidence_precision": 0.5}
    assert (
        retrieval_metrics(q, [{"unit_ids": ["a", "c"]}, {"unit_ids": ["d"]}])["evidence_recall"]
        == 1
    )
    assert retrieval_metrics({"answerable": False}, [])["evidence_recall"] is None


@pytest.mark.parametrize("citations", [[0], [2], [True], ["1"], [[1]], [1, 1], []])
def test_invalid_citations_are_not_silently_repaired(citations):
    with pytest.raises(ModelOutputError):
        validate_answer(
            {"abstained": False, "claims": [{"text": "测试事实", "citations": citations}]}, 1
        )


def test_abstention_cannot_hide_claims():
    with pytest.raises(ModelOutputError):
        validate_answer({"abstained": True, "reason": "不足", "claims": [{"text": "仍给事实"}]}, 1)


def test_summary_keeps_failure_denominator_and_unscored_as_missing():
    rows = [
        {"method": "R1", "status": "ok", "metrics": {"evidence_recall": 1}},
        {"method": "R1", "status": "error", "metrics": {"evidence_recall": 0}},
    ]
    s = summarize(rows)[0]
    assert s["attempted"] == 2 and s["failed"] == 1 and s["completion_rate"] == 0.5
    assert s["evidence_recall"] == 0.5
    assert s["faithfulness"] is None and s["faithfulness_n"] == 0


def test_no_price_is_unknown_instead_of_free():
    assert estimate_cost({"prompt_tokens": 10}, {})["cny_min"] is None


def test_price_uses_actual_hit_miss_output_separately():
    price = {
        "low": {"cached_input": 1, "input": 2, "output": 4},
        "high": {"cached_input": 2, "input": 4, "output": 8},
    }
    c = estimate_cost(
        {
            "prompt_tokens": 1000,
            "prompt_cache_hit_tokens": 600,
            "prompt_cache_miss_tokens": 400,
            "completion_tokens": 100,
        },
        {"prices_cny_per_million": price},
    )
    assert c["cny_min"] == pytest.approx(0.0018)
    assert c["cny_max"] == pytest.approx(0.0036)


def test_error_page_cannot_become_corpus():
    with pytest.raises(ValueError, match="选择器"):
        parse_article(
            "<html><h1>错误回报</h1></html>", {"id": "test", "url": "https://www.mohw.gov.tw/test"}
        )


def test_chunk_budgets_preserve_canonical_evidence_ids():
    class ToyTokenizer:
        model_max_length = 100

        def encode(self, text, add_special_tokens=True):
            return list(text) + ([0, 0] if add_special_tokens else [])

    doc = {
        "id": "d",
        "title": "题",
        "theme": "sleep",
        "url": "https://example.org",
        "units": canonical_units("d", ["甲乙。丙丁。戊己。庚辛。"]),
    }
    for budget in [10, 16]:
        chunks = build_chunks([doc], {"chunk_tokens": budget, "overlap_tokens": 3}, ToyTokenizer())
        assert {u for c in chunks for u in c["unit_ids"]} == {u["id"] for u in doc["units"]}
        assert all(len(ToyTokenizer().encode("题\n" + c["text"])) <= budget for c in chunks)


def test_exploration_cannot_run_test_split(tmp_path):
    with pytest.raises(ValueError, match="只允许dev"):
        run_evaluation({}, tmp_path / "absent.jsonl", split="test", exploratory=True)


def test_question_checks_unknown_evidence_and_cross_split_family():
    base = {
        "theme": "sleep",
        "kind": "direct",
        "question": "测试问题",
        "group_id": "same",
        "answerable": True,
        "keypoints": ["测试"],
        "evidence_groups": [[["missing"]]],
    }
    rows = [
        {**base, "id": "a", "split": "dev"},
        {**base, "id": "b", "split": "test", "question": "换一种说法"},
    ]
    errors = validate_questions(rows, [{"units": [{"id": "present"}]}])
    assert any("不存在" in e for e in errors)
    assert any("跨开发/测试" in e for e in errors)


def test_cli_help_works_when_redirected_with_windows_legacy_encoding():
    env = {**os.environ, "PYTHONIOENCODING": "cp1252"}
    completed = subprocess.run(
        [sys.executable, "-m", "campus_health.cli", "--help"],
        env=env,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr.decode("utf-8", errors="replace")
    assert "大学生日常健康知识" in completed.stdout.decode("utf-8")
