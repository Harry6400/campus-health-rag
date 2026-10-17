"""Synthetic fixtures for the read-only history adapter; no saved run is regenerated."""

import hashlib
import json
from datetime import date, datetime, timezone
from pathlib import Path

import pytest

from campus_health.run_history import (
    RETRIEVAL_KIND,
    discover_runs,
    filter_runs,
    load_record,
)


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")


def make_run(root, name, *, source_kind=None, time=None, **payload):
    path = root / "runs" / name
    path.mkdir(parents=True)
    manifest = {"kind": source_kind or name.split("_", 1)[-1]}
    if time is not None:
        manifest["time"] = time
    write_json(path / "manifest.json", manifest)
    for name, value in payload.items():
        write_json(path / (name + ".json"), value)
    return path


def evaluation_payload(*, judge_errors=0):
    return {
        "summary": {
            "kind": "audit_saved_outputs_not_new_model_calls",
            "rows": 12,
            "expected_rows": 12,
            "errors": 0,
            "judge_errors": judge_errors,
            "costs": {},
            "summary": [
                {
                    "method": "R3",
                    "completed": 12,
                    "attempted": 12,
                    "evidence_recall": 0,
                    "evidence_recall_n": 9,
                    "keypoint_coverage": None,
                    "keypoint_coverage_n": 0,
                }
            ],
        },
        "evaluation": {
            "exploratory": True,
            "methods": ["R3"],
            "generation_model": "historical-model",
            "corpus_sha256": "old-corpus-sha",
        },
    }


def interactive_payload():
    return {
        "question": "怎样在日常生活中增加活动？",
        "method": "B0",
        "answer": "测试原文",
        "structured": {"abstained": False, "claims": []},
        "evidence": [],
        "retrieval_seconds": 0.1,
        "generation_seconds": 0,
        "total_seconds": 0.1,
        "setup_seconds": 0.2,
        "usage": None,
        "cost": None,
    }


def test_retrieval_only_summary_without_evaluation_is_supported(tmp_path):
    summary = {
        "kind": RETRIEVAL_KIND,
        "rows": 90,
        "question_count": 30,
        "top_k_values": [2, 4, 6],
        "api_called": False,
        "paid_api_calls": 0,
        "top_k_selected": False,
        "selection_blocker": "Only retrieval was run.",
        "corpus_sha256": "retrieval-corpus-sha",
        "summary": [
            {
                "method": f"R3 retrieval only k={k}",
                "attempted": 30,
                "completed": 30,
                "failed": 0,
                "evidence_recall": 1,
                "evidence_recall_n": 24,
                "generation_seconds": None,
                "keypoint_coverage": None,
            }
            for k in (2, 4, 6)
        ],
    }
    path = make_run(
        tmp_path,
        "20261017T163710521764Z_candidate-dev-r3-retrieval-only-no-api",
        summary=summary,
    )
    record = discover_runs(tmp_path)[0]
    assert record.kind == "retrieval" and record.status == "saved"
    assert record.methods == ("R3",)
    assert record.corpus_sha == "retrieval-corpus-sha"
    assert not record.issues
    loaded = load_record(record)
    assert loaded["summary"] == summary
    assert "evaluation" not in loaded
    assert not (path / "evaluation.json").exists()


def test_complete_evaluation_preserves_scores_and_judge_failures(tmp_path):
    payload = evaluation_payload(judge_errors=2)
    make_run(tmp_path, "20261016T133932572718Z_exploratory-dev", **payload)
    record = discover_runs(tmp_path)[0]
    assert record.kind == "evaluation" and record.status == "errors"
    assert record.model == "historical-model"
    assert record.corpus_sha == "old-corpus-sha"
    loaded = load_record(record)
    assert loaded["summary"] == payload["summary"]
    row = loaded["summary"]["summary"][0]
    assert row["evidence_recall"] == 0 and row["evidence_recall_n"] == 9
    assert row["keypoint_coverage"] is None
    assert loaded["summary"]["judge_errors"] == 2


def test_missing_evaluation_never_becomes_retrieval_only(tmp_path):
    payload = evaluation_payload()
    payload.pop("evaluation")
    make_run(tmp_path, "20261016T133932572718Z_exploratory-dev", **payload)
    record = discover_runs(tmp_path)[0]
    assert record.kind == "evaluation" and record.status == "incomplete"
    assert any("evaluation.json" in issue for issue in record.issues)


def test_retrieval_directory_name_cannot_replace_explicit_summary_kind(tmp_path):
    make_run(
        tmp_path,
        "20261017T163710521764Z_candidate-dev-r3-retrieval-only-no-api",
        summary={"rows": 90, "summary": [{"method": "R3"}]},
    )
    record = discover_runs(tmp_path)[0]
    assert record.kind == "unknown" and record.status == "unknown"


@pytest.mark.parametrize("field", ["rows", "expected_rows", "errors", "judge_errors", "costs"])
def test_evaluation_requires_minimum_summary_fields(tmp_path, field):
    payload = evaluation_payload()
    del payload["summary"][field]
    make_run(tmp_path, "20261016T133932572718Z_exploratory-dev", **payload)
    record = discover_runs(tmp_path)[0]
    assert record.kind == "evaluation" and record.status == "incomplete"
    assert any(field in issue for issue in record.issues)


def test_early_interactive_result_needs_no_new_optional_api_fields(tmp_path):
    result = interactive_payload()
    make_run(tmp_path, "20261016T132127906669Z_interactive-not-evaluation", result=result)
    record = discover_runs(tmp_path)[0]
    assert record.kind == "interactive" and record.status == "saved"
    assert record.question == result["question"]
    assert record.methods == ("B0",)
    assert record.model == "" and record.corpus_sha is None
    assert load_record(record)["result"] == result


def test_corrupt_and_unknown_directories_remain_discoverable(tmp_path):
    broken = make_run(tmp_path, "20261017T163710521764Z_exploratory-dev")
    (broken / "summary.json").write_text("not json")
    unknown = tmp_path / "runs/unrecognized-directory"
    unknown.mkdir()
    (unknown / "notes.txt").write_text("fixture")
    records = discover_runs(tmp_path)
    assert len(records) == 2
    assert records[0].kind == "evaluation" and records[0].status == "incomplete"
    assert any("summary.json" in issue and "无法读取" in issue for issue in records[0].issues)
    assert records[1].kind == "unknown" and records[1].status == "unknown"
    assert filter_runs(records) == [records[0]]
    assert filter_runs(records, kind="全部") == records


@pytest.mark.parametrize("filename", ["summary.json", "evaluation.json", "manifest.json"])
def test_corrupt_evaluation_artifacts_do_not_break_other_records(tmp_path, filename):
    broken = make_run(
        tmp_path, "20261017T163710521764Z_exploratory-dev", **evaluation_payload()
    )
    (broken / filename).write_text("{")
    good = make_run(
        tmp_path, "20261016T132127906669Z_interactive-not-evaluation", result=interactive_payload()
    )
    records = discover_runs(tmp_path)
    assert {record.path for record in records} == {broken, good}
    assert records[0].status == "incomplete"
    assert any(filename in issue for issue in records[0].issues)
    assert records[1].status == "saved"


def test_process_payloads_show_known_errors_without_inventing_success(tmp_path):
    make_run(tmp_path, "20261016T131713645688Z_question-draft", validation={"count": 138, "errors": ["fixture"]})
    make_run(tmp_path, "20261017T164303799693Z_doctor", result={"api_key_configured": True})
    only_manifest = make_run(tmp_path, "20261017T164400000000Z_connection-check-not-evaluation")
    (only_manifest / "calls.jsonl").write_text("not parsed as a call log")
    records = {record.source_kind: record for record in discover_runs(tmp_path)}
    assert records["question-draft"].status == "errors"
    assert records["doctor"].status == "saved"
    assert records["connection-check-not-evaluation"].status == "incomplete"
    assert all(record.kind == "process" for record in records.values())
    assert "calls" not in load_record(records["connection-check-not-evaluation"])


def test_filter_searches_question_model_category_and_id_case_insensitively(tmp_path):
    make_run(tmp_path, "20261016T133932572718Z_exploratory-dev", **evaluation_payload())
    make_run(tmp_path, "20261016T132127906669Z_interactive-not-evaluation", result=interactive_payload())
    records = discover_runs(tmp_path)
    assert len(filter_runs(records, keyword="HISTORICAL-MODEL")) == 1
    assert len(filter_runs(records, keyword="增加活动")) == 1
    assert len(filter_runs(records, keyword="批量评测")) == 1
    assert len(filter_runs(records, keyword="  133932572718z  ")) == 1
    assert not filter_runs(records, keyword="not present")
    assert filter_runs(records, kind="evaluation", method="R3") == [records[0]]
    assert not filter_runs(records, kind="interactive", method="R3")
    assert filter_runs(records, status="已保存") == records


@pytest.mark.parametrize("category", ["evaluation", "retrieval"])
def test_batch_question_search_extracts_only_question_fields_without_writes(
    tmp_path, monkeypatch, category
):
    if category == "evaluation":
        payload = evaluation_payload()
        source_kind, filename = "exploratory-dev", "results.jsonl"
    else:
        payload = {
            "summary": {
                "kind": RETRIEVAL_KIND,
                "rows": 1,
                "question_count": 1,
                "api_called": False,
                "top_k_values": [4],
                "summary": [{"method": "R3 retrieval only k=4"}],
            }
        }
        source_kind, filename = "candidate-dev-r3-retrieval-only-no-api", "retrieval-results.jsonl"
    path = make_run(tmp_path, "20261017T163710521764Z_" + source_kind, **payload)
    line = {
        "question": "历史批量问题：怎样减少久坐？",
        "question_id": "BATCH-QUESTION-01",
        "id": "retrieval-question-01",
        "answer": "not-searchable-answer",
        "evidence": [{"text": "not-searchable-evidence"}],
    }
    (path / filename).write_text(json.dumps(line, ensure_ascii=False) + "\n")
    (path / "calls.jsonl").write_text("private-call-log")
    before = {file: file.read_bytes() for file in path.iterdir()}
    original_open = Path.open
    reads = []

    def read_only(file, mode="r", *args, **kwargs):
        assert mode in {"r", "rb"}
        assert file.name != "calls.jsonl"
        reads.append(file.name)
        return original_open(file, mode, *args, **kwargs)

    with monkeypatch.context() as guard:
        guard.setattr(Path, "open", read_only)
        records = discover_runs(tmp_path)
        assert filter_runs(records, keyword="减少久坐") == records
        assert filter_runs(records, keyword="batch-question-01") == records
        assert filter_runs(records, keyword="retrieval-question-01") == records
        assert not filter_runs(records, keyword="not-searchable-answer")
        assert not filter_runs(records, keyword="not-searchable-evidence")
        assert filename in reads
        assert "results" not in load_record(records[0])
    assert before == {file: file.read_bytes() for file in path.iterdir()}


def test_bad_question_jsonl_is_local_and_preserves_other_searchable_rows(tmp_path):
    path = make_run(tmp_path, "20261017T163710521764Z_exploratory-dev", **evaluation_payload())
    (path / "results.jsonl").write_text(
        json.dumps({"question": "仍可搜索的历史问题"}, ensure_ascii=False) + "\n{\n[]\n"
    )
    make_run(tmp_path, "20261016T132127906669Z_interactive-not-evaluation", result=interactive_payload())
    records = discover_runs(tmp_path)
    assert len(records) == 2
    assert records[0].status == "incomplete" and records[1].status == "saved"
    assert filter_runs(records, keyword="仍可搜索") == [records[0]]
    assert any("results.jsonl" in issue and "第 2 行" in issue for issue in records[0].issues)
    assert any("results.jsonl" in issue and "第 3 行" in issue for issue in records[0].issues)


def test_status_filters_keep_saved_errors_incomplete_and_unknown_distinct(tmp_path):
    make_run(tmp_path, "20261016T130000000000Z_exploratory-dev", **evaluation_payload())
    make_run(tmp_path, "20261016T140000000000Z_exploratory-dev", **evaluation_payload(judge_errors=1))
    make_run(tmp_path, "20261016T150000000000Z_exploratory-dev")
    (tmp_path / "runs/unknown").mkdir()
    records = discover_runs(tmp_path)
    for status in ("saved", "errors", "incomplete", "unknown"):
        filtered = filter_runs(records, kind="all", status=status)
        assert len(filtered) == 1 and filtered[0].status == status


def test_filter_date_bounds_are_inclusive_beijing_calendar_days(tmp_path):
    make_run(tmp_path, "before", source_kind="doctor", time="2026-10-16T15:59:59+00:00", result={})
    make_run(tmp_path, "first", source_kind="doctor", time="2026-10-16T16:00:00+00:00", result={})
    make_run(tmp_path, "last", source_kind="doctor", time="2026-10-17T15:59:59+00:00", result={})
    make_run(tmp_path, "after", source_kind="doctor", time="2026-10-17T16:00:00+00:00", result={})
    make_run(tmp_path, "unknown", source_kind="doctor", result={})
    records = discover_runs(tmp_path)
    assert records[-1].id == "unknown"
    assert all(record.started_at.tzinfo == timezone.utc for record in records[:-1])
    filtered = filter_runs(records, kind="全部", start_date=date(2026, 10, 17), end_date=date(2026, 10, 17))
    assert [record.id for record in filtered] == ["last", "first"]
    assert not filter_runs(records, kind="全部", start_date=date(2026, 10, 18), end_date=date(2026, 10, 16))


def test_directory_timestamp_is_utc_fallback_for_missing_or_naive_time(tmp_path):
    make_run(tmp_path, "20261017T163710521764Z_doctor", time="2026-10-01T00:00:00", result={})
    record = discover_runs(tmp_path / "runs")[0]
    assert record.started_at == datetime(2026, 10, 17, 16, 37, 10, 521764, tzinfo=timezone.utc)


def test_load_reports_artifact_corruption_after_discovery(tmp_path):
    path = make_run(tmp_path, "20261016T133932572718Z_exploratory-dev", **evaluation_payload())
    record = discover_runs(tmp_path)[0]
    (path / "summary.json").write_text("{")
    loaded = load_record(record)
    assert loaded["summary"] is None
    assert any("summary.json" in issue for issue in loaded["issues"])


def test_history_reads_no_call_logs_or_secrets_and_never_writes(tmp_path, monkeypatch):
    path = make_run(tmp_path, "20261016T132127906669Z_interactive-not-evaluation", result=interactive_payload())
    (path / "calls.jsonl").write_text("private-call-body")
    (path / ".env").write_text("PRIVATE=not-for-ui")
    write_json(tmp_path / "runs/development-budget-20261018.json", {"spent_cny": "0.1"})
    before = {file: hashlib.sha256(file.read_bytes()).hexdigest() for file in tmp_path.rglob("*") if file.is_file()}
    original = Path.read_text
    reads = []

    def tracked_read(file, *args, **kwargs):
        reads.append(file.name)
        assert file.name not in {"calls.jsonl", ".env", "development-budget-20261018.json"}
        return original(file, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", tracked_read)
    records = discover_runs(tmp_path)
    assert len(records) == 1
    assert "calls.jsonl" in records[0].artifacts
    load_record(records[0])
    after = {file: hashlib.sha256(file.read_bytes()).hexdigest() for file in tmp_path.rglob("*") if file.is_file()}
    assert before == after and reads


def test_missing_runs_and_invalid_nested_values_are_local(tmp_path):
    assert discover_runs(tmp_path) == []
    result = interactive_payload()
    result["method"] = {"invalid": "shape"}
    result["status"] = ["invalid"]
    make_run(tmp_path, "20261016T132127906669Z_interactive-not-evaluation", result=result)
    record = discover_runs(tmp_path)[0]
    assert record.kind == "interactive" and record.status == "incomplete"
    assert any("method" in issue for issue in record.issues)
