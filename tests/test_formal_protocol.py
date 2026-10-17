from __future__ import annotations

import json
from pathlib import Path

import pytest

from campus_health import evaluation
from campus_health.api_config import DEEPSEEK_URL, APIConfig
from campus_health.storage import append_jsonl, digest, write_json


def _approved_protocol(root: Path) -> dict:
    identity_path = root / evaluation.REVIEWER_IDENTITY_PATH
    identity_path.parent.mkdir(parents=True, exist_ok=True)
    scopes = sorted(evaluation.REQUIRED_REVIEW_SCOPES)
    identities = {
        "reviewers": [
            {
                "reviewer_id": f"R{number}",
                "name": f"Student {number}",
                "role": "student_team_member",
                "confirmed_at": f"2026-10-17T0{number}:00:00+08:00",
                "confirmation_text": "I reviewed and approve the listed scope.",
            }
            for number in [1, 2]
        ]
    }
    write_json(identity_path, identities)
    return {
        "status": "approved",
        "reviewers": [
            {
                "reviewer_id": f"R{number}",
                "reviewed_at": f"2026-10-17T0{number}:00:00+08:00",
                "scope": scopes,
            }
            for number in [1, 2]
        ],
        "reviewer_identity_record": {
            "path": evaluation.REVIEWER_IDENTITY_PATH.as_posix(),
            "sha256": digest(identity_path.read_bytes()),
        },
        "questions_sha256": "qset",
        "answer_prompt_sha256": "answer-prompt",
    }


def _review_sample_fixture(tmp_path: Path) -> tuple[Path, dict]:
    questions = []
    entries = []
    for theme in ["sleep", "diet", "exercise"]:
        for kind in ["direct", "paraphrase", "multi", "unanswerable"]:
            for index in range(2):
                question_id = f"{theme}-{kind}-{index}"
                question = {
                    "id": question_id,
                    "question": f"{theme} {kind} question {index}",
                    "theme": theme,
                    "kind": kind,
                    "split": "test",
                    "answerable": kind != "unanswerable",
                }
                questions.append(question)
                double = (kind == "direct" and index == 0) or (
                    kind == "unanswerable" and index == 0
                )
                if double:
                    reviewers = ["R1", "R2"]
                elif kind in {"direct", "unanswerable"}:
                    reviewers = ["R1"] if kind == "direct" else ["R2"]
                elif kind == "paraphrase":
                    reviewers = ["R1"] if index == 0 else ["R2"]
                else:
                    reviewers = ["R2"] if index == 0 else ["R1"]
                entries.append(
                    {
                        "question_id": question_id,
                        "theme": theme,
                        "kind": kind,
                        "answerable": question["answerable"],
                        "question_sha256": digest(question["question"]),
                        "double_review": double,
                        "reviewer_ids": reviewers,
                    }
                )
    path = tmp_path / "questions.jsonl"
    payload = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in questions)
    path.write_text(payload, encoding="utf-8")
    return path, {
        "questions_sha256": digest(path.read_bytes()),
        "unique_question_count": 24,
        "double_review_count": 6,
        "questions": entries,
    }


def test_protocol_requires_actual_group_review_and_frozen_fingerprints(tmp_path, monkeypatch):
    monkeypatch.setattr(evaluation, "ROOT", tmp_path)
    protocol = _approved_protocol(tmp_path)
    protocol_path = tmp_path / "review" / "protocol.json"
    write_json(protocol_path, protocol)
    monkeypatch.setattr(
        evaluation,
        "protocol_fingerprints",
        lambda settings, question_file, api: {
            "questions_sha256": "qset",
            "answer_prompt_sha256": "answer-prompt",
        },
    )
    api = APIConfig("deepseek", DEEPSEEK_URL, "deepseek-flash", "test-secret")

    assert evaluation.verify_protocol({}, tmp_path / "questions.jsonl", api) == protocol

    protocol["answer_prompt_sha256"] = "stale"
    write_json(protocol_path, protocol)
    with pytest.raises(ValueError, match="answer_prompt_sha256"):
        evaluation.verify_protocol({}, tmp_path / "questions.jsonl", api)


def test_protocol_rejects_missing_or_single_reviewer(tmp_path, monkeypatch):
    monkeypatch.setattr(evaluation, "ROOT", tmp_path)
    protocol = _approved_protocol(tmp_path)
    protocol["reviewers"] = protocol["reviewers"][:1]
    with pytest.raises(ValueError, match="至少要有两名"):
        evaluation._verify_human_approvals(protocol)


def test_protocol_rejects_identity_record_tampering(tmp_path, monkeypatch):
    monkeypatch.setattr(evaluation, "ROOT", tmp_path)
    protocol = _approved_protocol(tmp_path)
    identity_path = tmp_path / evaluation.REVIEWER_IDENTITY_PATH
    identities = json.loads(identity_path.read_text())
    identities["reviewers"][0]["confirmation_text"] = "changed"
    write_json(identity_path, identities)
    with pytest.raises(ValueError, match="哈希不符"):
        evaluation._verify_human_approvals(protocol)


def test_protocol_rejects_custom_endpoint_and_wrong_requested_model(tmp_path, monkeypatch):
    monkeypatch.setattr(evaluation, "ROOT", tmp_path)
    protocol = _approved_protocol(tmp_path)
    write_json(tmp_path / "review" / "protocol.json", protocol)
    with pytest.raises(ValueError, match="DeepSeek官方"):
        evaluation.verify_protocol(
            {}, tmp_path / "questions.jsonl", APIConfig("custom", "https://example.org", "deepseek-flash", "x")
        )
    with pytest.raises(ValueError, match="deepseek-flash"):
        evaluation.verify_protocol(
            {}, tmp_path / "questions.jsonl", APIConfig("deepseek", DEEPSEEK_URL, "deepseek-pro", "x")
        )


def test_protocol_must_exist_before_formal_run(tmp_path, monkeypatch):
    monkeypatch.setattr(evaluation, "ROOT", tmp_path)
    with pytest.raises(RuntimeError, match="缺少小组实际审核记录"):
        evaluation.verify_protocol(
            {},
            tmp_path / "questions.jsonl",
            APIConfig("deepseek", DEEPSEEK_URL, "deepseek-flash", "x"),
        )


def test_formal_run_rejects_subset_methods_before_protocol_or_api(tmp_path, monkeypatch):
    monkeypatch.setattr(evaluation, "load_corpus", lambda: [])
    monkeypatch.setattr(evaluation, "read_jsonl", lambda _: [{"split": "test", "theme": "sleep"}])
    monkeypatch.setattr(evaluation, "validate_questions", lambda *_: [])
    monkeypatch.setattr(
        evaluation,
        "verify_protocol",
        lambda *_: pytest.fail("partial methods must be rejected before protocol verification"),
    )
    with pytest.raises(ValueError, match="B0/R1/R2/R3"):
        evaluation.run_evaluation(
            {}, tmp_path / "questions.jsonl", methods=["R3"], exploratory=False
        )


def test_formal_run_rejects_disabled_ai_judging(tmp_path, monkeypatch):
    monkeypatch.setattr(evaluation, "load_corpus", lambda: [])
    monkeypatch.setattr(evaluation, "read_jsonl", lambda _: [{"split": "test", "theme": "sleep"}])
    monkeypatch.setattr(evaluation, "validate_questions", lambda *_: [])
    monkeypatch.setattr(
        evaluation,
        "verify_protocol",
        lambda *_: pytest.fail("missing AI scores must be rejected before protocol verification"),
    )
    with pytest.raises(ValueError, match="必须保留全部AI评分"):
        evaluation.run_evaluation(
            {}, tmp_path / "questions.jsonl", ai_judge=False, exploratory=False
        )


def test_protocol_fingerprints_never_include_api_key(tmp_path, monkeypatch):
    monkeypatch.setattr(evaluation, "ROOT", tmp_path)
    (tmp_path / "data").mkdir()
    (tmp_path / "review").mkdir()
    (tmp_path / "data" / "corpus-manifest.json").write_text(
        json.dumps({"sha256": "corpus"}), encoding="utf-8"
    )
    (tmp_path / "data" / "corpus.jsonl").write_text("{}\n", encoding="utf-8")
    (tmp_path / evaluation.REVIEW_SAMPLE_PATH).write_text("{}\n", encoding="utf-8")
    (tmp_path / "pyproject.toml").write_text("", encoding="utf-8")
    questions = tmp_path / "questions.jsonl"
    questions.write_text("{}\n", encoding="utf-8")
    monkeypatch.setattr(
        evaluation,
        "provenance",
        lambda: {"files": {}, "packages": {"openai": "2.0"}},
    )
    monkeypatch.setattr(evaluation, "validate_review_sample", lambda *_: None)
    api = APIConfig("deepseek", DEEPSEEK_URL, "deepseek-flash", "do-not-persist-this-key")

    fingerprints = evaluation.protocol_fingerprints(
        {"max_output_tokens": 700, "temperature": 0, "embedding_device": "cpu"},
        questions,
        api,
    )

    assert fingerprints["api"] == {
        "provider": "deepseek",
        "base_url": DEEPSEEK_URL,
        "model": "deepseek-flash",
    }
    assert "do-not-persist-this-key" not in json.dumps(fingerprints, ensure_ascii=False)


def test_review_sample_has_fixed_quotas_and_balanced_double_review(tmp_path):
    question_file, sample = _review_sample_fixture(tmp_path)

    evaluation.validate_review_sample(question_file, sample)

    sample["questions"][0]["question_sha256"] = "stale"
    with pytest.raises(ValueError, match="元数据不匹配"):
        evaluation.validate_review_sample(question_file, sample)


def test_audit_distinguishes_row_complete_from_error_free_completion(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    write_json(
        run / "evaluation.json",
        {"question_ids": ["q1"], "methods": ["B0"], "ai_judge": True},
    )
    append_jsonl(
        run / "results.jsonl",
        {
            "question_id": "q1",
            "method": "B0",
            "status": "ok",
            "judge_status": "error",
            "metrics": {},
        },
    )

    summary = evaluation.audit(run)

    assert summary["complete"] is True
    assert summary["errors"] == 0
    assert summary["judge_errors"] == 1
    assert summary["successful_complete"] is False


def test_audit_detects_missing_expected_output(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    write_json(
        run / "evaluation.json",
        {"question_ids": ["q1", "q2"], "methods": ["B0"], "ai_judge": False},
    )
    append_jsonl(
        run / "results.jsonl",
        {"question_id": "q1", "method": "B0", "status": "ok", "metrics": {}},
    )

    summary = evaluation.audit(run)

    assert summary["complete"] is False
    assert summary["missing_outputs"] == [["q2", "B0"]]
    assert summary["successful_complete"] is False
