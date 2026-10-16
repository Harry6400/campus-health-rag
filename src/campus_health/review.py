from __future__ import annotations

import csv
import json
from pathlib import Path

from .data import load_corpus
from .storage import ROOT, digest, read_json, read_jsonl, write_json


def export_review(run: Path) -> Path:
    """仅导出待填表，不替小组填写判断、身份或时间。"""
    rows = read_jsonl(run / "results.jsonl")
    if not rows:
        raise ValueError("没有真实输出，不能制作核验表")
    out = ROOT / "review" / "private" / run.name
    out.mkdir(parents=True, exist_ok=True)
    with (out / "outputs-to-review.csv").open("w", encoding="utf-8-sig", newline="") as f:
        keys = [
            "question_id",
            "method",
            "question",
            "status",
            "answer",
            "evidence",
            "reviewer",
            "reviewed_at",
            "keypoint_coverage",
            "faithfulness",
            "citation_support",
            "error_category",
            "comment",
        ]
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for r in rows:
            writer.writerow(
                {
                    "question_id": r["question_id"],
                    "method": r["method"],
                    "question": r["question"],
                    "status": r["status"],
                    "answer": r.get("answer", ""),
                    "evidence": json.dumps(r.get("evidence", []), ensure_ascii=False),
                }
            )
    write_json(
        out / "manifest.json",
        {
            "source_run": run.name,
            "results_sha256": digest((run / "results.jsonl").read_bytes()),
            "status": "awaiting_actual_group_review",
        },
    )
    return out


def protocol_template(settings: dict, question_file: Path) -> dict:
    load_corpus()
    return {
        "status": "pending",
        "reviewer": "",
        "reviewed_at": "",
        "questions_sha256": digest(question_file.read_bytes()),
        "corpus_sha256": read_json(ROOT / "data" / "corpus-manifest.json")["sha256"],
        "config_sha256": digest(json.dumps(settings, ensure_ascii=False, sort_keys=True)),
        "comment": "实际审核后由小组填写；AI不得自动改为approved或代签",
    }
