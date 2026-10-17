"""Read-only adapters for the different artifacts kept in ``runs/``.

Saved summaries are displayed as they are. This module does not audit outputs,
read API call logs, or turn missing values into successful results.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

BEIJING = timezone(timedelta(hours=8))
RETRIEVAL_KIND = "CPU_retrieval_only_not_RAG_answers_or_AI_scoring"
KIND_LABELS = {
    "evaluation": "批量评测",
    "retrieval": "纯检索对比",
    "interactive": "单次问答",
    "process": "准备与检查",
    "unknown": "其他或未知",
}
STATUS_LABELS = {
    "saved": "已保存",
    "errors": "有错误",
    "incomplete": "资料不完整",
    "unknown": "状态未知",
}
PROCESS_KINDS = {
    "source-intake",
    "api-smoke-not-evaluation",
    "retrieval-smoke",
    "doctor",
    "question-draft",
    "question-draft-v2",
    "parameter-sweep-dev",
    "clean-clone-mac",
    "judge-format-fix-check-not-regeneration",
    "local-final-verification",
    "api-ui-live-check-not-evaluation",
    "connection-check-not-evaluation",
    "candidate-dev-offline-only-license-gated",
}
JSON_ARTIFACTS = (
    "manifest.json",
    "summary.json",
    "evaluation.json",
    "result.json",
    "results.json",
    "setup.json",
    "validation.json",
    "experiments.json",
    "selection.json",
    "tokenizer-selection.json",
    "scope.json",
    "checks.json",
    "generation-api.json",
)


@dataclass(frozen=True)
class RunRecord:
    id: str
    path: Path
    kind: str
    source_kind: str
    label: str
    started_at: datetime | None
    methods: tuple[str, ...]
    status: str
    question: str
    model: str
    corpus_sha: str | None
    issues: tuple[str, ...]
    artifacts: tuple[str, ...]
    search_text: str = ""


def _read_json(path: Path, issues: list[str]) -> Any:
    if path.is_symlink():
        issues.append(f"{path.name}：符号链接未读取")
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, UnicodeError) as exc:
        # Do not expose paths, provider responses, or arbitrary error contents.
        issues.append(f"{path.name}：无法读取（{type(exc).__name__}）")
        return None


def _payload(path: Path, artifacts: tuple[str, ...], issues: list[str]) -> dict:
    payload = {}
    for name in JSON_ARTIFACTS:
        if name not in artifacts:
            continue
        value = _read_json(path / name, issues)
        if name in {"manifest.json", "summary.json", "evaluation.json"} and value is not None:
            if not isinstance(value, dict):
                issues.append(f"{name}：内容须为 JSON 对象")
        payload[Path(name).stem] = value
    return payload


def _mapping(value: Any) -> dict:
    return value if isinstance(value, dict) else {}


def _time(value: Any) -> datetime | None:
    if not isinstance(value, str):
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    return parsed.astimezone(timezone.utc) if parsed.tzinfo else None


def _started_at(name: str, manifest: dict) -> datetime | None:
    recorded = _time(manifest.get("time"))
    if recorded:
        return recorded
    try:
        return datetime.strptime(name.split("_", 1)[0], "%Y%m%dT%H%M%S%fZ").replace(
            tzinfo=timezone.utc
        )
    except ValueError:
        return None


def _source_kind(name: str, manifest: dict) -> str:
    if isinstance(manifest.get("kind"), str) and manifest["kind"]:
        return manifest["kind"]
    return re.sub(r"^\d{8}T\d{12}Z_", "", name)


def _classify(source_kind: str, artifacts: tuple[str, ...], payload: dict) -> str:
    summary = _mapping(payload.get("summary"))
    result = _mapping(payload.get("result"))
    if (
        "evaluation.json" in artifacts
        or source_kind in {"exploratory-dev", "formal-test"}
        or summary.get("kind") == "audit_saved_outputs_not_new_model_calls"
        or {"expected_rows", "errors", "judge_errors"}.issubset(summary)
    ):
        return "evaluation"
    if summary.get("kind") == RETRIEVAL_KIND:
        return "retrieval"
    if source_kind == "interactive-not-evaluation" or {
        "question",
        "method",
        "answer",
        "evidence",
    }.issubset(result):
        return "interactive"
    if source_kind in PROCESS_KINDS:
        return "process"
    return "unknown"


def _nonnegative_int(value: Any) -> bool:
    return type(value) is int and value >= 0


def _validate(kind: str, payload: dict, artifacts: tuple[str, ...]) -> list[str]:
    issues = []
    if kind in {"evaluation", "retrieval"}:
        required = {"rows": _nonnegative_int, "summary": lambda v: isinstance(v, list)}
        if kind == "evaluation":
            required.update(
                expected_rows=_nonnegative_int,
                errors=_nonnegative_int,
                judge_errors=_nonnegative_int,
                costs=lambda v: isinstance(v, dict),
            )
        else:
            required.update(
                question_count=_nonnegative_int,
                api_called=lambda v: v is False,
                top_k_values=lambda v: isinstance(v, list)
                and all(_nonnegative_int(n) for n in v),
            )
        summary = _mapping(payload.get("summary"))
        if "summary.json" not in artifacts:
            issues.append("summary.json：缺少已保存的摘要")
        else:
            for key, accepts in required.items():
                if not accepts(summary.get(key)):
                    issues.append(f"summary.json：字段 {key} 缺失或格式不兼容")
            if isinstance(summary.get("summary"), list) and not all(
                isinstance(row, dict) and isinstance(row.get("method"), str)
                for row in summary["summary"]
            ):
                issues.append("summary.json：方法摘要缺少有效的 method")
        if kind == "evaluation":
            spec = _mapping(payload.get("evaluation"))
            if "evaluation.json" not in artifacts:
                issues.append("evaluation.json：批量评测缺少实验说明")
            elif type(spec.get("exploratory")) is not bool:
                issues.append("evaluation.json：字段 exploratory 缺失或格式不兼容")
    elif kind == "interactive":
        result = _mapping(payload.get("result"))
        if "result.json" not in artifacts:
            issues.append("result.json：缺少单次问答结果，无法确认请求是否完成")
        else:
            required = {
                "question": lambda v: isinstance(v, str),
                "method": lambda v: isinstance(v, str) and v in {"B0", "R1", "R2", "R3"},
                "answer": lambda v: isinstance(v, str),
                "evidence": lambda v: isinstance(v, list)
                and all(isinstance(item, dict) for item in v),
                "structured": lambda v: isinstance(v, dict)
                and type(v.get("abstained")) is bool
                and isinstance(v.get("claims"), list),
            }
            for key, accepts in required.items():
                if not accepts(result.get(key)):
                    issues.append(f"result.json：字段 {key} 缺失或格式不兼容")
    elif kind == "process" and not any(
        name in payload and payload[name] is not None
        for name in (
            "result",
            "results",
            "validation",
            "experiments",
            "selection",
            "tokenizer-selection",
            "scope",
            "checks",
        )
    ):
        issues.append("运行目录：只有过程元信息，无法确认是否完成")
    elif kind == "unknown":
        issues.append("运行目录：记录类型无法确认，保留现有文件供检查")
    return issues


def _has_errors(value: Any) -> bool:
    if isinstance(value, list):
        return any(_has_errors(item) for item in value)
    if not isinstance(value, dict):
        return False
    status = value.get("status")
    if isinstance(status, str) and status in {
        "error", "failed", "api_error", "stopped", "blocked"
    }:
        return True
    if any(value.get(key) for key in ("error", "errors", "judge_error", "judge_errors", "failed")):
        return True
    if any(value.get(key) not in (None, 0) for key in ("returncode", "exit_code")):
        return True
    return any(_has_errors(value[key]) for key in ("summary", "results") if key in value)


def _methods(payload: dict) -> tuple[str, ...]:
    candidates = []
    spec = _mapping(payload.get("evaluation"))
    candidates.extend(spec.get("methods", []) if isinstance(spec.get("methods"), list) else [])
    candidates.append(_mapping(payload.get("result")).get("method"))
    summary = _mapping(payload.get("summary")).get("summary", [])
    if isinstance(summary, list):
        candidates.extend(row.get("method") for row in summary if isinstance(row, dict))
    found = {
        match
        for candidate in candidates
        if isinstance(candidate, str)
        for match in re.findall(r"(?<![A-Za-z0-9])(B0|R[123])(?![A-Za-z0-9])", candidate)
    }
    return tuple(method for method in ("B0", "R1", "R2", "R3") if method in found)


def _model(payload: dict) -> str:
    manifest = _mapping(payload.get("manifest"))
    config = _mapping(manifest.get("config"))
    spec = _mapping(payload.get("evaluation"))
    result = _mapping(payload.get("result"))
    models = [result.get("actual_model"), spec.get("generation_model")]
    for owner in (config, spec, result):
        models.append(_mapping(owner.get("generation_api")).get("model"))
    models.append(_mapping(payload.get("generation-api")).get("model"))
    return " · ".join(dict.fromkeys(value for value in models if isinstance(value, str) and value))


def _query_text(path: Path, kind: str, artifacts: tuple[str, ...], issues: list[str]) -> str:
    filename = {"evaluation": "results.jsonl", "retrieval": "retrieval-results.jsonl"}.get(kind)
    if not filename or filename not in artifacts:
        return ""
    file = path / filename
    if file.is_symlink():
        issues.append(f"{filename}：符号链接未读取")
        return ""
    queries = []
    try:
        with file.open(encoding="utf-8") as lines:
            for number, line in enumerate(lines, 1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except ValueError as exc:
                    issues.append(f"{filename}：第 {number} 行无法读取（{type(exc).__name__}）")
                    continue
                if not isinstance(row, dict):
                    issues.append(f"{filename}：第 {number} 行须为 JSON 对象")
                    continue
                # Keep only searchable question text/IDs, never answers or evidence.
                queries.extend(
                    row[key]
                    for key in ("question", "question_id", "id")
                    if isinstance(row.get(key), str) and row[key]
                )
    except (OSError, UnicodeError) as exc:
        issues.append(f"{filename}：无法读取（{type(exc).__name__}）")
    return "\n".join(dict.fromkeys(queries))


def _record(path: Path) -> RunRecord:
    issues = []
    if path.is_symlink():
        artifacts = ()
        issues.append("运行目录：符号链接未读取")
    else:
        try:
            artifacts = tuple(sorted(file.name for file in path.iterdir() if file.is_file()))
        except OSError as exc:
            artifacts = ()
            issues.append(f"运行目录：无法读取（{type(exc).__name__}）")
    payload = _payload(path, artifacts, issues)
    manifest = _mapping(payload.get("manifest"))
    source_kind = _source_kind(path.name, manifest)
    kind = _classify(source_kind, artifacts, payload)
    issues.extend(_validate(kind, payload, artifacts))
    search_text = _query_text(path, kind, artifacts, issues)
    summary = _mapping(payload.get("summary"))
    incomplete = summary.get("complete") is False or bool(summary.get("missing_outputs"))
    if _has_errors([value for key, value in payload.items() if key != "manifest"]):
        status = "errors"
    elif kind == "unknown":
        status = "unknown"
    elif issues or incomplete:
        status = "incomplete"
    else:
        status = "saved"
    spec = _mapping(payload.get("evaluation"))
    result = _mapping(payload.get("result"))
    corpus_sha = spec.get("corpus_sha256") if kind == "evaluation" else summary.get("corpus_sha256")
    if kind == "interactive":
        corpus_sha = result.get("corpus_sha256")
    return RunRecord(
        id=path.name,
        path=path,
        kind=kind,
        source_kind=source_kind,
        label=KIND_LABELS[kind],
        started_at=_started_at(path.name, manifest),
        methods=_methods(payload),
        status=status,
        question=result.get("question") if isinstance(result.get("question"), str) else "",
        model=_model(payload),
        corpus_sha=corpus_sha if isinstance(corpus_sha, str) and corpus_sha else None,
        issues=tuple(dict.fromkeys(issues)),
        artifacts=artifacts,
        search_text=search_text,
    )


def discover_runs(root: str | Path) -> list[RunRecord]:
    """Discover every direct run directory; ``root`` is a project root or ``runs``."""
    root = Path(root)
    directory = root if root.name == "runs" else root / "runs"
    if not directory.is_dir():
        return []
    records = [_record(path) for path in directory.iterdir() if path.is_dir()]
    return sorted(
        records,
        key=lambda record: (
            record.started_at is not None,
            record.started_at or datetime.min.replace(tzinfo=timezone.utc),
            record.id,
        ),
        reverse=True,
    )


def _day(value: date | datetime) -> date:
    return value.astimezone(BEIJING).date() if isinstance(value, datetime) else value


def filter_runs(
    records: list[RunRecord],
    keyword: str = "",
    kind: str = "main",
    method: str = "全部",
    status: str = "全部",
    start_date: date | None = None,
    end_date: date | None = None,
) -> list[RunRecord]:
    """Filter metadata only; date bounds are inclusive calendar days in Beijing."""
    keyword = keyword.strip().casefold()
    kind = next((key for key, label in KIND_LABELS.items() if kind == label), kind)
    status = next((key for key, label in STATUS_LABELS.items() if status == label), status)
    found = []
    for record in records:
        if kind == "main" and record.kind not in {"evaluation", "retrieval", "interactive"}:
            continue
        if kind not in {"main", "全部", "all"} and record.kind != kind:
            continue
        if method != "全部" and method not in record.methods:
            continue
        if status != "全部" and record.status != status:
            continue
        if keyword and keyword not in " ".join(
            (
                record.id,
                record.source_kind,
                record.kind,
                record.label,
                record.question,
                record.model,
                record.search_text,
            )
        ).casefold():
            continue
        if start_date is not None or end_date is not None:
            if record.started_at is None:
                continue
            day = record.started_at.astimezone(BEIJING).date()
            if start_date is not None and day < _day(start_date):
                continue
            if end_date is not None and day > _day(end_date):
                continue
        found.append(record)
    return found


def load_record(record: RunRecord) -> dict:
    """Read saved JSON artifacts only, returning local read/validation issues."""
    issues = list(record.issues)
    payload = _payload(record.path, record.artifacts, issues)
    issues.extend(_validate(record.kind, payload, record.artifacts))
    payload["issues"] = tuple(dict.fromkeys(issues))
    return payload
