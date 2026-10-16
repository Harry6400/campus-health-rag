from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import truststore
from dotenv import load_dotenv

# 使用系统证书库，兼容校园/本机受信代理；不关闭TLS校验。
truststore.inject_into_ssl()

ROOT = Path(os.environ.get("CAMPUS_HEALTH_HOME", Path.cwd())).resolve()
load_dotenv(ROOT / ".env", override=False)
os.environ.setdefault("HF_HOME", str(ROOT / "models" / "hf"))
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def digest(value: str | bytes) -> str:
    return hashlib.sha256(value.encode("utf-8") if isinstance(value, str) else value).hexdigest()


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
        name = f.name
    os.replace(name, path)


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def append_jsonl(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
        f.flush()


def safe_error(exc: Exception) -> str:
    message = f"{type(exc).__name__}: {exc}"
    for name in ("DEEPSEEK_API_KEY", "OPENAI_API_KEY"):
        key = os.getenv(name)
        if key:
            message = message.replace(key, "[REDACTED]")
    return message[:1500]


def config(path: Path | None = None) -> dict:
    return read_json(path or ROOT / "configs" / "default.json")


def provenance() -> dict:
    def git(*args):
        result = subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True)
        return result.stdout.strip() if result.returncode == 0 else None

    files = sorted((ROOT / "src").rglob("*.py")) + sorted((ROOT / "configs").glob("*.json"))
    fingerprints = {str(p.relative_to(ROOT)): digest(p.read_bytes()) for p in files}
    lock = ROOT / "uv.lock"
    if lock.exists():
        fingerprints["uv.lock"] = digest(lock.read_bytes())
    return {
        "time": now(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python": platform.python_version(),
        "commit": git("rev-parse", "HEAD"),
        "working_tree": git("status", "--porcelain"),
        "files": fingerprints,
        "packages": {
            p: importlib.metadata.version(p)
            for p in ["numpy", "torch", "sentence-transformers", "streamlit", "openai"]
        },
    }


def new_run(label: str, settings: dict) -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    path = ROOT / "runs" / f"{stamp}_{label}"
    path.mkdir(parents=True)
    write_json(path / "manifest.json", {"kind": label, "config": settings, **provenance()})
    return path
