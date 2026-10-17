#!/usr/bin/env python3
"""Create and verify RFC 3161 timestamps for project evidence files."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
import urllib.error
import urllib.request
import uuid
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
TSA_URL = "https://freetsa.org/tsr"
TSA_CERT_URL = "https://freetsa.org/files/tsa.crt"
CA_CERT_URL = "https://freetsa.org/files/cacert.pem"
# Fingerprints published on FreeTSA's service page. A certificate rotation must
# be checked against that page before changing these pins.
TSA_CERT_SHA256 = "8bfb0305bb64e2571ca507552ef3245cb1c2fee8728e0ff8689225081ea13467"
CA_CERT_SHA256 = "2151b61137ffa86bf664691ba67e7da0b19f98c758e3d228d5d8ebf27e044438"
SCHEMA_VERSION = 1
CHUNK_SIZE = 1024 * 1024
EXCLUDED_DIRS = {".git", ".cache", ".venv", ".runtime", "__pycache__", "data", "models", "private"}
EXCLUDED_NAMES = {".env", "secrets.toml", "credentials.json"}


class TimestampError(RuntimeError):
    """Raised for invalid evidence inputs or failed timestamp verification."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(CHUNK_SIZE), b""):
            digest.update(block)
    return digest.hexdigest()


def is_excluded(relative: Path) -> bool:
    if any(part in EXCLUDED_DIRS for part in relative.parts):
        return True
    parts = relative.parts
    if any(parts[index : index + 2] == ("evidence", "timestamps") for index in range(len(parts) - 1)):
        return True
    return any(name in EXCLUDED_NAMES or name.startswith(".env.") for name in relative.parts)


def resolve_input(root: Path, value: str) -> Path:
    candidate = Path(value)
    if not candidate.is_absolute():
        candidate = root / candidate
    if not candidate.exists():
        raise TimestampError(f"输入不存在：{value}")
    resolved = candidate.resolve()
    if resolved == root or root not in resolved.parents:
        raise TimestampError(f"只允许时间戳项目目录内的文件：{value}")
    relative = resolved.relative_to(root)
    if is_excluded(relative):
        raise TimestampError(f"输入路径属于排除项：{value}")
    if candidate.is_symlink():
        raise TimestampError(f"不接受符号链接：{value}")
    return resolved


def collect_files(root: Path, values: list[str]) -> list[dict[str, Any]]:
    found: dict[str, Path] = {}
    for value in values:
        target = resolve_input(root, value)
        candidates = target.rglob("*") if target.is_dir() else (target,)
        for candidate in candidates:
            if candidate.is_symlink() or not candidate.is_file():
                continue
            resolved = candidate.resolve()
            if root not in resolved.parents:
                continue
            relative = resolved.relative_to(root)
            if is_excluded(relative):
                continue
            found[relative.as_posix()] = resolved

    if not found:
        raise TimestampError("没有可纳入时间戳清单的普通文件")

    entries = []
    for name, path in sorted(found.items()):
        entries.append({"path": name, "bytes": path.stat().st_size, "sha256": sha256_file(path)})
    return entries


def check_files_match(root: Path, entries: list[dict[str, Any]]) -> None:
    for entry in entries:
        relative = Path(entry["path"])
        if relative.is_absolute() or ".." in relative.parts:
            raise TimestampError(f"清单含有越界路径：{entry['path']}")
        candidate = root / relative
        if candidate.is_symlink():
            raise TimestampError(f"证据文件变成符号链接：{entry['path']}")
        path = candidate.resolve()
        if root not in path.parents or not path.is_file():
            raise TimestampError(f"证据文件缺失或不是普通文件：{entry['path']}")
        if path.stat().st_size != entry["bytes"] or sha256_file(path) != entry["sha256"]:
            raise TimestampError(f"证据文件与时间戳清单不一致：{entry['path']}")


def run_openssl(args: list[str], *, capture: bool = True, cwd: Path = ROOT) -> str:
    if shutil.which("openssl") is None:
        raise TimestampError("找不到 openssl。需要安装 OpenSSL 3.x 并将其加入 PATH。")
    cwd = cwd.resolve()
    normalized_args = []
    for argument in args:
        candidate = Path(argument)
        if candidate.is_absolute():
            try:
                argument = candidate.relative_to(cwd).as_posix()
            except ValueError:
                pass
        normalized_args.append(argument)
    try:
        result = subprocess.run(
            ["openssl", *normalized_args],
            check=True,
            capture_output=capture,
            text=capture,
            cwd=cwd,
        )
    except subprocess.CalledProcessError as exc:
        detail = (exc.stderr or exc.stdout or "").strip()
        raise TimestampError(f"OpenSSL 命令失败：{' '.join(normalized_args)}\n{detail}") from exc
    return result.stdout.strip() if capture else ""


def download(url: str, destination: Path) -> None:
    request = urllib.request.Request(url, headers={"User-Agent": "campus-health-evidence-timestamp/1"})
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            data = response.read()
    except (urllib.error.URLError, TimeoutError) as exc:
        raise TimestampError(f"下载时间戳服务文件失败：{url}: {exc}") from exc
    if not data:
        raise TimestampError(f"服务返回空文件：{url}")
    destination.write_bytes(data)


def check_pinned_certificate(path: Path, expected_sha256: str, label: str) -> None:
    actual = sha256_file(path)
    if actual != expected_sha256:
        raise TimestampError(
            f"{label} SHA256 与 FreeTSA 公布值不符。请先人工核对服务官网的证书指纹；"
            f"当前为 {actual}。"
        )


def verify_token(proof_dir: Path, root: Path) -> str:
    manifest = proof_dir / "manifest.json"
    request = proof_dir / "request.tsq"
    response = proof_dir / "response.tsr"
    ca_cert = proof_dir / "cacert.pem"
    tsa_cert = proof_dir / "tsa.crt"
    for path in (manifest, request, response, ca_cert, tsa_cert):
        if not path.is_file():
            raise TimestampError(f"时间戳证明文件缺失：{path.name}")

    check_pinned_certificate(ca_cert, CA_CERT_SHA256, "FreeTSA CA 证书")
    check_pinned_certificate(tsa_cert, TSA_CERT_SHA256, "FreeTSA TSA 证书")

    # Verify the TSA signature, trust chain, and imprint against the manifest.
    by_data = run_openssl(
        [
            "ts", "-verify", "-in", str(response), "-data", str(manifest),
            "-CAfile", str(ca_cert), "-untrusted", str(tsa_cert),
        ], cwd=root,
    )
    # Also verify the response against the original request, including its nonce.
    by_request = run_openssl(
        [
            "ts", "-verify", "-in", str(response), "-queryfile", str(request),
            "-CAfile", str(ca_cert), "-untrusted", str(tsa_cert),
        ], cwd=root,
    )
    return f"{by_data}\n{by_request}\n"


def stamp(root: Path, values: list[str], label: str | None, output_root: Path) -> Path:
    entries = collect_files(root, values)
    manifest_data: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "record_id": uuid.uuid4().hex,
        "label": label,
        "hash_algorithm": "SHA-256",
        "timestamp_service": "FreeTSA RFC 3161 TSA",
        "files": entries,
    }
    manifest_bytes = (json.dumps(manifest_data, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
    manifest_hash = hashlib.sha256(manifest_bytes).hexdigest()

    output_root.mkdir(parents=True, exist_ok=True)
    final_dir = output_root / f"sha256-{manifest_hash[:16]}-{manifest_data['record_id'][:8]}"
    if final_dir.exists():
        raise TimestampError(f"证明目录已存在：{final_dir}")

    with tempfile.TemporaryDirectory(prefix=".pending-", dir=output_root) as temporary:
        proof_dir = Path(temporary)
        manifest = proof_dir / "manifest.json"
        manifest.write_bytes(manifest_bytes)
        request = proof_dir / "request.tsq"
        response = proof_dir / "response.tsr"
        ca_cert = proof_dir / "cacert.pem"
        tsa_cert = proof_dir / "tsa.crt"

        run_openssl(
            ["ts", "-query", "-data", str(manifest), "-sha256", "-cert", "-out", str(request)],
            cwd=root,
        )
        download(CA_CERT_URL, ca_cert)
        download(TSA_CERT_URL, tsa_cert)
        check_pinned_certificate(ca_cert, CA_CERT_SHA256, "FreeTSA CA 证书")
        check_pinned_certificate(tsa_cert, TSA_CERT_SHA256, "FreeTSA TSA 证书")

        post = urllib.request.Request(
            TSA_URL,
            data=request.read_bytes(),
            headers={"Content-Type": "application/timestamp-query"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(post, timeout=45) as result:
                response_bytes = result.read()
        except (urllib.error.URLError, TimeoutError) as exc:
            raise TimestampError(f"向 FreeTSA 请求时间戳失败：{exc}") from exc
        if not response_bytes:
            raise TimestampError("FreeTSA 返回了空时间戳响应")
        response.write_bytes(response_bytes)

        # Detect evidence changes during the online timestamp request.
        check_files_match(root, entries)
        verification = verify_token(proof_dir, root)
        (proof_dir / "verification.txt").write_text(verification, encoding="utf-8")
        info = run_openssl(["ts", "-reply", "-in", str(response), "-text"], cwd=root)
        (proof_dir / "timestamp-info.txt").write_text(info + "\n", encoding="utf-8")
        (proof_dir / "README.txt").write_text(
            "RFC 3161 timestamp proof. Verify from the project root with:\n"
            f"  python tools/timestamp_evidence.py verify {final_dir.relative_to(root).as_posix()}\n\n"
            "The signed response timestamps manifest.json. The manifest binds each listed file "
            "by SHA-256 and byte length. Folder names and filesystem modification times are not "
            "used as time evidence.\n",
            encoding="utf-8",
        )
        proof_dir.rename(final_dir)

    return final_dir


def verify(root: Path, proof_path: str) -> tuple[Path, str]:
    proof_dir = Path(proof_path)
    if not proof_dir.is_absolute():
        proof_dir = root / proof_dir
    proof_dir = proof_dir.resolve()
    if root not in proof_dir.parents:
        raise TimestampError("证明目录必须位于项目目录内")
    try:
        manifest = json.loads((proof_dir / "manifest.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise TimestampError(f"无法读取证明清单：{exc}") from exc
    if manifest.get("schema_version") != SCHEMA_VERSION or manifest.get("hash_algorithm") != "SHA-256":
        raise TimestampError("不支持的证明清单版本或哈希算法")
    entries = manifest.get("files")
    if not isinstance(entries, list) or not entries:
        raise TimestampError("清单中没有证据文件")
    check_files_match(root, entries)
    verification = verify_token(proof_dir, root)
    return proof_dir, verification


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT, help="项目根目录，默认使用本脚本所在项目")
    subparsers = parser.add_subparsers(dest="command", required=True)

    stamp_parser = subparsers.add_parser("stamp", help="对一组文件/目录生成 RFC 3161 时间戳")
    stamp_parser.add_argument("paths", nargs="+", help="项目根目录内的文件或目录；目录按文件内容递归清单")
    stamp_parser.add_argument("--label", help="供人阅读的记录说明，不作为时间来源")
    stamp_parser.add_argument(
        "--output-root", type=Path, default=Path("evidence/timestamps"), help="证明文件保存目录"
    )

    verify_parser = subparsers.add_parser("verify", help="离线核验时间戳和清单内文件")
    verify_parser.add_argument("proof_dir", help="项目根目录内的时间戳证明目录")

    args = parser.parse_args(argv)
    root = args.root.resolve()
    try:
        if args.command == "stamp":
            output_root = args.output_root
            if not output_root.is_absolute():
                output_root = root / output_root
            output_root = output_root.resolve()
            if root not in output_root.parents:
                raise TimestampError("证明输出目录必须位于项目目录内")
            proof_dir = stamp(root, args.paths, args.label, output_root)
            print(f"RFC 3161 时间戳已取得并完成本地验证：{proof_dir.relative_to(root).as_posix()}")
            print(f"签名时间见：{(proof_dir / 'timestamp-info.txt').relative_to(root).as_posix()}")
            return 0

        proof_dir, verification = verify(root, args.proof_dir)
        print(verification, end="")
        print(f"清单和 {len(json.loads((proof_dir / 'manifest.json').read_text(encoding='utf-8'))['files'])} 个源文件均匹配。")
        info = proof_dir / "timestamp-info.txt"
        if info.is_file():
            print(f"TSA 签发时间详情：{info.relative_to(root).as_posix()}")
        return 0
    except (TimestampError, OSError, json.JSONDecodeError) as exc:
        print(f"错误：{exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
