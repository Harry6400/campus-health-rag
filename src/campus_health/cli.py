from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

from .storage import ROOT, config, new_run, now, read_json, safe_error, write_json


def interactive(question: str, method: str, settings: dict, retriever=None) -> tuple[dict, Path]:
    from .data import load_corpus
    from .llm import answer_question, render_answer
    from .retrieval import Retriever

    run = new_run("interactive-not-evaluation", settings)
    start_setup = time.perf_counter()
    retriever = retriever or Retriever(load_corpus(), settings, dense=method in {"R2", "R3"})
    setup = time.perf_counter() - start_setup
    start = time.perf_counter()
    evidence = retriever.search(question, method)
    retrieval = time.perf_counter() - start
    gen_start = time.perf_counter()
    if method == "B0":
        value = {
            "abstained": False,
            "claims": [{"text": c["text"], "citations": [i]} for i, c in enumerate(evidence, 1)],
            "reason": "",
        }
        info = None
    else:
        value, info = answer_question(question, evidence, settings, run / "calls.jsonl")
    result = {
        "kind": "interactive_not_benchmark",
        "method": method,
        "question": question,
        "evidence": evidence,
        "structured": value,
        "answer": render_answer(value),
        "setup_seconds": setup,
        "retrieval_seconds": retrieval,
        "generation_seconds": time.perf_counter() - gen_start if method != "B0" else 0,
        "total_seconds": time.perf_counter() - start,
        "usage": info["usage"] if info else None,
        "cost": info["cost"] if info else None,
    }
    write_json(run / "result.json", result)
    return result, run


def doctor(settings: dict, network: bool) -> dict:
    from .data import load_corpus
    from .retrieval import model_path

    result = {
        "time": now(),
        "os": platform.platform(),
        "python": platform.python_version(),
        "root": str(ROOT),
        "api_key_configured": bool(os.getenv("DEEPSEEK_API_KEY")),
        "model_id": settings["model_id"],
        "model_revision": settings["model_revision"],
        "model_weights_present": (model_path(settings) / "model.safetensors").exists(),
        "protocol_reviewed": (ROOT / "review" / "protocol.json").exists(),
    }
    try:
        docs = load_corpus()
        result["corpus_documents"] = len(docs)
        result["corpus_status"] = read_json(ROOT / "data" / "corpus-manifest.json")["status"]
    except Exception as exc:
        result["corpus_error"] = safe_error(exc)
    if network and result["api_key_configured"]:
        import httpx

        response = httpx.get(
            os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com") + "/models",
            headers={"Authorization": "Bearer " + os.environ["DEEPSEEK_API_KEY"]},
            timeout=30,
        )
        result["api_status"] = response.status_code
        if response.is_success:
            result["models"] = [m["id"] for m in response.json()["data"]]
    return result


def main() -> None:
    # Windows重定向输出可能默认cp1252；帮助/中文日志也必须可执行。
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="大学生日常健康知识：真实运行、版本化评估")
    parser.add_argument("--config", type=Path)
    commands = parser.add_subparsers(dest="command", required=True)
    d = commands.add_parser("doctor")
    d.add_argument("--network", action="store_true")
    commands.add_parser("prepare")
    commands.add_parser("fetch-sources", help="开发取数，不用于正式复现")
    commands.add_parser("draft-questions", help="AI题目草稿，不能冒充人工金标准")
    commands.add_parser("demo")
    sweep = commands.add_parser("sweep", help="开发集单因素参数比较，不是正式测试")
    sweep.add_argument("--questions", type=Path, default=ROOT / "data" / "questions.draft.jsonl")
    sweep.add_argument("--limit", type=int)
    sweep.add_argument("--no-judge", action="store_true")
    a = commands.add_parser("ask")
    a.add_argument("question")
    a.add_argument("--method", default="R3", choices=["B0", "R1", "R2", "R3"])
    for name in ["reproduce", "evaluate"]:
        e = commands.add_parser(name)
        e.add_argument("--questions", type=Path, default=ROOT / "data" / "questions.draft.jsonl")
        e.add_argument("--split", choices=["dev", "test"], default="test")
        e.add_argument("--exploratory", action="store_true")
        e.add_argument("--limit", type=int)
        e.add_argument("--methods", nargs="+", default=["B0", "R1", "R2", "R3"])
        e.add_argument("--no-judge", action="store_true")
        e.add_argument("--resume", type=Path)
    a = commands.add_parser("audit")
    a.add_argument("run", type=Path)
    v = commands.add_parser("validate-questions")
    v.add_argument("path", type=Path, default=ROOT / "data" / "questions.draft.jsonl", nargs="?")
    x = commands.add_parser("export-review")
    x.add_argument("run", type=Path)
    args = parser.parse_args()
    settings = config(args.config)
    try:
        from . import data, evaluation

        if args.command == "doctor":
            result = doctor(settings, args.network)
            run = new_run("doctor", settings)
            write_json(run / "result.json", result)
            print(json.dumps(result, ensure_ascii=False, indent=2))
            print(run)
        elif args.command == "fetch-sources":
            run = new_run("source-intake", settings)
            result = data.fetch_sources()
            write_json(run / "result.json", result)
            print(run)
            if result["errors"]:
                raise RuntimeError("部分来源失败，不能视为数据就绪")
        elif args.command == "prepare":
            data.prepare()
        elif args.command == "draft-questions":
            print(evaluation.draft_questions(settings))
        elif args.command == "sweep":
            from .experiments import sweep

            print(sweep(settings, args.questions, args.limit, not args.no_judge))
        elif args.command == "demo":
            if not (ROOT / "data" / "corpus.jsonl").exists():
                data.prepare()
            app = Path(__file__).with_name("app.py")
            raise SystemExit(
                subprocess.call(
                    [
                        sys.executable,
                        "-m",
                        "streamlit",
                        "run",
                        str(app),
                        "--server.address",
                        "127.0.0.1",
                        "--server.headless",
                        "true",
                    ]
                )
            )
        elif args.command == "ask":
            result, run = interactive(args.question, args.method, settings)
            print(result["answer"])
            print("真实运行记录：", run)
        elif args.command in {"reproduce", "evaluate"}:
            if not (ROOT / "data" / "corpus.jsonl").exists():
                data.prepare()
            print(
                evaluation.run_evaluation(
                    settings,
                    args.questions,
                    args.split,
                    args.limit,
                    args.methods,
                    args.exploratory,
                    not args.no_judge,
                    args.resume,
                )
            )
        elif args.command == "audit":
            print(json.dumps(evaluation.audit(args.run), ensure_ascii=False, indent=2))
        elif args.command == "validate-questions":
            from .storage import read_jsonl

            errors = evaluation.validate_questions(read_jsonl(args.path), data.load_corpus())
            print(json.dumps({"errors": errors}, ensure_ascii=False, indent=2))
            if errors:
                raise SystemExit(1)
        elif args.command == "export-review":
            from .review import export_review

            print(export_review(args.run))
    except (Exception, KeyboardInterrupt) as exc:
        print("运行未完成：" + safe_error(exc), file=sys.stderr)
        raise SystemExit(1) from None


if __name__ == "__main__":
    main()
