from __future__ import annotations

from pathlib import Path

from .evaluation import run_evaluation
from .storage import new_run, write_json


def sweep(settings: dict, question_file: Path, limit: int | None, ai_judge: bool) -> Path:
    """每组只改分词或top-k；探索标签贯穿子运行，不作为正式测试。"""
    parent = new_run("parameter-sweep-dev", settings)
    runs = []
    for tokenizer in ["jieba", "char2"]:
        variant = {**settings, "tokenizer": tokenizer}
        path = run_evaluation(variant, question_file, "dev", limit, ["B0"], True, False)
        runs.append({"factor": "tokenizer", "value": tokenizer, "run": path.name})
        write_json(parent / "experiments.json", runs)
    for k in [2, 4, 6]:
        variant = {**settings, "top_k": k}
        path = run_evaluation(variant, question_file, "dev", limit, ["R3"], True, ai_judge)
        runs.append({"factor": "top_k", "value": k, "run": path.name})
        write_json(parent / "experiments.json", runs)
    return parent
