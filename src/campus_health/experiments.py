from __future__ import annotations

import math
from pathlib import Path

from .api_config import environment_api
from .evaluation import run_evaluation
from .storage import new_run, read_json, safe_error, write_json


def choose(candidates: list[dict], metrics: list[str], prefer) -> dict:
    """只比较完整且具有有效分母的指标；缺失值不能参与选参。"""
    for candidate in candidates:
        summary = candidate["summary"]
        for metric in metrics:
            value = summary.get(metric)
            if (
                not isinstance(value, (int, float))
                or not math.isfinite(value)
                or summary.get(metric + "_n", 0) <= 0
            ):
                raise ValueError(f"选参指标不可比较：{candidate['value']} {metric}")
    return max(
        candidates,
        key=lambda c: (*[c["summary"][metric] for metric in metrics], prefer(c["value"])),
    )


def sweep(settings: dict, question_file: Path, limit: int | None, ai_judge: bool) -> Path:
    """先选择分词，再以固定分词扫描k；失败停止，不自动启动正式测试。"""
    parent = new_run("parameter-sweep-dev", settings)
    runs = []
    identity = environment_api().public()
    write_json(parent / "generation-api.json", identity)

    def evaluate(factor, value, variant, method, judge):
        if environment_api().public() != identity:
            raise ValueError("开发运行API配置发生变化，停止混合结果")
        path = run_evaluation(
            variant, question_file, "dev", limit, [method], True, judge, stop_on_error=True
        )
        report = read_json(path / "summary.json")
        item = {"factor": factor, "value": value, "run": path.name}
        runs.append(item)
        write_json(parent / "experiments.json", runs)
        if not report["complete"] or report["errors"] or report["judge_errors"]:
            raise ValueError(f"开发运行不完整或有错误：{path.name}")
        return {**item, "summary": report["summary"][0]}

    try:
        tokenizers = [
            evaluate("tokenizer", value, {**settings, "tokenizer": value}, "B0", False)
            for value in ["jieba", "char2"]
        ]
        selected = choose(
            tokenizers, ["evidence_recall", "evidence_precision"], lambda value: value == "jieba"
        )
        write_json(parent / "tokenizer-selection.json", selected)
        variants = [
            evaluate(
                "top_k", k, {**settings, "tokenizer": selected["value"], "top_k": k},
                "R3", ai_judge,
            )
            for k in [2, 4, 6]
        ]
        if ai_judge:
            best = choose(
                variants,
                ["correct_refusal", "keypoint_coverage", "faithfulness", "citation_support"],
                lambda value: -value,
            )
            write_json(parent / "selection.json", {
                "status": "development_recommendation_awaiting_user_confirmation",
                "tokenizer": selected["value"], "top_k": best["value"],
                "tokenizer_candidates": tokenizers, "top_k_candidates": variants,
                "generation_api": identity, "formal_test_started": False,
            })
    except Exception as exc:
        write_json(parent / "stopped.json", {
            "status": "stopped_no_selection", "error": safe_error(exc), "runs": runs,
        })
        raise
    return parent
