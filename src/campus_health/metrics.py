from __future__ import annotations

from collections import defaultdict

import numpy as np


def retrieval_metrics(question: dict, evidence: list[dict]) -> dict:
    """每个evidence_group是一项必需依据；组内为可替代的原文单位集合。"""
    groups = question.get("evidence_groups", [])
    if not question.get("answerable"):
        return {"evidence_recall": None, "evidence_precision": None}
    if not groups:
        raise ValueError("可回答题缺少证据组")
    recalled = set().union(*(set(c["unit_ids"]) for c in evidence)) if evidence else set()
    gold = set().union(*(set(alt) for g in groups for alt in g))
    covered = sum(any(set(alt) <= recalled for alt in group) for group in groups)
    relevant = sum(bool(set(c["unit_ids"]) & gold) for c in evidence)
    return {
        "evidence_recall": covered / len(groups),
        "evidence_precision": relevant / len(evidence) if evidence else 0.0,
    }


def summarize(rows: list[dict]) -> list[dict]:
    methods = defaultdict(list)
    for row in rows:
        methods[row["method"]].append(row)
    summary = []
    for method, items in sorted(methods.items()):
        ok = [r for r in items if r["status"] == "ok"]
        result = {
            "method": method,
            "attempted": len(items),
            "completed": len(ok),
            "failed": len(items) - len(ok),
            "completion_rate": len(ok) / len(items),
        }
        for name in (
            "evidence_recall",
            "evidence_precision",
            "keypoint_coverage",
            "faithfulness",
            "citation_support",
            "citation_validity",
            "correct_refusal",
            "false_refusal",
        ):
            numbers = [
                r["metrics"][name] for r in items if r.get("metrics", {}).get(name) is not None
            ]
            result[name] = float(np.mean(numbers)) if numbers else None
            result[name + "_n"] = len(numbers)
        for key in ("retrieval_seconds", "generation_seconds", "total_seconds"):
            numbers = [r[key] for r in items if key in r]
            result[key] = (
                {
                    "mean": float(np.mean(numbers)),
                    "median": float(np.median(numbers)),
                    "p95": float(np.percentile(numbers, 95)),
                    "n": len(numbers),
                }
                if numbers
                else None
            )
        result["scoring_note"] = "回答质量仅对已成功且已评分项统计；同时报告完成率与各指标分母"
        summary.append(result)
    return summary
