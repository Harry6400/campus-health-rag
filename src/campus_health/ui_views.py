"""Presentation only: never recompute or fill missing experimental scores."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

THEMES = {"sleep": "睡眠", "diet": "饮食", "exercise": "运动"}
METHODS = {"B0": "B0 原文检索", "R1": "R1 关键词检索", "R2": "R2 语义检索", "R3": "R3 混合检索"}
EXAMPLES = {
    "睡眠": "根据原文，成人每日需要睡足多少小时？",
    "饮食": "根据冬令进补原则，如何做到油脂适量和天然调味？",
    "运动": "平时坐着学习，怎样在日常生活中增加活动？",
}


def score_cell(row: dict, metric: str) -> str:
    value, n = row.get(metric), row.get(metric + "_n")
    if value is None:
        return "不适用 / 未评分"
    return f"{value:.1%}" + (f"（{n}题）" if n is not None else "（分母未记录）")


def summary_rows(rows: list[dict]) -> list[dict]:
    return [
        {
            "方法": METHODS.get(r["method"], r["method"]),
            "完成 / 尝试": f"{r.get('completed', '—')} / {r.get('attempted', '—')}",
            "答案要点覆盖": score_cell(r, "keypoint_coverage"),
            "证据召回": score_cell(r, "evidence_recall"),
            "引用支持": score_cell(r, "citation_support"),
        }
        for r in rows
    ]


def quality_rows(rows: list[dict]) -> list[dict]:
    return [
        {
            "方法": METHODS.get(r["method"], r["method"]),
            **{
                label: score_cell(r, metric)
                for label, metric in [
                    ("证据精确率", "evidence_precision"),
                    ("事实有据率", "faithfulness"),
                    ("引用编号有效率", "citation_validity"),
                    ("正确拒答率", "correct_refusal"),
                    ("可答题误拒率↓", "false_refusal"),
                ]
            },
        }
        for r in rows
    ]


def latency_rows(rows: list[dict]) -> list[dict]:
    result = []
    for row in rows:
        for key, label in [
            ("retrieval_seconds", "检索"),
            ("generation_seconds", "生成"),
            ("total_seconds", "总响应"),
        ]:
            timing = row.get(key)
            if not timing:
                continue
            result.append(
                {
                    "方法": row["method"],
                    "阶段": label,
                    **{
                        name: (f"{timing[k]:.3f}" if timing.get(k) is not None else "未记录")
                        for k, name in [
                            ("mean", "平均（秒）"),
                            ("median", "中位数（秒）"),
                            ("p95", "P95（秒）"),
                        ]
                    },
                    "样本数": timing.get("n", "未记录"),
                }
            )
    return result


def run_label(name: str) -> str:
    try:
        stamp = datetime.strptime(name.split("_")[0], "%Y%m%dT%H%M%S%fZ")
        local = stamp.replace(tzinfo=timezone.utc).astimezone(timezone(timedelta(hours=8)))
        time_label = local.strftime("%m-%d %H:%M:%S") + "（北京时间）"
    except ValueError:
        return name
    kind = (
        "开发探索"
        if name.endswith("exploratory-dev")
        else "正式测试"
        if name.endswith("formal-test")
        else "其他记录"
    )
    return f"{time_label} · {kind}"
