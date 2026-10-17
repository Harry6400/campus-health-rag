"""Replay saved runs without making model calls or recalculating their scores."""

from __future__ import annotations

from pathlib import Path
from urllib.parse import urlsplit

import pandas as pd
import streamlit as st

from .api_ui import cost_rows
from .run_history import (
    BEIJING,
    KIND_LABELS,
    STATUS_LABELS,
    RunRecord,
    discover_runs,
    filter_runs,
    load_record,
)
from .storage import safe_error
from .ui_views import METHODS, latency_rows, quality_rows, score_cell, summary_rows

TYPE_OPTIONS = {"main": "问答与实验", **KIND_LABELS, "all": "全部记录"}


def clear_history_filters():
    for key, value in {
        "history_keyword": "",
        "history_kind": "main",
        "history_method": "全部",
        "history_status": "全部",
    }.items():
        st.session_state[key] = value
    st.session_state.pop("history_dates", None)
    st.session_state.pop("saved_run", None)


def show_all_history():
    clear_history_filters()
    st.session_state["history_kind"] = "all"


def public_details(value, root: Path):
    """Hide machine-specific project paths, including nested manifest values."""
    if isinstance(value, Path):
        value = str(value)
    if isinstance(value, str):
        for prefix in {str(root), str(root.resolve())}:
            value = value.replace(prefix, "<PROJECT_ROOT>")
        return value
    if isinstance(value, dict):
        return {public_details(k, root): public_details(v, root) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [public_details(item, root) for item in value]
    return value


def _corpus_status(previous: str | None, current: str | None):
    if previous and current:
        if previous != current:
            st.warning("资料版本不同：本记录使用历史语料，不能作为当前资料版本的实验结论。")
        else:
            st.caption("资料指纹与当前版本一致；这只表示语料相同，不代表其他实验条件一致。")
    else:
        st.warning("无法确认本记录与当前资料的版本关系：语料指纹缺失或当前资料未就绪。")


def _costs(costs):
    if not isinstance(costs, dict):
        st.caption("本记录未保存费用汇总，不能视为费用为0。")
        return
    for item in cost_rows(costs):
        with st.container(border=True):
            st.write(f"**{item['用途']}** · {item['费用估算区间']}")
            st.write(
                f"收到 {item['收到响应']} 次响应 · 输入 {item['输入token']:,} / "
                f"输出 {item['输出token']:,} token"
            )
            if item["失败请求费用未知"]:
                st.warning(f"另有 {item['失败请求费用未知']} 次失败请求，费用未知。")
    st.caption("按运行时价格估算，非账户实际扣费；未知费用不能视为0。")


def _evaluation(payload: dict):
    spec, report = payload.get("evaluation"), payload.get("summary")
    if isinstance(spec, dict) and type(spec.get("exploratory")) is bool:
        st.info("开发集探索 · 非正式测试" if spec["exploratory"] else "正式测试记录")
    else:
        st.warning("实验类别未完整记录，无法确认是否为正式测试。")
    st.caption("以下来自历史运行，并非刚刚重新调用模型；AI辅助评分仍需小组复核。")
    if not isinstance(report, dict):
        st.info("本次没有可读取的批量评测摘要，请选择其他记录或查看下方文件列表。")
        return
    cols = st.columns(3)
    cols[0].metric(
        "输出记录 / 预期",
        f"{report.get('rows', '未记录')} / {report.get('expected_rows', '未记录')}",
    )
    cols[1].metric("问答运行错误", report.get("errors", "未记录"))
    cols[2].metric("辅助评分错误", report.get("judge_errors", "未记录"))
    if report.get("errors") or report.get("judge_errors"):
        st.warning(
            "问答运行错误表示未完成回答；辅助评分错误表示评分未完成，两者不能当作正常拒答或满分。"
        )
    rows = report.get("summary")
    if isinstance(rows, list) and rows:
        st.header("方法效果")
        st.dataframe(pd.DataFrame(summary_rows(rows)), hide_index=True, width="stretch")
        st.caption(
            "百分比后括号是该指标的有效题数；不同指标的分母可能不同。缺失值不按0或100%处理。"
        )
        with st.expander("指标解释与完整质量指标"):
            st.write(
                "答案要点覆盖：答到了多少必答要点；证据召回：找到了多少必需依据；引用支持：引用是否支持对应说法。"
            )
            st.write(
                "事实有据率：回答事实是否受证据支持；引用编号有效不代表内容获支持。正确拒答只统计不应回答的题，误拒率只统计可回答题，越低越好。"
            )
            st.write("B0直接展示原文，没有拒答机制；不适用与未成功评分均保留缺失，不混进均值。")
            st.dataframe(pd.DataFrame(quality_rows(rows)), hide_index=True, width="stretch")
        with st.expander("耗时明细（秒）"):
            st.dataframe(pd.DataFrame(latency_rows(rows)), hide_index=True, width="stretch")
            st.caption("P95表示约95%的记录耗时不超过该值；准备/加载和AI评分不计入用户总响应。")
    else:
        st.info("本记录没有已保存的方法分数。")
    st.header("API用量与估算费用")
    _costs(report.get("costs"))


def _retrieval(payload: dict):
    st.info("CPU纯检索对比 · 未生成回答、未进行AI评分")
    st.write("此记录只比较证据检索，不是90题正式测试；检索召回率不能当作回答质量或拒答成绩。")
    report = payload.get("summary")
    if not isinstance(report, dict):
        st.info("本记录没有可读取的检索摘要。")
        return
    cols = st.columns(3)
    cols[0].metric("原始问题数", report.get("question_count", "未记录"))
    values = report.get("top_k_values")
    cols[1].metric("top-k配置数", len(values) if isinstance(values, list) else "未记录")
    cols[2].metric("检索输出条数", report.get("rows", "未记录"))
    if isinstance(values, list):
        st.caption("top-k：" + " / ".join(map(str, values)) + "；同一道问题可有多条配置输出。")
    rows = report.get("summary")
    if isinstance(rows, list) and rows:
        if any(type(row.get("failed")) in {int, float} and row["failed"] > 0 for row in rows):
            st.warning("此检索记录有失败项。失败不算检索成功，指标仍按已保存的有效题数展示。")
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "方法 / 配置": row["method"],
                        "完成 / 尝试": f"{row.get('completed', '未记录')} / {row.get('attempted', '未记录')}",
                        "检索失败": row.get("failed", "未记录"),
                        "证据召回": score_cell(row, "evidence_recall"),
                        "证据精确率": score_cell(row, "evidence_precision"),
                    }
                    for row in rows
                ]
            ),
            hide_index=True,
            width="stretch",
        )
        with st.expander("检索耗时（秒）"):
            timings = [{**row, "generation_seconds": None, "total_seconds": None} for row in rows]
            st.dataframe(pd.DataFrame(latency_rows(timings)), hide_index=True, width="stretch")
    if report.get("top_k_selected") is False:
        st.caption("本记录未完成参数选择；正式选择仍需按既定规则结合回答与拒答指标。")


def _interactive(payload: dict):
    st.info("单次问答回放 · 交互演示，不计入正式实验")
    st.caption("以下是已保存的历史回答；查看或切换记录不会重新调用模型。")
    result = payload.get("result")
    if not isinstance(result, dict):
        st.info("本记录没有可读取的问答结果。")
        return
    st.write("问题：" + str(result.get("question", "未记录")))
    method = result.get("method")
    st.caption(METHODS.get(method, str(method or "回答方式未记录")))
    target = result.get("generation_api")
    actual = result.get("actual_model")
    if actual or isinstance(target, dict):
        st.caption("实际 / 请求模型：" + str(actual or target.get("model", "未记录")))
    structured = result.get("structured")
    if method == "B0":
        st.info("B0直接展示检索原文，不生成概括。")
    elif isinstance(result.get("answer"), str):
        if isinstance(structured, dict) and structured.get("abstained") is True:
            st.info("未给出事实答案：" + result["answer"])
        else:
            st.write(result["answer"])
    evidence = result.get("evidence")
    if isinstance(evidence, list):
        st.subheader("已保存的原文证据")
        if not evidence:
            st.caption("本次未保存原文证据。")
        for i, item in enumerate(evidence, 1):
            if not isinstance(item, dict):
                continue
            with st.expander(f"[{i}] {item.get('title', '标题未记录')}", expanded=i == 1):
                st.write(item.get("text", "原文未记录"))
                if item.get("source_metadata"):
                    st.caption(str(item["source_metadata"]))
                url = item.get("url")
                if isinstance(url, str) and urlsplit(url).scheme in {"https", "http"}:
                    st.link_button("查看发布机构原文", url)
    timing = [
        f"{label} {result[key]:.3f} 秒"
        for key, label in (
            ("total_seconds", "总响应"),
            ("retrieval_seconds", "检索"),
            ("generation_seconds", "生成"),
        )
        if type(result.get(key)) in {int, float}
    ]
    if timing:
        st.caption(" · ".join(timing))
    if isinstance(result.get("usage"), dict):
        usage = result["usage"]
        st.caption(
            f"输入 {usage.get('prompt_tokens', '未返回')} / "
            f"输出 {usage.get('completion_tokens', '未返回')} token"
        )
    if isinstance(result.get("cost"), dict):
        cost = result["cost"]
        low, high = cost.get("cny_min"), cost.get("cny_max")
        if low is not None and high is not None:
            st.caption(f"费用估算：¥{low:.6f}—¥{high:.6f}，非账户实际扣费。")
        else:
            st.caption("费用未配置价格或无法确认，不能视为0。")


def _record_details(record: RunRecord, payload: dict, root: Path, issues):
    with st.expander("记录文件与技术详情"):
        st.write("运行目录：runs/" + record.id)
        if record.source_kind:
            st.write("原始类别：" + public_details(record.source_kind, root))
        for issue in issues:
            st.text(public_details(issue, root))
        st.write("已保存文件：")
        st.code(
            "\n".join(record.artifacts) if record.artifacts else "没有可列出的文件", language=None
        )
        st.caption("此页面只读取已保存的摘要、结果与配置，不读取API调用日志或个人复核内容。")
        if payload:
            st.json(public_details({k: v for k, v in payload.items() if k != "issues"}, root))


def render_history(current_sha: str | None, root: Path):
    records = discover_runs(root)
    st.text_input("搜索运行记录", placeholder="运行编号、问题、模型或类别", key="history_keyword")
    filters = st.columns(3)
    kind = filters[0].selectbox(
        "记录类型", list(TYPE_OPTIONS), format_func=TYPE_OPTIONS.get, key="history_kind"
    )
    method = filters[1].selectbox("检索方法", ["全部", *METHODS], key="history_method")
    status = filters[2].selectbox(
        "保存状态",
        ["全部", *STATUS_LABELS],
        format_func=lambda value: STATUS_LABELS.get(value, value),
        key="history_status",
    )
    dates = st.date_input(
        "运行日期（北京时间，可选范围）",
        value=(),
        format="YYYY-MM-DD",
        key="history_dates",
        help="起止日期均包含当天；选择日期后，未记录时间的运行将不显示。",
    )
    if not isinstance(dates, (tuple, list)):
        dates = (dates,) if dates else ()
    start, end = (dates[0] if dates else None), (dates[-1] if dates else None)
    filtered = filter_runs(
        records,
        keyword=st.session_state.get("history_keyword", ""),
        kind=kind,
        method=method,
        status=status,
        start_date=start,
        end_date=end,
    )
    st.caption(f"显示 {len(filtered)} / {len(records)} 条记录 · 从新到旧")
    st.button("清除筛选", key="clear_history_filters", on_click=clear_history_filters)
    if not filtered:
        st.session_state.pop("saved_run", None)
        st.info("没有匹配的运行记录。请清除筛选，或调整关键词、类型和日期。")
        if any(record.kind in {"process", "unknown"} for record in records):
            st.caption("还有准备、检查或不完整记录；选择“全部记录”可查看。")
            st.button("查看全部记录", key="history_show_all", on_click=show_all_history)
        if not records:
            st.caption("尚无已保存的运行。可以先在“证据问答”体验；正式实验需完成题集审核。")
        return
    by_id = {record.id: record for record in filtered}
    if st.session_state.get("saved_run") not in by_id:
        st.session_state["saved_run"] = filtered[0].id
    selected = st.selectbox(
        "选择已保存的运行",
        list(by_id),
        key="saved_run",
        format_func=lambda value: (
            (
                by_id[value].started_at.astimezone(BEIJING).strftime("%m-%d %H:%M:%S（北京时间）")
                if by_id[value].started_at
                else "时间未记录"
            )
            + f" · {by_id[value].label} · {value}"
        ),
    )
    record = by_id[selected]
    payload = load_record(record)
    issues = list(dict.fromkeys([*record.issues, *payload.get("issues", ())]))
    st.caption("运行编号：" + record.id + " · " + STATUS_LABELS.get(record.status, record.status))
    if issues:
        st.error("此记录不完整或格式不兼容，以下仅展示能够读取的已保存内容。可选择其他记录。")
    if record.kind in {"evaluation", "retrieval", "interactive"}:
        _corpus_status(record.corpus_sha, current_sha)
    try:
        if record.kind == "evaluation":
            _evaluation(payload)
        elif record.kind == "retrieval":
            _retrieval(payload)
        elif record.kind == "interactive":
            _interactive(payload)
        else:
            st.info("准备与检查记录" if record.kind == "process" else "其他或未知记录")
            st.write("此类记录不作为问答实验成绩；可在下方核对已有文件与过程信息。")
    except (KeyError, TypeError, ValueError, IndexError, AttributeError) as exc:
        st.error("此记录有不兼容的字段，无法完整展示。请选择其他记录或查看技术详情。")
        issues.append(safe_error(exc))
    _record_details(record, payload, root, issues)
