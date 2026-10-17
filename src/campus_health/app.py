from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
import streamlit as st

from campus_health.api_config import redact
from campus_health.api_ui import api_controls, apply_pending_model, cost_rows
from campus_health.cli import interactive
from campus_health.data import load_corpus
from campus_health.retrieval import Retriever
from campus_health.storage import ROOT, config, read_json, safe_error
from campus_health.ui_views import (
    EXAMPLES,
    METHODS,
    THEMES,
    latency_rows,
    quality_rows,
    run_label,
    summary_rows,
)

st.set_page_config(
    page_title="大学生健康知识 · 证据问答",
    page_icon="📖",
    layout="wide",
    initial_sidebar_state="expanded",
)
# Keep native widgets and focus styles; only constrain reading width and typography.
st.html("""<style>
.stMainBlockContainer {max-width: 1120px; padding-top: 4.25rem; padding-bottom: 3rem;}
h1,h2,h3 {text-wrap: balance;}
[data-testid="stMain"] [data-testid="stHeadingWithActionElements"] h1 {font-size: 2.5rem;}
.stApp { -webkit-font-smoothing: antialiased; }
[data-testid="stMetricValue"] {font-variant-numeric: tabular-nums;}
@media(max-width: 640px) {
 .stMainBlockContainer {padding: 4.25rem 1rem 2rem;}
 [data-testid="stMain"] [data-testid="stHeadingWithActionElements"] h1 {font-size: 1.75rem;}
}
</style>""")
st.title("大学生健康知识")
name, student_id = os.getenv("DEMO_NAME", ""), os.getenv("DEMO_STUDENT_ID", "")
if name or student_id:
    st.write(f"录屏身份：{name}　{student_id}")
st.caption("仅提供一般成年人健康知识，不用于诊断、处方或个体治疗。请勿输入他人隐私信息。")
settings = config()
apply_pending_model()
api = api_controls(settings)

try:
    docs = load_corpus()
except Exception as exc:
    st.error("资料尚未准备好：" + safe_error(exc))
    st.write("请在项目根目录的终端执行以下命令，然后刷新页面：")
    st.code("uv run --locked campus-health prepare", language="shell")
    st.stop()

model_label = api.model if api else "待配置模型"
methods = {
    "B0": "B0 · 原文检索（无需API）",
    "R1": f"R1 · 关键词检索 + {model_label}",
    "R2": f"R2 · 语义检索 + {model_label}",
    "R3": f"R3 · 混合检索 + {model_label}",
}
question_tab, records_tab, sources_tab = st.tabs(["证据问答", "运行记录", "资料来源"])


@st.cache_resource(show_spinner=False)
def get_retriever(dense: bool):
    return Retriever(load_corpus(), config(), dense=dense)


def fill_example(topic: str):
    st.session_state["draft_question"] = EXAMPLES[topic]


with question_tab:
    st.caption("选择示例或输入问题，再点击运行。示例只填入输入框，不会自动调用API。")
    for col, topic in zip(st.columns(3), EXAMPLES):
        col.button(
            f"试问{topic}",
            key=f"example_{topic}",
            on_click=fill_example,
            args=(topic,),
            width="stretch",
        )
    method = st.selectbox(
        "回答方式",
        list(methods),
        index=3 if api and api.api_key else 0,
        format_func=methods.get,
        key="question_method",
        help="B0直接展示本地原文；R1按关键词检索，R2按语义检索，R3结合两者，再由选定模型依据证据回答。",
    )
    with st.container(border=True):
        question = st.text_area(
            "你的问题",
            placeholder="例如：平时坐着学习，怎样在日常生活中增加活动？",
            max_chars=2000,
            key="draft_question",
            height=110,
        )
        if method == "B0":
            st.caption("本次仅在本机检索资料，不调用生成API。")
        elif api and api.api_key:
            st.caption("点击运行会向侧栏所示接口发送问题及检索证据，产生实际API用量。")
        else:
            st.info("此方式需要API密钥。请打开侧栏“接口、密钥与模型”配置，或选择B0。")
        submitted = st.button(
            "检索原文" if method == "B0" else "查找证据并回答",
            type="primary",
            width="stretch",
        )
    if submitted:
        st.session_state.pop("latest", None)
        if not question.strip():
            st.error("请输入问题，或先点击上方的一个示例。")
        elif method != "B0" and (not api or not api.api_key):
            st.error("未运行：此方式需要API密钥，可在侧栏配置或改用B0。")
        else:
            try:
                with st.status("正在准备本地检索……", expanded=True) as status:
                    if method in {"R2", "R3"}:
                        st.write("首次使用语义检索需要加载模型，耗时会长于后续查询。")
                    retriever = get_retriever(method in {"R2", "R3"})
                    status.update(
                        label="正在检索原文……" if method == "B0" else "正在检索并请求模型……"
                    )
                    result, run = interactive(question, method, settings, retriever, api=api)
                    st.session_state["latest"] = (result, str(run))
                    status.update(label="本次运行已完成", state="complete", expanded=False)
            except Exception as exc:
                message = redact(safe_error(exc), api.api_key if api else "")
                st.error("本次未完成，没有生成新结果。" + message)
    if "latest" in st.session_state:
        result, run = st.session_state["latest"]
        st.divider()
        st.subheader("本次结果")
        st.write("问题：" + result["question"])
        target = result.get("generation_api")
        st.caption(
            METHODS.get(result["method"], result["method"])
            + (
                f" · 实际模型：{result.get('actual_model') or target['model']}"
                if target
                else " · 无生成API"
            )
        )
        if result["method"] != "B0":
            if result["structured"]["abstained"]:
                st.info("未给出事实答案：" + result["answer"])
            else:
                with st.container(border=True):
                    st.write(result["answer"])
        cols = st.columns(3)
        cols[0].metric("检索耗时", f"{result['retrieval_seconds']:.3f} 秒")
        cols[1].metric("生成耗时", f"{result['generation_seconds']:.2f} 秒")
        cols[2].metric("总响应", f"{result['total_seconds']:.2f} 秒")
        if result.get("usage"):
            usage, cost = result["usage"], result.get("cost") or {}
            fee = (
                f"¥{cost['cny_min']:.6f}—¥{cost['cny_max']:.6f}（估算，非账户扣费）"
                if cost.get("cny_min") is not None
                else "未配置价格"
            )
            st.caption(
                f"输入 {usage.get('prompt_tokens', '未返回')} / "
                f"输出 {usage.get('completion_tokens', '未返回')} token · 费用：{fee}"
            )
        st.subheader("原文证据")
        if not result["evidence"]:
            st.warning("本次没有检索到原文证据。")
        cited = {n for claim in result["structured"]["claims"] for n in claim["citations"]}
        first_open = min(cited) if result["method"] != "B0" and cited else 1
        for i, c in enumerate(result["evidence"], 1):
            tag = " · 回答已引用" if result["method"] != "B0" and i in cited else ""
            with st.expander(f"[{i}] {c['title']}{tag}", expanded=i == first_open):
                st.write(c["text"])
                st.caption(c["source_metadata"])
                st.link_button("查看发布机构原文", c["url"])
        with st.expander("本次运行详情"):
            st.write("运行编号：" + Path(run).name)
            if target:
                st.write(f"请求接口：{target['base_url']} · 请求模型：{target['model']}")
            st.write(f"首次准备 / 加载：{result['setup_seconds']:.3f} 秒，不计入上方总响应。")
            st.write("此结果属于交互演示，不计入正式实验。")
            st.json({str(i): c["unit_ids"] for i, c in enumerate(result["evidence"], 1)})

with records_tab:
    summaries = sorted((ROOT / "runs").glob("*/summary.json"), reverse=True)
    if not summaries:
        st.info("尚无批量实验记录。可以先在“证据问答”体验；正式实验需完成题集审核。")
    else:
        path = st.selectbox(
            "选择已保存的运行",
            summaries,
            format_func=lambda p: run_label(p.parent.name),
            key="saved_run",
        )
        try:
            report = read_json(path)
            spec = read_json(path.parent / "evaluation.json")
            st.info("开发集探索 · 非正式测试" if spec["exploratory"] else "正式测试记录")
            st.caption("以下来自历史运行，并非刚刚重新调用模型；AI辅助评分仍需小组复核。")
            cols = st.columns(3)
            cols[0].metric("输出记录 / 预期", f"{report['rows']} / {report['expected_rows']}")
            cols[1].metric("问答运行错误", report["errors"])
            cols[2].metric("辅助评分错误", report["judge_errors"])
            if report["errors"] or report["judge_errors"]:
                st.warning(
                    "问答运行错误表示未完成回答；辅助评分错误表示评分未完成，两者不能当作正常拒答或满分。"
                )
            st.subheader("方法效果")
            st.dataframe(
                pd.DataFrame(summary_rows(report["summary"])), hide_index=True, width="stretch"
            )
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
                st.dataframe(
                    pd.DataFrame(quality_rows(report["summary"])), hide_index=True, width="stretch"
                )
            with st.expander("耗时明细（秒）"):
                st.dataframe(
                    pd.DataFrame(latency_rows(report["summary"])), hide_index=True, width="stretch"
                )
                st.caption("P95表示约95%的记录耗时不超过该值；准备/加载和AI评分不计入用户总响应。")
            st.subheader("API用量与估算费用")
            for item in cost_rows(report["costs"]):
                with st.container(border=True):
                    st.write(f"**{item['用途']}** · {item['费用估算区间']}")
                    st.write(
                        f"收到 {item['收到响应']} 次响应 · 输入 {item['输入token']:,} / 输出 {item['输出token']:,} token"
                    )
                    if item["失败请求费用未知"]:
                        st.warning(f"另有 {item['失败请求费用未知']} 次失败请求，费用未知。")
            st.caption("按运行时价格估算，非账户实际扣费；未知费用不能视为0。")
            with st.expander("原始记录与追溯信息"):
                st.write("运行编号：" + path.parent.name)
                st.write("摘要仅作展示，逐题输出和调用日志保存在对应运行目录。")
                st.json(report)
        except (OSError, ValueError, KeyError, TypeError) as exc:
            st.error("此记录不完整或格式不兼容，请选择其他记录。" + safe_error(exc))

with sources_tab:
    manifest = read_json(ROOT / "data" / "corpus-manifest.json")
    st.write(f"知识库：{len(docs)}份资料 · 版本 {manifest['version']}")
    st.warning("当前为开发资料与AI题集初稿，尚未完成小组冻结；不能视为已核验的金标准。")
    filters = st.columns([1, 2])
    theme = filters[0].selectbox("资料主题", ["全部", *THEMES.values()], key="source_theme")
    search = filters[1].text_input("查找资料", placeholder="输入标题关键词", key="source_search")
    filtered = [
        d
        for d in docs
        if (theme == "全部" or THEMES.get(d["theme"]) == theme)
        and search.strip().casefold() in d["title"].casefold()
    ]
    st.caption(f"显示 {len(filtered)} / {len(docs)} 份资料")
    if filtered:
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "主题": THEMES.get(d["theme"], d["theme"]),
                        "文章标题": d["title"],
                        "原文链接": d["url"],
                    }
                    for d in filtered
                ]
            ),
            hide_index=True,
            width="stretch",
            column_config={
                "原文链接": st.column_config.LinkColumn("原文", display_text="查看原文"),
            },
        )
    else:
        st.info("没有匹配的资料，请更换关键词或选择“全部”主题。")
    st.caption("使用范围为睡眠、饮食、运动的一般成年人科普。资料时间、地区与适用条件以原文为准。")
    with st.expander("资料版本与使用说明"):
        st.write(
            "只使用许可范围内的文字，不使用机构标志，不暗示机构认可本系统；原始资料与标准化文本分开保留。"
        )
        st.write(
            "来源、许可与文件校验记录见项目evidence/source-inventory.json；本页只提供原机构链接。"
        )
