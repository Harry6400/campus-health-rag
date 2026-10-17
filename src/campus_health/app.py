from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
import streamlit as st

from campus_health.api_config import redact
from campus_health.api_ui import api_controls, api_identity, apply_pending_model
from campus_health.cli import interactive
from campus_health.data import load_corpus
from campus_health.history_ui import render_history
from campus_health.retrieval import Retriever
from campus_health.storage import ROOT, config, read_json, safe_error
from campus_health.ui_views import (
    EXAMPLES,
    METHODS,
    THEMES,
)

st.set_page_config(
    page_title="大学生健康知识 · 证据问答",
    page_icon="📖",
    layout="wide",
    initial_sidebar_state="collapsed",
)
# Scope layout changes to our containers; retain native widgets and keyboard focus styles.
st.html("""<style>
.stMainBlockContainer {max-width: 1120px; padding-top: 3.5rem; padding-bottom: 3rem;}
h1,h2,h3 {text-wrap: balance;}
[data-testid="stMain"] h1 {font-size: 2rem; letter-spacing: -.035em;}
[data-testid="stMain"] h2 {font-size: 1.3rem;}
[data-testid="stMain"] h3 {font-size: 1.15rem;}
.stApp { -webkit-font-smoothing: antialiased; }
[data-testid="stMarkdownContainer"] p {line-height: 1.75;}
[data-testid="stMetricValue"] {font-variant-numeric: tabular-nums; font-size: 1.5rem;}
.st-key-question_panel {background: #FFFFFF; border-radius: 16px;}
.st-key-answer_panel {background: #FFFFFF; border-radius: 12px;}
.st-key-header_layout [data-testid="stHorizontalBlock"] {align-items: center;}
@media(max-width: 900px) {
 .st-key-result_layout [data-testid="stHorizontalBlock"] {flex-direction: column;}
 .st-key-result_layout [data-testid="stColumn"] {width: 100%; flex: 1 1 100%;}
}
@media(max-width: 640px) {
 .stMainBlockContainer {padding: 3.25rem 1rem 2rem;}
 [data-testid="stMain"] h1 {font-size: 1.6rem;}
}
</style>""")
settings = config()
apply_pending_model()
with st.container(key="header_layout"):
    heading, controls = st.columns([4, 1.3], vertical_alignment="center")
    with heading:
        st.title("大学生健康知识")
    with controls:
        api = api_controls(settings)
name, student_id = os.getenv("DEMO_NAME", ""), os.getenv("DEMO_STUDENT_ID", "")
if name or student_id:
    st.write(f"录屏身份：{name}　{student_id}")
st.write("睡眠、饮食、运动的一般成年人知识。仅供科普，不用于诊断、处方或个体治疗。")

docs, manifest, corpus_error = [], {}, None
try:
    docs = load_corpus()
    manifest = read_json(ROOT / "data" / "corpus-manifest.json")
except Exception as exc:
    corpus_error = safe_error(exc)

configured = bool(api and api.api_key)
verified = st.session_state.get("api_verified")
connection_checked = bool(verified and verified[0] == api_identity(api))
corpus_label = f"本地资料已就绪 · {len(docs)} 份" if not corpus_error else "本地资料待准备"
api_label = (
    f"{api.model} · 密钥已配置 · " + ("连接检查已通过" if connection_checked else "连接尚未检查")
    if configured
    else f"{api.model if api else '生成配置待完善'} · 未配置生成密钥 · 可用原文检索 B0"
)
st.caption(corpus_label + "　|　" + api_label)
methods = {
    "B0": "原文检索 B0 · 无需密钥",
    "R1": "关键词问答 R1",
    "R2": "语义问答 R2",
    "R3": "混合问答 R3",
}
method_help = {
    "B0": "按关键词查找本地原文，直接展示依据，不生成概括，不调用生成 API。",
    "R1": "按关键词查找原文，再由模型依据证据回答。",
    "R2": "按问题含义查找原文，再由模型依据证据回答。",
    "R3": "结合关键词和语义查找原文，再由模型依据证据回答。",
}
question_tab, records_tab, sources_tab = st.tabs(["证据问答", "运行记录", "资料来源"])


@st.cache_resource(show_spinner=False)
def get_retriever(dense: bool, corpus_sha: str | None):
    # The displayed source version and the cached index must describe the same corpus.
    return Retriever(load_corpus(), config(), dense=dense)


def fill_example(topic: str):
    st.session_state["draft_question"] = EXAMPLES[topic]


def switch_to_b0():
    st.session_state["question_method"] = "B0"
    st.session_state.pop("ui_failure", None)


def queue_retry():
    st.session_state["retry_requested"] = True


def clear_source_filters():
    st.session_state["source_theme"] = "全部"
    st.session_state["source_search"] = ""


def show_corpus_error():
    st.error("资料尚未准备好，暂时不能检索。")
    st.write("请让配置项目的 Agent 在项目根目录执行，再点击“重新检查资料”：")
    st.code("uv run --locked campus-health prepare", language="shell")
    st.caption("下载失败时检查网络；若文件校验失败，请按 README 恢复指定版本，不要跳过校验。")
    with st.expander("资料错误详情"):
        st.text(corpus_error)


def render_evidence(result):
    st.subheader("原文证据")
    if not result["evidence"]:
        st.warning("本次没有检索到原文证据。可以换一种问法，或在“资料来源”检查知识库范围。")
    cited = {n for claim in result["structured"]["claims"] for n in claim["citations"]}
    first_open = min(cited) if result["method"] != "B0" and cited else 1
    for i, c in enumerate(result["evidence"], 1):
        tag = " · 回答已引用" if result["method"] != "B0" and i in cited else ""
        with st.expander(f"[{i}] {c['title']}{tag}", expanded=i == first_open):
            st.write(c["text"])
            st.caption(c["source_metadata"])
            st.link_button("查看发布机构原文", c["url"])


with question_tab:
    st.write("**1 选择或输入问题　→　2 点击运行　→　3 对照原文核对**")
    if corpus_error:
        show_corpus_error()
        st.button("重新检查资料", key="recheck_corpus", on_click=get_retriever.clear)
    st.session_state.setdefault("draft_question", EXAMPLES["运动"])
    with st.container(border=True, key="question_panel"):
        question = st.text_area(
            "你的问题",
            placeholder="例如：平时坐着学习，怎样在日常生活中增加活动？",
            max_chars=2000,
            key="draft_question",
            height=110,
        )
        with st.container(horizontal=True, gap="small"):
            for topic in EXAMPLES:
                st.button(
                    f"试问{topic}",
                    key=f"example_{topic}",
                    on_click=fill_example,
                    args=(topic,),
                )
        method = st.selectbox(
            "回答方式",
            list(methods),
            index=3 if configured else 0,
            format_func=methods.get,
            key="question_method",
        )
        st.caption(method_help[method])
        if method != "B0" and configured:
            st.caption("点击运行才向“运行设置”中的接口发送问题与证据，并产生实际 API 用量。")
        elif method != "B0":
            st.info("此方式需要密钥。请打开页顶“运行设置”配置，或选择原文检索 B0。")
        submitted = st.button(
            "检索原文" if method == "B0" else "查找证据并回答",
            type="primary",
            width="stretch",
            disabled=bool(corpus_error),
        )
        st.caption("请勿输入他人隐私信息。示例只填入问题，不会自动运行。")
    signature = {
        "question": question.strip(),
        "method": method,
        "api": api_identity(api),
        "corpus": manifest.get("sha256"),
    }
    retry_requested = st.session_state.pop("retry_requested", False)
    if submitted or retry_requested:
        st.session_state.pop("latest", None)
        st.session_state.pop("latest_signature", None)
        st.session_state.pop("ui_failure", None)
        if not question.strip():
            st.session_state["ui_failure"] = "请输入问题，或先点击一个主题示例。"
        elif corpus_error:
            st.session_state["ui_failure"] = "资料尚未准备好，请先按上方提示准备资料。"
        elif method != "B0" and not configured:
            st.session_state["ui_failure"] = "未运行：此方式需要 API 密钥，可在页顶配置或改用 B0。"
        else:
            try:
                with st.status("正在准备本地检索……", expanded=True) as status:
                    st.write("首次使用需要准备固定版本的模型文件；下载失败时请检查网络后重试。")
                    if method in {"R2", "R3"}:
                        st.write("首次使用语义检索需要加载模型，耗时会长于后续查询。")
                    retriever = get_retriever(method in {"R2", "R3"}, manifest.get("sha256"))
                    status.update(
                        label="正在检索原文……" if method == "B0" else "正在检索并请求模型……"
                    )
                    result, run = interactive(question, method, settings, retriever, api=api)
                    st.session_state["latest"] = (result, str(run))
                    st.session_state["latest_signature"] = signature
                    status.update(label="本次运行已完成", state="complete", expanded=False)
            except Exception as exc:
                message = redact(safe_error(exc), api.api_key if api else "")
                st.session_state["ui_failure"] = "本次未完成，没有生成新结果。" + message
    if failure := st.session_state.get("ui_failure"):
        st.error(failure)
        st.write("可修改问题后重试；接口鉴权、限流或模型输出报错时，请在页顶“运行设置”检查配置。")
        with st.container(horizontal=True):
            st.button("重试本次问题", key="retry_question", on_click=queue_retry)
            st.button("改用原文检索 B0", key="switch_to_b0", on_click=switch_to_b0)
    if "latest" in st.session_state:
        result, run = st.session_state["latest"]
        st.divider()
        st.header("本次结果")
        if st.session_state.get("latest_signature") != signature:
            st.warning("上次结果：当前问题、回答方式、配置或资料版本已变化，修改尚未运行。以下仍是上次提交的结果。")
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
        if result["method"] == "B0":
            st.info("B0 直接展示检索原文，不生成概括。请阅读下方原文，并结合来源的适用条件判断。")
            render_evidence(result)
        else:
            with st.container(key="result_layout"):
                answer_col, evidence_col = st.columns([1, 1], gap="large")
                with answer_col:
                    st.subheader("回答")
                    if result["structured"]["abstained"]:
                        st.info("未给出事实答案：" + result["answer"])
                    else:
                        with st.container(border=True, key="answer_panel"):
                            st.write(result["answer"])
                    st.caption("回答中的 [编号] 对应原文证据。引用编号有效不代表内容一定受到支持。")
                with evidence_col:
                    render_evidence(result)
        st.caption(
            f"总响应 {result['total_seconds']:.3f} 秒 · 检索 {result['retrieval_seconds']:.3f} 秒"
            f" · 生成 {result['generation_seconds']:.2f} 秒 · 交互演示，不计入正式实验"
        )
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
        with st.expander("本次运行详情"):
            st.write("运行编号：" + Path(run).name)
            if target:
                st.write(f"请求接口：{target['base_url']} · 请求模型：{target['model']}")
            st.write(f"首次准备 / 加载：{result['setup_seconds']:.3f} 秒，不计入上方总响应。")
            st.write("此结果属于交互演示，不计入正式实验。")
            st.json({str(i): c["unit_ids"] for i, c in enumerate(result["evidence"], 1)})

with records_tab:
    render_history(manifest.get("sha256") if not corpus_error else None, ROOT)

with sources_tab:
    if corpus_error:
        st.info("资料未就绪。请在“证据问答”按提示准备并重新检查资料，历史运行仍可在“运行记录”查看。")
    else:
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
        if theme != "全部" or search:
            st.button("清除筛选", key="clear_source_filters", on_click=clear_source_filters)
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
            st.info("没有匹配的资料，请更换关键词或点击“清除筛选”。")
        st.caption(
            "使用范围为睡眠、饮食、运动的一般成年人科普。资料时间、地区与适用条件以原文为准。"
        )
        with st.expander("资料版本与使用说明"):
            st.write(
                "只使用许可范围内的文字，不使用机构标志，不暗示机构认可本系统；原始资料与标准化文本分开保留。"
            )
            st.write(
                "来源、许可与文件校验记录见项目evidence/source-inventory.json；本页只提供原机构链接。"
            )
