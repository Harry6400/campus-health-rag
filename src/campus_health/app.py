from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
import streamlit as st

from campus_health.cli import interactive
from campus_health.data import load_corpus
from campus_health.retrieval import Retriever
from campus_health.storage import ROOT, config, read_json, safe_error

st.set_page_config(page_title="大学生健康知识 · 证据问答", page_icon="📖", layout="wide")
st.title("大学生健康知识")
name, student_id = os.getenv("DEMO_NAME", ""), os.getenv("DEMO_STUDENT_ID", "")
if name or student_id:
    st.write(f"录屏身份：{name}　{student_id}")
st.info(
    "课程演示：提供一般健康知识与资料来源，不用于诊断、处方或个体治疗。请勿输入真实患者或他人隐私信息。"
)

try:
    docs = load_corpus()
except Exception as exc:
    st.error(safe_error(exc))
    st.code("uv run --locked campus-health prepare")
    st.stop()

settings = config()
methods = {
    "B0": "B0 · BM25 原文基线",
    "R1": "R1 · BM25 + DeepSeek",
    "R2": "R2 · 句向量 + DeepSeek",
    "R3": "R3 · 混合检索 + DeepSeek",
}
question_tab, records_tab, sources_tab = st.tabs(["证据问答", "运行记录", "资料来源"])


@st.cache_resource(show_spinner=False)
def get_retriever(dense: bool):
    return Retriever(load_corpus(), config(), dense=dense)


with question_tab:
    with st.form("question-form"):
        question = st.text_area(
            "你的问题", placeholder="例如：平时坐着学习，怎样在日常生活中增加活动？", max_chars=2000
        )
        method = st.selectbox("比较方法", list(methods), index=3, format_func=methods.get)
        submitted = st.form_submit_button("查找证据并回答", type="primary")
    if submitted:
        if not question.strip():
            st.error("请先输入问题。")
        elif method != "B0" and not os.getenv("DEEPSEEK_API_KEY"):
            st.error("未配置 DeepSeek 密钥。请在本地 .env 设置，或选择不调用 API 的 B0 基线。")
        else:
            try:
                with st.spinner("正在检索与运行；首次加载模型需要较长时间……"):
                    retriever = get_retriever(method in {"R2", "R3"})
                    result, run = interactive(question, method, settings, retriever)
                    st.session_state["latest"] = (result, str(run))
            except Exception as exc:
                st.error(safe_error(exc))
    if "latest" in st.session_state:
        result, run = st.session_state["latest"]
        st.header("回答" if result["method"] != "B0" else "检索原文")
        st.write(result["answer"])
        cols = st.columns(3)
        cols[0].metric("检索", f"{result['retrieval_seconds']:.3f} 秒")
        cols[1].metric("生成", f"{result['generation_seconds']:.2f} 秒")
        cols[2].metric("总响应", f"{result['total_seconds']:.2f} 秒")
        if result.get("usage"):
            usage = result["usage"]
            st.write(
                f"实际用量：输入 {usage.get('prompt_tokens', 0)} token · "
                f"输出 {usage.get('completion_tokens', 0)} token · "
                f"缓存命中 {usage.get('prompt_cache_hit_tokens', 0)} token"
            )
            cost = result.get("cost") or {}
            if cost.get("cny_min") is not None:
                st.write(
                    f"峰谷价费用估算：¥{cost['cny_min']:.6f}—¥{cost['cny_max']:.6f}，非账户扣费。"
                )
        st.header("原文证据")
        for i, c in enumerate(result["evidence"], 1):
            with st.expander(f"[{i}] {c['title']}"):
                st.write(c["text"])
                st.write(c["source_metadata"])
                st.link_button("查看发布机构原文", c["url"])
                st.caption("证据单位：" + ", ".join(c["unit_ids"]))
        st.caption("真实运行编号：" + Path(run).name)

with records_tab:
    summaries = sorted((ROOT / "runs").glob("*/summary.json"), reverse=True)
    if not summaries:
        st.warning("尚无批量运行结果。这里不展示模拟分数。")
    else:
        path = st.selectbox("选择已保存的真实运行", summaries, format_func=lambda p: p.parent.name)
        report = read_json(path)
        spec = read_json(path.parent / "evaluation.json")
        st.write(
            "运行类型：开发集探索（非正式测试）" if spec["exploratory"] else "运行类型：正式测试"
        )
        st.caption("以下为已保存输出的统计，不代表刚刚重新调用模型。AI评分尚需实际复核。")
        st.dataframe(pd.DataFrame(report["summary"]), hide_index=True)
        st.write("API用量与估算费用", report["costs"])
        st.write(
            f"记录 {report['rows']} / 预期 {report['expected_rows']}；运行错误 {report['errors']}；评分错误 {report['judge_errors']}"
        )

with sources_tab:
    manifest = read_json(ROOT / "data" / "corpus-manifest.json")
    st.write(f"版本：{manifest['version']} · {len(docs)}份资料")
    st.warning("当前数据与题集处于 AI 整理阶段；尚未完成小组冻结与正式验收。")
    st.dataframe(
        pd.DataFrame(
            [
                {
                    "编号": d["id"],
                    "主题": d["theme"],
                    "标题": d["title"],
                    "正文字符": len(d["text"]),
                    "原文": d["url"],
                }
                for d in docs
            ]
        ),
        hide_index=True,
    )
    st.caption("本项目只使用声明范围内的文字资料；不使用发布机构标志，也不暗示机构认可本系统。")
