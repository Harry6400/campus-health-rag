from __future__ import annotations

import hashlib
import os
from dataclasses import replace
from urllib.parse import urlsplit

import streamlit as st

from .api_config import DEEPSEEK_URL, page_api
from .llm import list_models, test_connection
from .storage import new_run


def clear_api_session():
    for key in list(st.session_state):
        if key.startswith("api_") or key == "latest":
            del st.session_state[key]
    st.session_state["api_use_local"] = False
    if "question_method" in st.session_state:
        st.session_state["question_method"] = "B0"


def api_controls(settings: dict):
    with st.sidebar:
        st.header("运行设置")
        provider_label = st.selectbox(
            "供应商", ["DeepSeek官方", "自定义兼容接口"], key="api_provider"
        )
        provider = "deepseek" if provider_label == "DeepSeek官方" else "custom"
        local_ready = bool(os.getenv("DEEPSEEK_API_KEY")) and st.session_state.get(
            "api_use_local", True
        )
        with st.expander("接口、密钥与模型", expanded=provider == "custom" or not local_ready):
            if provider == "deepseek":
                base_url = DEEPSEEK_URL
                st.caption("官方地址：https://api.deepseek.com")
                use_local = st.checkbox(
                    "使用本机.env / 环境变量密钥", value=True, key="api_use_local"
                )
                st.caption("页面密钥优先；不会显示本机已有密钥。")
            else:
                base_url = st.text_input(
                    "Base URL", placeholder="https://你的接口地址/v1", key="api_base_url"
                )
                use_local = False
                st.caption("仅支持兼容Chat Completions的接口；自定义地址不会继承本机DeepSeek密钥。")
            key = st.text_input("API Key（仅当前会话）", type="password", key=f"api_key_{provider}")
            default_model = os.getenv("DEEPSEEK_MODEL", "deepseek-flash")
            model = st.text_input(
                "模型ID",
                value=default_model if provider == "deepseek" else "",
                key=f"api_model_{provider}",
            )
            api, error = None, None
            try:
                api = page_api(provider, base_url, model, key)
                if provider == "deepseek" and not use_local and not key:
                    api = replace(api, api_key="")
            except ValueError as exc:
                error = str(exc)
            if error:
                st.warning(error + "；B0仍可运行。")
            identity = None
            if api:
                identity = (
                    api.provider,
                    api.base_url,
                    api.model,
                    hashlib.sha256(api.api_key.encode()).hexdigest(),
                )
            lookup_api = api
            if not lookup_api and not model.strip():
                try:
                    lookup_api = page_api(provider, base_url, "model-list-only", key)
                    if provider == "deepseek" and not use_local and not key:
                        lookup_api = replace(lookup_api, api_key="")
                except ValueError:
                    pass
            lookup_identity = (
                (
                    lookup_api.provider,
                    lookup_api.base_url,
                    hashlib.sha256(lookup_api.api_key.encode()).hexdigest(),
                )
                if lookup_api
                else None
            )
            if st.button("获取可用模型", disabled=not lookup_api or not lookup_api.api_key):
                try:
                    st.session_state["api_models"] = (
                        lookup_identity,
                        list_models(lookup_api, settings),
                    )
                except Exception as exc:
                    st.warning(str(exc))
            stored = st.session_state.get("api_models")
            if stored and lookup_identity == stored[0]:
                choices = stored[1]
                if choices:
                    st.selectbox("接口返回的模型", choices, key="api_selected_model")
                    if st.button("使用所选模型"):
                        st.session_state["api_pending_model"] = (
                            provider,
                            st.session_state["api_selected_model"],
                        )
                        st.rerun()
                else:
                    st.info("接口未返回模型列表，请手填模型ID。")
        if api:
            st.write(f"当前模型：{api.model}")
            st.caption(f"请求目标：{urlsplit(api.base_url).netloc}")
            if api.api_key:
                st.caption("密钥已配置 · 页面会话" if key else "密钥已配置 · 本机环境")
            else:
                st.info("尚未配置密钥，可以直接使用B0原文检索。")
        st.caption("连接测试会发送一次短请求，可能产生少量费用；不会自动测试。")
        if st.button("测试连接（少量费用）", disabled=not api or not api.api_key):
            run = new_run("connection-check-not-evaluation", {"generation_api": api.public()})
            try:
                with st.spinner("正在检查连接与JSON输出……"):
                    info = test_connection(api, settings, run / "calls.jsonl")
                st.session_state["api_verified"] = (
                    identity,
                    info["response"].get("model"),
                    info["usage"],
                )
            except Exception as exc:
                st.session_state.pop("api_verified", None)
                st.error(str(exc))
        verified = st.session_state.get("api_verified")
        if verified and verified[0] == identity:
            st.success(f"当前配置已通过连接与JSON检查；实际模型：{verified[1]}")
            st.caption("这不代表所有问答场景或该供应商的其他模型都已验证。")
        else:
            st.caption("当前配置尚未通过连接检查。")
        if verified and verified[0] == identity:
            usage = verified[2]
            st.caption(
                f"测试用量：输入 {usage.get('prompt_tokens', '未返回')} / "
                f"输出 {usage.get('completion_tokens', '未返回')} token"
            )
        st.button("清除会话配置", on_click=clear_api_session, width="stretch")
        st.caption("会话密钥不保存到磁盘；已有.env文件不会被修改。")
    return api


def apply_pending_model():
    pending = st.session_state.pop("api_pending_model", None)
    if pending:
        provider, model = pending
        st.session_state[f"api_model_{provider}"] = model


def cost_rows(costs: dict) -> list[dict]:
    rows = []
    labels = {"answer": "回答生成", "ai_judge": "AI辅助评分"}
    for purpose, item in costs.items():
        low, high = item.get("cny_min"), item.get("cny_max")
        fee = "未配置价格"
        if low is not None and high is not None:
            fee = f"¥{low:.6f}—¥{high:.6f}"
        rows.append(
            {
                "用途": labels.get(purpose, purpose),
                "收到响应": item.get("responses", 0),
                "输入token": item.get("prompt_tokens", 0),
                "输出token": item.get("completion_tokens", 0),
                "费用估算区间": fee,
                "失败请求费用未知": item.get("failed_attempts_unknown_cost", 0),
            }
        )
    return rows
