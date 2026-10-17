from __future__ import annotations

import os
from dataclasses import dataclass, field
from urllib.parse import urlsplit

DEEPSEEK_URL = "https://api.deepseek.com"


@dataclass(frozen=True)
class APIConfig:
    provider: str
    base_url: str
    model: str
    api_key: str = field(repr=False)

    def __post_init__(self):
        parsed = urlsplit(self.base_url)
        if (
            parsed.scheme not in {"http", "https"}
            or not parsed.hostname
            or parsed.username is not None
            or parsed.password is not None
            or parsed.query
            or parsed.fragment
        ):
            raise ValueError("Base URL须为无账号、查询参数和片段的HTTP(S)地址")
        if parsed.scheme != "https" and parsed.hostname not in {"localhost", "127.0.0.1", "::1"}:
            raise ValueError("非本机接口必须使用HTTPS，避免明文发送密钥")
        if self.provider not in {"deepseek", "custom"}:
            raise ValueError("未知供应商类型")
        if self.provider == "deepseek" and self.base_url.rstrip("/") != DEEPSEEK_URL:
            raise ValueError("DeepSeek官方配置只能发送到官方地址")
        if not self.model.strip():
            raise ValueError("请填写模型ID")
        if any(ord(c) < 32 or ord(c) == 127 for c in self.api_key):
            raise ValueError("密钥不能包含换行或控制字符")

    def public(self) -> dict:
        return redact(
            {"provider": self.provider, "base_url": self.base_url, "model": self.model}, self.api_key
        )


def environment_api() -> APIConfig:
    url = os.getenv("DEEPSEEK_BASE_URL", DEEPSEEK_URL).rstrip("/")
    return APIConfig(
        "deepseek" if url == DEEPSEEK_URL else "custom",
        url,
        os.getenv("DEEPSEEK_MODEL", "deepseek-flash"),
        os.getenv("DEEPSEEK_API_KEY", ""),
    )


def page_api(provider: str, base_url: str, model: str, key: str) -> APIConfig:
    """Never forward a local key to a different provider/endpoint implicitly."""
    url = DEEPSEEK_URL if provider == "deepseek" else base_url.strip().rstrip("/")
    if not key and provider == "deepseek":
        local_url = os.getenv("DEEPSEEK_BASE_URL", DEEPSEEK_URL).rstrip("/")
        if local_url == DEEPSEEK_URL:
            key = os.getenv("DEEPSEEK_API_KEY", "")
    return APIConfig(provider, url, model.strip(), key)


def redact(value, secret: str):
    if isinstance(value, str):
        return value.replace(secret, "[REDACTED]") if secret else value
    if isinstance(value, dict):
        return {redact(k, secret): redact(v, secret) for k, v in value.items()}
    if isinstance(value, list):
        return [redact(v, secret) for v in value]
    return value
