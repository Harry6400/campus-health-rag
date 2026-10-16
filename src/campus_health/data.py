from __future__ import annotations

import json
import re
import tarfile
from pathlib import Path

import httpx
from bs4 import BeautifulSoup
from opencc import OpenCC

from .storage import ROOT, digest, now, read_json, read_jsonl, write_json

CONVERTER = OpenCC("t2s")


def normalize(text: str) -> str:
    return re.sub(r"[ \t\u3000\xa0]+", " ", CONVERTER.convert(text)).strip()


def canonical_units(doc_id: str, paragraphs: list[str]) -> list[dict]:
    """证据单位独立于切块大小；长句在标点处或128字符处分割。"""
    units = []
    for pi, paragraph in enumerate(paragraphs, 1):
        for sentence in re.split(r"(?<=[。！？；])", paragraph):
            sentence = sentence.strip()
            if not sentence:
                continue
            pieces = [sentence]
            if len(sentence) > 128:
                pieces = [s.strip() for s in re.split(r"(?<=[，、：])", sentence) if s.strip()]
            for piece in pieces:
                for start in range(0, len(piece), 128):
                    text = piece[start : start + 128]
                    units.append(
                        {
                            "id": f"{doc_id}:u{len(units) + 1:04d}",
                            "paragraph": f"{doc_id}:p{pi:03d}",
                            "text": text,
                        }
                    )
    return units


def parse_article(html: str, source: dict) -> dict:
    soup = BeautifulSoup(html, "html.parser")
    is_hpa = "hpa.gov.tw" in source["url"]
    article = soup.select_one("article.cpArticle")
    if is_hpa:
        article = (
            soup.select_one(".contentBlock")
            or soup.select_one(".article-content")
            or soup.select_one(".articleContent")
        )
        if article is None:
            article = soup.select_one("#ContentPlaceHolder1_divContent")
    if article is None:
        raise ValueError(f"{source['id']} 正文选择器失效，拒绝索引整站导航")
    for el in article.select("script, style, iframe, img, figure, video, audio"):
        el.decompose()
    raw_paragraphs = [
        s.strip() for s in article.get_text("\n", strip=True).splitlines() if s.strip()
    ]
    paragraphs = [normalize(s) for s in raw_paragraphs if normalize(s)]
    title_el = soup.select_one("h2") if not is_hpa else soup.select_one(".pageTitle")
    title = normalize(title_el.get_text(" ", strip=True) if title_el else soup.title.get_text())
    metadata = soup.select_one(".contentTop") if is_hpa else soup.select_one(".poInfo")
    meta = metadata.get_text(" ", strip=True) if metadata else ""
    if sum(map(len, paragraphs)) < 100:
        raise ValueError(f"{source['id']} 正文过短，请检查原始页面")
    text = "\n".join(paragraphs)
    return {
        **source,
        "title": title,
        "text": text,
        "original_text": "\n".join(raw_paragraphs),
        "metadata_original": meta,
        "units": canonical_units(source["id"], paragraphs),
        "text_sha256": digest(text),
        "fetched_at": now(),
        "review_status": "ai_extraction_pending_group_review",
    }


def fetch_sources() -> dict:
    """开发期构建；复现不能重新爬取活网页替代冻结版本。"""
    manifest = read_json(ROOT / "configs" / "sources.json")
    target = ROOT / "data" / "raw"
    target.mkdir(parents=True, exist_ok=True)
    docs, errors = [], []
    with httpx.Client(timeout=45, follow_redirects=True) as client:
        for key, url in manifest["license_urls"].items():
            response = client.get(url)
            response.raise_for_status()
            (target / f"license_{key}.html").write_bytes(response.content)
        for source in manifest["sources"]:
            try:
                response = client.get(source["url"])
                response.raise_for_status()
                response.encoding = "utf-8"
                (target / f"{source['id']}.html").write_bytes(response.content)
                doc = parse_article(response.text, source)
                license_key = "hpa" if "hpa.gov.tw" in source["url"] else "mohw"
                doc.update(
                    {
                        "license": manifest["license"],
                        "license_url": manifest["license_urls"][license_key],
                        "raw_sha256": digest(response.content),
                    }
                )
                docs.append(doc)
                print(f"{source['id']}: {doc['title']} ({len(doc['text'])}字)", flush=True)
            except Exception as exc:
                errors.append({"source": source["id"], "error": str(exc)})
                print(f"{source['id']}: 失败 {exc}", flush=True)
    out = ROOT / "data" / "corpus.jsonl"
    out.write_text(
        "".join(json.dumps(d, ensure_ascii=False) + "\n" for d in docs),
        encoding="utf-8",
    )
    info = {
        "version": manifest["version"],
        "status": manifest["status"],
        "documents": len(docs),
        "characters": sum(len(d["text"]) for d in docs),
        "sha256": digest(out.read_bytes()),
        "errors": errors,
        "created_at": now(),
    }
    write_json(ROOT / "data" / "corpus-manifest.json", info)
    return info


def load_corpus() -> list[dict]:
    path = ROOT / "data" / "corpus.jsonl"
    if not path.exists():
        raise FileNotFoundError("缺少冻结资料，请先运行 campus-health prepare")
    manifest = read_json(ROOT / "data" / "corpus-manifest.json")
    if digest(path.read_bytes()) != manifest["sha256"]:
        raise ValueError("语料SHA256不符，停止运行；请恢复版本，不能静默使用修改数据")
    return read_jsonl(path)


def prepare() -> None:
    """从固定release下载，不回退到最新网页。"""
    if (ROOT / "data" / "corpus.jsonl").exists():
        docs = load_corpus()
        print(f"本地资料校验通过：{len(docs)}份")
        return
    release_file = ROOT / "configs" / "data-release.json"
    if not release_file.exists():
        raise RuntimeError("资料release尚未发布；开发使用 fetch-sources，正式复现不得爬取新版")
    release = read_json(release_file)
    archive = ROOT / "data" / "download.tar.gz"
    archive.parent.mkdir(parents=True, exist_ok=True)
    with httpx.Client(timeout=90, follow_redirects=True) as client:
        response = client.get(release["url"])
        response.raise_for_status()
    if digest(response.content) != release["sha256"]:
        raise ValueError("下载包校验不符")
    archive.write_bytes(response.content)
    with tarfile.open(archive) as tar:
        for member in tar.getmembers():
            if member.name not in release["files"] or not member.isfile():
                raise ValueError("资料包包含非预期文件")
            handle = tar.extractfile(member)
            (ROOT / "data" / Path(member.name).name).write_bytes(handle.read())
    load_corpus()
