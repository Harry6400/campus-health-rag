from __future__ import annotations

import json
import re
from pathlib import Path

import jieba
import numpy as np
from rank_bm25 import BM25Okapi

from .data import normalize
from .storage import ROOT, digest, read_json, write_json

jieba.setLogLevel(30)
_jieba_cache = ROOT / ".cache" / "jieba"
_jieba_cache.mkdir(parents=True, exist_ok=True)
jieba.dt.tmp_dir = str(_jieba_cache)


def tokenize(text: str, mode: str = "jieba") -> list[str]:
    text = normalize(text).lower()
    if mode == "char2":
        blocks = re.findall(r"[\u4e00-\u9fff]+|[a-z0-9]+", text)
        return [
            token
            for b in blocks
            for token in (
                [b] if len(b) < 2 or b.isascii() else [b[i : i + 2] for i in range(len(b) - 1)]
            )
        ]
    if mode != "jieba":
        raise ValueError("未知分词方式")
    return [t for t in jieba.lcut(text) if re.search(r"[\w\u4e00-\u9fff]", t)]


def model_path(settings: dict) -> Path:
    return ROOT / "models" / settings["model_id"].replace("/", "--") / settings["model_revision"]


def ensure_model(settings: dict) -> Path:
    from huggingface_hub import snapshot_download

    path = model_path(settings)
    if not all((path / f).exists() for f in ["config.json", "model.safetensors", "tokenizer.json"]):
        snapshot_download(
            repo_id=settings["model_id"],
            revision=settings["model_revision"],
            local_dir=path,
            ignore_patterns=["onnx/*", "*.bin", "*.h5", "*.ot"],
        )
    checksums = ROOT / "configs" / "model-files.json"
    if checksums.exists():
        for name, expected in read_json(checksums).items():
            p = path / name
            if not p.exists() or digest(p.read_bytes()) != expected:
                raise ValueError(f"固定模型文件校验失败：{name}")
    return path


def build_chunks(docs: list[dict], settings: dict, tokenizer) -> list[dict]:
    budget, overlap = settings["chunk_tokens"], settings["overlap_tokens"]
    if not 0 <= overlap < budget:
        raise ValueError("overlap必须非负且小于chunk_tokens")
    chunks = []
    for doc in docs:
        units, start = doc["units"], 0
        while start < len(units):
            end = start
            while end < len(units):
                text = doc["title"] + "\n" + "".join(u["text"] for u in units[start : end + 1])
                length = len(tokenizer.encode(text, add_special_tokens=True))
                if length > min(budget, tokenizer.model_max_length):
                    if end == start:
                        raise ValueError(f"证据单位过长：{units[start]['id']}；不允许静默截断")
                    break
                end += 1
            selected = units[start:end]
            chunks.append(
                {
                    "id": f"{doc['id']}:c{start:04d}-{end:04d}",
                    "doc_id": doc["id"],
                    "title": doc["title"],
                    "theme": doc["theme"],
                    "url": doc["url"],
                    "source_metadata": doc.get("metadata_original", ""),
                    "text": "".join(u["text"] for u in selected),
                    "unit_ids": [u["id"] for u in selected],
                }
            )
            if end >= len(units):
                break
            next_start = end
            while next_start > start + 1:
                candidate = "".join(u["text"] for u in units[next_start - 1 : end])
                if len(tokenizer.encode(candidate, add_special_tokens=False)) > overlap:
                    break
                next_start -= 1
            start = next_start
    return chunks


class Retriever:
    def __init__(self, docs: list[dict], settings: dict, dense: bool = True):
        import random

        import torch
        from transformers import AutoTokenizer

        random.seed(settings["seed"])
        np.random.seed(settings["seed"])
        torch.manual_seed(settings["seed"])
        torch.set_num_threads(4)
        torch.use_deterministic_algorithms(True)
        self.settings = settings
        path = ensure_model(settings)
        self.tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
        self.chunks = build_chunks(docs, settings, self.tokenizer)
        if not self.chunks:
            raise ValueError("资料库没有可检索段落")
        tokens = [
            tokenize(c["title"] + "\n" + c["text"], settings["tokenizer"]) for c in self.chunks
        ]
        self.bm25 = BM25Okapi(tokens)
        self.embedder, self.vectors = None, None
        if dense:
            from sentence_transformers import SentenceTransformer

            self.embedder = SentenceTransformer(str(path), device="cpu", local_files_only=True)
            texts = [c["title"] + "\n" + c["text"] for c in self.chunks]
            key = digest(
                json.dumps(
                    {"texts": texts, "revision": settings["model_revision"]},
                    ensure_ascii=False,
                    sort_keys=True,
                )
            )
            cache = ROOT / "data" / "index" / key
            cache.mkdir(parents=True, exist_ok=True)
            target = cache / "vectors.npy"
            if target.exists():
                self.vectors = np.load(target, allow_pickle=False)
                if self.vectors.shape[0] != len(texts):
                    raise ValueError("向量缓存行数不符")
            else:
                self.vectors = self.embedder.encode(
                    texts, normalize_embeddings=True, show_progress_bar=False, batch_size=32
                )
                np.save(target, self.vectors, allow_pickle=False)
            write_json(cache / "chunks.json", self.chunks)

    @staticmethod
    def order(scores) -> list[int]:
        return sorted(range(len(scores)), key=lambda i: (-float(scores[i]), i))

    def search(self, query: str, method: str, k: int | None = None) -> list[dict]:
        query = normalize(query)
        if not query or len(query) > 2000:
            raise ValueError("请输入1—2000字符的问题")
        k = k or self.settings["top_k"]
        if k < 1:
            raise ValueError("top-k必须为正整数")
        if method not in {"B0", "R1", "R2", "R3"}:
            raise ValueError("未知方法")
        if len(self.tokenizer.encode(query, add_special_tokens=True)) > 450:
            raise ValueError("问题超出嵌入模型输入预算，请缩短；不会静默截断")
        sparse = self.bm25.get_scores(tokenize(query, self.settings["tokenizer"]))
        if method in {"B0", "R1"}:
            scores = sparse
        else:
            if self.embedder is None:
                raise RuntimeError("当前索引未启用句向量")
            q = self.embedder.encode(
                [self.settings["query_instruction"] + query],
                normalize_embeddings=True,
                show_progress_bar=False,
            )[0]
            dense = self.vectors @ q
            scores = dense
            if method == "R3":
                scores = np.zeros(len(self.chunks), dtype=float)
                for ranking in [self.order(sparse), self.order(dense)]:
                    for rank, index in enumerate(ranking[: self.settings["rrf_depth"]], 1):
                        scores[index] += 1 / (self.settings["rrf_k"] + rank)
        return [
            {**self.chunks[i], "score": float(scores[i]), "rank": rank}
            for rank, i in enumerate(self.order(scores)[:k], 1)
        ]
