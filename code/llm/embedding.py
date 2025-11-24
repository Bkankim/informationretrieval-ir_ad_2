# llm/embedding.py

from __future__ import annotations

import os
from typing import Dict, List, Optional

import numpy as np
from openai import OpenAI
from tqdm import tqdm


class EmbeddingService:
    """
    Upstage Solar Embedding + 차원 축소(랜덤 프로젝션) + tqdm 진행률 표시

    - 원래 Upstage 임베딩: 4096차원 (solar-embedding-1-large-* 계열)  :contentReference[oaicite:4]{index=4}
    - ES 제한: dense_vector 최대 1536차원 (현재 환경) :contentReference[oaicite:5]{index=5}

    -> 여기서는 4096 → 1536 로 줄여서 ES에 넣도록 구현.
    """

    def __init__(
        self,
        passage_model: str = "solar-embedding-1-large-passage",
        query_model: str = "solar-embedding-1-large-query",
        output_dim: int = 1536,       # ES에 넣을 최종 차원 (<= 2048 이어야 함)
        projection_seed: int = 42,    # 문서/쿼리 모두 같은 랜덤 프로젝션 사용
    ):
        api_key = os.getenv("SOLAR_API_KEY") or os.getenv("OPENAI_API_KEY")
        if api_key is None:
            raise RuntimeError("SOLAR_API_KEY 또는 OPENAI_API_KEY 중 하나는 반드시 설정해야 합니다.")

        base_url = os.getenv("EMBEDDING_BASE_URL")
        if base_url is None:
            # LLM_BASE_URL 이 이미 /v1/solar 로 되어 있으면 그대로 사용
            llm_base = os.getenv("LLM_BASE_URL", "https://api.upstage.ai/v1")
            llm_base = llm_base.rstrip("/")
            if llm_base.endswith("/solar"):
                base_url = llm_base
            else:
                base_url = llm_base + "/solar"

        self._client = OpenAI(api_key=api_key, base_url=base_url)

        self.passage_model = passage_model
        self.query_model = query_model

        self.output_dim = output_dim
        self.projection_seed = projection_seed
        self._proj_matrix: Optional[np.ndarray] = None  # lazy init

    # -----------------------------
    # 내부 유틸
    # -----------------------------
    def _ensure_projection(self, input_dim: int) -> Optional[np.ndarray]:
        """
        input_dim (ex: 4096)에 대해 output_dim (ex: 1536)으로
        줄이는 랜덤 프로젝션 행렬을 lazy-initialize.
        """
        if self.output_dim is None:
            return None
        if input_dim <= self.output_dim:
            # 이미 ES 제한 이내면 굳이 줄이지 않음
            return None

        if self._proj_matrix is None:
            rng = np.random.default_rng(self.projection_seed)
            # (input_dim, output_dim) 행렬 생성
            # 1/sqrt(output_dim) 스케일로 정규분포 -> Johnson–Lindenstrauss 계열 랜덤 프로젝션
            self._proj_matrix = (
                rng.standard_normal(size=(input_dim, self.output_dim), dtype=np.float32)
                / np.sqrt(self.output_dim)
            )
        return self._proj_matrix

    def _embed(self, texts: List[str], model: str) -> List[List[float]]:
        """
        Upstage(OpenAI 호환) Embeddings API 호출 후,
        필요시 랜덤 프로젝션으로 차원 축소.
        """
        resp = self._client.embeddings.create(model=model, input=texts)
        raw_vectors = [item.embedding for item in resp.data]  # len == 4096

        if not raw_vectors:
            return []

        input_dim = len(raw_vectors[0])
        proj = self._ensure_projection(input_dim)

        if proj is None:
            # projection 안 쓰는 경우 (차원이 이미 <= output_dim)
            return raw_vectors

        # (N, input_dim) @ (input_dim, output_dim) -> (N, output_dim)
        arr = np.asarray(raw_vectors, dtype="float32")
        reduced = arr @ proj
        return reduced.tolist()

    # -----------------------------
    # 외부 인터페이스
    # -----------------------------
    # 쿼리 인코딩 (검색 질의)
    def encode(self, sentences: List[str]):
        return self._embed(sentences, self.query_model)

    # 문서 인코딩 (색인용)
    def encode_documents(self, docs: List[Dict[str, str]], batch_size: int = 100):
        embeddings: List[List[float]] = []
        for idx in tqdm(
            range(0, len(docs), batch_size),
            desc="Embedding documents (Upstage Solar → projected)",
            unit="batch",
        ):
            batch = docs[idx : idx + batch_size]
            contents = [doc["content"] for doc in batch]
            batch_embeds = self._embed(contents, self.passage_model)
            embeddings.extend(batch_embeds)
        return embeddings
