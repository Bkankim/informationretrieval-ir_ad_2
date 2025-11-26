# 과학 질의응답 IR 대회 (RAG) — Solar + Rank Graph 리팩토링

## Quick Facts
- 기간/팀: 2주, 4인
- 역할(본인): 원본 랭킹 로직을 그래프 기반 재랭킹으로 리팩토링, 프롬프트 고도화·튜닝
- 문제: 과학 문서에서 질문에 맞는 증거/답변 검색·랭킹 품질이 낮아 MAP/MRR 저조
- 지표: MAP/MRR 0.4242 → 0.8795 / 0.8818 (+107%, Public LB)
- 스택: Solar Embedding, Rank Graph, LangGraph, Elasticsearch, Prompt Tuning, Python

## Overview
과학 도메인 질의응답 대회에서 검색·랭킹 품질이 부족해 정답 회수가 낮았습니다. 2주 안에 RAG 파이프라인을 재설계하고 랭킹 로직을 그래프 기반으로 리팩토링해 MAP/MRR을 크게 끌어올리는 것이 목표였습니다.

## Results
- MAP: 0.4242 → 0.8795
- MRR: 0.4242 → 0.8818
- Public Leaderboard 기준 +107% 개선

## Stack
- 임베딩/랭킹: Solar Embedding, Rank Graph 리팩토링(노이즈 노드 제거·가중치 조정), Top-k·스코어 컷 튜닝
- RAG/파이프라인: LangGraph 기반 RAG, Elasticsearch BM25, 비과학 필터링
- LLM/프롬프트: Prompt Tuning(쿼리 리라이트·증거 강조), 응답 포맷 일관성
- 실행/환경: Python, `uv`/pip, `.env`로 키/ES 설정, Shell 스크립트로 ES 설치/기동
- 평가: MAP, MRR

## Approach (STAR)
- Situation: 초기 MAP/MRR 0.4242/0.4242로 과학 QA 검색·랭킹 품질 저조.
- Task: 2주 내 랭킹 지표 대폭 개선; 나는 랭킹·프롬프트·튜닝 담당.
- Action:
  - Solar 임베딩 도입, **원본 랭킹 로직을 그래프 기반 재랭킹으로 리팩토링**하며 노이즈 노드 제거·가중치 재설정
  - Top-k, 스코어 컷, 후보 필터링 튜닝으로 검색 후보 품질 개선
  - 프롬프트 고도화: 쿼리 리라이트, 증거 강조 템플릿, 포맷 일관성 확보
  - MAP/MRR 기반 실험 버전 관리·비교로 빠른 피드백 루프 구축
- Result: MAP/MRR 0.4242 → 0.8795 / 0.8818(+107%) 달성

## How to Run
> 실제 키/경로에 맞게 `.env`와 옵션을 설정하세요.
```bash
# 0) 의존성 설치
uv sync  # 또는 pip install -r code/requirements.txt

# 1) 환경 변수(.env) 설정
cp code/.env.example code/.env
# .env에 ES_HOST/ES_USERNAME/ES_PASSWORD/ES_CA_CERT, SOLAR_API_KEY 혹은 OPENAI_API_KEY 기입

# 2) Elasticsearch 설치/기동 (필요 시)
bash code/install_elasticsearch.sh
bash code/run_elasticsearch.sh
# 종료: bash code/stop_elasticsearch.sh

# 3) LangGraph RAG 실행 (인덱스 재사용 시 --skip-index)
uv run python code/scripts/rag_with_langgraph.py --skip-index --alpha 0.5 --topk 3

# 4) 제출/검증
# 결과 파일 예: code/sample_submission_hybrid2.csv
```

## Repo Structure (핵심 파일)
```
README.md
pyproject.toml
uv.lock
code/
  README.md               # LangGraph 기반 RAG 실행 가이드
  requirements.txt
  .env.example
  install_elasticsearch.sh
  run_elasticsearch.sh
  stop_elasticsearch.sh
  rag_with_elasticsearch.py
  run_once.py
  scripts/
    rag_with_langgraph.py # LangGraph RAG 실행 스크립트
  pipelines/
    langgraph_pipeline.py
    rag_callbacks.py
  retrieval/
    retriever.py
    elasticsearch_utils.py
    non_science.py        # 비과학 필터링
  llm/
    embedding.py
    generators.py
  config/
    settings.py
  data/
    documents.jsonl
    eval.jsonl
  experiments/
    experiment-log.md
    langgraph-skeleton.md
```

## Lessons / Next
- 그래프 리팩토링 + 프롬프트 고도화가 랭킹 지표 개선에 직접 기여.
- 다음: 도메인 확장, 경량 임베딩 실험, 하이브리드 검색(벡터+BM25), 에러 분석 자동화.

## Contact
- Author: Byeonghyeon Kim (랭크 그래프 리팩토링·프롬프트·튜닝)
- GitHub: https://github.com/Bkankim/informationretrieval-ir_ad_2
