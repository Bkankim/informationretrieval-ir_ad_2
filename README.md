# 과학 지식 질의 응답 시스템
**[Technical Case Study] Solar Embedding 기반 고성능 RAG 구축기**

> **🏆 Key Achievement**
> **Upstage Solar Embedding** 최적화를 통해 Baseline 대비 **검색 정확도(MAP) 107% 향상** (0.4242 → 0.8795) 달성.

## Team

| <img width="135" height="134" alt="image" src="https://github.com/user-attachments/assets/be58e056-194d-49f6-a3f7-8be86858774b" />| <img width="134" height="136" alt="image" src="https://github.com/user-attachments/assets/c20eaf3f-5939-4066-b425-590cf8d2d46a" /> | <img width="135" height="134" alt="image" src="https://github.com/user-attachments/assets/a2fa6cd6-fefe-41cf-8c65-7e314c503c3a" /> | <img width="135" height="135" alt="image" src="https://github.com/user-attachments/assets/18d39882-d1ba-4103-94c4-f8cead3088d5" /> | <img width="130" height="127" alt="image" src="https://github.com/user-attachments/assets/83acdead-ad49-4980-bba1-fbbbf1c6af8f" />|
| :--------------------------------------------------------------: | :--------------------------------------------------------------: | :--------------------------------------------------------------: | :--------------------------------------------------------------: | :--------------------------------------------------------------: |
|            김명철             |            김수환             |            김병현             |            송인섭             |            이가은             |
|                            팀장, PM, 데이터분석                             |                            코드/로직 고도화                             |                            랭그래프 리팩토링, 비과학 질의 고도화                             |                            프롬프트 고도화                             |                            Hybrid  로직 연구                             |

## 0. Executive Summary (Project Context)
과학 상식 관련 질의응답 시스템을 구축하는 프로젝트입니다. 단순 검색(BM25)만으로는 해결하기 어려운 복잡한 과학적 문맥을 이해하기 위해, **Upstage Solar Embedding**을 도입하여 Semantic Search 성능을 극대화했습니다.

**[Architecture Overview]**
![System Architecture](assets/architecture.png)

### Why This Project? (Technical Insights)
- **Challenge**: 초기 BM25 모델은 키워드 매칭에 의존하여, 문맥적 의미가 중요한 과학 질문(예: 인과관계 추론)에서 낮은 정확도(MAP 0.42)를 보였습니다.
- **Solution**: **Upstage Solar Embedding**을 활용한 Dense Search 파이프라인을 설계했습니다.
- **Result**: 검색 품질이 비약적으로 상승(MAP 0.88)하여, LLM이 정답을 생성하는 데 필요한 '증거 문서'를 거의 완벽하게 찾아낼 수 있었습니다.

### Environment
- OS: Linux (Ubuntu)
- GPU: NVIDIA RTX 3090
- Python 3.12
- Elasticsearch 8.8.0 (analysis-nori)
- LLM: Upstage Solar(OpenAI 호환) / 옵션으로 OpenAI, LangGraph 기반 파이프라인
- Embedding: solar-embedding-1-large-(passage/query) → 4096d → 랜덤 프로젝션 1536d

### Requirements
- `uv sync` 또는 `pip install -r code/requirements.txt`
- `.env` (예시: `code/.env.example`)
  - `ES_HOST`(default `http://localhost:9200`), `ES_USERNAME`, `ES_PASSWORD`, `ES_CA_CERT`, `ES_INDEX`(default `test`)
  - **`SOLAR_API_KEY` (필수)** 또는 `OPENAI_API_KEY`
  - 선택: `LLM_MODEL`(default `solar-pro2`), `LLM_BASE_URL`(default `https://api.upstage.ai/v1`)

## 1. Project Background (Competition Info)

### Overview
- 프로젝트명: 과학 지식 질의 응답 시스템 고도화

### Timeline
- 2025.11.14 - Start Date
- 2025.11.27 - Final submission deadline
- **Presentation**: [📄 프로젝트 발표 자료 (PDF) 보기](assets/presentation.pdf)

## 2. Components

### Directory
```
├── README.md
├── pyproject.toml
├── uv.lock
├── assets/                        # 발표자료 및 다이어그램
└── code
    ├── .env.example
    ├── README.md                      # LangGraph 실행 가이드
    ├── config/                        # 환경 설정 로더
    ├── data/                          # documents.jsonl, eval.jsonl (로컬 포함)
    ├── experiments/                   # 실험 로그(MAP/MRR 기록)
    ├── llm/                           # 임베딩/생성 모듈
    ├── pipelines/                     # LangGraph 스켈레톤, 콜백 정의
    ├── retrieval/                     # ES 유틸, hybrid BM25+dense, 비과학 필터
    ├── scripts/rag_with_langgraph.py  # 메인 실행 스크립트
    ├── rag_with_elasticsearch.py      # 상위 경로 호환 래퍼
    ├── install_elasticsearch.sh / run_elasticsearch.sh / stop_elasticsearch.sh
    └── requirements.txt
```

## 3. Data descrption

### Dataset overview
- `code/data/documents.jsonl`: 한국어 과학/상식 문단(`docid`, `content`)
- `code/data/eval.jsonl`: 평가 질의(`eval_id`, `msg` 멀티턴 대화 포함)
- 출력: `code/sample_submission_hybrid2.csv` (실행 시 생성, 기본 존재하지 않음)

### EDA
- 형식 검증 정도로 사용(별도 통계/시각화 없음)

### Data Processing
- Solar 임베딩 4096d → 1536d 랜덤 프로젝션 후 ES dense_vector 저장(cosine)
- BM25(nori analyzer) + dense KNN 점수 정규화 후 hybrid 검색(α 가중)
- 비과학 질의 정규식 필터로 검색/생성 스킵(topk 비움)

## 4. Modeling

### Model description
- Retriever: Elasticsearch BM25(`match`, nori) + dense KNN, hybrid 가중(α)
- Embedding: Upstage Solar passage/query 쌍, Johnson–Lindenstrauss 투영(1536d)
- Generator: Upstage Solar(OpenAI 호환) LLM, LangGraph 노드로 orchestration
- 비과학 필터: 규칙 기반 정규식(`retrieval/non_science.py`)

### Modeling Process
- 1) .env 로드 → 2) (옵션) 인덱스 재생성 + 임베딩 색인 → 3) 비과학 판별 → 4) LLM이 standalone query 생성 → 5) hybrid 검색(topk=3 기본, α=0.5 기본) → 6) LLM 최종 답변 → 7) `sample_submission_hybrid2.csv` 저장

## 5. Result

### Leader Board
- Rank: 1등
- Score:MAP 0.8795`, `MRR 0.8818

### Performance Metric (Leader Board)
- **Final Score**: MAP **0.7909** / MRR **0.7939** (Base Experiment)
- **Best Hybrid Setting**: MAP **0.8795** / MRR **0.8818**
    - 단순히 BM25에 의존했을 때보다, **Solar Embedding**을 통한 Dense Retrieval을 사용했을 때 약 2배 이상의 성능 향상을 확인했습니다.
    - 특히 묻는 대상의 속성이 복잡하게 얽힌 과학 지문에서 Solar Embedding의 문맥 이해도가 결정적이었습니다.
### Reference
- LangGraph, Elasticsearch 8.8.0 + analysis-nori, **Upstage Solar Embedding/LLM**, dense-only, 비과학 규칙 필터
