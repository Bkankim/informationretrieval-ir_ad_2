# 과학 지식 질의 응답 시스템
## Team

| ![박패캠](https://avatars.githubusercontent.com/u/156163982?v=4) | ![이패캠](https://avatars.githubusercontent.com/u/156163982?v=4) | ![최패캠](https://avatars.githubusercontent.com/u/156163982?v=4) | ![김패캠](https://avatars.githubusercontent.com/u/156163982?v=4) | ![오패캠](https://avatars.githubusercontent.com/u/156163982?v=4) |
| :--------------------------------------------------------------: | :--------------------------------------------------------------: | :--------------------------------------------------------------: | :--------------------------------------------------------------: | :--------------------------------------------------------------: |
|            [박패캠](https://github.com/UpstageAILab)             |            [이패캠](https://github.com/UpstageAILab)             |            [최패캠](https://github.com/UpstageAILab)             |            [김패캠](https://github.com/UpstageAILab)             |            [오패캠](https://github.com/UpstageAILab)             |
|                            팀장, 담당 역할                             |                            담당 역할                             |                            담당 역할                             |                            담당 역할                             |                            담당 역할                             |

## 0. Overview
### Environment
- OS: Linux (WSL2) x86_64
- GPU: NVIDIA RTX 3090
- Python 3.12
- Elasticsearch 8.8.0 (analysis-nori)
- LLM: Upstage Solar(OpenAI 호환) / 옵션으로 OpenAI, LangGraph 기반 파이프라인
- Embedding: solar-embedding-1-large-(passage/query) → 4096d → 랜덤 프로젝션 1536d

### Requirements
- `uv sync` 또는 `pip install -r code/requirements.txt`
- `.env` (예시: `code/.env.example`)
  - `ES_HOST`(default `http://localhost:9200`), `ES_USERNAME`, `ES_PASSWORD`, `ES_CA_CERT`, `ES_INDEX`(default `test`)
  - `SOLAR_API_KEY` 또는 `OPENAI_API_KEY` (둘 중 하나 필수)
  - 선택: `LLM_MODEL`(default `solar-pro2`), `LLM_BASE_URL`(default `https://api.upstage.ai/v1`)

## 1. Competiton Info

### Overview
- 대회명: 과학 지식 질의 응답 시스템

### Timeline
- 2025.11.14 - Start Date
- 2025.11.27 - Final submission deadline

## 2. Components

### Directory
```
├── README.md
├── pyproject.toml
├── uv.lock
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
- Rank: (비움)
- Score: MAP 0.7909 / MRR 0.7939 (`code/experiments/experiment-log.md` 기준)

### Presentation
- (발표 자료 링크 추가 예정)

## etc

### Meeting Log
- (회의록 링크 추가 예정)

### Reference
- LangGraph, Elasticsearch 8.8.0 + analysis-nori, Upstage Solar Embedding/LLM, hybrid BM25+dense, 비과학 규칙 필터
