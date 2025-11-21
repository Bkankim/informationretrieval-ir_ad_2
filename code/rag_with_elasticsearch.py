import os
import json
import time
import traceback

from elasticsearch import Elasticsearch, helpers
from sentence_transformers import SentenceTransformer
from openai import OpenAI
from dotenv import load_dotenv

# -------------------------------
# 1. .env 파일 불러오기
# -------------------------------
# .env 파일이 code/에 있든, 상위 폴더에 있든 자동 검색됨
load_dotenv()

ES_USERNAME = os.getenv("ES_USERNAME")
ES_PASSWORD = os.getenv("ES_PASSWORD")
ES_CA_CERT = os.getenv("ES_CA_CERT")   # ex) ./http_ca.crt
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
LLM_MODEL= "gpt-4o-mini" # gpt-4o-mini에서 변경

# 환경변수 확인(디버깅용)
print("[INFO] Loaded environment variables:")
print(f"ES_USERNAME: {ES_USERNAME}")
print(f"ES_PASSWORD: {'****' if ES_PASSWORD else None}")
print(f"ES_CA_CERT: {ES_CA_CERT}")
print(f"OPENAI_API_KEY: {'****' if OPENAI_API_KEY else None}")

# -------------------------------
# 2. SentenceTransformer 로드
# -------------------------------
model = SentenceTransformer("snunlp/KR-SBERT-V40K-klueNLI-augSTS")


def get_embedding(sentences):
    return model.encode(sentences)


def get_embeddings_in_batches(docs, batch_size=100):
    batch_embeddings = []
    for i in range(0, len(docs), batch_size):
        batch = docs[i:i + batch_size]
        contents = [doc["content"] for doc in batch]
        embeddings = get_embedding(contents)
        batch_embeddings.extend(embeddings)
        print(f"batch {i}")
    return batch_embeddings


# -------------------------------
# 3. Elasticsearch 연결 설정
# -------------------------------
# ES_CA_CERT는 절대경로 또는 상대경로 모두 허용됩니다.
# 예: code/http_ca.crt 또는 C:/Users/.../http_ca.crt
ES_HOST = os.getenv("ES_HOST", "http://localhost:9200")

es = Elasticsearch(
    ES_HOST,
    request_timeout=30,
)

print(es.info())

# -------------------------------
# 4. ES 인덱스 operation
# -------------------------------
def create_es_index(index, settings, mappings):
    if es.indices.exists(index=index):
        es.indices.delete(index=index)
    es.indices.create(index=index, settings=settings, mappings=mappings)


def delete_es_index(index):
    es.indices.delete(index=index)


def bulk_add(index, docs):
    actions = [{"_index": index, "_source": doc} for doc in docs]
    return helpers.bulk(es, actions)


# -------------------------------
# 4-1. sparse / dense 리트리버
# -------------------------------
def sparse_retrieve(query_str, size):
    query = {
        "match": {
            "content": {
                "query": query_str
            }
        }
    }
    result = es.search(index="test", query=query, size=size, sort="_score")
    return result


def dense_retrieve(query_str, size):
    query_embedding = get_embedding([query_str])[0]
    knn = {
        "field": "embeddings",
        "query_vector": query_embedding.tolist(),
        "k": size,
        "num_candidates": 100
    }
    return es.search(index="test", knn=knn)


def hybrid_retrieve(query_str, size, alpha=0.5): # 하이브리드 함수
    """
    sparse(BM25)와 dense(KNN) 결과를 가중 합으로 섞는 하이브리드 검색.
    - alpha: sparse 가중치 (0~1). 0.5면 동등한 비중.
    """
    # 각각 검색
    sparse = sparse_retrieve(query_str, size)
    dense = dense_retrieve(query_str, size)

    combined = {}

    def normalized_and_add(results, weight):
        hits = results.get("hits", {}).get("hits", [])
        if not hits:
            return
        # 점수 정규화
        max_score = max(h["_score"] for h in hits) or 1.0
        for h in hits:
            src = h.get("_source", {})
            docid = src.get("docid")
            if docid is None:
                # docid 없으면 스킵
                continue
            norm_score = (h["_score"] / max_score) * weight
            if docid not in combined:
                combined[docid] = {
                    "_source": src,
                    "_score": 0.0,
                }
            combined[docid]["_score"] += norm_score

    # sparse / dense 각각 반영
    normalized_and_add(sparse, alpha)
    normalized_and_add(dense, 1 - alpha)

    # 점수 순으로 정렬 후 상위 size개 선택
    merged_hits = sorted(
        [
            {"_source": v["_source"], "_score": v["_score"]}
            for v in combined.values()
        ],
        key=lambda x: x["_score"],
        reverse=True,
    )[:size]

    # sparse_retrieve와 비슷한 형태로 반환
    return {"hits": {"hits": merged_hits}}


def rrf_fusion(result_list, k=60):
    """
    여러 검색 결과 리스트를 Reciprocal Rank Fusion(RRF)으로 합치는 유틸리티 함수.

    Parameters
    ----------
    result_list : list[list[dict]]
        Elasticsearch search().get("hits", {}).get("hits", []) 형태의 결과 리스트.
        예: [sparse_hits, dense_hits]
    k : int
        RRF 상수. 일반적으로 10~60 사이 값을 사용.

    Returns
    -------
    dict
        {docid: {"_source": ..., "_score": rrf_score}} 형태의 딕셔너리
    """
    fused = {}

    for hits in result_list:
        if not hits:
            continue

        # 각 리트리버에서의 순위 (1위부터 시작)
        for rank, h in enumerate(hits, start=1):
            src = h.get("_source", {})
            docid = src.get("docid")
            if docid is None:
                # docid 없는 문서는 스킵
                continue

            # RRF 점수: 1 / (k + rank)
            score = 1.0 / (k + rank)

            if docid not in fused:
                fused[docid] = {
                    "_source": src,
                    "_score": 0.0,
                }
            fused[docid]["_score"] += score

    return fused


def hybrid_retrieve_rrf(query_str, size, k=60, per_retriever_k=None):
    """
    BM25 기반 sparse 검색과 dense 벡터 검색 결과를
    Reciprocal Rank Fusion(RRF)으로 합치는 하이브리드 검색 함수.

    Parameters
    ----------
    query_str : str
        검색 질의 문자열
    size : int
        최종으로 반환할 문서 수
    k : int
        RRF 상수 (기본 60)
    per_retriever_k : int or None
        각 리트리버에서 가져올 상위 문서 수.
        None이면 size와 동일하게 사용.
    """
    if per_retriever_k is None:
        # 검색 풀을 넓게 가져왔다가 RRF로 재정렬
        per_retriever_k = max(size, 50)

    # 1) 개별 리트리버 실행
    sparse = sparse_retrieve(query_str, per_retriever_k)
    dense = dense_retrieve(query_str, per_retriever_k)

    sparse_hits = sparse.get("hits", {}).get("hits", [])
    dense_hits = dense.get("hits", {}).get("hits", [])

    # 2) RRF 점수 계산
    fused = rrf_fusion([sparse_hits, dense_hits], k=k)

    # 3) 점수 순으로 정렬 후 상위 size개 선택
    merged_hits = sorted(
        [
            {"_source": v["_source"], "_score": v["_score"]}
            for v in fused.values()
        ],
        key=lambda x: x["_score"],
        reverse=True,
    )[:size]

    # 4) 기존 sparse_retrieve / hybrid_retrieve와 동일한 형식으로 반환
    return {"hits": {"hits": merged_hits}}


# -------------------------------
# 5. Elasticsearch 인덱스 설정
# -------------------------------
settings = {
    "analysis": {
        "analyzer": {
            "nori": {
                "type": "custom",
                "tokenizer": "nori_tokenizer",
                "decompound_mode": "mixed",
                "filter": ["nori_posfilter"]
            }
        },
        "filter": {
            "nori_posfilter": {
                "type": "nori_part_of_speech",
                "stoptags": ["E", "J", "SC", "SE", "SF", "VCN", "VCP", "VX"]
            }
        }
    }
}

mappings = {
    "properties": {
        "docid": {
            "type": "keyword"
        },
        "src": {
            "type": "keyword"
        },
        "content": {
            "type": "text",
            "analyzer": "nori",
            "fields": {
                "keyword": {
                    "type": "keyword"
                }
            }
        },
        "embeddings": {
            "type": "dense_vector",
            "dims": 768,
            "index": True,
            "similarity": "l2_norm"
        }
    }
}

# -------------------------------
# 6. 인덱스 생성
# -------------------------------
create_es_index("test", settings, mappings)

# -------------------------------
# 7. 데이터 로드 및 임베딩
# -------------------------------
index_docs = []
with open("./data/documents.jsonl", encoding="utf-8") as f:
    docs = [json.loads(line) for line in f]

embeddings = get_embeddings_in_batches(docs)

for doc, embedding in zip(docs, embeddings):
    doc["embeddings"] = embedding.tolist()
    index_docs.append(doc)

ret = bulk_add("test", index_docs)
print(ret)


# -------------------------------
# 8. RAG 구현
# -------------------------------
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY # type: ignore
client = OpenAI(
    timeout=10
)

llm_model = LLM_MODEL

persona_function_calling = """
당신은 검색 증강 생성(Retrieval-Augmented Generation, RAG) 시스템을 위한 질의 변환 도우미입니다.
사용자의 질문과 이전 대화 맥락을 읽고, 검색에 최적인 '독립 질의(standalone_query)'를 만들어야 합니다.

[역할]
- 사용자의 실제 의도를 파악하여, 검색 엔진에 넣을 수 있는 한 문장의 한국어 질의를 생성합니다.
- 모호한 대명사나 지시어는 모두 구체적인 명사/개념/인물명으로 치환해야 합니다.
  - 예: "그 사람" → "알베르트 아인슈타인"
  - 예: "이 사건" → "워터게이트 사건"
- 필요하다면, 영어 고유명사(인명, 지명, 이론명 등)를 함께 병기합니다.

[입력 형식]
- msg: user와 assistant의 대화 히스토리 전체 (리스트 형태)
  - 각 원소는 {"role": "user" or "assistant", "content": "..."} 구조입니다.

[출력 형식]
- JSON 객체 형태로 다음 한 가지만 포함하세요.
  - "standalone_query": (검색에 사용할 한 문장의 한국어 질의)

[주의사항]
- "standalone_query"는 절대 공백 문자열이 되면 안 됩니다.
- 답변에는 JSON 이외의 다른 텍스트를 포함하지 마세요.
"""


tools = [{
    "type": "function",
    "function": {
        "name": "standalone_query",
        "description": "대화 맥락을 반영하여, 검색 엔진에 넣기 좋은 한 문장의 질의문을 생성합니다.",
        "parameters": {
            "type": "object",
            "properties": {
                "standalone_query": {
                    "type": "string",
                    "description": "검색에 사용할 최종 한 문장의 한국어 질의"
                }
            },
            "required": ["standalone_query"]
        }
    }
}]


qa_persona = """
당신은 한국어 과학·상식 질문에 답하는 전문 Q&A 어시스턴트입니다.
당신에게는 검색 시스템으로부터 가져온 문서 조각(retrieved_context)이 주어지며,
반드시 이 문서들의 내용을 우선적으로 활용하여 답변해야 합니다.

[입력 설명]
- msg: 사용자와의 전체 대화 히스토리입니다.
- retrieved_context: 검색기로부터 가져온 문서 목록입니다.
  - 각 문서는 문단 형태의 텍스트이며, ko_MMLU, ARC 등 시험/퀴즈에서 추출된 과학·상식 지식입니다.
  - 내용은 물리, 화학, 생물, 지구과학, 인물, 역사, 사회 상식 등입니다.

[답변 생성 원칙]
1. 검색 문서 우선
   - 가능한 한 retrieved_context에 있는 내용에 기반해서만 답변하세요.
   - 당신이 사전 지식으로 알고 있더라도, 코퍼스 내용과 충돌하면 코퍼스를 우선합니다.
   - 코퍼스에 명시된 사실이 있으면, 그 내용을 중심으로 정리해서 설명하세요.

2. 사실성 & 정직성
   - 문서들 어디에도 정보가 없거나, 내용이 너무 부족해서 확신할 수 없다면:
     - 지어내지 말고, 모른다고 솔직히 말한 뒤,
     - 코퍼스에서 알 수 있는 범위(예: 일반적인 경향, 정의 수준)까지만 설명하세요.
   - 예시:
     - "제공된 자료에는 X에 대한 구체적인 내용은 없지만, 일반적으로는 ..."
     - "검색된 문서만으로는 Y에 대해 확실히 말하기 어렵습니다. 다만, ..."

3. 멀티턴 맥락 반영
   - msg 전체를 보고 사용자의 실제 질문이 무엇인지 이해해야 합니다.
   - 이전 발화의 감정/의도(걱정, 호기심 등)를 가볍게 반영하면 좋습니다.
     - 예: "기억 상실증이 무섭게 느껴질 수 있어요. 자료에 따르면, 주요 원인은 …"

4. 답변 스타일
   - 한국어로 친절하고 명확하게 설명합니다.
   - 기본은 3~6문장 정도의 단락으로 답하고, 필요하면 짧은 목록을 사용하세요.
   - 핵심 정보 → 이유/근거 → 간단한 정리 순서를 지향합니다.
   - 수치/연도/전문 용어는 가능하면 구체적으로 제시합니다.

[출력 형식]
- 한국어 자연문으로만 답변합니다. 불필요한 메타 설명(예: "다음은 답변입니다")은 넣지 마세요.
"""


def safe_chat_completion(
    max_retries=3,
    backoff_base=2,
    **kwargs
):
    """
    OpenAI ChatCompletion 호출 시 예외를 캐치하고
    지수 백오프로 여러 번 재시도하는 래퍼 함수.
    """
    for attempt in range(1, max_retries + 1):
        try:
            return client.chat.completions.create(**kwargs)
        except Exception as e:
            print(f"[WARN] OpenAI API 실패 (시도 {attempt}/{max_retries}): {e}")
            if attempt == max_retries:
                print("[ERROR] OpenAI API 연속 실패, 이 샘플은 빈 답변으로 넘어갑니다.")
                return None
            sleep_sec = backoff_base ** (attempt - 1)
            print(f"[INFO] {sleep_sec}초 대기 후 재시도...")
            time.sleep(sleep_sec)

def answer_question(messages):
    response = {"standalone_query": "", "topk": [], "references": [], "answer": ""}

    msg = [{"role": "system", "content": persona_function_calling}] + messages

    # 🔹 timeout 조금 늘리고, safe_chat_completion 사용
    result = safe_chat_completion(
        model=llm_model,
        messages=msg,
        tools=tools,  # type: ignore
        temperature=0,  # gpt5 x
        seed=1,
        timeout=20,   # 10 → 20초 정도로 여유
        max_retries=3
    )

    # 연속 실패한 경우
    if result is None:
        response["answer"] = ""
        return response

    if result.choices[0].message.tool_calls:
        tool_call = result.choices[0].message.tool_calls[0]
        function_args = json.loads(tool_call.function.arguments) # type: ignore
        standalone_query = function_args.get("standalone_query")

        # RRF 기반 하이브리드 리트리버 사용
        search_result = hybrid_retrieve_rrf(standalone_query, 3, k=60)
        # search_result = hybrid_retrieve(standalone_query, 3, alpha=0.5)
        response["standalone_query"] = standalone_query

        documents = search_result["hits"]["hits"]
        retrieved_context = []
        references = []
        for doc in documents:
            content = doc["_source"]["content"]
            docid = doc["_source"]["docid"]
            src = doc["_source"]["src"]
            references.append({"docid": docid, "src": src})
            retrieved_context.append(content)

        qa_msg = [
            {"role": "system", "content": qa_persona},
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "msg": messages,
                        "retrieved_context": retrieved_context,
                    },
                    ensure_ascii=False,
                ),
            },
        ]
        qaresult = safe_chat_completion(
            model=llm_model,
            messages=qa_msg,
            temperature=0,  # gpt5 x
            seed=1,
            timeout=20,
            max_retries=3,
        )

        response["topk"] = [doc["_source"]["docid"] for doc in documents]
        response["references"] = references

        if qaresult is None:
            response["answer"] = ""
        else:
            response["answer"] = qaresult.choices[0].message.content
    else:
        response["answer"] = result.choices[0].message.content

    return response


def eval_rag(eval_filename, output_filename):
    with open(eval_filename, encoding="utf-8") as f, open(output_filename, "w", encoding="utf-8") as of:
        idx = 0
        for line in f:
            j = json.loads(line)
            print(f'Test {idx}\nQuestion: {j["msg"]}')
            response = answer_question(j["msg"])
            print(f'Answer: {response["answer"]}\n')

            output = {
                "eval_id": j["eval_id"],
                "standalone_query": response["standalone_query"],
                "topk": response["topk"],
                "answer": response["answer"],
                "references": response["references"]
            }
            of.write(json.dumps(output, ensure_ascii=False) + "\n")
            idx += 1


eval_rag("./data/eval.jsonl", "sample_submission_hybrid2.csv")
