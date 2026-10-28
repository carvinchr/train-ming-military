"""
RAG Step 2: FastAPI RAG服务
提供 OpenAI 兼容的 /v1/chat/completions 接口，内部执行检索增强生成（检索 + 重排序）
用法：python 02_rag/rag_server.py --port 8080
"""
import argparse
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import List, Optional

import chromadb
import uvicorn
from chromadb.config import Settings as ChromaSettings
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger
from openai import OpenAI
from pydantic import BaseModel
from sentence_transformers import CrossEncoder, SentenceTransformer

sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import settings

RAG_SYSTEM = """你是一位专精明代军事史的学术助手，擅长明朝军事装备、兵器制造、军事制度等领域。

请基于以下检索到的史料原文回答问题。回答时：
1. 优先引用检索到的原文内容，保持学术准确性
2. 如果原文信息不足，可以结合背景知识补充，但需明确说明
3. 使用规范的学术语言，必要时说明文献来源
4. 如果问题超出明代军事范围，礼貌说明

检索到的相关史料：
{context}
"""

# 全局组件（lifespan 启动时初始化）
state = {
    "chroma_collection": None,
    "embed_model": None,
    "reranker": None,
    "llm_client": None,
    "llm_model_name": None,
    "top_k": settings.top_k,
    "rerank_top_k": settings.rerank_top_k,
}


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info(f"加载嵌入模型: {settings.embed_model}")
    state["embed_model"] = SentenceTransformer(settings.embed_model, trust_remote_code=True)

    logger.info(f"加载重排序模型: {settings.reranker_model}")
    state["reranker"] = CrossEncoder(settings.reranker_model, trust_remote_code=True)

    logger.info(f"连接ChromaDB: {settings.chroma_db_path}")
    client = chromadb.PersistentClient(
        path=str(settings.chroma_db_path),
        settings=ChromaSettings(anonymized_telemetry=False),
    )
    state["chroma_collection"] = client.get_collection(settings.collection_name)
    logger.info(f"集合大小: {state['chroma_collection'].count()} 条")

    state["llm_client"] = OpenAI(base_url=settings.llm_base_url, api_key=settings.llm_api_key)
    state["llm_model_name"] = settings.llm_model

    yield

    logger.info("RAG服务关闭")


app = FastAPI(title="明代军事装备RAG服务", version="1.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_methods=["*"],
    allow_headers=["*"],
)


class Message(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    model: Optional[str] = None
    messages: List[Message]
    temperature: Optional[float] = 0.7
    max_tokens: Optional[int] = 2048
    stream: Optional[bool] = False
    top_k: Optional[int] = None
    rerank_top_k: Optional[int] = None


class SearchRequest(BaseModel):
    query: str
    top_k: Optional[int] = None
    rerank_top_k: Optional[int] = None


def retrieve(query: str, top_k: int, rerank_top_k: int) -> List[dict]:
    """向量检索候选，再用 cross-encoder 重排序取前 rerank_top_k 条。"""
    embed_model = state["embed_model"]
    collection = state["chroma_collection"]

    query_embedding = embed_model.encode([query], normalize_embeddings=True).tolist()
    results = collection.query(
        query_embeddings=query_embedding,
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )

    candidates = []
    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        candidates.append({
            "text": doc,
            "source": meta.get("source", ""),
            "chapter": meta.get("chapter", ""),
            "vector_score": round(1 - dist, 4),
        })

    reranker = state["reranker"]
    pairs = [[query, c["text"]] for c in candidates]
    scores = reranker.predict(pairs)
    for c, s in zip(candidates, scores):
        c["rerank_score"] = round(float(s), 4)

    candidates.sort(key=lambda c: c["rerank_score"], reverse=True)
    return candidates[:rerank_top_k]


@app.get("/health")
def health():
    return {"status": "ok", "collection_count": state["chroma_collection"].count()}


@app.post("/v1/search")
def search(req: SearchRequest):
    """仅检索+重排序，不生成，用于调试"""
    top_k = req.top_k or state["top_k"]
    rerank_top_k = req.rerank_top_k or state["rerank_top_k"]
    results = retrieve(req.query, top_k, rerank_top_k)
    return {"results": results}


@app.post("/v1/chat/completions")
def chat(req: ChatRequest):
    # 提取最后一个用户消息作为检索查询
    user_messages = [m for m in req.messages if m.role == "user"]
    if not user_messages:
        raise HTTPException(status_code=400, detail="没有用户消息")
    query = user_messages[-1].content

    top_k = req.top_k or state["top_k"]
    rerank_top_k = req.rerank_top_k or state["rerank_top_k"]
    retrieved = retrieve(query, top_k, rerank_top_k)

    context_parts = []
    for i, r in enumerate(retrieved, 1):
        context_parts.append(
            f"[{i}] 来源：{r['source']}/{r['chapter']}（相关度：{r['rerank_score']}）\n{r['text']}"
        )
    context = "\n\n".join(context_parts)

    # 构建增强后的消息
    system_content = RAG_SYSTEM.format(context=context)
    augmented_messages = [{"role": "system", "content": system_content}]
    for m in req.messages:
        augmented_messages.append({"role": m.role, "content": m.content})

    # 调用LLM
    response = state["llm_client"].chat.completions.create(
        model=state["llm_model_name"],
        messages=augmented_messages,
        temperature=req.temperature,
        max_tokens=req.max_tokens,
    )

    answer = response.choices[0].message.content
    sources_info = "\n\n---\n**参考来源：**\n" + "\n".join(
        f"- [{r['source']}] {r['chapter']} 重排得分 {r['rerank_score']}" for r in retrieved
    )

    return {
        "id": response.id,
        "object": "chat.completion",
        "model": req.model or state["llm_model_name"],
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": answer + sources_info,
            },
            "finish_reason": "stop",
        }],
        "retrieved_contexts": retrieved,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--host", default="0.0.0.0")
    args = parser.parse_args()

    logger.info(f"RAG服务启动: http://{args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
