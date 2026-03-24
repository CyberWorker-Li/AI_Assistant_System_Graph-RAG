import json
import re

from ingestion.document_loader import load_course_documents
from knowledge.shared.config import Settings
from knowledge.shared.schemas import Evidence
from knowledge.nlu.analyzer import NLUAnalyzer
from knowledge.retrieval.strategy_router import StrategyRouter
from knowledge.generation.prompt_builder import build_prompt
from knowledge.graph_store.triple_extractor import TripleExtractor
from knowledge.vector_store.indexer import ChunkRecord, SimpleVectorIndexer
from knowledge.vector_store.vector_retriever import RetrievalEvidence, VectorRetriever
from knowledge.vector_store.rag_answerer import RagAnswerer
from langchain_text_splitters import RecursiveCharacterTextSplitter
from rank_bm25 import BM25Okapi
import jieba
from knowledge.graph_store.graph_store import KnowledgeGraphStore


def build_retriever_from_documents(settings: Settings) -> tuple[VectorRetriever, BM25Okapi, list[ChunkRecord], int, int]:
    # 直接指定用户提供的本地模型路径，避免网络问题
    local_model_path = "D:\\mine\\bjtu\\AI_Assistant_Model\\graph_maker_LLM\\AI_Assistant\\models\\bge-m3"
    indexer = SimpleVectorIndexer(
        embedding_model_name=local_model_path,
        local_files_only=True,  # 强制只使用本地文件
    )
    docs = load_course_documents(settings.data_dir)

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        separators=["\n\n", "\n", "。", "！", "？", "，", "、", " "],  # 优先按段落和句子切分
    )

    chunk_records: list[ChunkRecord] = []
    chunk_id = 1
    for d in docs:
        parts = text_splitter.split_text(d.text)
        for part in parts:
            chunk_records.append(
                ChunkRecord(
                    chunk_id=f"doc-{chunk_id}",
                    text=part,
                    metadata={"source": d.source_file, "type": d.source_type},
                )
            )
            chunk_id += 1

    if not docs:
        raise RuntimeError(f"未在数据目录发现可用课程文档: {settings.data_dir}")
    if not chunk_records:
        raise RuntimeError(f"课程文档未提取到有效文本块: {settings.data_dir}")

    # 1. 构建向量索引
    indexer.add_chunks(chunk_records)
    retriever = VectorRetriever(
        indexer,
        reranker_model_name=settings.reranker_model,
        enable_rerank=settings.enable_rerank, # 根据配置启用Reranker
        local_files_only=settings.local_files_only,
    )

    # 2. 构建BM25关键词索引
    print("正在构建BM25索引...")
    corpus = [rec.text for rec in chunk_records]
    tokenized_corpus = [list(jieba.cut_for_search(doc)) for doc in corpus]
    bm25 = BM25Okapi(tokenized_corpus)
    print("BM25索引构建完成。")

    return retriever, bm25, chunk_records, len(docs), len(chunk_records)


def _tokenize_query(question: str) -> set[str]:
    return set(re.findall(r"[\u4e00-\u9fa5A-Za-z0-9_]{2,}", question.lower()))


def _clean_question(question: str) -> str:
    return re.sub(r"^[\-•·\s]+", "", question).strip()


def _rewrite_question(question: str, intent: str, entities: list[str]) -> str:
    q = _clean_question(question)
    if intent == "ask_relation":
        base = " ".join(entities) if entities else q
        return f"{base} 关系 属于 包含 联系 分类"
    if intent == "ask_reason":
        base = " ".join(entities) if entities else q
        return f"{base} 原理 机制"
    return q


def _llm_chat(settings: Settings, system_prompt: str, user_prompt: str, model: str, temperature: float = 0.0) -> str:
    if settings.llm_framework != "langchain":
        raise RuntimeError(f"不支持的LLM框架: {settings.llm_framework}，当前仅支持 langchain")
    try:
        from langchain_core.messages import HumanMessage, SystemMessage
        from langchain_openai import ChatOpenAI
    except Exception as e:
        raise RuntimeError(f"缺少LangChain依赖，请安装 langchain-openai/langchain-core: {e}")
    llm = ChatOpenAI(
        model=model,
        api_key=settings.llm_api_key or "no-key",
        base_url=settings.llm_base_url,
        timeout=settings.llm_timeout,
        temperature=temperature,
    )
    result = llm.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt),
    ])
    content = str(result.content).strip()
    if not content:
        raise RuntimeError("LLM返回为空")
    return content


def _safe_json_loads(text: str) -> dict:
    """鲁棒的JSON加载函数，专门处理小模型输出的畸形JSON"""
    # 1. 尝试直接解析
    try:
        match = re.search(r"\{[\s\S]*\}", text)
        if match:
            content = match.group(0)
            # 简单修复常见的JSON错误：去掉列表末尾多余的逗号
            content = re.sub(r",\s*\]", "]", content)
            content = re.sub(r",\s*\}", "}", content)
            return json.loads(content)
    except:
        pass

    # 2. 如果失败，尝试手动正则提取关键字段
    data = {}
    # 提取 rewrite 字段
    rw_match = re.search(r'["\']rewrite["\']\s*:\s*["\']([^"\'\}]+)["\']', text)
    if rw_match:
        data["rewrite"] = rw_match.group(1).strip()
    
    # 提取 selected 列表
    sel_match = re.search(r'["\']selected["\']\s*:\s*\[([\s\d,]*)\]', text)
    if sel_match:
        nums = re.findall(r'\d+', sel_match.group(1))
        data["selected"] = [int(n) for n in nums]
        
    # 提取 items 结构（针对重排）
    if '"items"' in text:
        data["items"] = []
        item_blocks = re.findall(r'\{[^{}]*\}', text)
        for block in item_blocks:
            if "idx" in block:
                idx = re.search(r'"idx"\s*:\s*(\d+)', block)
                rel = re.search(r'"relevance"\s*:\s*([\d\.]+)', block)
                ans = re.search(r'"answerability"\s*:\s*([\d\.]+)', block)
                if idx:
                    data["items"].append({
                        "idx": int(idx.group(1)),
                        "relevance": float(rel.group(1)) if rel else 0.5,
                        "answerability": float(ans.group(1)) if ans else 0.5
                    })
    return data


def _llm_rewrite_query(question: str, settings: Settings, intent: str, entities: list[str]) -> tuple[str, list[str], bool]:
    if not settings.enable_llm_rewrite:
        return _rewrite_question(question, intent, entities), [], False
    prompt = (
        "将用户问题改写为更适合检索的查询。返回JSON: "
        "{\"rewrite\":\"...\",\"variants\":[\"...\",\"...\"]}。\n"
        f"问题: {question}\n意图: {intent}\n实体: {'、'.join(entities) if entities else '无'}"
    )
    content = _llm_chat(settings, "你是查询改写器，只输出JSON。", prompt, settings.llm_rerank_model, temperature=0)
    data = _safe_json_loads(content)
    rewrite = str(data.get("rewrite", "")).strip()
    
    if not rewrite:
        # 如果LLM解析彻底失败，回退到规则改写，不报错
        return _rewrite_question(question, intent, entities), [], False
        
    variants = [str(x).strip() for x in data.get("variants", []) if str(x).strip()]
    return rewrite, variants[:2], True


def _sparse_overlap_score(query: str, text: str) -> float:
    q = _tokenize_query(query)
    if not q:
        return 0.0
    t = _tokenize_query(text)
    return len(q & t) / max(len(q), 1)


def _reciprocal_rank_fusion(results: list[list[RetrievalEvidence]], k: int = 60) -> list[RetrievalEvidence]:
    fused_scores = {}
    # 建立一个从chunk_id到完整RetrievalEvidence对象的映射
    id_to_evidence = {evi.evidence_id: evi for res_list in results for evi in res_list}

    for res_list in results:
        for rank, evi in enumerate(res_list, 1):
            if evi.evidence_id not in fused_scores:
                fused_scores[evi.evidence_id] = 0
            fused_scores[evi.evidence_id] += 1 / (k + rank)

    reranked_ids = sorted(fused_scores.keys(), key=lambda x: fused_scores[x], reverse=True)
    
    # 根据融合后的排序，重新构建证据列表
    reranked_results = [id_to_evidence[id] for id in reranked_ids if id in id_to_evidence]
    return reranked_results


def _hierarchical_retrieve(question: str, plan, retriever: VectorRetriever, bm25: BM25Okapi, all_chunks: list[ChunkRecord], settings: Settings) -> tuple[list[RetrievalEvidence], bool]:
    rewrite, variants, used_llm_rewrite = _llm_rewrite_query(question, settings, plan.intent, plan.entities)
    queries = [rewrite] + variants
    candidate_k = max(settings.retrieval_candidate_k, plan.top_k + 5)

    # 1. 语义检索 (Dense Retrieval)
    vector_hits = []
    for q in queries:
        vector_hits.extend(retriever.retrieve(q, top_k=candidate_k))
    # 去重
    seen_ids = set()
    unique_vector_hits = []
    for hit in vector_hits:
        if hit.evidence_id not in seen_ids:
            unique_vector_hits.append(hit)
            seen_ids.add(hit.evidence_id)

    # 2. 关键词检索 (Sparse Retrieval)
    tokenized_query = list(jieba.cut_for_search(rewrite))
    bm25_scores = bm25.get_scores(tokenized_query)
    
    scored_chunks = sorted([(score, i) for i, score in enumerate(bm25_scores)], reverse=True)
    keyword_hits = []
    for score, i in scored_chunks[:candidate_k]:
        if score > 0:
            rec = all_chunks[i]
            keyword_hits.append(
                RetrievalEvidence(
                    evidence_id=rec.chunk_id,
                    content=rec.text,
                    score=score,
                    metadata=rec.metadata,
                )
            )

    # 3. 结果融合 (Reciprocal Rank Fusion)
    fused_hits = _reciprocal_rank_fusion([unique_vector_hits, keyword_hits])

    # 4. (可选) 文档级重排，与原逻辑保持一致
    doc_scores = {}
    for h in fused_hits:
        src = h.metadata.get("source", "")
        # 使用RRF的分数作为基础，可以设计更复杂的评分机制
        doc_scores[src] = max(doc_scores.get(src, -1e9), h.score) # 使用hit自身的分数
    top_docs = {k for k, _ in sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)[: settings.doc_top_n]}
    filtered = [h for h in fused_hits if h.metadata.get("source", "") in top_docs]
    
    return filtered[:candidate_k], used_llm_rewrite


def _split_sentences(text: str) -> list[str]:
    pieces = re.split(r"(?<=[。！？!?；;])\s*", text)
    sentences = [p.strip() for p in pieces if p and p.strip()]
    return sentences if sentences else ([text.strip()] if text.strip() else [])


def _relation_sentence_candidates(hits: list[RetrievalEvidence], entities: list[str]) -> list[RetrievalEvidence]:
    relation_terms = ["关系", "属于", "包含", "联系", "分类", "分为", "子类", "上位", "下位"]
    out: list[RetrievalEvidence] = []
    seen = set()
    for h in hits:
        for i, s in enumerate(_split_sentences(h.content), 1):
            entity_hits = sum(1 for e in entities if e and e in s)
            rel_hits = sum(1 for t in relation_terms if t in s)
            if entity_hits >= 2 and rel_hits >= 1:
                key = s.strip()
                if key in seen:
                    continue
                seen.add(key)
                out.append(
                    RetrievalEvidence(
                        evidence_id=f"{h.evidence_id}-s{i}",
                        content=s[:180],
                        score=h.score + entity_hits * 0.5 + rel_hits * 0.2,
                        metadata=h.metadata,
                    )
                )
    out.sort(key=lambda x: x.score, reverse=True)
    return out


def _extract_relation_snippet(text: str, entities: list[str]) -> str:
    relation_terms = ["关系", "属于", "包含", "联系", "分类", "分为", "子类", "上位", "下位"]
    candidates = []
    for s in _split_sentences(text):
        entity_hits = sum(1 for e in entities if e and e in s)
        rel_hits = sum(1 for t in relation_terms if t in s)
        if entity_hits >= 1 and rel_hits >= 1:
            score = entity_hits * 2 + rel_hits - len(s) / 140.0
            candidates.append((score, s))
    if not candidates:
        return ""
    candidates.sort(key=lambda x: x[0], reverse=True)
    return candidates[0][1][:140]


def _compress_text(question: str, text: str, max_sentences: int = 2, max_chars: int = 220, intent: str = "ask_general", entities: list[str] | None = None) -> str:
    sentences = _split_sentences(text)
    if not sentences:
        return ""
    q_terms = _tokenize_query(question)
    scored: list[tuple[int, int, int]] = []
    focus_entities = [e for e in (entities or []) if e]
    for idx, s in enumerate(sentences):
        s_terms = set(re.findall(r"[\u4e00-\u9fa5A-Za-z0-9_]{2,}", s.lower()))
        overlap = len(q_terms & s_terms)
        bonus = 0
        if intent == "ask_concept" and ("是" in s or "定义" in s or "指" in s):
            bonus += 3
        if intent == "ask_relation" and any(k in s for k in ["关系", "属于", "包含", "联系"]):
            bonus += 2
        if focus_entities and any(e in s for e in focus_entities):
            bonus += 2
        if intent == "ask_relation" and focus_entities and not any(e in s for e in focus_entities):
            bonus -= 2
        scored.append((overlap + bonus, -idx, idx))
    scored.sort(reverse=True)
    selected_idx = sorted([x[2] for x in scored[:max_sentences]])
    merged = " ".join(sentences[i] for i in selected_idx).strip()
    return merged[:max_chars]


def _rerank_hits_by_intent(hits: list[RetrievalEvidence], intent: str, entities: list[str]) -> list[RetrievalEvidence]:
    if not hits:
        return hits
    relation_terms = ["关系", "属于", "包含", "联系", "分类", "子类", "上位", "下位"]
    reason_terms = ["原理", "机制", "自注意力", "并行", "长距离", "依赖"]

    def score_hit(h: RetrievalEvidence) -> float:
        text = h.content
        score = h.score
        hit_entities = sum(1 for e in entities if e and e in text) if entities else 0
        if entities:
            score += hit_entities * 0.6
        if intent == "ask_relation":
            rel_hit = sum(1 for t in relation_terms if t in text)
            score += rel_hit * 0.2
            if entities and len(entities) >= 2 and hit_entities < 2:
                score -= 1.0
            if rel_hit == 0:
                score -= 0.4
        if intent == "ask_reason":
            score += sum(0.2 for t in reason_terms if t in text)
        if re.search(r"[\{\}\[\]=_;]{2,}|\bPipeline\(|\bfit_transform\(|\bX_train\b", text):
            score -= 0.8
        return score

    ranked = sorted(hits, key=score_hit, reverse=True)
    if intent == "ask_relation" and entities:
        hard_filtered = []
        for h in ranked:
            text = h.content
            entity_hits = sum(1 for e in entities if e and e in text)
            rel_hits = sum(1 for t in relation_terms if t in text)
            if entity_hits >= 2 and rel_hits >= 1:
                hard_filtered.append(h)
        if hard_filtered:
            return hard_filtered
    return ranked


def _llm_rerank_hits(question: str, hits: list[RetrievalEvidence], settings: Settings, intent: str, entities: list[str]) -> tuple[list[RetrievalEvidence], bool, str]:
    if not settings.enable_llm_rerank:
        return hits, False, "disabled"
    if intent not in {"ask_relation", "ask_reason"}:
        return hits, False, "intent_skip"
    if settings.llm_framework == "langchain" and "localhost" not in settings.llm_base_url and not settings.llm_api_key:
        return hits, False, "no_api_key"
    if not hits:
        return hits, False, "empty_hits"
    candidates = hits[:8]
    lines = []
    for i, h in enumerate(candidates, 1):
        lines.append(f"[{i}] {h.content[:260].replace(chr(10), ' ')}")
    entity_text = "、".join(entities) if entities else "无"
    user_prompt = (
        f"问题: {question}\n"
        f"意图: {intent}\n"
        f"核心实体: {entity_text}\n"
        "请评估每条候选证据的相关性(relevance 0-1)和可回答性(answerability 0-1)，并输出JSON。\n"
        "格式: {\"items\":[{\"idx\":1,\"relevance\":0.9,\"answerability\":0.8}],\"selected\":[1,3]}\n"
        "候选证据:\n" + "\n".join(lines)
    )
    content = _llm_chat(settings, "你是推理重排器，只输出JSON对象。", user_prompt, settings.llm_rerank_model, temperature=0)
    data = _safe_json_loads(content)
    
    items = data.get("items", [])
    selected = data.get("selected", [])
    
    if not items and not selected:
        return hits, False, "llm_parse_failed"
        
    score_map = {}
    for it in items:
        idx = int(it.get("idx", 0))
        rel = float(it.get("relevance", 0))
        ans = float(it.get("answerability", 0))
        score_map[idx] = rel * 0.6 + ans * 0.4
    order = [int(x) for x in selected if int(x) >= 1]
    if not order:
        order = sorted(score_map.keys(), key=lambda i: score_map[i], reverse=True)
    picked = []
    used = set()
    for idx in order:
        j = idx - 1
        if 0 <= j < len(candidates) and j not in used:
            if score_map.get(idx, 0.0) >= 0.45:
                used.add(j)
                picked.append(candidates[j])
    for i, h in enumerate(candidates):
        if i not in used:
            picked.append(h)
    return picked + hits[len(candidates):], True, "ok"

def _compress_hits(question: str, hits: list[RetrievalEvidence], intent: str, entities: list[str]) -> list[RetrievalEvidence]:
    out: list[RetrievalEvidence] = []
    seen = set()
    if intent == "ask_concept":
        max_sentences, max_chars = 1, 140
    elif intent == "ask_relation":
        max_sentences, max_chars = 1, 120
    else:
        max_sentences, max_chars = 2, 200
    filtered_hits = _rerank_hits_by_intent(hits, intent, entities)
    if intent == "ask_relation" and entities:
        relation_hits = _relation_sentence_candidates(filtered_hits, entities)
        if relation_hits:
            filtered_hits = relation_hits
    for h in filtered_hits:
        if intent == "ask_relation" and entities:
            relation_snippet = _extract_relation_snippet(h.content, entities)
            compact = relation_snippet if relation_snippet else _compress_text(question, h.content, max_sentences=max_sentences, max_chars=max_chars, intent=intent, entities=entities)
        else:
            compact = _compress_text(question, h.content, max_sentences=max_sentences, max_chars=max_chars, intent=intent, entities=entities)
        text = (compact if compact else h.content[:220]).strip()
        if not text or text in seen:
            continue
        seen.add(text)
        out.append(
            RetrievalEvidence(
                evidence_id=h.evidence_id,
                content=text,
                score=h.score,
                metadata=h.metadata,
            )
        )
    return out


def _to_vector_evidences(hits: list[RetrievalEvidence], all_chunks: list[ChunkRecord]) -> list[Evidence]:
    out: list[Evidence] = []
    # 建立ID索引方便查找邻居
    id_map = {c.chunk_id: i for i, c in enumerate(all_chunks)}
    
    for h in hits:
        content = h.content
        # 双向上下文扩展：同时抓取上一个和下一个文本块
        curr_idx = id_map.get(h.evidence_id)
        if curr_idx is not None:
            # 抓取上一个块
            if curr_idx > 0:
                prev_chunk = all_chunks[curr_idx - 1]
                if prev_chunk.metadata.get("source") == h.metadata.get("source"):
                    content = f"[补充上一段]: {prev_chunk.text}\n\n---\n{content}"
            # 抓取下一个块
            if curr_idx + 1 < len(all_chunks):
                next_chunk = all_chunks[curr_idx + 1]
                if next_chunk.metadata.get("source") == h.metadata.get("source"):
                    content += f"\n\n---\n[补充下一段]: {next_chunk.text}"
        
        out.append(
            Evidence(
                evidence_id=h.evidence_id,
                evidence_type="vector",
                content=content,
                score=h.score,
                source_ref=h.metadata.get("source", "vector_store"),
                metadata=h.metadata,
            )
        )
    return out


def _build_graph_evidence(graph_store: KnowledgeGraphStore, entities: list[str]) -> list[Evidence]:
    if not entities:
        return []
    
    # 从预构建的图谱中检索与实体相关的子图
    triples = graph_store.retrieve_subgraph_for_entities(entities, k=2)
    
    if not triples:
        return []
        
    # 将三元组格式化为可读文本
    triple_text = "；".join([f"({s})-[:{r}]->({o})" for s, r, o in triples[:12]]) 
    return [
        Evidence(
            evidence_id="graph-knowledge",
            evidence_type="graph",
            content=f"【知识图谱关联信息】：{triple_text}",
            score=2.0,  # 进一步提高图权重
            source_ref="knowledge_graph",
            metadata={"triple_count": str(len(triples))},
        )
    ]


def main() -> None:
    settings = Settings()
    nlu = NLUAnalyzer()
    router = StrategyRouter()
    extractor = TripleExtractor()
    retriever, bm25, all_chunks, doc_count, chunk_count = build_retriever_from_documents(settings)
    answerer = RagAnswerer(settings)

    graph_store = KnowledgeGraphStore()
    if settings.enable_graph_on_start:
        from pathlib import Path
        cache_path = settings.graph_cache_path
        if cache_path and Path(cache_path).exists():
            try:
                graph_store.load_from(cache_path)
                print(f"已从缓存加载知识图谱：nodes={graph_store.graph.number_of_nodes()} edges={graph_store.graph.number_of_edges()}")
            except Exception:
                pass
        if graph_store.graph.number_of_nodes() == 0:
            print("\n正在构建知识图谱，这可能需要几分钟时间，请稍候...")
            chunks = all_chunks[:settings.graph_build_max_chunks] if settings.graph_build_max_chunks > 0 else all_chunks
            graph_store.build_from_chunks(chunks, extractor, use_llm=True)
            if cache_path:
                try:
                    graph_store.save_to(cache_path)
                except Exception:
                    pass
            print(f"知识图谱构建完成，包含 {graph_store.graph.number_of_nodes()} 个节点和 {graph_store.graph.number_of_edges()} 条边。")

    print(
        f"AI Assistant CLI 已启动，直接回车可退出。文档数={doc_count}，文本块数={chunk_count}，"
        f"local_rerank=off，llm_rerank={settings.enable_llm_rerank}"
    )
    while True:
        question = input("\n请输入问题: ").strip()
        if not question:
            print("已退出。")
            break

        question = _clean_question(question)
        plan = nlu.analyze(question)
        # 实体标准化
        plan.entities = [e.replace('的', '') for e in plan.entities if e]
        plan.top_k = settings.top_k
        route_mode = router.route(plan)

        try:
            raw_hits, used_llm_rewrite = _hierarchical_retrieve(question, plan, retriever, bm25, all_chunks, settings)
            reranked_hits, llm_used, llm_reason = _llm_rerank_hits(question, raw_hits, settings, plan.intent, plan.entities)
            hits = _compress_hits(question, reranked_hits, plan.intent, plan.entities)
            
            # 对于复杂推理问题，增加Top-K数量
            current_top_k = plan.top_k + 2 if plan.question_type == "multi_hop" else plan.top_k
            hits = hits[:current_top_k]
            
            evidences = _to_vector_evidences(hits, all_chunks) # 传入all_chunks用于扩展上下文
            if route_mode in {"graph_first", "hybrid"}:
                graph_evidences = _build_graph_evidence(graph_store, plan.entities)
                if route_mode == "graph_first":
                    evidences = graph_evidences + evidences
                else:
                    evidences.extend(graph_evidences)
            prompt = build_prompt(question, evidences)
            answer = answerer.answer(question, hits, intent=plan.intent)
        except Exception as e:
            print(f"\n[ERROR] 调用失败: {e}")
            continue
        if settings.enable_debug_output:
            print(f"\n[NLU] intent={plan.intent}, question_type={plan.question_type}, entities={plan.entities}")
            print(f"[Route] {route_mode}")
            print(f"[LLM-REWRITE] used={used_llm_rewrite}")
            print(f"[LLM-RERANK] used={llm_used}, reason={llm_reason}")
            print("[Prompt]\n" + prompt)
            print("\n[Answer]\n" + answer)
        else:
            print(answer)


if __name__ == "__main__":
    main()