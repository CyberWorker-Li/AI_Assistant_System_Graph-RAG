import os
from dataclasses import dataclass


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


@dataclass
class Settings:
    top_k: int = int(os.getenv("AI_ASSISTANT_TOP_K", "3"))
    graph_html_output: str = os.getenv("AI_ASSISTANT_GRAPH_HTML", "knowledge_graph.html")
    data_dir: str = os.getenv("AI_ASSISTANT_DATA_DIR", "data")
    chunk_size: int = int(os.getenv("AI_ASSISTANT_CHUNK_SIZE", "800"))
    chunk_overlap: int = int(os.getenv("AI_ASSISTANT_CHUNK_OVERLAP", "200"))
    embedding_model: str = os.getenv("AI_ASSISTANT_EMBEDDING_MODEL", "BAAI/bge-m3")
    reranker_model: str = os.getenv("AI_ASSISTANT_RERANKER_MODEL", "BAAI/bge-reranker-base")
    enable_rerank: bool = _env_bool("AI_ASSISTANT_ENABLE_RERANK", True)
    hf_endpoint: str = os.getenv("HF_ENDPOINT", "https://hf-mirror.com")
    local_files_only: bool = _env_bool("AI_ASSISTANT_LOCAL_FILES_ONLY", False)
    enable_llm_rewrite: bool = _env_bool("AI_ASSISTANT_ENABLE_LLM_REWRITE", True)
    enable_llm_rerank: bool = _env_bool("AI_ASSISTANT_ENABLE_LLM_RERANK", True)
    enable_llm_answer: bool = _env_bool("AI_ASSISTANT_ENABLE_LLM_ANSWER", True)
    llm_framework: str = os.getenv("AI_ASSISTANT_LLM_FRAMEWORK", "langchain")
    llm_api_key: str = os.getenv("AI_ASSISTANT_LLM_API_KEY", "")
    llm_base_url: str = os.getenv("AI_ASSISTANT_LLM_BASE_URL", "http://localhost:11434/v1")
    llm_rerank_model: str = os.getenv("AI_ASSISTANT_LLM_RERANK_MODEL", "llama3.1:8b")
    llm_answer_model: str = os.getenv("AI_ASSISTANT_LLM_ANSWER_MODEL", "llama3.1:8b")
    llm_api_url: str = os.getenv("AI_ASSISTANT_LLM_API_URL", "")
    llm_timeout: int = int(os.getenv("AI_ASSISTANT_LLM_TIMEOUT", "45"))
    retrieval_candidate_k: int = int(os.getenv("AI_ASSISTANT_RETRIEVAL_CANDIDATE_K", "8"))
    doc_top_n: int = int(os.getenv("AI_ASSISTANT_DOC_TOP_N", "2"))