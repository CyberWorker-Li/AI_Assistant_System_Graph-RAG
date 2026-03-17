# AI_Assistant_System_Graph-RAG
《人工智能基础》课程AI助教
==========================================================================
   AI Assistant: 基于 Graph-RAG 与 混合检索的本地课程助教系统
==========================================================================

1. 项目简介
-----------
本项目是一个针对课程文档（如 .docx）设计的智能问答助手。它集成了最前沿的 
Graph-RAG（图增强检索）与 混合检索（向量 + 关键词）技术，旨在解决传统 RAG 
在处理复杂推理、长上下文关联以及专有名词匹配时的局限性。

2. 技术路线 (Technical Stack)
----------------------------
*   核心框架: LangChain (用于流程编排、文本切分、LLM 调用)
*   本地语言模型 (LLM): Llama 3.1 (8B-Instruct-Q2_K) via Ollama
*   文本向量化 (Embedding): BAAI/bge-m3 (本地加载，支持多语言、长文本)
*   重排模型 (Reranker): BAAI/bge-reranker-base (本地加载，提升检索精度)
*   知识图谱 (Graph): NetworkX (构建实体-关系网络)
*   关键词检索: Rank-BM25 + Jieba (针对中文优化的混合检索)
*   向量数据库: FAISS (轻量级本地向量索引)

3. 运行逻辑与过程 (Core Workflow)
--------------------------------
A. 启动预处理阶段 (Initialization):
   1. 文档加载: 从 data 目录读取课程测试文档。
   2. 智能切分: 使用 RecursiveCharacterTextSplitter 按语义边界切分块。
   3. 向量构建: 调用本地 bge-m3 生成嵌入向量并构建 FAISS 索引。
   4. 关键词构建: 使用 jieba 对全文分词并构建 BM25 索引。
   5. 全量建图: 调用 LLM 遍历文本块，提取(实体-关系-实体)三元组，构建持久化知识图谱。

B. 问答执行阶段 (Query Process):
   1. 意图分析 (NLU): 识别问题中的意图、实体及复杂度（Simple/Multi-hop）。
   2. 查询改写: LLM 将原始问题改写为更适合检索的多个候选查询。
   3. 混合检索: 同时执行向量检索(语义)与 BM25 检索(关键词)。
   4. 检索融合: 使用 RRF (倒数排序融合) 算法合并两路检索结果。
   5. 图谱增强: 根据识别出的实体，在图谱中进行“模糊匹配”并提取相关子图路径。
   6. 深度重排 (Rerank): 使用 CrossEncoder 模型对候选证据进行精细打分排序。
   7. 证据压缩: 提取证据中最相关的片段，并自动扩展上下文（抓取前后相邻块）。
   8. 生成回答: LLM 结合文本证据与图谱结构信息，生成可追溯、带引用的答案。

4. 环境依赖 (Prerequisites)
--------------------------
*   Python 3.10+
*   已安装并启动 Ollama (https://ollama.com)
*   核心依赖库安装:
    pip install langchain langchain-openai langchain-text-splitters 
    pip install sentence-transformers faiss-cpu rank-bm25 jieba networkx requests

5. 关键配置说明 (Configuration)
------------------------------
项目配置主要通过 `knowledge/shared/config.py` 及环境变量控制：

*   LLM 接口:
    - AI_ASSISTANT_LLM_BASE_URL: 默认为 http://localhost:11434/v1 (Ollama)
    - AI_ASSISTANT_LLM_ANSWER_MODEL: 推荐使用 llama3.1:8b-instruct-q2_K
*   本地模型路径 (必须在 cli.py/vector_retriever.py 中确认路径一致):
    - Embedding 路径: models/bge-m3
    - Reranker 路径: models/bge-reranker-base
*   检索参数:
    - chunk_size: 800 (建议范围 600-1000)
    - chunk_overlap: 200 (保留足够的语义重叠)
    - enable_rerank: True (启用深度重排以获得更高精度)

6. 移植与部署建议
---------------
1. 模型准备: 确保 models 目录下包含完整的 bge-m3 和 bge-reranker-base 文件。
2. Ollama 准备: 执行 `ollama pull llama3.1:8b-instruct-q2_K`。
3. 数据放置: 将你的课程文档（.docx）放入 AI_Assistant/data 目录下。
4. 启动: 直接运行根目录下的 `run.bat`，它会自动配置 PYTHONPATH 并启动程序。

7. 故障排除
----------
*   若遇到 "JSON Decode Error": 本项目已集成鲁棒解析器，会自动尝试修复小模型畸形输出。
*   若遇到 "ModuleNotFoundError": 请务必通过 `run.bat` 启动或手动设置 PYTHONPATH 为项目根目录。
*   网络超时: 已配置为本地优先模式，除首次下载库外，运行过程无需外网连接。
