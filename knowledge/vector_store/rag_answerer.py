from pathlib import Path
from typing import List

from knowledge.shared.config import Settings
from knowledge.vector_store.vector_retriever import RetrievalEvidence


class RagAnswerer:
    def __init__(self, settings: Settings):
        self.settings = settings

    def answer(self, question: str, evidences: List[RetrievalEvidence], intent: str = "ask_general") -> str:
        if not evidences:
            return "根据当前知识库，我暂时找不到相关证据。"
        top = evidences[:3]
        lines = [f"问题：{question}", "参考证据："]
        for i, e in enumerate(top, 1):
            snippet = e.content[:140].replace("\n", " ")
            source = Path(e.metadata.get("source", "unknown")).name
            lines.append(f"[{i}] {snippet}（来源: {source}）")

        try:
            generated = self._llm_answer(question, top, intent)
            lines.append("综合回答：" + generated)
        except Exception as e:
            lines.append(f"综合回答生成失败：{e}")
        return "\n".join(lines)

    def _llm_answer(self, question: str, evidences: List[RetrievalEvidence], intent: str) -> str:
        if not self.settings.enable_llm_answer:
            raise RuntimeError("LLM回答开关未启用")
        try:
            from langchain_core.messages import HumanMessage, SystemMessage
            from langchain_openai import ChatOpenAI
        except Exception as e:
            raise RuntimeError(f"缺少LangChain依赖，请安装 langchain-openai/langchain-core: {e}")
        evidence_lines = []
        for i, e in enumerate(evidences, 1):
            evidence_lines.append(f"[{i}] {e.content}")
        prompt = (
            "请严格基于参考证据回答问题，不能使用证据外知识。"
            "若证据不足请明确说明。必须在句末标注引用编号，如[1][2]。\n"
            f"意图: {intent}\n问题: {question}\n证据:\n" + "\n".join(evidence_lines)
        )
        llm = ChatOpenAI(
            model=self.settings.llm_answer_model,
            api_key=self.settings.llm_api_key or "no-key",
            base_url=self.settings.llm_base_url,
            timeout=self.settings.llm_timeout,
            temperature=0.1,
        )
        result = llm.invoke([
            SystemMessage(content="你是课程助教，回答必须可追溯。"),
            HumanMessage(content=prompt),
        ])
        content = str(result.content).strip()
        if not content:
            raise RuntimeError("LLM返回为空")
        return content