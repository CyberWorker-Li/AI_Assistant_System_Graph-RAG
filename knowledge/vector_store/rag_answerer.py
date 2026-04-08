from pathlib import Path
from typing import List

from knowledge.shared.config import Settings
from knowledge.shared.schemas import Evidence


def _tail_chars(text: str, max_chars: int) -> str:
    text = str(text or "")
    if max_chars <= 0:
        return ""
    if len(text) <= max_chars:
        return text
    return text[-max_chars:]


class RagAnswerer:
    def __init__(self, settings: Settings):
        self.settings = settings

    def answer(self, question: str, evidences: List[Evidence], intent: str = "ask_general", conversation_context: str = "") -> str:
        if not evidences:
            return "根据当前知识库，我暂时找不到相关证据。"

        try:
            ctx = str(conversation_context or "").strip()
            if ctx and hasattr(self.settings, "session_max_chars"):
                ctx = _tail_chars(ctx, int(getattr(self.settings, "session_max_chars", 0) or 0)).strip()
            generated = self._llm_answer(question, evidences, intent, conversation_context=ctx)
            if self.settings.enable_llm_polish and self.settings.polish_base_url and self.settings.polish_api_key:
                try:
                    generated = self._polish_text(generated)
                except Exception:
                    pass
            if self.settings.concise_answer:
                return generated

            top = evidences[:3]
            lines = [f"问题：{question}", "参考证据："]
            for i, e in enumerate(top, 1):
                snippet = e.content[:140].replace("\n", " ")
                source = Path(e.source_ref or "unknown").name
                lines.append(f"[{i}] {snippet}（来源: {source}）")
            lines.append("综合回答：" + generated)
            return "\n".join(lines)
        except Exception as e:
            return f"综合回答生成失败：{e}"

    def _llm_answer(self, question: str, evidences: List[Evidence], intent: str, conversation_context: str = "") -> str:
        if not self.settings.enable_llm_answer:
            raise RuntimeError("LLM回答开关未启用")
        try:
            from langchain_core.messages import HumanMessage, SystemMessage
            from langchain_openai import ChatOpenAI
        except Exception as e:
            raise RuntimeError(f"缺少LangChain依赖，请安装 langchain-openai/langchain-core: {e}")
        evidence_lines = []
        for i, e in enumerate(evidences, 1):
            source = Path(e.source_ref or "unknown").name
            content = e.content
            evidence_lines.append(f"[{i}] {content}\n（来源: {source}）")
        history_block = ""
        if conversation_context:
            history_block = (
                "\n对话历史（仅用于理解指代与省略，不作为事实来源；如与证据冲突，以证据为准）：\n"
                + conversation_context
                + "\n"
            )
        prompt = (
            "请严格基于参考证据回答问题，不能使用证据外知识。"
            "不要改写或替换问题中的专有名词。"
            "若证据不足请明确说明。"
            "必须在句末标注引用编号，如[1][2]。\n"
            f"意图: {intent}\n问题: {question}"
            + history_block
            + "证据:\n"
            + "\n".join(evidence_lines)
        )
        llm = ChatOpenAI(
            model=self.settings.llm_answer_model,
            api_key=self.settings.llm_api_key or "no-key",
            base_url=self.settings.llm_base_url,
            timeout=self.settings.llm_timeout,
            temperature=0.1,
        )
        result = llm.invoke([
            SystemMessage(content="你是课程助教。回答必须可追溯、严格引用证据；当对话历史与证据不一致时，以证据为准。"),
            HumanMessage(content=prompt),
        ])
        content = str(result.content).strip()
        if not content:
            raise RuntimeError("LLM返回为空")
        return content

    def _polish_text(self, text: str) -> str:
        from langchain_core.messages import HumanMessage, SystemMessage
        from langchain_openai import ChatOpenAI
        llm = ChatOpenAI(
            model=self.settings.polish_model,
            api_key=self.settings.polish_api_key or "no-key",
            base_url=self.settings.polish_base_url,
            timeout=self.settings.llm_timeout,
            temperature=0.2,
        )
        result = llm.invoke([
            SystemMessage(content="你是中文写作润色助手，在不改变事实并保留所有引用编号如[1][2]的前提下优化语言表达。只输出润色后的文本。"),
            HumanMessage(content=text),
        ])
        return str(result.content).strip() or text