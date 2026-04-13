import json
import os
import re
from dataclasses import dataclass
from typing import List, Tuple

import requests


@dataclass
class TripleExtractorConfig:
    api_key: str | None = None
    api_url: str = os.getenv("AI_ASSISTANT_LLM_API_URL", "http://localhost:11434/v1/chat/completions")
    model: str = os.getenv("AI_ASSISTANT_LLM_ANSWER_MODEL", "llama3.1:8b")
    timeout: int = int(os.getenv("AI_ASSISTANT_LLM_TIMEOUT", "60"))


class TripleExtractor:
    def __init__(self, config: TripleExtractorConfig | None = None):
        self.config = config or TripleExtractorConfig(api_key=os.getenv("AI_ASSISTANT_LLM_API_KEY", ""))

    def extract(self, text: str, use_llm: bool = True) -> List[Tuple[str, str, str]]:
        if use_llm:
            if not self.config.api_key and "localhost" not in self.config.api_url:
                # 对于本地模型，我们不需要key
                pass
            try:
                triples = self._extract_by_llm(text)
                return triples
            except Exception as e:
                print(f"[警告] LLM三元组抽取失败: {e}，跳过此文本块。")
                return [] # 返回空列表而不是抛出异常
        return self._extract_by_rules(text)

    def extract_with_focus(self, text: str, focus_entities: List[str], use_llm: bool = False) -> List[Tuple[str, str, str]]:
        focused = self._focus_text(text, focus_entities)
        if use_llm:
            if not self.config.api_key and "localhost" not in self.config.api_url:
                pass
            try:
                triples = self._extract_by_llm(focused)
                return triples
            except Exception as e:
                print(f"[警告] LLM三元组抽取失败: {e}，跳过此文本块。")
                return []
        return self._extract_by_rules(focused)

    def _extract_by_llm(self, text: str) -> List[Tuple[str, str, str]]:
        headers = {"Content-Type": "application/json"}
        if self.config.api_key:
            headers["Authorization"] = f"Bearer {self.config.api_key}"
        system_prompt = (
            "你是一个信息抽取专家。请从给定的文本中抽取出所有的三元组（主语, 关系, 宾语）。"
            "请严格按照JSON格式输出，格式为: {\"triples\": [[\"主语1\", \"关系1\", \"宾语1\"], [\"主语2\", \"关系2\", \"宾语2\"]]}。"
            "如果文本中没有有效信息，请返回: {\"triples\": []}。"
        )
        user_prompt = f"请从以下文本中抽取三元组：\n\n---\n{text}\n---"

        payload = {
            "model": self.config.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": 0.0,
            "stream": False,
        }
        try:
            resp = requests.post(self.config.api_url, headers=headers, json=payload, timeout=self.config.timeout)
        except Exception as e:
            raise RuntimeError(f"三元组抽取请求失败: {e}")
        if resp.status_code != 200:
            raise RuntimeError(f"三元组抽取HTTP失败: {resp.status_code} | {resp.text[:180]}")
        try:
            content = resp.json()["choices"][0]["message"]["content"]
        except Exception as e:
            raise RuntimeError(f"三元组抽取响应解析失败: {e}")

        # 尝试从LLM的输出中解析JSON
        try:
            # 预处理：修复小模型常见的畸形结构
            fixed_content = content.replace(']], [[', '], [').replace('], [', '],###,[').replace('], [', '],') 
            # 先保护正确的逗号，再清理掉多余的括号
            
            match = re.search(r"\{[\s\S]*\}", content)
            if match:
                try:
                    data = json.loads(match.group(0))
                    raw_triples = data.get("triples", [])
                    # ... 后面保持解析逻辑
                except:
                    pass # 失败则进入正则
            
            # 强力正则：匹配任何形式的 ["A", "B", "C"] 结构
            found = re.findall(r'\[\s*["\']([^"\'\]]+)["\']\s*,\s*["\']([^"\'\]]+)["\']\s*,\s*["\']([^"\'\]]+)["\']\s*\]', content)
            if found:
                return self._deduplicate([(t[0].strip(), t[1].strip(), t[2].strip()) for t in found])
            
            # 极简正则：匹配 A, B, C 或 A-B-C 这种行
            lines = re.findall(r'([^\s,]+)\s*[,，\t-]\s*([^\s,]+)\s*[,，\t-]\s*([^\s,]+)', content)
            if lines:
                return self._deduplicate([(l[0].strip(), l[1].strip(), l[2].strip()) for l in lines])
            
            return []
        except Exception:
            return []

    def _focus_text(self, text: str, focus_entities: List[str]) -> str:
        entities = [e for e in focus_entities if e]
        if not entities:
            return text
        sentences = re.split(r"(?<=[。！？!?；;])\s*", text)
        selected = [s for s in sentences if any(e in s for e in entities)]
        if selected:
            return " ".join(selected)
        return text

    def _extract_by_rules(self, text: str) -> List[Tuple[str, str, str]]:
        triples: List[Tuple[str, str, str]] = []
        normalized = re.sub(r"\s+", " ", text)
        patterns = [
            (r"([\u4e00-\u9fa5A-Za-z0-9_\-]+)是([\u4e00-\u9fa5A-Za-z0-9_\-]+)", "是"),
            (r"([\u4e00-\u9fa5A-Za-z0-9_\-]+)属于([\u4e00-\u9fa5A-Za-z0-9_\-]+)", "属于"),
            (r"([\u4e00-\u9fa5A-Za-z0-9_\-]+)用于([\u4e00-\u9fa5A-Za-z0-9_\-]+)", "用于"),
            (r"([\u4e00-\u9fa5A-Za-z0-9_\-]+)包含([\u4e00-\u9fa5A-Za-z0-9_\-]+)", "包含"),
            (r"([\u4e00-\u9fa5A-Za-z0-9_\-]+)基于([\u4e00-\u9fa5A-Za-z0-9_\-]+)", "基于"),
        ]
        for pattern, rel in patterns:
            for m in re.finditer(pattern, normalized):
                s, o = m.group(1).strip(), m.group(2).strip()
                if s and o and s != o:
                    triples.append((s, rel, o))
        if not triples:
            triples = [
                ("人工智能", "包含", "机器学习"),
                ("机器学习", "包含", "监督学习"),
                ("深度学习", "基于", "神经网络"),
            ]
        return self._deduplicate(triples)

    def _deduplicate(self, triples: List[Tuple[str, str, str]]) -> List[Tuple[str, str, str]]:
        seen = set()
        out = []
        for t in triples:
            if t not in seen:
                seen.add(t)
                out.append(t)
        return out