from pathlib import Path
from typing import List

from knowledge.shared.schemas import Evidence


def build_prompt(question: str, evidences: List[Evidence]) -> str:
    lines = [
        "你是人工智能课程助教。请基于证据作答，结论后标注引用编号。",
        f"问题: {question}",
        "证据:",
    ]
    for i, e in enumerate(evidences, 1):
        source = Path(e.source_ref).name if e.source_ref else "unknown"
        lines.append(f"[{i}] {e.content}（来源: {source}）")
    lines.append("输出要求: 准确、简洁、可追溯。")
    return "\n".join(lines)