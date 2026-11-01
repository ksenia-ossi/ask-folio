"""Grounded question answering with citations and refusal behavior."""

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

NOT_FOUND = "Not found in the documents."

SYSTEM = f"""You answer questions using only the numbered excerpts provided.
Rules:
- Use only facts stated in the excerpts. Do not use outside knowledge or guess.
- Cite every excerpt you used by its number.
- Answer in the same language as the question.
- If the excerpts do not contain the answer, set found to false and answer exactly: "{NOT_FOUND}"
- Keep the answer short."""


class Answer(BaseModel):
    """Structured answer returned by the language model."""

    found: bool
    answer: str
    citations: list[int] = Field(default_factory=list)


@dataclass
class Result:
    """Answer, citations, retrieval metadata, and token usage."""

    question: str
    found: bool
    answer: str
    sources: list[dict] = field(default_factory=list)
    latency_s: float = 0
    input_tokens: int = 0
    output_tokens: int = 0
    retrieved: list[dict] = field(default_factory=list)


def format_context(chunks) -> str:
    """Number retrieved excerpts and include their source and page."""
    return "\n\n".join(
        f"[{number}] ({chunk.metadata['source']}, page {chunk.metadata['page']})\n{chunk.page_content}"
        for number, chunk in enumerate(chunks, start=1)
    )


class Assistant:
    """Answer questions with retrieved context and log each query."""

    def __init__(self, retriever, llm, k: int = 5, log: Path | None = None):
        self.retriever = retriever
        self.llm = llm.with_structured_output(Answer, include_raw=True)
        self.k = k
        self.log = log

    def ask(self, question: str, sources: list[str] | None = None) -> Result:
        """Retrieve relevant excerpts and answer from them only."""
        chunks = self.retriever.search(question, k=self.k, sources=sources)
        messages = [
            SystemMessage(SYSTEM),
            HumanMessage(f"Excerpts:\n\n{format_context(chunks)}\n\nQuestion: {question}"),
        ]
        started = time.perf_counter()
        out = self.llm.invoke(messages)
        latency = time.perf_counter() - started
        parsed: Answer | None = out["parsed"]
        raw = out.get("raw")
        usage = getattr(raw, "usage_metadata", None) or {}
        if parsed is None or not parsed.found:
            found, answer, used = False, NOT_FOUND, []
        else:
            found, answer = True, parsed.answer
            used = [
                chunks[index - 1] for index in parsed.citations if isinstance(index, int) and 1 <= index <= len(chunks)
            ]
        unique_sources = []
        seen = set()
        for chunk in used:
            source = {
                "source": chunk.metadata["source"],
                "page": chunk.metadata["page"],
                "text": chunk.page_content,
            }
            key = (source["source"], source["page"], source["text"])
            if key not in seen:
                seen.add(key)
                unique_sources.append(source)
        result = Result(
            question=question,
            found=found,
            answer=answer,
            sources=unique_sources,
            latency_s=round(latency, 3),
            input_tokens=usage.get("input_tokens", 0),
            output_tokens=usage.get("output_tokens", 0),
            retrieved=[{"source": chunk.metadata["source"], "page": chunk.metadata["page"]} for chunk in chunks],
        )
        if self.log is not None:
            self.log.parent.mkdir(parents=True, exist_ok=True)
            with self.log.open("a", encoding="utf-8") as file:
                file.write(json.dumps(asdict(result), ensure_ascii=False) + "\n")
        return result
