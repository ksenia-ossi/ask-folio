"""FastAPI service for document management and grounded questions."""

import hashlib
import json
import re
import threading
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel

from docrag.answer import Assistant
from docrag.embed import DEFAULT_EMBEDDING_MODEL, Embedder
from docrag.index import DEFAULT_CHUNK_SIZE, Index
from docrag.llm import make_llm
from docrag.retrieve import Retriever

UPLOAD_FILES = File(...)


class AskRequest(BaseModel):
    """Question request payload."""

    question: str
    sources: list[str] | None = None


def create_app(
    index_dir: Path,
    embedder_factory=Embedder,
    llm_factory=make_llm,
    provider: str = "anthropic",
    model: str | None = None,
    retriever: str = "hybrid",
    k: int = 5,
    base_url: str | None = None,
) -> FastAPI:
    """Build an API app without loading models until they are needed."""
    index_dir = Path(index_dir)
    index_lock = threading.Lock()
    state = {"index": None, "embedder": None, "llm": None}
    app = FastAPI()

    def get_index() -> Index:
        if state["index"] is None:
            if state["embedder"] is None:
                if embedder_factory is Embedder:
                    state["embedder"] = Embedder(manifest()["embedding_model"])
                else:
                    state["embedder"] = embedder_factory()
            state["index"] = Index.open(index_dir, embedder=state["embedder"])
        return state["index"]

    def manifest() -> dict:
        path = index_dir / "manifest.json"
        return (
            json.loads(path.read_text(encoding="utf-8"))
            if path.exists()
            else {
                "embedding_model": DEFAULT_EMBEDDING_MODEL,
                "chunk_size": DEFAULT_CHUNK_SIZE,
                "documents": {},
            }
        )

    def document_list() -> list[dict]:
        return [
            {"name": name, "pages": data["pages"], "chunks": data["chunks"]}
            for name, data in manifest()["documents"].items()
        ]

    @app.get("/", response_class=HTMLResponse)
    async def home():
        page = Path(__file__).parent / "web" / "index.html"
        return HTMLResponse(page.read_text(encoding="utf-8"))

    @app.get("/api/documents")
    async def list_documents():
        data = manifest()
        return {
            "documents": [
                {"name": name, "pages": row["pages"], "chunks": row["chunks"]}
                for name, row in data["documents"].items()
            ],
            "settings": {
                "embedding_model": data["embedding_model"],
                "chunk_size": data["chunk_size"],
                "retriever": retriever,
                "k": k,
                "provider": provider,
                "model": model or ("claude-haiku-5-5" if provider == "anthropic" else "gpt-4o-mini"),
            },
        }

    @app.post("/api/documents")
    async def upload_documents(files: list[UploadFile] = UPLOAD_FILES):
        upload_dir = index_dir / "files"
        upload_dir.mkdir(parents=True, exist_ok=True)
        paths = []
        skipped = []
        for upload in files:
            original = (upload.filename or "upload.pdf").replace("\\", "/").split("/")[-1]
            filename = re.sub(r"[\x00-\x1f/\\\\]", "_", original).strip(" .") or "upload"
            if Path(filename).suffix.lower() != ".pdf":
                skipped.append({"name": filename, "reason": "not a PDF file"})
                continue
            content = await upload.read()
            target = upload_dir / filename
            digest = hashlib.sha256(content).digest()
            suffix = target.suffix
            stem = target.stem
            number = 1
            while target.exists() and hashlib.sha256(target.read_bytes()).digest() != digest:
                number += 1
                target = upload_dir / f"{stem}-{number}{suffix}"
            if not target.exists():
                target.write_bytes(content)
            paths.append(target)
        report = {"added": [], "updated": [], "unchanged": [], "skipped": skipped}
        if paths:
            try:
                with index_lock:
                    result = get_index().add(paths)
                report["added"] = result.added
                report["updated"] = result.updated
                report["unchanged"] = result.unchanged
                report["skipped"].extend({"name": name, "reason": reason} for name, reason in result.skipped)
            except (FileNotFoundError, ValueError) as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc
            except Exception as exc:
                raise HTTPException(status_code=500, detail=str(exc)) from exc
        report["documents"] = document_list()
        return report

    @app.delete("/api/documents/{name:path}")
    async def delete_document(name: str):
        data = manifest()
        if name not in data["documents"]:
            raise HTTPException(status_code=404, detail="Document not found")
        file_path = Path(data["documents"][name]["path"])
        try:
            with index_lock:
                index = get_index()
                index.remove([name])
            resolved = file_path.resolve()
            files_root = (index_dir / "files").resolve()
            if resolved.is_relative_to(files_root) and resolved.is_file():
                resolved.unlink()
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="Document not found") from exc
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc
        return {"documents": document_list()}

    @app.get("/api/documents/{name:path}/pdf")
    async def get_pdf(name: str):
        data = manifest()
        row = data["documents"].get(name)
        if row is None:
            raise HTTPException(status_code=404, detail="Document not found")
        path = Path(row["path"])
        if not path.is_file():
            raise HTTPException(status_code=404, detail="PDF file not found")
        return FileResponse(path, media_type="application/pdf")

    @app.post("/api/ask")
    async def ask(request: AskRequest):
        if not request.question.strip():
            raise HTTPException(status_code=400, detail="Question cannot be empty")
        if not manifest()["documents"]:
            raise HTTPException(status_code=400, detail="No documents indexed yet")
        try:
            index = get_index()
            if not index.chunks:
                raise HTTPException(status_code=400, detail="No documents indexed yet")
            if state["llm"] is None:
                state["llm"] = llm_factory(provider=provider, model=model, base_url=base_url)
            search = Retriever(index.chunks, index.matrix, index.embedder, mode=retriever)
            result = Assistant(search, state["llm"], k=k, log=index_dir / "queries.jsonl").ask(
                request.question,
                sources=request.sources,
            )
        except HTTPException:
            raise
        except RuntimeError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc
        return {
            "found": result.found,
            "answer": result.answer,
            "sources": result.sources,
            "latency_s": result.latency_s,
            "input_tokens": result.input_tokens,
            "output_tokens": result.output_tokens,
        }

    return app
