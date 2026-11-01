"""FastAPI service for document management and grounded questions."""

import hashlib
import json
import re
import threading
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel

from askfolio.answer import Assistant
from askfolio.embed import DEFAULT_EMBEDDING_MODEL, Embedder
from askfolio.index import DEFAULT_CHUNK_SIZE, Index
from askfolio.llm import make_llm
from askfolio.retrieve import Retriever

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
    index_lock = threading.RLock()
    state = {"index": None, "manifest_mtime": None, "embedder": None, "llm": None, "retriever": None}
    app = FastAPI()
    manifest_path = index_dir / "manifest.json"

    def manifest_mtime() -> int | None:
        try:
            return manifest_path.stat().st_mtime_ns
        except FileNotFoundError:
            return None

    def read_manifest() -> dict:
        return (
            json.loads(manifest_path.read_text(encoding="utf-8"))
            if manifest_path.exists()
            else {
                "embedding_model": DEFAULT_EMBEDDING_MODEL,
                "chunk_size": DEFAULT_CHUNK_SIZE,
                "documents": {},
            }
        )

    def get_index() -> Index:
        with index_lock:
            observed_mtime = manifest_mtime()
            if state["index"] is not None and state["manifest_mtime"] == observed_mtime:
                return state["index"]
            while True:
                observed_mtime = manifest_mtime()
                data = read_manifest()
                if observed_mtime != manifest_mtime():
                    continue
                model_name = data["embedding_model"]
                if state["embedder"] is None or getattr(state["embedder"], "name", None) != model_name:
                    state["embedder"] = Embedder(model_name) if embedder_factory is Embedder else embedder_factory()
                try:
                    index = Index.open(index_dir, embedder=state["embedder"])
                except ValueError:
                    if observed_mtime != manifest_mtime():
                        continue
                    raise
                if observed_mtime != manifest_mtime():
                    continue
                state["index"] = index
                state["manifest_mtime"] = observed_mtime
                state["retriever"] = None
                return index

    def manifest() -> dict:
        with index_lock:
            if state["index"] is not None and state["manifest_mtime"] != manifest_mtime():
                get_index()
            return read_manifest()

    def mark_index_changed() -> None:
        state["manifest_mtime"] = manifest_mtime()
        state["retriever"] = None

    def document_list() -> list[dict]:
        return [
            {"name": name, "pages": data["pages"], "chunks": data["chunks"]}
            for name, data in manifest()["documents"].items()
        ]

    @app.get("/", response_class=HTMLResponse)
    def home():
        page = Path(__file__).parent / "web" / "index.html"
        return HTMLResponse(page.read_text(encoding="utf-8"))

    @app.get("/api/documents")
    def list_documents():
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
    def upload_documents(files: list[UploadFile] = UPLOAD_FILES):
        upload_dir = index_dir / "files"
        upload_dir.mkdir(parents=True, exist_ok=True)
        paths = []
        newly_written = {}
        skipped = []
        for upload in files:
            original = (upload.filename or "upload.pdf").replace("\\", "/").split("/")[-1]
            filename = re.sub(r"[\x00-\x1f/\\\\]", "_", original).strip(" .") or "upload"
            if Path(filename).suffix.lower() != ".pdf":
                skipped.append({"name": filename, "reason": "not a PDF file"})
                continue
            content = upload.file.read()
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
                newly_written[target.name] = target
            paths.append(target)
        report = {"added": [], "updated": [], "unchanged": [], "skipped": skipped}
        if paths:
            try:
                with index_lock:
                    result = get_index().add(paths)
                    if result.added or result.updated:
                        mark_index_changed()
                report["added"] = result.added
                report["updated"] = result.updated
                report["unchanged"] = result.unchanged
                report["skipped"].extend({"name": name, "reason": reason} for name, reason in result.skipped)
                for name, _ in result.skipped:
                    path = newly_written.get(Path(name).name)
                    if path is not None:
                        path.unlink(missing_ok=True)
            except (FileNotFoundError, ValueError) as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc
            except Exception as exc:
                raise HTTPException(status_code=500, detail=str(exc)) from exc
        report["documents"] = document_list()
        return report

    @app.delete("/api/documents/{name:path}")
    def delete_document(name: str):
        data = manifest()
        if name not in data["documents"]:
            raise HTTPException(status_code=404, detail="Document not found")
        file_path = Path(data["documents"][name]["path"])
        try:
            with index_lock:
                index = get_index()
                index.remove([name])
                mark_index_changed()
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
    def get_pdf(name: str):
        data = manifest()
        row = data["documents"].get(name)
        if row is None:
            raise HTTPException(status_code=404, detail="Document not found")
        path = Path(row["path"])
        if not path.is_file():
            raise HTTPException(status_code=404, detail="PDF file not found")
        return FileResponse(path, media_type="application/pdf")

    @app.post("/api/ask")
    def ask(request: AskRequest):
        if not request.question.strip():
            raise HTTPException(status_code=400, detail="Question cannot be empty")
        if not manifest()["documents"]:
            raise HTTPException(status_code=400, detail="No documents indexed yet")
        try:
            with index_lock:
                index = get_index()
                if not index.chunks:
                    raise HTTPException(status_code=400, detail="No documents indexed yet")
                if state["retriever"] is None:
                    state["retriever"] = Retriever(index.chunks, index.matrix, index.embedder, mode=retriever)
                if state["llm"] is None:
                    state["llm"] = llm_factory(provider=provider, model=model, base_url=base_url)
                search = state["retriever"]
                llm = state["llm"]
            result = Assistant(search, llm, k=k, log=index_dir / "queries.jsonl").ask(
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
