"""Persistent, incremental PDF chunk and embedding storage."""

import hashlib
import json
import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pymupdf
from langchain_core.documents import Document

from docrag.embed import DEFAULT_EMBEDDING_MODEL, Embedder
from docrag.ingest import find_pdfs, load_pages, split

DEFAULT_CHUNK_SIZE = 400


@dataclass
class IngestReport:
    """Summary of one index update."""

    added: list[str] = field(default_factory=list)
    updated: list[str] = field(default_factory=list)
    unchanged: list[str] = field(default_factory=list)
    skipped: list[tuple[str, str]] = field(default_factory=list)


class Index:
    """A persistent chunk collection with an aligned embedding matrix."""

    def __init__(self, directory: Path, embedder, chunk_size: int, overlap: int):
        self.directory = Path(directory)
        self.embedder = embedder
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.embedding_model = getattr(embedder, "name", DEFAULT_EMBEDDING_MODEL)
        self._documents: dict[str, dict] = {}
        self.chunks: list[Document] = []
        self.matrix = np.empty((0, 0), dtype=np.float32)

    @classmethod
    def open(
        cls,
        directory: Path,
        embedder=None,
        chunk_size: int | None = None,
        overlap: int | None = None,
    ) -> "Index":
        """Load an index, or create an empty one with the requested settings."""
        directory = Path(directory)
        manifest_path = directory / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else None
        if manifest:
            stored_size = manifest["chunk_size"]
            stored_overlap = manifest["overlap"]
            if chunk_size is not None and chunk_size != stored_size:
                raise ValueError("index chunk size differs; pass --rebuild to create a new index")
            if overlap is not None and overlap != stored_overlap:
                raise ValueError("index overlap differs; pass --rebuild to create a new index")
            resolved_size = stored_size
            resolved_overlap = stored_overlap
            model_name = manifest["embedding_model"]
            if embedder is None:
                embedder = Embedder(model_name)
            elif getattr(embedder, "name", model_name) != model_name:
                raise ValueError("index embedding model differs; pass --rebuild to create a new index")
        else:
            resolved_size = DEFAULT_CHUNK_SIZE if chunk_size is None else chunk_size
            resolved_overlap = resolved_size // 5 if overlap is None else overlap
            if embedder is None:
                embedder = Embedder()
        if resolved_size <= 0 or resolved_overlap < 0 or resolved_overlap >= resolved_size:
            raise ValueError("chunk size must be positive and overlap must be between zero and chunk size")
        index = cls(directory, embedder, resolved_size, resolved_overlap)
        if manifest:
            index._documents = manifest["documents"]
            chunk_path = directory / "chunks.jsonl"
            index.chunks = []
            if chunk_path.exists():
                for line in chunk_path.read_text(encoding="utf-8").splitlines():
                    row = json.loads(line)
                    index.chunks.append(
                        Document(
                            page_content=row["text"],
                            metadata={
                                "source": row["source"],
                                "page": row["page"],
                                "chunk_id": row["chunk_id"],
                            },
                        )
                    )
            matrix_path = directory / "embeddings.npy"
            index.matrix = (
                np.load(matrix_path).astype(np.float32) if matrix_path.exists() else np.empty((0, 0), dtype=np.float32)
            )
            if len(index.chunks) != len(index.matrix):
                raise ValueError("index chunks and embeddings are inconsistent; pass --rebuild")
            if len(index.chunks) != sum(row["chunks"] for row in index._documents.values()):
                raise ValueError("index manifest and chunks are inconsistent; pass --rebuild")
        index.directory.mkdir(parents=True, exist_ok=True)
        return index

    @staticmethod
    def _sha256(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as file:
            for block in iter(lambda: file.read(1024 * 1024), b""):
                digest.update(block)
        return digest.hexdigest()

    @staticmethod
    def _atomic_write(path: Path, content: bytes) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, temp_name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.")
        try:
            with os.fdopen(fd, "wb") as file:
                file.write(content)
                file.flush()
                os.fsync(file.fileno())
            os.replace(temp_name, path)
        finally:
            if os.path.exists(temp_name):
                os.unlink(temp_name)

    def _save(self) -> None:
        chunk_rows = [
            {
                "text": chunk.page_content,
                "source": chunk.metadata["source"],
                "page": chunk.metadata["page"],
                "chunk_id": row,
            }
            for row, chunk in enumerate(self.chunks)
        ]
        self._atomic_write(
            self.directory / "chunks.jsonl",
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in chunk_rows).encode("utf-8"),
        )
        fd, temp_name = tempfile.mkstemp(dir=self.directory, prefix=".embeddings.")
        try:
            with os.fdopen(fd, "wb") as file:
                np.save(file, self.matrix.astype(np.float32))
                file.flush()
                os.fsync(file.fileno())
            os.replace(temp_name, self.directory / "embeddings.npy")
        finally:
            if os.path.exists(temp_name):
                os.unlink(temp_name)
        manifest = {
            "version": 1,
            "embedding_model": self.embedding_model,
            "chunk_size": self.chunk_size,
            "overlap": self.overlap,
            "documents": self._documents,
        }
        self._atomic_write(
            self.directory / "manifest.json",
            json.dumps(manifest, ensure_ascii=False, indent=2).encode("utf-8"),
        )

    def _display_names(self, paths: list[Path]) -> dict[str, str]:
        existing_paths = {row["path"]: name for name, row in self._documents.items()}
        assigned = {str(path): existing_paths[str(path)] for path in paths if str(path) in existing_paths}
        pending = [path for path in paths if str(path) not in existing_paths]
        basename_counts = {}
        for path in pending:
            basename_counts[path.name] = basename_counts.get(path.name, 0) + 1
        existing_names = set(self._documents)
        for path in pending:
            if basename_counts[path.name] > 1:
                parent_parts = list(path.parent.parts)
                for depth in range(1, len(parent_parts) + 1):
                    candidate = Path(*parent_parts[-depth:], path.name).as_posix()
                    if candidate not in existing_names and candidate not in assigned.values():
                        break
            else:
                candidate = path.name
                if candidate in existing_names or candidate in assigned.values():
                    parent_parts = list(path.parent.parts)
                    for depth in range(1, len(parent_parts) + 1):
                        candidate = Path(*parent_parts[-depth:], path.name).as_posix()
                        if candidate not in existing_names and candidate not in assigned.values():
                            break
            assigned[str(path)] = candidate
        return assigned

    def add(
        self,
        paths: list[Path],
        ocr: str = "auto",
        rebuild: bool = False,
        progress=None,
        ocr_language: str = "eng",
    ) -> IngestReport:
        """Incrementally index PDFs and preserve unchanged embeddings."""
        pdfs = find_pdfs(paths)
        report = IngestReport()
        if rebuild:
            self._documents = {}
            self.chunks = []
            self.matrix = np.empty((0, 0), dtype=np.float32)
        names = self._display_names(pdfs)
        current_by_path = {row["path"]: name for name, row in self._documents.items()}
        replacements = []
        additions = []

        def notify(position: int, name: str, status: str) -> None:
            if progress is not None:
                progress(position, len(pdfs), name, status)

        for position, pdf in enumerate(pdfs, start=1):
            path = str(pdf.resolve())
            name = names[path]
            digest = self._sha256(pdf)
            existing_name = current_by_path.get(path)
            if existing_name and self._documents[existing_name]["sha256"] == digest:
                report.unchanged.append(existing_name)
                notify(position, existing_name, "unchanged")
                continue
            try:
                pages = load_pages(pdf, name, ocr=ocr, ocr_language=ocr_language)
                if not pages:
                    report.skipped.append((name, "no extractable text"))
                    notify(position, name, "skipped")
                    continue
                chunks = split(pages, self.chunk_size, self.overlap)
                if not chunks:
                    report.skipped.append((name, "no extractable text"))
                    notify(position, name, "skipped")
                    continue
                with pymupdf.open(pdf) as document:
                    page_count = document.page_count
            except (ValueError, OSError, RuntimeError, pymupdf.FileDataError) as exc:
                report.skipped.append((name, str(exc)))
                notify(position, name, "skipped")
                continue
            vectors = self.embedder.embed_documents([chunk.page_content for chunk in chunks])
            vectors = np.asarray(vectors, dtype=np.float32)
            if vectors.ndim != 2 or len(vectors) != len(chunks):
                raise ValueError("embedder returned an invalid document matrix")
            if existing_name:
                replacements.append((existing_name, name, path, digest, page_count, chunks, vectors))
                report.updated.append(existing_name)
                notify(position, existing_name, "updated")
            else:
                additions.append((name, path, digest, page_count, chunks, vectors))
                report.added.append(name)
                notify(position, name, "added")
        removed_chunks = {name for name, *_ in replacements}
        kept_chunks = [chunk for chunk in self.chunks if chunk.metadata["source"] not in removed_chunks]
        kept_matrix_indices = [
            i for i, chunk in enumerate(self.chunks) if chunk.metadata["source"] not in removed_chunks
        ]
        matrix_parts = [self.matrix[kept_matrix_indices]] if kept_matrix_indices else []
        updated_docs = {name: row for name, row in self._documents.items() if name not in removed_chunks}
        for old_name, new_name, path, digest, page_count, chunks, vectors in replacements:
            updated_docs[new_name] = {
                "path": path,
                "sha256": digest,
                "pages": page_count,
                "chunks": len(chunks),
            }
            for chunk in chunks:
                chunk.metadata["source"] = new_name
            kept_chunks.extend(chunks)
            matrix_parts.append(vectors)
        for name, path, digest, page_count, chunks, vectors in additions:
            updated_docs[name] = {"path": path, "sha256": digest, "pages": page_count, "chunks": len(chunks)}
            kept_chunks.extend(chunks)
            matrix_parts.append(vectors)
        if additions or replacements or rebuild:
            self._documents = updated_docs
            self.chunks = kept_chunks
            self.matrix = np.concatenate(matrix_parts, axis=0) if matrix_parts else np.empty((0, 0), dtype=np.float32)
            for chunk_id, chunk in enumerate(self.chunks):
                chunk.metadata["chunk_id"] = chunk_id
            self._save()
        return report

    def remove(self, names: list[str]) -> list[str]:
        """Remove named documents from the index."""
        missing = [name for name in names if name not in self._documents]
        if missing:
            raise KeyError(f"unknown document: {missing[0]}")
        removed = set(names)
        indices = [i for i, chunk in enumerate(self.chunks) if chunk.metadata["source"] not in removed]
        self.chunks = [self.chunks[i] for i in indices]
        self.matrix = (
            self.matrix[indices]
            if indices
            else np.empty((0, self.matrix.shape[1] if self.matrix.ndim == 2 else 0), dtype=np.float32)
        )
        for name in removed:
            del self._documents[name]
        for chunk_id, chunk in enumerate(self.chunks):
            chunk.metadata["chunk_id"] = chunk_id
        self._save()
        return names

    def documents(self) -> list[dict]:
        """Return document names and index statistics."""
        return [
            {"name": name, "pages": row["pages"], "chunks": row["chunks"], "path": row["path"]}
            for name, row in self._documents.items()
        ]
