"""Command-line interface for the DocRAG index."""

import argparse
import json
import os
import sys
from pathlib import Path

from docrag.answer import Assistant
from docrag.embed import DEFAULT_EMBEDDING_MODEL, Embedder
from docrag.evaluate import load_questions, normalize, retrieval_hit_rate, score, summarize
from docrag.index import Index
from docrag.ingest import load_pages, split
from docrag.llm import make_llm
from docrag.retrieve import Retriever

NO_DOCUMENTS_ERROR = "No documents indexed yet. Run `docrag ingest PATH`."


class StoredEmbedder:
    """Index metadata holder for operations that do not embed text."""

    def __init__(self, name: str):
        self.name = name


def index_dir_from(args) -> Path:
    return Path(args.index).expanduser()


def open_index(args, embedder=None) -> Index:
    """Open the selected index using stored model settings by default."""
    directory = index_dir_from(args)
    if embedder is None:
        manifest_path = directory / "manifest.json"
        if getattr(args, "no_model", False):
            model_name = DEFAULT_EMBEDDING_MODEL
            if manifest_path.exists():
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                model_name = manifest["embedding_model"]
            embedder = StoredEmbedder(model_name)
    return Index.open(directory, embedder=embedder)


def print_report(report):
    for label in ("added", "updated", "unchanged"):
        values = getattr(report, label)
        print(f"{label.capitalize()}: {', '.join(values) if values else 'none'}")
    for name, reason in report.skipped:
        print(f"Skipped: {name}: {reason}")
    print(
        f"Totals: added={len(report.added)}, updated={len(report.updated)}, "
        f"unchanged={len(report.unchanged)}, skipped={len(report.skipped)}"
    )


def print_progress(position, total, name, status):
    print(f"[{position}/{total}] {name}: {status}", file=sys.stderr)


def report_add(args):
    directory = index_dir_from(args)
    manifest_path = directory / "manifest.json"
    previous_model = None
    if manifest_path.exists():
        previous_model = json.loads(manifest_path.read_text(encoding="utf-8"))["embedding_model"]
    if args.rebuild:
        for filename in ("manifest.json", "chunks.jsonl", "embeddings.npy"):
            (directory / filename).unlink(missing_ok=True)
    model_name = args.embedding_model or previous_model
    embedder = Embedder(model_name) if model_name else None
    index = Index.open(
        directory,
        embedder=embedder,
        chunk_size=args.chunk_size,
        overlap=args.overlap,
    )
    report = index.add(
        args.paths,
        ocr=args.ocr,
        rebuild=args.rebuild,
        progress=print_progress,
        ocr_language=args.ocr_language,
    )
    print_report(report)


def command_list(args):
    if not (index_dir_from(args) / "manifest.json").exists():
        print("No documents indexed.")
        return
    args.no_model = True
    index = open_index(args)
    for row in index.documents():
        print(f"{row['name']}: {row['pages']} pages, {row['chunks']} chunks")
    if not index.documents():
        print("No documents indexed.")


def command_remove(args):
    args.no_model = True
    index = open_index(args)
    for name in index.remove(args.names):
        print(f"Removed: {name}")


def make_assistant(args, index=None):
    index = index or open_index(args)
    search = Retriever(index.chunks, index.matrix, index.embedder, mode=args.retriever)
    llm = make_llm(args.provider, args.model, args.base_url)
    return Assistant(search, llm, k=args.k, log=index_dir_from(args) / "queries.jsonl")


def require_chunks(index):
    if not index.chunks:
        raise RuntimeError(NO_DOCUMENTS_ERROR)


def show_result(result, as_json=False):
    if as_json:
        print(
            json.dumps(
                {
                    "found": result.found,
                    "answer": result.answer,
                    "sources": result.sources,
                    "retrieved": result.retrieved,
                    "latency_s": result.latency_s,
                    "input_tokens": result.input_tokens,
                    "output_tokens": result.output_tokens,
                },
                ensure_ascii=False,
            )
        )
        return
    print(result.answer)
    for source in result.sources:
        print(f"  source: {source['source']}, page {source['page']}")
    print(f"  ({result.latency_s} s, {result.input_tokens} input / {result.output_tokens} output tokens)")


def command_ask(args):
    index = open_index(args)
    if args.paths:
        print_report(index.add(args.paths, progress=print_progress, ocr_language=args.ocr_language))
    require_chunks(index)
    result = make_assistant(args, index).ask(args.question, _sources(args.source))
    show_result(result, args.json)


def _sources(values):
    return [name for group in (values or []) for name in group] or None


def command_chat(args):
    index = open_index(args)
    require_chunks(index)
    assistant = make_assistant(args, index)
    print("Ask a question. Press Ctrl-D or enter an empty line to quit.")
    while True:
        try:
            question = input("> ")
        except EOFError:
            print()
            return
        if not question.strip():
            return
        show_result(assistant.ask(question, _sources(args.source)))


def indexed_pages(index):
    pages = []
    for row in index.documents():
        pages.extend(load_pages(Path(row["path"]), row["name"]))
    return pages


def command_eval(args):
    index = open_index(args)
    require_chunks(index)
    assistant = make_assistant(args, index)
    page_text = {
        (page.metadata["source"], page.metadata["page"]): normalize(page.page_content) for page in indexed_pages(index)
    }
    rows = []
    for question in load_questions(args.questions):
        rows.append(score(question, assistant.ask(question["question"]), page_text))
        mark = "ok " if rows[-1]["correct"] else "BAD"
        print(f"{mark} {question['id']}: {rows[-1]['answer'][:90]}")
    summary = {
        "model": args.model,
        "provider": args.provider,
        "retriever": args.retriever,
        "chunk_size": index.chunk_size,
        "k": args.k,
    }
    summary |= summarize(rows)
    output_dir = index_dir_from(args) / "eval"
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "answers.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def command_sweep(args):
    index = open_index(args)
    pages = indexed_pages(index)
    rows = []
    for size in args.sizes:
        chunks = split(pages, size)
        matrix = index.embedder.embed_documents([chunk.page_content for chunk in chunks])
        for mode in ("bm25", "dense", "hybrid"):
            search = Retriever(chunks, matrix, index.embedder, mode=mode)
            hit = retrieval_hit_rate(load_questions(args.questions), search, args.k)
            row = {"chunk_size": size, "chunks": len(chunks), "retriever": mode, f"hit@{args.k}": round(hit, 3)}
            rows.append(row)
            print(row)
    output_dir = index_dir_from(args) / "eval"
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "retrieval_sweep.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")


def command_serve(args):
    import uvicorn

    from docrag.server import create_app

    uvicorn.run(
        create_app(
            index_dir_from(args),
            provider=args.provider,
            model=args.model,
            retriever=args.retriever,
            k=args.k,
            base_url=args.base_url,
        ),
        host=args.host,
        port=args.port,
    )


def add_answer_options(parser):
    parser.add_argument("--source", nargs="+", action="append")
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--retriever", choices=("dense", "bm25", "hybrid"), default="hybrid")
    parser.add_argument("--provider", choices=("anthropic", "openai"), default="anthropic")
    parser.add_argument("--model")
    parser.add_argument("--base-url")


def build_parser():
    parser = argparse.ArgumentParser(prog="docrag")
    parser.add_argument("--index", type=Path, default=Path(os.environ.get("DOCRAG_INDEX", ".docrag")))
    commands = parser.add_subparsers(dest="command", required=True)

    ingest = commands.add_parser("ingest")
    ingest.add_argument("paths", type=Path, nargs="+")
    ingest.add_argument("--chunk-size", type=int)
    ingest.add_argument("--overlap", type=int)
    ingest.add_argument("--embedding-model")
    ingest.add_argument("--ocr", choices=("auto", "off"), default="auto")
    ingest.add_argument("--ocr-language", default="eng")
    ingest.add_argument("--rebuild", action="store_true")
    ingest.set_defaults(func=report_add)

    commands.add_parser("list").set_defaults(func=command_list)
    remove = commands.add_parser("remove")
    remove.add_argument("names", nargs="+")
    remove.set_defaults(func=command_remove)

    ask = commands.add_parser("ask")
    ask.add_argument("question")
    ask.add_argument("paths", type=Path, nargs="*")
    ask.add_argument("--ocr-language", default="eng")
    add_answer_options(ask)
    ask.add_argument("--json", action="store_true")
    ask.set_defaults(func=command_ask)

    chat = commands.add_parser("chat")
    add_answer_options(chat)
    chat.set_defaults(func=command_chat)

    evaluate = commands.add_parser("eval")
    evaluate.add_argument("--questions", type=Path, required=True)
    add_answer_options(evaluate)
    evaluate.set_defaults(func=command_eval)

    sweep = commands.add_parser("sweep")
    sweep.add_argument("--questions", type=Path, required=True)
    sweep.add_argument("--sizes", type=int, nargs="+", default=[400, 800, 1600])
    sweep.add_argument("--k", type=int, default=5)
    sweep.set_defaults(func=command_sweep)

    serve = commands.add_parser("serve")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)
    add_answer_options(serve)
    serve.set_defaults(func=command_serve)
    return parser


def main(argv=None):
    try:
        args = build_parser().parse_args(argv)
        if hasattr(args, "k") and args.k < 1:
            raise SystemExit("--k must be positive")
        args.func(args)
    except (FileNotFoundError, ValueError, RuntimeError, KeyError) as exc:
        print(f"docrag: error: {exc}", file=sys.stderr)
        raise SystemExit(1) from None


if __name__ == "__main__":
    main()
