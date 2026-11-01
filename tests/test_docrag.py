import hashlib
import json

import numpy as np
import pymupdf
import pytest
from fastapi.testclient import TestClient
from langchain_core.documents import Document
from langchain_core.messages import AIMessage

from docrag.answer import NOT_FOUND, Answer, Assistant
from docrag.evaluate import is_correct, normalize, page_hit, score, summarize, supported
from docrag.index import Index
from docrag.ingest import find_pdfs, load_pages, split
from docrag.retrieve import Retriever, reciprocal_rank_fusion, tokenize


class FakeEmbedder:
    name = "fake"
    dimensions = 64

    def __init__(self):
        self.calls = 0

    def _vector(self, text):
        vector = np.zeros(self.dimensions, dtype=np.float32)
        for token in text.lower().split():
            digest = hashlib.sha256(token.encode()).digest()
            position = int.from_bytes(digest[:4], "big") % self.dimensions
            vector[position] += 1
        norm = np.linalg.norm(vector)
        return vector / norm if norm else vector

    def embed_documents(self, texts):
        self.calls += 1
        return np.array([self._vector(text) for text in texts], dtype=np.float32)

    def embed_query(self, text):
        return self._vector(text)


def make_pdf(path, texts):
    path.parent.mkdir(parents=True, exist_ok=True)
    document = pymupdf.open()
    for text in texts:
        page = document.new_page()
        if text:
            page.insert_text((72, 72), text)
    document.save(path)
    document.close()
    return path


@pytest.fixture
def pdf(tmp_path):
    return make_pdf(tmp_path / "report.pdf", ["first page text", "", "third page about net sales"])


def test_find_pdfs_recurses_uppercase_deduplicates_and_validates(tmp_path, pdf):
    uppercase = make_pdf(tmp_path / "nested" / "UPPER.PDF", ["other text"])
    assert find_pdfs([tmp_path, pdf, uppercase]) == sorted([pdf.resolve(), uppercase.resolve()], key=str)
    with pytest.raises(FileNotFoundError):
        find_pdfs([tmp_path / "missing.pdf"])
    text = tmp_path / "note.txt"
    text.write_text("not pdf")
    with pytest.raises(ValueError, match="not a PDF"):
        find_pdfs([text])


def test_load_pages_preserves_nonempty_page_numbers(pdf):
    pages = load_pages(pdf, "report.pdf", ocr="off")
    assert [page.metadata["page"] for page in pages] == [1, 3]
    assert all(page.metadata["source"] == "report.pdf" for page in pages)
    assert pages[0].page_content == "first page text"


def test_load_pages_warns_once_when_tesseract_is_missing(pdf, monkeypatch, caplog):
    from docrag import ingest

    monkeypatch.setattr(ingest.shutil, "which", lambda name: None)
    pages = load_pages(pdf, "report.pdf")
    assert [page.metadata["page"] for page in pages] == [1, 3]
    warnings = [record for record in caplog.records if "Tesseract is not installed" in record.message]
    assert len(warnings) == 1


def test_load_pages_passes_ocr_language_and_resolves_tessdata(tmp_path, monkeypatch):
    from docrag import ingest

    calls = []

    class Page:
        def get_text(self, textpage=None):
            return "Texte français" if textpage is not None else ""

        def get_textpage_ocr(self, language, full, dpi):
            calls.append((language, full, dpi))
            return object()

    class Pdf:
        is_encrypted = False
        page_count = 1

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return None

        def __iter__(self):
            return iter([Page()])

    monkeypatch.setattr(ingest.pymupdf, "open", lambda path: Pdf())
    monkeypatch.setattr(ingest.pymupdf, "get_tessdata", lambda: "/tessdata")
    monkeypatch.setattr(ingest.shutil, "which", lambda name: "/usr/bin/tesseract")
    monkeypatch.setattr(ingest.os, "environ", dict(ingest.os.environ))
    ingest.os.environ.pop("TESSDATA_PREFIX", None)
    pages = load_pages(tmp_path / "scan.pdf", "scan.pdf", ocr_language="fra")
    assert pages[0].page_content == "Texte français"
    assert calls == [("fra", True, 300)]
    assert ingest.os.environ["TESSDATA_PREFIX"] == "/tessdata"


def test_index_skips_pdf_with_no_extractable_text(tmp_path):
    blank = make_pdf(tmp_path / "blank.pdf", [""])
    index = Index.open(tmp_path / "index", FakeEmbedder())
    report = index.add([blank], ocr="off")
    assert report.skipped == [("blank.pdf", "no extractable text")]
    assert index.documents() == []


def test_index_skips_corrupt_pdf_and_keeps_other_files(tmp_path, pdf):
    corrupt = tmp_path / "corrupt.pdf"
    corrupt.write_bytes(b"not a pdf")
    index = Index.open(tmp_path / "idx", FakeEmbedder())
    report = index.add([corrupt, pdf], ocr="off")
    assert report.added == ["report.pdf"]
    assert len(report.skipped) == 1 and report.skipped[0][0] == "corrupt.pdf"
    assert "could not read PDF" in report.skipped[0][1]


def test_split_preserves_metadata_and_numbers_chunks():
    pages = [Document("word " * 400, metadata={"source": "source.pdf", "page": 7})]
    chunks = split(pages, chunk_size=200)
    assert len(chunks) > 1
    assert all(chunk.metadata["page"] == 7 for chunk in chunks)
    assert [chunk.metadata["chunk_id"] for chunk in chunks] == list(range(len(chunks)))


def test_index_persists_incrementally_updates_removes_and_validates(tmp_path, pdf):
    directory = tmp_path / "index"
    embedder = FakeEmbedder()
    index = Index.open(directory, embedder=embedder, chunk_size=80)
    first = index.add([pdf], ocr="off")
    assert first.added == ["report.pdf"]
    previous_chunks = [chunk.page_content for chunk in index.chunks]
    previous_matrix = index.matrix.copy()
    calls = embedder.calls
    reopened = Index.open(directory, embedder=embedder)
    assert [chunk.page_content for chunk in reopened.chunks] == previous_chunks
    np.testing.assert_array_equal(reopened.matrix, previous_matrix)
    assert reopened.add([pdf], ocr="off").unchanged == ["report.pdf"]
    assert embedder.calls == calls
    pdf.unlink()
    make_pdf(pdf, ["changed contents", "", "new net sales figure"])
    assert reopened.add([pdf], ocr="off").updated == ["report.pdf"]
    assert reopened.chunks[0].page_content.startswith("changed")
    assert len(reopened.documents()[0]["path"]) > 0
    with pytest.raises(ValueError, match="--rebuild"):
        Index.open(directory, embedder=embedder, chunk_size=40)
    assert reopened.remove(["report.pdf"]) == ["report.pdf"]
    assert reopened.documents() == []
    assert reopened.chunks == []


def test_index_progress_reports_each_file_status(tmp_path):
    first = make_pdf(tmp_path / "a.pdf", ["original text"])
    second = make_pdf(tmp_path / "b.pdf", ["original contents"])
    blank = make_pdf(tmp_path / "c.pdf", [""])
    index = Index.open(tmp_path / "index", FakeEmbedder())
    index.add([first, second], ocr="off")
    make_pdf(second, ["updated contents"])
    events = []
    report = index.add(
        [first, second, blank],
        ocr="off",
        progress=lambda *event: events.append(event),
    )
    assert report.unchanged == ["a.pdf"]
    assert report.updated == ["b.pdf"]
    assert report.skipped == [("c.pdf", "no extractable text")]
    assert events == [
        (1, 3, "a.pdf", "unchanged"),
        (2, 3, "b.pdf", "updated"),
        (3, 3, "c.pdf", "skipped"),
    ]


def test_duplicate_basenames_get_disambiguated_names(tmp_path):
    first = make_pdf(tmp_path / "a" / "report.pdf", ["alpha text"])
    second = make_pdf(tmp_path / "b" / "report.pdf", ["beta text"])
    index = Index.open(tmp_path / "index", FakeEmbedder())
    report = index.add([first, second], ocr="off")
    assert set(report.added) == {"a/report.pdf", "b/report.pdf"}
    assert {row["name"] for row in index.documents()} == {"a/report.pdf", "b/report.pdf"}


def test_index_rebuild_renumbers_chunks(tmp_path, pdf):
    index = Index.open(tmp_path / "index", FakeEmbedder())
    index.add([pdf], ocr="off")
    result = index.add([pdf], ocr="off", rebuild=True)
    assert result.added == ["report.pdf"]
    assert [chunk.metadata["chunk_id"] for chunk in index.chunks] == list(range(len(index.chunks)))


def test_unicode_tokenizer_and_accent_insensitive_bm25():
    assert tokenize("Résultats net ПРИБЫЛЬ") == ["resultats", "net", "прибыль"]
    chunks = [Document("résultat financier annuel", metadata={"source": "french.pdf", "page": 4, "chunk_id": 0})]
    result = Retriever(chunks, None, None, mode="bm25").search("resultat annuel", k=1)
    assert result[0].metadata["page"] == 4


def test_retrieval_modes_filter_sources_and_fuse():
    chunks = [
        Document("net sales reached ten million euros", metadata={"source": "a.pdf", "page": 1, "chunk_id": 0}),
        Document("the annual report describes net sales", metadata={"source": "b.pdf", "page": 2, "chunk_id": 1}),
        Document("the document has a different topic", metadata={"source": "a.pdf", "page": 3, "chunk_id": 2}),
    ]
    embedder = FakeEmbedder()
    matrix = embedder.embed_documents([chunk.page_content for chunk in chunks])
    for mode in ("dense", "bm25", "hybrid"):
        retriever = Retriever(chunks, matrix, embedder, mode=mode)
        assert len(retriever.search("net sales", k=2)) == 2
        assert all(
            chunk.metadata["source"] == "a.pdf" for chunk in retriever.search("net sales", k=5, sources=["a.pdf"])
        )
    assert reciprocal_rank_fusion([[chunks[0], chunks[1]], [chunks[1], chunks[2]]], k=3)[0] is chunks[1]


class FakeLLM:
    def __init__(self, parsed):
        self.parsed = parsed

    def with_structured_output(self, schema, include_raw):
        return self

    def invoke(self, messages):
        raw = AIMessage("", usage_metadata={"input_tokens": 12, "output_tokens": 3, "total_tokens": 15})
        return {"raw": raw, "parsed": self.parsed}


class FixedRetriever:
    def __init__(self, chunks):
        self.chunks = chunks

    def search(self, query, k, sources=None):
        return self.chunks[:k]


def test_answer_deduplicates_citations_includes_text_and_logs(tmp_path):
    chunks = [
        Document("answer text", metadata={"source": "doc.pdf", "page": 2, "chunk_id": 0}),
        Document("more answer text", metadata={"source": "doc.pdf", "page": 2, "chunk_id": 1}),
    ]
    log = tmp_path / "index" / "queries.jsonl"
    result = Assistant(
        FixedRetriever(chunks), FakeLLM(Answer(found=True, answer="42", citations=[2, 1, 2, 99])), k=2, log=log
    ).ask("question")
    assert result.found
    assert result.sources == [
        {"source": "doc.pdf", "page": 2, "text": "more answer text"},
        {"source": "doc.pdf", "page": 2, "text": "answer text"},
    ]
    assert (result.input_tokens, result.output_tokens) == (12, 3)
    assert json.loads(log.read_text())["sources"] == result.sources


@pytest.mark.parametrize("parsed", [None, Answer(found=False, answer="maybe", citations=[1])])
def test_answer_refuses_and_drops_citations(parsed):
    chunk = Document("some text", metadata={"source": "doc.pdf", "page": 1, "chunk_id": 0})
    result = Assistant(FixedRetriever([chunk]), FakeLLM(parsed), k=1).ask("unknown?")
    assert (result.found, result.answer, result.sources) == (False, NOT_FOUND, [])


def test_evaluation_metrics():
    question = {
        "id": "q1",
        "question": "How much?",
        "answerable": True,
        "expect": ["10959"],
        "source": "report.pdf",
        "pages": [110],
    }
    assert normalize("€10,959 m") == "10959m"
    assert is_correct(question, True, "Net sales were €10,959m.")
    assert not is_correct(question, False, NOT_FOUND)
    assert is_correct({"answerable": False}, False, NOT_FOUND)
    assert not is_correct({"answerable": False}, True, "About 5 billion")
    assert page_hit(question, [{"source": "report.pdf", "page": 110}])
    assert not page_hit(question, [{"source": "other.pdf", "page": 110}])
    page_text = {("report.pdf", 110): "net sales10959"}
    assert supported(question, [{"source": "report.pdf", "page": 110}], page_text)
    result = type(
        "Result",
        (),
        {
            "found": True,
            "answer": "10,959",
            "sources": [{"source": "report.pdf", "page": 110, "text": "net sales 10,959"}],
            "retrieved": [{"source": "report.pdf", "page": 110}],
            "latency_s": 1.0,
            "input_tokens": 10,
            "output_tokens": 2,
        },
    )()
    row = score(question, result, page_text)
    assert row["correct"] and row["retrieval_hit"] and row["citation_ok"]
    summary = summarize(
        [
            row,
            {
                "answerable": False,
                "found": True,
                "correct": False,
                "latency_s": 3.0,
                "input_tokens": 10,
                "output_tokens": 2,
            },
        ]
    )
    assert summary["answer_accuracy"] == 1 and summary["hallucinated_answers"] == 1


def test_server_upload_list_ask_delete_pdf_and_empty_index(tmp_path):
    from docrag.server import create_app

    llm = FakeLLM(Answer(found=True, answer="The result is 42.", citations=[1]))
    app = create_app(
        tmp_path / "index",
        embedder_factory=FakeEmbedder,
        llm_factory=lambda **kwargs: llm,
    )
    client = TestClient(app)
    empty = TestClient(create_app(tmp_path / "empty", embedder_factory=FakeEmbedder, llm_factory=lambda **kwargs: llm))
    assert empty.post("/api/ask", json={"question": "anything"}).status_code == 400
    first = pymupdf.open()
    first.new_page().insert_text((72, 72), "The answer is 42.")
    pdf_one = first.tobytes()
    first.close()
    second = pymupdf.open()
    second.new_page().insert_text((72, 72), "A separate document.")
    pdf_two = second.tobytes()
    second.close()
    response = client.post(
        "/api/documents",
        files=[
            ("files", ("first.pdf", pdf_one, "application/pdf")),
            ("files", ("second.PDF", pdf_two, "application/pdf")),
            ("files", ("notes.txt", b"text file", "text/plain")),
        ],
    )
    assert response.status_code == 200
    assert len(response.json()["added"]) == 2
    assert response.json()["skipped"] == [{"name": "notes.txt", "reason": "not a PDF file"}]
    listed = client.get("/api/documents").json()
    assert {row["name"] for row in listed["documents"]} == {"first.pdf", "second.PDF"}
    answer = client.post("/api/ask", json={"question": "What is the answer?"})
    assert answer.status_code == 200
    assert answer.json()["sources"][0]["text"]
    pdf_response = client.get("/api/documents/first.pdf/pdf")
    assert pdf_response.status_code == 200 and pdf_response.headers["content-type"] == "application/pdf"
    assert client.delete("/api/documents/first.pdf").status_code == 200
    assert client.delete("/api/documents/first.pdf").status_code == 404


def test_server_reloads_index_after_manifest_changes(tmp_path):
    from docrag.server import create_app

    index_dir = tmp_path / "index"
    first = make_pdf(tmp_path / "first.pdf", ["Original indexed text."])
    second = make_pdf(tmp_path / "second.pdf", ["New external indexed text."])
    Index.open(index_dir, FakeEmbedder()).add([first], ocr="off")
    llm = FakeLLM(Answer(found=True, answer="The answer.", citations=[1]))
    client = TestClient(create_app(index_dir, embedder_factory=FakeEmbedder, llm_factory=lambda **kwargs: llm))
    assert client.post("/api/ask", json={"question": "original"}).status_code == 200

    Index.open(index_dir, FakeEmbedder()).add([second], ocr="off")
    response = client.post("/api/ask", json={"question": "external", "sources": ["second.pdf"]})
    assert response.status_code == 200
    assert response.json()["sources"][0]["source"] == "second.pdf"


def test_server_removes_new_upload_when_index_skips_it(tmp_path):
    from docrag.server import create_app

    index_dir = tmp_path / "index"
    llm = FakeLLM(Answer(found=False, answer=NOT_FOUND, citations=[]))
    client = TestClient(create_app(index_dir, embedder_factory=FakeEmbedder, llm_factory=lambda **kwargs: llm))
    response = client.post(
        "/api/documents",
        files=[("files", ("broken.pdf", b"not a PDF", "application/pdf"))],
    )
    assert response.status_code == 200
    assert response.json()["skipped"][0]["name"] == "broken.pdf"
    assert not (index_dir / "files" / "broken.pdf").exists()


def test_cli_ingest_and_list_with_fake_embedder(tmp_path, pdf, monkeypatch, capsys):
    from docrag import cli

    monkeypatch.setattr(cli, "Embedder", lambda *args, **kwargs: FakeEmbedder())
    cli.main(["--index", str(tmp_path / "index"), "ingest", str(pdf), "--ocr", "off"])
    captured = capsys.readouterr()
    assert "added=1" in captured.out
    assert "[1/1] report.pdf: added" in captured.err
    cli.main(["--index", str(tmp_path / "index"), "list"])
    assert "report.pdf" in capsys.readouterr().out


def test_cli_ask_paths_reports_ingest_progress(tmp_path, monkeypatch, capsys):
    from docrag import cli

    pdf = make_pdf(tmp_path / "report.pdf", ["report text"])
    monkeypatch.setattr(cli, "Embedder", lambda *args, **kwargs: FakeEmbedder())
    monkeypatch.setattr(
        cli,
        "make_llm",
        lambda *args, **kwargs: FakeLLM(Answer(found=False, answer=NOT_FOUND, citations=[])),
    )
    cli.main(
        [
            "--index",
            str(tmp_path / "index"),
            "ask",
            "question",
            str(pdf),
            "--ocr-language",
            "fra",
        ]
    )
    assert "[1/1] report.pdf: added" in capsys.readouterr().err


def test_cli_empty_operations_error_without_creating_list_index(tmp_path, monkeypatch, capsys):
    from docrag import cli

    index_dir = tmp_path / "missing"
    cli.main(["--index", str(index_dir), "list"])
    assert capsys.readouterr().out == "No documents indexed.\n"
    assert not index_dir.exists()

    monkeypatch.setattr(cli, "Embedder", lambda *args, **kwargs: FakeEmbedder())
    monkeypatch.setattr(cli, "make_llm", lambda **kwargs: pytest.fail("LLM must not be created"))
    questions = tmp_path / "questions.jsonl"
    questions.write_text("")
    commands = [
        ["ask", "question"],
        ["chat"],
        ["eval", "--questions", str(questions)],
    ]
    for command in commands:
        with pytest.raises(SystemExit) as error:
            cli.main(["--index", str(index_dir), *command])
        assert error.value.code == 1
        assert capsys.readouterr().err == "docrag: error: No documents indexed yet. Run `docrag ingest PATH`.\n"
