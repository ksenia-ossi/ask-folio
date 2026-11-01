# ask-folio

Ask questions about your own PDFs. Point it at any number of files or folders, in any language; it extracts the text, splits it into chunks, indexes them on disk and answers questions using only those chunks. Every answer cites the file and page it comes from. When the documents do not contain the answer, it says `Not found in the documents.` instead of guessing.

It works from the command line or from a small web page where you drop PDFs and ask questions.

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
export ANTHROPIC_API_KEY=...         # never commit it

askfolio ingest ~/papers/ contracts/2025.pdf
askfolio ask "What notice period does the contract set?"
askfolio serve                         # web page on http://127.0.0.1:8000
```

`askfolio ask QUESTION PATH...` indexes the given files first, so a single command is enough for a one-off question.

## Commands

| command | what it does |
|---|---|
| `askfolio ingest PATH...` | index PDFs (files or folders, searched recursively). Re-running it only processes new or modified files. |
| `askfolio list` | indexed documents with page and chunk counts |
| `askfolio remove NAME...` | remove documents from the index |
| `askfolio ask QUESTION [PATH...]` | one answer with citations; `--source NAME` restricts it to some documents, `--json` for machine output |
| `askfolio chat` | interactive questions in the terminal |
| `askfolio serve` | web page: upload PDFs, choose which documents to search, ask, open the cited page |
| `askfolio eval --questions FILE` | score answers on your own question set |
| `askfolio sweep --questions FILE` | compare chunk sizes and retrievers on retrieval only (no LLM calls) |

The index lives in `.askfolio/` by default; use `--index DIR` or `ASKFOLIO_INDEX` to keep separate collections. Ingest options: `--chunk-size`, `--overlap`, `--embedding-model`, `--ocr auto|off`, `--ocr-language` (Tesseract codes such as `fra` or `eng+fra`), `--rebuild`.

## How it works

1. **Parsing** (`askfolio/ingest.py`): PyMuPDF reads each page. Pages without a text layer (scans) go through Tesseract OCR when it is installed (`apt install tesseract-ocr tesseract-ocr-fra`). Corrupt, encrypted or empty PDFs are reported as skipped; the other files are still indexed.
2. **Chunking**: LangChain `RecursiveCharacterTextSplitter`, 400 characters with 20% overlap by default. Each chunk keeps its file name and PDF page number (1-based, not the number printed on the page).
3. **Index** (`askfolio/index.py`): chunks, embeddings and a manifest with a SHA-256 per file are stored on disk. Unchanged files are never re-embedded. Two different files with the same name get their parent folder in the name (`2024/report.pdf`, `2025/report.pdf`).
4. **Retrieval** (`askfolio/retrieve.py`): dense search with a multilingual embedding model run locally (`intfloat/multilingual-e5-small`), BM25 keyword search with accent-insensitive Unicode tokens, and hybrid (both lists merged with reciprocal rank fusion, the default).
5. **Answer** (`askfolio/answer.py`): the LLM gets the top 5 numbered excerpts and returns structured output `{found, answer, citations}`. The prompt allows only facts from the excerpts and asks for an answer in the language of the question. If `found` is false or the output cannot be parsed, the answer is replaced by the not-found sentence and citations are dropped.
6. **Logging**: each query appends latency, tokens, cited and retrieved pages to `.askfolio/queries.jsonl`.

The defaults (hybrid retrieval, 400-character chunks) come from the retrieval sweep below. It was measured on two English financial reports; for other documents, run `askfolio sweep` on a few questions of your own.

## Models

- **Anthropic** (default): `claude-haiku-5-5`, needs `ANTHROPIC_API_KEY`. Change with `--model`.
- **Any OpenAI-compatible API** (OpenAI, OpenRouter, a local Ollama or vLLM): `pip install -e ".[openai]"`, then for example
  ```bash
  askfolio ask "..." --provider openai --base-url http://localhost:11434/v1 --model qwen2.5:7b
  ```
  `OPENAI_API_KEY` is required except for localhost endpoints. With a local model nothing leaves the machine; the model has to support structured output (tool calling) for answers to be parsed.

The embedding model always runs locally; it is downloaded from Hugging Face on first use.

## Evaluating on your documents

`eval` and `sweep` read a JSONL file, one question per line:

```json
{"id": "q1", "question": "What notice period does the contract set?", "answerable": true, "expect": ["3 months", "three months"], "source": "2025.pdf", "pages": [4]}
{"id": "q2", "question": "Who won the 2030 World Cup?", "answerable": false, "expect": [], "source": null, "pages": []}
```

Metrics: answer accuracy (an expected string appears in the answer), citation accuracy (a cited page contains an expected string), retrieval hit@k (a retrieved chunk is on a reference page), correct refusals and hallucinated answers on unanswerable questions. Results go to `.askfolio/eval/`.

## Evaluation

`make eval` downloads two Pernod Ricard reports (the FY25 integrated annual report and the FY26 results presentation, 192 pages in total), indexes them and runs [eval/questions.jsonl](eval/questions.jsonl): 34 questions, 28 with an expected value and reference page, 6 whose answer is not in the documents. The question set comes from [annual-report-rag](https://github.com/ksenia-ossi/annual-report-rag). Results are in [eval/results/](eval/results/).

Answers (hybrid retrieval, 400-character chunks, top 5 excerpts, `claude-haiku-5-5`):

| metric | result |
|---|---|
| correct answers (answerable questions) | 21/28 (75%) |
| citations on a page that contains the answer | 89% |
| retrieval hit@5 | 0.75 |
| correct refusals (unanswerable questions) | 6/6 |
| invented answers on unanswerable questions | 0 |
| latency per question, median / max | 0.94 s / 2.73 s |
| tokens per question, input / output (mean) | 1,497 / 107 |

Retrieval sweep, hit@5 on the 28 answerable questions (no LLM calls):

| chunk size | BM25 | dense | hybrid |
|---|---|---|---|
| 400 | 0.68 | 0.61 | **0.75** |
| 800 | 0.68 | 0.57 | 0.57 |
| 1600 | 0.71 | 0.71 | 0.68 |

Where it fails (7 of 28):

- **Layout** (2): on a key-figures page the amounts are printed before their region labels, so text extraction puts each number next to the wrong region and the model swaps Europe and Asia/Rest of the World.
- **Retrieval miss** (3): the right page is not in the top 5 (worldwide headcount, FY26 net sales, the two markets that weighed on FY26 sales), so the tool answers `Not found in the documents.` instead of guessing.
- **Wrong figure** (1): FY26 organic growth given as -3.6% instead of -3.9%, although the right page was retrieved; -3.6% does not appear in the report text.
- **Incomplete answer** (1): the excerpt was found, but the answer leaves out the 5% figure.

## Development

```bash
make install
make lint test
```

Tests use a fake embedder and a fake LLM: no network and no API key needed.

## Limitations

- Text is extracted without layout: tables, charts and multi-column pages lose their structure, and a figure can end up in a different chunk from its caption.
- OCR quality depends on the scan and on installing the right Tesseract language.
- With the default provider, questions and retrieved excerpts are sent to the Anthropic API; use a local model for confidential documents.
- The model can still misread a correct excerpt. The answer is a pointer to the cited page, not a replacement for reading it.
