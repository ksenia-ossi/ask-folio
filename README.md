# docrag

## Install

```bash
make install
export ANTHROPIC_API_KEY=...
```

## Usage

```bash
docrag ingest path/to/pdfs/
docrag ask "What does the document say?"
docrag chat
docrag serve
```

The persistent index defaults to `.docrag`; set `DOCRAG_INDEX` or pass `--index DIR` to change it.

## Evaluation

`docrag eval --questions FILE` and `docrag sweep --questions FILE` accept JSONL records:

```json
{"id":"q1","question":"What is the result?","answerable":true,"expect":["42"],"source":"report.pdf","pages":[3]}
```

## OpenAI-compatible providers

Install `docrag[openai]` and use `--provider openai`. For Ollama, use `--base-url http://localhost:11434/v1 --model MODEL`.
