# Benchmarks

Raw output of `python cli.py eval` for the three configurations compared in the
README. Committed as evidence: the numbers quoted there are reproducible, not
asserted.

| File | Embedder | Reranker | Answerer |
|---|---|---|---|
| `a_lexical_extractive.json` | hashing (no downloads) | lexical | extractive (no API key) |
| `b_semantic_extractive.json` | `all-MiniLM-L6-v2` | cross-encoder | extractive (no API key) |
| `c_semantic_groq.json` | `all-MiniLM-L6-v2` | cross-encoder | Groq `openai/gpt-oss-120b` |

All three run the same 53-question golden set: 45 answerable from the sample
corpus, 8 deliberately out-of-corpus to measure refusal behaviour. Relevance
labels are verbatim phrases from the source PDFs, so the metrics survive changes
to chunk size, overlap or the parser — see `src/verirag/eval/dataset.py`.

Reproduce:

```bash
python -m scripts.make_sample_pdfs
python cli.py ingest data/raw
python cli.py --provider extractive eval --json benchmarks/a_lexical_extractive.json
```

Each report includes a `threshold_calibration` block reporting the score
distributions for in-corpus vs out-of-corpus questions. In configuration A the
`separation` is negative, meaning no single refusal threshold can separate the two
classes with a keyword embedder; in B and C it is positive. That finding is why
refusal thresholds are a property of the reranker rather than a global constant.
