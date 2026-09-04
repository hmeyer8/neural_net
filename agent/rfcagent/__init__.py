"""rfcagent — a grounded question-answering agent over the IETF RFC corpus.

The system answers questions about internet standards, cites the section it used,
refuses when the corpus does not support an answer, and checks whether the document
it found has been superseded.

Layout:

    corpus/       fetch, parse, chunk. Produces the units that get indexed.
    index/        BM25, dense embeddings, fusion, reranking. Produces ranked hits.
    providers/    the seam between the agent loop and whatever model is behind it.
    tools/        the four tools the agent can call, and their contracts.
    guardrails/   input validation, PII detection, output schema, refusal path.
    obs/          tracing: spans, tokens, latency, cost.
    serve/        the FastAPI service.

See ROADMAP.md for what is built, what is measured, and what is still planned.
"""

__version__ = "0.1.0"
