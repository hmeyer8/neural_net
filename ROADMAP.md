# Roadmap

Two tracks have run in this repo. A third starts now and is the current focus.

| Track | Status | Why |
|---|---|---|
| Fundamentals (`micrograd/`, `makemore/`) | **Done** | The engine is understood. Marginal return on more went to zero — journal, 8/13. |
| Applied CV (`cv/`) | **Paused** | The drill files (`reps.py`, `call_box.md`) are live and still run daily. The project work is parked. `cv/README.md` says so plainly. |
| Applied LLM systems (`agent/`) | **Active** | Started 2026-09-04. The track described below. |

---

## The track: a grounded question-answering agent over the IETF RFC corpus

An agent that answers technical questions about internet standards, cites the exact section it
used, refuses when the corpus doesn't support an answer, and knows when the document it found has
been superseded.

### Why this corpus

I needed a corpus where answers could be graded objectively rather than eyeballed, and where
retrieval is genuinely hard rather than keyword lookup wearing a costume. RFCs give four things
almost nothing else does:

1. **Normative language with defined meaning.** RFC 2119 fixes what MUST, SHOULD, and MAY mean.
   "Is this a MUST or a SHOULD" has exactly one correct answer, so a wrong answer is a gradeable
   failure rather than a matter of taste.

2. **A supersession graph.** Every RFC carries `Obsoletes`, `Obsoleted-By`, `Updates`, and
   `Updated-By` links, plus a status (Internet Standard, Proposed Standard, Historic, …). This is
   the part I actually care about. **The hard problem in a normative corpus is not finding the
   text — it is knowing whether the text you found is still in force.** Cosine similarity cannot
   answer that. It's a graph query over structured metadata. So the agent needs a real tool
   alongside retrieval, and that is the difference between a RAG demo and an agent.

3. **Checkable ground truth.** RFC numbers, section numbers, and statuses are exact strings. A
   citation is either right or it isn't. No judge required for the part that matters most.

4. **Public domain, offline, ~9,700 documents.** Large enough that chunking and ranking choices
   move the numbers. Small enough to index on a laptop in minutes.

The transferable claim: normative text + a supersession graph + an obligation to cite is the same
shape as policy, regulation, and procedure. Solve the shape and the domain is a swap.

### Why an agent and not a retrieval pipeline

A single retrieve-then-generate pass cannot answer "is this still current?", because the answer
depends on metadata the retriever never sees. Multi-step isn't decoration here, it's load-bearing:

    "Is the SHOULD in RFC 2616 section 14.9 still current guidance for cache-control?"

      step 1  search_corpus("cache-control no-store")   -> 2616 s14.9 text
      step 2  rfc_metadata(2616)                        -> Obsoleted-By 7230..7235; status Historic
      step 3  search_corpus(..., restrict_to=[7234])    -> 7234 s5.2.1.5, the live text
      step 4  answer citing 7234, noting 2616 is obsolete

Skip step 2 and the system confidently cites a document obsoleted in 2014. That failure is silent,
which is what makes it the one worth building against.

---

## Rules carried over

The four working rules in the top-level README apply unchanged. Two are specific to this track.

**No number, no claim.** Every retrieval or ranking decision is justified by a measured delta
against a baseline on a held-out question set, logged in `experiments/experiments.csv`. "Reranking
helps" is not an engineering statement. "Reranking moved recall@5 from 0.71 to 0.84 on 120 held-out
questions at +380 ms p50" is.

**The eval set exists before the thing it evaluates.** Build the agent first and the eval gets
shaped, unconsciously, to fit what the agent already does. Golden set first, every time.

---

## Weeks

Each week has a **gate**. A week isn't done because the days elapsed; it's done when the gate passes.

### Week 1 — Corpus, retrieval baseline, first number

- Ingest the corpus; parse into documents with section structure and the metadata graph.
- Chunking: implement it, then *measure two strategies against each other* instead of picking one.
- **BM25 by hand.** ~50 lines. Not because a library would be wrong in production, but because the
  interview question "why does hybrid retrieval work" deserves an answer from the scoring function
  rather than a vibe about keywords.
- Golden question set: 120 questions with known answer sections, split train/dev/test.
- Dense retrieval with `bge-small-en-v1.5`, then reciprocal rank fusion over both. Runs on the
  local RTX 3050 when present and falls back to CPU otherwise — same numbers either way, and the
  device is recorded per run because it moves latency but not quality.

**Gate:** a table in `agent/README.md` giving recall@5 and MRR for BM25, dense, and hybrid on dev,
with question count and seed. Three rows in the ledger.

### Week 2 — The agent

- Provider seam: `EchoProvider` (deterministic, offline — this is what CI runs), `OllamaProvider`
  (local), `AnthropicProvider`, `BedrockProvider`. One interface; the loop never knows which is behind it.
- Tool contracts and four tools: `search_corpus`, `rfc_metadata`, `fetch_section`, `escalate`.
- The agent loop, by hand, no framework. Turn accounting, tool-result plumbing, parallel tool
  calls, stop conditions — all things I want to be able to explain rather than import.
- Structured output: every answer is a validated object with claims, citations, and a confidence.

**Gate:** the trace of a multi-step supersession question, end to end, in the README — answer, tool
calls in order, citations, total tokens.

### Week 3 — Evaluation and observability

- Offline eval: retrieval metrics, citation precision/recall, groundedness, abstention quality.
- LLM-as-judge with a written rubric, plus the step most people skip: **agreement between the judge
  and my own labels on a 50-item sample.** An uncalibrated judge is an opinion with a decimal point.
- Regression gate: eval runs in CI against `EchoProvider` with recorded fixtures, so a PR that
  breaks retrieval fails before merge.
- Tracing: every run emits a span tree with per-step tokens, latency, cost, and tool outcome.

**Gate:** `agent/docs/eval_report.md` with a baseline, the current system, a per-failure-mode
breakdown, and judge-vs-human agreement. CI green on a PR.

### Week 4 — Serving, infrastructure, write-up

- FastAPI service, Dockerfile, health and readiness endpoints, request-scoped trace IDs.
- Guardrails: input validation, PII detection on inbound text, output schema enforcement, and a
  hard refusal path when groundedness falls below threshold.
- Terraform for the AWS shape (Bedrock, S3, OpenSearch Serverless, Lambda, Step Functions),
  **written and `terraform validate`d, not applied.** I don't have an account funded for this, and
  saying so is better than implying a deployment I never made.
- `agent/docs/`: architecture, tool contracts, model card, and a limitations page that is honest.

**Gate:** `docker run` serves the agent. CI green. The results table holds real numbers. The
limitations page names at least five things this system does badly.

---

## What I am deliberately not doing

Same discipline as `cv/call_box.md`. A scope that grows never closes.

- **No fine-tuning.** Retrieval and context are where the wins are at this scale — and the ablation
  proves it instead of me assuming it.
- **No agent framework** (LangChain, LlamaIndex, CrewAI). Not because they're bad. Because the loop
  is ~150 lines, and writing it is the only way to earn an opinion about when a framework pays for
  itself. That opinion is the actual interview answer.
- **No multi-agent orchestration.** One agent with four tools, measured, beats a swarm that demos.
- **No vector database service.** A flat index over 9,700 documents is the correct call at this
  scale. Knowing where that stops being true is the point.
- **No chat UI.** The interface is an API and an eval harness. A chat window hides exactly the
  failures this is built to find.

---

## The parallel study track

Building and studying are different activities and I've stopped pretending otherwise.

- `agent/call_box.md` — the bounded set of LLM/RAG/agent concepts held cold, same three-tier format
  as the CV call box. Closed while building. Open on Sunday.
- `agent/reps.py` — the daily drill. Same protocol as `cv/reps.py`: ten minutes, closed book,
  predict before running, throw the answers away afterward.
- `docs/interview_prep.md` — questions I should be able to answer cold, answered from this build
  rather than from a blog post.
