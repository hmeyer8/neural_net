# rfcagent

A grounded question-answering agent over the IETF RFC corpus. It answers questions about internet
standards, cites the exact section it used, refuses when the corpus doesn't support an answer, and
checks whether the document it found has been superseded.

**Started 2026-09-04. In progress.** The status section below distinguishes what is built and
measured from what is specified and from what is only planned. Nothing in this README claims a
number that isn't in `experiments/experiments.csv`.

---

## The problem this is actually about

Retrieval-augmented generation demos are easy because the demo question has an answer sitting in one
chunk. Normative corpora aren't like that. The failure that matters looks like this:

> **Q:** Is the `SHOULD` in RFC 2616 §14.9 still the current guidance for cache control?
>
> **A naive RAG system:** retrieves RFC 2616 §14.9, finds the SHOULD, quotes it, cites it.
> Confident, well-formatted, correctly quoted, and **wrong** — RFC 2616 was obsoleted in 2014.

Nothing about that answer looks wrong. The retrieval worked. The quote is accurate. The citation
points at real text. The system failed anyway, because **the hard problem in a normative corpus is
not finding the text — it is knowing whether the text you found is still in force.**

That question cannot be answered by similarity search. "Is this obsolete" is not semantically
encoded in the passage; it lives in the document's metadata, in a graph of `Obsoletes` /
`Obsoleted-By` / `Updates` / `Updated-By` edges. Answering it requires a second, structured lookup
that the retriever never triggers on its own.

**That is the whole reason this is an agent and not a pipeline.** Multi-step tool use isn't
decoration here; remove it and the system produces silent, confident, checkable errors.

```
"Is the SHOULD in RFC 2616 §14.9 still current guidance for cache-control?"

  1  search_corpus("cache-control no-store")      -> RFC2616 §14.9
  2  rfc_metadata(2616)                           -> Obsoleted-By: 7230…7235, status HISTORIC
  3  search_corpus(..., restrict_to=[7234])       -> RFC7234 §5.2.1.5, the live text
  4  answer citing RFC7234, flagging 2616 obsolete
```

## Why this corpus

I needed a corpus I could grade objectively rather than eyeball. RFCs give four things:

| | |
|---|---|
| **Normative language with fixed meaning** | RFC 2119/8174 define MUST, SHOULD, MAY. "Is this a MUST or a SHOULD" has one right answer, so a wrong answer is a gradeable failure rather than a matter of taste. Note the case-sensitivity: a lowercase "may" in prose is *not* a permission grant, and treating it as one manufactures obligations the document never made. |
| **A supersession graph** | ~9,700 documents with four kinds of directed edge and a status field. Structured data access, right next to semantic retrieval, in one system. |
| **Checkable ground truth** | RFC numbers, section numbers, and statuses are exact strings. Citation precision needs no LLM judge. |
| **Public domain, offline, laptop-scale** | Big enough that chunking and ranking choices move the numbers. Small enough to index in minutes on a laptop GPU, and still tractable on CPU. |

The transferable claim: normative text + a supersession graph + an obligation to cite is the same
shape as policy, regulation, and procedure. Solve the shape and the domain is a swap.

---

## Architecture

```
question
   │
   ├─ guardrails/input        validate, detect PII, reject or redact before anything is logged
   │
   ├─ agent/loop              hand-written; no framework
   │     │
   │     ├─ search_corpus     hybrid retrieval  → index/{bm25, dense, fuse, rerank}
   │     ├─ rfc_metadata      supersession graph → corpus/parse (structured, exact)
   │     ├─ fetch_section     full section text by citation
   │     └─ escalate          hand off to a human with the trace attached
   │
   ├─ guardrails/output       schema validation, citation verification, groundedness floor
   │
   └─ obs/trace               span tree: tokens, latency, cost, tool outcome, per step
```

**Providers are a seam, not a dependency.** `EchoProvider` is deterministic and offline and is what
CI runs; `OllamaProvider` is the local default; `AnthropicProvider` and `BedrockProvider` are there
for when a hosted model is wanted. The loop never knows which is behind it. That seam is why the
eval suite can gate a pull request without a network call or a cent of spend.

---

## Status

Honest state, updated as things land.

**Built and running**

- [x] Configuration with a `config_hash` that ties every reported number to the settings that produced it
- [x] Corpus fetch — metadata index and bulk text, cached and rate-limited. The RFC Editor retired its HTTP tarball, so this tries rsync (their recommended bulk method) and falls back to ~9,800 throttled HTTP requests, tolerating the handful of 1969-era RFCs that exist only as scans
- [x] Metadata parsing — the full supersession graph out of `rfc-index.xml`
- [x] Text parsing — page furniture stripped, TOC rejected, sections split with line spans
- [x] RFC 2119/8174 normative-keyword extraction, case-sensitive
- [x] **Corpus ingested and the parser validated against all of it** — 9,828 documents (1969–2026), 7 with no plain-text version. See below.

### Corpus and parser validation

The parser's docstring claimed *"a parser you have not tested against 1989 is a parser that works on 2014."* With the corpus on disk that got tested rather than asserted.

| | |
|---|---|
| Documents on disk | 9,828 of 9,835 indexed (7 exist only as scans) |
| Parse throughput | 9,828 documents in 18.2s (540 docs/s) |
| Sections per document | median 23, mean 30.5, max 998 (RFC 8881, NFSv4.1) |
| Documents with no numbered sections | 641 — **all of them pre-1995**, zero after |

Two real bugs, both found only by sweeping the whole corpus:

- **RFC 2626** produced **550 phantom sections** from lines reading `2000  found at line 3182:` — it's a Y2K survey full of bare years, and a heading test that only checks *shape* accepts every one of them.
- **RFC 1035** wrapped a sentence onto a line beginning `25 (SMTP).  If this bit is set…`, which parsed as section 25.

Both are fixed by treating section numbers as a **sequence** rather than a shape: a top-level number never jumps more than one past the highest already seen. RFC 2626 drops 629 → 78 sections; RFC 7234, 2616, 8446, 9110, 793 and the ~997-section NFS specs lose nothing.

The 641 unstructured documents are a genuine **limitation, not a bug** — early RFCs are memos using Roman numerals and ALL-CAPS headings, so there is no numbered structure to find. They cannot be chunked below document level, which will show up in retrieval and belongs in the limitations page rather than being quietly ignored.

**Specified, with a passing-or-failing grader, not yet implemented**

- [ ] `index/bm25.py` — 26 tests in `tests/test_bm25.py`, currently 26 failing. Written as a spec on purpose; see the module docstring for why by hand.
- [ ] `corpus/chunk.py` — two strategies to compare, not one to pick

**Planned — see [`../ROADMAP.md`](../ROADMAP.md) for the week each is due**

- [ ] Dense retrieval, reciprocal rank fusion, cross-encoder reranking
- [ ] Golden question set (120 questions, train/dev/test)
- [ ] Provider seam and the agent loop
- [ ] Eval harness: retrieval metrics, citation precision, groundedness, abstention quality
- [ ] LLM-as-judge **with judge-vs-human agreement measured**, because an uncalibrated judge is an opinion with a decimal point
- [ ] Tracing, FastAPI service, Dockerfile, CI eval gate
- [ ] Terraform for the AWS shape — written and validated, *not* applied

## Results

There are none yet, and there is no table here pretending otherwise. The first measured numbers are
the week-1 gate: recall@5 and MRR for BM25, dense, and hybrid on a held-out dev set, with the
question count and the seed. They land here and in `experiments/experiments.csv` at the same time.

What will be reported, so the shape is fixed before the numbers exist and can't be chosen to flatter
them:

| Measured | Why it's the right measurement |
|---|---|
| recall@5, MRR, nDCG@10 | Retrieval quality, separated from generation quality. If retrieval is the bottleneck, no amount of prompt work fixes it. |
| Citation precision / recall | Exact-match against known answer sections. No judge required. |
| Groundedness | Does every claim trace to a retrieved span? LLM-judged, and the judge is itself calibrated against my labels. |
| Abstention quality | On questions the corpus genuinely can't answer, does it refuse? A system that answers everything has no precision to trade. |
| p50/p95 latency, tokens, cost per query | An answer that takes 40 seconds and $0.30 is a different product than one that takes 2 seconds. |
| Obsolescence catch rate | The one this project exists for: of questions whose top hit is an obsoleted RFC, how often does the answer say so? |

---

## Running it

```bash
uv sync
uv run rfcagent status
```

```bash
uv run rfcagent fetch --index
```

```bash
uv run rfcagent show 2616
```

```bash
uv run pytest agent/tests -v
```

### Hardware

Encoders run on CUDA when a usable GPU is present and on CPU otherwise, resolved once in
`config.resolve_device` and passed down — a system that silently runs half on each is a system whose
latency numbers mean nothing. `rfcagent status` prints the device actually in use; `RFCAGENT_DEVICE=cpu`
forces the CPU path.

On this machine (RTX 3050 6GB) `bge-small-en-v1.5` encodes at **2,988 texts/s against 412 texts/s on
CPU — 7.3×**. That is a two-minute full-corpus index instead of a twenty-minute one, which is the
difference between running the chunking ablation twice and running it once and hoping.

The GPU changes latency, not quality. recall@5 is identical on either device, so the device is
excluded from `Settings.fingerprint()` and carried as its own ledger column instead: quality keyed by
config hash, latency keyed by config hash *and* device. And a seed alone is not reproducibility on
CUDA — cuDNN benchmarks algorithms at runtime and some reductions accumulate nondeterministically,
so `set_seed()` pins deterministic algorithms for any run whose number gets reported.

## Deliberately not doing

A scope that grows never closes — same discipline as `cv/call_box.md`.

No fine-tuning. No agent framework. No multi-agent orchestration. No hosted vector database. No chat
UI. The reasoning for each is in [`../ROADMAP.md`](../ROADMAP.md), and "I chose not to, here's the
tradeoff" is a better interview answer than a dependency I can't defend.
