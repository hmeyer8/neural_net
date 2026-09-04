"""BM25 — lexical retrieval, implemented by hand.

WHAT THIS IS
    A spec file. Every function below is a signature, a docstring, and an empty
    body. You fill the bodies from the formula, run `pytest agent/tests/test_bm25.py`,
    and it tells you what passed. Same protocol as `cv/reps.py`.

WHY BY HAND
    `rank_bm25` is one pip install away and in production I would probably use
    OpenSearch, which has BM25 built in. This is not production. The reason to write
    it is that "why does hybrid retrieval beat dense alone" has a real answer that
    lives inside the scoring function, and an answer recited from a blog post sounds
    exactly like an answer derived from the formula right up until the follow-up
    question. Roughly fifty lines buys the ability to handle the follow-up.

THE FORMULA

    For a query Q and document D:

        score(D, Q) = SUM over terms q in Q of:

                                    f(q, D) * (k1 + 1)
            IDF(q)  *  ------------------------------------------
                       f(q, D) + k1 * (1 - b + b * |D| / avgdl)

    where
        f(q, D)   term frequency: times q occurs in D
        |D|       length of D in tokens
        avgdl     mean document length across the corpus
        k1        term-frequency saturation, default 1.2
        b         length normalization strength, default 0.75

    and the IDF variant used here is the non-negative one:

                        N - n(q) + 0.5
        IDF(q) = ln( ------------------ + 1 )
                        n(q) + 0.5

        N         number of documents
        n(q)      number of documents containing q

    The trailing "+ 1" is not decoration. Without it, a term appearing in more than
    half the corpus gets a negative IDF, and a document is then *punished* for
    containing a query term. Ranking a document lower because it matched more of
    the query is indefensible, so the +1 variant is the one to implement.

THE TWO KNOBS, AND WHAT THEY ACTUALLY DO
    Predict each of these before you run the tests. Write the prediction down. The
    gap between the prediction and the test result is the lesson.

    k1 controls **saturation**. As k1 -> 0 the term-frequency factor collapses to 1
    and BM25 becomes binary: a document either has the term or doesn't, and the
    tenth occurrence counts for nothing. As k1 grows, term frequency matters more
    nearly linearly. The default 1.2 says "the second occurrence tells me a lot more
    than the twentieth" — which is the empirically right shape for prose.

    b controls **length normalization**. At b=0 length is ignored entirely and long
    documents win by having more room to match. At b=1 the frequency is fully
    divided by relative length. The default 0.75 is a compromise, and it is a
    compromise worth revisiting on this corpus specifically: RFC sections range from
    a two-line stub to forty pages of ABNF, which is a far wider length distribution
    than the news articles these defaults were tuned on. Sweep it in week 1 rather
    than accepting it.

WHAT NOT TO DO
    Do not read `agent/tests/test_bm25.py` for the answer. The exact-value test in
    there was computed by hand from the formula above; reading it to reverse out an
    implementation is the same as looking up the solution, and it converts a
    retrieval attempt into reading practice.
"""

from __future__ import annotations

import re
from collections.abc import Sequence

from ..config import settings
from .types import Hit

# Tokens: runs of letters and digits, with internal hyphens preserved so that
# "no-store", "max-age", and "content-length" survive as single terms. They are
# header field names in this corpus, not two words that happen to be adjacent,
# and splitting them costs precision on exactly the queries this system exists for.
_TOKEN_RE = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)*")


def tokenize(text: str) -> list[str]:
    """Lowercase `text` and split it into terms.

    Use `_TOKEN_RE` above. No stemming and no stopword removal: both are tunable
    choices that should be *measured* on the golden set before being adopted, and
    baking them into the tokenizer makes them impossible to ablate later.

    >>> tokenize("A cache MUST NOT store a no-store response.")
    ['a', 'cache', 'must', 'not', 'store', 'a', 'no-store', 'response']
    """
    raise NotImplementedError


class BM25Index:
    """An in-memory BM25 index over a fixed set of documents.

    Built once, queried many times. Fitting is O(total tokens); a query is
    O(sum over query terms of the length of that term's postings list), which is why
    the inverted index below is a dict from term to postings rather than a scan.

    Attributes you will need to populate in `fit_tokenized`:

        self.doc_ids      list[str]              position i <-> document i
        self.doc_len      list[int]              token count of document i
        self.avgdl        float                  mean of doc_len
        self.n_docs       int                    len(doc_ids)
        self.postings     dict[str, dict[int, int]]
                                                 term -> {doc index: term frequency}
        self.df           dict[str, int]         term -> document frequency n(q)

    `postings` and `df` are redundant with each other — df[t] == len(postings[t]) —
    and storing both is deliberate. IDF is computed once per query term and reading
    it from a dict beats taking len() of a postings list every time.
    """

    def __init__(self, k1: float | None = None, b: float | None = None) -> None:
        self.k1 = settings.retrieval.bm25_k1 if k1 is None else k1
        self.b = settings.retrieval.bm25_b if b is None else b
        self.doc_ids: list[str] = []
        self.doc_len: list[int] = []
        self.doc_text: list[str] = []
        self.avgdl: float = 0.0
        self.n_docs: int = 0
        self.postings: dict[str, dict[int, int]] = {}
        self.df: dict[str, int] = {}

    # -- building ---------------------------------------------------------

    def fit(self, documents: Sequence[tuple[str, str]]) -> BM25Index:
        """Tokenize and index `(doc_id, text)` pairs. Returns self, so it chains."""
        raise NotImplementedError

    def fit_tokenized(self, documents: Sequence[tuple[str, list[str]]]) -> BM25Index:
        """Index pre-tokenized `(doc_id, tokens)` pairs. Returns self.

        This is where the real work happens; `fit` should tokenize and delegate here.
        Keeping the tokenized entry point public is what lets the tests pin exact
        scores without the tokenizer in the way.

        An empty corpus is legal and must not divide by zero — `avgdl` of an empty
        index is 0.0 and every query returns nothing.
        """
        raise NotImplementedError

    # -- scoring ----------------------------------------------------------

    def idf(self, term: str) -> float:
        """Inverse document frequency for `term`, using the non-negative variant.

        A term that appears in no document has n(q) = 0. Decide what that should
        return and make it consistent with `score` — the formula gives a positive
        number for an unseen term, which is harmless only because f(q, D) is then
        always 0. Returning 0.0 explicitly is also defensible. Either way, the
        contract is that an out-of-vocabulary term contributes nothing to any
        document's score, and the tests check that.
        """
        raise NotImplementedError

    def score(self, query_terms: Sequence[str], doc_index: int) -> float:
        """BM25 score of one document against already-tokenized query terms.

        Repeated query terms count repeatedly — the outer SUM in the formula is over
        term *occurrences* in Q, not over the distinct set. Deduplicating here is a
        defensible variant, but it is a different scoring function, so if you choose
        it, choose it on purpose and write down why.
        """
        raise NotImplementedError

    def search(self, query: str, k: int | None = None) -> list[Hit]:
        """Top `k` documents for `query`, best first.

        Score only documents that contain at least one query term — walking the
        postings lists rather than the whole corpus is the entire point of an
        inverted index.

        Ties must break deterministically. Two chunks with identical scores that
        swap order between runs will produce a recall@5 that moves without any
        change to the system, and chasing that ghost is a bad afternoon. Break ties
        on `doc_id`.

        Each `Hit` gets `retriever="bm25"`, its 0-based `rank`, and its `text`.
        """
        raise NotImplementedError
