"""Grader for `rfcagent.index.bm25`.

Run:  uv run pytest agent/tests/test_bm25.py -v

These tests check *behavior*, not implementation. Where a number is asserted
exactly, it was computed by hand from the formula in the module docstring — the
derivation is in the comment above the assertion so the test is checkable rather
than magic.

Reading this file to reverse out an implementation defeats the drill. The failure
messages are meant to be enough.
"""

from __future__ import annotations

import math

import pytest

from rfcagent.index.bm25 import BM25Index, tokenize

# --------------------------------------------------------------------------
# the worked example
#
# Three documents, pre-tokenized so the tokenizer is not in the way:
#
#   d0  cache control no store     len 4
#   d1  cache cache control        len 3
#   d2  proxy server               len 2
#
#   N = 3,  avgdl = (4 + 3 + 2) / 3 = 3.0
#
# Query "cache" appears in d0 and d1, so n(q) = 2 and
#
#   IDF = ln((3 - 2 + 0.5) / (2 + 0.5) + 1) = ln(1.6) = 0.470003629245736
#
# d0:  f = 1, |D| = 4
#      denom = 1 + 1.2 * (1 - 0.75 + 0.75 * 4/3) = 1 + 1.2 * 1.25 = 2.5
#      score = 0.470003629... * (1 * 2.2) / 2.5 = 0.413603193736247
#
# d1:  f = 2, |D| = 3
#      denom = 2 + 1.2 * (1 - 0.75 + 0.75 * 3/3) = 2 + 1.2 * 1.0 = 3.2
#      score = 0.470003629... * (2 * 2.2) / 3.2 = 0.646254990212887
#
# d1 outranks d0 despite both being "about" caching: twice the term frequency in a
# shorter document. That is the whole of BM25 in one comparison.
# --------------------------------------------------------------------------

WORKED = [
    ("d0", ["cache", "control", "no", "store"]),
    ("d1", ["cache", "cache", "control"]),
    ("d2", ["proxy", "server"]),
]

IDF_CACHE = math.log(1.6)
SCORE_D0 = IDF_CACHE * 0.88
SCORE_D1 = IDF_CACHE * 1.375


@pytest.fixture
def worked() -> BM25Index:
    return BM25Index(k1=1.2, b=0.75).fit_tokenized(WORKED)


# --------------------------------------------------------------------------
# tokenizer
# --------------------------------------------------------------------------


def test_tokenize_lowercases_and_splits():
    assert tokenize("A cache MUST NOT store") == ["a", "cache", "must", "not", "store"]


def test_tokenize_keeps_internal_hyphens():
    """`no-store` and `max-age` are header field names, not adjacent words.

    Splitting them turns a precise query into a fuzzy one and costs precision on
    exactly the questions this corpus is built around.
    """
    assert tokenize("Cache-Control: no-store, max-age=0") == [
        "cache-control",
        "no-store",
        "max-age",
        "0",
    ]


def test_tokenize_drops_punctuation_and_keeps_digits():
    assert tokenize("See RFC 7234, section 5.2.1.5.") == [
        "see",
        "rfc",
        "7234",
        "section",
        "5",
        "2",
        "1",
        "5",
    ]


def test_tokenize_empty():
    assert tokenize("") == []
    assert tokenize("   ...  ") == []


# --------------------------------------------------------------------------
# fitting
# --------------------------------------------------------------------------


def test_fit_records_corpus_statistics(worked: BM25Index):
    assert worked.n_docs == 3
    assert worked.doc_ids == ["d0", "d1", "d2"]
    assert worked.doc_len == [4, 3, 2]
    assert worked.avgdl == pytest.approx(3.0)


def test_fit_builds_document_frequencies(worked: BM25Index):
    assert worked.df["cache"] == 2
    assert worked.df["control"] == 2
    assert worked.df["proxy"] == 1
    assert "missing" not in worked.df


def test_fit_builds_postings_with_term_frequencies(worked: BM25Index):
    """d1 contains "cache" twice; the postings list has to say so."""
    assert worked.postings["cache"] == {0: 1, 1: 2}
    assert worked.postings["proxy"] == {2: 1}


def test_fit_from_raw_text_matches_fit_tokenized():
    raw = BM25Index().fit([("a", "cache control no store"), ("b", "cache cache control")])
    tok = BM25Index().fit_tokenized(
        [("a", ["cache", "control", "no", "store"]), ("b", ["cache", "cache", "control"])]
    )
    assert raw.doc_len == tok.doc_len
    assert raw.df == tok.df
    assert raw.postings == tok.postings


def test_empty_corpus_does_not_divide_by_zero():
    """avgdl is a mean, and a mean over nothing is the classic silent crash."""
    index = BM25Index().fit_tokenized([])
    assert index.n_docs == 0
    assert index.avgdl == 0.0
    assert index.search("cache") == []


# --------------------------------------------------------------------------
# idf
# --------------------------------------------------------------------------


def test_idf_matches_hand_computation(worked: BM25Index):
    assert worked.idf("cache") == pytest.approx(IDF_CACHE, abs=1e-12)


def test_idf_is_never_negative(worked: BM25Index):
    """A term in every document must not make matching it a penalty.

    This is what the "+ 1" inside the log buys. Without it a term with n(q) > N/2
    scores negative and a document is ranked *lower* for containing a query term.
    """
    index = BM25Index().fit_tokenized(
        [("a", ["http"]), ("b", ["http"]), ("c", ["http"])]
    )
    assert index.idf("http") >= 0.0


def test_idf_is_higher_for_rarer_terms(worked: BM25Index):
    assert worked.idf("proxy") > worked.idf("cache")


# --------------------------------------------------------------------------
# scoring
# --------------------------------------------------------------------------


def test_score_matches_hand_computation(worked: BM25Index):
    assert worked.score(["cache"], 0) == pytest.approx(SCORE_D0, abs=1e-12)
    assert worked.score(["cache"], 1) == pytest.approx(SCORE_D1, abs=1e-12)


def test_shorter_document_with_more_hits_wins(worked: BM25Index):
    assert worked.score(["cache"], 1) > worked.score(["cache"], 0)


def test_document_without_the_term_scores_zero(worked: BM25Index):
    assert worked.score(["cache"], 2) == pytest.approx(0.0)


def test_out_of_vocabulary_term_contributes_nothing(worked: BM25Index):
    """An unseen term must not shift any score, whatever idf() returns for it."""
    base = worked.score(["cache"], 0)
    assert worked.score(["cache", "zzzznotaword"], 0) == pytest.approx(base, abs=1e-12)


def test_repeated_query_terms_count_repeatedly(worked: BM25Index):
    single = worked.score(["cache"], 0)
    assert worked.score(["cache", "cache"], 0) == pytest.approx(2 * single, abs=1e-12)


def test_multi_term_query_sums_over_terms(worked: BM25Index):
    combined = worked.score(["cache", "control"], 0)
    parts = worked.score(["cache"], 0) + worked.score(["control"], 0)
    assert combined == pytest.approx(parts, abs=1e-12)


# --------------------------------------------------------------------------
# the two knobs
# --------------------------------------------------------------------------

_EQUAL_LENGTH = [
    ("one", ["cache", "x", "y", "z"]),  # tf = 1
    ("two", ["cache", "cache", "y", "z"]),  # tf = 2
]


def test_k1_controls_saturation():
    """Small k1 flattens term frequency toward binary; large k1 lets it matter.

    Predict the direction before reading the assertion. Most people get this one
    backwards the first time.
    """
    flat = BM25Index(k1=0.01, b=0.75).fit_tokenized(_EQUAL_LENGTH)
    steep = BM25Index(k1=10.0, b=0.75).fit_tokenized(_EQUAL_LENGTH)

    flat_ratio = flat.score(["cache"], 1) / flat.score(["cache"], 0)
    steep_ratio = steep.score(["cache"], 1) / steep.score(["cache"], 0)

    assert flat_ratio == pytest.approx(1.0, abs=0.02), (
        "at k1 -> 0 the second occurrence should add almost nothing"
    )
    assert steep_ratio > flat_ratio


_UNEQUAL_LENGTH = [
    ("short", ["cache", "x"]),
    ("long", ["cache"] + ["x"] * 20),
]


def test_b_controls_length_normalization():
    """b=0 ignores length entirely; b=1 divides frequency fully by relative length."""
    off = BM25Index(k1=1.2, b=0.0).fit_tokenized(_UNEQUAL_LENGTH)
    full = BM25Index(k1=1.2, b=1.0).fit_tokenized(_UNEQUAL_LENGTH)

    assert off.score(["cache"], 0) == pytest.approx(off.score(["cache"], 1)), (
        "at b=0, document length must not affect the score at all"
    )
    assert full.score(["cache"], 0) > full.score(["cache"], 1)


# --------------------------------------------------------------------------
# search
# --------------------------------------------------------------------------


def test_search_returns_hits_best_first(worked: BM25Index):
    hits = worked.search("cache", k=3)
    assert [h.doc_id for h in hits] == ["d1", "d0"]
    assert hits[0].score > hits[1].score


def test_search_labels_hits(worked: BM25Index):
    hit = worked.search("cache", k=1)[0]
    assert hit.retriever == "bm25"
    assert hit.rank == 0


def test_search_respects_k(worked: BM25Index):
    assert len(worked.search("cache control", k=1)) == 1


def test_search_skips_zero_scoring_documents(worked: BM25Index):
    """d2 shares no term with the query and must not appear at all."""
    assert "d2" not in {h.doc_id for h in worked.search("cache control", k=10)}


def test_search_on_unmatched_query_returns_empty(worked: BM25Index):
    assert worked.search("zzzznotaword") == []


def test_search_breaks_ties_deterministically():
    """Identical documents must come back in a stable order, every run.

    Without this, recall@5 drifts between runs with no change to the system, and
    the resulting ghost regression costs an afternoon to chase.
    """
    index = BM25Index().fit_tokenized(
        [("b", ["cache"]), ("a", ["cache"]), ("c", ["cache"])]
    )
    first = [h.doc_id for h in index.search("cache", k=3)]
    assert first == ["a", "b", "c"]
    assert first == [h.doc_id for h in index.search("cache", k=3)]
