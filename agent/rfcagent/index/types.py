"""Shared retrieval types.

Every retriever in this package — BM25, dense, fused, reranked — returns the same
`list[Hit]`. That uniformity is what makes fusion and evaluation possible without
special-casing, and it is why `Hit` carries `retriever` and `rank`: after fusion you
still need to know where a result came from and where it sat in its own list.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Hit:
    """One retrieved chunk with the score that put it there.

    `score` is only comparable *within* one retriever. BM25 scores are unbounded
    sums over query terms; cosine similarities live in [-1, 1]; reranker logits are
    something else again. Comparing them directly is the single most common way to
    build a broken hybrid retriever, and it is exactly the mistake reciprocal rank
    fusion exists to avoid — RRF throws the scores away and uses only the ranks.
    """

    doc_id: str
    score: float
    rank: int
    retriever: str = ""
    text: str = ""
    meta: dict = field(default_factory=dict)

    def replace(self, **kwargs) -> Hit:
        from dataclasses import replace as _replace

        return _replace(self, **kwargs)
