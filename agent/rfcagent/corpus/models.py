"""Corpus data types.

Three levels, and the distinction between them is load-bearing for the whole system:

    RfcMeta   one document's *metadata*, including the supersession graph.
              This is what `rfc_metadata` serves. It is never embedded and never
              retrieved by similarity — it is looked up by number, exactly.

    Section   one numbered section of one document, with its text.
              This is what `fetch_section` serves, and it is the unit a citation
              points at.

    Chunk     a retrievable span, derived from sections by a chunking strategy.
              This is what goes in the indices. A chunk always knows which section
              it came from, because an answer that cannot name its section cannot
              be cited and therefore cannot be graded.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

# Normative keywords as fixed by RFC 2119 / RFC 8174. The agent is graded on
# reporting these correctly, so they are a closed set defined in one place.
NORMATIVE_KEYWORDS = (
    "MUST NOT",
    "SHALL NOT",
    "SHOULD NOT",
    "NOT RECOMMENDED",
    "MUST",
    "SHALL",
    "REQUIRED",
    "SHOULD",
    "RECOMMENDED",
    "MAY",
    "OPTIONAL",
)

# Longest-first so "MUST NOT" wins over "MUST".
_NORMATIVE_RE = re.compile(
    r"\b(" + "|".join(re.escape(k) for k in NORMATIVE_KEYWORDS) + r")\b"
)


def normative_keywords(text: str) -> tuple[str, ...]:
    """Every RFC 2119 keyword appearing in `text`, in order of first appearance.

    Case-sensitive on purpose. RFC 8174 is explicit that the requirement-level
    meaning applies only when the words appear in all capitals; a lowercase "may"
    in prose is not a permission grant. Lowercasing here would manufacture
    normative claims that the document does not make.
    """
    seen: list[str] = []
    for match in _NORMATIVE_RE.finditer(text):
        kw = match.group(1)
        if kw not in seen:
            seen.append(kw)
    return tuple(seen)


@dataclass(frozen=True)
class RfcMeta:
    """One RFC's metadata, straight from the RFC Editor index.

    The four graph edges are the reason this project is an agent. `obsoleted_by`
    in particular is the field that turns a confident wrong answer into a correct
    one, and no amount of embedding quality substitutes for reading it.
    """

    number: int
    title: str
    current_status: str
    publication_status: str
    authors: tuple[str, ...] = ()
    year: int | None = None
    month: str | None = None
    stream: str | None = None
    obsoletes: tuple[int, ...] = ()
    obsoleted_by: tuple[int, ...] = ()
    updates: tuple[int, ...] = ()
    updated_by: tuple[int, ...] = ()
    also: tuple[str, ...] = ()  # STD/BCP/FYI identifiers this RFC is part of
    keywords: tuple[str, ...] = ()
    abstract: str | None = None
    page_count: int | None = None

    @property
    def doc_id(self) -> str:
        return f"RFC{self.number}"

    @property
    def url(self) -> str:
        return f"https://www.rfc-editor.org/rfc/rfc{self.number}.txt"

    @property
    def is_obsolete(self) -> bool:
        """True when a later RFC has replaced this one outright.

        Distinct from `is_updated`: an update amends part of a document that is
        otherwise still in force, while obsoletion replaces it. Conflating the two
        is one of the failure modes the eval set deliberately probes.
        """
        return bool(self.obsoleted_by)

    @property
    def is_updated(self) -> bool:
        return bool(self.updated_by)

    @property
    def is_current(self) -> bool:
        return not self.is_obsolete and self.current_status.upper() != "HISTORIC"


@dataclass(frozen=True)
class Section:
    """One numbered section of one RFC."""

    rfc: int
    number: str  # "5.2.1.5"; "" for front matter before the first numbered section
    title: str
    text: str
    line_start: int
    line_end: int

    @property
    def citation(self) -> str:
        return f"RFC{self.rfc} §{self.number}" if self.number else f"RFC{self.rfc}"

    @property
    def normative(self) -> tuple[str, ...]:
        return normative_keywords(self.text)


@dataclass(frozen=True)
class Chunk:
    """A retrievable span. The unit that goes into BM25 and the dense index."""

    chunk_id: str  # "RFC7234:5.2.1.5:0" — stable across rebuilds at a fixed config
    rfc: int
    section: str
    section_title: str
    rfc_title: str
    text: str
    ordinal: int = 0  # position within its section, for windowed chunking

    @property
    def citation(self) -> str:
        return f"RFC{self.rfc} §{self.section}" if self.section else f"RFC{self.rfc}"

    def for_embedding(self) -> str:
        """The text actually embedded.

        The section heading and document title are prepended because a chunk from
        the middle of a long section otherwise carries no signal about what it is
        about — the subject sits in the heading, which is three pages up. This is
        cheap context injection and it is worth measuring rather than assuming;
        week 1 runs it both ways.
        """
        head = f"{self.rfc_title} — §{self.section} {self.section_title}".strip(" —")
        return f"{head}\n\n{self.text}" if head else self.text


@dataclass
class Document:
    """An RFC: its metadata plus its parsed sections."""

    meta: RfcMeta
    sections: list[Section] = field(default_factory=list)

    @property
    def number(self) -> int:
        return self.meta.number

    def section(self, number: str) -> Section | None:
        for s in self.sections:
            if s.number == number:
                return s
        return None
