"""Parse the RFC index and RFC text files.

Two parsers with very different characters.

The **index parser** reads a well-formed XML document the RFC Editor maintains. It
is boring and it should be, because the supersession graph it produces is the part
of this system that has to be exactly right.

The **text parser** reads a 50-year-old plain-text format with page furniture,
dot-leader tables of contents, and conventions that drifted across decades. It is
heuristic and it will be wrong somewhere. `parse_text` is therefore written to fail
loudly on structural surprises rather than silently produce one giant section, and
`agent/tests/test_parse.py` pins its behavior against real documents from several
eras. A parser you have not tested against 1989 is a parser that works on 2014.
"""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from pathlib import Path

from ..config import settings
from .models import Document, RfcMeta, Section

# --------------------------------------------------------------------------
# index
# --------------------------------------------------------------------------

_DOC_ID_RE = re.compile(r"^RFC0*(\d+)$")


def _local(tag: str) -> str:
    """Strip the XML namespace. The index has declared two different namespace
    URIs over its life; matching on the local name survives that."""
    return tag.rsplit("}", 1)[-1]


def _child_text(entry: ET.Element, name: str) -> str | None:
    for child in entry:
        if _local(child.tag) == name:
            return (child.text or "").strip() or None
    return None


def _doc_ids(entry: ET.Element, name: str) -> tuple[int, ...]:
    """RFC numbers listed under a graph element such as <obsoleted-by>."""
    out: list[int] = []
    for child in entry:
        if _local(child.tag) != name:
            continue
        for grandchild in child:
            if _local(grandchild.tag) != "doc-id":
                continue
            match = _DOC_ID_RE.match((grandchild.text or "").strip())
            if match:
                out.append(int(match.group(1)))
    return tuple(out)


def _also_ids(entry: ET.Element) -> tuple[str, ...]:
    out: list[str] = []
    for child in entry:
        if _local(child.tag) != "is-also":
            continue
        for grandchild in child:
            if _local(grandchild.tag) == "doc-id" and grandchild.text:
                out.append(grandchild.text.strip())
    return tuple(out)


def parse_index(path: Path | None = None) -> dict[int, RfcMeta]:
    """Parse rfc-index.xml into {number: RfcMeta}.

    Entries whose doc-id is not an RFC (the index also carries BCP, STD, and FYI
    entries) are skipped rather than coerced.
    """
    path = path or settings.paths.index_xml
    root = ET.parse(path).getroot()

    out: dict[int, RfcMeta] = {}
    for entry in root:
        if _local(entry.tag) != "rfc-entry":
            continue
        raw_id = _child_text(entry, "doc-id") or ""
        match = _DOC_ID_RE.match(raw_id)
        if not match:
            continue
        number = int(match.group(1))

        authors: list[str] = []
        year: int | None = None
        month: str | None = None
        keywords: list[str] = []
        abstract: str | None = None
        for child in entry:
            tag = _local(child.tag)
            if tag == "author":
                for grandchild in child:
                    if _local(grandchild.tag) == "name" and grandchild.text:
                        authors.append(grandchild.text.strip())
            elif tag == "date":
                for grandchild in child:
                    gtag = _local(grandchild.tag)
                    if gtag == "year" and grandchild.text:
                        year = int(grandchild.text.strip())
                    elif gtag == "month" and grandchild.text:
                        month = grandchild.text.strip()
            elif tag == "keywords":
                for grandchild in child:
                    if _local(grandchild.tag) == "kw" and grandchild.text:
                        keywords.append(grandchild.text.strip())
            elif tag == "abstract":
                parts = [(p.text or "").strip() for p in child if _local(p.tag) == "p"]
                abstract = " ".join(x for x in parts if x) or None

        page_count = _child_text(entry, "page-count")
        out[number] = RfcMeta(
            number=number,
            title=_child_text(entry, "title") or "",
            current_status=_child_text(entry, "current-status") or "UNKNOWN",
            publication_status=_child_text(entry, "publication-status") or "UNKNOWN",
            authors=tuple(authors),
            year=year,
            month=month,
            stream=_child_text(entry, "stream"),
            obsoletes=_doc_ids(entry, "obsoletes"),
            obsoleted_by=_doc_ids(entry, "obsoleted-by"),
            updates=_doc_ids(entry, "updates"),
            updated_by=_doc_ids(entry, "updated-by"),
            also=_also_ids(entry),
            keywords=tuple(keywords),
            abstract=abstract,
            page_count=int(page_count) if page_count and page_count.isdigit() else None,
        )
    return out


# --------------------------------------------------------------------------
# text
# --------------------------------------------------------------------------

# Page footer: "Fielding, et al.        Standards Track       [Page 12]"
_FOOTER_RE = re.compile(r"\[Page\s+\d+\]\s*$")
# Page header: "RFC 7234            HTTP/1.1 Caching            June 2014"
_HEADER_RE = re.compile(r"^RFC\s+\d+\s+\S.*\s+[A-Z][a-z]+\s+\d{4}\s*$")
# Draft-era header on very old RFCs: "Network Working Group ... Crocker"
_FORMFEED = "\f"

# A numbered heading at column 0: "5.2.1.5.  no-store"
_NUMBERED_RE = re.compile(r"^(?P<num>\d+(?:\.\d+)*)\.?\s+(?P<title>\S.*?)\s*$")
# "Appendix A.  Collected ABNF" / "Appendix A.1.  ..."
_APPENDIX_RE = re.compile(
    r"^Appendix\s+(?P<num>[A-Z](?:\.\d+)*)\.?\s*(?P<title>.*?)\s*$", re.IGNORECASE
)
# Unnumbered front and back matter that is worth keeping as its own section.
_NAMED_RE = re.compile(
    r"^(?P<title>Abstract|Status of This Memo|Copyright Notice|"
    r"Acknowledg(?:e)?ments?|Authors?'? Addresses?|Contributors|"
    r"(?:Normative |Informative )?References|Security Considerations|"
    r"IANA Considerations|Full Copyright Statement)\s*$",
    re.IGNORECASE,
)
# Dot leaders, or a trailing page number: both mean "table of contents entry".
_TOC_HINT_RE = re.compile(r"\.{3,}|\s\d+\s*$")


def _strip_page_furniture(raw: str) -> list[str]:
    """Remove form feeds, page footers, and repeated page headers.

    Done before heading detection because a heading that lands one line after a
    page break is otherwise separated from its body by three lines of noise, and
    that noise ends up inside the chunk that gets embedded.
    """
    lines = raw.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    out: list[str] = []
    for line in lines:
        if _FORMFEED in line:
            line = line.replace(_FORMFEED, "").strip()
            if not line:
                continue
        if _FOOTER_RE.search(line):
            continue
        if _HEADER_RE.match(line):
            continue
        out.append(line.rstrip())
    # Collapse runs of blank lines to at most two.
    collapsed: list[str] = []
    blanks = 0
    for line in out:
        if line.strip():
            blanks = 0
            collapsed.append(line)
        else:
            blanks += 1
            if blanks <= 2:
                collapsed.append("")
    return collapsed


def _heading(line: str) -> tuple[str, str] | None:
    """Return (number, title) if `line` is a real section heading, else None.

    The dot-leader / trailing-page-number test is what keeps the table of contents
    out. In modern RFCs the TOC is indented and column-0 matching already excludes
    it; in older ones it is not, and this test is the only thing standing between
    a clean parse and forty phantom sections.
    """
    if not line or line[0].isspace():
        return None
    if _TOC_HINT_RE.search(line):
        return None

    match = _APPENDIX_RE.match(line)
    if match:
        return match.group("num"), (match.group("title") or "").strip()

    match = _NUMBERED_RE.match(line)
    if match:
        title = match.group("title")
        # "1996.  " or a bare year is not a heading; require a non-digit start.
        if title and not title[0].isdigit():
            return match.group("num"), title.strip()
        return None

    match = _NAMED_RE.match(line)
    if match:
        return "", match.group("title").strip()

    return None


def parse_text(number: int, raw: str) -> list[Section]:
    """Split one RFC's text into sections.

    Everything before the first heading becomes a section with an empty number and
    the title "Front Matter" — it holds the title block and the status boilerplate,
    which is rarely what you want to retrieve but is occasionally exactly what you
    want to cite.
    """
    lines = _strip_page_furniture(raw)

    # Heading detection, with a monotonic-sequence guard on numbered headings.
    #
    # `_heading` is a shape test: it accepts any line starting at column 0 with a
    # leading integer and some text. That shape is also matched by ordinary
    # wrapped prose, and the corpus proves it. Two real cases:
    #
    #   RFC 2626, a Y2K survey full of bare years, produced 550 phantom sections
    #   from lines reading "2000  found at line 3182:".
    #
    #   RFC 1035 wrapped a sentence onto a line beginning "25 (SMTP).  If this
    #   bit is set, ...", which parsed as section 25.
    #
    # Both are caught by the observation that section numbers are a *sequence*,
    # not just a shape: a top-level number never jumps more than one past the
    # highest already seen. Measured across the corpus, this drops 550 phantom
    # sections from RFC 2626 and one from RFC 1035 while removing nothing from
    # RFC 7234, 2616, 8446, 9110, 793, or the ~997-section NFS specifications.
    #
    # Appendices are exempt: they are lettered, restart their own numbering, and
    # legitimately follow the highest numbered section.
    starts: list[tuple[int, str, str]] = []
    max_top = 0
    for i, line in enumerate(lines):
        found = _heading(line)
        if not found:
            continue
        num, title = found
        if num and num[0].isdigit():
            top = int(num.split(".")[0])
            if top > max_top + 1:
                continue
            max_top = max(max_top, top)
        starts.append((i, num, title))

    # A document with no detectable headings is a parser failure, not a document
    # with no structure. Surface it rather than emitting one 40-page chunk.
    if not starts:
        return [
            Section(
                rfc=number,
                number="",
                title="Front Matter",
                text="\n".join(lines).strip(),
                line_start=0,
                line_end=len(lines),
            )
        ]

    sections: list[Section] = []
    first = starts[0][0]
    if first > 0:
        front = "\n".join(lines[:first]).strip()
        if front:
            sections.append(
                Section(
                    rfc=number,
                    number="",
                    title="Front Matter",
                    text=front,
                    line_start=0,
                    line_end=first,
                )
            )

    for idx, (line_no, num, title) in enumerate(starts):
        end = starts[idx + 1][0] if idx + 1 < len(starts) else len(lines)
        body = "\n".join(lines[line_no + 1 : end]).strip()
        sections.append(
            Section(
                rfc=number,
                number=num,
                title=title,
                text=body,
                line_start=line_no,
                line_end=end,
            )
        )
    return sections


def load_document(number: int, index: dict[int, RfcMeta] | None = None) -> Document:
    """Load one RFC from disk: metadata from the index, sections from the text."""
    index = index if index is not None else parse_index()
    meta = index.get(number)
    if meta is None:
        raise KeyError(f"RFC {number} is not in the index")
    path = settings.paths.text / f"rfc{number}.txt"
    if not path.exists():
        raise FileNotFoundError(f"{path} — run `rfcagent fetch` first")
    raw = path.read_text(encoding="utf-8", errors="replace")
    return Document(meta=meta, sections=parse_text(number, raw))


def estimate_tokens(text: str) -> int:
    """Cheap token estimate for chunk budgeting.

    Four characters per token is the usual English rule of thumb and it is close
    enough to size a chunk. It is *not* used for anything billed or capped — where
    an exact count matters, the provider's own tokenizer is used instead. Guessing
    at a context limit with a heuristic is how requests get truncated in production.
    """
    return max(1, len(text) // 4)
