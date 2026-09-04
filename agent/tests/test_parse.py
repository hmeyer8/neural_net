"""Tests for the RFC text and index parsers.

The text parser is heuristic — it reads a 50-year-old plain-text format whose
conventions drifted across decades — so these tests pin the specific failures
found by sweeping the full 9,828-document corpus, not just a happy path.

Tests needing the real corpus are marked `network` (the data has to be fetched
first) and skip cleanly when it isn't on disk, so CI stays offline.
"""

from __future__ import annotations

import pytest
from rfcagent.config import settings
from rfcagent.corpus.models import normative_keywords
from rfcagent.corpus.parse import estimate_tokens, parse_text

# --------------------------------------------------------------------------
# the sequence guard
#
# `_heading` is a shape test, and ordinary wrapped prose matches that shape.
# These are the two real cases found in the corpus.
# --------------------------------------------------------------------------


def test_year_like_numbers_are_not_sections():
    """RFC 2626 produced 550 phantom sections from lines reading "2000  found at...".

    A Y2K survey document is full of bare years at the start of a line. Without a
    sequence guard each one becomes a section, and the document's real structure
    is buried under garbage that then gets chunked, embedded, and cited.
    """
    raw = "\n".join([
        "1.  Introduction",
        "",
        "   Text of the introduction.",
        "",
        "2000  found at line 3182:",
        "",
        "   Some matched line.",
        "",
        "1900  found at line 8:",
        "",
        "   Another matched line.",
        "",
        "2.  Discussion",
        "",
        "   Real second section.",
    ])
    numbers = [s.number for s in parse_text(2626, raw)]
    assert "1" in numbers and "2" in numbers
    assert "2000" not in numbers
    assert "1900" not in numbers


def test_wrapped_prose_starting_with_a_number_is_not_a_section():
    """RFC 1035 wrapped a sentence onto a line beginning "25 (SMTP).  If this bit..."."""
    raw = "\n".join([
        "1.  Introduction",
        "",
        "   For example, if PROTOCOL=TCP (6), the 26th bit corresponds to TCP port",
        "25 (SMTP).  If this bit is set, a SMTP server should be listening on TCP",
        "port 25; if zero, SMTP service is not supported.",
        "",
        "2.  Discussion",
        "",
        "   Body.",
    ])
    numbers = [s.number for s in parse_text(1035, raw)]
    assert numbers.count("1") == 1
    assert "25" not in numbers


def test_sequential_sections_all_survive_the_guard():
    """The guard must not eat legitimate structure, including deep subsections."""
    raw = "\n".join([
        "1.  One", "", "   a", "",
        "2.  Two", "", "   b", "",
        "2.1.  Two One", "", "   c", "",
        "2.1.1.  Two One One", "", "   d", "",
        "3.  Three", "", "   e",
    ])
    numbers = [s.number for s in parse_text(1, raw)]
    assert numbers == ["1", "2", "2.1", "2.1.1", "3"]


def test_appendices_are_exempt_from_the_numeric_guard():
    """Appendices are lettered and legitimately follow the highest numbered section."""
    raw = "\n".join([
        "1.  Introduction", "", "   a", "",
        "Appendix A.  Collected ABNF", "", "   b", "",
        "Appendix B.  Examples", "", "   c",
    ])
    numbers = [s.number for s in parse_text(1, raw)]
    assert "A" in numbers and "B" in numbers


# --------------------------------------------------------------------------
# page furniture and tables of contents
# --------------------------------------------------------------------------


def test_page_furniture_is_stripped():
    raw = "\n".join([
        "1.  Introduction",
        "",
        "   Body before the page break.",
        "Fielding, et al.             Standards Track                    [Page 3]",
        "\f",
        "RFC 7234                    HTTP/1.1 Caching                   June 2014",
        "",
        "   Body after the page break.",
    ])
    body = parse_text(7234, raw)[0].text
    assert "[Page 3]" not in body
    assert "Standards Track" not in body
    assert "June 2014" not in body
    assert "Body before" in body and "Body after" in body


def test_table_of_contents_entries_are_rejected():
    """Dot leaders and trailing page numbers mean "table of contents", not "section"."""
    raw = "\n".join([
        "Table of Contents",
        "",
        "1. Introduction ....................................................4",
        "2. Caching .........................................................5",
        "",
        "1.  Introduction",
        "",
        "   The real section body.",
    ])
    sections = [s for s in parse_text(7234, raw) if s.number == "1"]
    assert len(sections) == 1
    assert "real section body" in sections[0].text


def test_document_with_no_headings_returns_front_matter_not_nothing():
    """Early RFCs genuinely have no numbered structure. That is a fact, not a crash."""
    raw = "\n".join([
        "Network Working Group                                   Steve Crocker",
        "Request for Comments: 1                                          UCLA",
        "",
        "                         Title:   Host Software",
    ])
    sections = parse_text(1, raw)
    assert len(sections) == 1
    assert sections[0].number == ""
    assert "Host Software" in sections[0].text


# --------------------------------------------------------------------------
# normative keywords — RFC 2119 / 8174
# --------------------------------------------------------------------------


def test_normative_keywords_are_case_sensitive():
    """RFC 8174 is explicit: requirement-level meaning applies only in all-capitals.

    Lowercasing here would manufacture obligations the document never made, which
    is the single worst failure available to a system that answers questions about
    what a specification requires.
    """
    assert normative_keywords("A cache MUST NOT store this.") == ("MUST NOT",)
    assert normative_keywords("A cache may store this.") == ()
    assert normative_keywords("It should work, but MUST be checked.") == ("MUST",)


def test_must_not_beats_must():
    """Longest-first matching, or every MUST NOT is silently reported as a MUST."""
    assert normative_keywords("MUST NOT") == ("MUST NOT",)
    assert normative_keywords("SHOULD NOT") == ("SHOULD NOT",)


def test_normative_keywords_are_deduplicated_in_order():
    text = "MUST do this. SHOULD do that. MUST do the other."
    assert normative_keywords(text) == ("MUST", "SHOULD")


def test_estimate_tokens_is_never_zero():
    assert estimate_tokens("") >= 1
    assert estimate_tokens("a" * 400) == 100


# --------------------------------------------------------------------------
# against the real corpus
# --------------------------------------------------------------------------

_HAVE_CORPUS = (settings.paths.text / "rfc7234.txt").exists()
needs_corpus = pytest.mark.skipif(_HAVE_CORPUS is False, reason="corpus not fetched")


@needs_corpus
@pytest.mark.network
def test_readme_example_target_parses():
    """RFC 7234 s5.2.1.5 is the citation the README's worked example produces.

    If this section stops parsing, the project's headline example is broken and
    the eval set's ground truth moves underneath it.
    """
    raw = (settings.paths.text / "rfc7234.txt").read_text(encoding="utf-8", errors="replace")
    section = next(s for s in parse_text(7234, raw) if s.number == "5.2.1.5")
    assert section.title == "no-store"
    assert "MUST NOT" in section.normative
    assert section.citation == "RFC7234 §5.2.1.5"


@needs_corpus
@pytest.mark.network
def test_large_specifications_keep_their_structure():
    """The NFS specs really do have ~1000 sections; the guard must not touch them."""
    raw = (settings.paths.text / "rfc8881.txt").read_text(encoding="utf-8", errors="replace")
    assert len(parse_text(8881, raw)) > 900


@needs_corpus
@pytest.mark.network
def test_rfc2626_phantom_sections_stay_fixed():
    """629 sections before the guard, ~78 after. A regression here is silent."""
    raw = (settings.paths.text / "rfc2626.txt").read_text(encoding="utf-8", errors="replace")
    sections = parse_text(2626, raw)
    assert len(sections) < 150
    assert not any(s.number in {"2000", "1900"} for s in sections)
