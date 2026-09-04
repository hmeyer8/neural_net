"""Fetch the RFC corpus from the RFC Editor.

**Correction, 2026-09-04:** the original version of this module pointed `fetch_bulk`
at `https://www.rfc-editor.org/in-notes/tar/RFC-all.txt.tar.gz`. That endpoint is
gone — it 404s. The RFC Editor retired the HTTP tarball and now serves bulk mirrors
over **rsync only** (module `rfcs-text-only`; see
https://www.rfc-editor.org/series/rfc-download/). This machine doesn't have `rsync`
installed, so `fetch_bulk` below tries rsync first and falls back to fetching every
document over plain HTTP, one request at a time, politely rate-limited.

Two paths:

    fetch_bulk()    the whole corpus. rsync if available (fast, one connection,
                     the RFC Editor's own recommended method); otherwise ~9,700
                     individual HTTP requests at a floor of one every 0.2s, which
                     takes on the order of half an hour and is the honest cost of
                     not having rsync.

    fetch_text(n)   one document. Use this for a handful — a test fixture, a
                     document the index gained since the last bulk pull.

Everything is cached on disk and nothing re-downloads unless `force=True`. The
corpus is gitignored: it is public domain and cheap to refetch, and committing
hundreds of megabytes of text to a portfolio repo helps nobody.
"""

from __future__ import annotations

import shutil
import subprocess
import time
from collections.abc import Iterable, Iterator
from pathlib import Path

import httpx

from ..config import settings

INDEX_URL = "https://www.rfc-editor.org/rfc-index.xml"
TEXT_URL = "https://www.rfc-editor.org/rfc/rfc{n}.txt"
RSYNC_HOST = "rsync.rfc-editor.org"
RSYNC_MODULE = "rfcs-text-only"

# Identify the client honestly. An anonymous scraper is indistinguishable from a
# hostile one, and operators are right to treat it that way.
USER_AGENT = (
    "rfcagent/0.1 (portfolio project; https://github.com/hmeyer8/neural_net) "
    "python-httpx"
)

# Politeness floor for per-document HTTP fetches — the fallback path when rsync
# isn't available. This is what makes ~9,800 sequential requests defensible rather
# than a scrape.
MIN_DELAY_S = 0.2
# Above this many documents via fetch_many specifically, point the caller at
# fetch_bulk instead. fetch_bulk itself has no such cap — it *is* the bulk path.
BATCH_WARN_THRESHOLD = 50

# Transient-failure retry policy for the bulk path. A run that takes half an hour
# will meet a dropped connection eventually; treating that as fatal throws away
# every document fetched so far.
MAX_ATTEMPTS = 5
BACKOFF_BASE_S = 1.0
BACKOFF_CAP_S = 30.0

_SHARED_CLIENT: httpx.Client | None = None


def _client(timeout: float = 60.0) -> httpx.Client:
    """A pooled, keep-alive client, created once and reused.

    The first version of this module built a fresh `httpx.Client` inside every
    call to `fetch_text`, which meant a new TCP connection and TLS handshake per
    document — 9,835 of them for one bulk run. That is wasteful on both ends, and
    from the server's side a client that reconnects for every single request is
    close to indistinguishable from something hostile. It is very likely what got
    this client's connection dropped partway through the first full run.

    One pooled client with keep-alive is both faster and better behaved.
    """
    global _SHARED_CLIENT
    if _SHARED_CLIENT is None or _SHARED_CLIENT.is_closed:
        _SHARED_CLIENT = httpx.Client(
            headers={"User-Agent": USER_AGENT},
            timeout=timeout,
            follow_redirects=True,
            limits=httpx.Limits(max_connections=4, max_keepalive_connections=2),
        )
    return _SHARED_CLIENT


def _get_with_retry(url: str, *, attempts: int = MAX_ATTEMPTS) -> httpx.Response:
    """GET `url`, retrying transient failures with exponential backoff.

    Retries dropped connections, timeouts, 429, and 5xx. Does **not** retry other
    4xx — a 404 means the document genuinely isn't there, and retrying it four
    more times is just noise against a volunteer-run server. `fetch_many` depends
    on that 404 arriving promptly so it can record the gap and move on.

    Honors `Retry-After` when the server sends one: if an operator tells you how
    long to wait, guessing something shorter is rude and usually counterproductive.
    """
    delay = BACKOFF_BASE_S
    last_exc: Exception | None = None

    for attempt in range(1, attempts + 1):
        try:
            response = _client().get(url)
            if response.status_code == 429 or response.status_code >= 500:
                retry_after = response.headers.get("retry-after")
                wait = float(retry_after) if retry_after and retry_after.isdigit() else delay
                if attempt == attempts:
                    response.raise_for_status()
                time.sleep(min(wait, BACKOFF_CAP_S))
                delay = min(delay * 2, BACKOFF_CAP_S)
                continue
            response.raise_for_status()
            return response
        except httpx.HTTPStatusError:
            # Non-retryable status (404 and friends) — surface it immediately.
            raise
        except (httpx.ConnectError, httpx.ReadError, httpx.WriteError,
                httpx.RemoteProtocolError, httpx.TimeoutException) as exc:
            last_exc = exc
            # A dropped connection may have poisoned the pool; force a fresh one.
            global _SHARED_CLIENT
            if _SHARED_CLIENT is not None:
                _SHARED_CLIENT.close()
                _SHARED_CLIENT = None
            if attempt == attempts:
                break
            time.sleep(min(delay, BACKOFF_CAP_S))
            delay = min(delay * 2, BACKOFF_CAP_S)

    raise httpx.ConnectError(
        f"{url} failed after {attempts} attempts: {last_exc}"
    ) from last_exc


def fetch_index(*, force: bool = False) -> Path:
    """Download the RFC Editor metadata index. One file, one request.

    This is the supersession graph. It is the single most important download in
    the project and the smallest.
    """
    dest = settings.paths.index_xml
    if dest.exists() and not force:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    response = _get_with_retry(INDEX_URL)
    dest.write_bytes(response.content)
    return dest


def fetch_text(number: int, *, force: bool = False) -> Path:
    """Download one RFC's text. Returns the cached path if it already exists."""
    dest = settings.paths.text / f"rfc{number}.txt"
    if dest.exists() and not force:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    response = _get_with_retry(TEXT_URL.format(n=number))
    # RFCs are ASCII by specification; the handful with UTF-8 bodies declare it.
    dest.write_bytes(response.content)
    return dest


def fetch_many(
    numbers: Iterable[int],
    *,
    delay: float = MIN_DELAY_S,
    force: bool = False,
    allow_large_batch: bool = False,
    skip_404: bool = True,
) -> Iterator[Path]:
    """Fetch several RFCs one at a time over HTTP, politely.

    Yields each path as it lands so a caller can show progress. Raises rather than
    silently sending hundreds of requests when `fetch_bulk()` (rsync, or the same
    loop this function runs, run deliberately) is what the caller actually wants.

    A handful of the earliest RFCs (1969-era) have no plain-text version — only a
    scanned PDF exists, or nothing at all. With `skip_404=True` (the default) a 404
    is recorded to `fetch_many.missing_numbers` and the loop continues rather than
    aborting the whole run over a handful of documents from before plain text
    existed. Pass `skip_404=False` when a 404 should be a hard failure instead —
    e.g. re-fetching a specific set of numbers you already know should exist.
    """
    fetch_many.missing_numbers = []
    wanted = list(numbers)
    if len(wanted) > BATCH_WARN_THRESHOLD and not allow_large_batch:
        raise ValueError(
            f"{len(wanted)} documents requested one at a time. Use fetch_bulk() for "
            f"the full corpus, or pass allow_large_batch=True if you mean it."
        )
    delay = max(delay, MIN_DELAY_S)
    for i, number in enumerate(wanted):
        dest = settings.paths.text / f"rfc{number}.txt"
        if dest.exists() and not force:
            yield dest
            continue
        if i:
            time.sleep(delay)
        try:
            yield fetch_text(number, force=force)
        except httpx.HTTPStatusError as exc:
            if skip_404 and exc.response.status_code == 404:
                fetch_many.missing_numbers.append(number)
                continue
            raise


fetch_many.missing_numbers = []  # populated during the most recent fetch_many() call


def _rsync_available() -> bool:
    return shutil.which("rsync") is not None


def _fetch_bulk_rsync(*, progress: bool) -> int:
    text_dir = settings.paths.text
    text_dir.mkdir(parents=True, exist_ok=True)
    source = f"{RSYNC_HOST}::{RSYNC_MODULE}"
    if progress:
        print(f"rsync {source} -> {text_dir}", flush=True)
    subprocess.run(
        ["rsync", "-az", "--info=progress2" if progress else "-q", source, str(text_dir)],
        check=True,
    )
    return len(list(text_dir.glob("rfc*.txt")))


def _all_index_numbers() -> list[int]:
    """Every RFC number the metadata index knows about.

    Parsed directly rather than through `corpus.parse.parse_index` to avoid a
    circular import — `parse.py` reads corpus paths that `fetch.py` populates.
    """
    import xml.etree.ElementTree as ET

    root = ET.parse(settings.paths.index_xml).getroot()
    out: list[int] = []
    for entry in root:
        if not entry.tag.rsplit("}", 1)[-1] == "rfc-entry":
            continue
        for child in entry:
            if child.tag.rsplit("}", 1)[-1] == "doc-id" and child.text:
                digits = child.text.strip().lstrip("RFC").lstrip("0")
                if digits.isdigit():
                    out.append(int(digits))
                break
    return sorted(out)


def fetch_bulk(*, force: bool = False, progress: bool = True) -> int:
    """Fetch the full text corpus: every RFC number in the metadata index.

    Tries rsync first — it's the RFC Editor's own recommended bulk method, one
    connection instead of thousands. Falls back to sequential, rate-limited HTTP
    when rsync isn't installed, which is slower but no less honest a way to end up
    with the same files on disk.

    Requires the index to already be fetched (`fetch_index()`), since that's where
    the full list of RFC numbers comes from.
    """
    if _rsync_available():
        try:
            return _fetch_bulk_rsync(progress=progress)
        except subprocess.CalledProcessError as exc:
            if progress:
                print(f"rsync failed ({exc}); falling back to HTTP", flush=True)

    if not settings.paths.index_xml.exists():
        raise FileNotFoundError(
            "no rfc-index.xml on disk — run fetch_index() first; fetch_bulk() needs "
            "it for the list of RFC numbers when rsync isn't available"
        )

    # Resume by set difference, not by a file count.
    #
    # This previously bailed out whenever more than 1,000 documents were already
    # on disk, on the theory that a populated directory meant a finished corpus.
    # It does not: a run interrupted at 2,895 of 9,835 hit that guard on the next
    # invocation, fetched nothing, and *reported success*. A resume path that
    # silently declares a partial corpus complete is worse than no resume path at
    # all, because everything downstream then indexes a third of the corpus and
    # reports recall against it with nothing looking wrong.
    numbers = _all_index_numbers()
    have = set(local_numbers())
    known_missing = load_missing()
    outstanding = [n for n in numbers if n not in have and n not in known_missing]

    if not outstanding and not force:
        if progress:
            print(
                f"complete: {len(have)} documents on disk, "
                f"{len(known_missing)} known to have no plain text",
                flush=True,
            )
        return len(have)

    target = numbers if force else outstanding
    if progress:
        print(
            f"rsync unavailable — {len(have)} on disk, fetching {len(target)} over "
            f"HTTP at {MIN_DELAY_S}s/request "
            f"(~{len(target) * MIN_DELAY_S / 60:.0f} min)",
            flush=True,
        )
    written = 0
    for i, path in enumerate(
        fetch_many(target, force=force, allow_large_batch=True, skip_404=True),
        start=1,
    ):
        written += 1
        if progress and i % 500 == 0:
            print(f"  {i}/{len(target)} ({path.name})", flush=True)

    # Accumulate across runs rather than overwrite: a resumed run only sees the
    # gaps in its own slice, and clobbering the file would lose the rest.
    newly_missing = set(fetch_many.missing_numbers)
    if newly_missing:
        all_missing = sorted(known_missing | newly_missing)
        _missing_path().write_text(
            "\n".join(str(n) for n in all_missing) + "\n", encoding="utf-8"
        )
        if progress:
            print(
                f"  {len(newly_missing)} more with no plain-text version "
                f"({len(all_missing)} total; list: {_missing_path()})",
                flush=True,
            )
    if progress:
        print(
            f"done: {written} written, {len(local_numbers())} documents on disk",
            flush=True,
        )
    return written


def _missing_path() -> Path:
    return settings.paths.corpus / "missing.txt"


def load_missing() -> set[int]:
    """RFC numbers already known to have no plain-text version.

    Persisted so a resumed run does not re-request documents the server has
    already told us are not there. Without it, every resume re-walks the same
    404s against a volunteer-run server, which is exactly the behavior the
    politeness delay elsewhere in this module exists to avoid.
    """
    path = _missing_path()
    if not path.exists():
        return set()
    return {int(tok) for tok in path.read_text().split() if tok.isdigit()}

def local_numbers() -> list[int]:
    """Every RFC number whose text is already on disk."""
    text_dir = settings.paths.text
    if not text_dir.exists():
        return []
    out = []
    for path in text_dir.glob("rfc*.txt"):
        stem = path.stem[3:]
        if stem.isdigit():
            out.append(int(stem))
    return sorted(out)
