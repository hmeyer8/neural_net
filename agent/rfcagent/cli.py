"""Command line entry point.

    rfcagent fetch --index          metadata index only (one request, ~12 MB)
    rfcagent fetch --bulk           every RFC text file (rsync if available, else
                                     ~9,800 rate-limited HTTP requests, ~30 min)
    rfcagent fetch --rfc 7234 2616  a handful of documents, rate limited
    rfcagent status                 what is on disk
    rfcagent show 7234              parsed sections and the supersession graph

Subcommands are added as the components behind them land. A subcommand that
prints "not built yet" is deliberate — see ROADMAP.md for the week it arrives.
"""

from __future__ import annotations

import argparse
import sys

from .config import settings


def _cmd_fetch(args: argparse.Namespace) -> int:
    from .corpus import fetch

    if args.index or not (args.bulk or args.rfc):
        path = fetch.fetch_index(force=args.force)
        size_mb = path.stat().st_size / 1e6
        print(f"index: {path} ({size_mb:.1f} MB)")
    if args.bulk:
        count = fetch.fetch_bulk(force=args.force)
        print(f"text:  {settings.paths.text} ({count} documents)")
    if args.rfc:
        for path in fetch.fetch_many(args.rfc, force=args.force):
            print(f"  {path.name}")
    return 0


def _cmd_status(_: argparse.Namespace) -> int:
    from .corpus import fetch

    paths = settings.paths
    index_ok = paths.index_xml.exists()
    numbers = fetch.local_numbers()
    from .config import device_report

    print(f"data root       {paths.data}")
    print(f"index           {'present' if index_ok else 'MISSING — run: rfcagent fetch --index'}")
    print(f"text documents  {len(numbers)}")
    if numbers:
        print(f"                rfc{numbers[0]} .. rfc{numbers[-1]}")
    print(f"device          {device_report()}")
    print(f"embed batch     {settings.retrieval.embed_batch_size()}")
    print(f"rerank batch    {settings.retrieval.rerank_batch_size()}")
    print(f"config hash     {settings.fingerprint()}")
    return 0


def _cmd_show(args: argparse.Namespace) -> int:
    from .corpus.parse import load_document

    doc = load_document(args.number)
    meta = doc.meta
    print(f"RFC {meta.number}  {meta.title}")
    print(f"  status        {meta.current_status}  ({meta.stream or 'unknown stream'})")
    print(f"  published     {meta.month or ''} {meta.year or ''}".rstrip())
    for label, values in (
        ("obsoletes", meta.obsoletes),
        ("obsoleted by", meta.obsoleted_by),
        ("updates", meta.updates),
        ("updated by", meta.updated_by),
    ):
        if values:
            print(f"  {label:<13} {', '.join(f'RFC{v}' for v in values)}")
    if meta.is_obsolete:
        print("  ** OBSOLETE — answers citing this document need the successor **")
    print(f"\n  {len(doc.sections)} sections parsed")
    for section in doc.sections[: args.limit]:
        norm = ", ".join(section.normative[:4])
        marker = f"  [{norm}]" if norm else ""
        number = section.number or "—"
        print(f"    {number:<10} {section.title[:56]:<56} {len(section.text):>6}c{marker}")
    if len(doc.sections) > args.limit:
        print(f"    ... {len(doc.sections) - args.limit} more (--limit to see them)")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="rfcagent", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p_fetch = sub.add_parser("fetch", help="download the corpus")
    p_fetch.add_argument("--index", action="store_true", help="metadata index only")
    p_fetch.add_argument(
        "--bulk", action="store_true", help="all RFC text (rsync or HTTP fallback)"
    )
    p_fetch.add_argument("--rfc", type=int, nargs="+", metavar="N", help="specific RFCs")
    p_fetch.add_argument("--force", action="store_true", help="re-download cached files")
    p_fetch.set_defaults(func=_cmd_fetch)

    p_status = sub.add_parser("status", help="what is on disk")
    p_status.set_defaults(func=_cmd_status)

    p_show = sub.add_parser("show", help="parsed sections and supersession graph")
    p_show.add_argument("number", type=int)
    p_show.add_argument("--limit", type=int, default=30)
    p_show.set_defaults(func=_cmd_show)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
