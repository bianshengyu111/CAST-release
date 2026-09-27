#!/usr/bin/env python3
"""Audit the bibliography of the manuscript for entries with no in-text citation.

Reproduces the audit reported in the response letter (Reviewer 1, bibliographic
note): it parses the manuscript's ``thebibliography`` block, collects every
in-text citation marker, and reports the entries that are never cited.

Usage
-----
    python scripts/audit_citations.py [path/to/manuscript.tex]

If no path is given, the script looks for ``manuscript.tex`` next to the
repository root and, failing that, prints usage.  The manuscript itself is not
part of this repository (it is submitted separately), so the path is supplied by
the reader.

The script is deliberately dependency-free: standard library only.
"""
from __future__ import annotations

import os
import re
import sys

BIB_BEGIN = "\\begin{thebibliography}"
BIB_END = "\\end{thebibliography}"

# In-text citation markers.  The manuscript uses two conventions:
#   * hard-coded bracketed numbers, e.g. "[12]" or "[3, 7]" or "[4]--[9]";
#   * \cite{key} commands.
BRACKET = re.compile(r"(?<![\w.\[])\[(\d+(?:\s*[,–-]\s*\d+)*)\]")
CITE = re.compile(r"\\cite[A-Za-z]*\*?(?:\[[^\]]*\])?\{([^}]*)\}")
BIBITEM = re.compile(r"\\bibitem(?:\s*\[[^\]]*\])?\s*\{([^},]+)")
RANGE_SEP = re.compile(r"\s*[,–-]\s*")


def parse(text: str):
    """Return (ordered bibitem keys, body text with the bibliography removed)."""
    try:
        i = text.index(BIB_BEGIN)
        j = text.index(BIB_END, i)
    except ValueError:
        raise SystemExit("No thebibliography environment found in this file.")

    bib_block = text[i:j]
    body = text[:i]

    keys = BIBITEM.findall(bib_block)
    keys = [k.strip() for k in keys]
    return keys, body, bib_block


def numeric_markers(body: str) -> set:
    """Numbers that appear as bracketed citation markers in the body."""
    out = set()
    for m in BRACKET.finditer(body):
        for part in RANGE_SEP.split(m.group(1)):
            part = part.strip()
            if part.isdigit():
                out.add(int(part))
    return out


def cited_keys(body: str) -> set:
    out = set()
    for m in CITE.finditer(body):
        for k in m.group(1).split(","):
            out.add(k.strip())
    return out


def main(argv):
    if len(argv) > 1:
        path = argv[1]
    else:
        here = os.path.dirname(os.path.abspath(__file__))
        path = os.path.join(os.path.dirname(here), "manuscript.tex")
        if not os.path.exists(path):
            print(__doc__)
            return 2

    with open(path, encoding="utf-8", errors="replace") as fh:
        text = fh.read()

    keys, body, bib_block = parse(text)
    nums = numeric_markers(body)
    ck = cited_keys(body)

    print("manuscript      : %s" % path)
    print("bibliography    : %d entries" % len(keys))
    print("numeric markers : %d distinct" % len(nums))
    print("\\cite keys      : %s" % (", ".join(sorted(ck)) or "none"))
    print()

    uncited = []
    for n, key in enumerate(keys, start=1):
        by_number = n in nums
        by_key = key in ck
        if not (by_number or by_key):
            uncited.append((n, key))

    # Off-by-one guard: warn if markers exceed the bibliography, which usually
    # means the reference list was renumbered without updating the text.
    over = sorted(x for x in nums if x > len(keys))
    if over:
        print("WARNING: markers beyond the reference list: %s" % over)

    if uncited:
        print("Uncited entries (%d):" % len(uncited))
        for n, key in uncited:
            print("  [%d] %s" % (n, key))
    else:
        print("Uncited entries: none")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
