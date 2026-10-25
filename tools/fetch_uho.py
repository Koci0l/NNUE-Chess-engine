#!/usr/bin/env python3
"""Download a UHO opening book from official-stockfish/books.

The books are published as .zip files containing a plain EPD file (one FEN per
line), which the arena reads directly:

    python3 tools/fetch_uho.py UHO_4060_v1.epd
    python3 tools/fetch_uho.py UHO_Lichess_4852_v1.epd -o books/

Requires network access to api.github.com; if that is not available, download
the zip manually from https://github.com/official-stockfish/books and unzip it.
"""

import argparse
import io
import os
import sys
import urllib.request
import zipfile

URL = ("https://api.github.com/repos/official-stockfish/books/contents/"
       "{name}.zip")

KNOWN = [
    "UHO_4060_v1.epd",
    "UHO_4060_v2.epd",
    "UHO_4060_v3.epd",
    "UHO_4060_v4.epd",
    "UHO_Lichess_4852_v1.epd",
]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("book", nargs="?", default="UHO_4060_v1.epd",
                    help="book file name, e.g. UHO_4060_v1.epd (default) or "
                         "UHO_Lichess_4852_v1.epd; known: " + ", ".join(KNOWN))
    ap.add_argument("-o", "--out", default=".", help="output directory (default .)")
    args = ap.parse_args()

    name = args.book
    if name.endswith(".zip"):
        name = name[:-4]

    url = URL.format(name=name)
    print(f"downloading {url}")

    req = urllib.request.Request(
        url, headers={"Accept": "application/vnd.github.raw"})

    try:
        data = urllib.request.urlopen(req, timeout=300).read()
    except Exception as exc:                     # noqa: BLE001
        print(f"error: download failed: {exc}", file=sys.stderr)
        print("       download it manually from "
              "https://github.com/official-stockfish/books", file=sys.stderr)
        return 1

    print(f"  {len(data)} bytes")

    os.makedirs(args.out, exist_ok=True)

    try:
        with zipfile.ZipFile(io.BytesIO(data)) as zf:
            zf.extractall(args.out)
            for member in zf.namelist():
                path = os.path.join(args.out, member)
                size = os.path.getsize(path) if os.path.exists(path) else 0
                print(f"extracted {path} ({size} bytes)")
    except zipfile.BadZipFile:
        print("error: downloaded data is not a zip file "
              "(GitHub API rejected the request?)", file=sys.stderr)
        return 1

    print("done. use it with:  ./arena --uho " + name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
