"""Read-only phrase search over all COCO captions for dictionary reviewers."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import re
import sqlite3


def search(database: Path, expressions: list[str], limit: int):
    db = sqlite3.connect(f"file:{database.resolve()}?mode=ro", uri=True)
    results = []
    for expression in expressions:
        tokens = re.findall(r"\w+", expression)
        if not tokens:
            raise ValueError("Search expressions must contain words")
        query = " AND ".join('"' + token.replace('"', '""') + '"' for token in tokens)
        pattern = re.compile(r"(?<!\w)" + re.escape(expression) + r"(?!\w)", re.I)
        found = []
        for row in db.execute(
            "SELECT c.caption_id,c.image_id,c.split,c.caption FROM caption_fts f "
            "JOIN captions c ON c.rowid=f.rowid WHERE caption_fts MATCH ?",
            (query,),
        ):
            if pattern.search(row[3]):
                found.append(row)
        # Stable spread across source IDs, not hand-picked favorable sentences.
        found.sort(key=lambda r: (r[0] * 2654435761) % 4294967296)
        examples = found[:limit]
        results.append(
            {
                "expression": expression,
                "caption_count": len(found),
                "counts_by_split": dict(Counter(r[2] for r in found)),
                "examples": [
                    {"caption_id": r[0], "image_id": r[1], "split": r[2], "caption": r[3]} for r in examples
                ],
                "scope": "Exact case-insensitive phrase with word boundaries in ALL train and validation captions, without filtering by image labels.",
            }
        )
    db.close()
    return results


def caption_ids(database: Path, ids: list[int]):
    db = sqlite3.connect(f"file:{database.resolve()}?mode=ro", uri=True)
    rows = []
    for caption_id in ids:
        row = db.execute(
            "SELECT caption_id,image_id,split,caption FROM captions WHERE caption_id=?", (caption_id,)
        ).fetchone()
        if row is not None:
            rows.append({"caption_id": row[0], "image_id": row[1], "split": row[2], "caption": row[3]})
    db.close()
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", type=Path, required=True)
    parser.add_argument("--phrases", nargs="+")
    parser.add_argument("--caption-ids", nargs="+", type=int)
    parser.add_argument("--limit", type=int, default=8)
    args = parser.parse_args()
    if not 1 <= args.limit <= 30:
        parser.error("limit must be between 1 and 30")
    if args.phrases:
        result = search(args.database, args.phrases, args.limit)
    elif args.caption_ids:
        result = caption_ids(args.database, args.caption_ids)
    else:
        parser.error("Provide --phrases or --caption-ids")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
