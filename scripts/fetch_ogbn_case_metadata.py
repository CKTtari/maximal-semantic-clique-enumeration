#!/usr/bin/env python3
"""Fetch public paper metadata for the archived ogbn-arxiv case."""

from __future__ import annotations

import csv
import gzip
import io
import json
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from zipfile import ZipFile


ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT / "dataset" / "downloads" / "ogbn-arxiv" / "arxiv.zip"
CASE_FILE = ROOT / "experiments" / "quality" / "ogbn-arxiv-cases.json"
OUTPUT_FILE = ROOT / "experiments" / "quality" / "ogbn-arxiv-case-metadata.json"
ENDPOINT = (
    "https://api.semanticscholar.org/graph/v1/paper/batch"
    "?fields=title,authors,year,externalIds,url"
)


def load_paper_ids(vertices: set[int]) -> dict[int, int]:
    result: dict[int, int] = {}
    with ZipFile(ARCHIVE) as archive:
        with archive.open("arxiv/mapping/nodeidx2paperid.csv.gz") as member:
            with gzip.GzipFile(fileobj=member) as compressed:
                text = io.TextIOWrapper(compressed, encoding="utf-8")
                rows = csv.reader(text)
                next(rows)
                for row in rows:
                    vertex = int(row[0])
                    if vertex in vertices:
                        result[vertex] = int(row[1])
    if set(result) != vertices:
        raise ValueError(f"Missing paper identifiers for {sorted(vertices - set(result))}")
    return result


def main() -> None:
    cases = json.loads(CASE_FILE.read_text(encoding="utf-8"))
    vertices = [int(vertex) for vertex in cases[0]["vertices"]]
    paper_ids = load_paper_ids(set(vertices))
    requested_ids = [f"MAG:{paper_ids[vertex]}" for vertex in vertices]
    request = urllib.request.Request(
        ENDPOINT,
        data=json.dumps({"ids": requested_ids}).encode("utf-8"),
        headers={"Content-Type": "application/json", "User-Agent": "ACMSC-research-artifact/1.0"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        records = json.loads(response.read().decode("utf-8"))
    if len(records) != len(vertices) or any(record is None for record in records):
        raise ValueError("Semantic Scholar returned incomplete metadata")

    papers = []
    for index, (vertex, requested_id, record) in enumerate(zip(vertices, requested_ids, records), 1):
        if not record.get("title") or not record.get("authors") or not record.get("year"):
            raise ValueError(f"Incomplete paper record for {requested_id}")
        papers.append(
            {
                "case_index": index,
                "vertex": vertex,
                "requested_mag_id": requested_id.removeprefix("MAG:"),
                "resolved_paper_id": record["paperId"],
                "resolved_mag_id": record.get("externalIds", {}).get("MAG"),
                "title": record["title"],
                "year": record["year"],
                "authors": [author["name"] for author in record["authors"]],
                "external_ids": record.get("externalIds", {}),
                "url": record.get("url"),
            }
        )

    output = {
        "schema_version": 1,
        "retrieved_utc": datetime.now(timezone.utc).isoformat(),
        "source": {
            "provider": "Semantic Scholar Academic Graph API",
            "endpoint": ENDPOINT,
            "requested_identifier_type": "MAG",
            "note": "Resolved identifiers retain API-side merged-paper redirects when present.",
        },
        "ogbn_mapping": "arxiv/mapping/nodeidx2paperid.csv.gz in the official ogbn-arxiv archive",
        "case_rank": cases[0]["rank"],
        "papers": papers,
    }
    OUTPUT_FILE.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
