#!/usr/bin/env python3
"""Build FB15K-237 or WN18RR graph files and MiniLM vector chunks."""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parents[1]
MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_triples(paths: Iterable[Path]) -> tuple[set[str], set[tuple[str, str]]]:
    entities: set[str] = set()
    edges: set[tuple[str, str]] = set()
    for path in paths:
        with path.open("r", encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, 1):
                fields = line.rstrip(chr(10)).split(chr(9))
                if len(fields) != 3:
                    raise ValueError(f"Malformed triple at {path}:{line_number}")
                left, _, right = fields
                entities.update((left, right))
                if left != right:
                    edges.add((left, right) if left < right else (right, left))
    return entities, edges


def fb15k_inputs(input_dir: Path) -> tuple[list[Path], dict[str, list[str]]]:
    triples = [input_dir / name for name in ("train.txt", "valid.txt", "test.txt")]
    semantics: dict[str, list[str]] = defaultdict(list)
    text_path = input_dir / "text_emnlp.txt"
    with text_path.open("r", encoding="utf-8") as stream:
        for line in stream:
            fields = line.rstrip(chr(10)).split(chr(9))
            if len(fields) != 4:
                continue
            left, path, right, _ = fields
            words = [
                token
                for token in path.split(":")
                if token and not token.startswith("<") and not token.startswith("[")
            ]
            text = " ".join(words)
            if text:
                semantics[left].append(text)
                semantics[right].append(text)
    return triples, semantics


def wn18rr_inputs(input_dir: Path) -> tuple[list[Path], dict[str, list[str]]]:
    triples = [input_dir / name for name in ("train.tsv", "dev.tsv", "test.tsv")]
    semantics: dict[str, list[str]] = defaultdict(list)
    with (input_dir / "entity2text.txt").open("r", encoding="utf-8") as stream:
        for line in stream:
            fields = line.rstrip(chr(10)).split(chr(9), 1)
            if len(fields) == 2:
                semantics[fields[0]].append(fields[1])
    with (input_dir / "wordnet-mlj12-definitions.txt").open("r", encoding="utf-8") as stream:
        for line in stream:
            fields = line.rstrip(chr(10)).split(chr(9), 2)
            if len(fields) >= 3:
                semantics[fields[0]].append(fields[2])
    return triples, semantics


def write_graph(
    output: Path,
    entities: list[str],
    edges: set[tuple[str, str]],
) -> None:
    mapping = {entity: index for index, entity in enumerate(entities)}
    normalized = sorted((mapping[left], mapping[right]) for left, right in edges)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    with temporary.open("w", encoding="ascii", newline=chr(10)) as stream:
        stream.write(f"{len(entities)} {len(normalized)}" + chr(10))
        for left, right in normalized:
            stream.write(f"{left} {right}" + chr(10))
    temporary.replace(output)


def encode(
    texts: list[str],
    batch_size: int,
    device_name: str,
    revision: str | None,
):
    import numpy as np
    import torch
    import transformers
    from transformers import AutoModel, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=revision)
    model = AutoModel.from_pretrained(MODEL_ID, revision=revision)
    device = torch.device(device_name)
    model.to(device)
    model.eval()
    rows = []
    with torch.no_grad():
        for start in range(0, len(texts), batch_size):
            batch = texts[start : start + batch_size]
            inputs = tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=256,
                return_tensors="pt",
            )
            inputs = {key: value.to(device) for key, value in inputs.items()}
            outputs = model(**inputs)
            token_vectors = outputs[0]
            mask = inputs["attention_mask"].unsqueeze(-1).expand(token_vectors.size()).float()
            pooled = torch.sum(token_vectors * mask, 1) / torch.clamp(mask.sum(1), min=1e-9)
            pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)
            rows.append(pooled.cpu().numpy())
            print(f"Encoded {min(start + batch_size, len(texts))}/{len(texts)}")
    values = np.concatenate(rows, axis=0)
    commit = getattr(model.config, "_commit_hash", None)
    versions = {
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "numpy": np.__version__,
        "resolved_model_commit": commit,
    }
    return values, versions


def write_vectors(index_path: Path, chunk_path: Path, values) -> None:
    import numpy as np

    index_path.parent.mkdir(parents=True, exist_ok=True)
    chunk_temp = chunk_path.with_suffix(chunk_path.suffix + ".tmp")
    with chunk_temp.open("wb") as stream:
        stream.write(struct.pack("<ii", values.shape[0], values.shape[1]))
        np.asarray(values, dtype="<f8", order="C").tofile(stream)
    chunk_temp.replace(chunk_path)

    try:
        listed = chunk_path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        listed = str(chunk_path.resolve())
    index_temp = index_path.with_suffix(index_path.suffix + ".tmp")
    index_temp.write_text(
        f"{values.shape[0]} {values.shape[1]}" + chr(10) + listed + chr(10),
        encoding="utf-8",
    )
    index_temp.replace(index_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("FB15K-237", "WN18RR"), required=True)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--graph-output", type=Path)
    parser.add_argument("--index-output", type=Path)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--model-revision")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_dir = args.input_dir.resolve()
    if args.dataset == "FB15K-237":
        triple_paths, semantics = fb15k_inputs(input_dir)
    else:
        triple_paths, semantics = wn18rr_inputs(input_dir)
    required = triple_paths + [
        input_dir / (
            "text_emnlp.txt" if args.dataset == "FB15K-237" else "entity2text.txt"
        )
    ]
    if args.dataset == "WN18RR":
        required.append(input_dir / "wordnet-mlj12-definitions.txt")
    missing = [path for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(", ".join(map(str, missing)))

    graph_output = args.graph_output or (
        ROOT / "dataset" / "dataset" / f"{args.dataset}_edges.txt"
    )
    index_output = args.index_output or (
        ROOT / "dataset" / "vectors" / f"{args.dataset}_vectors.index"
    )
    chunk_output = index_output.parent / f"{args.dataset}_vectors.json.chunk.0.bin"
    provenance_output = index_output.parent / f"{args.dataset}_provenance.json"
    existing = [path for path in (graph_output, index_output, chunk_output, provenance_output) if path.exists()]
    if existing and not args.force:
        raise FileExistsError("Refusing to overwrite without --force: " + ", ".join(map(str, existing)))

    entity_set, edges = read_triples(triple_paths)
    entities = sorted(entity_set)
    write_graph(graph_output, entities, edges)
    texts = [" ".join(semantics.get(entity, [])[:5]) or "unknown entity" for entity in entities]
    values, versions = encode(texts, args.batch_size, args.device, args.model_revision)
    write_vectors(index_output, chunk_output, values)

    source_files = {}
    for path in required:
        source_files[path.name] = {
            "bytes": path.stat().st_size,
            "sha256": sha256(path),
        }
    record = {
        "schema_version": 1,
        "dataset": args.dataset,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_files": source_files,
        "transformation": {
            "graph": "union all split triples, remove self-loops, ignore relation and direction, deduplicate, sort entity identifiers",
            "text": "at most five descriptions per entity; fallback text is 'unknown entity'",
            "model": MODEL_ID,
            "model_revision_requested": args.model_revision,
            "pooling": "attention-mask mean pooling followed by L2 normalization",
            "max_tokens": 256,
            **versions,
        },
        "output": {
            "vertices": len(entities),
            "edges": len(edges),
            "dimension": int(values.shape[1]),
            "graph": {"path": str(graph_output), "bytes": graph_output.stat().st_size, "sha256": sha256(graph_output)},
            "index": {"path": str(index_output), "bytes": index_output.stat().st_size, "sha256": sha256(index_output)},
            "chunk": {"path": str(chunk_output), "bytes": chunk_output.stat().st_size, "sha256": sha256(chunk_output)},
        },
    }
    provenance_output.write_text(json.dumps(record, indent=2) + chr(10), encoding="utf-8")
    print(f"Wrote {graph_output}")
    print(f"Wrote {index_output} and {chunk_output}")
    print(f"Wrote {provenance_output}")


if __name__ == "__main__":
    main()