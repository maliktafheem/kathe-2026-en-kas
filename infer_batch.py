"""Translate a CSV of English sentences into Kashmiri.

The input needs an `ID` column and a `sentence` column. The output has an `ID`
column and a `kashmiri_text` column, in the same row order, which is the
KATHE 2026 submission format.

    python infer_batch.py --input examples/sample_input.csv --output predictions.csv
"""

from __future__ import annotations

import argparse
import csv
import re
import time
from pathlib import Path

from load_model import BATCH_SIZE, MODEL_ID, load

# Degenerate decoding can repeat a punctuation mark until it hits max_length.
# No reference contains such a run, so collapsing one is always safe. This
# changed 0 of the 1,730 rows in the scored submission; it is a guard, not a
# scoring trick.
PUNCTUATION_RUN = re.compile(r"([۔،؛؟.,;?!])(?:\s*\1)+")


def read_input(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"{path} has no rows.")
    if not {"ID", "sentence"}.issubset(rows[0]):
        raise ValueError(f"{path} needs an ID column and a sentence column.")
    if any(not row["ID"].strip() or not row["sentence"].strip() for row in rows):
        raise ValueError("Every ID and sentence must be non-empty.")
    if len({row["ID"] for row in rows}) != len(rows):
        raise ValueError("The IDs must be unique.")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="CSV with ID,sentence")
    parser.add_argument("--output", type=Path, required=True, help="CSV to write")
    parser.add_argument("--model", default=MODEL_ID, help="model id or local path")
    parser.add_argument("--device", help="cuda or cpu; detected automatically if unset")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    args = parser.parse_args()

    rows = read_input(args.input)
    # Repeated sentences are translated once. Beam search is deterministic, so
    # this only saves time; it does not change any output.
    sources = list(dict.fromkeys(row["sentence"] for row in rows))
    print(f"{len(rows)} rows, {len(sources)} unique sentences")

    translator = load(args.model, args.device)
    print(f"loaded {args.model} on {translator.device}")

    started = time.time()
    translations = translator.translate(
        sources, batch_size=args.batch_size, progress=True
    )
    by_source = dict(zip(sources, translations, strict=True))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["ID", "kashmiri_text"])
        writer.writeheader()
        for row in rows:
            text = PUNCTUATION_RUN.sub(r"\1", by_source[row["sentence"]])
            if not text.strip():
                raise ValueError(f"Row {row['ID']} produced an empty translation.")
            writer.writerow({"ID": row["ID"], "kashmiri_text": text})

    print(f"wrote {args.output} in {time.time() - started:.0f}s")


if __name__ == "__main__":
    main()
