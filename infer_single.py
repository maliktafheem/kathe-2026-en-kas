"""Translate one English sentence into Kashmiri.

    python infer_single.py "The sun rises in the east."
"""

from __future__ import annotations

import argparse

from load_model import MODEL_ID, load


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sentence", help="the English sentence to translate")
    parser.add_argument("--model", default=MODEL_ID, help="model id or local path")
    parser.add_argument("--device", help="cuda or cpu; detected automatically if unset")
    args = parser.parse_args()

    if not args.sentence.strip():
        parser.error("The sentence is empty.")

    translator = load(args.model, args.device)
    print(translator.translate([args.sentence])[0])


if __name__ == "__main__":
    main()
