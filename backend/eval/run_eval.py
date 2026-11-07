"""Evaluate DocPilot on a labelled question set.

Retrieval metrics (no LLM needed):
  hit@k  - share of questions whose expected document appears in the top-k chunks
  MRR    - mean reciprocal rank of the first chunk from the expected document
End-to-end metrics (runs the full LangGraph agent):
  answer accuracy - answer contains the expected key fact
  citation accuracy - at least one citation points at the expected document

Usage:  python -m eval.run_eval [--k 5] [--no-agent]
"""

from __future__ import annotations

import argparse
import json
import uuid
from pathlib import Path

from app.graph import get_graph, turn_input
from app.vectorstore import search

DATASET = Path(__file__).with_name("dataset.jsonl")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--no-agent", action="store_true", help="only evaluate retrieval")
    args = parser.parse_args()

    rows = [json.loads(line) for line in DATASET.read_text().splitlines() if line.strip()]
    hits, rr, correct, cited_right = 0, 0.0, 0, 0

    for row in rows:
        sources = [d.metadata.get("source") for d, _ in search(row["question"], k=args.k)]
        if row["expected_source"] in sources:
            hits += 1
            rr += 1 / (sources.index(row["expected_source"]) + 1)

        if not args.no_agent:
            out = get_graph().invoke(turn_input(row["question"]), {"configurable": {"thread_id": str(uuid.uuid4())}})
            ok = row["expected_answer_contains"].lower() in out["answer"].lower()
            cit = any(c["source"] == row["expected_source"] for c in out.get("citations", []))
            correct += ok
            cited_right += cit
            print(f"{'PASS' if ok and cit else 'FAIL'} | {row['question']}\n       -> {out['answer']}")

    n = len(rows)
    print("\n=== Results ===")
    print(f"questions:          {n}")
    print(f"hit@{args.k}:             {hits / n:.0%}")
    print(f"MRR:                {rr / n:.2f}")
    if not args.no_agent:
        print(f"answer accuracy:    {correct / n:.0%}")
        print(f"citation accuracy:  {cited_right / n:.0%}")


if __name__ == "__main__":
    main()
