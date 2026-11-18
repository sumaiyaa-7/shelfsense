"""Command-line interface.

    python -m shelfsense.cli "Which products are below reorder point in Boston?"
    python -m shelfsense.cli            # interactive mode
"""
from __future__ import annotations

import sys

from .agents.orchestrator import Copilot


def show(copilot: Copilot, question: str) -> None:
    r = copilot.ask(question)
    print(f"\n[route: {r.route} | {r.latency_ms:.0f} ms | tokens: {r.total_tokens} | trace: {r.trace_id}]")
    if r.sql:
        print(f"SQL: {r.sql}")
    print(r.answer)


def main() -> None:
    copilot = Copilot()
    if len(sys.argv) > 1:
        show(copilot, " ".join(sys.argv[1:]))
        return
    print("ShelfSense - ask about store policies or sales/inventory data. Ctrl+C to quit.")
    try:
        while True:
            q = input("\n> ").strip()
            if q:
                show(copilot, q)
    except (KeyboardInterrupt, EOFError):
        print()


if __name__ == "__main__":
    main()
