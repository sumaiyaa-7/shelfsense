"""Evaluation harness.

    python -m shelfsense.evals.run_evals

1. Retrieval A/B test: BM25 vs vector vs hybrid on every policy question
   (hit@1, hit@k, MRR) with bootstrap CIs and paired permutation tests.
2. End-to-end run with the configured LLM provider: routing accuracy, answer
   keyword recall, groundedness, SQL execution accuracy, latency p50/p95.

Writes reports/eval_report.md, reports/eval_results.json, reports/retrieval_ab.png.
"""
from __future__ import annotations

import json
from pathlib import Path

from ..agents.data_agent import connect_readonly
from ..agents.orchestrator import Copilot
from ..config import ROOT, get_settings
from ..observability.tracing import Tracer
from ..retrieval.retriever import MODES, Retriever
from .metrics import groundedness, hit_at_k, keyword_recall, reciprocal_rank, sql_result_match
from .stats import bootstrap_ci, paired_permutation_test, percentile

GOLDEN = Path(__file__).with_name("golden.jsonl")
REPORTS = ROOT / "reports"


def load_golden() -> list[dict]:
    return [json.loads(line) for line in GOLDEN.read_text().splitlines() if line.strip()]


def _prefix(item: dict, retriever: Retriever) -> str:
    """Chunk-id prefix of the gold section (doc-sectionIndex-)."""
    for c in retriever.chunks:
        if c.doc == item["expected_doc"] and c.section == item["expected_section"]:
            return c.chunk_id.rsplit("-", 1)[0] + "-"
    raise KeyError(f"gold section not found for {item['id']}")


def retrieval_ab(items: list[dict], retriever: Retriever, k: int) -> dict:
    policy_items = [i for i in items if "expected_doc" in i]
    per_mode: dict[str, dict[str, list[float]]] = {}
    for mode in MODES:
        rr, h1, hk = [], [], []
        for it in policy_items:
            ids = [h.chunk.chunk_id for h in retriever.search(it["question"], k=k, mode=mode)]
            pre = _prefix(it, retriever)
            rr.append(reciprocal_rank(ids, pre)); h1.append(hit_at_k(ids, pre, 1)); hk.append(hit_at_k(ids, pre, k))
        per_mode[mode] = {"mrr": rr, "hit@1": h1, f"hit@{k}": hk}
    summary = {m: {name: bootstrap_ci(v) for name, v in d.items()} for m, d in per_mode.items()}
    tests = {
        f"hybrid_vs_{other}": paired_permutation_test(per_mode["hybrid"]["mrr"], per_mode[other]["mrr"])
        for other in ("bm25", "vector")
    }
    return {"n": len(policy_items), "summary": summary, "p_values_mrr": tests, "per_question": per_mode}


def end_to_end(items: list[dict], copilot: Copilot) -> dict:
    con = connect_readonly(copilot.settings.db_path)
    rows, latencies = [], []
    for it in items:
        r = copilot.ask(it["question"])
        latencies.append(r.latency_ms)
        row = {"id": it["id"], "expected_route": it["route"], "route": r.route, "route_ok": float(r.route == it["route"])}
        if "keywords" in it:
            row["keyword_recall"] = keyword_recall(r.answer, it["keywords"])
            row["groundedness"] = groundedness(r.answer.split("**Data:**")[0], r.contexts)
        if "reference_sql" in it:
            row["sql_exec_match"] = sql_result_match(con, it["reference_sql"], r.rows, it.get("unordered", False))
        row["latency_ms"] = r.latency_ms
        row["tokens"] = r.total_tokens
        rows.append(row)

    def agg(key):
        vals = [r[key] for r in rows if key in r]
        return {"n": len(vals), "ci": bootstrap_ci(vals)}

    return {
        "rows": rows,
        "metrics": {k: agg(k) for k in ("route_ok", "keyword_recall", "groundedness", "sql_exec_match")},
        "latency_ms": {"p50": percentile(latencies, 50), "p95": percentile(latencies, 95)},
        "failures": [r for r in rows if r["route_ok"] < 1 or r.get("keyword_recall", 1) < 1 or r.get("sql_exec_match", 1) < 1],
    }


def fmt(ci) -> str:
    m, lo, hi = ci
    return f"{m:.2f} [{lo:.2f}, {hi:.2f}]"


def write_report(ab: dict, e2e: dict, provider: str, k: int) -> str:
    lines = [
        "# ShelfSense Evaluation Report", "",
        f"LLM provider: `{provider}` | golden questions: {len(e2e['rows'])} | 95% bootstrap CIs in brackets", "",
        f"## 1. Retrieval A/B test ({ab['n']} policy questions, k={k})", "",
        f"| mode | MRR | hit@1 | hit@{k} |", "|---|---|---|---|",
    ]
    for mode, s in ab["summary"].items():
        lines.append(f"| {mode} | {fmt(s['mrr'])} | {fmt(s['hit@1'])} | {fmt(s[f'hit@{k}'])} |")
    lines += ["", "Paired permutation test on per-question reciprocal rank (two-sided):", ""]
    lines += [f"- {name}: p = {p:.3f}" for name, p in ab["p_values_mrr"].items()]
    lines += ["", "## 2. End-to-end pipeline", "", "| metric | n | mean [95% CI] |", "|---|---|---|"]
    for name, m in e2e["metrics"].items():
        lines.append(f"| {name} | {m['n']} | {fmt(m['ci'])} |")
    lines += ["", f"Latency: p50 = {e2e['latency_ms']['p50']:.1f} ms, p95 = {e2e['latency_ms']['p95']:.1f} ms", ""]
    lines += ["## 3. Failure analysis", ""]
    if not e2e["failures"]:
        lines.append("No failures on the golden set.")
    for f in e2e["failures"]:
        lines.append(f"- `{f['id']}`: " + ", ".join(f"{k}={v}" for k, v in f.items() if k not in ("id", "latency_ms", "tokens")))
    return "\n".join(lines) + "\n"


def plot(ab: dict, k: int, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    metrics = ["mrr", "hit@1", f"hit@{k}"]
    modes = list(ab["summary"])
    colors = {"bm25": "#9AA5B1", "vector": "#5B8DEF", "hybrid": "#1F3A93"}
    fig, ax = plt.subplots(figsize=(8, 4.2))
    width = 0.26
    for j, mode in enumerate(modes):
        means = [ab["summary"][mode][m][0] for m in metrics]
        err = [[means[i] - ab["summary"][mode][m][1] for i, m in enumerate(metrics)],
               [ab["summary"][mode][m][2] - means[i] for i, m in enumerate(metrics)]]
        xs = [i + (j - 1) * width for i in range(len(metrics))]
        ax.bar(xs, means, width, yerr=err, capsize=4, label=mode, color=colors.get(mode))
    ax.set_xticks(range(len(metrics)), [m.upper() for m in metrics])
    ax.set_ylim(0, 1.25)
    ax.set_ylabel("score (95% bootstrap CI)")
    ax.set_title(f"Retrieval A/B test: BM25 vs vector vs hybrid (n={ab['n']} questions)")
    ax.legend(frameon=False, ncol=3, loc="upper center")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=160)


def main() -> None:
    settings = get_settings()
    items = load_golden()
    retriever = Retriever.from_dir(settings.policies_dir)
    ab = retrieval_ab(items, retriever, settings.top_k)
    copilot = Copilot(settings, tracer=Tracer(REPORTS / "eval_traces.jsonl"))
    e2e = end_to_end(items, copilot)

    REPORTS.mkdir(exist_ok=True)
    (REPORTS / "eval_traces.jsonl").unlink(missing_ok=True)
    report = write_report(ab, e2e, settings.llm_provider, settings.top_k)
    (REPORTS / "eval_report.md").write_text(report)
    (REPORTS / "eval_results.json").write_text(json.dumps({"retrieval_ab": ab, "end_to_end": e2e}, indent=2))
    plot(ab, settings.top_k, REPORTS / "retrieval_ab.png")
    print(report)


if __name__ == "__main__":
    main()
