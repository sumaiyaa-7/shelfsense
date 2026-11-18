# ShelfSense Evaluation Report

LLM provider: `mock` | golden questions: 26 | 95% bootstrap CIs in brackets

## 1. Retrieval A/B test (18 policy questions, k=4)

| mode | MRR | hit@1 | hit@4 |
|---|---|---|---|
| bm25 | 0.92 [0.78, 1.00] | 0.89 [0.72, 1.00] | 0.94 [0.83, 1.00] |
| vector | 0.86 [0.72, 0.97] | 0.78 [0.56, 0.94] | 0.94 [0.83, 1.00] |
| hybrid | 0.92 [0.78, 1.00] | 0.89 [0.72, 1.00] | 0.94 [0.83, 1.00] |

Paired permutation test on per-question reciprocal rank (two-sided):

- hybrid_vs_bm25: p = 1.000
- hybrid_vs_vector: p = 0.494

## 2. End-to-end pipeline

| metric | n | mean [95% CI] |
|---|---|---|
| route_ok | 26 | 0.96 [0.88, 1.00] |
| keyword_recall | 18 | 0.89 [0.72, 1.00] |
| groundedness | 18 | 0.90 [0.76, 1.00] |
| sql_exec_match | 10 | 1.00 [1.00, 1.00] |

Latency: p50 = 0.9 ms, p95 = 2.2 ms

## 3. Failure analysis

- `p08`: expected_route=policy, route=data, route_ok=0.0, keyword_recall=0.0, groundedness=0.0
- `p16`: expected_route=policy, route=policy, route_ok=1.0, keyword_recall=0.0, groundedness=1.0
