# ShelfSense Evaluation Report

LLM provider: `mock` | golden questions: 47 | 95% bootstrap CIs in brackets

## 1. Retrieval A/B test (33 policy questions, k=4)

| mode | MRR | hit@1 | hit@4 |
|---|---|---|---|
| bm25 | 0.87 [0.76, 0.96] | 0.82 [0.67, 0.94] | 0.94 [0.85, 1.00] |
| vector | 0.84 [0.73, 0.93] | 0.76 [0.61, 0.91] | 0.94 [0.85, 1.00] |
| hybrid | 0.87 [0.76, 0.96] | 0.82 [0.67, 0.94] | 0.94 [0.85, 1.00] |

Paired permutation test on per-question reciprocal rank (two-sided):

- hybrid_vs_bm25: p = 1.000
- hybrid_vs_vector: p = 0.489

## 2. End-to-end pipeline

| metric | n | mean [95% CI] |
|---|---|---|
| route_ok | 47 | 0.98 [0.94, 1.00] |
| keyword_recall | 33 | 0.85 [0.73, 0.97] |
| groundedness | 33 | 0.93 [0.86, 0.99] |
| sql_exec_match | 17 | 1.00 [1.00, 1.00] |

Latency: p50 = 0.8 ms, p95 = 2.1 ms

## 3. Failure analysis

- `p08`: expected_route=policy, route=data, route_ok=0.0, keyword_recall=0.0, groundedness=0.0
- `p16`: expected_route=policy, route=policy, route_ok=1.0, keyword_recall=0.0, groundedness=1.0
- `p20`: expected_route=policy, route=policy, route_ok=1.0, keyword_recall=0.0, groundedness=1.0
- `p22`: expected_route=policy, route=policy, route_ok=1.0, keyword_recall=0.0, groundedness=1.0
- `p30`: expected_route=policy, route=policy, route_ok=1.0, keyword_recall=0.0, groundedness=1.0
