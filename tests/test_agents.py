from shelfsense.agents.router import rule_route
from shelfsense.evals.stats import bootstrap_ci, paired_permutation_test


def test_router():
    assert rule_route("Can a customer return an opened printer?").label == "policy"
    assert rule_route("Top 5 products by revenue in Boston").label == "data"
    assert rule_route("Which Boston products are running low and what is the reorder policy?").label == "hybrid"


def test_policy_answer_is_cited(copilot):
    r = copilot.ask("How long do rewards last before they expire?")
    assert r.route == "policy"
    assert "90 days" in r.answer
    assert r.citations and r.citations[0].startswith("rewards_program")


def test_data_answer_matches_sql(copilot):
    r = copilot.ask("What are the top 3 products by revenue in Edison?")
    assert r.route == "data" and len(r.rows) == 3
    revenues = [row[1] for row in r.rows]
    assert revenues == sorted(revenues, reverse=True)


def test_trace_recorded(copilot):
    r = copilot.ask("Show revenue by store in March")
    trace = copilot.tracer.get(r.trace_id)
    names = {c["name"] for c in trace["children"]}
    assert {"router", "data_agent"} <= names


def test_unknown_data_question_degrades_gracefully(copilot):
    r = copilot.data_agent.answer("what is the meaning of life in sql")
    assert r.error == "no_template"


def test_bootstrap_ci_contains_mean():
    m, lo, hi = bootstrap_ci([1, 0, 1, 1, 0, 1, 1, 1])
    assert lo <= m <= hi


def test_permutation_test():
    assert paired_permutation_test([1] * 20, [0] * 20) < 0.01
    assert paired_permutation_test([1, 0, 1, 0], [1, 0, 1, 0]) == 1.0


def test_regressions_found_by_evals(copilot):
    # "restocking fee" must not trigger the low-stock query
    r = copilot.ask("What were Edison's top 3 products by revenue and what is the restocking fee on opened printers?")
    assert r.route == "hybrid" and "SUM(x.revenue)" in r.sql and "reorder_point" not in r.sql
    # "sell" (not just "sales") should map to a totals query
    assert copilot.ask("How many units of furniture did Boston sell in March?").sql is not None
    # "which category" is a data question
    assert copilot.ask("Which category brings in the most money in Edison?").route == "data"
