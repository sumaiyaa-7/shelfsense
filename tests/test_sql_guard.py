import pytest

from shelfsense.agents.sql_guard import UnsafeSQLError, validate_sql


def test_select_gets_limit():
    assert validate_sql("SELECT * FROM sales").endswith("LIMIT 200")


def test_existing_limit_kept_and_fences_stripped():
    assert validate_sql("```sql\nSELECT name FROM products LIMIT 3;\n```") == "SELECT name FROM products LIMIT 3"


def test_cte_allowed():
    validate_sql("WITH t AS (SELECT * FROM sales) SELECT COUNT(*) FROM t")


@pytest.mark.parametrize("sql", [
    "DELETE FROM sales",
    "DROP TABLE stores",
    "SELECT 1; DROP TABLE sales",
    "UPDATE inventory SET on_hand = 0",
    "SELECT * FROM sqlite_master",
    "ATTACH DATABASE 'x.db' AS x",
    "PRAGMA table_info(sales)",
    "",
])
def test_unsafe_sql_rejected(sql):
    with pytest.raises(UnsafeSQLError):
        validate_sql(sql)


def test_readonly_connection_blocks_writes(database):
    import sqlite3
    from shelfsense.agents.data_agent import connect_readonly
    con = connect_readonly(database)
    with pytest.raises(sqlite3.OperationalError):
        con.execute("DELETE FROM sales")
