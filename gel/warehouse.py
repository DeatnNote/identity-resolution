"""DuckDB connection with the source views and metric definitions loaded; named-query runner."""
import re
from pathlib import Path

import duckdb

SQL = Path(__file__).resolve().parent.parent / "sql"


def connect(root="."):
    con = duckdb.connect()
    base = Path(root).resolve().as_posix()
    for f in ("00_sources.sql", "10_metrics.sql"):
        con.execute((SQL / f).read_text().replace("'data/", f"'{base}/data/"))
    return con


def named_queries(path=SQL / "20_growth.sql"):
    text = Path(path).read_text()
    parts = re.split(r"^-- name: (\w+)\s*$", text, flags=re.M)
    return {parts[i]: parts[i + 1].strip().rstrip(";") for i in range(1, len(parts), 2)}


def query(con, name):
    return con.execute(named_queries()[name]).df()
