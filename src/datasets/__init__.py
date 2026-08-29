"""Datasets a user has pointed this app at, and the measuring behind the SQL.

The agent that writes that SQL lives in `src/agents/dataset_agent.py` with the
rest of them; what is here is what it is handed — a measured schema card and a
sealed local file.

`src/connectors/` attaches somebody's *store* and searches it. This attaches
somebody's *dataset* — a URL to parquet or CSV — measures it, and answers
questions about it in SQL rather than by embedding it.

The split between the two halves is the same one `profile_service.py` makes,
for the same reason:

    add()    slow. Network, a model call, a file written. Seconds to minutes.
    query()  fast. Local, sealed, milliseconds — inside a turn somebody is
             listening through.

They meet at a Postgres row and a local DuckDB file.
"""
