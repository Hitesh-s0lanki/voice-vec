"""The `/datasets` contract: what was attached, what was understood, what it answered.

Three shapes, and the split between the first two is the same one the profile
makes internally. `Dataset` is the row — enough for a list, cheap to fetch, safe
to poll while a build runs. `DatasetDetail` adds the schema block, which is
thousands of characters and is only ever wanted for one dataset at a time.

`QueryResult` carries `sql` on success *and* on failure, deliberately. A number
that came out of a database nobody can see the query for is a number somebody
has to take on trust, and the whole argument for answering in SQL rather than
by retrieval is that the answer is checkable.
"""

from datetime import datetime
from typing import Any

from src.schemas.wire import Wire


class Dataset(Wire):
    """One attached dataset, as the panel lists it."""

    dataset_id: str
    url: str
    kind: str = ""
    location: str = ""
    # pending | ok | degraded | failed. `pending` is what a freshly added
    # dataset returns, because the build runs behind the request.
    status: str = "pending"
    # The first line of the card — what this dataset is, in a few words.
    title: str = ""
    card: str = ""
    rows: int = 0
    bytes: int = 0
    tables: int = 0
    # Only ever set on `failed`, and always set when it is.
    error: str = ""
    built_at: datetime | None = None
    created_at: datetime | None = None


class DatasetList(Wire):
    datasets: list[Dataset] = []
    # How many more this account may attach, so the panel can disable the form
    # rather than letting somebody paste a URL and be refused.
    limit: int = 0
    enabled: bool = True


class DatasetColumn(Wire):
    """One measured column. The panel's version of a line in the schema card."""

    name: str
    type: str = ""
    coverage: float = 0.0
    distinct: int | None = None
    values: list[str] = []
    examples: list[str] = []
    avg_bytes: int = 0
    filterable: bool = True
    constant: bool = False
    wide: bool = False


class DatasetTable(Wire):
    name: str
    rows: int = 0
    # None when the source would not say cheaply. Different from equal-to-rows,
    # which means the whole file is here.
    total: int | None = None
    sampled: bool = False
    path: str = ""
    columns: list[DatasetColumn] = []
    # Columns the source has that were too expensive to pull, and their size.
    # Named rather than omitted, so nothing writes SQL against them.
    omitted: list[str] = []


class DatasetDetail(Dataset):
    """Everything measured, plus the block the SQL writer is handed."""

    summary: str = ""
    topics: list[str] = []
    good_for: list[str] = []
    not_for: list[str] = []
    schema_card: str = ""
    tables_detail: list[DatasetTable] = []
    notes: list[str] = []


class AddDataset(Wire):
    """A URL, and nothing else.

    Deliberately nothing else: there is nowhere to file a dataset under
    somebody else's name, and no place for a credential, because only public
    URLs are accepted.
    """

    url: str


class QueryRequest(Wire):
    """A question in English, or SQL written by hand. One of the two.

    Both go through the same guard. The SQL path is not a privileged one — it
    is the same read-only sealed connection, and relaxing it for a human would
    make the relaxation reachable from an HTTP request.
    """

    question: str = ""
    sql: str = ""


class QueryResult(Wire):
    dataset_id: str
    # What actually ran. Present even when the query failed.
    sql: str = ""
    columns: list[str] = []
    rows: list[dict[str, Any]] = []
    # True when the row cap cut the result short — the difference between
    # "12 rows" and "the first 12 of more".
    truncated: bool = False
    ms: float = 0.0
    attempts: int = 0
    error: str = ""


class DatasetRemoved(Wire):
    dataset_id: str
    removed: bool = True
