"""HTTP surface for datasets: attach a URL, see what was understood, ask it things.

Every route requires a signed-in caller, and "signed in" means a Clerk session
token whose signature checked out — the same rule as `/connectors`, for a
weaker but real reason. A dataset holds no credential, so this is not about
secrets; it is that a dataset is a file this deployment downloaded and stores on
its own disk, and anonymous callers attaching those is an open invitation to
fill it.

Two shapes worth noticing:

**Adding returns immediately, with `status: "pending"`.** Resolving the URL is
synchronous, because that is where the errors worth showing somebody live — a
typo, a gated repo, a URL that is not a dataset. Everything after it runs on a
worker, so a 500 MB pull does not hold an HTTP connection open for a minute.
The panel polls the list.

**`POST /datasets/{id}/query` answers with the SQL it ran, and returns 200 for a
query that failed.** A failed query is a result, not a server error: the caller
gets the reason and the statement, which is what makes it fixable. A 500 would
lose both to an error handler.
"""

from __future__ import annotations

import logging
from typing import Annotated, Any

from fastapi import APIRouter, Depends, Header, HTTPException, status

from src.core.clerk import Verifier, get_verifier
from src.agents.dataset_agent import Answer, DatasetAgent, get_dataset_agent
from src.datasets.profile import DatasetProfile
from src.datasets.service import DatasetError, DatasetService, get_dataset_service
from src.datasets.store import DatasetRow
from src.schemas.datasets import (
    AddDataset,
    Dataset,
    DatasetColumn,
    DatasetDetail,
    DatasetList,
    DatasetRemoved,
    DatasetTable,
    QueryRequest,
    QueryResult,
)

log = logging.getLogger("vec.datasets")

router = APIRouter(prefix="/datasets", tags=["datasets"])

ServiceDep = Annotated[DatasetService, Depends(get_dataset_service)]
AgentDep = Annotated[DatasetAgent, Depends(get_dataset_agent)]


def require_user(
    verifier: Annotated[Verifier, Depends(get_verifier)],
    authorization: Annotated[str | None, Header(description="Bearer <clerk token>")] = None,
) -> str:
    """The signed-in account, or 401. Identical to `/connectors` on purpose."""
    scheme, _, token = (authorization or "").partition(" ")
    user_id = verifier.user_id(token if scheme.lower() == "bearer" else None)

    if not user_id:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Sign in to manage datasets.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return user_id


UserDep = Annotated[str, Depends(require_user)]


def to_wire(row: DatasetRow) -> Dataset:
    profile = DatasetProfile.from_json(row.profile)
    return Dataset(
        dataset_id=row.dataset_id,
        url=row.url,
        kind=row.kind,
        location=row.location,
        status=row.status,
        title=_title(row, profile),
        card=row.card,
        rows=row.rows,
        bytes=row.bytes,
        tables=len(profile.observation.tables) if profile else 0,
        error=row.error,
        built_at=row.built_at,
        created_at=row.created_at,
    )


def to_detail(row: DatasetRow) -> DatasetDetail:
    """The row plus everything measured.

    Built from the stored JSON rather than re-measuring, which is the point of
    storing it: this endpoint is what a panel opens, and opening a panel should
    not touch a DuckDB file.
    """
    profile = DatasetProfile.from_json(row.profile)
    base = to_wire(row).model_dump()
    understanding = profile.understanding if profile else None

    return DatasetDetail(
        **base,
        summary=understanding.summary if understanding else "",
        topics=list(understanding.topics) if understanding else [],
        good_for=list(understanding.good_for) if understanding else [],
        not_for=list(understanding.not_for) if understanding else [],
        schema_card=row.schema_card,
        notes=list(profile.observation.notes) if profile else [],
        tables_detail=[
            DatasetTable(
                name=table.name,
                rows=table.rows,
                total=table.total,
                sampled=table.sampled,
                path=table.path,
                omitted=[f"{o.name} ({o.megabytes:,.0f} MB)" for o in table.omitted],
                columns=[
                    DatasetColumn(
                        name=column.name,
                        type=column.type,
                        coverage=column.coverage,
                        distinct=column.distinct,
                        values=list(column.values),
                        examples=list(column.examples),
                        avg_bytes=column.avg_bytes,
                        filterable=column.filterable,
                        constant=column.constant,
                        wide=column.wide,
                    )
                    for column in table.columns
                ],
            )
            for table in (profile.observation.tables if profile else ())
        ],
    )


def to_result(answer: Answer) -> QueryResult:
    result = answer.result
    return QueryResult(
        dataset_id=answer.dataset_id,
        sql=answer.sql,
        columns=list(result.columns) if result else [],
        rows=[_jsonable(record) for record in (result.to_dicts() if result else [])],
        truncated=result.truncated if result else False,
        ms=round(answer.ms, 1),
        attempts=answer.attempts,
        error=answer.error,
    )


@router.get("", response_model=DatasetList, summary="Every dataset you have attached")
def list_datasets(service: ServiceDep, user_id: UserDep) -> DatasetList:
    rows = service.list(user_id)
    return DatasetList(
        datasets=[to_wire(row) for row in rows],
        limit=service.max_per_user,
        enabled=service.configured,
    )


@router.post(
    "",
    response_model=Dataset,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Attach a dataset by URL",
)
def add_dataset(body: AddDataset, service: ServiceDep, user_id: UserDep) -> Dataset:
    """202, not 201: the row exists but the dataset does not answer anything yet.

    The URL is resolved before this returns, so a bad one is a 400 with a
    message about the URL. The pull, the measurement and the model call all
    happen after, and the row's `status` is how the panel follows them.
    """
    try:
        row = service.add(user_id, body.url)
    except DatasetError as error:
        raise HTTPException(status_code=error.status_code, detail=str(error)) from error
    return to_wire(row)


@router.get("/{dataset_id}", response_model=DatasetDetail, summary="What was understood")
def get_dataset(dataset_id: str, service: ServiceDep, user_id: UserDep) -> DatasetDetail:
    row = service.get(user_id, dataset_id)
    if row is None:
        raise HTTPException(status_code=404, detail=f"No dataset called {dataset_id}.")
    return to_detail(row)


@router.post(
    "/{dataset_id}/rebuild",
    response_model=Dataset,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Pull and measure it again",
)
def rebuild(dataset_id: str, service: ServiceDep, user_id: UserDep) -> Dataset:
    """Re-pull under the current settings.

    Which is the escape hatch for the column budget: raising
    `dataset_column_budget_mb` changes what gets materialised, and this is how
    an already-attached dataset picks that up without being removed and re-added.
    """
    row = service.get(user_id, dataset_id)
    if row is None:
        raise HTTPException(status_code=404, detail=f"No dataset called {dataset_id}.")

    service.schedule(user_id, dataset_id)
    return to_wire(row)


@router.post("/{dataset_id}/query", response_model=QueryResult, summary="Ask it something")
def query(
    dataset_id: str,
    body: QueryRequest,
    agent: AgentDep,
    user_id: UserDep,
) -> QueryResult:
    """A question in English, or SQL. 200 even when the query failed — see above."""
    if body.sql.strip():
        return to_result(agent.run(user_id, dataset_id, body.sql))
    if not body.question.strip():
        raise HTTPException(status_code=400, detail="Send a question or some SQL.")
    return to_result(agent.ask(user_id, dataset_id, body.question))


@router.delete("/{dataset_id}", response_model=DatasetRemoved, summary="Detach a dataset")
def remove(dataset_id: str, service: ServiceDep, user_id: UserDep) -> DatasetRemoved:
    """Deletes the row and the local file. Idempotent."""
    return DatasetRemoved(dataset_id=dataset_id, removed=service.remove(user_id, dataset_id))


def _title(row: DatasetRow, profile: DatasetProfile | None) -> str:
    if profile and profile.understanding and profile.understanding.title:
        return profile.understanding.title
    return row.location or row.dataset_id


def _jsonable(record: dict[str, Any]) -> dict[str, Any]:
    """Row values FastAPI can serialise.

    DuckDB hands back nested types as Python structures, which encode fine, and
    occasionally something that does not — a `bytes` blob is the one that turns
    a working endpoint into a 500 the first time somebody attaches a dataset
    with a binary column.
    """
    clean: dict[str, Any] = {}
    for key, value in record.items():
        if isinstance(value, (str, int, float, bool, type(None), list, dict)):
            clean[key] = value
        else:
            clean[key] = str(value)
    return clean
