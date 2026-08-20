"""HTTP surface for in-corpus question suggestions."""

from typing import Annotated

from fastapi import APIRouter, Depends, Query

from src.schemas.suggestions import SuggestionsResponse
from src.services.suggestions_service import (
    SuggestionsService,
    get_suggestions_service,
)

router = APIRouter(prefix="/suggestions", tags=["suggestions"])

SuggestionsServiceDep = Annotated[SuggestionsService, Depends(get_suggestions_service)]


@router.get("", response_model=SuggestionsResponse, summary="Questions this index can answer")
def read_suggestions(
    service: SuggestionsServiceDep,
    limit: Annotated[int, Query(ge=1, le=20)] = 4,
) -> SuggestionsResponse:
    """A rotating sample of questions the corpus demonstrably answers.

    Without this the first question anyone asks is one the corpus has never
    heard of, and a correct abstention reads as a broken system.
    """
    return service.sample(limit)
