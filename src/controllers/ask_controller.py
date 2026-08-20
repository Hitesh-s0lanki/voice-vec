"""HTTP surface for the RAG pipeline."""

from typing import Annotated

from fastapi import APIRouter, Depends

from src.schemas.ask import AskRequest, AskResponse
from src.services.ask_service import AskService, get_ask_service

router = APIRouter(prefix="/ask", tags=["ask"])

AskServiceDep = Annotated[AskService, Depends(get_ask_service)]


# Deliberately `def`, not `async def`: the pipeline is synchronous CPU work
# (ONNX forward pass, vector search), so FastAPI runs it in the threadpool and
# one slow answer cannot stall the event loop.
@router.post("", response_model=AskResponse, summary="Answer a transcript from the corpus")
def ask(request: AskRequest, service: AskServiceDep) -> AskResponse:
    """Transcript in, grounded answer or honest abstention out.

    Never raises for a rejected question: `status` carries the outcome and
    HTTP 200 means the pipeline ran. `abstained` is a success.
    """
    return service.ask(request)
