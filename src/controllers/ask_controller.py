"""HTTP surface for the RAG pipeline."""

from typing import Annotated

from fastapi import APIRouter, Depends, Header

from src.core.clerk import Verifier, get_verifier
from src.schemas.ask import AskRequest, AskResponse
from src.services.ask_service import AskService, get_ask_service

router = APIRouter(prefix="/ask", tags=["ask"])

AskServiceDep = Annotated[AskService, Depends(get_ask_service)]


def maybe_user(
    verifier: Annotated[Verifier, Depends(get_verifier)],
    authorization: Annotated[str | None, Header(description="Bearer <clerk token>")] = None,
) -> str | None:
    """Who is asking, if anybody. Signed out is a first-class answer here.

    Unlike `/integrations`, this route does not require an account — it
    answered anonymously before connectors existed and still does. The identity
    only decides *which vector store* answers: a user who connected Pinecone is
    searched against theirs, everybody else against the deployment's.

    So an unverified token is not an error, it is anonymity, and the worst it
    can cost somebody is being answered from the default corpus.
    """
    scheme, _, token = (authorization or "").partition(" ")
    return verifier.user_id(token if scheme.lower() == "bearer" else None)


UserDep = Annotated[str | None, Depends(maybe_user)]


# Deliberately `def`, not `async def`: the pipeline is synchronous CPU work
# (ONNX forward pass, vector search), so FastAPI runs it in the threadpool and
# one slow answer cannot stall the event loop.
@router.post("", response_model=AskResponse, summary="Answer a transcript from the corpus")
def ask(request: AskRequest, service: AskServiceDep, user_id: UserDep) -> AskResponse:
    """Transcript in, grounded answer or honest abstention out.

    Never raises for a rejected question: `status` carries the outcome and
    HTTP 200 means the pipeline ran. `abstained` is a success.
    """
    return service.ask(request, user_id=user_id)
