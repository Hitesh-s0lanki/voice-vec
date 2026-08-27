"""HTTP surface for conversations.

Who is asking arrives on every route, because every route needs it and none of
them should be able to forget it:

    Authorization: Bearer …   a Clerk session token, verified here. Its `sub`
                              is the account, and nothing else is taken as one.
    x-session-id              the browser's own `sess_…`, minted client-side
                              and kept in localStorage — what an anonymous
                              visitor is known by

The account half is a *token* rather than a user id header for the reason any
id-in-a-header scheme fails: anyone can type one. The signature is what makes
`sub` mean something. The browser id is not verified and does not need to be —
it grants exactly what holding that browser already grants.

`identify` turns the two into an `Owner`, and the store's SQL matches a row on
either column, so signing in widens what you can see rather than hiding what
you just said. A route that cannot name an owner returns nothing rather than
everything.

Writing goes the other way: the voice socket appends messages as they are
spoken, so nothing here creates a message. `POST /conversations` exists for a
client that wants a URL before it opens a socket; the socket is happy to open
one itself.
"""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, Header, HTTPException, Query, Response, status

from src.chat.store import Owner
from src.core.clerk import Verifier, get_verifier
from src.schemas.chat import (
    Adopted,
    AdoptConversations,
    ConversationDetail,
    ConversationList,
    ConversationSummary,
    CreateConversation,
    RenameConversation,
)
from src.services.conversation_service import (
    ConversationService,
    get_conversation_service,
)

router = APIRouter(prefix="/conversations", tags=["conversations"])

ServiceDep = Annotated[ConversationService, Depends(get_conversation_service)]


def identify(
    verifier: Annotated[Verifier, Depends(get_verifier)],
    authorization: Annotated[str | None, Header(description="Bearer <clerk token>")] = None,
    x_session_id: Annotated[str | None, Header(description="Anonymous browser id")] = None,
) -> Owner:
    """The caller, as far as anyone here is concerned.

    A token that does not verify is not an error — it makes the caller
    anonymous, which is a state this API serves perfectly well. Anything else
    would turn an expired token (Clerk's live about a minute) into a failure in
    front of someone who is signed in and simply idled.

    The browser id is trimmed and length-capped: it lands in an indexed column,
    and a header is whatever the client felt like sending.
    """
    scheme, _, token = (authorization or "").partition(" ")

    return Owner(
        user_id=verifier.user_id(token if scheme.lower() == "bearer" else None),
        session_id=(x_session_id or "").strip()[:128] or None,
    )


OwnerDep = Annotated[Owner, Depends(identify)]


@router.post(
    "",
    response_model=ConversationSummary,
    status_code=status.HTTP_201_CREATED,
    summary="Open a conversation",
)
def create_conversation(
    service: ServiceDep, owner: OwnerDep, body: CreateConversation | None = None
) -> ConversationSummary:
    if not owner.known:
        raise HTTPException(
            status_code=400, detail="Send x-session-id (or x-user-id) to open a conversation."
        )

    body = body or CreateConversation()
    return service.create(owner, title=body.title, language=body.language)


@router.get("", response_model=ConversationList, summary="Your conversations, newest first")
def list_conversations(
    service: ServiceDep,
    owner: OwnerDep,
    limit: Annotated[int, Query(ge=1, le=100)] = 30,
) -> ConversationList:
    return service.list(owner, limit=limit)


@router.post(
    "/adopt",
    response_model=Adopted,
    summary="Claim this browser's conversations for the signed-in account",
)
def adopt_conversations(
    body: AdoptConversations,
    owner: OwnerDep,
    service: ServiceDep,
) -> Adopted:
    """Called once after signing in.

    Declared before `/{conversation_id}` because FastAPI matches routes in
    order and `adopt` is a perfectly good conversation id as far as a path
    parameter is concerned.
    """
    if not owner.user_id:
        raise HTTPException(status_code=401, detail="Sign in first.")

    return Adopted(moved=service.adopt(owner.user_id, body.session_id))


@router.get("/{conversation_id}", response_model=ConversationDetail, summary="One thread")
def read_conversation(
    conversation_id: str, service: ServiceDep, owner: OwnerDep
) -> ConversationDetail:
    """404 covers both "no such conversation" and "not yours" — see the store."""
    detail = service.read(conversation_id, owner)
    if detail is None:
        raise HTTPException(status_code=404, detail="No such conversation.")
    return detail


@router.patch("/{conversation_id}", response_model=ConversationSummary, summary="Rename")
def rename_conversation(
    conversation_id: str, body: RenameConversation, service: ServiceDep, owner: OwnerDep
) -> ConversationSummary:
    summary = service.rename(conversation_id, owner, body.title)
    if summary is None:
        raise HTTPException(status_code=404, detail="No such conversation.")
    return summary


@router.delete(
    "/{conversation_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete a conversation and its messages",
)
def delete_conversation(conversation_id: str, service: ServiceDep, owner: OwnerDep) -> Response:
    if not service.delete(conversation_id, owner):
        raise HTTPException(status_code=404, detail="No such conversation.")
    return Response(status_code=status.HTTP_204_NO_CONTENT)
