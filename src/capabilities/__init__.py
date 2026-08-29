"""What this person has connected, and which of it bears on the question.

The agent does not carry a description of every connected store in its prompt.
It carries one tool, `find_capability`, and this is what answers it:

    catalogue.py   every connector profile, dataset and authorised toolkit,
                   turned into `Capability` — what it is good for, and the
                   call that uses it
    index.py       the same semantic search retrieval uses, over the cards
                   instead of over passages

The reason for the indirection is in docs/23-capabilities.md: a prompt that
lists everything grows with what somebody connects, is paid for on every turn
including the ones that need none of it, and gets *recited* — a model handed a
dataset card answers from the card, and the card carries real numbers.
"""

from src.capabilities.catalogue import (
    DATASET,
    STORE,
    TOOLKIT,
    Capability,
    Catalogue,
    get_catalogue,
)
from src.capabilities.index import CapabilityIndex, Match, get_capability_index

__all__ = [
    "DATASET",
    "STORE",
    "TOOLKIT",
    "Capability",
    "CapabilityIndex",
    "Catalogue",
    "Match",
    "get_capability_index",
    "get_catalogue",
]
