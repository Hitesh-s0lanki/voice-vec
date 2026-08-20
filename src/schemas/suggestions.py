"""What this index can actually answer."""

from pydantic import BaseModel, ConfigDict, Field
from pydantic.alias_generators import to_camel


class Suggestion(BaseModel):
    model_config = ConfigDict(alias_generator=to_camel, populate_by_name=True)

    query: str
    query_type: str
    confidence: float


class SuggestionsResponse(BaseModel):
    model_config = ConfigDict(alias_generator=to_camel, populate_by_name=True)

    suggestions: list[Suggestion] = Field(
        description="Questions verified to come back answered against this index"
    )
    corpus: str = Field(description="What the corpus is, in one line, for the UI to show")
