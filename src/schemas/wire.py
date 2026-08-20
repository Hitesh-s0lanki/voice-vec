"""The base every wire model shares.

Field names go over the wire in camelCase because the client is TypeScript;
Python keeps snake_case on this side.
"""

from pydantic import BaseModel, ConfigDict
from pydantic.alias_generators import to_camel


class Wire(BaseModel):
    model_config = ConfigDict(alias_generator=to_camel, populate_by_name=True)
