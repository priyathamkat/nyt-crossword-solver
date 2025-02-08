from typing import Literal

from pydantic import BaseModel


class Clue(BaseModel):
    orientation: Literal["across", "down"]
    position: int
    clue: str
    length: int


class MiniCrossword(BaseModel):
    clues: list[Clue]
