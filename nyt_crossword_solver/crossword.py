from typing import Literal, Optional

from pydantic import BaseModel


class Clue(BaseModel):
    position: tuple[int, int]
    clue: str
    length: int
    answer: Optional[str] = None
    consistency_score: Optional[float] = None


class MiniCrossword(BaseModel):
    grid: list[str]
    across: list[Clue]
    down: list[Clue]


def get_spanning_cells(clue: Clue, orientation: Literal["across", "down"]) -> list[tuple[int, int]]:
    """
    Get the cells that the answer of the clue spans.

    Args:
        clue (Clue): The clue.
        orientation (Literal["across", "down"]): The orientation of the clue.

    Returns:
        list[tuple[int, int]]: The cells that the answer of the clue spans.
    """
    if orientation == "across":
        return [(clue.position[0], clue.position[1] + i) for i in range(clue.length)]
    else:
        return [(clue.position[0] + i, clue.position[1]) for i in range(clue.length)]


def get_intersecting_clues(crossword: MiniCrossword, orientation: str, idx: int) -> list[tuple[int, int, Clue]]:
    """
    Get the intersecting clues of the answer at a given position.

    Args:
        crossword (MiniCrossword): The crossword puzzle.
        orientation (str): The orientation of the answer.
        idx (int): The index of the answer.

    Returns:
        list[tuple[int, int, Clue]]: The intersecting clues of the answer. Each tuple contains the position of the
            intersecting cell in the answer, the position of the intersecting cell in the intersecting clue, and
            the intersecting clue.
    """
    other_orientation = "across" if orientation == "down" else "down"
    candidate_clue: Clue = getattr(crossword, orientation)[idx]
    candidate_spanning_cells = get_spanning_cells(candidate_clue, orientation)
    intersecting_clues = []
    for other_clue in getattr(crossword, other_orientation):
        other_spanning_cells = get_spanning_cells(other_clue, other_orientation)
        for pos, cell in enumerate(candidate_spanning_cells):
            try:
                other_idx = other_spanning_cells.index(cell)
                intersecting_clues.append((pos, other_idx, other_clue))
            except ValueError:
                pass
    return intersecting_clues


def consistency_score(crossword: MiniCrossword, orientation: str, idx: int) -> float:
    """
    Calculate the consistency score of the answer at a given position based on its length and intersecting candidates.

    Args:
        crossword (MiniCrossword): The crossword puzzle.
        orientation (str): The orientation of the answer.
        idx (int): The index of the answer.

    Returns:
        float: The consistency score of the answer.
    """
    score = 0
    candidate_clue: Clue = getattr(crossword, orientation)[idx]
    candidate_answer = candidate_clue.answer
    if len(candidate_answer) != candidate_clue.length:
            return 0
    delta = 1 / candidate_clue.length
    intersecting_clues = get_intersecting_clues(crossword, orientation, idx)
    for pos, other_pos, other_clue in intersecting_clues:
            try:
                if candidate_answer[pos] == other_clue.answer[other_pos]:
                    score += delta
            except IndexError:  # Answer is too short
                pass
    return round(score, 2)
