import argparse
import asyncio
import json
from math import sqrt
from pathlib import Path
from typing import Literal

from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.messages import MultiModalMessage
from autogen_core import CancellationToken, Image
from autogen_ext.models.openai import OpenAIChatCompletionClient
from PIL import Image as PILImage
from pydantic import BaseModel

from nyt_crossword_solver.crossword import Clue, MiniCrossword


class ExtractedClueFormat(BaseModel):
    orientation: Literal["across", "down"]
    clue: str


class ExtractedCluesFormat(BaseModel):
    clues: list[ExtractedClueFormat]


async def extract_clues_from_image(image: PILImage) -> list[ExtractedClueFormat]:
    """Extract clues from an image of a crossword puzzle.

    Args:
        image (Image): The image to analyze.

    Returns:
        list[ExtractedClueFormat]: The extracted clues.
    """

    image = Image(image)
    model_client = OpenAIChatCompletionClient(
        model="gpt-4o",
        response_format=ExtractedCluesFormat,
    )

    system_message = (
        """Analyze the image of a crossword puzzle and extract the clues including the orientation (across or down),"""
        """ position, clue, and length of each answer. Instructions for determining the answer length:\n"""
        """    1. Locate the grid position corresponding to the clue number.\n"""
        """    2. Count the number of consecutive empty squares starting from this position. You must count in """
        """the direction specified by the clue's orientation (Across → Right, Down → Downward). \n"""
        """    3. Stop counting when reaching either a black (blocked) square or the edge of the puzzle grid."""
    )
    agent = AssistantAgent(name="crossword_clues_extractor", model_client=model_client, system_message=system_message)
    response = await agent.on_messages(
        [MultiModalMessage(content=[image], source="User")], cancellation_token=CancellationToken()
    )
    return ExtractedCluesFormat(**json.loads(response.chat_message.content)).clues


async def construct_from_image(image: PILImage, grid_structure: str) -> MiniCrossword:
    """Construct a crossword puzzle from an image.

    Args:
        image (Image): The image to analyze.

    Returns:
        MiniCrossword: The constructed crossword puzzle.
    """

    clues = await extract_clues_from_image(image)
    across_clues = [clue for clue in clues if clue.orientation == "across"]
    down_clues = [clue for clue in clues if clue.orientation == "down"]

    grid_size = int(sqrt(len(grid_structure)))
    grid = [grid_structure[i * grid_size : (i + 1) * grid_size] for i in range(grid_size)]

    crossword = MiniCrossword(grid=grid, across=[], down=[])

    current_across_clue_idx = 0
    current_down_clue_idx = 0
    for i in range(grid_size):
        for j in range(grid_size):
            if grid[i][j] == "#":
                continue

            if j == 0 or grid[i][j - 1] == "#":  # Potential start of an across clue
                length_across = 0
                while j + length_across < grid_size and grid[i][j + length_across] != "#":
                    length_across += 1
                if length_across > 2:  # Minimum length of a word is 3
                    crossword.across.append(
                        Clue(
                            position=(i, j),
                            clue=across_clues[current_across_clue_idx].clue,
                            length=length_across,
                        )
                    )
                    current_across_clue_idx += 1

            if i == 0 or grid[i - 1][j] == "#":  # Potential start of a down clue
                length_down = 0
                while i + length_down < grid_size and grid[i + length_down][j] != "#":
                    length_down += 1
                if length_down > 2:  # Minimum length of a word is 3
                    crossword.down.append(
                        Clue(
                            position=(i, j),
                            clue=down_clues[current_down_clue_idx].clue,
                            length=length_down,
                        )
                    )
                    current_down_clue_idx += 1

    return crossword


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Construct a crossword puzzle from an image.")
    parser.add_argument("image_path", type=str, help="Path to the image to analyze.")
    parser.add_argument(
        "grid_structure",
        type=str,
        help=(
            """Structure of the crossword grid as a string. Left to right, top to bottom."""
            """Use '#' for black squares and _ for white squares."""
        ),
    )
    parser.add_argument("output_path", type=str, help="Path to save the constructed crossword puzzle.")
    args = parser.parse_args()

    image_path = Path(args.image_path)
    output_path = Path(args.output_path)
    output_path = output_path / f"{image_path.stem}.json"

    image = PILImage.open(str(image_path))
    crossword = asyncio.run(construct_from_image(image, args.grid_structure))
    crossword = crossword.model_dump()
    with open(output_path, "w") as f:
        json.dump(crossword, f, indent=4)
