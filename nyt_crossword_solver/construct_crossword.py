import argparse
import asyncio
import json
from pathlib import Path

from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.messages import MultiModalMessage
from autogen_core import CancellationToken, Image
from autogen_ext.models.openai import OpenAIChatCompletionClient
from PIL import Image as PILImage

from nyt_crossword_solver.crossword import MiniCrossword


async def construct_from_image(image: PILImage) -> MiniCrossword:
    """Construct a crossword puzzle from an image.

    Args:
        image (Image): The image to analyze.

    Returns:
        MiniCrossword: The constructed crossword puzzle.
    """

    image = Image(image)
    model_client = OpenAIChatCompletionClient(
        model="gpt-4o",
        response_format=MiniCrossword,
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
    return response.chat_message.content


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Construct a crossword puzzle from an image.")
    parser.add_argument("image_path", type=str, help="The path to the image to analyze.")
    parser.add_argument("output_path", type=str, help="The path to save the constructed crossword puzzle.")
    args = parser.parse_args()

    image_path = Path(args.image_path)
    output_path = Path(args.output_path)
    output_path = output_path / f"{image_path.stem}.json"

    image = PILImage.open(str(image_path))
    crossword = asyncio.run(construct_from_image(image))
    crossword = json.loads(crossword)
    if MiniCrossword.model_validate(crossword):
        print("The crossword puzzle was successfully constructed.")
    else:
        print("Failed to construct a proper crossword puzzle.")
    with open(output_path, "w") as f:
        json.dump(crossword, f, indent=4)
