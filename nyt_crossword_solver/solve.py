import argparse
import asyncio
import json
from pathlib import Path

from autogen_agentchat.messages import TextMessage
from autogen_core import CancellationToken

from nyt_crossword_solver.agents import candidates_generator
from nyt_crossword_solver.crossword import MiniCrossword


async def main():
    parser = argparse.ArgumentParser(description="Generate candidate answers for a crossword clue.")
    parser.add_argument("puzzle_path", type=str, help="The path to the crossword puzzle JSON file.")

    puzzle_path = Path(parser.parse_args().puzzle_path)

    with open(puzzle_path) as f:
        puzzle = json.load(f)
    crossword = MiniCrossword(**puzzle)
    for clue in crossword.clues:
        response = await candidates_generator.on_messages(
            [TextMessage(content=f"""Clue: {clue.clue}, Length: {clue.length}""", source="User")],
            cancellation_token=CancellationToken(),
        )
        candidates = json.loads(response.chat_message.content)["candidates"]
        print(f"Clue: {clue.clue} -------- Candidates: {", ".join(candidates)}")


asyncio.run(main())
