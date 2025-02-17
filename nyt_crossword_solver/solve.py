import argparse
import asyncio
import json
from copy import deepcopy
from itertools import product
from pathlib import Path

from autogen_agentchat.agents import UserProxyAgent
from autogen_agentchat.teams import RoundRobinGroupChat
from tqdm import trange

from nyt_crossword_solver.agents import candidates_generator_factory
from nyt_crossword_solver.crossword import MiniCrossword, consistency_score, get_intersecting_clues
from nyt_crossword_solver.tools import filter_invalid_characters


async def generate_candidates(
    clue: str, length: int, exclude: list[str] = None, intersections: list[tuple[int, str]] = None
) -> list[str]:
    """Generate candidate answers for a crossword clue.

    Args:
        clue (str): The crossword clue.
        length (int): The length of the answer.
        exclude (list[str], optional): The list of candidates to exclude. Defaults to None.

    Returns:
        list[str]: The candidate answers.
    """
    instruction = f"""Clue: {clue}, Length: {length}."""
    if intersections:
        for intersection in intersections:
            instruction += (
                f""" The character at position {intersection[0] + 1} in the answer is likely to be {intersection[1]}."""
            )
    if exclude:
        instruction += f""" Exclude the following candidates: {", ".join(exclude)}."""
    instruction_agent = UserProxyAgent(
        name="instruction_agent",
        input_func=lambda _: instruction,
    )
    candidates_generator = candidates_generator_factory()
    team = RoundRobinGroupChat([instruction_agent, candidates_generator], max_turns=2)
    result = await team.run()
    return json.loads(result.messages[-1].content)["candidates"]


def best_solution(
    crossword: MiniCrossword, across_candidates: list[list[str]], down_candidates: list[list[str]]
) -> MiniCrossword:
    """Find the best solution for a crossword puzzle.

    Args:
        crossword (MiniCrossword): The crossword puzzle.
        across_candidates (list[list[str]]): The candidate answers for the across clues.
        down_candidates (list[list[str]]): The candidate answers for the down clues.

    Returns:
        MiniCrossword: The best solution for the crossword puzzle.
    """
    best_score = 0
    best_crossword = None
    for sample_across_candidates in product(*across_candidates):
        for sample_down_candidates in product(*down_candidates):
            score = 0
            for i, clue in enumerate(crossword.across):
                clue.answer = filter_invalid_characters(sample_across_candidates[i])
            for i, clue in enumerate(crossword.down):
                clue.answer = filter_invalid_characters(sample_down_candidates[i])
            for i, clue in enumerate(crossword.across):
                clue_consistency_score = consistency_score(crossword, "across", i)
                clue.consistency_score = clue_consistency_score
                score += clue_consistency_score
            for i, clue in enumerate(crossword.down):
                clue_consistency_score = consistency_score(crossword, "down", i)
                clue.consistency_score = clue_consistency_score
                score += clue_consistency_score
            if score > best_score:
                best_score = score
                best_crossword = deepcopy(crossword)
    return best_crossword


async def main():
    parser = argparse.ArgumentParser(description="Generate candidate answers for a crossword clue.")
    parser.add_argument("puzzle_path", type=str, help="The path to the crossword puzzle JSON file.")
    parser.add_argument("max_improvements", type=int, help="The maximum number of improvements to make.")

    args = parser.parse_args()

    puzzle_path = Path(args.puzzle_path)

    with open(puzzle_path) as f:
        puzzle = json.load(f)
    crossword = MiniCrossword(**puzzle)

    print("Generating candidate answers for the crossword clues...", end="", flush=True)
    across_tasks = [generate_candidates(clue.clue, clue.length) for clue in crossword.across]
    across_candidates = await asyncio.gather(*across_tasks)
    down_tasks = [generate_candidates(clue.clue, clue.length) for clue in crossword.down]
    down_candidates = await asyncio.gather(*down_tasks)
    print(" Done.")

    for i in trange(args.max_improvements, desc="Improving solution"):
        best_crossword = best_solution(crossword, across_candidates, down_candidates)

        worst_score = 1
        worst_clue_orientation = None
        worst_clue_idx = None

        for i, clue in enumerate(best_crossword.across):
            if clue.consistency_score < worst_score:
                worst_score = clue.consistency_score
                worst_clue_orientation = "across"
                worst_clue_idx = i
        for i, clue in enumerate(best_crossword.down):
            if clue.consistency_score < worst_score:
                worst_score = clue.consistency_score
                worst_clue_orientation = "down"
                worst_clue_idx = i

        if worst_score == 1:
            break
        else:
            if worst_clue_orientation == "across":
                intersections = get_intersecting_clues(best_crossword, "across", worst_clue_idx)
                intersections_to_consider = []
                for intersection in intersections:
                    if intersection[2].consistency_score > 0.5:
                        try:
                            intersections_to_consider.append((intersection[0], intersection[2].answer[intersection[1]]))
                        except IndexError:  # Other answer is too short
                            pass
                candidates = await generate_candidates(
                    best_crossword.across[worst_clue_idx].clue,
                    best_crossword.across[worst_clue_idx].length,
                    exclude=across_candidates[worst_clue_idx],
                    intersections=intersections_to_consider,
                )
                across_candidates[worst_clue_idx].extend(candidates)
            else:
                intersections = get_intersecting_clues(best_crossword, "across", worst_clue_idx)
                intersections_to_consider = []
                for intersection in intersections:
                    if intersection[2].consistency_score > 0.5:
                        try:
                            intersections_to_consider.append((intersection[0], intersection[2].answer[intersection[1]]))
                        except IndexError:  # Other answer is too short
                            pass
                candidates = await generate_candidates(
                    best_crossword.down[worst_clue_idx].clue,
                    best_crossword.down[worst_clue_idx].length,
                    exclude=down_candidates[worst_clue_idx],
                    intersections=intersections_to_consider,
                )
                down_candidates[worst_clue_idx].extend(candidates)

    for clue in best_crossword.across + best_crossword.down:
        print(f"Clue: {clue.clue} -------- Answer: {clue.answer} -------- Consistency Score: {clue.consistency_score}")


asyncio.run(main())
