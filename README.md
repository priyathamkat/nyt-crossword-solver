# Solver for NY Times Crossword
An LLM-agent-based solver for the NY Times mini crossword. Note that the algorithm in this solver is likely not very optimal, but it is by no means slow. I wrote this purely as an exercise in using LLM agents.
# Usage
## Setup
- Install `uv` package manager: [https://github.com/astral-sh/uv](https://github.com/astral-sh/uv)
- Run `uv sync`
- Add `OPENAI_API_KEY` and `GEMINI_API_KEY` to a `.env` file
- Set `UV_ENV_FILE` to `.env`:
    ```bash
    export UV_ENV_FILE=.env
    ```
## Extracting a puzzle
You can prepare a puzzle for solving by running the following command:
```bash
uv run <puzzle_screenshot> <puzzle_grid> <output_folder>
```
See the [puzzles/images](puzzles/images/) folder for some example screenshots. Grid structure is a string denoting the shape of the grid. Starting from the top and going from left to right, indicate a black square with `#` and an empty square with `_`. For example, for the puzzle in [puzzles/images/puzzle-2025-02-17.png](puzzles/images/puzzle-2025-02-17.png), the grid structure is: `____________________#___#`. You can set output folder to [puzzles/extracted](puzzles/extracted/) if you desire.
## Solving a puzzle
To solve a puzzle, simply run:
```bash
uv run nyt_crossword_solver/solve.py <puzzle_json> <max_improvement_steps>
```
`puzzle_json` is the path to the `json` file extracted in the previous step. The solver first gets a list of candidate answers for each clue and then attemps to improve on it by first tackling the candidates which have low "consistency" with intersecting candidates. You can set `max_improvement_steps` to how many attemps you want the solver to make when trying to fix these inconsistent candidates. I recommend 10.
# Credits
All the screenshots of puzzles included in this repo are from [NY Times](https://www.nytimes.com/crosswords/game/mini).
# License
All the code in this repo is available under a permissive MIT License. Please use the following BibTex entry if you wish to cite this repo:
```
@misc{priyathamkat/nyt_crossword_solver,
  author       = {Priyatham Kattakinda},
  title        = {NYT Crossword Solver},
  year         = {2025},
  publisher    = {GitHub},
  journal      = {GitHub repository},
  howpublished = {\url{https://github.com/priyathamkat/nyt-crossword-solver}},
  note         = {Accessed: 2025-02-17}
}
```
