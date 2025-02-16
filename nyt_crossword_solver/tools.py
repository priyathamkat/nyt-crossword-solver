import functools
import os

from autogen_core.tools import FunctionTool
from google import genai
from google.genai import types as genai_types

_google_genai_client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
_google_generate_content_config = genai_types.GenerateContentConfig(
    tools=[genai_types.Tool(google_search=genai_types.GoogleSearch())], response_modalities=["TEXT"]
)


def nyt_crossword_tool(fn: callable):
    description = fn.__doc__.split("Args:")[0]  # Use everything before the Args section as the description.

    @functools.wraps(fn)
    async def wrapper(*args, **kwargs):
        return await fn(*args, **kwargs)

    return FunctionTool(wrapper, description=description, strict=True)


@nyt_crossword_tool
async def answer_len(answer: str) -> int:
    """Return the length of answer to a clue. Does not count special characters including spaces.

    Args:
        answer (str): The answer to a clue.

    Returns:
        int: The length of the answer.
    """
    return len([char for char in answer if char.isalpha()])


@nyt_crossword_tool
async def ask_oracle(query: str) -> str:
    """Find accurate answers to a query.

    Args:
        query (str): The query to search the for.

    Returns:
        str: The answer to the query.
    """
    prompt = (
        f"""Always search the web and answer the following question."""
        f"""ONLY give the answer WITHOUT any exta details in as few words as possible:\nQuery: {query}"""
    )
    response = _google_genai_client.models.generate_content(
        model="gemini-2.0-flash",
        contents=prompt,
        config=_google_generate_content_config,
    )
    return response.candidates[0].content.parts[-1].text
