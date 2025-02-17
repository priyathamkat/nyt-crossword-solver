from autogen_agentchat.agents import AssistantAgent
from autogen_ext.models.openai import OpenAIChatCompletionClient
from pydantic import BaseModel

from nyt_crossword_solver.tools import answer_len, ask_oracle, get_nth_character


class CandidatesGeneratorFormat(BaseModel):
    candidates: list[str]


candidates_generator_model_client = OpenAIChatCompletionClient(
    model="gpt-4o",
    response_format=CandidatesGeneratorFormat,
)
candidates_generator_system_prompt = (
    """Generate a list of candidate answers for the crossword clue of given length. If the clue ends in a question """
    """mark, it's  tricky clue that needs out-of-the-box thinking to answer, such as interpreting a word(s) in the """
    """clue literally. Replace any accented characters with their unaccented equivalents (e.g., e instead of é)."""
    """Answer may contain multiple words."""
    """Available tools:\n"""
    """`answer_len`: Determine the length of a candidate answer. If the output of this tool does not match the """
    """given length, try to generate other more appropriate candidates and filter out the candidates with """
    """incorrect length.\n"""
    """`ask_oracle`: Get accurate answers to well-formed, grammatically correct questions. """
    """Use this tool when you need accurate and up-to-date information.\n"""
    """`get_nth_character`: Get the nth character of a candidate answer.\n"""
)


def candidates_generator_factory():
    """Create a candidates generator agent."""
    return AssistantAgent(
        name="candidates_generator",
        model_client=candidates_generator_model_client,
        tools=[answer_len, ask_oracle, get_nth_character],
        system_message=candidates_generator_system_prompt,
        reflect_on_tool_use=True,
    )
