from autogen_agentchat.agents import AssistantAgent
from autogen_ext.models.openai import OpenAIChatCompletionClient
from pydantic import BaseModel

from nyt_crossword_solver.tools import answer_len, ask_oracle, get_nth_character


class ContextFormat(BaseModel):
    context: str


context_model_client = OpenAIChatCompletionClient(
    model="gpt-4o",
    response_format=ContextFormat,
)
context_system_prompt = (
    """Analyze the clue and using the `ask_oracle` tool, generate a summary of relevant context that can be used """
    """to answer a crossword clue. Make sure to provide information that is very current or is unlikely to be """
    """known by an LLM without access to the internet."""
    """`ask_oracle`: Get accurate answers to well-formed, grammatically correct questions."""
)


def context_agent_factory():
    """Create a context agent."""
    return AssistantAgent(
        name="context",
        model_client=context_model_client,
        tools=[ask_oracle],
        system_message=context_system_prompt,
        reflect_on_tool_use=True,
    )


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
    """`get_nth_character`: Get the nth character of a candidate answer.\n"""
)


def candidates_generator_factory():
    """Create a candidates generator agent."""
    return AssistantAgent(
        name="candidates_generator",
        model_client=candidates_generator_model_client,
        tools=[answer_len, get_nth_character],
        system_message=candidates_generator_system_prompt,
        reflect_on_tool_use=True,
    )


class CorrectnessFormat(BaseModel):
    is_correct: bool


correctness_model_client = OpenAIChatCompletionClient(
    model="gpt-4o",
    response_format=CorrectnessFormat,
)
correctness_system_prompt = (
    """Determine if the candidate answer to a crossword clue is likely to be correct. If the clue ends in a """
    """question  mark, it's  tricky clue that needs out-of-the-box thinking to answer, such as interpreting a """
    """word(s) in the clue literally."""
)


def correctness_agent_factory():
    """Create a correctness agent."""
    return AssistantAgent(
        name="correctness",
        model_client=correctness_model_client,
        system_message=correctness_system_prompt,
    )
