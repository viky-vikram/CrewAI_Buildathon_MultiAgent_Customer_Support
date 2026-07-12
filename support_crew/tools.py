"""CrewAI tools. The file-saving tool is assigned ONLY to the Entry Agent."""

from crewai.tools import tool

from . import config, storage


@tool("Save Support Record")
def save_support_record(query: str, assistant_answer: str, web_search_answer: str) -> str:
    """Append a customer-support record to answers.txt using UTF-8 encoding.

    Pass the ORIGINAL user query, the COMPLETE unmodified Assistant answer,
    and the COMPLETE unmodified Web Search Assistant answer. The tool writes
    them in a fixed, readable format and never overwrites earlier records.
    """
    record_id = storage.append_record(query, assistant_answer, web_search_answer)
    return (
        f"Record {record_id} successfully appended to "
        f"{config.ANSWERS_FILE.name} at {config.ANSWERS_FILE}"
    )
