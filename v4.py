#persistant memory added
import sys
sys.stdout.reconfigure(encoding="utf-8")

import json
import os
from pathlib import Path

from groq import Groq
from dotenv import load_dotenv

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))
MODEL = "qwen/qwen3.8-27b"

WORKSPACE = Path(__file__).parent / "workspace"
WORKSPACE.mkdir(parents=True, exist_ok=True)

MEMORY_FILE = Path(__file__).parent / "memory.md"


# --- memory: a markdown file, that's it ---

def load_memory() -> str:
    if MEMORY_FILE.is_file():
        return MEMORY_FILE.read_text(encoding="utf-8")
    return "(nothing saved yet)"


def save_memory(fact: str) -> str:
    """Append one fact about the user to long-term memory."""
    try:
        with MEMORY_FILE.open("a", encoding="utf-8") as f:
            f.write(f"- {fact}\n")
        return f"Saved: {fact}"
    except OSError as e:
        return f"Error saving memory: {e}"


# --- tools ---

def list_files() -> str:
    return "\n".join(p.name for p in WORKSPACE.iterdir()) or "(empty)"


def read_file(filename: str) -> str:
    path = (WORKSPACE / filename).resolve()

    if not path.is_relative_to(WORKSPACE.resolve()):
        return "Error: Access outside the workspace is not allowed."

    try:
        return (
            path.read_text(encoding="utf-8")
            if path.is_file()
            else f"Error: No file named {filename}"
        )
    except OSError as e:
        return f"Error reading file: {e}"


TOOLS = {
    "list_files": list_files,
    "read_file": read_file,
    "save_memory": save_memory,
}

TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "list_files",
            "description": "List the files in the user's workspace folder.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read one file from the user's workspace folder.",
            "parameters": {
                "type": "object",
                "properties": {
                    "filename": {"type": "string"},
                },
                "required": ["filename"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "save_memory",
            "description": (
                "Save one short fact about the user to long-term memory. "
                "Use whenever you learn something worth remembering: their "
                "name, preferences, projects, or recurring tasks."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "fact": {"type": "string"},
                },
                "required": ["fact"],
            },
        },
    },
]


SYSTEM_PROMPT = """You are a helpful personal assistant.

Here is what you remember about the user from previous sessions:
{memory}

When you learn a new lasting fact about the user, save it with save_memory."""


def run_agent(messages: list) -> str:
    while True:
        response = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            tools=TOOL_SCHEMAS,
        )

        message = response.choices[0].message

        if not message.tool_calls:
            answer = message.content or ""
            messages.append(
                {"role": "assistant", "content": answer}
            )
            return answer

        # Preserve the assistant's tool-call message
        messages.append(message.model_dump(exclude_none=True))

        for call in message.tool_calls:
            tool_name = call.function.name

            try:
                args = json.loads(call.function.arguments or "{}")
                print(f"  [tool] {tool_name}({args})")

                if tool_name not in TOOLS:
                    result = f"Error: Unknown tool {tool_name}"
                else:
                    result = TOOLS[tool_name](**args)

            except (json.JSONDecodeError, TypeError, OSError) as e:
                result = f"Tool error: {e}"

            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": call.id,
                    "content": str(result),
                }
            )


if __name__ == "__main__":
    # Conversation history persists across turns.
    messages = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT.format(memory=load_memory()),
        }
    ]

    print(f"v4 assistant ({MODEL}) - memory loaded - Ctrl+C to quit")

    try:
        while True:
            question = input("\nYou: ").strip()

            if not question:
                continue

            messages.append({"role": "user", "content": question})
            print("\nAssistant:", run_agent(messages))

    except (EOFError, KeyboardInterrupt):
        print("\nBye!")
