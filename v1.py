#llm with tools
import sys
sys.stdout.reconfigure(encoding="utf-8")

import json
from pathlib import Path
from groq import Groq
import os
from dotenv import load_dotenv

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))
MODEL = "qwen/qwen3.8-27b"

WORKSPACE = Path(__file__).parent / "workspace"
WORKSPACE.mkdir(parents=True, exist_ok=True)


def list_files() -> str:
    """List the files in the assistant's workspace folder."""
    return "\n".join(p.name for p in WORKSPACE.iterdir()) or "(empty)"


def read_file(filename: str) -> str:
    """Read a file from the workspace folder."""
    path = (WORKSPACE / filename).resolve()

    # Prevent access to files outside the workspace
    if not path.is_relative_to(WORKSPACE.resolve()):
        return "Error: Access outside the workspace is not allowed."

    if not path.is_file():
        return f"Error: No file named {filename}"

    try:
        return path.read_text(encoding="utf-8")
    except OSError as e:
        return f"Error reading file: {e}"


TOOLS = {
    "list_files": list_files,
    "read_file": read_file,
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
                    "filename": {"type": "string"}
                },
                "required": ["filename"],
            },
        },
    },
]


def chat(user_message: str) -> str:
    messages = [
        {
            "role": "system",
            "content": "You are a helpful personal assistant. "
                       "Use the available tools when necessary.",
        },
        {"role": "user", "content": user_message},
    ]

    response = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        tools=TOOL_SCHEMAS,
    )

    message = response.choices[0].message

    # If the model doesn't request a tool, return its response
    if not message.tool_calls:
        return message.content or ""

    # Add the assistant's tool-call message to the conversation
    messages.append(message.model_dump(exclude_none=True))

    # Execute the requested tools
    for call in message.tool_calls:
        tool_name = call.function.name

        if tool_name not in TOOLS:
            result = f"Error: Unknown tool {tool_name}"
        else:
            try:
                args = json.loads(call.function.arguments or "{}")
                print(f"[tool] {tool_name}({args})")
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

    # Ask the model to generate a final response using the tool results
    final = client.chat.completions.create(
        model=MODEL,
        messages=messages,
    )

    return final.choices[0].message.content or ""


if __name__ == "__main__":
    print(f"v1 assistant ({MODEL}) - Ctrl+C to quit")

    try:
        while True:
            question = input("\nYou: ").strip()

            if not question:
                continue

            print("\nAssistant:", chat(question))

    except (EOFError, KeyboardInterrupt):
        print("\nBye!")
