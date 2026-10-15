#mcp added
import sys
sys.stdout.reconfigure(encoding="utf-8")

import asyncio
import json
import os
from contextlib import AsyncExitStack
from pathlib import Path

from dotenv import load_dotenv
from groq import AsyncGroq
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

load_dotenv()

client = AsyncGroq(api_key=os.getenv("GROQ_API_KEY"))
MODEL = "qwen/qwen3.8-27b"

WORKSPACE = Path(__file__).parent / "workspace"
WORKSPACE.mkdir(parents=True, exist_ok=True)

# Each server is just a command to launch
MCP_SERVERS = {
    "time": StdioServerParameters(
        command="uvx",
        args=["mcp-server-time"],
    ),
    "fetch": StdioServerParameters(
        command="uvx",
        args=["mcp-server-fetch"],
    ),
}


# --- Local tools ---

def list_files() -> str:
    return "\n".join(p.name for p in WORKSPACE.iterdir()) or "(empty)"


def read_file(filename: str) -> str:
    path = (WORKSPACE / filename).resolve()

    if not path.is_relative_to(WORKSPACE.resolve()):
        return "Error: Access outside the workspace is not allowed."

    return (
        path.read_text(encoding="utf-8")
        if path.is_file()
        else f"Error: No file named {filename}"
    )


LOCAL_TOOLS = {
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
                    "filename": {"type": "string"},
                },
                "required": ["filename"],
            },
        },
    },
]


# --- MCP component ---

mcp_sessions: dict[str, ClientSession] = {}


async def connect_mcp(stack: AsyncExitStack):
    """Launch each MCP server and merge its tools into TOOL_SCHEMAS."""
    for name, params in MCP_SERVERS.items():
        read, write = await stack.enter_async_context(
            stdio_client(params)
        )

        session = await stack.enter_async_context(
            ClientSession(read, write)
        )

        await session.initialize()

        tools = (await session.list_tools()).tools

        for tool in tools:
            # Avoid duplicate tool names
            if tool.name in LOCAL_TOOLS or tool.name in mcp_sessions:
                raise ValueError(
                    f"Duplicate tool name detected: {tool.name}"
                )

            mcp_sessions[tool.name] = session

            TOOL_SCHEMAS.append(
                {
                    "type": "function",
                    "function": {
                        "name": tool.name,
                        "description": tool.description or "",
                        "parameters": tool.input_schema,
                    },
                }
            )

        print(f"  [mcp] connected '{name}': {[t.name for t in tools]}")


async def call_tool(name: str, args: dict) -> str:
    """Run a local tool or an MCP server tool."""
    if name in LOCAL_TOOLS:
        return LOCAL_TOOLS[name](**args)

    session = mcp_sessions.get(name)

    if session is None:
        return f"Error: Unknown tool {name}"

    result = await session.call_tool(name, args)

    if result.isError:
        return "\n".join(
            c.text for c in result.content if hasattr(c, "text")
        ) or "MCP tool execution failed."

    return "\n".join(
        c.text for c in result.content if hasattr(c, "text")
    )


# --- Agent loop ---

async def run_agent(user_message: str) -> str:
    messages = [
        {
            "role": "system",
            "content": "You are a helpful personal assistant. "
                       "Use available tools when necessary.",
        },
        {"role": "user", "content": user_message},
    ]

    while True:
        response = await client.chat.completions.create(
            model=MODEL,
            messages=messages,
            tools=TOOL_SCHEMAS,
        )

        message = response.choices[0].message

        if not message.tool_calls:
            return message.content or ""

        # Preserve the assistant's tool calls in the conversation
        messages.append(message)

        for call in message.tool_calls:
            try:
                args = json.loads(call.function.arguments or "{}")
                print(f"  [tool] {call.function.name}({args})")

                result = await call_tool(call.function.name, args)

            except (json.JSONDecodeError, TypeError, OSError) as e:
                result = f"Tool error: {e}"

            except Exception as e:
                result = f"Tool error: {e}"

            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": call.id,
                    "content": str(result),
                }
            )


async def main():
    async with AsyncExitStack() as stack:
        await connect_mcp(stack)

        print(
            f"\nv3 assistant ({MODEL}) "
            f"with {len(TOOL_SCHEMAS)} tools - Ctrl+C to quit"
        )

        try:
            while True:
                question = await asyncio.to_thread(input, "\nYou: ")

                if not question.strip():
                    continue

                answer = await run_agent(question)
                print("\nAssistant:", answer)

        except (EOFError, KeyboardInterrupt):
            print("\nBye!")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nBye!")
