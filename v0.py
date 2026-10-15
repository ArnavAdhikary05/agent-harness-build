#normal llm
import sys
sys.stdout.reconfigure(encoding="utf-8")

from groq import Groq
import os
from dotenv import load_dotenv

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))
MODEL = "qwen/qwen3.8-27b"

def chat(user_message: str) -> str:
    completion = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": "You are a helpful personal assistant"},
            {"role": "user", "content": user_message},
        ],
        temperature=1,
        max_completion_tokens=2048,
        top_p=1,
        reasoning_effort="medium",
        stream=True,
    )

    response = ""

    for chunk in completion:
        content = chunk.choices[0].delta.content
        if content:
            response += content

    return response

if __name__ == "__main__":
    print(f"v0 assistant ({MODEL}) - ctrl + c to quit")
    try:
        while True:
            question = input("\nyou: ")
            print("\nassistant: ", chat(question))
    except(EOFError, KeyboardInterrupt):
        print("\nBye!")