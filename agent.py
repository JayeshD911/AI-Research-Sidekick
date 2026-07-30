# agent.py — brain + loop.  [REFERENCE SOLUTION · Research Sidekick]
# Class 2's agent.py, grown up: runs on FREE Groq (Class 1's base_url trick),
# takes `history` so it REMEMBERS within a chat (Class 2), and drives a whole
# TOOL MENU (read the web / save findings / recall findings).

import os
import json
from openai import OpenAI
from dotenv import load_dotenv
from tools import TOOLS_SCHEMA, TOOL_FUNCTIONS

load_dotenv()

# Same OpenAI client, different brain — pointed at Groq's free API (Class 1).
client = OpenAI(
    api_key=os.getenv("GROQ_API_KEY"),
    base_url="https://api.groq.com/openai/v1",
)
MODEL = "llama-3.3-70b-versatile"   # free Groq model that supports tool calling

SYSTEM_PROMPT = (
    "You are Research Sidekick, a sharp, friendly research assistant.\n"
    "Your workflow:\n"
    "• When the user shares a URL or asks about a page, call read_webpage and "
    "answer using ONLY what you found — never make things up.\n"
    "• When you find something clearly worth keeping (or the user asks you to "
    "remember it), call save_finding with a short topic.\n"
    "• When the user asks what you've saved, or you need past notes to answer, "
    "call list_findings.\n"
    "For multi-source jobs, read each page first, then save the key points, then "
    "summarise. Cite which source each point came from. Reply in clean markdown."
)

MAX_TOOL_HOPS = 8   # safety cap so the loop can't run forever


def run_agent(user_message, history=None):
    """One chat turn: think -> (use any tools it needs) -> answer.
    `history` = in-chat memory; the notes file = across-session memory."""

    # system prompt + past turns + the new message.
    # Gradio (type="messages") gives history as [{"role","content"}, ...] —
    # exactly the format the API wants, so we drop it straight in.
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    if history:
        messages.extend(history)
    messages.append({"role": "user", "content": user_message})

    # The loop: keep going while the model wants to use tools. This is what lets
    # it do MULTI-STEP jobs (read page A, read page B, save the difference...).
    for _ in range(MAX_TOOL_HOPS):
        response = client.chat.completions.create(
            model=MODEL, messages=messages, tools=TOOLS_SCHEMA)
        msg = response.choices[0].message

        # No tool requested → this is the final answer.
        if not msg.tool_calls:
            return msg.content

        # Otherwise: record the request, run each tool, feed results back.
        messages.append(msg)
        for call in msg.tool_calls:
            fn = TOOL_FUNCTIONS.get(call.function.name)
            args = json.loads(call.function.arguments)
            result = fn(**args) if fn else f"Unknown tool: {call.function.name}"
            messages.append({
                "role": "tool",                 # the third role, from Class 2
                "tool_call_id": call.id,
                "content": result,
            })
        # loop again so the model can use the results (and maybe call more tools)

    # Ran out of hops — force a final answer with no more tools.
    final = client.chat.completions.create(model=MODEL, messages=messages)
    return final.choices[0].message.content


if __name__ == "__main__":
    # Smoke test of the whole menu (no in-chat memory needed for one-offs):
    print(run_agent("Read https://anthropic.com and save one key finding about them."))
    print("---")
    print(run_agent("What have you saved so far?"))
