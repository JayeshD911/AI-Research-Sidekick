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

MODEL = "llama-3.3-70b-versatile"   # free Groq model that supports tool calling


def _get_client():
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError(
            "Missing GROQ_API_KEY. Add it to your .env file before running the app."
        )
    return OpenAI(api_key=api_key, base_url="https://api.groq.com/openai/v1")


def _normalize_history(history):
    """Accept Gradio history in either tuple or message-dict form."""
    if not history:
        return []

    normalized = []
    for item in history:
        if isinstance(item, dict):
            role = item.get("role")
            content = item.get("content")
            if role and content is not None:
                normalized.append({"role": role, "content": str(content)})
        elif isinstance(item, (list, tuple)) and len(item) >= 2:
            role, content = item[0], item[1]
            if isinstance(role, str) and content is not None:
                normalized.append({"role": role, "content": str(content)})
    return normalized

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

    try:
        client = _get_client()
    except RuntimeError as exc:
        return str(exc)

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages.extend(_normalize_history(history))
    messages.append({"role": "user", "content": user_message})

    for _ in range(MAX_TOOL_HOPS):
        try:
            response = client.chat.completions.create(
                model=MODEL,
                messages=messages,
                tools=TOOLS_SCHEMA,
            )
            msg = response.choices[0].message
        except Exception as exc:
            return f"I could not reach the model. Check your API key or network connection. Error: {exc}"

        if not msg.tool_calls:
            return msg.content or "I don't have a reply right now."

        assistant_message = {
            "role": "assistant",
            "content": msg.content or "",
            "tool_calls": [
                {
                    "id": call.id,
                    "type": call.type,
                    "function": {
                        "name": call.function.name,
                        "arguments": call.function.arguments,
                    },
                }
                for call in msg.tool_calls
            ],
        }
        messages.append(assistant_message)

        for call in msg.tool_calls:
            fn = TOOL_FUNCTIONS.get(call.function.name)
            try:
                args = json.loads(call.function.arguments)
                result = fn(**args) if fn else f"Unknown tool: {call.function.name}"
            except Exception as exc:
                result = f"Tool '{call.function.name}' failed: {exc}"

            messages.append({
                "role": "tool",
                "tool_call_id": call.id,
                "content": str(result),
            })

    try:
        final = client.chat.completions.create(model=MODEL, messages=messages)
        return final.choices[0].message.content or "I reached the tool limit without a final answer."
    except Exception as exc:
        return f"The conversation reached the tool limit, and the final reply could not be generated. Error: {exc}"


if __name__ == "__main__":
    # Smoke test of the whole menu (no in-chat memory needed for one-offs):
    print(run_agent("Read https://anthropic.com and save one key finding about them."))
    print("---")
    print(run_agent("What have you saved so far?"))
