# app.py — the chat UI + public link.  [REFERENCE SOLUTION · Research Sidekick]
# Run:  python app.py    then open the local (and public) link it prints.

import gradio as gr
from agent import run_agent


def chat(message, history):
    # Gradio hands us `history` (all past turns). We pass it into the agent so
    # it REMEMBERS the conversation — the class-2/app.py homework, completed.
    # (Findings saved via the tools live in notes.json and persist across restarts.)
    return run_agent(message, history)


gr.ChatInterface(
    fn=chat,
    # type="messages",
    title="🔬 Research Sidekick — reads the web, keeps your notes",
    description=(
        "Share links and I'll read them, save the important bits, and recall "
        "them later — even after a restart. I remember our chat, too. "
        "Try researching a topic across a couple of pages."
    ),
    examples=[
        "Read https://en.wikipedia.org/wiki/Large_language_model and save the 3 key ideas",
        "What findings have you saved so far?",
        "Read https://anthropic.com and https://openai.com and save one difference between them",
    ],
).launch(share=True)
