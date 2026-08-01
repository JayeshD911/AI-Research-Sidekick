# tools.py — the agent's "hands": a MENU of tools.  [REFERENCE SOLUTION]
# A tool = a normal Python function + a schema describing it (like get_price in
# class-2/agent.py). Here the agent gets three:
#   • read_webpage   — READ the live web (Class 1 scraper)
#   • save_finding   — WRITE something worth remembering to a file
#   • list_findings  — READ those saved findings back (survives restarts)
#
# The save/recall pair is what gives the agent memory ACROSS sessions — using
# nothing but Python's own json + file I/O. No database.

import os
import json
import requests
from bs4 import BeautifulSoup

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                  "AppleWebKit/537.36 (KHTML, like Gecko) "
                  "Chrome/120.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
}

MAX_CHARS = 6000               # keep scraped text inside free-tier limits
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
NOTES_FILE = os.path.join(BASE_DIR, "notes.json")      # our tiny "database" — just a JSON file on disk


# ── Tool 1: read a web page (Class 1 scraper) ───────────────────────────────
def read_webpage(url):
    """Fetch a URL and return its readable text."""
    print(f"🔧 tool called: read_webpage({url})")   # makes tool use visible

    if not url.startswith(("http://", "https://")):
        url = "https://" + url

    try:
        response = requests.get(url, headers=HEADERS, timeout=15)
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        return f"Could not fetch the website. Error: {e}"

    soup = BeautifulSoup(response.text, "html.parser")
    title = soup.title.string if soup.title and soup.title.string else "No title found"

    for tag in soup(["script", "style", "nav", "footer", "header", "img", "input"]):
        tag.decompose()

    text = soup.get_text(separator="\n", strip=True)
    return f"Title: {title}\n\nPage contents:\n{text[:MAX_CHARS]}"


# ── little helpers: load/save the notes file safely ─────────────────────────
def _load_notes():
    if not os.path.exists(NOTES_FILE):
        return []
    try:
        with open(NOTES_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return []            # corrupt/missing → start fresh instead of crashing


def _save_notes(notes):
    with open(NOTES_FILE, "w", encoding="utf-8") as f:
        json.dump(notes, f, indent=2, ensure_ascii=False)


# ── Tool 2: save a finding (WRITE — this is cross-session memory) ────────────
def save_finding(topic, finding):
    """Save something worth remembering, filed under a topic."""
    print(f"🔧 tool called: save_finding(topic={topic!r})")
    notes = _load_notes()
    notes.append({"topic": topic, "finding": finding})
    _save_notes(notes)
    return f"Saved under '{topic}'. You now have {len(notes)} findings on file."


# ── Tool 3: list findings (READ them back, optionally by topic) ─────────────
def list_findings(topic=None):
    """Return saved findings. If `topic` is given, only matching ones."""
    print(f"🔧 tool called: list_findings(topic={topic!r})")
    notes = _load_notes()
    if topic:
        notes = [n for n in notes if topic.lower() in n["topic"].lower()]
    if not notes:
        return "No saved findings yet."
    return "\n".join(f"- [{n['topic']}] {n['finding']}" for n in notes)


# ── The schema: tells the model each tool exists and WHEN to use it ─────────
TOOLS_SCHEMA = [
    {
        "type": "function",
        "function": {
            "name": "read_webpage",
            "description": (
                "Fetch and read the text of a web page. Use whenever the user "
                "shares a URL or asks about the contents of a specific website "
                "or article. Read the page before answering questions about it."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {"type": "string", "description": "The full URL to read"},
                },
                "required": ["url"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "save_finding",
            "description": (
                "Save an important fact, insight, or conclusion so it can be "
                "recalled later (even in a future session). Use this when the "
                "user asks to remember/save something, or when you've found "
                "something clearly worth keeping."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "topic": {"type": "string", "description": "Short label to file it under"},
                    "finding": {"type": "string", "description": "The fact/insight to remember"},
                },
                "required": ["topic", "finding"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_findings",
            "description": (
                "Look up previously saved findings. Use when the user asks what "
                "you've saved/remember, or when you need your past notes to "
                "answer. Optionally filter by topic."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "topic": {"type": "string", "description": "Optional topic filter"},
                },
                "required": [],
            },
        },
    },
]

# ── Registry: lets the loop find the real function by its name ──────────────
# Add a new tool = add its function above, a schema entry, and one line here.
TOOL_FUNCTIONS = {
    "read_webpage": read_webpage,
    "save_finding": save_finding,
    "list_findings": list_findings,
}


if __name__ == "__main__":
    # Quick self-test of the whole menu:
    print(read_webpage("https://anthropic.com")[:300])
    print(save_finding("demo", "This is a saved finding."))
    print(list_findings())
