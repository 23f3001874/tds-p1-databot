import io
import json
import os
import re
import sys
import threading
import time
import traceback
from collections import defaultdict
from contextlib import redirect_stdout
from datetime import datetime, timezone

import requests
from fastapi import FastAPI
from fastapi.responses import PlainTextResponse

BOT_TOKEN = os.environ["BOT_TOKEN"]
BASE_URL = os.environ["BASE_URL"].rstrip("/")

LLM_API_KEY = os.environ.get("LLM_API_KEY") or os.environ.get("OPENROUTER_API_KEY") or os.environ.get("AIPIPE_TOKEN")
LLM_BASE_URL = os.environ.get("LLM_BASE_URL", "https://openrouter.ai/api/v1")
MODEL = os.environ.get("MODEL", "openai/gpt-4o")

TELEGRAM_API = f"https://api.telegram.org/bot{BOT_TOKEN}"
LOG_PATH = "run.jsonl"
LOG_LOCK = threading.Lock()

TOOL_TIMEOUT_SECONDS = 210  # wall-clock budget out of the ~300s grader timeout
MAX_TOOL_STEPS = 10
HISTORY_TURNS = 20

app = FastAPI()

# per-chat conversation history
histories = defaultdict(list)


def log_event(event):
    entry = {"timestamp": datetime.now(timezone.utc).isoformat(), **event}
    with LOG_LOCK:
        with open(LOG_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, default=str) + "\n")


@app.get("/health")
def health():
    return {"ok": True, "time": datetime.now(timezone.utc).isoformat()}


@app.get("/run.jsonl")
def run_log():
    if not os.path.exists(LOG_PATH):
        return PlainTextResponse("", media_type="application/json")
    with open(LOG_PATH, "r", encoding="utf-8") as f:
        return PlainTextResponse(f.read(), media_type="application/json")


SYSTEM_PROMPT = """You are a data-analyst agent replying inside a Telegram conversation.

Rules:
1. Answer the LATEST user message. Earlier messages in this chat are context (some tasks are multi-turn: setup message(s), then the real question).
2. You have a tool `run_python(code)` that executes Python server-side and returns captured stdout. Use it whenever the question gives you inline data to compute on, or points at a specific dataset/URL, or you are genuinely confident of a real, stable, fetchable URL (e.g. a specific data.gov.in/api.data.gov.in resource, a specific Wikipedia page, a specific known CSV/XLSX endpoint). pandas, numpy, requests, BeautifulSoup, openpyxl are available.
3. NEVER invent or guess a plausible-looking URL (e.g. a made-up example.com or mospi.gov.in file path) just to attempt a fetch. A fabricated URL will fail and wastes a step for nothing — it does not make your answer more reliable. If you do not know a real, specific, working URL for the requested dataset, skip the tool entirely and answer directly from your own knowledge in one step.
4. For questions about a well-known published statistic (a national ranking, a commonly-cited government figure, etc.) where you are not sure of an exact fetchable URL: think carefully and recall the precise correct fact from your training knowledge BEFORE answering — this is usually more reliable than a guessed fetch that fails and forces a rushed fallback answer.
5. If you do attempt a fetch and it fails, do not immediately guess — try at most one alternative real source if you know one, then give your single best, carefully-reasoned answer from knowledge. Do not let a failed fetch attempt degrade the quality of your final reasoning.
6. Your final reply must be ONLY a single JSON object and NOTHING else — no markdown fences, no prose like "Here is the answer:", nothing before or after it.
7. Match the exact JSON shape the question asks for — same keys, same nesting, same type (number vs string vs list). Never add extra keys beyond what's asked, except a "log_url" key which will be overwritten automatically — you may put any placeholder there.
8. If a message is only setup text ("I'll send data next"), still reply with a small JSON acknowledgement — every message needs a reply.
9. Never crash or refuse — if you truly cannot compute something, give your best single answer in the requested shape rather than an error or explanation."""

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "run_python",
            "description": "Execute Python code server-side and return captured stdout (last 8000 chars). Use print() to output results. pandas, numpy, requests, bs4 (BeautifulSoup), openpyxl are available.",
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {"type": "string", "description": "Python code to execute"}
                },
                "required": ["code"],
            },
        },
    }
]


def run_python_tool(code):
    buf = io.StringIO()
    try:
        with redirect_stdout(buf):
            exec_globals = {"__name__": "__main__"}
            exec(code, exec_globals)
        out = buf.getvalue()
    except Exception:
        out = buf.getvalue() + "\n" + traceback.format_exc()
    return out[-8000:]


def call_llm(messages, allow_tools=True):
    payload = {
        "model": MODEL,
        "messages": messages,
        "temperature": 0,
    }
    if allow_tools:
        payload["tools"] = TOOLS
    resp = requests.post(
        f"{LLM_BASE_URL}/chat/completions",
        headers={
            "Authorization": f"Bearer {LLM_API_KEY}",
            "Content-Type": "application/json",
        },
        json=payload,
        timeout=60,
    )
    resp.raise_for_status()
    return resp.json()


def extract_json(text):
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\n", "", text)
        text = re.sub(r"\n```$", "", text)
        text = text.strip()

    depth = 0
    start = None
    for i, ch in enumerate(text):
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0 and start is not None:
                candidate = text[start:i + 1]
                try:
                    return json.loads(candidate)
                except json.JSONDecodeError:
                    continue
    raise ValueError(f"no balanced JSON object found in: {text!r}")


def agent_reply(chat_id, user_message):
    deadline = time.monotonic() + TOOL_TIMEOUT_SECONDS

    history = histories[chat_id]
    history.append({"role": "user", "content": user_message})
    history[:] = history[-HISTORY_TURNS:]

    messages = [{"role": "system", "content": SYSTEM_PROMPT}] + history

    final_text = None
    for step in range(MAX_TOOL_STEPS):
        time_left = deadline - time.monotonic()
        if time_left <= 0:
            messages.append({
                "role": "user",
                "content": "Time is up. Answer NOW with only the JSON object, no more tool calls.",
            })

        try:
            data = call_llm(messages, allow_tools=time_left > 0)
        except Exception as e:
            log_event({"type": "llm_error", "chat_id": chat_id, "error": str(e)})
            break

        choice = data["choices"][0]
        msg = choice["message"]
        log_event({"type": "llm_step", "chat_id": chat_id, "step": step, "message": msg})

        tool_calls = msg.get("tool_calls")
        if tool_calls and time_left > 0:
            messages.append(msg)
            for tc in tool_calls:
                if tc["function"]["name"] == "run_python":
                    try:
                        args = json.loads(tc["function"]["arguments"])
                    except json.JSONDecodeError:
                        args = {"code": ""}
                    code = args.get("code", "")
                    result = run_python_tool(code)
                    log_event({"type": "tool_call", "chat_id": chat_id, "code": code,
                               "result": result})
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc["id"],
                        "content": result,
                    })
            continue
        else:
            final_text = msg.get("content") or ""
            break

    if final_text is None:
        final_text = ""

    try:
        parsed = extract_json(final_text)
    except Exception as e:
        log_event({"type": "parse_error", "chat_id": chat_id, "raw": final_text, "error": str(e)})
        parsed = {"answer": "internal error"}

    if not isinstance(parsed, dict):
        parsed = {"answer": parsed}
    if "answer" not in parsed:
        parsed = {"answer": parsed}

    parsed["log_url"] = f"{BASE_URL}/run.jsonl"

    history.append({"role": "assistant", "content": json.dumps(parsed)})
    history[:] = history[-HISTORY_TURNS:]

    return parsed


def send_message(chat_id, text):
    requests.post(f"{TELEGRAM_API}/sendMessage",
                   json={"chat_id": chat_id, "text": text}, timeout=30)


def poll_loop():
    offset = None
    while True:
        try:
            resp = requests.get(f"{TELEGRAM_API}/getUpdates",
                                 params={"timeout": 30, "offset": offset}, timeout=40)
            data = resp.json()
            for update in data.get("result", []):
                offset = update["update_id"] + 1
                message = update.get("message")
                if not message or "text" not in message:
                    continue
                chat_id = message["chat"]["id"]
                text = message["text"]
                log_event({"type": "incoming", "chat_id": chat_id, "text": text})
                try:
                    reply = agent_reply(chat_id, text)
                except Exception:
                    log_event({"type": "handler_crash", "chat_id": chat_id,
                               "error": traceback.format_exc()})
                    reply = {"answer": "internal error", "log_url": f"{BASE_URL}/run.jsonl"}
                reply_text = json.dumps(reply)
                send_message(chat_id, reply_text)
                log_event({"type": "outgoing", "chat_id": chat_id, "text": reply_text})
        except Exception:
            log_event({"type": "poll_loop_error", "error": traceback.format_exc()})
            time.sleep(5)


def keepalive_loop():
    while True:
        time.sleep(600)
        try:
            requests.get(f"{BASE_URL}/health", timeout=10)
        except Exception:
            pass


@app.on_event("startup")
def startup():
    threading.Thread(target=poll_loop, daemon=True).start()
    threading.Thread(target=keepalive_loop, daemon=True).start()
