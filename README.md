# tds-p1-databot

Data-analyst Telegram bot for TDS Project 1, Q5. Replies to every Telegram
message with exactly one JSON object: `{"answer": ..., "log_url": "..."}`.

## Architecture

Single FastAPI app (`bot.py`) running three concurrent pieces in one process:

- `GET /health` — keep-alive/sanity check
- `GET /run.jsonl` — public run log (JSONL, one event per line)
- Background thread: Telegram `getUpdates` long-poll loop → agent loop → `sendMessage`
- Background thread: self-ping `/health` every 10 minutes (keeps free hosts awake)

The agent loop gives the LLM (via OpenRouter) one tool, `run_python`, which
executes Python server-side (pandas/numpy/requests/BeautifulSoup/openpyxl
available) and returns captured stdout. It loops tool-calls until the model
returns a final plain-text answer, then extracts the first balanced JSON
object from that text.

## Environment variables

| Var | Description |
|---|---|
| `BOT_TOKEN` | Telegram bot token from @BotFather |
| `OPENROUTER_API_KEY` | OpenRouter API key |
| `BASE_URL` | Public base URL of this deployment, e.g. `https://tds-p1-databot.onrender.com` |
| `MODEL` | (optional) OpenRouter model id, default `openai/gpt-4o` |

## Deploy (Render)

1. Push this repo to GitHub (public).
2. Render → New → Web Service → connect this repo.
3. Build command: `pip install -r requirements.txt`
4. Start command: `uvicorn bot:app --host 0.0.0.0 --port $PORT`
5. Set the environment variables above.
6. After first deploy, set `BASE_URL` to the assigned `*.onrender.com` URL and
   redeploy (env var changes need a manual redeploy trigger on Render).

## Verify

```bash
curl https://<your-host>/health
wget https://<your-host>/run.jsonl
```

Then message the bot on Telegram and confirm it replies with exactly one JSON
object.
