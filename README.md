<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="assets/callshield-logo-dark.svg">
    <img src="assets/callshield-logo.svg" alt="CallShield" height="72">
  </picture>
</p>

<p align="center"><b>Hear who's really calling</b><br>Carrier-side protection from voice-clone phone scams</p>

<p align="center">
  <img src="https://img.shields.io/badge/Thirduni-2026-1E8C6E" alt="Thirduni 2026">
  <img src="https://img.shields.io/badge/python-3.11%2B-0B1622?logo=python&logoColor=white" alt="Python 3.11+">
</p>

# CallShield carrier agent

A LangChain agent that runs at the phone carrier. It works with any model behind any AI gateway. On calls for subscribers who switched CallShield on, it reads the live transcript, calls tools, and flags voice-clone and impersonation scams before money moves.

## Setup (once)
```bash
python -m venv .venv
.venv/Scripts/pip install -r requirements.txt
copy .env.example .env
```
Then fill in `.env`:

| Variable | What |
|---|---|
| `CALLSHIELD_GATEWAY_URL` | your gateway's base URL, usually ending in `/v1` |
| `CALLSHIELD_GATEWAY_KEY` | gateway API key |
| `CALLSHIELD_MODEL` | model name the gateway routes (must support tool calling) |
| `CALLSHIELD_PROTOCOL` | `openai` for OpenAI-compatible gateways (LiteLLM, OpenRouter, Portkey, Cloudflare, Vercel, Kong, vLLM, Ollama) or `anthropic` for a Messages-API gateway |
| `CALLSHIELD_EXTRA_HEADERS` | optional JSON of extra headers, e.g. `{"x-portkey-config":"..."}` |
| `CALLSHIELD_WARN_AT` / `CALLSHIELD_ESCALATE_AT` | risk thresholds (default 0.45 / 0.65) |
| `CALLSHIELD_ACTION_WEBHOOK` | optional URL that receives every action as JSON |
| `CALLSHIELD_SERVER_TOKEN` | optional bearer token for the HTTP API |

## Use
```bash
.venv/Scripts/python callshield_agent.py check            # gateway reachable + model calls tools
.venv/Scripts/python callshield_agent.py run --all        # the sample calls in samples/
.venv/Scripts/python callshield_agent.py serve --port 8080
```

### HTTP API (`serve`)
The carrier's speech-to-text posts each line as it is said:

| Request | Body | Returns |
|---|---|---|
| `POST /calls` | `{"subscriber": {...}, "caller": {...}}` | `{"call_id": "..."}` |
| `POST /calls/{id}/utterances` | `{"speaker": "caller" or "callee", "text": "...", "t": 12.3}` | risk, verdict, actions taken this turn |
| `POST /calls/{id}/end` | | final verdict and every action |
| `GET /health` | | status |

See `samples/*.json` for the `subscriber` and `caller` shapes:
- **subscriber:** consent flag, saved contacts, trusted contact.
- **caller:** number, display name, STIR/SHAKEN attestation, number age, voice-clone score.

### Embed it in Python
```python
from callshield_agent import CallShield, GatewayConfig, load_env
load_env()
shield = CallShield(GatewayConfig.from_env())
call = shield.start_call(subscriber, caller)
result = shield.on_utterance(call.call_id, "caller", "Grandma, it's me, I'm in trouble")
```

## How it decides
- **Tools the model calls:**
  - `get_call_setup_signals` and `get_voice_clone_score`, once per call.
  - `record_signals`, every caller turn.
  - `warn_callee`, `suggest_callback` and `alert_trusted_contact`, only when the policy allows them.
- **The model never decides on its own.** The risk score and thresholds are computed in code. Every action tool re-checks the policy and the subscriber's consent, so a model that tries the wrong action is blocked, and the block is logged.

## Plug in for production
- **Transcripts:** feed real-time speech-to-text from the call's media stream into `/utterances`.
- **Voice-clone score:** fill `caller.voice_clone_score` from an audio deepfake detector, or replace `get_voice_clone_score` with a call to it.
- **Actions:** point `CALLSHIELD_ACTION_WEBHOOK` at the carrier's in-call banner, SMS and app services.
- **Thresholds:** tune the weights and thresholds on labelled call data.


---

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="assets/langchain-lockup-white.svg">
    <img src="assets/langchain-lockup-black.svg" alt="LangChain" height="36">
  </picture>
</p>
<p align="center">Made with ❤️ and <a href="https://www.langchain.com/">LangChain</a> for <b>Thirduni 2026</b></p>
