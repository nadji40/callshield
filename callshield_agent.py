from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import threading
import urllib.request
import uuid
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Literal

from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from pydantic import BaseModel, Field

HERE = os.path.dirname(os.path.abspath(__file__))


def load_env(path: str = os.path.join(HERE, ".env")) -> None:
    if not os.path.exists(path):
        return
    for line in open(path, encoding="utf8"):
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


@dataclass
class GatewayConfig:
    url: str
    key: str
    model: str
    protocol: Literal["openai", "anthropic"] = "openai"
    timeout: float = 30.0
    headers: dict = field(default_factory=dict)

    @classmethod
    def from_env(cls, **override) -> "GatewayConfig":
        e = os.environ.get
        vals = {"url": e("CALLSHIELD_GATEWAY_URL"), "key": e("CALLSHIELD_GATEWAY_KEY"), "model": e("CALLSHIELD_MODEL"),
                "protocol": e("CALLSHIELD_PROTOCOL", "openai"), "timeout": float(e("CALLSHIELD_TIMEOUT", "30")),
                "headers": json.loads(e("CALLSHIELD_EXTRA_HEADERS") or "{}")}
        vals.update({k: v for k, v in override.items() if v is not None})
        missing = [n for n, k in (("CALLSHIELD_GATEWAY_URL", "url"), ("CALLSHIELD_GATEWAY_KEY", "key"), ("CALLSHIELD_MODEL", "model")) if not vals[k]]
        if missing:
            sys.exit(f"Missing config: {', '.join(missing)}. Put them in {os.path.join(HERE, '.env')} (see .env.example) or pass flags.")
        if vals["protocol"] not in ("openai", "anthropic"):
            sys.exit("CALLSHIELD_PROTOCOL must be 'openai' or 'anthropic'.")
        vals["url"] = vals["url"].rstrip("/")
        return cls(**vals)

    def masked(self) -> str:
        k = self.key
        return f"{self.protocol} gateway {self.url} · model {self.model} · key {k[:4]}…{k[-2:] if len(k) > 6 else ''}"


def make_chat_model(cfg: GatewayConfig):
    if cfg.protocol == "anthropic":
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(model=cfg.model, base_url=cfg.url, api_key=cfg.key, max_tokens=4000,
                             default_request_timeout=cfg.timeout, max_retries=2, default_headers=cfg.headers or None)
    from langchain_openai import ChatOpenAI
    return ChatOpenAI(model=cfg.model, base_url=cfg.url, api_key=cfg.key, timeout=cfg.timeout, max_retries=2,
                      use_responses_api=False, default_headers=cfg.headers or None)


PaymentRail = Literal["none", "bank_transfer", "wire", "gift_card", "crypto", "cash_courier", "payment_app", "other"]
Impersonation = Literal["none", "family_or_friend", "authority", "bank", "company", "unknown"]


class ScamSignals(BaseModel):
    money_request: bool = Field(description="The caller asks the callee to send or pay money")
    amount: str | None = Field(description="Amount requested, as said (e.g. 'two thousand dollars'), or null")
    payment_rail: PaymentRail = Field(description="How the caller wants to be paid; 'none' if not mentioned")
    urgency: float = Field(ge=0, le=1, description="Time pressure: 0 none, 1 'right now or something terrible happens'")
    secrecy_request: bool = Field(description="The caller asks the callee not to tell anyone or to keep it private")
    impersonation: Impersonation = Field(description="Who the caller claims to be, if that claim drives the request")
    claimed_name: str | None = Field(description="First name the caller claims to be (e.g. 'Daniel'), or null if none")
    authority_handoff: bool = Field(description="A second party (lawyer, police, doctor, bank agent) takes over or is announced")
    emotional_pressure: float = Field(ge=0, le=1, description="Fear, guilt or distress used to push a decision")
    rationale: str = Field(description="One short sentence naming the strongest evidence")


WARN_AT = float(os.environ.get("CALLSHIELD_WARN_AT", 0.45))
ESCALATE_AT = float(os.environ.get("CALLSHIELD_ESCALATE_AT", 0.65))
RISKY_RAILS = {"gift_card", "crypto", "wire", "cash_courier"}


def network_risk(caller: dict, sub: dict) -> tuple[float, list[str]]:
    why, r = [], 0.0
    att = caller.get("stir_shaken_attestation", "C")
    r += {"A": 0.0, "B": 0.08, "C": 0.15}.get(att, 0.15)
    if att != "A":
        why.append(f"caller ID not verified (attestation {att})")
    if caller.get("number_age_days", 9999) < 30:
        r += 0.10
        why.append(f"number active {caller['number_age_days']} days")
    saved = {c["name"].lower(): c["number"] for c in sub.get("contacts", [])}
    claimed = caller.get("display_name", "").lower()
    if claimed in saved and saved[claimed] != caller.get("number"):
        r += 0.20
        why.append(f"claims to be '{caller['display_name']}' but not from their saved number")
    return r, why


def content_risk(s: ScamSignals) -> float:
    r = 0.20 * s.money_request + 0.15 * s.urgency + 0.15 * s.secrecy_request + 0.10 * s.authority_handoff
    r += 0.05 * s.emotional_pressure + (0.10 if s.impersonation in ("family_or_friend", "authority") and s.money_request else 0)
    r += 0.15 if s.payment_rail in RISKY_RAILS else (0.05 if s.payment_rail != "none" else 0)
    return r


def verdict_for(risk: float) -> str:
    return "SCAM LIKELY" if risk >= ESCALATE_AT else ("SUSPICIOUS" if risk >= WARN_AT else "LOOKS NORMAL")


@dataclass
class CallSession:
    call_id: str
    subscriber: dict
    caller: dict
    transcript: list = field(default_factory=list)
    setup: dict | None = None
    voice: float | None = None
    risk: float = 0.0
    peak: float = 0.0
    last_signals: dict | None = None
    fired: dict = field(default_factory=dict)
    tool_log: list = field(default_factory=list)
    turn_events: list = field(default_factory=list)
    lock: threading.Lock = field(default_factory=threading.Lock)

    @property
    def enabled(self) -> bool:
        return bool(self.subscriber.get("callshield_enabled"))

    def saved_contact(self, name: str | None) -> dict | None:
        names = {n.lower() for n in (name, self.caller.get("display_name")) if n}
        return next((c for c in self.subscriber.get("contacts", []) if c["name"].lower() in names), None)

    def allowed_actions(self, money_request: bool, claimed_name: str | None) -> list[str]:
        acts = []
        if self.risk >= WARN_AT:
            acts.append("warn_callee")
        if self.risk >= ESCALATE_AT and money_request:
            if self.saved_contact(claimed_name):
                acts.append("suggest_callback")
            if self.subscriber.get("trusted_contact"):
                acts.append("alert_trusted_contact")
        return [a for a in acts if a not in self.fired]


def make_tools(s: CallSession, deliver):
    def logged(name, out):
        s.tool_log.append(name)
        return out

    def guard(action: str) -> str | None:
        if not s.enabled:
            return "subscriber has not switched CallShield on"
        if s.last_signals is None:
            return "call record_signals first"
        if action not in s.allowed_actions(s.last_signals["money_request"], s.last_signals.get("claimed_name")):
            return f"not allowed by policy at risk {s.risk:.2f}"
        return None

    def blocked(action: str, why: str) -> dict:
        s.turn_events.append({"type": "blocked", "action": action, "reason": why})
        return logged(action, {"status": "blocked", "reason": why})

    def fire(action: str, detail: str) -> dict:
        s.fired[action] = detail
        event = {"type": "action", "action": action, "detail": detail, "risk": round(s.risk, 2)}
        s.turn_events.append(event)
        deliver(s, event)
        return logged(action, {"status": "done", "detail": detail})

    @tool(description="Network facts the carrier has at call setup: caller-ID attestation (STIR/SHAKEN A/B/C), "
                      "how long the number has existed, and whether the caller's display name matches a saved contact "
                      "calling from a different number. Call once per call.")
    def get_call_setup_signals() -> dict:
        r, why = network_risk(s.caller, s.subscriber)
        s.setup = {"network_risk": round(r, 2), "findings": why or ["nothing unusual"],
                   "caller_display_name": s.caller.get("display_name")}
        return logged("get_call_setup_signals", s.setup)

    @tool(description="Score from the carrier's audio deepfake detector on the caller's voice, 0 (human) to 1 "
                      "(synthetic). Call once per call.")
    def get_voice_clone_score() -> dict:
        s.voice = float(s.caller.get("voice_clone_score", 0.0))
        return logged("get_voice_clone_score", {"voice_clone_score": s.voice,
                                                "reading": "likely synthetic" if s.voice >= 0.5 else "likely human"})

    @tool(args_schema=ScamSignals,
          description="Record the scam indicators in what the CALLER has said so far. Call this every turn. Returns "
                      "the fused risk score and the only actions the carrier policy allows right now.")
    def record_signals(**signals) -> dict:
        sig = ScamSignals(**signals)
        vc = s.voice or 0.0
        s.risk = min(1.0, (s.setup or {}).get("network_risk", 0.0) + (0.35 * vc if vc >= 0.5 else 0.0) + content_risk(sig))
        s.peak = max(s.peak, s.risk)
        s.last_signals = sig.model_dump()
        return logged("record_signals", {"risk": round(s.risk, 2), "warn_at": WARN_AT, "escalate_at": ESCALATE_AT,
                                         "allowed_actions": s.allowed_actions(sig.money_request, sig.claimed_name),
                                         "already_done": list(s.fired)})

    @tool(description="Play a soft tone and show a banner on the callee's phone during the call.")
    def warn_callee() -> dict:
        if why := guard("warn_callee"):
            return blocked("warn_callee", why)
        return fire("warn_callee", "in-call tone + banner: 'This call shows signs of a scam. Don't send money yet.'")

    @tool(description="Offer the callee a one-tap button to hang up and call the person the caller claims to be, "
                      "on that person's saved number.")
    def suggest_callback(contact_name: str) -> dict:
        if why := guard("suggest_callback"):
            return blocked("suggest_callback", why)
        c = s.saved_contact(contact_name)
        if not c:
            return blocked("suggest_callback", f"no saved contact named {contact_name}")
        return fire("suggest_callback", f"button: 'Call {c['name']} on their saved number {c['number']}'")

    @tool(description="Text the subscriber's trusted contact that they are on a suspicious call asking for money.")
    def alert_trusted_contact(amount: str | None = None) -> dict:
        if why := guard("alert_trusted_contact"):
            return blocked("alert_trusted_contact", why)
        tc = s.subscriber["trusted_contact"]
        return fire("alert_trusted_contact",
                    f"SMS to {tc['name']}: '{s.subscriber['name']} is on a call that asks for {amount or 'money'}. Check on them.'")

    return [get_call_setup_signals, get_voice_clone_score, record_signals, warn_callee, suggest_callback, alert_trusted_contact]


SYSTEM = """You are CallShield, a scam-protection agent running inside a phone carrier. The subscriber
switched it on and consented to live analysis. After each thing the caller says you get the
transcript so far. Each turn:
1. If the setup facts are not known yet, call get_call_setup_signals and get_voice_clone_score.
2. Call record_signals with the indicators in what the CALLER has said so far. Judge behaviour, not
   identity: a real relative can ask for money too, and everyday money talk is not a scam.
3. Call exactly the actions listed in allowed_actions, nothing else. (For suggest_callback, pass the
   name the caller claims to be. For alert_trusted_contact, pass the amount if one was said.)
4. Finish with one short sentence on what you see. Never address the caller."""


def log_action(s: CallSession, event: dict) -> None:
    print(f"[callshield] {s.call_id} {event['action']}: {event['detail']}", file=sys.stderr, flush=True)


def webhook_action(url: str):
    def deliver(s: CallSession, event: dict) -> None:
        body = json.dumps({"call_id": s.call_id, "subscriber": s.subscriber.get("name"), **event}).encode()
        try:
            urllib.request.urlopen(urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"}), timeout=5)
        except Exception as e:
            print(f"[callshield] webhook failed for {event['action']}: {e}", file=sys.stderr)
    return deliver


class CallShield:
    def __init__(self, cfg: GatewayConfig, deliver=None, max_steps: int = 8):
        self.cfg = cfg
        self.llm = make_chat_model(cfg)
        hook = os.environ.get("CALLSHIELD_ACTION_WEBHOOK")
        self.deliver = deliver or (webhook_action(hook) if hook else log_action)
        self.max_steps = max_steps
        self.sessions: dict[str, CallSession] = {}
        self._lock = threading.Lock()

    def start_call(self, subscriber: dict, caller: dict, call_id: str | None = None) -> CallSession:
        s = CallSession(call_id or uuid.uuid4().hex[:12], subscriber, caller)
        with self._lock:
            self.sessions[s.call_id] = s
        return s

    def end_call(self, call_id: str) -> dict:
        with self._lock:
            s = self.sessions.pop(call_id)
        return self.summary(s)

    @staticmethod
    def summary(s: CallSession) -> dict:
        return {"call_id": s.call_id, "analyzed": s.enabled, "verdict": verdict_for(s.peak) if s.enabled else None,
                "peak_risk": round(s.peak, 2), "actions": s.fired, "tool_calls": s.tool_log, "last_signals": s.last_signals}

    def on_utterance(self, call_id: str, speaker: str, text: str, t: float | None = None) -> dict:
        s = self.sessions[call_id]
        with s.lock:
            who = "CALLER" if speaker.lower() == "caller" else "CALLEE"
            s.transcript.append(f"{who}: {text}")
            s.turn_events = []
            if not s.enabled or who != "CALLER":
                return {"call_id": call_id, "analyzed": s.enabled, "risk": round(s.risk, 2),
                        "verdict": verdict_for(s.risk), "events": []}
            before = len(s.tool_log)
            note = self._agent_turn(s)
            return {"call_id": call_id, "analyzed": True, "t": t, "risk": round(s.risk, 2), "verdict": verdict_for(s.risk),
                    "events": s.turn_events, "tools": s.tool_log[before:], "note": note}

    def _agent_turn(self, s: CallSession) -> str:
        tools = make_tools(s, self.deliver)
        by_name = {t.name: t for t in tools}
        llm = self.llm.bind_tools(tools)
        known = (f"\nSetup facts already known: {json.dumps(s.setup)}; voice_clone_score={s.voice}."
                 if s.setup is not None and s.voice is not None else "\nSetup facts: not fetched yet.")
        msgs = [SystemMessage(SYSTEM), HumanMessage(
            f"Callee: {s.subscriber.get('name')}. Caller ID shows: {s.caller.get('display_name')}.{known}\n"
            f"Actions already taken: {list(s.fired) or 'none'}.\n\nTranscript so far:\n" + "\n".join(s.transcript))]
        for _ in range(self.max_steps):
            ai = llm.invoke(msgs)
            msgs.append(ai)
            if not ai.tool_calls:
                return (ai.text if isinstance(getattr(ai, "text", None), str) else str(ai.content)).strip()
            for tc in ai.tool_calls:
                t = by_name.get(tc["name"])
                try:
                    out = t.invoke(tc["args"]) if t else {"error": f"unknown tool {tc['name']}"}
                except Exception as e:
                    out = {"error": f"{type(e).__name__}: {e}"}
                msgs.append(ToolMessage(content=json.dumps(out), tool_call_id=tc["id"], name=tc["name"]))
        return "(stopped: step limit reached)"


def bar(x: float) -> str:
    n = round(x * 20)
    col = "\033[32m" if x < WARN_AT else ("\033[33m" if x < ESCALATE_AT else "\033[31m")
    return f"{col}{'█' * n}{'·' * (20 - n)}\033[0m {x:.2f}"


def cmd_check(cfg: GatewayConfig) -> None:
    print(cfg.masked())

    @tool(description="Connectivity probe. Call it with ok=true.")
    def ping(ok: bool) -> str:
        return "pong"

    ai = make_chat_model(cfg).bind_tools([ping]).invoke([HumanMessage("Call the ping tool with ok=true. Do not write anything else.")])
    if ai.tool_calls and ai.tool_calls[0]["name"] == "ping":
        print("OK: the gateway answered and the model calls tools. CallShield is ready.")
    else:
        sys.exit("The gateway answered, but the model did not call the tool. Pick a model with tool calling.")


def cmd_run(cfg: GatewayConfig, paths: list[str], report: str | None) -> None:
    quiet = None if os.environ.get("CALLSHIELD_ACTION_WEBHOOK") else (lambda s, e: None)
    shield = CallShield(cfg, deliver=quiet)
    print(cfg.masked())
    results = []
    for p in paths:
        call = json.load(open(p, encoding="utf8"))
        s = shield.start_call(call["subscriber"], call["caller"])
        print(f"\n\033[1m═══ {call.get('title', os.path.basename(p))} ═══\033[0m")
        if not s.enabled:
            print("   CallShield is off for this subscriber; the call passes untouched.")
        for u in call["transcript"]:
            print(f"   [{u.get('t', 0):5.1f}s] {u['speaker']:6s} {u['text']}")
            r = shield.on_utterance(s.call_id, u["speaker"], u["text"], u.get("t"))
            for e in r["events"]:
                if e["type"] == "action":
                    print(f"   \033[1m▶ {e['action']}\033[0m  {e['detail']}")
                else:
                    print(f"   \033[2m✕ {e['action']} blocked: {e['reason']}\033[0m")
            if "note" in r:
                print(f"          risk {bar(r['risk'])}  {r['note'][:150]}")
        summ = shield.end_call(s.call_id)
        if summ["analyzed"]:
            print(f"   result: \033[1m{summ['verdict']}\033[0m · peak risk {summ['peak_risk']:.2f} · actions {list(summ['actions']) or 'none'}")
        results.append({"call": call.get("title", p), **summ})
    if report:
        json.dump(results, open(report, "w", encoding="utf8"), indent=1, ensure_ascii=False)
        print(f"\nreport written to {report}")


def cmd_serve(cfg: GatewayConfig, host: str, port: int) -> None:
    shield = CallShield(cfg)
    token = os.environ.get("CALLSHIELD_SERVER_TOKEN")

    class Handler(BaseHTTPRequestHandler):
        def _send(self, code: int, body: dict) -> None:
            data = json.dumps(body).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def _body(self) -> dict:
            n = int(self.headers.get("Content-Length", 0))
            return json.loads(self.rfile.read(n)) if n else {}

        def _authed(self) -> bool:
            if token and self.headers.get("Authorization") != f"Bearer {token}":
                self._send(401, {"error": "unauthorized"})
                return False
            return True

        def do_GET(self):
            if self.path == "/health":
                self._send(200, {"ok": True, "model": cfg.model, "open_calls": len(shield.sessions)})
            else:
                self._send(404, {"error": "not found"})

        def do_POST(self):
            if not self._authed():
                return
            parts = [p for p in self.path.split("/") if p]
            try:
                if parts == ["calls"]:
                    b = self._body()
                    s = shield.start_call(b["subscriber"], b["caller"], b.get("call_id"))
                    return self._send(201, {"call_id": s.call_id, "analyzed": s.enabled})
                if len(parts) == 3 and parts[0] == "calls" and parts[2] == "utterances":
                    if parts[1] not in shield.sessions:
                        return self._send(404, {"error": f"unknown call {parts[1]}"})
                    b = self._body()
                    return self._send(200, shield.on_utterance(parts[1], b["speaker"], b["text"], b.get("t")))
                if len(parts) == 3 and parts[0] == "calls" and parts[2] == "end":
                    if parts[1] not in shield.sessions:
                        return self._send(404, {"error": f"unknown call {parts[1]}"})
                    return self._send(200, shield.end_call(parts[1]))
                self._send(404, {"error": "not found"})
            except (KeyError, json.JSONDecodeError) as e:
                self._send(400, {"error": f"bad request: {e}"})
            except Exception as e:
                self._send(502, {"error": f"{type(e).__name__}: {e}"})

        def log_message(self, fmt, *args):
            print(f"[callshield] {self.command} {self.path} {args[1] if len(args) > 1 else ''}", file=sys.stderr, flush=True)

    print(f"CallShield serving on http://{host}:{port}  ({cfg.masked()})", flush=True)
    ThreadingHTTPServer((host, port), Handler).serve_forever()


def main() -> None:
    global WARN_AT, ESCALATE_AT
    sys.stdout.reconfigure(encoding="utf-8")
    load_env()
    WARN_AT = float(os.environ.get("CALLSHIELD_WARN_AT", WARN_AT))
    ESCALATE_AT = float(os.environ.get("CALLSHIELD_ESCALATE_AT", ESCALATE_AT))
    ap = argparse.ArgumentParser(description="CallShield: carrier-side scam flagging through any AI gateway")
    ap.add_argument("--gateway-url")
    ap.add_argument("--api-key")
    ap.add_argument("--model")
    ap.add_argument("--protocol", choices=["openai", "anthropic"])
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("check", help="test the gateway and the model's tool calling")
    r = sub.add_parser("run", help="analyze call files")
    r.add_argument("calls", nargs="*")
    r.add_argument("--all", action="store_true", help="every file in samples/")
    r.add_argument("--report")
    sv = sub.add_parser("serve", help="HTTP API for the carrier's live transcription")
    sv.add_argument("--host", default="127.0.0.1")
    sv.add_argument("--port", type=int, default=8080)
    a = ap.parse_args()

    cfg = GatewayConfig.from_env(url=a.gateway_url, key=a.api_key, model=a.model, protocol=a.protocol)
    import anthropic
    import openai
    try:
        if a.cmd == "check":
            cmd_check(cfg)
        elif a.cmd == "run":
            paths = sorted(glob.glob(os.path.join(HERE, "samples", "*.json"))) if a.all else a.calls
            if not paths:
                ap.error("give call files or --all")
            cmd_run(cfg, paths, a.report)
        else:
            cmd_serve(cfg, a.host, a.port)
    except (openai.AuthenticationError, anthropic.AuthenticationError):
        sys.exit(f"The gateway rejected the API key ({cfg.url}).")
    except (openai.APIConnectionError, anthropic.APIConnectionError):
        sys.exit(f"Can't reach the gateway at {cfg.url}. Is the URL right (usually ends in /v1)?")
    except (openai.NotFoundError, anthropic.NotFoundError) as e:
        sys.exit(f"The gateway doesn't know model '{cfg.model}' or this path: {e}")


if __name__ == "__main__":
    main()
