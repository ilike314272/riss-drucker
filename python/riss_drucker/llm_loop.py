"""Generic language-model-in-the-loop model building (any molecule, any target).

The loop is molecule-independent; an *adapter* supplies the problem:

    adapter.system                  system prompt
    adapter.reference()             a known proposal (dict) or None
    adapter.prompt(history)         the prompt text (facts, grammar, target, earlier attempts)
    adapter.evaluate(proposal, log) -> (candidate, score) ; raises with an explanatory message to
                                       reject; score["score"] is lower-is-better, score["summary"] text
    adapter.feedback(score)         -> dict sent back to the model next round
    adapter.relation(cand, score, ref_cand, ref_score) -> str (how a proposal relates to the reference)
    adapter.describe(proposal)      -> str (commentary)

Nothing is trusted on the model's word: every proposal is compiled, certified and scored by the
adapter. Backends: `claude -p` (Claude Code print mode on the user's login), the Anthropic API,
or a replayed transcript. Usage/rate/credit limits stop the loop cleanly; transient errors are
retried; budget and token caps; Ctrl-C saves; --resume re-scores saved answers without calls.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import threading
import time
from pathlib import Path


class LimitReached(Exception):
    """Usage, rate or credit limit: stop the loop, keep what was done, resume later."""


class TransientError(Exception):
    """Overload, timeout or network failure: worth retrying after a pause."""


LIMIT_PAT = re.compile(r"usage limit|limit reached|limit will reset|rate.?limit|quota|credit balance|"
                       r"too many requests|\b429\b|budget", re.I)
TRANSIENT_PAT = re.compile(r"overloaded|\b5\d\d\b|timed? ?out|connection|temporar|ECONNRESET|network", re.I)


def classify(msg: str) -> Exception:
    if LIMIT_PAT.search(msg):
        return LimitReached(msg)
    if TRANSIENT_PAT.search(msg):
        return TransientError(msg)
    return RuntimeError(msg)


def say(msg: str = "", indent: int = 0):
    """Running commentary (stderr, flushed, timestamped)."""
    pad = " " * indent
    for i, line in enumerate(str(msg).splitlines() or [""]):
        print(f"{time.strftime('%H:%M:%S') if i == 0 else '        '} {pad}{line}", file=sys.stderr, flush=True)


class Heartbeat:
    """Prints 'still waiting' every `every` seconds while a call is in flight."""

    def __init__(self, what: str, every: float = 60.0, log=say):
        self.what, self.every, self.t0, self.log = what, every, time.time(), log
        self._stop = threading.Event()
        self._th = threading.Thread(target=self._run, daemon=True)

    def _run(self):
        while not self._stop.wait(self.every):
            self.log(f"... still waiting for {self.what} ({time.time() - self.t0:.0f} s)", 2)

    def __enter__(self):
        self._th.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()


# ----------------------------------------------------------------------------- backends
def claude_cli_complete(system: str, model=None, timeout=1800, budget_left=None):
    """Claude Code print mode, tools disabled, one turn; uses the logged-in subscription unless
    ANTHROPIC_API_KEY is set. `budget_left()` (USD) is passed as --max-budget-usd. After each call
    `complete.last_usage` holds the reported cost and tokens."""
    if os.environ.get("ANTHROPIC_API_KEY"):
        print("note: ANTHROPIC_API_KEY is set, so `claude -p` will bill the API key, not your subscription",
              file=sys.stderr)

    def complete(prompt: str) -> str:
        cmd = ["claude", "-p", "--tools", "", "--max-turns", "1", "--output-format", "json",
               "--append-system-prompt", system]
        if model:
            cmd += ["--model", model]
        if budget_left is not None and budget_left() is not None:
            cmd += ["--max-budget-usd", f"{max(budget_left(), 0.01):.2f}"]
        try:
            out = subprocess.run(cmd, input=prompt, capture_output=True, text=True, timeout=timeout)
        except subprocess.TimeoutExpired:
            raise TransientError(f"claude -p timed out after {timeout} s")
        except FileNotFoundError:
            raise RuntimeError("`claude` not found: install Claude Code (curl -fsSL https://claude.ai/install.sh | bash)")
        try:
            data = json.loads(out.stdout)
        except (json.JSONDecodeError, ValueError):
            data = {}
        complete.last_usage = {"cost_usd": data.get("total_cost_usd"),
                               "input_tokens": (data.get("usage") or {}).get("input_tokens"),
                               "output_tokens": (data.get("usage") or {}).get("output_tokens"),
                               "models_served": sorted((data.get("modelUsage") or {}).keys()) or None,
                               "model_requested": model}
        if out.returncode or data.get("is_error") or "result" not in data:
            msg = (f"claude -p failed (exit {out.returncode}): {data.get('result') or data.get('subtype') or ''} "
                   f"{out.stderr[-2000:]} {out.stdout[-500:] if not data else ''}")
            raise classify(msg)
        return data["result"]
    complete.last_usage = {}
    return complete


def api_complete(system: str, model: str = "claude-opus-5-5"):
    from .llm import anthropic_complete
    inner = anthropic_complete(model, system=system)

    def complete(prompt: str) -> str:
        try:
            text = inner(prompt)
        except Exception as e:                 # anthropic.RateLimitError (429), overloaded (529), low credit, ...
            status = getattr(e, "status_code", None)
            msg = f"{type(e).__name__} (status {status}): {e}"
            if status == 429 or "credit" in str(e).lower():
                raise LimitReached(msg)
            if status in (500, 502, 503, 504, 529) or "Connection" in type(e).__name__ or "Timeout" in type(e).__name__:
                raise TransientError(msg)
            raise classify(msg)
        complete.last_usage = dict(inner.last_usage)
        return text
    complete.last_usage = {}
    return complete


def replay_complete(path):
    answers = [r["response"] for r in json.loads(Path(path).read_text())["rounds"]]

    def complete(prompt: str) -> str:
        if not answers:
            raise RuntimeError("replay transcript exhausted")
        return answers.pop(0)
    complete.last_usage = {}
    return complete


def with_retries(complete, prompt, retries=3, wait=(30, 90, 270), log=print):
    """Retry transient failures with backoff; limits and other errors propagate at once."""
    for i in range(retries + 1):
        try:
            return complete(prompt)
        except TransientError as e:
            if i == retries:
                raise RuntimeError(f"gave up after {retries} retries: {e}")
            log(f"  transient failure ({str(e)[:200]}); retry {i + 1}/{retries} in {wait[min(i, len(wait) - 1)]} s")
            time.sleep(wait[min(i, len(wait) - 1)])


def resumed(saved: list, live):
    """Serve the saved answers of an interrupted run first (no calls, no usage), then the live backend."""
    saved = list(saved)

    def complete(prompt: str) -> str:
        if saved:
            complete.last_usage = {"resumed": True}
            return saved.pop(0)
        if live is None:
            raise RuntimeError("no saved answers left and no live backend")
        text = live(prompt)
        complete.last_usage = getattr(live, "last_usage", {})
        return text
    complete.last_usage = {}
    complete.pending = lambda: len(saved)
    return complete


def parse_answer(text: str) -> dict:
    m = re.search(r"```(?:json)?\s*(\{.*\})\s*```", text, re.S)
    if m:
        blob = m.group(1)
    elif "{" in text and "}" in text:
        blob = text[text.index("{"):text.rindex("}") + 1]
    else:
        raise ValueError("no JSON object found in the answer (expected one ```json block)")
    d = json.loads(blob)
    if "spec" not in d:
        raise ValueError("answer lacks 'spec'")
    return d


def strip_private(d):
    if isinstance(d, dict):
        return {k: strip_private(v) for k, v in d.items() if not str(k).startswith("_")}
    return d


# ----------------------------------------------------------------------------- the loop
def run_loop(adapter, make_live, *, rounds=4, transcript="transcript.json", max_budget_usd=None,
             max_total_tokens=None, retries=3, retry_wait=30.0, resume=None, backend="", model=None, log=say):
    """Run the proposal loop. make_live(budget_left) builds the live completion function.
    Returns dict(best=(label, candidate, score, proposal), reference=(cand, score), rounds, usage, stop_reason)."""
    usage = {"cost_usd": 0.0, "tokens": 0, "calls": 0}

    def budget_left():
        return None if max_budget_usd is None else max_budget_usd - usage["cost_usd"]
    live = make_live(budget_left)
    saved_rounds = []
    if resume:
        saved_rounds = [r for r in json.loads(Path(resume).read_text())["rounds"] if "response" in r]
        for r in saved_rounds:
            u = r.get("usage") or {}
            usage["cost_usd"] += u.get("cost_usd") or 0.0
            usage["tokens"] += (u.get("input_tokens") or 0) + (u.get("output_tokens") or 0)
        log(f"resuming: {len(saved_rounds)} saved answers will be re-evaluated without calls")
    complete = resumed([r["response"] for r in saved_rounds], live)

    ref = adapter.reference()
    ref_c = ref_s = None
    best = None
    if ref is not None:
        log("scoring the reference model")
        ref_c, ref_s = adapter.evaluate(ref, log)
        log(ref_s["summary"], 2)
        log(f"reference score: {ref_s['score']:.4g} -- proposals must match or beat this", 2)
        best = ("reference", ref_c, ref_s, ref)

    history, recs, stop_reason = [], [], "running"

    def save():
        Path(transcript).write_text(json.dumps({"backend": backend, "model": model, "rounds": recs,
                                                "usage_total": usage, "stop_reason": stop_reason},
                                               indent=1, default=str))
    for k in range(rounds):
        live_call = complete.pending() == 0
        if live_call and max_budget_usd is not None and usage["cost_usd"] >= max_budget_usd:
            stop_reason = f"budget cap reached ({usage['cost_usd']:.2f} >= {max_budget_usd} USD)"
            break
        if live_call and max_total_tokens is not None and usage["tokens"] >= max_total_tokens:
            stop_reason = f"token cap reached ({usage['tokens']} >= {max_total_tokens})"
            break
        prompt = adapter.prompt(history)
        t0 = time.time()
        log("")
        if complete.pending():
            log(f"round {k + 1}/{rounds}: re-using saved answer (no call)")
        else:
            log(f"round {k + 1}/{rounds}: asking the model ({len(prompt) // 1000} kB prompt"
                + (f", feedback from {len(history)} earlier attempt(s)" if history else "") + ")")
        try:
            with Heartbeat("the model", log=log):
                text = with_retries(complete, prompt, retries, wait=(retry_wait, 3 * retry_wait, 9 * retry_wait),
                                    log=lambda m: log(m, 2))
        except LimitReached as e:
            stop_reason = f"usage/rate limit: {str(e)[:500]}"
            break
        except KeyboardInterrupt:
            stop_reason = "interrupted by user"
            break
        except Exception as e:
            stop_reason = f"language-model call failed: {type(e).__name__}: {str(e)[:500]}"
            break
        u = dict(getattr(complete, "last_usage", {}) or {})
        if not u.get("resumed"):
            usage["calls"] += 1
            usage["cost_usd"] += u.get("cost_usd") or 0.0
            usage["tokens"] += (u.get("input_tokens") or 0) + (u.get("output_tokens") or 0)
        rec = {"round": k + 1, "response": text, "seconds": round(time.time() - t0, 1), "usage": u}
        log(f"answer received after {rec['seconds']:.0f} s" + (f" (reported usage ${u['cost_usd']:.2f})"
                                                               if u.get("cost_usd") else "")
            + (f" from {', '.join(u['models_served'])}" if u.get("models_served") else ""), 2)
        try:
            prop = parse_answer(text)
            log("proposal:", 2)
            log(adapter.describe(prop), 4)
            cand, sc = adapter.evaluate(prop, log)
            relation = adapter.relation(cand, sc, ref_c, ref_s) if ref_c is not None else "no reference"
            fb = {"status": "compiled, certified and scored", "relation_to_known_model": relation,
                  **adapter.feedback(sc), "score": sc["score"],
                  "reference_score": ref_s["score"] if ref_s else None}
            rec.update(proposal=prop, score=strip_private(sc), relation=relation,
                       equivalent_to_reference=relation.startswith("equivalent"))
            better = best is None or sc["score"] < best[2]["score"] - 1e-4 or (
                best[0] == "reference" and sc["score"] <= ref_s["score"] + 1e-4)
            if better:
                best = (f"llm round {k + 1}", cand, sc, prop)
            log(sc["summary"], 2)
            log(f"ACCEPTED: score {sc['score']:.4g}" + (f" vs reference {ref_s['score']:.4g}" if ref_s else "")
                + f"; {relation}" + ("; now the best model" if better else "; not better than the best so far"), 2)
        except Exception as e:          # every failure goes back to the model verbatim
            fb = {"status": "rejected", "error": f"{type(e).__name__}: {e}"}
            rec["error"] = fb["error"]
            log(f"REJECTED: {fb['error']}", 2)
            log("(this message goes back to the model in the next round)", 4)
        history.append({"round": k + 1, "proposal": rec.get("proposal", text[:4000]), "feedback": fb})
        recs.append(rec)
        save()
    if stop_reason == "running":
        stop_reason = "completed all rounds"
    save()
    if stop_reason != "completed all rounds":
        log(f"stopped early: {stop_reason}\n  {len(recs)} round(s) saved in {transcript}; the best model so far "
            f"is written. Continue later with:  --resume {transcript}")
    log("")
    log(f"done ({stop_reason}); selected: {best[0] if best else 'nothing'}; {usage['calls']} call(s)"
        + (f", reported usage ${usage['cost_usd']:.2f}" if usage["cost_usd"] else ""))
    return {"best": best, "reference": (ref_c, ref_s), "rounds": recs, "usage": usage, "stop_reason": stop_reason}
