"""Reasoning setting of the served model, request by request; one mapping for every bench. Pure functions.

Modes (option --reasoning of the benches):
  default  nothing is sent, the server decides (serve.sh turns reasoning off with --default-chat-template-kwargs)
  off      chat_template_kwargs {"enable_thinking": false}
  low, medium, xhigh
           chat_template_kwargs {"enable_thinking": true, "reasoning_effort": <level>}
The chat template of the checkpoint only accepts low, medium and xhigh (xhigh by default): "high" returns a 400. In
a reasoning mode the token budget rises to BUDGET_MIN at least, and a `length` finish is a budget too short, never a
wrong answer: it is counted apart.
"""
MODES = ("default", "off", "low", "medium", "xhigh")
REASONING = ("low", "medium", "xhigh")
BUDGET_MIN = 16384


def resolve(reasoning=None, no_think=False):
    """Mode from the options: --no-think is an alias of off; without option, default."""
    if reasoning is not None and reasoning not in MODES:
        raise ValueError(f"unknown reasoning mode: {reasoning}; known: {', '.join(MODES)}")
    if no_think and reasoning not in (None, "off"):
        raise ValueError(f"--no-think (off) contradicts --reasoning {reasoning}")
    if no_think:
        return "off"
    return reasoning or "default"


def template_kwargs(mode):
    """chat_template_kwargs to send for this mode, or None when nothing must be sent (default)."""
    if mode == "default":
        return None
    if mode == "off":
        return {"enable_thinking": False}
    if mode in REASONING:
        return {"enable_thinking": True, "reasoning_effort": mode}
    raise ValueError(f"unknown reasoning mode: {mode}")


def budget(mode, max_tokens):
    return max(max_tokens, BUDGET_MIN) if mode in REASONING else max_tokens


def apply(payload, mode, budget_field="max_tokens"):
    """Copy of the request body set for this mode. In default, the body is returned unchanged."""
    body = dict(payload)
    kw = template_kwargs(mode)
    if kw is not None:
        body["chat_template_kwargs"] = dict(body.get("chat_template_kwargs") or {}, **kw)
    if budget_field in body:
        body[budget_field] = budget(mode, body[budget_field])
    return body


def measures(response, seconds):
    """Reasoning and answer tokens of a /v1/chat/completions response. Reasoning tokens come from
    usage.completion_tokens_details.reasoning_tokens (vLLM v0.31); absent, they are None, never 0."""
    choice = (response.get("choices") or [{}])[0]
    msg = choice.get("message") or {}
    usage = response.get("usage") or {}
    total = usage.get("completion_tokens")
    rea = (usage.get("completion_tokens_details") or {}).get("reasoning_tokens")
    return {"prompt_tokens": usage.get("prompt_tokens"), "reasoning_tokens": rea,
            "answer_tokens": (total - rea) if (total is not None and rea is not None) else total,
            "reasoning_chars": len(msg.get("reasoning") or msg.get("reasoning_content") or ""),
            "finish": choice.get("finish_reason"), "seconds": round(seconds, 1)}


def self_check():
    """Fixtures of the mapping; returns the number of failures."""
    cases = [
        (resolve(), "default"), (resolve(None, True), "off"), (resolve("low"), "low"),
        (template_kwargs("default"), None), (template_kwargs("off"), {"enable_thinking": False}),
        (template_kwargs("medium"), {"enable_thinking": True, "reasoning_effort": "medium"}),
        (budget("off", 600), 600), (budget("xhigh", 600), BUDGET_MIN), (budget("low", 32000), 32000),
        (apply({"max_tokens": 10}, "default"), {"max_tokens": 10}),
        (apply({"max_tokens": 10, "chat_template_kwargs": {"x": 1}}, "low"),
         {"max_tokens": BUDGET_MIN, "chat_template_kwargs": {"x": 1, "enable_thinking": True, "reasoning_effort": "low"}}),
    ]
    bad = sum(got != want for got, want in cases)
    for bad_call in (lambda: resolve("high"), lambda: resolve("low", True)):
        try:
            bad_call(); bad += 1
        except ValueError:
            pass
    return bad


if __name__ == "__main__":
    n = self_check()
    print(f"reasoning_modes self-check: {'OK' if n == 0 else f'{n} FAIL'}")
    raise SystemExit(1 if n else 0)
