#!/usr/bin/env python3
"""Witness of the literal <tool_call>: what the model emits, against what the parser returns (read-only).

Three cases (prompts/toolcall.json): French with a declared tool, French without tool, Hungarian with a tool. The
prompt asks to explain the literal string <tool_call> without calling a tool. Unprompted cases (--cases
code_without_tool,prose_fr_without_tool): the probe's prompts, no tool declared; any call there is invented. Case
code_with_tool: the same code prompt with the Bash tool declared; a call to Bash is an agent's move, a call to any
other name is invented. N series of 4 requests per case, temperature 0.7, one request at a time.
--reasoning {default,off,low,medium,xhigh} (reasoning_modes.py). --system TEXT puts a system message before the
prompt (none by default).

Called names: every <function=NAME> outside code in the answer, against the declared tools. The summary counts, per
case, the draws with at least one undeclared name ("invented") and those whose calls are all declared.

Chat path (default): /v1/chat/completions with return_token_ids; the generated tokens are decoded again by
/detokenize, special tokens included, and compared with the message the parser returns. That separates a choice of
the model from a fabrication of the parser without replaying the request another way.
Raw path (--raw): /tokenize (messages and tools) then /v1/completions, skip_special_tokens false, no parser.

Classes (chat path):
  complete            the model writes text, the returned content carries at least 90 % of the visible text
  short_stop          the model stops by itself before 20 tokens, the returned content is whole: a model cause
  content_lost        the model's text is longer than the returned content: the parser lost it
  model_call          the model writes a structured call outside code, the parser returns it: a model choice
  fabricated_call     the parser returns a call that the model did not write outside code
  missed_call         the model writes a structured call outside code, the parser returns none
  call_from_reasoning the parser returns a call that the model only sketched in its reasoning, outside code
  length_finish       reasoning mode, `length` finish without a call: budget too short, never a wrong answer
A "structured" call is <tool_call> followed by <function=, or a bare <function=, outside a code block and outside
backticks.
  python3 -I bench/toolcall.py <label> [--series 3] [--cases hu_with_tool] [--raw] [--system TEXT]
      results/toolcall-<label>.json
  python3 -I bench/toolcall.py --self-check        classification fixtures only, no network
"""
import argparse, os, re, sys, time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import reasoning_modes as rm  # noqa: E402

SPECIAL = re.compile(r"<\|im_end\|>|<\|endoftext\|>|<\|im_start\|>")
MARKER = re.compile(r"<tool_call>\s*<function=|<function=")
NAME = re.compile(r"<function=([^>\s]+)>")
SHORT = 20
WHOLE = 0.9


def in_code(text, pos):
    """True if pos falls in an open ``` block or between two backticks of the same line."""
    before = text[:pos]
    if before.count("```") % 2:
        return True
    line = before[before.rfind("\n") + 1:].replace("```", "")
    return line.count("`") % 2 == 1


def structured_call(raw):
    for m in MARKER.finditer(raw):
        if not in_code(raw, m.start()):
            return m.start()
    return None


def called_names(text):
    """Names of the structured calls outside code, in order."""
    return [m.group(1) for m in NAME.finditer(text) if not in_code(text, m.start())]


def undeclared(names, tools):
    declared = {t["function"]["name"] for t in tools or []}
    return [n for n in names if n not in declared]


def build_messages(prompt, system=None):
    return ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": prompt}]


def markers(raw):
    out = {}
    for name, pattern in (("tool_call", "<tool_call>"), ("function", "<function=")):
        pos = [m.start() for m in re.finditer(re.escape(pattern), raw)]
        out[name] = len(pos)
        out[name + "_outside_code"] = sum(not in_code(raw, p) for p in pos)
    return out


def visible(raw):
    return SPECIAL.sub("", re.sub(r"^\s*<think>.*?</think>", "", raw, flags=re.S)).strip()


def split(raw, reasoning):
    """(reasoning, answer) of the generated text; without </think> in reasoning mode, all is reasoning."""
    if not reasoning:
        return "", raw
    if "</think>" not in raw:
        return raw, ""
    i = raw.index("</think>")
    return re.sub(r"^\s*<think>", "", raw[:i]), raw[i + len("</think>"):]


def classify(raw, tokens, content=None, calls=None, reasoning=False, finish=None):
    """content and calls None: raw path (model side only). tokens: those of the answer."""
    rea, answer = split(raw, reasoning)
    model_calls = structured_call(answer) is not None
    if content is None:
        return "model_call" if model_calls else ("short_stop" if tokens < SHORT else "text")
    if calls and model_calls:
        return "model_call"
    if calls and structured_call(rea) is not None:
        return "call_from_reasoning"
    if calls:
        return "fabricated_call"
    if model_calls:
        return "missed_call"
    if reasoning and finish == "length":
        return "length_finish"
    if len(content.strip()) < WHOLE * len(visible(answer)):
        return "content_lost"
    return "short_stop" if tokens < SHORT else "complete"


def self_check():
    call = "<tool_call>\n<function=Bash>\n<parameter=command>ls</parameter>\n</function>\n</tool_call><|im_end|>"
    long_fr = "La chaine litterale `<tool_call>` est une balise de controle. " * 6 + "<|im_end|>"
    cases = [
        ("whole text", (long_fr, 120, visible(long_fr), []), "complete"),
        ("stop after the backtick", ("La chaine litterale `<|im_end|>", 6, "La chaine litterale `", []), "short_stop"),
        ("text lost after the literal", (long_fr, 200, "La chaine litterale `", []), "content_lost"),
        ("call written by the model", (call, 30, "", [{"name": "Bash"}]), "model_call"),
        ("model call not returned", (call, 30, "", []), "missed_call"),
        ("function= between backticks", ("Pelda: `<function=Bash>` a nev.<|im_end|>", 29, "", [{"name": "Bash"}]), "fabricated_call"),
        ("call in a code block", ("Exemple :\n```\n<tool_call>\n<function=Bash>\n```\nFin.<|im_end|>", 40, "", [{"name": "Bash"}]),
         "fabricated_call"),
        ("bare function= outside code", ("Voici <function=Bash> en clair<|im_end|>", 12, "", [{"name": "Bash"}]), "model_call"),
        ("raw: structured call", (call, 30, None, None), "model_call"),
        ("raw: short stop", ("La chaine litterale `<|im_end|>", 6, None, None), "short_stop"),
        ("raw: text", (long_fr, 120, None, None), "text"),
    ]
    thought = "Je relis la demande et je pese chaque terme avant de repondre. " * 8
    sketch = "Je pourrais ecrire <tool_call>\n<function=Bash>\n<parameter=command>ls</parameter>\n</function>\n</tool_call> mais non. "
    R = {"reasoning": True}
    cases += [
        ("long reasoning then whole answer", (thought + "</think>\n\n" + long_fr, 120, visible(long_fr), []), "complete", R),
        ("call sketched in reasoning, returned by the parser", (sketch + "</think>\n\nTexte.<|im_end|>", 30, "", [{"name": "Bash"}]),
         "call_from_reasoning", R),
        ("sketch in reasoning, not returned, whole answer", (sketch + "</think>\n\n" + long_fr, 120, visible(long_fr), []), "complete", R),
        ("call written in the answer after reasoning", (thought + "</think>\n\n" + call, 30, "", [{"name": "Bash"}]), "model_call", R),
        ("cut by the budget in reasoning", (thought * 4, 0, "", []), "length_finish", dict(R, finish="length")),
        ("stop in reasoning", (thought + "<|im_end|>", 0, "", []), "short_stop", R),
        ("raw with reasoning: sketch only in reasoning", (sketch + "</think>\n\n" + long_fr, 120, None, None), "text", R),
        ("raw with reasoning: call in the answer", (thought + "</think>\n\n" + call, 30, None, None), "model_call", R),
        ("without reasoning, a literal </think> stays text",
         ("Le marqueur </think> ferme le raisonnement. " * 4 + "<|im_end|>", 60, "Le marqueur </think> ferme le raisonnement. " * 4, []),
         "complete"),
    ]
    red = 0
    for name, args, want, *opt in cases:
        got = classify(*args, **(opt[0] if opt else {}))
        red += got != want
        print(f"  {'ok ' if got == want else 'RED'}  {name}: {got} (expected {want})")
    bash = [{"type": "function", "function": {"name": "Bash"}}]
    two = "<tool_call>\n<function=Bash>\n</function>\n</tool_call>\n<tool_call>\n<function=read_file>\n</function>\n</tool_call>"
    names = [
        ("called names, outside code only", called_names("```\n<function=list_files>\n```\n" + call), ["Bash"]),
        ("called names, between backticks ignored", called_names("Write `<function=read_file>` to call."), []),
        ("declared call, no undeclared name", undeclared(called_names(call), bash), []),
        ("no tool declared, every name is undeclared", undeclared(called_names(call), None), ["Bash"]),
        ("declared then undeclared", undeclared(called_names(two), bash), ["read_file"]),
        ("messages without system", build_messages("p"), [{"role": "user", "content": "p"}]),
        ("messages with system", build_messages("p", "s"), [{"role": "system", "content": "s"}, {"role": "user", "content": "p"}]),
    ]
    for name, got, want in names:
        red += got != want
        print(f"  {'ok ' if got == want else 'RED'}  {name}: {got} (expected {want})")
    total = len(cases) + len(names)
    print(f"self-check: {total - red}/{total}")
    return 1 if red else 0


def excerpt(raw):
    pos = structured_call(raw)
    start = max(0, (pos if pos is not None else 0) - 60)
    return raw[start:start + 200]


def one_request(c, model, prompt, tools, raw_only, temperature, max_tokens, mode, system=None):
    messages = build_messages(prompt, system)
    t0 = time.time()
    if raw_only:
        body = {"model": model, "messages": messages, "add_generation_prompt": True}
        if tools:
            body["tools"] = tools
        body = rm.apply(body, mode)
        ids = c.call_json("/tokenize", body)["tokens"]
        d = c.call_json("/v1/completions", {"model": model, "prompt": ids, "max_tokens": rm.budget(mode, max_tokens),
                                             "temperature": temperature, "skip_special_tokens": False})
        ch = d["choices"][0]; raw = ch["text"]; tokens = d["usage"]["completion_tokens"]
        reasoning = mode in rm.REASONING
        rea, answer = split(raw, reasoning)
        names = called_names(answer)
        r = {"class": classify(raw, tokens, reasoning=reasoning), "finish": ch["finish_reason"], "tokens": tokens,
             "visible": len(visible(answer)), "markers": markers(raw), "called_names": names,
             "undeclared_names": undeclared(names, tools), "raw_excerpt": excerpt(raw), "seconds": round(time.time() - t0, 1)}
        if reasoning:
            r["reasoning_markers"] = markers(rea)
        return r
    body = {"model": model, "messages": messages, "max_tokens": max_tokens, "temperature": temperature,
            "return_token_ids": True}
    if tools:
        body["tools"] = tools
    body = rm.apply(body, mode)
    d = c.call_json("/v1/chat/completions", body)
    mes = rm.measures(d, time.time() - t0)
    ch = d["choices"][0]; m = ch["message"]
    ids = ch.get("token_ids")
    if ids is None:
        sys.exit("the server does not return token_ids (return_token_ids): no witness possible on the chat path")
    raw = c.call_json("/detokenize", {"model": model, "tokens": ids})["prompt"]
    content = m.get("content") or ""
    calls = [{"name": x["function"]["name"], "arguments": x["function"]["arguments"][:120]} for x in (m.get("tool_calls") or [])]
    tokens = d["usage"]["completion_tokens"]
    reasoning = bool(mes["reasoning_chars"] or mes["reasoning_tokens"])
    rea, answer = split(raw, reasoning)
    answer_tokens = mes["answer_tokens"] if mes["answer_tokens"] is not None else tokens
    names = called_names(answer)
    r = {"class": classify(raw, answer_tokens, content, calls, reasoning, ch["finish_reason"]), "finish": ch["finish_reason"],
         "tokens": tokens, "content": len(content), "visible": len(visible(answer)), "calls": calls, "markers": markers(raw),
         "called_names": names, "undeclared_names": undeclared(names, tools), "code_block_returned": "```" in content,
         "content_start": content[:70], "raw_excerpt": excerpt(raw), "reasoning_tokens": mes["reasoning_tokens"],
         "seconds": mes["seconds"]}
    if reasoning:
        r["reasoning_markers"] = markers(rea)
    if r["class"] not in ("complete", "short_stop"):
        r.update(raw_full=raw, content_full=content)
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("label", nargs="?", default="unlabelled")
    ap.add_argument("--series", type=int, default=3)
    ap.add_argument("--per-series", type=int, default=4)
    ap.add_argument("--cases", default="hu_with_tool,fr_with_tool,fr_without_tool")
    ap.add_argument("--raw", action="store_true")
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--max-tokens", type=int, default=3000)
    ap.add_argument("--reasoning", choices=rm.MODES, default=None)
    ap.add_argument("--system", default=None, help="system message put before the prompt (none by default)")
    ap.add_argument("--self-check", action="store_true")
    a = ap.parse_args()
    if a.self_check:
        return self_check()
    import common as c
    mode = rm.resolve(a.reasoning)
    p = c.prompts("toolcall.json")
    known = dict(p["cases"], **p["provoked"], **p.get("unprompted", {}))
    names = a.cases.split(",")
    unknown = [n for n in names if n not in known]
    if unknown:
        sys.exit(f"unknown case(s): {unknown}; known: {list(known)}")
    model = c.served_model()
    path = "raw (/v1/completions)" if a.raw else "chat (return_token_ids)"
    res = c.header("toolcall", a.label, path=path, temperature=a.temperature, reasoning=mode, series=a.series,
                   per_series=a.per_series, cases=names, system=a.system)
    print(f"<tool_call> witness, {path}, temperature {a.temperature}, reasoning {mode}, system {a.system!r}")
    rows = []
    for s in range(1, a.series + 1):
        for name in names:
            prompt, with_tool = known[name]
            for i in range(a.per_series):
                r = one_request(c, model, prompt, p["tools"] if with_tool else None, a.raw, a.temperature, a.max_tokens, mode,
                                a.system)
                r.update(series=s, case=name, rank=i + 1, time=time.strftime("%H:%M:%S"))
                rows.append(r)
                print(f"  {name:16} {r['class']:20} finish {r['finish']:7} tokens {r['tokens']:5}")
    classes = sorted({r["class"] for r in rows})
    res["summary"] = {n: {k: sum(1 for r in rows if r["case"] == n and r["class"] == k) for k in classes} for n in names}
    res["summary"]["total"] = {k: sum(1 for r in rows if r["class"] == k) for k in classes}
    res["called"] = {}
    for n in names:
        mine = [r for r in rows if r["case"] == n]
        res["called"][n] = {
            "draws": len(mine), "with_undeclared_call": sum(1 for r in mine if r["undeclared_names"]),
            "with_declared_calls_only": sum(1 for r in mine if r["called_names"] and not r["undeclared_names"]),
            "without_call": sum(1 for r in mine if not r["called_names"]),
            "code_block_returned": sum(1 for r in mine if r.get("code_block_returned")),
            "length_finishes": sum(1 for r in mine if r["finish"] == "length"),
            "names": sorted({x for r in mine for x in r["called_names"]})}
    print("total: " + ", ".join(f"{k} {v}" for k, v in res["summary"]["total"].items()))
    for n, v in res["called"].items():
        print(f"  {n}: {v}")
    res["responses"] = rows
    res["requests"] = {"sent": len(rows), "errors": 0}
    c.write_result(res, f"toolcall-{a.label}.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
