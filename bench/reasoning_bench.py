#!/usr/bin/env python3
"""Reasoning bench under YaRN, and long context with a needle (accuracy, not speed).

Does serving 524 288 tokens with YaRN (factor 2 over 262 144) cost reasoning? Two conditions on the SAME server,
without restarting it:
  - SHORT: each of 15 items alone (short context). YaRN is active globally: a drop here is a cost of YaRN itself.
  - DEEP:  a subset of items after a neutral filler that pushes the question deep into the context, plus a
           tagged needle placed at the very top. Tests reasoning AND retrieval at depth.
Verdict: PASS if DEEP accuracy >= SHORT accuracy - 15 points AND SHORT accuracy >= 80 %.

A deep pass is a long prefill that saturates a single-model server for minutes: run it when nobody else needs it.
The witness is a correct ANSWER, never a 200. A `length` finish is counted apart (budget too short).
  python3 -I bench/reasoning_bench.py <label> [--short-only] [--depths 128000,300000] [--deep-items 3]
      [--reasoning default|off|low|medium|xhigh]            writes results/reasoning-<label>.json
"""
import argparse, os, re, sys, time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as c  # noqa: E402
import reasoning_modes as rm  # noqa: E402

SUF = "\n\nRaisonne si besoin, puis termine par une ligne EXACTEMENT ainsi : ANSWER: <reponse>"
ITEMS = [
    ("r01", "Ana a deux fois l'age de Ben. Dans 6 ans, Ana aura 1,5 fois l'age de Ben. Quel age a Ben aujourd'hui ?" + SUF, {"answer": "6"}),
    ("r02", "On lance une piece equilibree 4 fois. Probabilite d'obtenir exactement 2 faces ? Donne une fraction irreductible." + SUF, {"answer": "3/8", "alt": ["0.375"]}),
    ("r03", "Cinq coureurs finissent. Dee finit avant Eve mais apres Cal. Bo finit dernier. Al finit avant Cal. Qui finit troisieme ?" + SUF, {"answer": "Dee"}),
    ("r04", "Quel est le reste de 7^100 divise par 5 ?" + SUF, {"answer": "1"}),
    ("r05", "Combien de zeros terminent 25! (factorielle de 25) ?" + SUF, {"answer": "6"}),
    ("r06", "Un train de 300 km separe deux gares ; deux trains partent l'un vers l'autre a 70 et 80 km/h. Un oiseau a 120 km/h fait l'aller-retour jusqu'a la rencontre. Combien de km parcourt l'oiseau ?" + SUF, {"answer": "240"}),
    ("r07", "Si tous les Bloops sont des Razzies et que certains Razzies sont des Lazzies, peut-on conclure avec CERTITUDE que certains Bloops sont des Lazzies ? Reponds oui ou non." + SUF, {"answer": "non"}),
    ("r08", "Trace ce code et donne la sortie :\n\ndef f(n):\n    a,b=0,1\n    for _ in range(n):\n        a,b=b,a+b\n    return a\nprint(f(10))" + SUF, {"answer": "55"}),
    ("r09", "Combien y a-t-il de nombres entiers de 1 a 100 inclus divisibles par 3 ou par 5 ?" + SUF, {"answer": "47"}),
    ("r10", "Une chemise coute 40 EUR apres une remise de 20 pour cent. Quel etait son prix avant la remise, en EUR ?" + SUF, {"answer": "50"}),
    ("r11", "Resous pour x entier : 3x + 7 = 2x + 19." + SUF, {"answer": "12"}),
    ("r12", "Combien de secondes y a-t-il dans 2,5 heures ? Donne le nombre entier." + SUF, {"answer": "9000"}),
    ("r13", "Un nombre premier strictement superieur a 2 peut-il etre pair ? Reponds oui ou non." + SUF, {"answer": "non"}),
    ("r14", "Un rectangle mesure 12 cm sur 7 cm. Quelle est son aire en centimetres carres ? Donne le nombre." + SUF, {"answer": "84"}),
    ("r15", "J'ai 3 pommes, j'en mange une puis j'en achete le double de ce qu'il me reste. Combien de pommes ai-je ?" + SUF, {"answer": "6"}),
]
DEEP_IDS = ["r01", "r08", "r11", "r09", "r06"]
NEEDLE = "REPERE-BANC-YARN : le mot de passe du coffre fictif est GIRAFE-7731."
NEEDLE_Q = ("Tout en haut de ce document se trouve une ligne REPERE-BANC-YARN. "
            "Quel est le mot de passe du coffre fictif qu'elle donne ?" + SUF)
NEEDLE_ANS = {"answer": "GIRAFE-7731", "alt": ["GIRAFE 7731", "girafe-7731"]}


def filler(approx_tokens):
    """Neutral filler of about approx_tokens tokens (about 3.6 characters per token), numbered paragraphs."""
    base = ("Paragraphe de remplissage neutre, sans information utile a la question posee. "
            "Il sert uniquement a occuper la fenetre de contexte pour mesurer le raisonnement en profondeur. ")
    out, n, total = [], 0, 0
    while total < int(approx_tokens * 3.6):
        n += 1
        line = f"[bloc {n:05d}] {base}"
        out.append(line); total += len(line) + 1
    return "\n".join(out)


def ask(model, prompt, thinking, max_tokens, timeout, mode):
    body = {"model": model, "messages": [{"role": "user", "content": prompt}], "max_tokens": max_tokens,
            "temperature": 0, "chat_template_kwargs": {"enable_thinking": thinking}}
    if mode in rm.REASONING:
        body = rm.apply(body, mode)
    t0 = time.time()
    d = c.call_json("/v1/chat/completions", body, timeout=timeout)
    dt = time.time() - t0
    return d["choices"][0]["message"].get("content") or "", rm.measures(d, dt)


def grade(output, spec):
    m = re.findall(r"ANSWER\s*:\s*(.+)", output, flags=re.I)
    got = (m[-1] if m else output.strip().splitlines()[-1] if output.strip() else "").strip()
    got = got.strip("*`. ").replace(" ", "").lower()
    cands = [x.replace(" ", "").lower() for x in [spec["answer"]] + spec.get("alt", [])]
    # Exact first; a substring only for answers of two characters or more ("1" must not match "18500").
    return any(got == x for x in cands) or any(len(x) >= 2 and x in got for x in cands)


def run_set(model, items, thinking, max_tokens, timeout, mode, prefix=""):
    ok, detail = 0, []
    for iid, prompt, spec in items:
        try:
            out, mes = ask(model, (prefix + "\n\n" + prompt) if prefix else prompt, thinking, max_tokens, timeout, mode)
            good = grade(out, spec)
        except Exception as e:
            good, mes = False, {"finish": "error", "error": str(e)[:200]}
        ok += good
        detail.append({"id": iid, "ok": good, **mes})
        print(f"  {iid:4s} {'OK  ' if good else 'MISS'} {mes.get('seconds', 0):6.1f} s  prompt {mes.get('prompt_tokens')}"
              f"{'  LENGTH' if mes.get('finish') == 'length' else ''}", flush=True)
    return ok, detail


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("label")
    ap.add_argument("--reasoning", choices=rm.MODES, default="off")
    ap.add_argument("--max-tokens", type=int, default=2048)
    ap.add_argument("--depths", default="128000,300000")
    ap.add_argument("--deep-items", type=int, default=3)
    ap.add_argument("--short-only", action="store_true")
    ap.add_argument("--deep-only", action="store_true")
    a = ap.parse_args()
    mode = a.reasoning
    thinking = mode in rm.REASONING
    model = c.served_model()
    res = c.header("reasoning", a.label, reasoning=mode, max_tokens=rm.budget(mode, a.max_tokens), temperature=0,
                   depths=a.depths, deep_items=a.deep_items)
    res["short"], res["deep"] = {}, {}
    count = lambda det: sum(1 for x in det if x.get("finish") == "length")
    sent = errors = 0
    if not a.deep_only:
        print(f"-- SHORT ({len(ITEMS)} items) --", flush=True)
        ok, det = run_set(model, ITEMS, thinking, a.max_tokens, 300, mode)
        res["short"] = {"ok": ok, "total": len(ITEMS), "accuracy": round(ok / len(ITEMS), 3), "length_finishes": count(det),
                        "detail": det}
        sent += sum(1 for x in det if x.get("finish") != "error"); errors += sum(1 for x in det if x.get("finish") == "error")
        print(f"   SHORT accuracy {ok}/{len(ITEMS)}", flush=True)
    if not a.short_only and a.deep_items > 0:
        deep = [it for it in ITEMS if it[0] in DEEP_IDS][: a.deep_items]
        for depth in [int(x) for x in a.depths.split(",") if x.strip()]:
            pre = NEEDLE + "\n\n" + filler(depth)
            print(f"-- DEEP at about {depth} tokens ({len(deep)} items + needle): slow prefill --", flush=True)
            ok, det = run_set(model, deep, thinking, a.max_tokens, 1800, mode, prefix=pre)
            try:
                out, mes = ask(model, pre + "\n\n" + NEEDLE_Q, thinking, a.max_tokens, 1800, mode)
                needle = grade(out, NEEDLE_ANS)
            except Exception as e:
                needle, mes = False, {"finish": "error", "error": str(e)[:200]}
            print(f"  needle {'OK' if needle else 'MISS'}  prompt {mes.get('prompt_tokens')}", flush=True)
            res["deep"][str(depth)] = {"ok": ok, "total": len(deep), "accuracy": round(ok / max(1, len(deep)), 3),
                                       "needle_ok": needle, "needle": mes, "detail": det,
                                       "real_prompt_tokens": max([x.get("prompt_tokens") or 0 for x in det] + [mes.get("prompt_tokens") or 0]),
                                       "length_finishes": count(det) + (mes.get("finish") == "length")}
            allx = det + [mes]
            sent += sum(1 for x in allx if x.get("finish") != "error"); errors += sum(1 for x in allx if x.get("finish") == "error")
    acc_s = res["short"].get("accuracy")
    acc_d = min([v["accuracy"] for v in res["deep"].values()], default=None)
    reasons = []
    if acc_s is not None and acc_s < 0.80:
        reasons.append(f"SHORT accuracy {acc_s:.0%} under the 80 % floor")
    if acc_s is not None and acc_d is not None and acc_d < acc_s - 0.15:
        reasons.append(f"DEEP accuracy {acc_d:.0%} under SHORT {acc_s:.0%} minus 15 points")
    if res["deep"] and not all(v["needle_ok"] for v in res["deep"].values()):
        reasons.append("a needle was not retrieved at depth")
    res["verdict"] = "PASS" if not reasons else "FAIL"
    res["reasons"] = reasons
    res["requests"] = {"sent": sent, "errors": errors}
    print(f"== VERDICT: {res['verdict']} {'; '.join(reasons)}")
    c.write_result(res, f"reasoning-{a.label}.json")
    return 0 if res["verdict"] == "PASS" else 3


if __name__ == "__main__":
    sys.exit(main())
