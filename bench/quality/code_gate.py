#!/usr/bin/env python3
"""Code-generation gate through /v1/chat/completions: four tasks that produce real code (a guarded Solidity token,
an ethers v6 script from an ABI, a typed React hook) in short context and one in long context (a 70 000-character
document made of this repository's own scripts, frozen by its sha256 in the result). Full answers are saved so that
judge_code.py can check whether the code holds.

  python3 -I bench/quality/code_gate.py <label> [--reasoning off|low|medium|xhigh|default]
      writes results/code-gate-<label>.json ; then: python3 -I bench/quality/judge_code.py results/code-gate-<label>.json
"""
import argparse, glob, hashlib, os, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import common as c  # noqa: E402
import reasoning_modes as rm  # noqa: E402

SYSTEM = {"role": "system", "content": "Tu es un ingenieur. Reponds avec du code correct et complet, pret a "
          "compiler/executer. Pas de blabla."}


def long_document(cap=70000):
    """This repository's bench and serve scripts, concatenated in path order (no .env, no data)."""
    parts = []
    for p in sorted(glob.glob(os.path.join(c.REPO, "serve", "*.sh")) + glob.glob(os.path.join(c.REPO, "bench", "*.py"))):
        parts.append("// ==== %s ====\n%s" % (os.path.relpath(p, c.REPO), open(p, encoding="utf-8").read()))
    return "\n\n".join(parts)[:cap]


def tasks(doc):
    return [
        ("solidity_guard",
         "Ecris en Solidity 0.8.x un ERC20 minimal ou transfer et transferFrom REVERTENT sauf si "
         "l'emetteur ET le destinataire sont whitelistes via un contrat externe IWhitelist obtenu par "
         "getWhitelistAddress(). Inclus l'interface IWhitelist, une custom error NotWhitelisted(address), "
         "et le gate applique aux deux fonctions."),
        ("abi_runner_ethers",
         "Voici une ABI partielle: "
         '[{"type":"function","name":"balanceOf","stateMutability":"view","inputs":[{"name":"a","type":"address"}],"outputs":[{"type":"uint256"}]},'
         '{"type":"function","name":"decimals","stateMutability":"view","inputs":[],"outputs":[{"type":"uint8"}]}]. '
         "Ecris un script Node ethers v6 qui lit RPC_URL, TOKEN et HOLDER depuis l'environnement, "
         "instancie le contrat, lit decimals() et balanceOf(HOLDER), et imprime le solde formate. "
         "Gere les erreurs proprement, process.exit(1) si une variable manque."),
        ("react_hook_ts",
         "Ecris un hook React TypeScript useChainHead(rpcUrl: string) qui interroge eth_blockNumber "
         "toutes les 5s par fetch JSON-RPC, rend {blockNumber: number|null, loading: boolean, error: "
         "string|null}, annule proprement a l'unmount (pas de setState apres demontage), typage strict."),
        ("code_long",
         "Voici un ensemble de scripts shell et python d'un depot :\n\n" + doc +
         "\n\nEcris un nouveau script bash serve/status.sh, dans le MEME style que serve/serve.sh "
         "(set -euo pipefail, configuration par SERVE_CONFIG, aucune cle en argument), qui affiche l'etat du "
         "conteneur (docker inspect : demarrage, redemarrages), les dix dernieres lignes de son journal, et "
         "interroge /v1/models avec la cle lue dans API_KEY_FILE."),
    ]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("label")
    ap.add_argument("--max-tokens", type=int, default=3072)
    ap.add_argument("--reasoning", choices=rm.MODES, default="off")
    a = ap.parse_args()
    model = c.served_model()
    doc = long_document()
    res = c.header("code-gate", a.label, reasoning=a.reasoning, temperature=0.1,
                   max_tokens=rm.budget(a.reasoning, a.max_tokens),
                   long_document={"chars": len(doc), "sha256": hashlib.sha256(doc.encode()).hexdigest()})
    res["results"] = []
    for name, prompt in tasks(doc):
        body = rm.apply({"model": model, "messages": [SYSTEM, {"role": "user", "content": prompt}],
                         "max_tokens": a.max_tokens, "temperature": 0.1, "stream": False}, a.reasoning)
        t0 = time.time()
        try:
            d = c.call_json("/v1/chat/completions", body, timeout=1800)
            msg = d["choices"][0]["message"]
            r = {"name": name, "content": msg.get("content") or "", **rm.measures(d, time.time() - t0)}
        except Exception as e:
            r = {"name": name, "error": str(e)[:300]}
        res["results"].append(r)
        print(f"{name:18} " + (f"ERROR {r['error']}" if "error" in r else
                               f"{r['answer_tokens']} tokens, finish {r['finish']}, {r['seconds']} s"))
    res["requests"] = {"sent": sum(1 for r in res["results"] if "error" not in r),
                       "errors": sum(1 for r in res["results"] if "error" in r)}
    c.write_result(res, f"code-gate-{a.label}.json")


if __name__ == "__main__":
    main()
