"""Temoin de la garde des marqueurs (patch blazux 13) sur nos neuf fichiers du parseur, hors service.

Joue le moteur de flux du parseur qwen3 comme les tests de blazux (StreamingParserEngine, sans tokenizer,
decoupages 1, 7 et 10 000 caracteres). Cas A et B : controles. C et D : bloc de code dans un element de
liste, indente de quatre puis de trois espaces. E : la reponse de production du 17/10 a 14:25:44.
"""
import json
from vllm.parser.engine.events import EventType
from vllm.parser.engine.streaming_parser_engine import StreamingParserEngine
from vllm.parser.qwen3 import qwen3_config

def parse(text, chunk, thinking):
    e = StreamingParserEngine(qwen3_config(thinking=thinking), None)
    ev = []
    for i in range(0, len(text), chunk):
        ev.extend(e.feed(text[i:i + chunk], []))
    ev.extend(e.finish())
    texte = "".join(x.value for x in ev if x.type == EventType.TEXT_CHUNK)
    appels = sum(1 for x in ev if x.type == EventType.TOOL_CALL_START)
    noms = "".join(x.value for x in ev if x.type == EventType.TOOL_NAME)
    return texte, appels, noms

APPEL = "<tool_call>\n<function=Bash>\n<parameter=command>\nls -l\n</parameter>\n</function>\n</tool_call>\n"
def indente(bloc, n):
    return "".join(" " * n + l + "\n" for l in bloc.splitlines())
piece = json.load(open("/t/piece.json"))
cas = [
    ("A bloc au niveau principal, non indente", False,
     "Voici le format :\n\n```xml\n" + APPEL + "```\n\nC'est tout.\n", "texte"),
    ("B test de blazux : ``` indente de 4 au niveau principal, appel en colonne 0", True,
     "Short thought.</think>\n\n    ```\n" + APPEL.rstrip("\n"), "appel"),
    ("C element de liste, bloc indente de 4 espaces", False,
     "1.  **Format :** par exemple :\n" + indente("```xml\n" + APPEL + "```", 4) + "2.  Fin de la liste.\n", "texte"),
    ("D element de liste, bloc indente de 3 espaces", False,
     "1. **Format :** par exemple :\n" + indente("```xml\n" + APPEL + "```", 3) + "2. Fin de la liste.\n", "texte"),
    ("E reponse de production du 17/10 a 14:25:44", False, piece["texte"], "texte"),
]
rouges = 0
for nom, thinking, text, attendu in cas:
    resultats = set()
    for chunk in (1, 7, 10_000):
        t, n, noms = parse(text, chunk, thinking)
        resultats.add((n, noms, len(t), t == text.split("</think>")[-1]))
    print(nom)
    for n, noms, lt, entier in sorted(resultats):
        print(f"   appels {n} {noms or '-'} ; texte rendu {lt} sur {len(text.split('</think>')[-1])} caracteres ; entier {entier}")
    n, noms, lt, entier = sorted(resultats)[0]
    vu = "appel" if n else ("texte" if entier else "texte coupe")
    print(f"   attendu d'une garde juste : {attendu} ; rendu : {vu} ; decoupages concordants : {len(resultats) == 1}")
    if nom.startswith("E"):
        t, _, _ = parse(text, 10_000, False)
        print(f"   egal au contenu de production ({len(piece['contenu_production'])} car.) : {t == piece['contenu_production']}")
