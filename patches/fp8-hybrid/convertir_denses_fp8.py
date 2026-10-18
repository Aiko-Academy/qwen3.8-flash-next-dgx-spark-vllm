#!/usr/bin/env python3
"""Convertit les couches denses BF16 du checkpoint nvidia en FP8 e4m3 par blocs 128x128, en flux.

Pourquoi pas tools/fp8_convert.py tel quel : il charge chaque shard ENTIER en memoire (load_file), soit
environ 11 Go par shard de 10 Go, quand le Spark sert le modele avec 15 Go disponibles. Ici un seul tenseur
est en memoire a la fois : l'en-tete safetensors de sortie est calcule d'abord, puis chaque tenseur est
recopie octet pour octet ou converti. Le calcul lui-meme est celui de fp8_convert.py, IMPORTE du fichier
vendorise apres verification de son empreinte (aucune recopie de la formule).

Sortie : un dossier voisin <src>-fp8hybrid. Les fichiers sans tenseur vise y sont des LIENS PHYSIQUES vers
l'original (zero octet de plus, resolus dans un conteneur qui ne monte que la sortie) ; l'index est une vraie
copie reecrite ; les shards qui portent un tenseur vise sont reecrits en entier (vLLM lit tous les tenseurs
d'un fichier). L'original n'est ouvert qu'en lecture.

Usage (dans un conteneur de l'image vLLM, CPU seul, le dossier parent monte en ecriture) :
  convertir_denses_fp8.py convertir <src> <dst>
  convertir_denses_fp8.py verifier  <src> <dst>
  convertir_denses_fp8.py etat-source <src>     empreinte stat de chaque fichier (temoin d'intangibilite)
"""
import hashlib
import importlib.util
import json
import os
import re
import shutil
import struct
import sys
import time

FP8_CONVERT_SHA = "52c80c0d6303384dd644fcb60663c28750d9c700343cf2b8258fcf8d1f890629"  # blazux@7d0798e
INDEX = "model.safetensors.index.json"
CHUNK = 64 << 20


def charger_fp8_convert():
    chemin = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fp8_convert.py")
    h = hashlib.sha256(open(chemin, "rb").read()).hexdigest()
    if h != FP8_CONVERT_SHA:
        sys.exit(f"ERREUR: fp8_convert.py modifie depuis l'audit ({h})")
    argv = sys.argv
    sys.argv = [chemin, "/inutilise"]  # fp8_convert.py lit argv[1] a l'import ; main() n'est pas appele
    try:
        spec = importlib.util.spec_from_file_location("fp8_convert", chemin)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
    finally:
        sys.argv = argv
    return mod


def lire_entete(chemin):
    with open(chemin, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        return n, json.loads(f.read(n))


def ecrire_entete(f, entete):
    brut = json.dumps(entete, separators=(",", ":")).encode()
    brut += b" " * ((8 - (8 + len(brut)) % 8) % 8)  # donnees alignees sur 8 octets, comme safetensors
    f.write(struct.pack("<Q", len(brut)))
    f.write(brut)


def oublier(fd):
    try:
        os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)  # ne pas chasser la table PLE du cache
    except (AttributeError, OSError):
        pass


def convertir_shard(fc, src, dst, cibles):
    """Reecrit un shard : cibles en FP8 + weight_scale_inv juste apres, le reste recopie a l'octet."""
    import torch

    n, ent = lire_entete(src)
    base = 8 + n
    meta = ent.pop("__metadata__", None)
    ordre = sorted(ent, key=lambda k: ent[k]["data_offsets"][0])
    sortie, pos = ({"__metadata__": meta} if meta is not None else {}), 0
    for k in ordre:
        t = ent[k]
        if k in cibles:
            o, i = t["shape"]
            if t["dtype"] != "BF16" or o % fc.BLOCK or i % fc.BLOCK:
                raise SystemExit(f"ERREUR: cible inattendue {k} {t['dtype']} {t['shape']}")
            sortie[k] = {"dtype": "F8_E4M3", "shape": [o, i], "data_offsets": [pos, pos + o * i]}
            pos += o * i
            ks = k[: -len(".weight")] + ".weight_scale_inv"
            taille = (o // fc.BLOCK) * (i // fc.BLOCK) * 4
            sortie[ks] = {"dtype": "F32", "shape": [o // fc.BLOCK, i // fc.BLOCK], "data_offsets": [pos, pos + taille]}
            pos += taille
        else:
            a, b = t["data_offsets"]
            sortie[k] = {"dtype": t["dtype"], "shape": t["shape"], "data_offsets": [pos, pos + b - a]}
            pos += b - a
    pire, nb = 0.0, 0
    tmp = dst + ".tmp"
    with open(src, "rb") as fi, open(tmp, "wb") as fo:
        ecrire_entete(fo, sortie)
        for k in ordre:
            a, b = ent[k]["data_offsets"]
            fi.seek(base + a)
            if k in cibles:
                w = torch.frombuffer(bytearray(fi.read(b - a)), dtype=torch.bfloat16).reshape(ent[k]["shape"])
                q, scale, rel = fc.block_quant(w)
                pire, nb = max(pire, rel), nb + 1
                fo.write(q.contiguous().view(torch.uint8).numpy().tobytes())
                fo.write(scale.to(torch.float32).contiguous().numpy().tobytes())
            else:
                reste = b - a
                while reste:
                    morceau = fi.read(min(CHUNK, reste))
                    fo.write(morceau)
                    reste -= len(morceau)
        fo.flush()
        os.fsync(fo.fileno())
        oublier(fi.fileno())
        oublier(fo.fileno())
    os.rename(tmp, dst)
    return nb, pire, list(sortie)


def convertir(src, dst):
    fc = charger_fp8_convert()
    if os.path.exists(dst):
        sys.exit(f"ERREUR: {dst} existe deja (verifier, ou le retirer a la main)")
    part = dst + ".partial"
    if os.path.isdir(part):
        shutil.rmtree(part)  # un essai interrompu : le dossier est le notre
    os.makedirs(part)
    idx = json.load(open(os.path.join(src, INDEX)))
    wm = idx["weight_map"]
    cibles = {n for n in wm if fc.TARGETS.search(n)}
    par_shard = {}
    for n in cibles:
        par_shard.setdefault(wm[n], set()).add(n)
    print(f"{len(cibles)} tenseurs vises sur {len(par_shard)} shards", flush=True)
    for nom in sorted(os.listdir(src)):
        p = os.path.join(src, nom)
        if os.path.isfile(p) and not os.path.islink(p) and nom != INDEX and nom not in par_shard:
            os.link(p, os.path.join(part, nom))
    pire, total, t0 = 0.0, 0, time.time()
    for i, (shard, noms) in enumerate(sorted(par_shard.items()), 1):
        nb, rel, cles = convertir_shard(fc, os.path.join(src, shard), os.path.join(part, shard), noms)
        pire, total = max(pire, rel), total + nb
        for k in cles:
            if k != "__metadata__":
                wm[k] = shard
        print(f"[{i}/{len(par_shard)}] {shard}: {nb} tenseurs, erreur max {rel:.4f}, {time.time() - t0:.0f} s", flush=True)
    idx.setdefault("metadata", {})["total_size"] = sum(
        os.path.getsize(os.path.join(part, f)) for f in set(wm.values()))
    with open(os.path.join(part, INDEX), "w") as f:
        json.dump(idx, f, indent=2)
    if pire >= 0.10:
        sys.exit(f"ERREUR: erreur relative maximale {pire:.4f} >= 0,10 (seuil de fp8_convert.py)")
    resume = {"tenseurs": total, "shards": sorted(par_shard), "erreur_relative_max": round(pire, 5),
              "fp8_convert_sha256": FP8_CONVERT_SHA, "source": src,
              "index_source_sha256": hashlib.sha256(open(os.path.join(src, INDEX), "rb").read()).hexdigest()}
    with open(os.path.join(part, ".prepared"), "w") as f:
        json.dump(resume, f, indent=2)
    os.rename(part, dst)
    print(f"termine : {total} tenseurs, erreur relative max {pire:.4f}, {time.time() - t0:.0f} s", flush=True)


def sha_fichier(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while True:
            b = f.read(CHUNK)
            if not b:
                break
            h.update(b)
        oublier(f.fileno())
    return h.hexdigest()


def verifier(src, dst):
    import torch

    fc = charger_fp8_convert()
    ko = []
    isrc = json.load(open(os.path.join(src, INDEX)))["weight_map"]
    idst = json.load(open(os.path.join(dst, INDEX)))["weight_map"]
    cibles = {n for n in isrc if fc.TARGETS.search(n)}
    attendu = set(isrc) | {n[: -len(".weight")] + ".weight_scale_inv" for n in cibles}
    if set(idst) != attendu:
        ko.append(f"cles de l'index : {len(set(idst) ^ attendu)} ecarts")
    couches = {int(n.split(".")[3]) for n in cibles}
    print(f"index : {len(idst)} cles (source {len(isrc)} + {len(cibles)} echelles), {len(cibles)} tenseurs vises, {len(couches)} couches")
    # chaque fichier de l'index existe et porte exactement les cles qui lui sont attribuees
    vu = {}
    for f in sorted(set(idst.values())):
        p = os.path.join(dst, f)
        if not os.path.exists(p):
            ko.append(f"fichier absent : {f}")
            continue
        _, ent = lire_entete(p)
        ent.pop("__metadata__", None)
        attribuees = {k for k, v in idst.items() if v == f}
        if set(ent) != attribuees:
            ko.append(f"{f} : en-tete et index divergent ({len(set(ent) ^ attribuees)} cles)")
        for k in ent:
            vu[k] = vu.get(k, 0) + 1
    doubles = [k for k, c in vu.items() if c > 1]
    if doubles:
        ko.append(f"{len(doubles)} cles presentes dans deux fichiers")
    # cibles en FP8, echelles en F32 de la bonne forme
    convertis = set()
    for n in cibles:
        _, ent = lire_entete(os.path.join(dst, idst[n]))
        s = n[: -len(".weight")] + ".weight_scale_inv"
        o, i = ent[n]["shape"]
        if ent[n]["dtype"] == "F8_E4M3" and ent[s]["dtype"] == "F32" and ent[s]["shape"] == [o // 128, i // 128]:
            convertis.add(n)
    print(f"convertis en FP8 avec echelle : {len(convertis)} / {len(cibles)}")
    if convertis != cibles:
        ko.append(f"{len(cibles - convertis)} cibles non converties")
    # liens physiques : meme inode que l'original
    shards_conv = {idst[n] for n in cibles}
    liens = [f for f in os.listdir(dst) if os.path.isfile(os.path.join(dst, f)) and f not in shards_conv
             and f not in (INDEX, ".prepared")]
    pas_lies = [f for f in liens if os.stat(os.path.join(dst, f)).st_ino != os.stat(os.path.join(src, f)).st_ino]
    print(f"liens physiques : {len(liens) - len(pas_lies)} / {len(liens)} fichiers partagent l'inode de l'original")
    if pas_lies:
        ko.append(f"non lies : {pas_lies}")

    def octets(chemin, nom):
        n, ent = lire_entete(chemin)
        a, b = ent[nom]["data_offsets"]
        with open(chemin, "rb") as f:
            f.seek(8 + n + a)
            return f.read(b - a)

    # temoin dont la reponse doit etre OUI : chaque tenseur non vise d'un shard reecrit est identique a
    # l'octet a l'original. Une seule passe par shard, qui donne aussi son empreinte.
    empreintes, compares, differents = {}, 0, []
    entetes_src = {}  # un en-tete de shard fait plusieurs Mo : le lire une fois, pas une fois par tenseur
    for sh in sorted(shards_conv):
        p = os.path.join(dst, sh)
        n, ent = lire_entete(p)
        ent.pop("__metadata__", None)
        h = hashlib.sha256()
        with open(p, "rb") as fd:
            h.update(fd.read(8 + n))
            for k in sorted(ent, key=lambda k: ent[k]["data_offsets"][0]):
                a, b = ent[k]["data_offsets"]
                garder = k not in cibles and not k.endswith("weight_scale_inv")
                if garder:
                    fsrc = isrc[k]
                    if fsrc not in entetes_src:
                        entetes_src[fsrc] = lire_entete(os.path.join(src, fsrc))
                    ns, es = entetes_src[fsrc]
                    fs = open(os.path.join(src, fsrc), "rb")
                    fs.seek(8 + ns + es[k]["data_offsets"][0])
                reste, egal = b - a, True
                while reste:
                    m = fd.read(min(CHUNK, reste))
                    h.update(m)
                    if garder and egal and fs.read(len(m)) != m:
                        egal = False
                    reste -= len(m)
                if garder:
                    oublier(fs.fileno())
                    fs.close()
                    compares += 1
                    if not egal:
                        differents.append(k)
            oublier(fd.fileno())
        empreintes[sh] = h.hexdigest()
    print(f"tenseurs non vises des shards reecrits identiques a l'original : {compares - len(differents)} / {compares}")
    if differents:
        ko.append(f"tenseur non vise modifie : {differents[:3]}")
    # dequantification de trois tenseurs convertis, contre l'original BF16
    for n in sorted(cibles)[:: max(1, len(cibles) // 3)][:3]:
        w = torch.frombuffer(bytearray(octets(os.path.join(src, isrc[n]), n)), dtype=torch.bfloat16)
        o, i = lire_entete(os.path.join(src, isrc[n]))[1][n]["shape"]
        w = w.reshape(o, i).float()
        q = torch.frombuffer(bytearray(octets(os.path.join(dst, idst[n]), n)), dtype=torch.float8_e4m3fn).reshape(o, i)
        s = n[: -len(".weight")] + ".weight_scale_inv"
        sc = torch.frombuffer(bytearray(octets(os.path.join(dst, idst[s]), s)), dtype=torch.float32).reshape(o // 128, i // 128)
        deq = q.float() * sc.repeat_interleave(128, 0).repeat_interleave(128, 1)
        rel = ((deq - w).abs().amax() / w.abs().amax()).item()
        print(f"dequantification {n} : erreur relative max {rel:.4f}")
        if rel >= 0.10:
            ko.append(f"erreur {rel:.4f} sur {n}")
    print("empreintes des shards reecrits :")
    for f in sorted(shards_conv):
        print(f"  {empreintes[f]}  {f}", flush=True)
    if ko:
        print("VERIFICATION ROUGE : " + " ; ".join(ko))
        sys.exit(1)
    print("VERIFICATION VERTE")


def etat_source(src):
    """Une ligne par fichier : nom, taille, mtime_ns, inode ; empreinte des petits fichiers."""
    for nom in sorted(os.listdir(src)):
        p = os.path.join(src, nom)
        if not os.path.isfile(p):
            continue
        st = os.stat(p)
        h = sha_fichier(p) if st.st_size < (64 << 20) else "-"
        print(f"{nom} {st.st_size} {st.st_mtime_ns} {st.st_ino} {h}")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        sys.exit(__doc__)
    cmd, args = sys.argv[1], sys.argv[2:]
    if cmd == "convertir" and len(args) == 2:
        convertir(*args)
    elif cmd == "verifier" and len(args) == 2:
        verifier(*args)
    elif cmd == "etat-source" and len(args) == 1:
        etat_source(*args)
    else:
        sys.exit(__doc__)
