"""Temoin du crochet piste 3a dans l image v0.31.0, sur CPU (conteneur --network none, mtp.py accroche et jeu montes,
VLLM_MTP_DRAFT_VOCAB pose) : crochet actif, logits du jeu egaux aux logits pleins, autres a -inf, argmax egal."""
import os, types, torch, numpy as np
import vllm.models.qwen4_exp.nvidia.mtp as m
f = m.Qwen4ExpMTP.compute_logits
print("crochet actif :", f.__name__ == "_dv_compute_logits")
ids = np.load(os.environ["VLLM_MTP_DRAFT_VOCAB"]).astype(np.int64)
V, H = 248320, 64
torch.manual_seed(0)
w = torch.randn(V, H, dtype=torch.bfloat16)
self = types.SimpleNamespace(lm_head=types.SimpleNamespace(weight=w), logits_processor=types.SimpleNamespace(org_vocab_size=V))
h = torch.randn(3, H, dtype=torch.bfloat16)
out = f(self, h)
ref = h @ w.T
masque = torch.full((V,), float("-inf")); masque[torch.from_numpy(ids)] = 0
attendu = (ref.float() + masque).argmax(-1)
print("forme", tuple(out.shape), "argmax egal a l'argmax plein restreint :", bool((out.argmax(-1) == attendu).all()))
print("hors jeu a -inf :", bool(torch.isinf(out[:, masque.isinf()]).all()), "; dans le jeu egaux :", bool(torch.equal(out[:, torch.from_numpy(ids)], ref[:, torch.from_numpy(ids)])))
