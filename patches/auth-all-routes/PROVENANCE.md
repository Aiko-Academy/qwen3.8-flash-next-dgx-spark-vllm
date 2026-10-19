# Provenance: the API key required on every route

vLLM v0.32.0 checks the key only on paths that start with `/v1`, `/v2`, `/inference` or `/cohere`. On a server
started with a key, `POST /invocations` still serves inference without it (measured on our server on 2026-10-19:
200 without a key), and `/tokenize`, `/detokenize`, `/metrics`, `/version`, `/ping` and `/openapi.json`
answer without it too (vllm-project/vllm#58144, open). This overlay requires the key on every route except
`GET /health`, a liveness check that returns no data.

## Sources

| item | value |
|---|---|
| base file | `vllm/entrypoints/serve/middleware/authenticate.py` of the v0.32.0 tag (sha256 `ceb576aa`), Apache-2.0, vLLM contributors |
| upstream issue | vllm-project/vllm#58144 (2026-09-22, open) |
| change | `OPEN_PATHS = {("GET", "/health")}` replaces the prefix test; OPTIONS stays open as upstream |
| licence | Apache-2.0 |

## Checks

| check | result |
|---|---|
| middleware in the v0.32.0 image, CPU, no network (2026-10-19, 11:48) | v0.32.0 file: /metrics, /version, /ping, /openapi.json, /tokenize, /detokenize and /invocations pass without a key. This file: 401 on all of them without a key or with a wrong key, 200 with the key; GET /health open |
| served on 2026-10-19 (12:32 to 12:44) | the same, over HTTP; our clients (OpenAI and Anthropic APIs, a metrics reader that sends the key) answered |

Before you serve it: every client that reads `/metrics`, `/tokenize`, `/detokenize` or `/version` must send the
key. The benches of this repository send it when `VLLM_API_KEY` or `VLLM_API_KEY_FILE` is set.

sha256 of the overlay: `88bdb56207642cc297f4cd2f8c45a45c10ffe030911b35bcb6a4adfb00eacbcb`.
