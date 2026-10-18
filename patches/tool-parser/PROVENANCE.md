# Provenance: repaired Qwen tool parser (literal `<tool_call>` in prose)

Defect of the v0.31.0 parser on this model: the text that follows a literal `<tool_call>` written in an answer is
lost on the chat path, or becomes a fabricated tool call. Replaying the same generation without the parsers gives
the full text. This patch changes no decoding, no kernel, no weight, no KV.

## Source

| item | value |
|---|---|
| repository | https://github.com/blazux/qwen3.8-Flash-DGX, commit `a9ec989cdaaa813fcd3755ba87ff6c43ed7c2156` (2026-10-16), Apache-2.0 (`LICENSE-blazux`, abbreviated; full text in `../../LICENSES/blazux-Apache-2.0.txt`) |
| patch 19 | original fixes and fixtures by Chaos2522 in blazux/qwen3.8-Flash-DGX issue #31, ported by blazux; Apache-2.0 (`tests/issue31_safe_output/LICENSE` and `NOTICE` of the commit above) |
| upstream patches | `upstream-patches/`: `qwen-tool-preamble.patch` (12), `qwen-tool-marker-guard.patch` (13, requires 12), `qwen-safe-output.patch` (19, requires 12 and 13), verbatim |
| base | image `vllm/vllm-openai:v0.31.0`, files under `/usr/local/lib/python3.12/dist-packages/vllm/` (`../../serve/image-base.sha`) |

## How the overlay was made

The nine files were read from the image with `docker run --rm --entrypoint cat`, then the three patches applied in
the order 12, 13, 19 with `patch --batch --forward --fuzz=0 -p1`: three successes, no offset. The nine files compile
(`py_compile`). `vllm-overlay/` holds the result (modified files of vLLM, Apache-2.0), mounted read-only on the image
path. `EMPREINTES.sha256` lists the sha256 of the result. The upstream tests (67 methods, 21 fixtures) target a source
tree that the image does not ship: not replayed here.

## Remaining defect

The marker guard of patch 13 only recognises a code block indented by three spaces at most. Inside a list item, a
block indented by four spaces is not seen, and a `<tool_call>` example in it opens a real call or cuts the answer.
Witness without a server: `marker-guard-witness/run.sh` (cases C and E red, A, B and D as expected). Case E replays
to the byte an answer of the served model to our witness prompt (Hungarian, 2026-10-17 14:25:44, `piece.json`).
Not covered by the two deliberate exclusions of blazux's README.
