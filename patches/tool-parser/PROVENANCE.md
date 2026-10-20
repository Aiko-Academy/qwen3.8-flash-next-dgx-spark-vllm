# Provenance: repaired Qwen tool parser (literal `<tool_call>` in prose)

Defect of the v0.31.0 parser on this model: the text that follows a literal `<tool_call>` written in an answer is
lost on the chat path, or becomes a fabricated tool call. Replaying the same generation without the parsers gives
the full text. This patch changes no decoding, no kernel, no weight, no KV.

## Source

| item | value |
|---|---|
| repository | https://github.com/blazux/qwen3.8-Flash-DGX, commit `427907ac2f9ef6d5c5ef055188d6e5f71a2a033f` (2026-10-19), Apache-2.0 (`LICENSE-blazux`, abbreviated, unchanged since `a9ec989`; full text in `../../LICENSES/blazux-Apache-2.0.txt`) |
| patch 19 | original fixes and fixtures by Chaos2522 in blazux/qwen3.8-Flash-DGX issue #31, ported by blazux; Apache-2.0 (`tests/issue31_safe_output/LICENSE` and `NOTICE` of the commit above). Chat Completions follow-up of 2026-10-19 by blazux, on a report by FinSnow (issue #56) |
| upstream patches | `upstream-patches/`: `qwen-tool-preamble.patch` (12), `qwen-tool-marker-guard.patch` (13, requires 12), `qwen-safe-output.patch` (19, requires 12 and 13), verbatim |
| base | image `vllm/vllm-openai:v0.32.0`, files under `/usr/local/lib/python3.12/dist-packages/vllm/` (`../../serve/image-base.sha`) |

## How the overlay was made

The nine files were read from the v0.32.0 image with `docker run --rm --entrypoint cat`, then the three patches
applied in the order 12, 13, 19 with GNU `patch --batch --forward --fuzz=0 -p1`: no fuzz, line offsets only, except
two hunks of patch 19 in `responses/serving.py` (the Responses API changed in v0.32.0). Aiko Academy ported those two
by hand: the non-streaming path wraps `_make_response_output_items` (now called only when no streamed items exist)
in the `GenerationError` handler, and the streaming checks sit before `maybe_filter_parallel_tool_calls`. A second,
independent port (the overlay's whole diff against v0.31.0 applied to v0.32.0) gives the same eight other files. The
nine files import in the image (`tests/overlay-import.sh`). On v0.31.0 the three patches applied with no offset. `vllm-overlay/` holds the result (modified files of vLLM, Apache-2.0), mounted read-only on the image
path. `EMPREINTES.sha256` lists the sha256 of the result. The upstream tests (67 methods, 21 fixtures) target a source
tree that the image does not ship: not replayed here.

## Chat Completions follow-up (2026-10-19)

The first port of patch 19 (commit `a9ec989`, served by this recipe until v0.32.0-2) enforced the offered-name list on
both APIs. On Chat Completions, a call to a tool the request did not offer then became an empty turn ending in `stop`,
and an unoffered call followed by valid ones delivered the first valid call at index 1. blazux restored the earlier
Chat contract in `427907ac` (issue #56): known names and `functions.` / `tools.` aliases resolve first, an unknown name
is passed through unchanged for the client to reject, Responses stays strict. The follow-up touches two of the nine
files, `parser/qwen3.py` and `parser/engine/parser_engine.py` (the rejection warning now carries the bounded name). Both
were rebuilt as above, from the image files with patches 12, 13 and the new 19, no fuzz; the seven other files are
byte-identical to the first port, and the `responses/serving.py` section of patch 19 did not change, so the hand port
described above is kept as is.

Witness: blazux's `chat_compat_tests.py` and `chat_fixture.py` at `427907ac` (synthetic engine frames through the
installed Chat producer; no model, GPU or network), run in a throwaway CPU container of the image. The fixture names
the `qwen3_coder` parser; the served `qwen3_xml` maps to the same `Qwen3EngineToolParser`. On 2026-10-20, with the first
port mounted, `test_unknown_call_reaches_chat_client_instead_of_empty_stop` failed (`['stop']` instead of
`['tool_calls']`) while the bare image passed it.

## Remaining defect

The marker guard of patch 13 only recognises a code block indented by three spaces at most. Inside a list item, a
block indented by four spaces is not seen, and a `<tool_call>` example in it opens a real call or cuts the answer.
Witness without a server: `marker-guard-witness/run.sh` (cases C and E red, A, B and D as expected). Case E replays
to the byte an answer of the served model to our witness prompt (Hungarian, 2026-10-17 14:25:44, `piece.json`).
Not covered by the two deliberate exclusions of blazux's README.
