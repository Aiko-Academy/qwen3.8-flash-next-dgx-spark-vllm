#!/usr/bin/env bash
# Plays temoin.py against the nine tool-parser files of this repository, in a container of the pinned image with
# no network and no GPU: a serving container is not touched. Cases C and E stay red as long as the marker guard
# (upstream patch 13) does not recognise a code block indented inside a list item; A, B and D are controls.
#   patches/tool-parser/marker-guard-witness/run.sh
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
IMAGE="${IMAGE:-vllm/vllm-openai:v0.32.0}"
VP=/usr/local/lib/python3.12/dist-packages/vllm
M=()
for f in entrypoints/openai/responses/protocol.py entrypoints/openai/responses/serving.py parser/abstract_parser.py \
         parser/engine/adapters.py parser/engine/parser_engine.py parser/engine/parser_engine_config.py \
         parser/engine/streaming_parser_engine.py parser/nemotron_v3.py parser/qwen3.py; do
  M+=(-v "$HERE/../vllm-overlay/$f:$VP/$f:ro")
done
docker run --rm --network none --cpus 2 --memory 4g -e CUDA_VISIBLE_DEVICES= "${M[@]}" -v "$HERE":/t:ro \
  --entrypoint python3 "$IMAGE" /t/temoin.py 2>&1 | grep -v -E "^(INFO|WARNING|DEBUG|W[0-9]{4} )"
