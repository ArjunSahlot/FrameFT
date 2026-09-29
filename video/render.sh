#!/usr/bin/env bash
# Render every scene at 1080p30 (settings in manim.cfg), three at a time.   bash render.sh [SCENE ...]
set -euo pipefail
cd "$(dirname "$0")"
ALL="scenes_1.py:S01Intro scenes_1.py:S02Problem scenes_1.py:S03Landscape scenes_2.py:S04Frames
scenes_2.py:S05aFusion scenes_2.py:S05bFusion scenes_3.py:S06Build scenes_3.py:S07Update scenes_3.py:S08Knobs
scenes_4.py:S09Training scenes_4.py:S10Results scenes_4.py:S11Experiment scenes_4.py:S12Recap"
if [ "$#" -gt 0 ]; then
  ALL=$(for want in "$@"; do for item in $ALL; do if [ "${item#*:}" = "$want" ]; then echo "$item"; fi; done; done)
fi
mkdir -p build/logs
echo $ALL | tr ' ' '\n' | xargs -P "${JOBS:-3}" -I{} bash -c 'f=${1%%:*}; s=${1#*:}; manim "$f" "$s" > "build/logs/$s.log" 2>&1 && echo "done $s" || echo "FAILED $s"' _ {}
