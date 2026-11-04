#!/usr/bin/env bash
# Turn a folder of frame_NNNNN.png files into a GIF with ffmpeg.
# Two passes: the first builds a 256-colour palette from all frames, the
# second maps the frames onto it, which keeps watercolour gradients smooth.
#
# Usage: scripts/make_gif.sh FRAMES_DIR OUT.gif [FPS] [WIDTH] [HOLD_SECONDS]
#   FPS           frames per second (default 12)
#   WIDTH         output width in pixels, -1 keeps the frame size (default -1)
#   HOLD_SECONDS  how long to hold the last frame (default 2)
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "usage: $0 FRAMES_DIR OUT.gif [FPS] [WIDTH] [HOLD_SECONDS]" >&2
  exit 1
fi

frames_dir=$1
out=$2
fps=${3:-12}
width=${4:--1}
hold=${5:-2}

if ! command -v ffmpeg >/dev/null 2>&1; then
  echo "ffmpeg not found on PATH" >&2
  exit 1
fi
if ! compgen -G "$frames_dir/frame_*.png" >/dev/null; then
  echo "no frame_*.png files in $frames_dir" >&2
  exit 1
fi

palette=$(mktemp --suffix=.png)
trap 'rm -f "$palette"' EXIT

filters="fps=$fps,scale=$width:-1:flags=lanczos,tpad=stop_mode=clone:stop_duration=$hold"

ffmpeg -v error -y -framerate "$fps" -i "$frames_dir/frame_%05d.png" \
  -vf "$filters,palettegen=stats_mode=full" "$palette"
ffmpeg -v error -y -framerate "$fps" -i "$frames_dir/frame_%05d.png" -i "$palette" \
  -lavfi "$filters[x];[x][1:v]paletteuse=dither=sierra2_4a" -loop 0 "$out"

echo "$out"
