# Backrun

**A watercolor simulator where the water is in charge, and a painter that learns to use it.**

A backrun is what happens when fresh water touches a wash that has started to dry. The new water pushes the pigment outward and leaves a hard, cauliflower-shaped edge. Beginners get them by accident. Painters learn to cause them on purpose.

This project builds a small watercolor physics model on a grid, gives a painter a brush, a target image and a score, and lets it get the water wrong until it stops. Everything a painter does is saved as a stroke file, so any painting can be replayed exactly, stroke by stroke, bloom by bloom.

I do these experiments for fun.

## Status

Early. The simulator works and is tested. The first painter is a search with no learning, and its first painting is far from its target: the sky came out three times too dark, the horizon is a ragged band and the sun is a smudge. That is written down here because it is the starting line the learned painters have to beat.

| Part | State |
|---|---|
| Simulator: water flow, pigment movement, evaporation, settling, granulation, edge darkening | Done, 71 tests, about 5 ms per step at 256 x 192 on one CPU core |
| Pigment rendering with the Kubelka-Munk model, eight pigments | Done. The pigment numbers are design choices, not measurements |
| Stroke file format, byte-identical replay from file and seed | Done and tested |
| Scorer, kept in its own module the painter never reads | Done |
| Search painter, coarse to fine, no learning | Done, and weak |
| Learned painter trained with the simulator in the loop | Not started |
| A studio in the browser where you can watch it paint | Not started |

## What it looks like so far

Three stroke files, each rendered by replaying the file:

| Flat wash | Wet in wet | A wet stroke into a damp wash |
|---|---|---|
| ![A rectangle of ultramarine wash with a mottled centre and a thin darker rim](examples/flat_wash.png) | ![Three soft strokes, rose, blue and green, feathering into a damp yellow wash that keeps a hard outer edge](examples/wet_in_wet.png) | ![A pale patch inside a blue wash, ringed by a darker irregular band of pushed pigment](examples/backrun.png) |

The rim on the flat wash and the feathering in the second image come out of the physics; nothing in the code draws them. The third image is the honest one: a wet stroke on a damp wash gives a pale patch with a darker ring of pushed pigment, which is the start of a backrun and still a long way from the crisp branching edge of a real one. [examples/README.md](examples/README.md) records what was tried.

## How it works

The paper is a grid. Each cell holds water, pigment floating in the water, pigment that has settled onto the paper, and a paper height from a seeded texture. Each simulation step does five things, in order, only where the paper is wet:

1. **Water moves** downhill along the water surface and the paper texture.
2. **Pigment moves** with the water and spreads a little on its own.
3. **Water evaporates**, faster at the edge of the wet area. Pigment stays behind, so it concentrates where the water last was. That is why edges darken.
4. **Pigment settles** onto the paper, more in the paper's valleys for granulating pigments, and some lifts back into the water.
5. The wet area is updated.

Rendering uses the Kubelka-Munk model from Curtis, Anderson, Seims, Fleischer and Salesin, "Computer-Generated Watercolor" (SIGGRAPH 1997), which is also where the physics above comes from, simplified so one step runs in a few milliseconds. The full plan is in [docs/DESIGN.md](docs/DESIGN.md).

A painting is a JSON Lines file of strokes and drying periods:

```json
{"op": "canvas", "width": 256, "height": 192, "seed": 7, "paper": "cold_press"}
{"op": "stroke", "path": [[12.0, 30.5], [40.0, 20.0], [70.0, 44.0]], "width": 6.0, "water": 0.8, "pigments": {"french_ultramarine": 0.6}}
{"op": "dry", "steps": 200}
```

## Try it

```bash
pip install -e ".[dev]"
python -m pytest -q
```

Replay a stroke file to a PNG:

```bash
python -m backrun.cli render examples/wet_in_wet.jsonl out.png
```

Let the search painter paint a built-in test target (no downloads, about two minutes on a laptop):

```bash
python -m backrun.cli_paint synthetic:horizon:1 runs/horizon --strokes 40 --candidates 16 --seed 7
```

The run folder gets `strokes.jsonl`, `final.png`, `target.png`, a `ledger.jsonl` with the score after every stroke, and a `frames/` folder. `scripts/make_gif.sh runs/horizon/frames runs/horizon/run.gif` turns the frames into a GIF if ffmpeg is installed.

## The score

`backrun/score.py` compares a painting to its target and returns a few numbers. `total` is the one the painter tries to lower:

```
total = 0.01 * lab_l2 + 4.0 * (1 - ssim) + 0.002 * strokes
```

`lab_l2` is the mean squared colour distance in CIELAB, `ssim` is structural similarity on brightness, and `strokes` is how many strokes were used. Lower is better. The scorer lives in its own module, imports nothing from any painter, and a test checks that the painter imports only the `score` function. A learned perceptual distance may be added later, pinned by hash.

## Data and consent

Targets are my own photos, public-domain images and the built-in synthetic scenes. No paintings by any artist, living or dead, are used as training data or as targets. The look comes from the physics of water and pigment on paper.

## The catch

It paints what it is shown. Taste stays with whoever picks the targets. The numbers only say how close a painting got to its target and how many strokes it took; whether the result is a good watercolor is for you to judge by looking. Two simulators will never match pixel for pixel, so when a browser version arrives, the stroke file plus seed stays the artifact of record and the Python render of it is the official image.

## Tools

I actively use Codex, Claude, DeepSeek through Hermes, and Kimi to experiment and develop Backrun. They help me try implementations, debug problems, inspect ideas, and review changes. I also use DeepSeek through Hermes and Kimi for security audits.

## License

MIT. See [LICENSE](LICENSE).
