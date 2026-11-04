# Backrun

**A watercolor simulator where the water is in charge, and a painter that learns to use it.**

A backrun is what happens when fresh water touches a wash that has started to dry. The new water pushes the pigment outward and leaves a hard, cauliflower-shaped edge. Beginners get them by accident. Painters learn to cause them on purpose.

This project builds a small watercolor physics model on a grid, gives a painter a brush, a target image and a score, and lets it get the water wrong until it stops. Everything a painter does is saved as a stroke file, so any painting can be replayed exactly, stroke by stroke, bloom by bloom.

I do these experiments for fun.

## Status

Early. The simulator works and is tested. The first painter is a search with no learning. Its very first painting had a sky three times too dark, a ragged horizon and a sun like a smudge, because it judged each stroke while the paper was still wet. It now judges strokes on a prediction of how they will dry, and the same target comes out with a horizon line, a dark sea and a sky that lightens toward it. It is still a long way from the target, and that is the starting line the learned painters have to beat.

| Target on the left, painting on the right | The same run, stroke by stroke |
|---|---|
| ![A synthetic horizon scene: lavender sky, orange sun, dark teal sea, next to its watercolor: a pale sky with blue washes, an orange sun blob in place, a dark slate sea under a clean horizon](docs/assets/horizon-side-by-side.png) | ![The horizon painting appearing over forty strokes: washes first, the sea, then the sun](docs/assets/horizon-run.gif) |

Forty stroke slots, sixteen candidates each, seed 7, about fifty seconds on one CPU core. Score `total` 5.61, lower is better; the first painter scored 6.48 on the same target. One seed on one target, so a direction rather than a result.

The learned painter, after one run of imitation training on 1,000 of the search painter's paintings, on a sealed target it never saw:

| Sealed target on the left, the learned painter on the right |
|---|
| ![A synthetic horizon target with a lavender sky, orange sun and dark sea, next to the learned painter's version: a cerulean band of sky, bare paper where the gradient should be, a small yellow mark for the sun, and a dark ultramarine sea with a ragged top edge](docs/assets/learned-sealed-003.png) |

One simulator step per stroke, no search, about six seconds. It has learned the layout: sky above, sea below, the sun about where it belongs. It has not learned the pale gradient, which it leaves as bare paper, nor the sun as a disc. Score `total` 8.85 averaged over five sealed targets and three paper seeds, between a blank sheet at 24.02 and the search painter at 5.26.

| Part | State |
|---|---|
| Simulator: water flow, pigment movement, evaporation, settling, granulation, edge darkening | Done, 71 tests, about 5 ms per step at 256 x 192 on one CPU core |
| Pigment rendering with the Kubelka-Munk model, eight pigments | Done. The pigment numbers are design choices, not measurements |
| Stroke file format, byte-identical replay from file and seed | Done and tested |
| Scorer, kept in its own module the painter never reads | Done |
| Search painter, coarse to fine, no learning | Done, and weak |
| Learned painter | Stage A (imitation of the search painter) now paints. The first GPU run scored no better than a blank sheet, because a regression head averaged the search painter's varied strokes into one muddy dab. A head that picks a start cell and a bin for each other number, trained on 1,000 ledgers, scores `total` 8.85 on the sealed set against 24.02 for a blank sheet and 5.26 for the search painter, with 26 times less simulation. One run, five sealed targets, so a direction rather than a result. Numbers in [docs/TRAINING.md](docs/TRAINING.md) |
| A language model that writes stroke files from a text prompt, rewarded by a judge against a pool I rate | Code built and tested against a fake model: the gate, candidate generation with a critic, the rated pool and its rating desk, the pairwise judge, the reward checklist and the trainer script. No real model called yet, no pool rated, no training run. Plan in [docs/SCRIBE.md](docs/SCRIBE.md) |
| A studio in the browser where you can watch it paint | First pass built: a drafting arm replays a run stroke by stroke with the palette, jars, drying rack and a dashboard. Runs locally from `studio/`, not published yet. See [studio/README.md](studio/README.md) |

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

## A second painter: a language model writes the stroke file

The stroke file is a small language, so a language model can write one from a prompt such as "a harbour at dusk, two boats, low tide", and the simulator paints it in a few seconds. There is no target image to match, so the reward is a judgement: a vision model is shown the painting next to a painting from a pool I rated by hand and asked which is the better watercolour of the subject. A gate throws out files that are broken or that try to game the judge with one wide dark wash. The recipe comes from Surya Narreddi and Cameron Franz, who trained a model to paint by writing p5.brush JavaScript, and from Sergio Paniego's open reproduction; what changes here is the medium, with real water and drying time, and a render that takes seconds instead of a minute. The plan, the gate rules, the reward and what is not done are in [docs/SCRIBE.md](docs/SCRIBE.md).

## Try it

```bash
pip install -e ".[dev]"
python -m pytest -q
```

Replay a stroke file to a PNG:

```bash
python -m backrun.cli render examples/wet_in_wet.jsonl out.png
```

Let the search painter paint a built-in test target (no downloads, about a minute on a laptop):

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

Targets are my own photos, public-domain images and the built-in synthetic scenes. Every painting in the judge's pool is a stroke file written by a model and rendered by this simulator, rated by me. No paintings by any artist, living or dead, are used as training data, as targets or as references. The look comes from the physics of water and pigment on paper.

## The catch

It paints what it is shown. Taste stays with whoever picks the targets. The numbers only say how close a painting got to its target and how many strokes it took; whether the result is a good watercolor is for you to judge by looking. Two simulators will never match pixel for pixel, so when a browser version arrives, the stroke file plus seed stays the artifact of record and the Python render of it is the official image.

## Tools

I actively use Codex, Claude, DeepSeek through Hermes, and Kimi to experiment and develop Backrun. They help me try implementations, debug problems, inspect ideas, and review changes. I also use DeepSeek through Hermes and Kimi for security audits.

## License

MIT. See [LICENSE](LICENSE).
