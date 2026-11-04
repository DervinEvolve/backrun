# Backrun design brief

A backrun is what happens when fresh water hits a wash that has started to dry:
the new water pushes pigment outward and leaves a hard, cauliflower-shaped edge.
Beginners get them by accident. Painters learn to cause them on purpose. This
project trains a model to paint watercolors by giving it a simulator where the
water is in charge, and letting it get the water wrong until it stops.

Three parts, built in this order:

1. **The simulator.** A small, deterministic watercolor physics model on a grid.
   Strokes in, pigment on paper out. CPU only.
2. **The scorer.** A separate process that compares a painting to a target and
   returns numbers. The painter never sees the scorer's code.
3. **The painter.** First a search painter with no learning (so there is
   something to watch on day one), then a neural policy trained on a GPU.

Everything is reproducible from a stroke list and a seed. The repo publishes
stroke lists, not just pictures.

## 1. Simulator

Based on the physics in Curtis, Anderson, Seims, Fleischer and Salesin,
"Computer-Generated Watercolor" (SIGGRAPH 1997), simplified so that one step on
a 256 x 192 grid runs in a few milliseconds in NumPy.

### State

All arrays are `float32`, shape `(H, W)` unless noted. Pigments are a fixed
list of `K` named pigments; pigment arrays have shape `(K, H, W)`.

| Array | Meaning |
|---|---|
| `paper_height` | Paper texture, 0 to 1. Fractal noise plus a little fibre grain. Fixed per seed. |
| `water` | Water on the surface, 0 and up. A cell is wet when `water > wet_threshold`. |
| `wet_mask` | Boolean, derived from `water` each step. Fluid only moves inside it. |
| `pigment_fluid` | Pigment suspended in the water, per pigment. |
| `pigment_deposited` | Pigment dried onto the paper, per pigment. |
| `flow_u`, `flow_v` | Water velocity. Simplified: driven by the gradient of `water + slope_weight * paper_height`, damped. |

### Step order

Each `step()` runs, in this order, inside the wet mask only:

1. **Move water.** Compute flow from the gradient of water surface height
   (`water + slope_weight * paper_height`). Move water downhill with an upwind
   scheme, clamp so no cell goes negative. Damp the velocity.
2. **Move pigment.** Pigment in fluid advects with the water flow (upwind) and
   diffuses within the wet mask with a per-pigment diffusion rate.
3. **Evaporate.** Reduce `water` by a base rate everywhere wet, plus an extra
   rate at the boundary of the wet mask. Pigment in fluid is not removed, so
   it concentrates as water leaves. This is what makes edges darken.
4. **Deposit and lift.** Some fluid pigment settles onto the paper each step:
   `deposit = pigment_fluid * density_k * (1 - water_norm) * granulation_term`,
   where `granulation_term` is larger in paper valleys for granulating pigments.
   Some deposited pigment lifts back into fluid when the cell is wet:
   `lift = pigment_deposited * staining_k_inverse * water_norm`. When a cell's
   water drops below `wet_threshold`, all its remaining fluid pigment deposits.
5. **Update wet mask.**

Backruns are not coded explicitly. They should emerge: a wet stroke placed on a
damp region raises the water surface locally, water flows outward into the
damp zone, carries pigment to the moving front, and the front deposits as it
dries. If they do not emerge, say so in the README rather than faking them.

### Pigments

Rendering uses the Kubelka-Munk model as in the Curtis paper. Each pigment has
absorption `K` and scattering `S` per RGB channel. Rather than hand-typing K
and S, derive them from two colours the way the paper does: the pigment's
colour thin on white paper, and thick on black. Start with eight pigments with
plausible colours (quinacridone rose, cadmium yellow, French ultramarine,
cerulean blue, burnt umber, Hooker's green, Indian red, Payne's grey) and three
per-pigment behaviour numbers: `density` (how fast it settles), `staining`
(how hard it is to lift), `granulation` (how much it collects in paper
valleys). Values are design choices, not measurements; mark them so.

Rendering composites the deposited pigment layers over the paper colour with
Kubelka-Munk, then applies a faint paper-texture shading. Output is
`uint8 (H, W, 3)`.

### Stroke format

A painting is a JSON Lines file. Each line is one operation:

```json
{"op": "stroke", "path": [[12.0, 30.5], [40.0, 20.0], [70.0, 44.0]], "width": 6.0,
 "water": 0.8, "pigments": {"french_ultramarine": 0.6, "burnt_umber": 0.1}}
{"op": "dry", "steps": 200}
```

- `path`: 2 to 8 control points in pixel coordinates, drawn as a Catmull-Rom
  spline. `width` in pixels. The brush deposits water and fluid pigment along
  the spline with a soft falloff and a slight random edge jitter from the seed.
- `water`: 0 to 1, how loaded the brush is. Low water with pigment gives a
  dry-brush look that skips paper valleys.
- `pigments`: name to concentration, 0 to 1.
- `dry`: run the simulator `steps` times with no new input. Strokes placed
  without a `dry` between them meet wet-in-wet.

A top-level header line carries `{"op": "canvas", "width": 256, "height": 192,
"seed": 7, "paper": "cold_press"}`.

### API

```python
from backrun.sim import Watercolor
sim = Watercolor(width=256, height=192, seed=7)
sim.apply(stroke_dict)          # one op, either stroke or dry
sim.step(n=1)                   # advance physics
img = sim.render()              # uint8 (H, W, 3)
sim2 = sim.copy()               # deep copy for search painters
sim.state                       # dict of arrays, read-only use
Watercolor.from_jsonl(path)     # replay a file to its final state
```

Determinism test: replaying the same file with the same seed must produce the
same bytes.

Speed target: at 256 x 192, `step()` under 10 ms on one CPU core.

## 2. Scorer

`backrun/score.py` exposes `score(image, target) -> dict` with:

- `lab_l2`: mean squared distance in CIELAB.
- `ssim`: structural similarity on luminance, computed with SciPy, no extra
  dependency.
- `strokes`: stroke count, passed in by the caller.
- `total`: a fixed weighted sum, lower is better, documented in the README.

A learned perceptual distance (LPIPS) comes later, behind an optional extra,
and only with its weights pinned by hash. The scorer lives in its own module
with no imports from the painter, and the painter never reads the scorer's
code. That separation is the point, and the README says so.

## 3. Painter

### v0: search painter (no learning)

Given a target image resized to the canvas, build the painting greedily:

1. Coarse to fine: a few wide washes first, then medium strokes, then detail.
2. For each stroke slot, sample `N` candidate strokes. Colour comes from the
   target under the stroke's path, matched to the nearest mixable pigment
   combination. Position is biased toward regions where the current painting
   differs most from the target.
3. Simulate each candidate on a copy, score, keep the best if it improves the
   total score, otherwise skip. Candidates are judged by their dried future,
   not their wet look: each is scored on the simulator's dry preview
   (`render(predict_dry=True)`, all fluid pigment treated as deposited where
   it is), and a stroke is kept only if it lowers the predicted dry score of
   the whole painting. After each phase's real dry the painting is
   re-measured, so the ledger records both the prediction and the truth.
4. Insert `dry` ops between phases so washes set before detail goes on.

This is slow and dumb and it is enough to watch. It also provides the
baseline every learned painter must beat.

### v1: neural policy (later, on a GPU)

A small policy network that looks at the target and the current canvas and
outputs the next stroke's parameters. Train with the simulator in the loop.
Two routes, both documented when we get there: reinforcement learning
against the scorer, or supervised learning on search-painter stroke lists
followed by fine-tuning. A differentiable version of the simulator would
allow gradient training and is a contribution in its own right; it is not
required for v1.

## Observability

`backrun/record.py` writes a PNG frame every `k` steps into a run folder with
a `ledger.jsonl` of every op and the score after it. A script turns frames
into a GIF with ffmpeg. Every published result links its stroke file, its
seed and its ledger.

## Data and consent

Targets are the owner's own photos and public-domain images. No paintings by
living or dead artists are used as training data or as targets. Style comes
from the simulator's physics. The README states this plainly.

## What this project does not claim

It paints what it is shown. It learns a medium, not taste. Whether the
results look like good watercolors is for the viewer; the numbers only say
how close the painting is to the target and how many strokes it took.

## Build order

- [ ] Phase 1: simulator, pigments, stroke format, render, determinism test, speed test, three example stroke files (flat wash, wet-in-wet bloom, intentional backrun) with PNG outputs.
- [ ] Phase 2: scorer, search painter, recorder, one target painted end to end on CPU, GIF.
- [ ] Phase 3: README with results, plain English, the catch.
- [ ] Phase 4: neural policy on Modal, with the search painter as baseline.
- [ ] Phase 5: the studio room (browser, skeuomorphic) that replays ledgers.
