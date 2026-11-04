# Backrun studio

A live three.js room where visitors watch the Backrun painter make a watercolor, replayed stroke by stroke from a
recorded run folder. Pass 1 built the room, the hero assets and the stage cameras; Pass 2 makes the room paint and
adds the dashboard, the score sheet and the 30 s demo. The plan is `../docs/STUDIO.md`.

## Run

```
npm install          # once
npm run dev          # http://127.0.0.1:5426/
```

The app opens on the Stroke camera and plays the gallery's first run (`public/runs/gallery.json`) at 1x.

- Space plays and pauses, `-` and `+` step through the speeds (0.25x, 1x, 2x, 5x, 10x, 50x, 150x), Step advances one
  1/60 s step while paused, the scrubber jumps anywhere (phase bands, a tick per stroke, a blue tick per rack dry).
- Keys `1` to `8` fly to the stage cameras in `src/data/cameras.json` (Prompt and target, Calibration, Load, Search,
  Stroke, Rinse, Dry, Reveal); `0` or `o` is the orthographic cut-room overview. Tour plays the stages in order,
  jumping the clock to each stage's event and slowing to 0.25x around it; any click or drag stops it.
- Click the brush or the arm for its card (stroke, width, water, pigments, what it is doing) with Follow the tip;
  click a well, a jar, the rack or the target print for theirs; a pigment row in the legend flies to its well. Esc or
  a click on the floor closes a card.
- Candidates toggles the rejected candidate strokes. About says what is recorded and what is staging.
- The score sheet stays face down until the last op; the guess field keeps a guess in this browser's localStorage
  (`backrun.guess.<run>`) and the revealed card shows how far off it was.
- URL options: `?run=<name>` picks a packed run, `?t=<seconds>` starts the replay clock there, `?view=<id>` opens
  on a stage camera, `?paused`, `?ao=0` turns off ambient occlusion, `?still` renders on demand at pixel ratio 1
  (stills and checks), `?figure` shows a 1.8 m mannequin for scale, `?asset=<name>` the single-asset preview.

On a phone width (under 760 px) the room takes the top half and the panels stack below it, scrolling vertically;
the stage cameras widen their field of view on a narrow screen.

## Pack a run

```
node scripts/pack-run.mjs ../runs/horizon_v1 horizon_v1
```

copies `ledger.jsonl`, `run.json`, `strokes.jsonl`, `target.png`, `final.png` and `frames/` unchanged into
`public/runs/<name>/`, adds the run to `public/runs/gallery.json`, and writes three derived files:

- `index.json`: the run settings with every score removed, op and stroke counts, the phases, the timings, and the
  frame list. The recorder does not say which op a frame belongs to, so the packer rebuilds each frame's simulator
  step from the recorder's rule (a blank frame before the first op, then one whenever the step count reaches the next
  multiple of `frames_every`; strokes are one step, settle dries run in chunks of `frames_every`, phase and final
  dries in blocks of 25 split into those chunks, and a last frame at finish). For horizon_v1 the rebuilt steps match
  the ledger at every op and the frame count matches the folder (50).
- `ops.json`: the ledger without `score`, `predicted_score`, `candidate_score`, `predicted_before_dry` or the
  rejected candidates' scores, which is what the app reads.
- `scores.json`: those scores and the final score, fetched only when the replay nears its end, so the face-down sheet
  holds in the app. The raw `ledger.jsonl` and `run.json` are still in the folder for anyone reading the files.

The root `.gitignore` ignores `runs/` and `frames/`; `studio/.gitignore` re-includes `public/runs/` so packed gallery
runs are committed with the studio. horizon_v1 is 3.8 MB.

## What moves, and from which recorded field

| What you see | Driven by |
|---|---|
| Stroke order, counter, timeline ticks | `ledger.jsonl` order, `op.op` |
| Brush tip path on the sheet | `op.path`, Catmull-Rom through the points, resampled by arc length, pixel to sheet mm (1.19 mm per pixel) |
| Which brush is in the collet | `phase` (washes: 50 mm flat, medium: size 12 round, detail: size 4 round) |
| Wells the brush dips into, mixing slant | `op.pigments` (main pigment first, a slant touch when there are two) |
| Rinse | the main pigment of `op.pigments` changing from the previous stroke |
| Palette cake levels | cumulative `pigment_added` per pigment (the most used cake ends at 30 percent) |
| Clean jar level | cumulative `op.water` |
| Rinse jar murk and tint | number of rinses so far, tinted by the pigments rinsed out |
| Sheet colour | `frames/` (see below) |
| Ghost strokes | `rejected` (path, width, pigments) on each stroke line |
| Panel to the rack and back | `dry_kind == "phase"` dries |
| Phase card | `phase`, `phase_index`, `phase_count` |
| Strokes laid, wet area, simulator steps, recorded time, candidates tried | stroke lines, `wet_pixels`, `sim_steps`, `t`, `candidates_considered` |
| Pigment legend (strokes and amount) | `op.pigments`, cumulative `pigment_added` |
| Score sheet, chart, guess difference | `scores.json` (final score and `score.total` after every op) |
| Target print, prompt card, credit slip | `target.png`, `run.json` `target_path`, `painter`, `seed` |

Timeline (`src/replay.js`): each op gets a start and a duration. A stroke lasts its recorded wall time (`t` minus the
previous op's `t`) or the time its choreography needs, whichever is longer: travel from the last stroke, a rinse and
a blot when the main pigment changes, a dip per pigment, a slant touch for two pigments, the travel down, the lay
(path length at 190 mm/s, 0.9 to 2.1 s) and the lift. A settle dry is a wait of 0.14 s per step (at least 0.28 s); a
phase dry is the carry out (2.4 s), a hold of 0.012 s per step (at least 2.4 s) and the carry back; the final dry
0.03 s per step (at least 2.2 s). horizon_v1 plays 4 min 35 s at 1x (the run took 48 s), about 5.5 s at 50x per
minute of replay.

Sheet (`src/sheet.js`): frames were saved every 20 simulator steps, so up to seven strokes fall between two frames.
At any time the sheet shows the latest frame at or before the current simulator step, with the next frame revealed
under a soft mask: the footprint of every stroke since that frame, drawn along the path up to the brush (blurred, at
the stroke's width). During dries the two frames cross-fade by simulator step. A fixed paper grain is multiplied
over it and the paper's normal map gives the tooth under the lamp.

Ghosts: during each stroke's search window (from the op's start until just before the brush reaches the sheet) its
recorded rejected candidates (up to 8) appear one by one in their mixed pigment colour with a dashed centre line,
flicker, and vanish worst first, the runner-up last, before the kept stroke lands.

Every value above is a function of the replay clock alone (no state carried between frames; the arm's inverse
kinematics starts from the same seed each frame), so any frame renders the same in any order and at any speed.
Speed changes how many 1/60 s steps run per frame, never the step.

## Staging versus recorded data

Recorded: everything in the table's right-hand column, the frames, and the order and length of the ops.

Staging (choreography derived from the ledger, stated in the app's About panel):

- the arm's moves: travel arcs, the dip in each well, the slant touch, the rinse swirl and rim blot, the hover above
  the last stroke, the lift; the simulator has no brush, palette or jar;
- the rinse rule (the main pigment changes) and the murk scale;
- the brush change at the pot in the middle of each phase dry; the brush is swapped while the collet is over the pot
  rather than placed into its slot and picked up;
- the panel's carry to the drying rack and back during the phase dries (the dry itself and its length are recorded);
- on-screen durations (the minimum per stroke, the per-step dry rates);
- the cake and jar level scales (proportional to the recorded amounts, scaled to look right);
- the sun creeping and the lamp brightening while the brush is on the paper;
- the mask that reveals the next frame under the brush, since frames are sparser than strokes.

## What is still placeholder or missing

- Not shown because the run does not record it: the water and pigment fields (the Water and Pigment layers), the
  painter's error map (Attention), calibration colours (the calibration strip keeps its fixed stripes and is not
  painted on screen), where the one skipped slot fell (the KPI says "1 slot skipped in the run"), per-op cost in
  money.
- One gallery run; the gallery drawer and switching runs from the UI are not built (`?run=` works).
- Themes (Late afternoon, Evening, North light) are not built; only Late afternoon.
- Right-click menus, the sketchbook card, visitor figures sitting down and the queue board are Pass 1 set dressing.
- Textures are still procedural canvas textures (no CC0 scans: those hosts are not reachable from the build
  machine); no baked lightmaps.
- Glassware, cup, mug, bowl, bottles, book and blind are the Pass 1 hand-built stand-ins (see `CREDITS.md`).

## Look changes in Pass 2

Soft contact shadows from N8AO (npm `n8ao`, half resolution, warm dark AO colour) in place of GTAO; softer shadow
filtering on the sun and lamp; bench top with rubbed pale patches and fine scratches in its procedural grain; paper
tooth from a multiplied grain on the sheet canvas and a stronger paper normal map; glass that thickens and greens
toward its silhouette (a small shader patch with its own program cache key); brass and nickel as anisotropic
physical metal with a brushed roughness and normal map; visible water in both jars.

## Checks and the demo

With the dev server running:

```
node scripts/probe.mjs --contacts                     # brush tip against sheet, cakes, IK error over the whole run
node scripts/probe.mjs --times 18,57 --view rinse,dry # stills into review/pass2/probe/
node scripts/stills.mjs --out review/pass2/stills     # overview and the eight stage cameras
node scripts/record-demo.mjs                          # 30 s demo, 1920 x 1080 30 fps, review/pass2/demo.mp4 and demo-preview.gif
node scripts/record-demo.mjs --scale 0.5 --out review/pass2/round1 --keep   # fast 960 x 540 preview
```

`record-demo.mjs` replaces requestAnimationFrame, performance.now and Date.now with a virtual clock advanced 1/30 s
per captured frame, drives the app only with real mouse moves, clicks, a drag and key presses (a drawn cursor and a
click ring show them), logs progress to `record.log`, runs the frame guard (`src/frame-guard.js`) on the brush, the
sheet and the score card every frame into `guard.json`, then encodes H.264 yuv420p crf 18 with faststart and a
480 px GIF preview. `review/` is gitignored.

BUDGET_PLACEHOLDER

## Files

- `src/main.js`: assembly, materials, lights, cut-room walls, cameras and flights, post, the replay clock, applying
  the replay to the scene, picking and cards, the tour, `window.__app` for checks and the demo.
- `src/replay.js`: the ledger timeline and every per-time query (frames, ghosts, amounts, panel, phase, events).
- `src/sheet.js`: the sheet canvas (frames, reveal mask, ghosts, grain).
- `src/choreo.js`: the brush stops and travel arcs.
- `src/ik.js`: damped least squares IK for the arm.
- `src/dashboard.js`, `src/dashboard.css`: the panels.
- `src/frame-guard.js`: copied from the skill's frame guard.
- `scripts/pack-run.mjs`, `scripts/record-demo.mjs`, `scripts/probe.mjs`, `scripts/stills.mjs`, `scripts/count.mjs`,
  `scripts/compress.mjs`; `blender/` as in Pass 1.
