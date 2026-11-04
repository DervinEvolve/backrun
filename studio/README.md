# Backrun studio (Pass 1: assets and stage)

A live three.js room where visitors will watch the Backrun painter make a watercolor. This pass builds the room, the
hero assets, the lighting, the stage cameras and the overview. Nothing is driven by a run yet; that is Pass 2. The
plan is `../docs/STUDIO.md`.

## Run

```
npm install          # once
npm run dev          # http://127.0.0.1:5426/
```

- Keys `1` to `8`: the stage cameras from `src/data/cameras.json` (Prompt and target, Calibration, Load, Search,
  Stroke, Rinse, Dry, Reveal). `0` or `o`: the orthographic cut-room overview. Double-click toggles
  overview and Stroke.
- Drag to orbit, wheel to zoom. The free orbit is limited: it stays in front of the bench (azimuth -85 to 85
  degrees), inside the room box, at most 3.2 m from its target and above 0.9 m once you drag away from a preset.
  The overview pans and zooms but does not rotate.
- `?still` renders one frame at pixel ratio 1 (used by the stills script), `?ao=0` turns off ambient occlusion,
  `?figure` shows a standing 1.8 m mannequin for scale, `?view=<id>` opens on a stage camera,
  `?asset=<name>&yaw=<deg>` shows one GLB on a plain floor beside a 1.8 m block and a 1 m ruler.

Review stills (dev server must be running):

```
node scripts/stills.mjs                         # overview + 8 stage cameras, 1920 x 1080, into review/pass1/
node scripts/stills.mjs --assets arm,palette    # per-asset scale previews
node scripts/count.mjs                          # triangles and bytes per GLB
```

Rebuilding assets: `npm run assets` runs `blender/build_all.py` on Blender 4.0.2 headless into `blender/out/`, then
`scripts/compress.mjs` meshopt-compresses them into `public/models/` and writes `manifest.json`. Build one family
with `blender -b --factory-startup --python blender/build_all.py -- rack`. Do not pipe `compress.mjs` into a pager
or `head`; a broken pipe leaves a half-written GLB.

## What is placeholder

- The sheet shows `public/placeholder/wet_in_wet.png`, an example simulator output, not a run. The target print,
  prompt card, credit slip and queue slips are canvas-drawn placeholder cards.
- Palette cake levels, jar water levels and the rinse jar murk are fixed staging values in `src/main.js`.
- The arm holds one static inverse-kinematics pose with the brush tip on the sheet (tip error 0.01 mm, tip on the
  sheet plane). No motion yet.
- Lights are all real time. The plan's 1K baked lightmaps for the static room are not made yet.
- Surface textures are procedural (`src/textures.js`), not CC0 scans; no KTX2 textures yet.
- The SimReady glassware, cup, mug, bowl, bottles, open book and blind could not be downloaded (see `CREDITS.md`);
  hand-built stand-ins fill those places. The two SimReady cardboard boxes under the bench are real imports.
- No UI beyond a small walnut corner panel with the view name; the dashboard is Pass 2.

## Budget used

Counted with `node scripts/count.mjs` on the shipped, meshopt-compressed files:

| File | Triangles | Bytes |
|---|---:|---:|
| arm.glb | 6,240 | 161,328 |
| bench.glb | 1,644 | 52,328 |
| board.glb | 546 | 21,016 |
| brushes.glb | 1,408 | 38,572 |
| corkboards.glb | 796 | 29,440 |
| jar.glb | 1,336 | 26,292 |
| lamp.glb | 1,480 | 36,236 |
| mannequin.glb (standing + seated) | 2,920 | 86,404 |
| palette.glb | 5,672 | 88,420 |
| props.glb | 1,850 | 67,984 |
| rack.glb | 848 | 29,840 |
| room.glb | 1,180 | 44,128 |
| shelf.glb | 2,704 | 74,628 |
| sidetable.glb | 604 | 17,996 |
| stool.glb | 484 | 16,900 |
| simready/box_cardboard_a01.glb | 554 | 55,852 |
| simready/box_cardboard_b01.glb | 560 | 71,052 |
| **Total in files** | **30,826** | **918,416** |

In the assembled scene (two jars, six stools, two boxes, one seated figure; the standing figure hidden) the renderer
draws about 40,000 triangles in 357 meshes, about 650 draw calls a frame including the sun and lamp shadow passes.
Against the plan's 30 MB room budget the models use 0.9 MB; with the placeholder image, the Sepia SVG and the app
code the whole `public/` folder is about 1 MB.

## Files

- `blender/build_all.py`, `blender/lib.py`: every hand-built asset, one GLB per family, attach points as named
  empties (listed in `blender/build_report.json`).
- `src/main.js`: assembly from `src/data/layout.json`, materials, lights, the cut-room walls, stage cameras,
  orbit limits, post (GTAO, bloom on lamp and window), the static arm pose. `window.__app` exposes `setView`,
  `views`, `renderOnce`, `stats`, `camera` and `poseOnSheet` for checks.
- `src/preview.js`: the single-asset scale preview.
- `scripts/stills.mjs`, `scripts/count.mjs`, `scripts/compress.mjs`.
