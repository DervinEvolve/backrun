# Credits

Every model, texture and image the studio ships, with where it came from.

## Imported models

| File in `public/models/` | Source | Author | Licence | Link | Changes |
|---|---|---|---|---|---|
| `simready/box_cardboard_a01.glb` | NVIDIA SimReady assets, web GLB conversion | NVIDIA (conversion by codepawl) | CC BY 4.0 | https://huggingface.co/datasets/codepawl/simready-glb (source `nvidia/simready-assets`, https://huggingface.co/datasets/nvidia/simready-assets) | scaled to 0.6 in the app, colour darkened 10 percent; paper stock under the bench |
| `simready/box_cardboard_b01.glb` | same | NVIDIA (conversion by codepawl) | CC BY 4.0 | same | scaled to 0.58 in the app, colour darkened 10 percent; paper stock under the bench |

Credit line for the app's help: "Cardboard boxes: NVIDIA SimReady assets, CC BY 4.0."

### SimReady props named in the plan but not imported

The plan lists `container_jar_glass_*`, `dishware_cup_tea_a01`, `dishware_mug_a01`, `dishware_bowl_a01`,
`bottle_laboratory_b01`, `book_open_a01` and `blinds_window_2m_open_a01`. Those files are stored through Hugging
Face's Xet storage, which redirects every download to `us.aws.cdn.hf.co`; this environment's network policy refuses
that host (HTTP 403 on CONNECT), so none of them could be fetched. Only the two boxes, which are served directly from
`huggingface.co`, came through. The shelf jars, lab bottles, mug, bowl, tea cup, notebook and window blind in the
scene are hand-built in Blender (`blender/build_all.py`) as stand-ins and carry no third-party licence. To import the
real props, allow `us.aws.cdn.hf.co` in the environment's network settings and re-run the fetch.

## Hand-built

All other models (room shell, bench, drawing board and painting panel, desk arm, brushes, palette, water jars, drying
rack, stools, side table, shelf and its contents, lamp, cork boards, small props, mannequins) are built by
`blender/build_all.py` for this project, under the repository's licence.

Surface textures (oak floor, limewashed brick, limewash plaster, wood grain, cork, paper tooth) are generated
procedurally at load time by `src/textures.js`; no scanned texture is downloaded.

## Project images

- `public/brand/sepia.svg`: the Sepia mark, copied from `docs/brand/sepia.svg` (MIT, like the repository).
- `public/placeholder/wet_in_wet.png`: an example output of the Backrun simulator, used on the sheet until a run
  folder is loaded. Staging only.

## Libraries added in Pass 2

- `n8ao` 1.10.1 (npm), MIT licence, by N8python: the ambient occlusion pass.

## Run data

`public/runs/horizon_v1/` is a packed copy of the repository's own recorded run `runs/horizon_v1` (built-in synthetic
target "horizon", search painter `search_v0`, seed 7), under the repository's licence.
