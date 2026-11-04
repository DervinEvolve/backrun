Your folder: /home/user/backrun/studio (create it). Port 5426.

# Backrun studio

Subject: a live 3D room in the browser where visitors watch the Backrun painter make a watercolor, stroke by stroke,
by replaying a recorded run (later, streaming one as it happens). Show only what Backrun really has: a simulator
with water, suspended pigment, deposited pigment and a wet area; eight pigments; strokes and dry periods; a search
painter that tries candidate strokes and keeps the best; three phases (washes, medium, detail); a scorer kept apart
from the painter. Everything the studio shows comes from a run folder or is marked as staging.

Shape: live app with a dashboard (`references/brief.md`), recorded as a 30 s demo film. There is no separate film
deliverable.

Quality bar: my reference is a recorded live browser app of a robot arm painting an impasto seascape in a
dark studio, with PBR brick, bench and window light, and a dense dashboard (strokes laid, colours mixed, paint used,
painting time, pigment legend with per-colour counts, speed buttons, a card naming the painting and the current
pass). The studio reaches that level live, in real time, with PBR materials and baked plus real-time lighting, but
warm instead of dark blue. The reference is for study only and never ships.

## Look

**Lived-in isometric**, varied for one warm room.

Kept from the look: real scale; a room furnished and slightly cluttered with what a painter really owns; muted,
realistic materials with wear (paint drips on the bench, water rings, a chipped well); light tells the state; the
overview camera is a high three-quarter view with the two camera-side walls cut to a 0.3 m stub so the room reads
like a diorama; small translucent panels in the lower corners; a right-click menu on objects with their real actions.

Changed for this subject: one room, so no floor switcher; the overview is orthographic but every stage camera is
perspective (field of view 30 to 40) and close, because the subject is a 30 cm sheet; no crisis and no weather, the
"state" the light tells is the painting's phase.

The warm variation, from my Observable-Lab "quiet science" palette:

- Structure: wood `#8B7355` (bench, board, shelves, floor in a darker oiled oak), brass `#B5A642` (arm, lamp, rail,
  jar rings, pins), glass `#E8E4D9` (jars, window), warm walls `#F5F2EB` (limewash plaster; one wall of limewashed
  brick behind the board, a nod to the reference's brick wall). Paper is cold-press white `#F4F0E6`, porcelain
  `#F7F5EF` with a slight blue-grey glaze shadow.
- States, used only on UI marks and the dashboard: working `#FFB74D` (brush on paper), healthy `#7CB342` (stroke
  kept), blocked `#E57373` (slot skipped, budget cap hit), stable `#64B5F6` (dry, phase set).
- Light: late-afternoon sun through a tall window on the left wall (about 3500 K, low, long soft shadows across the
  floor and bench), a warm brass lamp over the board (about 2700 K, the brightest pool in the room), a dim cool sky
  fill from the window. Bake the static room (walls, floor, shelves) to lightmaps at 1K; keep the sun and lamp as
  real-time lights for the board, arm, jars and palette so their shadows move. Bloom only on the lamp bulb and the
  window.
- The colour in the room is the paint. Everything else stays muted so the pigments and the sheet carry the saturation.
- UI skin: the look's small corner panels, in translucent dark walnut (`#2B2219` at 88 percent) with paper-coloured
  text and brass hairline accents only on focus states; one serif for the painting's name, one sans with tabular
  figures for numbers. House UI rules hold (no eyebrow labels, no left-border callouts, no row dividers, an icon whose
  shape differs per state on every status).

Why not clean ops world: it is a cool white-grey operations floor with one teal accent seen from above, and here the
subject is one intimate desk whose whole point is colour, which a grey world with a competing accent would flatten.

Themes (one quiet button, remembered per viewer): **Late afternoon** (default, above); **Evening** (the dark theme:
sun gone, window deep blue, the lamp and a desk candle-warm bounce light the board, UI panels stay walnut); **North
light** (an even, slightly cool overcast morning with cream paper-coloured UI panels and dark ink text, for
projection and print). Each theme changes the lights, the window and the wall and floor materials as well as the UI.

## The room and its real parts

A small studio about 5.0 m by 4.0 m, ceiling 2.8 m. Window on the left (west) wall, 1.2 m wide by 1.8 m tall, sill at
0.9 m. Sizes below are working numbers; Pass 1 checks each against its reference sheet and marks any it could not
verify.

Centre, on a bench (1.8 m by 0.8 m, top at 0.76 m):

- **Drawing board**, 600 by 450 mm, 18 mm thick, tilted 12 degrees toward the viewer on a hinged prop. On it the
  **painting panel**: a 4 mm carrier board holding the **sheet**, 305 by 229 mm (a 12 by 9 inch block size, which is
  exactly the simulator's 4:3 canvas), held by kraft tape on all four sides. One canvas pixel is 1.19 mm on the
  sheet, so a 6 px stroke is 7 mm wide. The sheet shows the painting.
- **The brush holder: a small jointed desk arm**, brass and wood, like a drafting arm or an anglepoise rather than an
  industrial robot. Base plate clamped to the bench behind the board's right corner, with the **Sepia badge
  enamelled on the base plate**. Five joints: base yaw, shoulder, elbow, wrist pitch, and a collet that grips the
  brush ferrule and rolls it. Reach about 0.75 m, enough for the whole sheet, the palette, both jars and the brush
  pot. Upper and lower arm about 0.32 m each, brass tube with wooden knuckle blocks and a visible tension spring.
  Why an arm and not an unseen hand: the painter is a program, and a plain mechanism says so honestly, where a hand
  would suggest a person is painting; a floating brush hides how loading, rinsing and pressure happen; and a jointed
  arm turns every dip, rinse and brush change into travel the viewer can follow. Inverse kinematics put the brush tip
  on the stroke path every frame; the tip lowers with the stroke's width (a wider stroke presses more of the hair).
- **Brushes** in a brush pot, three of them, chosen by stroke width (washes 41 to 69 mm wide, medium 14 to 28 mm,
  detail 2.4 to 9 mm on the sheet, from the painter's phase settings): a 50 mm flat wash brush, a size 12 round and a
  size 4 round. The arm puts one back and takes the next at a phase change.
- **Porcelain palette**, about 240 by 150 mm, eight round wells (one per pigment, in the simulator's order:
  quinacridone rose, cadmium yellow, French ultramarine, cerulean blue, burnt umber, Hooker's green, Indian red,
  Payne's grey) plus two flat mixing slants. Each well holds a moist cake whose level drops as that pigment is used.
  A two-pigment stroke shows the brush touching both wells and the mix on a slant.
- **Two water jars**, glass, about 90 mm across and 110 mm tall (about 500 ml), with brass rim rings. The **clean
  jar** loses level as water is loaded onto the brush (the sum of stroke water loads). The **rinse jar** clouds as the
  brush is rinsed, and its murk is also the **cost meter**: its opacity follows running cost, with a brass ring
  etched on the glass at the budget cap. Tint comes from the pigments rinsed so far.
- **Calibration strip**, a small card of eight stripes leaning on the palette: the search painter really paints one
  pigment calibration sheet before it starts, and the strip shows its measured colours.
- **Drying rack**, left of the board, a wooden rack about 0.60 m wide, 0.30 m deep, 0.45 m tall with three slots.
  Between phases the painting panel rides a short brass rail from the board into the rack, the water sheen retreats
  there, and it rides back. Short settle dries inside a phase stay on the board.
- **Score sheet**, a small card face down beside the board on the bench, revealed at the end.
- **Target print**, pinned with two brass pins on a cork strip above the board, the same 4:3 size as the sheet; beside
  it the **prompt card** (the prompt text, or "Built-in test scene: horizon" or "Owner's photo" for targets that had
  no prompt) and a small credit line for the image's source.
- **Lamp** over the board, brass, on a wall bracket.

Around the room:

- **Shelf of sketchbooks**, one per painter version (today one: `search_v0`; later each learned painter gets its own),
  with the Sepia badge on each spine. Clicking one opens its card: version, runs in the gallery, revealed scores.
- **Paper drawer** under the bench holding fresh sheets; a new run's sheet comes from here.
- **Queue board** on the right wall, cork with paper slips: waiting prompts in tier (b); in tier (a) it lists the
  gallery's recorded paintings.
- **Stools** for visitors, six to eight, along a low rail behind the bench, each with a simple wooden-mannequin figure
  and a paper name tag when a visitor sits there.
- Set dressing: a side table with an open notebook, a tea cup, bottles and jars on the shelf, a plant on the sill, a
  blind on the window, boxes of paper under the bench (small pieces from SimReady, wooden pieces hand-built; below).

The mascot: **Sepia, a small cuttlefish**, the project's own mark (source `docs/brand/sepia.svg`, MIT like the repo).
Sepia pigment was made from cuttlefish ink, and a cuttlefish mixes colour on its own skin. Round body seen from the
side in Payne's grey ink wash with a warm ochre underside, big calm eye with a W-shaped pupil, one front arm holding a
round brush, a faint watercolor blot behind as the badge ground; flat, two or three tones, readable at 24 px. It
appears enamelled on the arm's base plate, on the sketchbook spines, on the loading screen and in the dashboard's
corner. Nowhere else, and never animated as a character.

## What moves

- The arm, continuously: brush change at phase starts, dip in one or two wells, touch the mixing slant for a two-pigment
  stroke, travel, lower, follow the stroke's Catmull-Rom path at the recorded width, lift, rinse when the next stroke's
  pigments differ, blot on the jar rim.
- The sheet, from recorded frames: the painting appears under the brush as it moves (see Data, "Frames").
- Palette well levels, the clean jar level, the rinse jar murk and tint.
- The painting panel on its rail to and from the drying rack between phases.
- The calibration strip being painted at the start (one stripe per pigment, revealed in turn).
- The score sheet turning over at the end.
- The light: sun angle creeps over the run (a function of the replay clock, not the real clock); the lamp brightens
  when the brush is on the paper. Theme changes swap the whole setting.
- Hidden layers, each behind one toggle, off by default (below).
- Visitors sitting down and leaving (tier (b) only), slips appearing and leaving the queue board.

Staging, stated in the app's help: dips, rinses, brush changes and the panel's ride to the rack are choreography the
studio derives from the ledger; the simulator has no rinse and no rack. The stroke path, width, water, pigments, the
order of ops, the drying and the frames are recorded fact.

## Hidden layers made visible

Switchable from the dashboard, one toggle each, off by default. Each is labelled with the value it shows.

1. **Water**: a glassy sheen on the sheet where it is wet, from the recorded low-resolution water field, driving the
   sheet material's clearcoat and roughness so the window and lamp reflect in it. It retreats as the sheet dries,
   fastest at the edges, as the simulator's edge evaporation does. Label: wet area in cm² and percent.
2. **Pigment in suspension**: motes drifting just above wet cells, count from the recorded fluid pigment, colour from
   the dominant pigment in that cell, drift speed from recorded flow speed; they slow and sink as pigment deposits,
   and vanish where the cell dries. Large, soft motes with short trails, never a haze.
3. **The painter's attention**: faint warm heat on the sheet where the painting differs most from the target, from
   the painter's own error map for that slot (the map it draws candidate positions from). Label: "where the painter
   measures the most difference".
4. **Rejected candidates**: during each slot's search time, the candidate strokes it tried appear as faint ghost
   strokes in their pigment colour that flicker and vanish one by one; the kept one solidifies into the brush path.
   A slot where nothing helped shows all ghosts fading and a "slot skipped" mark.

## The score and the prediction game

The score sheet lies face down beside the board. It shows nothing until the run ends; then it turns over with
`total`, `lab_l2`, `ssim` and the stroke count, and a small line chart of `total` after every op from the ledger.
Lower `total` is better; the card says so in plain words and gives the README's formula.

Before the reveal, visitors may guess the final `total` (a slider with a short note on what the numbers mean, and the
gallery's past results as a guide). After the reveal the card shows their guess against the result. In tier (a) a
guess lives only in the visitor's browser; a shared tally of everyone's guesses needs tier (b)'s backend.

Honest wording for the card: the painter never reads the scorer's code, but the v0 search painter does call the score
function as a black box to rank its candidates. The face-down sheet hides the score from visitors, not from the
painter. The ledger holds a score after every op, so the studio's packer moves all scores into a separate file that
loads only at the reveal.

## Dashboard

The reference's density, in the look's corner panels:

- **Painting card**, top left: the painting's name, the target thumbnail, painter version, and the phase card
  ("Washes, phase 1 of 3", "Detail, phase 3 of 3"), with "stroke 12 of 40" and the slot being searched.
- **KPI strip**: strokes laid (with slots skipped), wet area (cm² and percent), painting time in simulator steps and
  in recorded wall time, running cost (compute seconds now, money in tier (b)), and the score once revealed. Every
  number is computed from the ledger at the current replay time and counts to its new value.
- **Pigment legend**, bottom left: eight rows, each a swatch from the calibration strip, the pigment name, strokes
  using it, and amount used as a bar and a number. Clicking a row flies the camera to that well.
- **Layers**: four toggles (Water, Pigment, Attention, Candidates).
- **Timeline strip**, bottom: play and pause (Space), Step (one fixed step while paused), a scrubber with the three
  phase bands, a tick per stroke and a mark per skipped slot, and speeds 0.25x, 1x, 2x, 5x, 10x (keys - and +) plus
  50x and 150x for long runs. "Slow down at events" is on in the tour.
- **Stage bar**, numbers 1 to 8 (below), with Next, Previous and Tour.
- **Score card**: face down, the guess slider, then the reveal.
- **Gallery and queue**: a drawer of prebuilt paintings (the "sites": choosing one switches the run, target, prompt
  and sheet) and, in tier (b), a prompt box and the queue.
- Sepia in the dashboard's corner, small.
- Detail cards open on click for the arm and brush (current op, width, water, pigments), each well, each jar, the
  rack, a sketchbook, the target, the prompt card, the score sheet and a visitor's tag. Esc or a click on empty
  floor closes a card. "Follow" on the brush keeps the camera on the tip at close, mid and wide.

## Data: the seeded simulation is the ledger

The brief's seeded simulation is the Backrun run folder itself. The studio replays recorded runs deterministically;
the same run gives the same frames at every speed. A later live mode streams a run folder while it is being written.

Replay clock: a fixed-step studio clock (1/60 s steps). The timeline is built once from the ledger: each op gets a
segment whose length is its recorded wall time (search time for a stroke, simulated drying for a dry), with a floor
so a stroke's dip, travel and lift are always readable (about 1.2 s at 1x). At 1x a v0 run plays about as long as it
took (a 40-stroke run took about two minutes). Speed changes how many fixed steps run per frame, never the step.

What drives what, from today's files:

| Source | Field | Drives |
|---|---|---|
| `strokes.jsonl` line 1 | `width`, `height`, `seed`, `paper` | sheet size in pixels, the paper's grain card |
| `ledger.jsonl` | `index` | stroke counter, timeline ticks |
| | `t` | segment lengths, wall-time KPI |
| | `sim_steps` | simulator-steps KPI, which frame is current |
| | `op.op` (`stroke` or `dry`) | arm action or drying |
| | `op.path` | brush tip path (Catmull-Rom through the points, pixel to sheet mm) |
| | `op.width` | brush choice, how far the tip presses |
| | `op.water` | clean jar level, how wet the brush looks |
| | `op.pigments` | which wells, mixing slant, rinse decision, legend counts |
| | `op.steps` on `dry` | drying length; long dries between phases send the panel to the rack |
| | `score` | moved to the reveal file; the line chart on the score card |
| `run.json` | `canvas`, `seed`, `painter`, `strokes_requested`, `candidates_per_stroke` | painting card, sketchbook shelf |
| | `frames_every`, `frames` | frame lookup |
| | `timings` (`calibration_s`, `phase_*_s`, `paint_s`, `elapsed_s`) | phase bands, cost |
| | `painter_stats` (`strokes_accepted`, `slots_skipped`, `evaluations`) | KPI totals, cost |
| | `final_score` | the score sheet |
| | `status` | live mode: running or finished |
| `frames/frame_NNNNN.png` | | the sheet's colour |
| `target.png`, `final.png` | | target print; the last frame |

Frames: today a frame is written every 20 simulator steps and a stroke counts as one step, so several strokes can
fall between two frames. The studio must not invent paint. Until the recorder writes a frame after every stroke, the
sheet blends from one frame to the next under a mask that follows the brush tip along each stroke's footprint, and
the help text says so. With per-stroke frames the mask only reveals the true next frame as the brush passes.

What today's files cannot draw: phase names, per-pigment amounts, wet area, the water and pigment fields, the
attention map, rejected candidates, skipped slots, calibration colours and per-op cost. Those are the recorder
changes at the end. Until they exist, the phase is inferred from the long dries (settle dries are 2 or 3 steps,
phase dries are up to 600), amounts are estimated as concentration times water times footprint area and marked
"estimate", and the four hidden layers stay greyed out with a note.

Live mode: a small server tails a run folder (the recorder already appends one ledger line per op) and pushes new
lines and frames over server-sent events or a WebSocket; the clock follows the live edge, visitors can scrub back,
and the score sheet stays face down until `status` is `finished`.

## Visitors and prompts

Visitors join the studio and can ask for a painting, in two tiers.

**(a) Gallery of prebuilt prompts**, first. Each entry has its prompt, its target image already generated, and its
run already recorded and packed. Choosing one replays at once. This is a static site: the app plus packed run folders
on any static host, no server and no running cost beyond hosting.

**(b) Free-text prompts**, later. A prompt goes to a queue; it is moderated; an open text-to-image model makes the
target at the canvas's 4:3 size; the trained painter paints it with the recorder on; the studio streams the run as it
happens. This needs a small backend: a queue, moderation, a GPU worker, storage for run folders, and a live stream.

Who did what, stated plainly in the app: the picture's content comes from a separate open text-to-image model,
credited by name and licence on the prompt card and in the credits (I pick the model; its name, licence,
version and generation seed go into `run.json`); the watercolor handling is the painter's. Both the prompt and the
generated target hang above the board so nobody mistakes which model did what.

Presence (tier (b) only, since it needs shared state): other visitors appear as wooden figures with name tags on the
stools, at most eight shown, the rest as a count. The queue board lists waiting prompts as slips with a position
number; a slip moves to the cork strip above the board when its run starts.

Open questions:

1. Per-prompt GPU cost (image generation plus painting) and a budget cap per day and per visitor; what the studio
   shows when the cap is hit (the rinse jar reaches its etched ring and the queue pauses).
2. Prompt moderation before generation: which filter, what is refused, and whether prompts naming a living artist
   or "in the style of" a named artist are refused, as the project's data-and-consent rule suggests.
3. Tier (b) needs a small backend; tier (a) is a static site. Who runs and pays for the backend, and when.
4. Which open text-to-image model, under a licence that allows this use, and its credit line.
5. Generated targets are neither my photos nor public-domain images, so the README's "Data and consent"
   section needs a line about them before the gallery ships. Visitors choosing prompts also moves "taste" from the
   owner to visitors, which the README's "The catch" should mention.
6. Whether shared guesses for the prediction game are wanted in tier (b).

## Assets

Small props come from the NVIDIA SimReady props in their web GLB conversion on Hugging Face (dataset
`codepawl/simready-glb`, 992 GLB files, about 1.4 GB in all, CC BY 4.0 confirmed on NVIDIA's source card; credit
NVIDIA, source `nvidia/simready-assets`). The catalog is industrial, hospital, office and food props: it has no stool,
wooden table, open shelf, lamp, easel, drawing board, plant or rug, and its chairs are office and hospital models that
look wrong in a studio. So SimReady supplies small glassware, a window blind, a book and boxes; everything wooden or
painterly is built by hand. Fetch only the files listed (about 5 to 7 MB before recompression), once I say yes; rescale each to the size
below, restyle its materials to the warm palette, and record it in `CREDITS.md`.

| SimReady file | Use | Size in metres (W x D x H) | File size |
|---|---|---|---|
| `container_jar_glass_*` (pick 2 to 3 of the 8) | jars on the shelf; reference and fallback for the hero water jars | 0.09 x 0.09 x 0.11 | 0.3 to 0.5 MB each |
| `dishware_cup_tea_a01` | tea cup on the side of the bench | 0.09 x 0.09 x 0.06 | 0.1 MB |
| `dishware_mug_a01` | mug holding spare brushes on the shelf | 0.09 x 0.09 x 0.10 | 0.1 MB |
| `dishware_bowl_a01` | small bowl of rags or sponges | 0.15 x 0.15 x 0.06 | 0.1 MB |
| `bottle_laboratory_b01` (x2 to 3) | gum arabic and masking fluid bottles on the shelf | 0.05 x 0.05 x 0.12 | 0.1 MB |
| `book_open_a01` | open notebook on the side table | 0.30 x 0.22 x 0.03 | 0.8 MB |
| `blinds_window_2m_open_a01` | blind on the window, scaled to it | 1.20 x 0.05 x 1.80 | 1.9 MB |
| cardboard boxes and crates (1 to 2) | paper stock under the bench | 0.50 x 0.35 x 0.30 | small |
| `desk_a01` (optional) | only if restyled to wood reads right; otherwise skip | 1.40 x 0.70 x 0.75 | 1.7 MB |

Hand-built in Blender instead, from reference sheets, at these sizes: the wooden workbench (1.80 x 0.80 x 0.76),
visitor stools (0.35 diameter x 0.65), the open shelf (0.90 x 0.30 x 1.80), a side table (0.60 x 0.45 x 0.70), the
cork queue board (1.20 x 0.02 x 0.80), a rope rail with posts (2.40 long, 0.90 high), a plant on the sill (0.25 x
0.25 x 0.45, simple), and the visitor figures (wooden mannequins, 1.0 m seated height).

Hero objects, built by hand in Blender from a reference sheet each (`references/real-objects.md`): the porcelain
palette, the three brushes, the paper block and painting panel with tape, the two water jars, the drying rack, the
desk arm with its collet and spring, the drawing board on its prop, the lamp, the score card, the calibration strip
and the sketchbooks. The arm's joints are separate named objects with pivots at the origin; attach points exported as
named empties: brush tip, collet, each well centre and cake surface, each mixing slant, each jar's rim and water
surface, the brush pot slots, the rack slots, the rail ends, the sheet's four corners.

The Sepia badge SVG (from `docs/brand/sepia.svg`) and a 3D enamel-pin version of it are hero assets built by hand.

Room shell (walls, window with frame and muntins, floor, brick wall, skirting) is built in Blender; surface materials
from CC0 scans (Poly Haven, ambientCG): oiled oak floor, limewash plaster, limewashed brick, worn bench wood, brass,
porcelain, cold-press paper normal map. Textures at 1K, compressed (KTX2), lightmaps at 1K.

Blender 4.0.2 is installed, while Pass 1's brief names Blender 5.x. Run Pass 1 on 4.0.2 and check that the glTF
exporter options it relies on (NLA-track animation export) exist in that version; mark anything that differs in
`ASSETS.md`.

Budget: the whole room under 30 MB. Working split: hand-built heroes and furniture 10 MB, SimReady props 5 MB
(restyled and recompressed, skipping `desk_a01` if not used), room shell and lightmaps 7 MB, textures 6 MB, UI and
fonts 2 MB. Each packed gallery run is extra (frames as WebP, about 1 to 2 MB per run)
and loads only when chosen.

Bench-scale checks from the brief: brush tip within 1 mm of the sheet while a stroke is laid, collet on the ferrule,
brush in the well and jar within 1 mm of the surface it touches, panel on its rail, no surfaces sharing a plane
(sheet 0.5 mm above the carrier, tape 0.2 mm above the sheet, carrier above the board).

## Stages and stage cameras

Each stage has a perspective camera preset in one data file, number keys 1 to 8, framed so the thing doing the work
fills about a quarter of the frame, from the side that shows the moving part.

1. **Prompt and target**: front, slightly low, on the cork strip with the target print and prompt card. Event: the
   target is pinned.
2. **Calibration**: close on the strip by the palette as each stripe is painted.
3. **Load**: from the right, over the palette, brush dipping into a well and onto the mixing slant.
4. **Search**: square-on above the sheet, attention and candidate layers on; event: the kept candidate solidifies.
5. **Stroke**: low three-quarter from the right across the tilted board, following the brush tip; event: the brush
   lifts at the end of the path.
6. **Rinse**: close on the two jars, the brush swirling, the murk rising.
7. **Dry**: on the drying rack from the left with the window behind, water layer on; event: the sheen is gone.
8. **Reveal**: over the bench toward the score sheet and the finished sheet; event: the card turns over.

The camera view that shows the subject best: a low three-quarter view from the front right across the tilted board
(stage 5), close enough that the sheet fills about a third of the frame, with the arm, the palette and the jars in
the edge of the frame and the window light raking across the wet paper. The orthographic cut-room overview is for
orientation and for seeing the visitors and the queue board.

**Guided tour**: the Tour button plays stages 1 to 8 in order, holding each until its event has happened once (it
jumps the clock to the next instance of that event, slowing to 0.25x for the second around it), then moves on; any
click or drag stops it. On a finished replay the tour ends with the reveal; in live mode it skips stage 8 until the
run is finished.

## The 30 s demo recording

Recorded from the live app with the brief's virtual clock and real clicks; it is the demo of the app, not a separate
film.

- **0-5 s**: slow orbit from the window side over the room in late light, the arm laying a wide wash, the lamp pool
  on the board, Sepia catching light on the arm's base plate.
- **5-12 s**: click the brush; its card opens (stroke 9 of 40, cerulean and ultramarine, water 0.84) and updates as
  the camera follows the tip along the stroke; turn on the Water layer and the sheen spreads behind the brush.
- **12-18 s**: click "French ultramarine" in the pigment legend; the camera flies to its well as it drops a step, the
  amount counts up; Candidates layer on for one search, ghosts flicker and one solidifies.
- **18-24 s**: open the gallery and choose another painting; a fresh sheet comes from the drawer, the new target and
  prompt are pinned, and the speed goes to 50x with the control on screen while the washes fill and the panel rides to
  the rack and back.
- **24-30 s**: pull back to a wide three-quarter view where the room fills the frame; a guess is set on the score
  card and the sheet turns over with the result.

## Recorder changes needed

For `backrun/record.py` and its caller in `backrun/painter_search.py`, so the studio can draw the legend, the phase
card and the hidden layers. All are additions; existing fields stay as they are.

1. A frame after every stroke op (and every `frames_every` steps during dries), and a `frame` field on each ledger
   line naming the frame written after that op.
2. `phase` (name), `phase_index` and `phase_count` on every ledger line, and `dry_kind` on dry ops (`settle`,
   `phase`, `final`).
3. Per-stroke `pigment_added`: the amount of each pigment actually put on the sheet, from the change in the
   simulator's total pigment per pigment across the op; and `water_added`.
4. `wet_pixels` after every op (count of wet cells), and the same at every saved frame.
5. Per frame, small fields beside the PNG (64 x 48, 8-bit PNG or a packed array): water depth, total fluid pigment,
   dominant fluid pigment index, and flow speed.
6. Per stroke slot, the candidates tried: each candidate's op, its rank, and whether it was kept; and a ledger line
   with `op: "skip"` for slots where nothing helped. Candidate scores, if recorded, go to a separate scores file.
7. Per slot, the painter's error map downsampled to 32 x 24, 8-bit.
8. Per op `search_s` (wall time spent on candidates) and `evaluations` (candidates simulated), for the cost meter.
9. In `run.json`: the calibration colours per pigment, the painter version and code commit, the machine type, and for
   generated targets the prompt, the image model's name, version, licence and seed.
10. Optional: a recorder flag that writes all scores (per op and final) to `scores.jsonl` instead of the ledger, so
    the face-down sheet holds even for someone reading the raw files.

## Pass 1 notes (assets and stage)

What Pass 1 changed against the plan above, and what it could not do:

- **SimReady props blocked.** Every SimReady file except the small cardboard boxes is stored in Hugging Face's Xet
  storage and downloads from `us.aws.cdn.hf.co`, which my cloud environment's network policy refuses. The glass jars,
  tea cup, mug, bowl, lab bottles, open book and window blind are hand-built stand-ins in `blender/build_all.py`.
  The two boxes (`box_cardboard_a01`, `box_cardboard_b01`) are imported and credited in `studio/CREDITS.md`; the
  third, loose box on the floor was dropped since the plan wants one or two. Allowing that host would let a later
  pass swap the stand-ins for the real props.
- **Drying rack turned.** The rack is now 0.56 m wide along the bench, 0.30 m deep, 0.45 m tall, closed at its west
  end and open toward the board, with three slots running along x, so the panel slides in edge-on along the brass
  rail as it leaves the board. To fit it beside the board on the 1.80 m bench, the board, lamp and cork strip moved
  0.10 m east (board centre x = 0.07), the rack sits at x = -0.60, and the score card moved to the front right of the
  board. The rail's board end meets the board's left edge (x = -0.23).
- **Overview cut.** The overview looks from the south-west, so the window wall and the south wall drop to their
  stubs; the sill and the plant on it hide with the window wall. The cut walls still cast the window's shadow (they
  stay in the shadow pass on layer 1) but are kept out of the ambient occlusion pass.
- **Orbit.** The free orbit stays in front of the bench (azimuth -85 to 85 degrees), inside a box just inside the
  walls, and within 3.2 m of its target.
- **Seated figure.** The seated mannequin's thighs pointed backward; it now sits forward with its feet on the
  stool's foot ring.
- **Not done in Pass 1:** baked 1K lightmaps, CC0 scan textures and KTX2 (surfaces are procedural canvas textures),
  the Sepia enamel pin as a separate 3D asset beyond the base-plate badge, and the reference sheets in
  `studio/blender/refs/` (the folder is empty; the hero sizes follow the dimension tables above and were checked
  against a 1.8 m block and a 1 m ruler in `?asset=` previews, not against reference photos).

## Pass 2 notes (replay, dashboard, demo)

What Pass 2 changed against the plan above, and what it could not do:

- **The ledger now carries most of the recorder changes.** `phase`, `phase_index`, `phase_count`, `dry_kind`,
  `pigment_added`, `wet_pixels`, `candidates_considered`, `search_s` and up to eight `rejected` candidates with their
  paths are on the ledger lines, so the phase card, the legend amounts, the wet-area KPI and the ghost strokes come
  from recorded fields, not estimates. Still missing: a frame per stroke (frames stay every 20 steps, so the sheet
  reveals the next frame under the brush's footprint), the per-frame water and pigment fields, the error map, the
  calibration colours, skip lines for skipped slots, and cost in money.
- **Frame steps are rebuilt.** The recorder does not say which op a frame belongs to; `scripts/pack-run.mjs` replays
  the recorder's frame rule over the ledger (including the blank frame `cli_paint` saves before the first op) and
  checks it against `sim_steps` and the frame count.
- **Scores split off at packing.** The app reads `ops.json` (no scores) and fetches `scores.json` only near the end.
- **On-screen timing.** Each stroke takes the longer of its recorded wall time and its choreography (dips, a rinse
  when the main pigment changes, the lay at 190 mm/s with a 0.9 s floor), so a 48 s run replays in about 4.6 minutes
  at 1x; 50x and 150x cover it quickly.
- **Brush change** happens at the pot during each phase dry, swapped while the collet is over the pot (staging).
- **Hidden layers.** Only Candidates is built (on by default so the ghosts are seen; the plan has all layers off by
  default). Water, Pigment and Attention need the per-frame fields and the error map.
- **Not built in Pass 2:** the three themes, the gallery drawer (one run so far; `?run=` switches), the calibration
  strip being painted, right-click menus, and the sketchbook and visitor cards. AO is now N8AO instead of GTAO.
- **The demo's beat 18-24 s** plays the panel's ride to the rack at 1x after a 5x stretch, then 50x through the
  medium phase, instead of switching to a second gallery painting (there is only one).

### Pass 2 review rounds (recording and self-review)

- **Recording.** The interrupted take had reached frame 455 of 900; memory stayed near 1 GB through a full take, so
  the stop was the container, not a leak. Takes are half scale (960 x 540, 30 fps, about 17 minutes each on this
  box), encoded at the captured size under `nice -n 19`, with a 360 px GIF under 4 MB. A dev-server reload during a
  take restarts the app on a fresh clock, so the recorder now stops the take if the page navigates.
- **Round 1** found the panel swinging through open air on its way to the rack, the opening too close for the arm's
  raised moves, a 45 degree roll on the flight to Search (the preset's up vector set at once), Follow snapping its
  distance. **Round 2** found the camera sweeping through a post after too large a pull-back, and two KPIs reading
  against the run files (624 candidates against 640 evaluations, 48.5 s "of 48.4 s"). **Round 3** left one guard
  fault (brush above the frame for the first 0.7 s) and no camera frame inside a mesh box.
- **Panel carry, changed from the plan.** Instead of lifting the panel and swinging it to the rail, it pivots upright
  on its front edge on the board, slides back upright on the board to the rail's line, then west and down about 2 cm
  onto the rail into the rack, seated from the carrier's own vertices (within 0.1 mm of the board plane or the rail).
  It is still staging with no carrier in the scene; the dry and its length are recorded.
- **KPIs.** Candidates tried counts the laid strokes' `candidates_considered` and says "of 640, 16 in the skipped
  slot" from `painter_stats.evaluations`; recorded time is measured against the last ledger `t`.
- **Measured** (software rendering, no GPU on the build box): 0.7 fps on the overview and 1.3 fps on the Stroke
  camera at 1280 x 720; about 40,000 triangles in view (104,000 drawn with the two shadow maps), 885 and 618 draw
  calls with shadow passes, 358 meshes; JS heap 11.9 MB. Draw calls are the next thing to cut.
- **Still open:** the full-size take, a GPU frame rate, the plan's Water layer and second-painting beats (no data and
  one run), and the Reveal camera leaving the brush at the frame's left edge.

## Pass 3 notes (the learned painter in the gallery)

The gallery has a second run: the learned painter (the Stage A v2 discrete-head policy, `shared/stage_a_v2.pt`,
5,252,619 parameters, likeliest stroke each slot) on sealed target 3, recorded with `python -m train.record_policy`
at 256 x 192, 40 slots, frames every 20 steps as horizon_v1 was, and packed as `policy_v2_sealed_003` (2.6 MB, 39
frames). The recorder rolls the policy out the way `train.policy.rollout` does and then dries the sheet until it is
set, as the evaluation does; its final total, 9.02 (lab_l2 708.8, ssim 0.538), and its strokes match the same
painting in the Modal evaluation exactly. Each stroke line says one candidate and an empty rejected list. A selector
under the painting card switches runs (it reloads on `?run=`), with a plain label per run kept in `gallery.json`.
For the learned painter the dashboard says so, the slot line reads "one look, one stroke, no candidates", the
candidates KPI becomes Looks ("1 per stroke, no candidates"), Candidates is greyed out and About explains why. The
policy has no dry between phases, so the brush is now swapped at the pot at the start of a phase's first stroke when
no phase dry came before it; the search painter's run is unchanged (its swaps stay in the phase dries).

What I saw in the browser at 50x (`studio/review/pass3/`): the arm lays the first wash with the tip on the sheet, the
brush changes at the pot as the medium and detail phases start, the sheet fills in with the blue sea, the pale sky
band and the yellow ring the policy makes of the sun, and the score sheet turns at the end with total 9.02, the
ledger's final total, and the kept guess compared against it. The learned painter's replay is short (2 min 56 s at
1x against 4 min 35 s), since a stroke takes only its choreography: the policy's recorded time to look and act is under
a third of a second per stroke. Its waits of 2 or 3 steps show as "waiting" between strokes, and the last dry is 575 steps.

## Pass 4 notes (the scribe in the gallery)

- A third run in the gallery: the peach hibiscus Qwen3 Coder Next wrote from a text prompt, the one painting in
  the first pool I rated love. `python -m scribe.record` replays the stroke file through the recorder, so the
  studio paints it exactly as the simulator did, waits included; 42 strokes, 3,867 simulator steps, 67 frames at
  one every 60 steps (one every 20 gave 196 frames and 11 MB, too much for the repo).
- There is no target image for a scribe painting, so `target.png` is a prompt card: the subject written on blank
  paper of the same size. The dashboard reads the subject from `run.json` for the title.
- No score sheet either, because there is nothing to match. At the reveal the sheet shows the painting's rating
  (love, okay, nope, or not rated), the judge's win share when the candidate carries one, and the model that
  wrote the file. The "candidates tried" tile reads "none, the file was written blind".
- Checked in a headless browser: title, painter line, about text and the reveal all read from the run; no page
  errors. The replay itself is unchanged, since the ledger has the same shape as the learned painter's (one
  phase, no candidates, settle dries between strokes).
