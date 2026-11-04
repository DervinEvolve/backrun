# Examples

Each `.jsonl` file is a stroke list with a canvas header; each `.png` next to
it is its render at 256 x 192. Rebuild any of them with:

```
python -m backrun.cli render examples/backrun.jsonl examples/backrun.png
```

Replaying a file gives the same bytes every time on the same machine (this is
tested).

## flat_wash

Three overlapping horizontal strokes of French ultramarine laid wet into each
other, then left to dry for 1200 steps. The wash dries with a thin, darker
line along its outer edge: in the render the outer 3-pixel ring is about 40%
darker than the interior (mean darkness 0.47 against 0.33). Ultramarine is set
to granulate, and the interior is mottled, with more pigment in paper valleys
than on peaks (about 0.48 against 0.36 in deposited amount). The rim is
one to two pixels wide and stair-stepped at this resolution.

## wet_in_wet

A pale cadmium yellow wash, then three short strokes (quinacridone rose,
French ultramarine, Hooker's green) dropped into it while it is still wet.
The three strokes have soft, feathered edges with no hard line, while the
outer edge of the yellow wash, which met dry paper, has a hard, slightly
darker rim. The strokes spread only a few pixels; they do not bloom far.

## backrun

A wash of French ultramarine with a little burnt umber, 230 steps of drying
(at that point the wash is damp: mean water about 0.11, and some cells near
the edge have already dried), then one loaded stroke (water 1.0, weak
ultramarine) across the middle, then drying to the end.

What the image shows: a pale patch where the wet stroke landed, surrounded by
a darker, irregular band where pigment was pushed outward and settled. Along
the top of the band there is one short, sharp dark line; elsewhere the band
is soft-edged. Faint tide lines show further out in the wash.

What it does not show: the crisp, branching, cauliflower-shaped edge of a
real backrun. The ring is a rounded blob that roughly follows the shape of
the stroke. So the simulator produces a pale bloom with a dark margin that
is related to a backrun, not a convincing one.

Changing only the drying time before the wet stroke (not shipped as files,
just edit the `dry` line): with no drying the stroke makes a pale patch with
a diffuse edge and no clear ring; at 150 steps the ring appears; at 230 steps
(this file) it is darkest and partly hard; after 600 steps the wash is fully
dry and the stroke just sits on top as a separate glaze with its own thin
rim, with no spreading. The ring is strongest when the wash is damp, which
is the condition under which real backruns form. Nothing in the code places
the ring; it comes from water flowing outward, lifting and carrying pigment,
and depositing it where the water runs out.
