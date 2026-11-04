# Training the painter

This is the plan for Phase 4: a learned painter that beats the search painter
at the same stroke budget, with a fraction of the simulation per painting.
What has run so far: on CPU in a cloud container, the search-painter
baseline, the sealed-set report on the search painter, ledger generation
and small Stage A smoke runs end to end; on Modal, one Stage A run on a
GPU (section "First GPU run"), whose painter scored no better than a blank
sheet. The policy head was rebuilt after it (section "The policy"). Numbers
marked "estimate" are guesses to be replaced by measurements.

## What the learned painter has to beat

The search painter simulates every candidate stroke before choosing one.
On the horizon target with 40 stroke slots and 16 candidates per slot it
simulates 640 strokes to lay 40, and takes about 42 seconds on one CPU
core. Its scores on three synthetic targets, one seed each, with the
uncapped dry-preview painter, are the baseline (lower is better):

| Target | total | lab_l2 | ssim | strokes used |
|---|---|---|---|---|
| horizon | 5.61 | 331.3 | 0.445 | 39 |
| discs | 13.53 | 1152.3 | 0.518 | 38 |
| sky | 4.31 | 222.9 | 0.495 | 31 |

A learned painter wins if it scores lower than this on held-out targets
with the same 40 strokes and one simulator step per stroke, averaged over
three seeds.

## The policy

Think of the search painter as someone who tries sixteen strokes on scrap
paper before every real one. The learned painter has to look at the sheet
and know.

**Input, downsampled to 128 x 96:** the target; the current sheet as it
would look dry (`render(predict_dry=True)`); the wet mask; the per-pixel
error between the two renders; and the phase as a one-hot over the three
phases. Eleven channels in all (`OBS_CHANNELS` in `train/policy.py`).

**Output, one stroke:** three path control points (six numbers in canvas
coordinates), width, water, eight pigment concentrations, and a wait, which
is how many drying steps to run before the next stroke. The wait is the
part that matters most for watercolor and the part no pixel painter has.
These 17 numbers, each in 0..1, are the action layout in
`train/policy.py`; the stroke files and the studio see only them.

**Network:** a small convolutional encoder, about five million parameters
(5,252,619 in the default config), followed by a discrete, spatial head
(format `discrete-v2`). The first head regressed the 17 numbers directly
and collapsed onto one average stroke (section "First GPU run"); the
second picks among choices instead of averaging them.

* *Where the stroke starts.* The encoder's second stage gives a 24 x 32
  feature map. The deeper stages' maps are brought to that size and added
  to it (a 1 x 1 convolution and bilinear upsampling each, then one 3 x 3
  convolution), so each cell also sees the whole picture. A 1 x 1
  convolution turns each of the 768 cells into one logit; the stroke's
  first path point is in the chosen cell. A second 1 x 1 convolution gives,
  per cell, where in the cell the point sits (two numbers in 0..1), trained
  only at the true cell.
* *Everything else, given the start.* The feature vector at the start cell
  (the true one in training, the chosen one when painting), the average
  and max pooled deepest stage, and the cell's position go into an MLP
  that gives, for each of the other 15 numbers, logits over 32 equal bins
  on 0..1. The two further path points are expressed relative to the
  start point, as (dx, dy) within plus or minus half the canvas, clipped.
  Width, water, the eight concentrations and the wait are binned as they
  are. A concentration in the lowest bin decodes to no pigment.
* *Decoding.* The most likely cell, then the most likely bin of each
  number at that cell, turned back into 17 numbers from bin centres
  (`argmax`), or both drawn from the softmax with a temperature
  (`sample`). The network also gives the log probability and entropy of
  an action, which Stage B will need.

The bin count (32) and the relative range (0.5) are fields of
`PolicyConfig`. The sparsity penalty that kept mixes to one or two
pigments is still available on the decoded concentrations but is off by
default: the bins can put a pigment at zero directly. Checkpoints from the
first head are refused with a message saying they must be retrained.

**Stopping:** a fixed budget of strokes per painting at first. A learned
stop comes later, once the stroke penalty in the score is tuned.

## Two training stages

### Stage A: imitation, cheap

Generate ledgers with the search painter on many targets, then train the
policy to predict the accepted stroke from the state before it. This is
behaviour cloning. The loss is a cross-entropy for the start cell and for
each of the 15 binned numbers (with label smoothing of 0.05), plus a
smooth L1 loss on the sub-cell offset. `train.jsonl` logs every part
separately, with the start cell's top-1 and top-5 accuracy and the share of
binned numbers predicted within one bin. It needs no GPU for data: one painting is about 40
seconds of CPU, so a thousand paintings is about eleven CPU hours, which
is minutes of wall time fanned out across containers. Training the network
on a thousand paintings, forty strokes each, is under an hour on one
mid-range GPU (estimate).

Expected result: a policy roughly as good as the search painter, forty
times cheaper per painting, and probably a little worse on exact placement.

### Stage B: reinforcement, where the gains are

Fine-tune the same policy by painting with the simulator in the loop and
rewarding the drop in `total` each stroke produces on the dried sheet.
Rollouts run on CPU workers, since the simulator is NumPy; the network
updates run on one GPU. Four to eight GPU hours is the first budget
(estimate). The reward is the scorer's number on the painting as it will
dry, so the painter is paid for patience: a stroke that looks good wet
and bad dry earns nothing.

A later option is a PyTorch port of the simulator so rollouts run in
batches on the GPU. It is more work and would make the simulator
differentiable as a side effect, which opens a third training route by
gradient. Not needed for Stage B.

## Targets and the sealed set

Training targets: the built-in synthetic scenes with random seeds, the
owner's photographs, and public-domain photographs, all letterboxed to
256 x 192. No paintings by anyone are used, as the README states.

A sealed set of fifty targets is drawn once, hashed, and never used for
training, selection or early stopping. Every published number is measured
on the sealed set with three seeds. The scorer stays in its own module and
the training code imports only its `score` function, which a test already
checks. This separation is the same rule the earlier research on
self-improving systems kept coming back to: gains are real only where the
check is outside the loop.

## What gets reported

`train/evaluate.py` produces this report. Its first run on the search
painter (five sealed targets, three seeds, 128 x 96, 40 strokes, 16
candidates) gave a mean dried `total` of 6.45 with a standard deviation of
1.29, about 2,600 simulator steps per painting counting every candidate,
and a washes-phase prediction gap of about 1.15. The full table is in
`runs/eval/` when the command is run.

For the search painter and each learned painter, on the sealed set:
`total`, `lab_l2`, `ssim`, strokes used, simulator steps per painting,
wall time per painting, and the gap between the score the painter
predicted while painting and the score of the fully dried sheet. That last
number is the honesty meter. In the first search-painter runs the washes
phase was optimistic by about 0.8 to 2.6 points of `total`.

Two baselines place a learned painter between doing nothing and the
search painter: `blank` lays no strokes and only dries the sheet for 2000
steps, and `mean:LEDGER_ROOT` lays the dataset-mean stroke of those ledgers
in every slot. `python -m train.evaluate compare OUT.md REPORT_DIR ...`
puts reports side by side and, when a blank and a search report are among
them, gives each painter's share of the gap from blank to search that it
closes (0 is no better than a blank sheet, 1 is as good as the search
painter).

Every report also carries a collapse detector, since a policy that paints
the same stroke every time is the first failure to look for. Per painting:
the standard deviation of the strokes' start points in pixels, along each
axis, and the number of distinct strokes once paths are rounded to 1 px
and width and concentrations to 0.05; the report gives the mean over
paintings. The search painter, at 128 x 96, spreads its start points by
about 37 px in x and 25 px in y and lays as many distinct strokes as
strokes.

## First Stage A smoke run (CPU)

A check that the whole Stage A path works, run in the cloud container on
CPU with two worker processes under `nice`. It is far too small to say
anything about how good a learned painter can be.

Data: 24 synthetic training targets (8 seeds of each of the three kinds,
`runs/data/train_smoke`), painted by the search painter at 128 x 96 with
20 stroke slots and 8 candidates per slot (`runs/ledgers_smoke`). That
gave 24 paintings and 473 accepted strokes; mean dried `total` of the
ledgers 9.48. The example cache for all 24 paintings is 17 MB. At most
0.5% of any painting's error-map pixels were above 1 and were clipped
when cached.

Training: the `small` network config (47,657 parameters), 8 epochs, batch
32, learning rate 1e-3 with a cosine schedule, split by painting into 21
paintings (413 strokes) for training and 3 (60 strokes) for validation.

| epoch | train loss | validation loss |
|---|---|---|
| 1 | 0.997 | 0.703 |
| 4 | 0.598 | 0.515 |
| 8 | 0.584 | 0.508 |

Measured on the training paintings, the loss went from 1.238 at
initialisation to 0.584. The final epoch's parts (training / validation):
points 0.215 / 0.186, width 0.130 / 0.125, water 0.118 / 0.100, wait 0.066 /
0.064, concentration cross-entropy 0.373 / 0.323, sparsity 0.328 / 0.291.
Validation is lower than training here only because three paintings is a
small sample. The loss had nearly stopped falling after epoch 4.

Sealed set, first 5 targets, one seed (seed 1), 128 x 96, 40 stroke slots
(the evaluator's default, twice the 20 the ledgers used), dried before
scoring:

| painter | total | lab_l2 | ssim | strokes | sim steps per painting | wall per painting |
|---|---|---|---|---|---|---|
| search, 16 candidates | 6.44 (std 1.23) | 385.5 | 0.371 | 36.4 | 2,628 | 16.5 s |
| smoke policy (`best.pt`) | 25.01 (std 2.57) | 2,345 | 0.629 | 40 | 124 | 1.1 s |

The smoke policy is much worse than the search painter: about four times
the `total`, six times the `lab_l2`. That is the expected state before
real training: 47 thousand parameters, 413 strokes and 104 optimiser
steps cannot learn where strokes go. Its higher `ssim` comes with a
colour error six times larger, so it does not mean a better painting.
Its predicted and dried totals agree within 0.01; on a painting this far
from its target that agreement tells little. What the run does show: ledgers cache, decode
and train, checkpoints save, load and resume, and `train.evaluate`
paints the sealed set with the checkpoint, at about one twentieth of the
search painter's simulation per painting.

Wall time: ledgers 72 s, cache 10 s, training 12 s, policy evaluation 4 s,
search evaluation 52 s. The reports are in
`runs/eval/search_limit5_128x96_seed1` and
`runs/eval/policy_smoke_limit5_128x96_seed1`.

## First GPU run

Stage A ran on Modal with the first head (the one that regressed the 17
numbers): 100 ledgers, 10 epochs, 177 s on an A10G. The
loss fell from 1.21 to 0.41, with 0.40 on validation. On the sealed set (5
targets x 3 seeds, 256 x 192, 40 strokes, dried before scoring):

| painter | total | std | lab_l2 | ssim |
|---|---|---|---|---|
| Stage A policy, first head | 23.57 | 1.52 | 2,211 | 0.654 |
| blank sheet, dried 2000 steps | 24.02 | 2.13 | | |
| dataset-mean stroke, 40 times | 24.54 | 2.36 | | |
| search painter, at 128 x 96 | about 6.4 | | | |

The full-size (256 x 192) search number is not measured yet. The learned
painter was no better than a blank sheet.

Why: it had learned one stroke. The local smoke policy, which shows the
same thing in its stroke file
(`runs/eval/policy_smoke_limit5_128x96_seed1/paintings/sealed_000_s1/strokes.jsonl`),
lays the same stroke 40 times: a 9 px vertical dab at the centre of the
sheet, with all eight pigments loaded at about their dataset-mean
concentrations (1.05 in total, a dark mud), its start point moving by less
than a pixel over the whole painting. The search painter's strokes spread
20 to 30 px along each axis and use 1.9 pigments on average. From any
given state the search painter might accept quite different strokes; a
regression loss (smooth L1 on positions, a per-pigment cross-entropy on
concentrations) is smallest for the average of them, and the average of
strokes all over the sheet is a stroke in the middle with a bit of every
colour. The network learned that average and little else, so it painted
the same thing whatever it saw. This is why the head is now discrete
(section "The policy"): a softmax over cells and bins can put its weight
on one of several good answers instead of between them.

A note on `ssim`: the blank sheet scores a higher `ssim` (0.65) than the
search painter (0.37). On smooth targets an empty sheet has almost no
structure to get wrong, and SSIM rewards that. This is a known weakness of
SSIM, and it is why `total` weights `lab_l2` heavily. The scorer is not
changed.

## Stage A v2 smoke run (CPU)

The discrete head, trained the same way as the first smoke run (same 24
ledgers and example cache, the `small` config, now 82,059 parameters, 8
epochs, batch 32, learning rate 1e-3, 21 paintings for training and 3 for
validation), on CPU in the cloud container. Training took 15 s. It is a
check that the new head trains, decodes and paints, and says nothing yet
about how good it can be.

| epoch | train loss | validation loss | cell top-1 / top-5 (val) | bins within one (val) |
|---|---|---|---|---|
| 1 | 10.16 | 9.71 | 3.3% / 3.3% | 37.6% |
| 4 | 8.05 | 8.28 | 5.0% / 6.7% | 61.2% |
| 8 | 7.54 | 8.18 | 3.3% / 8.3% | 60.7% |

The loss is a sum of cross-entropies, so it is not comparable with the
first head's. Chance for the start cell is 0.13% top-1 and 0.65% top-5.
The offset loss stayed at about 0.25, which is what predicting the middle
of every cell gives: the offset head did not learn in 8 epochs.

Sealed set, first 5 targets, seed 1, 128 x 96, 40 stroke slots, dried
before scoring:

| painter | total | std | lab_l2 | ssim | start spread x, y (px) | distinct strokes |
|---|---|---|---|---|---|---|
| blank sheet | 24.32 | 2.32 | 2,294 | 0.654 | 0, 0 | 0 |
| dataset-mean stroke (`runs/ledgers_smoke`) | 24.87 | 2.55 | 2,330 | 0.629 | 0, 0 | 1 |
| first-head smoke policy | 25.01 | 2.57 | 2,345 | 0.629 | 0.14, 0.68 | 3.2 |
| v2 smoke policy | 24.30 | 2.24 | 2,282 | 0.649 | 2.18, 3.63 | 8.0 |
| search, 16 candidates | 6.44 | 1.23 | 385.5 | 0.371 | 37.3, 25.2 | 36.4 |

The v2 smoke policy is no better than a blank sheet either. It no longer
mixes every pigment: each stroke carries French ultramarine alone or no
pigment at all (clear water), and is 2 to 3.4 px wide. But it still
repeats nearly the same stroke, and in a place that hardly matters: on
three targets all 40 strokes start on the left edge, between 66 and 90 px
down; on the other two they start along the top edge at the right, carry
no pigment and so leave the sheet blank. With argmax decoding, a sheet that hardly
changes after each small stroke gives nearly the same observation, so a
policy that responds weakly to the observation repeats itself. Whether a
properly trained v2 head escapes this needs the GPU run on the 100 or
more ledgers; this smoke run, with 413 training strokes, cannot say.

Wall time: training 15 s, policy evaluation 4.4 s, blank 1.9 s, mean
stroke 3.9 s. The reports are in `runs/eval/policy_v2_smoke_limit5_128x96_seed1`,
`runs/eval/blank_limit5_128x96_seed1`, `runs/eval/mean_smoke_limit5_128x96_seed1`
and side by side in `runs/eval/compare_v2_limit5_128x96_seed1.md`.

## Second GPU run: the discrete head at full scale

The same pipeline on Modal with the discrete head: 1,000 ledgers (the 100
from the first run plus 900 more, 256 x 192, 40 slots, 16 candidates), a
cache of 36,541 strokes, 900 paintings for training and 100 for validation,
20 epochs of the default config (5,252,619 parameters) on an A10G, 2,348 s.
Validation loss fell from 7.34 to 6.89 at epoch 8 and then rose every epoch
to 8.55 while training loss kept falling to 4.83, so the run overfit after
epoch 8 and the saved weights are the EMA from epoch 8. At the last epoch
the start-cell head was right 2.9% of the time (top-5 11.6%, chance 0.13%)
and the bin heads were within one bin 74.0% of the time.

Sealed set, first 5 targets x 3 seeds (7, 8, 9), 256 x 192, 40 stroke
slots, scores on the fully dried sheet, all painters on the same 15
paintings:

| painter | total | std | lab_l2 | ssim | sim steps per painting | wall per painting | start spread x, y (px) | distinct strokes | gap closed |
|---|---|---|---|---|---|---|---|---|---|
| blank sheet | 24.02 | 2.13 | 2,293 | 0.729 | 2,000 | 0.1 s | 0, 0 | 0 | 0 |
| first head (100 ledgers, 10 epochs) | 23.57 | 1.52 | 2,211 | 0.654 | 102 | 10.7 s | not measured | not measured | 0.02 |
| discrete head, likeliest stroke | 8.85 | 1.36 | 716.6 | 0.600 | 112 | 6.5 s | 75.4, 36.0 | 19.0 | 0.81 |
| discrete head, sampled strokes | 10.31 | 1.72 | 827.8 | 0.512 | 437 | 8.7 s | 81.6, 56.5 | 40.0 | 0.73 |
| search, 16 candidates | 5.26 | 0.67 | 312.0 | 0.483 | 2,897 | 50.0 s | 78.7, 53.6 | 35.3 | 1 |

"Gap closed" is how far a painter gets from the blank sheet (0) to the
search painter (1) on `total`. The discrete head closes 81% of that gap
with 26 times less simulation than the search painter, in an eighth of the
wall time. This is the first learned painter that paints. It is one run
and one seed of training on five sealed targets, so a direction rather
than a result.

What the numbers say about what to do next:

* Picking the likeliest stroke beats sampling by 1.5 points of `total`,
  so argmax is the default. Sampling spreads strokes more and never
  repeats, but places them worse.
* The likeliest-stroke painter repeats itself: 19 distinct strokes of 40.
  When the sheet barely changes, the same observation gives the same
  stroke. Stage B, which rewards the drop in score, is the direct fix; so
  is a stroke-history channel in the observation.
* The run overfit from epoch 8. More ledgers are cheap (900 cost about a
  dollar of CPU time), and flips of target and sheet together would double
  the data for free. Twenty epochs were twelve too many at this data size.

## Where it runs

`train/modal_app.py` defines a Modal app named `backrun` with these
functions:

* `make_targets` (CPU): writes the sealed set to the data volume if it is
  not there yet (it is drawn once and never overwritten; if present, its
  manifest is checked), then the training targets.
* `generate_ledgers` (CPU): plans the paintings still to do from
  `ledgers/index.jsonl` and fans them out with `.map` over chunks of
  targets (`paint_chunk`, eight CPUs each, one painting per CPU). Only this
  function writes the index, so an interrupted job resumes where it stopped.
* `train_imitation` (one GPU, an A10G unless `--gpu` says otherwise): calls
  `train.imitation.train_imitation`, keeps the example cache on the data
  volume and the checkpoints on the checkpoint volume, commits the
  checkpoint volume after every epoch, and resumes from `last.pt`.
* `evaluate` (CPU): runs `train.evaluate` on the sealed set for `search`,
  `blank`, `mean` (the dataset-mean stroke of `/data/ledgers`) or a
  checkpoint and writes the report under `eval/` on the data volume. When
  blank and search reports of the same size and limit are already there,
  the report ends with a table placing the painter between them.
* `finetune_rl`: raises `NotImplementedError`; Stage B is not written yet.

Two volumes: `backrun-data` at `/data` (`sealed/`, `train/`, `ledgers/`,
`cache/`, `eval/`) and `backrun-checkpoints` at `/checkpoints` (one folder
per training run). The image pins Python 3.11, installs NumPy, SciPy,
Pillow and PyTorch, copies `backrun/`, `train/`, `pyproject.toml` and
`README.md` from the local checkout and runs `pip install -e .[train]`, so
the simulator on the GPU box is the code on the machine that launched it.

One command per stage, from a machine where `modal setup` has been done:

```
modal run train/modal_app.py --stage targets
modal run train/modal_app.py --stage ledgers
modal run train/modal_app.py --stage imitation
modal run train/modal_app.py --stage evaluate --painter search
modal run train/modal_app.py --stage evaluate --painter stage_a/best.pt
modal run train/modal_app.py --stage evaluate --painter blank
modal run train/modal_app.py --stage evaluate --painter mean
modal run train/modal_app.py --stage all
```

Useful knobs: `--seeds-per-kind` (targets), `--limit`, `--chunk`,
`--strokes`, `--candidates`, `--width`, `--height` (ledgers), `--run-name`,
`--epochs`, `--batch`, `--lr`, `--config`, `--gpu` (training),
`--eval-limit`, `--seeds` (evaluation). A first cheap pass is
`--stage all --limit 100 --epochs 10 --eval-limit 5`.

Modal's client speaks gRPC, which the cloud session that wrote this cannot
carry, so `modal run` is launched from the owner's machine or by another
operator. The app was checked here only as far as importing it: the
module loads with Modal 1.6.3 installed, and also without Modal, and a
test checks it exposes the five functions. It has since been run on Modal
for the first GPU run (section "First GPU run").

The same steps run locally without Modal:

```
python -m train.data                    # sealed set (once) and training targets
python -m train.ledgers runs/data/train runs/ledgers --workers 2
python -m train.imitation runs/ledgers runs/ckpt/stage_a --cache runs/cache --device auto
python -m train.evaluate policy:runs/ckpt/stage_a/best.pt runs/eval/stage_a
python -m train.evaluate policy:runs/ckpt/stage_a/best.pt@sample runs/eval/stage_a_sample   # draw strokes instead of taking the likeliest
```

Environment note: in the cloud container used so far, PyTorch came from
PyPI (2.15.0, run on CPU), because the CPU-only wheel index at
download.pytorch.org is blocked there. The PyPI wheel for Linux brings the
CUDA helper packages, several gigabytes, which filled that container's
disk on the first try. This applies to that container only; the project
installs PyTorch however the machine allows.

## Prompts to targets, later

For the studio's prompt gallery, a text-to-image model turns a prompt into
a target before the painter starts. The painter neither knows nor cares
where the target came from. The model is a dependency and is credited by
name, version and licence in `run.json` and on the prompt card, as a font
or a dataset would be. The choice is the owner's; the plan needs one under
a licence that allows this use and that fits the GPU in hand.

## Cost, in round numbers

Estimates, to be replaced by the bills: Stage A under five dollars, Stage B
ten to thirty dollars at mid-range GPU prices, evaluation under a dollar.
The search painter baseline costs nothing but CPU time.

## Order of work

1. `train/data.py`: targets, letterboxing, the sealed set with its hash file.
2. `train/ledgers.py`: run the search painter at scale, one folder per painting.
3. `train/policy.py`: the network and the stroke parameterisation, with a test that a policy output replays through the simulator unchanged.
4. `train/imitation.py`: Stage A. Written; run once on a GPU with the first head (no better than a blank sheet); head rebuilt as discrete and smoke-tested on CPU.
5. `train/evaluate.py`: the sealed-set report, run first on the search painter so the baseline exists before any learned number.
6. `train/rl.py`: Stage B.
7. `train/modal_app.py`: wrap the above. Written; ran the first GPU run on Modal.
