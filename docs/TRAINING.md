# Training the painter

This is the plan for Phase 4: a learned painter that beats the search painter
at the same stroke budget, with a fraction of the simulation per painting.
What has run so far, all on CPU in a cloud container: the search-painter
baseline, the sealed-set report on the search painter, ledger generation,
and a small Stage A smoke run end to end (section "First Stage A smoke run
(CPU)" below). No GPU training has run yet. Numbers marked "estimate" are
guesses to be replaced by measurements.

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

**Network:** a small convolutional encoder, about five million parameters,
with a global pooling head. Nothing exotic; the simulator is the hard part.

**Output, one stroke:** three path control points (six numbers in canvas
coordinates), width, water, eight pigment concentrations with a sparsity
penalty so mixes stay to one or two pigments, and a wait, which is how
many drying steps to run before the next stroke. The wait is the part that
matters most for watercolor and the part no pixel painter has.

**Stopping:** a fixed budget of strokes per painting at first. A learned
stop comes later, once the stroke penalty in the score is tuned.

## Two training stages

### Stage A: imitation, cheap

Generate ledgers with the search painter on many targets, then train the
policy to predict the accepted stroke from the state before it. This is
behaviour cloning. It needs no GPU for data: one painting is about 40
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
* `evaluate` (CPU): runs `train.evaluate` on the sealed set for `search` or
  a checkpoint and writes the report under `eval/` on the data volume.
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
test checks it exposes the five functions. It has not been run on Modal.

The same steps run locally without Modal:

```
python -m train.data                    # sealed set (once) and training targets
python -m train.ledgers runs/data/train runs/ledgers --workers 2
python -m train.imitation runs/ledgers runs/ckpt/stage_a --cache runs/cache --device auto
python -m train.evaluate policy:runs/ckpt/stage_a/best.pt runs/eval/stage_a
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
4. `train/imitation.py`: Stage A. Written; smoke-tested on CPU.
5. `train/evaluate.py`: the sealed-set report, run first on the search painter so the baseline exists before any learned number.
6. `train/rl.py`: Stage B.
7. `train/modal_app.py`: wrap the above. Written and import-checked; not run on Modal yet.
