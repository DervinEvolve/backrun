# Training the painter

This is the plan for Phase 4: a learned painter that beats the search painter
at the same stroke budget, with a fraction of the simulation per painting.
Nothing here has run yet. Numbers marked "estimate" are guesses to be
replaced by measurements.

## What the learned painter has to beat

The search painter simulates every candidate stroke before choosing one.
On the horizon target with 40 stroke slots and 16 candidates per slot it
simulates 640 strokes to lay 40, and takes about 42 seconds on one CPU
core. Its scores on three synthetic targets, one seed each, with the
uncapped dry-preview painter, are the baseline (lower is better):

| Target | total | lab_l2 | ssim |
|---|---|---|---|
| horizon | 5.61 | measured in the run folder | measured in the run folder |
| discs | 13.53 | | |
| sky | 4.31 | | |

A learned painter wins if it scores lower than this on held-out targets
with the same 40 strokes and one simulator step per stroke, averaged over
three seeds.

## The policy

Think of the search painter as someone who tries sixteen strokes on scrap
paper before every real one. The learned painter has to look at the sheet
and know.

**Input, downsampled to 128 x 96:** the target; the current sheet as it
would look dry (`render(predict_dry=True)`); the wet mask; the per-pixel
error between the two renders; and the phase as a one-hot. Nine channels.

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

For the search painter and each learned painter, on the sealed set:
`total`, `lab_l2`, `ssim`, strokes used, simulator steps per painting,
wall time per painting, and the gap between the score the painter
predicted while painting and the score of the fully dried sheet. That last
number is the honesty meter. In the first search-painter runs the washes
phase was optimistic by about 0.8 to 2.6 points of `total`.

## Where it runs

`train/modal_app.py` defines four functions on Modal: `generate_ledgers`
(CPU, fanned out), `train_imitation` (one GPU), `finetune_rl` (one GPU plus
CPU rollout workers) and `evaluate` (CPU, sealed set). Two volumes:
`backrun-data` for targets and ledgers, `backrun-checkpoints` for weights.
The image pins Python 3.11, PyTorch, NumPy, SciPy and Pillow, and installs
this repository from its own checkout so the simulator on the GPU box is
the same commit as the one on the laptop.

Modal's client speaks gRPC, which the cloud session that wrote this cannot
carry, so `modal run` is launched from the owner's machine or by another
operator. The scripts are written so that is one command per stage.

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
4. `train/imitation.py`: Stage A.
5. `train/evaluate.py`: the sealed-set report, run first on the search painter so the baseline exists before any learned number.
6. `train/rl.py`: Stage B.
7. `train/modal_app.py`: wrap the above.
