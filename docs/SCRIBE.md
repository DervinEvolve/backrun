# Scribe: a language model that writes stroke files

The scribe line trains a language model to write Backrun stroke files from a
text prompt such as "a harbour at dusk, two boats, low tide". The simulator
paints each file, a gate throws out files that are broken or that try to game
the judge, and the reward comes from a vision model asked, again and again,
which of two paintings is the better watercolour of the subject. One of the
two is always a painting from a rated pool.

The code lives in `scribe/`. Everything below the trainer has tests. The first
pool exists and is rated; see "The first pool" below. The trainer itself has
been written and parsed but never run.

## Where the recipe comes from

Surya Narreddi and Cameron Franz trained a language model with reinforcement
learning to write p5.brush JavaScript that renders to watercolour paintings,
rewarded by a vision judge comparing each render against a hand-rated pool of
reference paintings. Their write-up is "Training AI to Paint with Code" at
[surya.website](https://surya.website). Sergio Paniego published an open
reproduction with TRL at
[huggingface.co/blog/train-to-paint-with-code](https://huggingface.co/blog/train-to-paint-with-code).
The scribe line follows their recipe and borrows their lessons; the mistakes
in adapting it are ours.

What changes here: Backrun's stroke file is a much smaller language than
JavaScript (two operations and a handful of numbers), and the simulator paints
a file in a few seconds on one CPU core, so the render step needs no browser.

The lessons from their write-up that are built in:

1. A judge asked for an absolute score from 0 to 10 gives numbers that bunch
   near the bottom. Asked "which of these two is better", it spreads out. The
   scribe judge only ever compares.
2. Their nine reward signals stopped improving because five judges agreed
   with each other at a correlation of 0.85 to 0.95, and a code-length term
   worth a third of the reward was maxed out by step 30 and gave no gradient
   after that. So every reward term here is measured on a batch before
   training: its spread, how often it sits at 0 or 1, and how it correlates
   with every other term.
3. A long API reference made models invent methods. A short, opinionated
   allowlist worked. The scribe system prompt is 20 lines.
4. The pool needs colour coverage. They added supplement paintings where
   colours were thin. The pool here reports thin hues and lightness levels so
   supplement prompts can be aimed at them.
5. When the judge or the network fails, that rollout is left out of its
   group. Scoring it zero would teach the model that it did something wrong.
6. They rated 1,664 generations into love, okay and nope; the 117 love
   ratings seeded the pool. Each rollout is compared with two references, in
   both presentation orders.

## The loop

```
  prompt ("a harbour at dusk")
     |
     v
  language model --> reply text --> extract the JSON Lines block
                                          |
                                          v
                                    gate: static checks
                                          | pass
                                          v
                     simulator: replay from the canvas header, dry 2,000 steps
                                          |
                                          v
                                    gate: rendered checks
                         fail -> reward 0 | pass
                                          v
             +----------------------------+--------------------------+
             |                            |                          |
       economy term             aesthetic term (optional)     pairwise judge
       10 to 80 strokes          HTTP scorer, off by default   candidate vs 2 references
             |                            |                    x 2 orders = 4 questions
             +-------------> weighted sum <----------------------+
                                  |
                 any term None -> reward None (rollout excluded)
                                  |
                                  v
                        GRPO update of LoRA adapters
```

## The stroke language as the model sees it

`scribe/format.py` holds the system prompt. It tells the model the paper size
and seed, the two operations, each stroke field and its range, the eight
pigment names, that about 1,500 dry steps dries a wet wash, and the order a
watercolourist works in: big pale washes, dry, middle values, dry, small dark
details. It asks for JSON Lines inside one fenced block and nothing else.

`extract_jsonl` pulls the stroke lines out of a reply. It takes the first
fenced block that holds any `{...}` line, plus any stray `{...}` lines right
after the closing fence, and falls back to every line starting with `{` when
there is no fence. Trailing commas are dropped. A `canvas` line from the model
is ignored, because the gate supplies the header.

## The gate

`scribe/gate.py`. Every threshold is a module constant marked as a design
choice.

Static checks, all reported at once so the model can fix them in one retry:

- every line parses as a JSON object;
- only `stroke` and `dry` operations;
- a stroke path has 2 to 8 points, all inside the 256 x 192 paper;
- width from 1 to 0.4 of the shorter side (76.8 pixels);
- water from 0 to 1;
- pigments only from the eight names, concentrations 0 to 1, at most three
  per stroke;
- 6 to 200 strokes;
- at least one dry operation, and at most 20,000 dry steps in total.

Rendered checks, after replaying the file and drying 2,000 more steps:

- coverage: the share of pixels whose colour differs from bare paper by more
  than Delta E 3 must be between 5% and 98%, so a nearly blank sheet fails and
  so does a sheet with no paper showing;
- mean lightness (CIELAB L*) of at least 35, which rejects the all-black wash;
- variety: the painted pixels must hold either two hues at least one 30-degree
  bin apart (each with 5% of the painted pixels) or two lightness levels at
  least one 10-unit bin apart (each with 8%), which rejects the single flat
  wash.

The cheat the tests check by name is six full-width Payne's grey strokes and a
dry: it passes every static check, then fails on coverage and on variety.

## The pool and rating

`scribe/generate.py` fills a pool folder. For each prompt, generator model
and candidate number it asks the model, gates the reply, retries once with
the gate's reasons if it failed, and, when a critic model is set and the
prompt has a reference photo, shows the critic the render and the photo, asks
for three concrete improvements and asks the generator to revise. Every
round's stroke file and render is kept. Each candidate folder has
`strokes.jsonl`, `final.png` at 256 x 192, `preview.png` at 512 wide, the
round files and `meta.json` (model, prompt, round, gate result, token usage,
time). `index.jsonl` gets one line per candidate. A rerun skips candidates
already indexed; a candidate whose API call failed is left out of the index
so the next run tries it again.

The prompts file is JSON Lines:

```json
{"id": "harbour", "subject": "a harbour at dusk", "prompt": "Paint a small harbour at dusk with two boats at low tide.", "reference": "photos/harbour.jpg"}
```

`scribe/rate.py` serves a rating page on 127.0.0.1. It shows one candidate at
a time at the preview size with its subject and prompt, but not the model that
wrote it. Keys: L love, O okay, N nope, U undo. The sheet slides into one of
three labelled drawers. Each rating appends a line to `ratings.jsonl`; the
last line for an id wins, and undo appends a line that restores the previous
tier, so a run of undos walks back through the history.

`scribe/pool.py` reads the pool:

- `summary()` counts per tier and per generator model;
- `colour_coverage()` builds a 12-bin hue histogram and a 10-bin lightness
  histogram over the painted pixels of the love and okay paintings, counts in
  how many paintings each bin is present, and writes one plain line naming the
  thin bins, for example "Thin in 140 rated paintings (present in fewer than
  7): hues teal (180-210 deg), cyan (210-240 deg); lightness L* 10-20";
- `sample_references(rng, k=2, tiers=("love", "okay"), mix=0.5)` draws
  references deterministically from a seed, each from love with probability
  `mix`;
- `manifest()` hashes every rated id, its tier and its files, so a training
  run records exactly which pool it used.

The rating workflow:

1. Write 100 to 200 prompts with the subjects you care about, some with your
   own reference photos.
2. Generate two to four candidates per prompt from several models.
3. Rate them all. At about two seconds each, 1,600 candidates take about an
   hour, longer with breaks.
4. Read the coverage line. Write supplement prompts for the thin hues and
   values, generate, rate again.
5. Stop when love has at least 100 paintings and no bin you care about is
   thin.

## The reward

`scribe/judge.py` asks a vision model, with two images: "Which of these is
the better watercolour of SUBJECT?", followed by a short criterion
(translucent washes, soft wet edges, visible blooms and backruns, paper
showing through, a composition that reads) and "Answer with one letter: A or
B." Each of the two references is shown once with the candidate as A and once
as B, so a judge that always says A scores every candidate 0.5. The score is
the share of the four comparisons the candidate won. A reply that is not
clearly A or B, or a failed call, counts as a comparison failure and is left
out of the share; if more than half fail the judge returns None.

`scribe/reward.py` combines the terms. Weights, in `RewardWeights`:

| Term | Weight | What it measures |
|---|---|---|
| gate | 0.05 | 1 if the file passed the gate |
| economy | 0.05 | 1 if the file has 10 to 80 strokes; very long files were the waste in the original recipe |
| aesthetic | 0.30 | an HTTP scorer such as HPSv3 at `SCRIBE_AESTHETIC_URL`, mapped to 0 to 1; off by default, and then its weight moves to pairwise with a note in the result |
| pairwise | 0.60 | the judge's share of wins |

A file that fails the gate scores 0 and costs no judge calls. If any term is
None because a service failed, the whole reward is None.

### Pre-training checklist

Before spending GPU time:

1. Generate a batch from the base model itself: `python -m scribe.generate
   prompts.jsonl batch --models BASE --per 8` against any OpenAI-compatible
   endpoint that serves it.
2. Score it: `python -m scribe.reward score batch --pool pool --judge-model
   JUDGE --out batch-terms.jsonl`. This prints the report.
3. Read the report. Any term with a standard deviation under 0.02 is flagged
   as giving no gradient; any pair correlated above 0.8 is flagged as the same
   signal twice. Expect gate and economy to be flagged if the base model
   already passes the gate every time; then either drop their weight to zero
   or accept that they act as floors.
4. Check the judge: the share of comparison failures should be small, and
   swapping the order should flip the answer most of the time.
5. Record the pool manifest hash. The trainer writes it into
   `run_config.json`.

## The trainer

`scribe/train_grpo.py` uses TRL's GRPOTrainer. Defaults from Paniego's
configuration: LoRA on all linear layers, bf16, gradient checkpointing,
learning rate 5e-5 with a constant schedule after 5 warmup steps,
`scale_rewards="none"`, 8 generations per prompt, top-p 0.95, top-k 20. Max
completion length is 4,096 tokens because stroke files are short. The default
base model is `Qwen/Qwen3-4B-Instruct-2507`, text only and without a thinking
step, so the reply budget goes on the file; `Qwen/Qwen3.5-4B` is tagged
image-text-to-text and stays an untried option.

The reward function returns None for an excluded rollout. TRL's current
source marks a completion NaN when every reward function returns None, leaves
it out of the group mean and gives it zero advantage. Older TRL releases added
it in as 0, so the script checks the installed version for that handling and
refuses to start without it.

Each reward call appends one line per rollout and one summary line to
`terms.jsonl`, so `python -m scribe.reward report OUT/terms.jsonl --step N`
gives the same report on any step of a run. The judge's token usage goes to
`usage.jsonl`.

## The first pool

Pool v1 was generated on 4 November 2026 from the 36 starter subjects, 8
candidates per subject from each of four generators, with one critic round
from `qwen/qwen3-vl-30b-a3b-instruct` and no reference photos. Kimi K2.6 was
in the first batch and dropped after 23 candidates because it spent its whole
reply budget on hidden reasoning. The run was resumed five times across
container restarts; the index is append-only, so nothing was generated twice.

| Generator | Accepted | Rejected by the gate | Pass rate | Median strokes |
|---|---|---|---|---|
| Gemini 3.1 Flash Lite | 279 | 9 | 97% | 10 |
| DeepSeek V4 Flash | 243 | 45 | 84% | 19 |
| Qwen3 Coder Next | 218 | 70 | 76% | 22 |
| GLM 5.3 Flash | 163 | 125 | 57% | 34 |
| Kimi K2.6 | 18 | 5 | 78% | 44 |

921 accepted, 254 rejected, no failed calls. GLM and DeepSeek failed mostly
on format (no stroke lines, no dry op, too few strokes); Qwen failed on
substance (too many dry steps, one flat colour, paper fully covered). Gemini
almost never failed and paints the sparsest files. 847 of the 921 accepted
files are the revised one after the critic. The whole pool cost $3.74 in
OpenRouter credit, read from the usage ledger: $3.67 for generation and $0.07
for the critic, about $0.004 per accepted painting.

Rating. I rated the first 108 at the desk, in about 25 minutes, and kept 15
of them (1 love, 14 okay). At my request Codex, an AI coding agent, reviewed
the remaining 813 from contact sheets and recorded a tier for each, keeping
254 (9 love, 245 okay). So of the 921 ratings, 108 are mine and 813 are an AI
reviewer's. We ranked the four generators in the same order (GLM 5.3 Flash
best, Qwen3 Coder Next worst), but Codex kept paintings at about twice my
rate, so the keep set is mostly its taste rather than mine. I keep that in
mind when reading anything the judge is calibrated against, and the ratings
file carries no label of who rated what; the split is by line number, 1 to
108 and 109 to 921.

Totals: love 10, okay 259, nope 652. Per generator, love and okay together:
GLM 61%, Gemini 26%, DeepSeek 26%, Qwen 13%. Colour coverage over the 269
keeps is thin in green and teal (present in fewer than 14 paintings) and in
dark values below L* 20, so supplement prompts should aim at green subjects
and night or shadow scenes.

The ratings log, the index, the prompts and a manifest with the pool hash
(`f641308c…`) are in `shared/pool_v1/`. The images are not in the repo; the
manifest hash covers each rated candidate's stroke file and render, so a
training run can record exactly which pool it used.

## Judge calibration

Before any GPU time, the judge has to agree with the ratings, because GRPO
pushes the model toward whatever the judge rewards. `scribe/calibrate.py`
scores rated pool paintings the way the reward does: two references of the
same subject, never the painting itself, each shown in both orders. It
reports the AUC, the chance a random keep (love or okay) outscores a random
nope, with a bootstrap interval; the order-flip rate, how often swapping A and
B changes the answer; and direct same-subject pairs of one keep against one
nope, where a good judge should pick the keep well above half the time.

The ground truth is my 108 ratings, which hold only 15 keeps, so every
interval is wide. Twelve judges were tried on the same 108 paintings and 40
direct pairs on 4 November 2026, about $9 of credit in all, including three
first attempts that had to be discarded. The reports, rows and pairs are in
`shared/judge_calibration/`.

| judge | scored of 108 | AUC [95%] | order flips | direct pairs valid of 40 | keep picked | cost, 512 calls |
|---|---|---|---|---|---|---|
| Qwen3-VL-235B | 108 | 0.66 [0.51, 0.82] | 0.20 | 40 | 0.74 | about $0.03 |
| Claude Opus 5.5, low reasoning | 108 | 0.66 [0.52, 0.79] | 0.07 | 40 | 0.79 | $1.55 |
| Claude Sonnet 5.5, low reasoning | 108 | 0.63 [0.48, 0.78] | 0.10 | 37 | 0.80 | $0.32 |
| Gemini 3.1 Pro, low reasoning | 99 | 0.62 [0.49, 0.75] | 0.11 | 26 | 0.81 | $3.20 |
| Kimi K2.6, reasoning off | 108 | 0.59 [0.45, 0.74] | 0.26 | 40 | 0.76 | $0.05 |
| GPT-5 mini, minimal reasoning | 108 | 0.59 [0.44, 0.73] | 0.32 | 40 | 0.70 | $0.04 |
| GPT-5.2, minimal reasoning | 106 | 0.57 [0.41, 0.73] | 0.20 | 25 | 0.84 | $0.52 |
| Claude Haiku 5.5, reasoning off | 108 | 0.56 [0.40, 0.74] | 0.26 | 40 | 0.71 | $0.01 |
| Grok 4.7, low reasoning | 108 | 0.55 [0.40, 0.72] | 0.21 | 40 | 0.69 | $1.18 |
| Qwen3.5 397B, low reasoning | 37 | 0.51 [0.24, 0.80] | 0.15 | 7 | 0.93 | $1.03 |
| Qwen3-VL-30B | 108 | 0.50 [0.38, 0.63] | 0.74 | 2 | - | about $0.02 |
| Gemini 3.1 Flash Lite | 108 | 0.47 [0.31, 0.63] | 0.45 | 40 | 0.60 | about $0.05 |
| GLM-4.6V | 0 | - | - | 0 | - | - |

What I take from it:

- The judge I had wired in, Qwen3-VL-30B, is a coin flip with a strong
  position bias: swapping the order changed its answer three times in four.
  It is out.
- Four judges show a real but modest signal: Qwen3-VL-235B, Opus 5.5, Sonnet
  5.5 and Gemini 3.1 Pro, all near AUC 0.65 and all picking my keep in about
  four direct pairs of five. No lower bound clears 0.5 by much, and that is a
  limit of 15 keeps as much as of the judges.
- Opus 5.5 is the most consistent, with 7% order flips and 2.5% on direct
  pairs. Qwen3-VL-235B matches its AUC at about a fiftieth of the price but
  flips one answer in five. Asking both orders averages flips out, so for
  training the two are closer than the flip rate suggests.
- Models that reason by default answer with nothing unless the request
  turns reasoning off or caps the effort, and the Claude, Grok and Qwen3.5
  endpoints refuse to turn it off, so the judge takes extra request fields
  and the calibration command has `--no-reasoning`, `--reasoning-effort`
  and `--max-tokens`. The prompt now asks for the letter alone, and a letter
  standing alone on the first line counts even when prose follows.
- Within one generator the signal is weaker, and for Qwen3 Coder Next it is
  strongest, so part of what the judges pick up is a generator's style
  rather than quality. More of my own ratings, especially keeps, would
  sharpen all of this; the 813 reviewer ratings are not used as truth here.

## The GPU route

`scribe/modal_app.py` runs the scribe line on Modal, mirroring the painter's
app. The container gets `backrun/`, `scribe/` and the reference pack
`shared/pool_v1/keeps/`, which is the 269 love and okay paintings exported
as a small pool of their own (`Pool.export`), so nothing has to be uploaded
to a volume first. The OpenRouter key comes from a Modal secret. From the
Mac:

```bash
modal secret create openrouter SCRIBE_API_KEY=...
```

The OpenRouter address is set in the image, so the secret only needs the key.

```bash
modal run scribe/modal_app.py --stage judge-check
```

```bash
modal run scribe/modal_app.py --stage sample
```

```bash
modal run scribe/modal_app.py --stage warmup --run-name warm_v1
```

`judge-check` makes one judge call on a CPU container and prints the answer
or the error, so a bad key shows up before GPU time is spent; `score --name
BATCH` re-scores a batch already on the volume the same way.

```bash
modal run scribe/modal_app.py --stage train --run-name grpo_v1 --model-path /checkpoints/scribe/warm_v1/merged --max-steps 30
```

```bash
modal run scribe/modal_app.py --stage evaluate --run-name grpo_v1 --model-path /checkpoints/scribe/warm_v1/merged
```

The judge defaults to Claude Opus 5.5 at low reasoning effort with a
400-token budget, the setting the calibration found; `--judge-model`,
`--judge-no-reasoning`, `--judge-reasoning-effort` and `--judge-max-tokens`
change it. The base model defaults to `Qwen/Qwen3-4B-Instruct-2507`, a
text-only model that does not think first.

`sample` is the go or no-go that both reviewers asked for before any
training. The base model writes eight stroke files for each of the 36
prompts with vLLM, the gate and the judge score them the way the reward
will, and the report in `/data/scribe/samples/base/report.md` says what
share passed the gate, what share was cut off at the token limit, and how
many of the 36 prompt groups have any reward spread. GRPO only learns from a
group whose rollouts score differently, so a batch of flat groups means a
short warm-up on the keeps comes first, and the report says so in one
sentence.

The first base sample, on 4 November 2026, settled the go or no-go: 288
files from Qwen3-4B-Instruct, 15% through the gate, none cut off. The
rejections were near misses: 171 files had four or five strokes where six is
the minimum, 34 misspelt a pigment or asked for white, 7 put a point off the
paper. So the model knows the language and has never seen a finished file,
which is what `warmup` is for: `scribe/sft.py` fine-tunes LoRA adapters on
the 269 keeps for three epochs, prompt in and fenced stroke file out, with
the loss on the completion only, then merges the adapter into the base model
on the checkpoint volume and samples it the same way as `sample`, with the
base numbers in the next column of the report.

`train` runs the trainer with vLLM colocated on the same GPU, LoRA on all
linear layers, four gradient accumulation steps so one optimizer step sees
four prompts, and the reference pack as the pool; `--model-path` starts it
from the warm-up's merged model. Output goes to `/checkpoints/scribe/<run>/`. `evaluate` merges the run's adapter into the
base model, samples it exactly as `sample` did, and writes a report whose
table has the base sample's numbers in the next column.

The first warm-up, `warm_v1`, lifted the gate pass rate from 15% to 66% on
the same 36 prompts. It also made 22% of replies run to the 4,096-token
limit, where the base model never did, and generation took four and a half
minutes instead of half a minute, so some replies ramble or loop instead of
closing the file; the report now quotes the tail of one such reply. The
judge calls in both runs failed on the key (OpenRouter answered "User not
found"), so neither batch has reward numbers yet.

Status: `sample` and `warmup` have run on Modal. The trainer has not run
yet; its first run is a debugging run.

## Showing it in the studio

A scribe painting is a stroke file, so the studio can replay it like any
other run. `python -m scribe.record CANDIDATE OUT` replays one pool or batch
candidate through the recorder, writes the prompt on a blank sheet as the
target card, and `studio/scripts/pack-run.mjs` packs it into the gallery.
The dashboard names the painter "Scribe", shows the prompt instead of a
target, and at the reveal shows the rating and the judge's numbers instead
of a score sheet. The first one in the gallery is the hibiscus I rated love.
After a GRPO run, the same two commands put a checkpoint's painting next to
the base model's for the same prompt.

## Costs

Round-number estimates, except the first pool and the judge calibration, which were billed.

- Pool generation: pool v1, 1,175 candidates with one critic round each,
  cost $3.74 with the four cheap generators above. The estimate before it ran
  was 30 to 200 dollars.
- Rating: my 108 took 25 minutes, so a full pool of this size is three to
  four hours by hand.
- Training: 200 steps of 8 rollouts with 4 judge calls each is about 6,400
  judge calls, a few dollars to perhaps 20 with a small vision model. GPU time
  for a 4B model with LoRA on one A100, if a step takes about a minute: three
  to four hours, roughly 10 to 30 dollars.

## Data and consent

Every image in the pool is a stroke file written by a model and rendered by
this simulator. The ratings are mine for the first 108 candidates of pool v1
and an AI reviewer's for the other 813, as described above. Reference photos shown to
the critic are my own or in the public domain. No paintings by any
artist are used, as references, as training data or as targets.

## What is not done

- Generation, the critic and the judge have run against real models (pool
  v1 and the calibration above). The reward as a whole has run only with
  the scripted fake client.
- The trainer has never run. The Modal wrapper is written but has not been
  run; its image pins are a first guess.
- The aesthetic scorer's server is not part of this repo, and its score range
  has to be set from real scores before use.
- The rating page has been used for one pool on a Mac and in a headless
  browser here; it has no tests beyond its handlers.
