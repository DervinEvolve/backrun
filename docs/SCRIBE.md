# Scribe: a language model that writes stroke files

The scribe line trains a language model to write Backrun stroke files from a
text prompt such as "a harbour at dusk, two boats, low tide". The simulator
paints each file, a gate throws out files that are broken or that try to game
the judge, and the reward comes from a vision model asked, again and again,
which of two paintings is the better watercolour of the subject. One of the
two is always a painting from a pool the owner rated by hand.

The code lives in `scribe/`. Everything below the trainer has tests. The
trainer itself has been written and parsed but never run.

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
base model is `Qwen/Qwen3.5-4B`; the id was checked on the Hub API, loading it
was not.

The reward function returns None for an excluded rollout. TRL's current
source marks a completion NaN when every reward function returns None, leaves
it out of the group mean and gives it zero advantage. Older TRL releases added
it in as 0, so the script checks the installed version for that handling and
refuses to start without it.

Each reward call appends one line per rollout and one summary line to
`terms.jsonl`, so `python -m scribe.reward report OUT/terms.jsonl --step N`
gives the same report on any step of a run. The judge's token usage goes to
`usage.jsonl`.

## Costs

Round-number estimates only. Nothing has been billed yet.

- Pool generation: 1,600 candidates at about three calls each (generate,
  sometimes a retry or a critic round), a few thousand tokens per call: on the
  order of 15 million tokens, roughly 30 to 200 dollars depending on the
  models.
- Rating: one to three hours of the owner's time per 1,600 candidates.
- Training: 200 steps of 8 rollouts with 4 judge calls each is about 6,400
  judge calls, a few dollars to perhaps 20 with a small vision model. GPU time
  for a 4B model with LoRA on one A100, if a step takes about a minute: three
  to four hours, roughly 10 to 30 dollars.

## Data and consent

Every image in the pool is a stroke file written by a model and rendered by
this simulator, and every rating is the owner's. Reference photos shown to
the critic are the owner's own or in the public domain. No paintings by any
artist are used, as references, as training data or as targets.

## What is not done

- No real API call has been made: the client was tested against a local
  server and a scripted fake. An unauthenticated request to OpenRouter's
  public model list went through the proxy; the Hugging Face router was
  refused by the proxy from this machine.
- Generation, critique, judging and the reward ran only with the scripted
  fake client.
- No pool exists yet, so the coverage report and the thresholds of the gate
  have not met real model output.
- The trainer has never run. The `hf jobs` and `modal run` commands in its
  docstring are unverified, and the Modal wrapper app it mentions is not
  written.
- The aesthetic scorer's server is not part of this repo, and its score range
  has to be set from real scores before use.
- The rating page has handler tests and its script passes a syntax check.
  Nobody has looked at it in a browser yet.
