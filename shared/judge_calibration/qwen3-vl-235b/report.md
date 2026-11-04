# Judge calibration

Judge `qwen/qwen3-vl-235b-a22b-instruct`, 2 same-subject references per painting, both orders, never the painting itself. AUC is the chance a random keep outscores a random nope (0.5 is a coin flip). Agreement is win share above one half for a keep and below for a nope, ties left out; it is capped by the design, so read the AUC.

| set | scored | keeps / nopes | AUC [95%] | mean win share keep / nope | agreement | order flips |
|---|---|---|---|---|---|---|
| all | 108 | 15 / 93 | 0.662 [0.511, 0.818] | 0.567 / 0.341 | 0.711 | 0.199 |
| rated by owner | 108 | 15 / 93 | 0.662 [0.505, 0.810] | 0.567 / 0.341 | 0.711 | 0.199 |
| deepseek/deepseek-v4-flash | 29 | 3 / 26 | 0.410 [0.135, 0.788] | 0.250 / 0.346 | 0.640 | 0.224 |
| google/gemini-3.1-flash-lite | 35 | 3 / 32 | 0.688 [0.219, 0.974] | 0.583 / 0.281 | 0.793 | 0.214 |
| moonshotai/kimi-k2.6 | 5 | 2 / 3 | 0.167 [0.000, 0.500] | 0.250 / 0.667 | 0.000 | 0.000 |
| qwen/qwen3-coder-next | 24 | 3 / 21 | 0.921 [0.746, 1.000] | 0.833 / 0.214 | 0.842 | 0.083 |
| z-ai/glm-5.3-flash | 15 | 4 / 11 | 0.614 [0.330, 0.864] | 0.750 / 0.659 | 0.500 | 0.367 |

## Direct same-subject pairs, one keep against one nope

- 40 valid pairs of 40 asked; the judge picked the keep 0.738 of the time; order flipped the answer in 0.175.
- rated by mixed: 6 pairs, keep picked 0.750
- rated by owner: 1 pairs, keep picked 1.000
- rated by reviewer: 33 pairs, keep picked 0.727
