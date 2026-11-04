# Judge calibration

Judge `google/gemini-3.1-flash-lite`, 2 same-subject references per painting, both orders, never the painting itself. AUC is the chance a random keep outscores a random nope (0.5 is a coin flip). Agreement is win share above one half for a keep and below for a nope, ties left out; it is capped by the design, so read the AUC.

| set | scored | keeps / nopes | AUC [95%] | mean win share keep / nope | agreement | order flips |
|---|---|---|---|---|---|---|
| all | 108 | 15 / 93 | 0.470 [0.314, 0.633] | 0.417 / 0.454 | 0.554 | 0.454 |
| rated by owner | 108 | 15 / 93 | 0.470 [0.301, 0.641] | 0.417 / 0.454 | 0.554 | 0.454 |
| deepseek/deepseek-v4-flash | 29 | 3 / 26 | 0.314 [0.038, 0.770] | 0.250 / 0.490 | 0.474 | 0.483 |
| google/gemini-3.1-flash-lite | 35 | 3 / 32 | 0.562 [0.281, 0.828] | 0.500 / 0.453 | 0.625 | 0.429 |
| moonshotai/kimi-k2.6 | 5 | 2 / 3 | 0.000 [0.000, 0.000] | 0.125 / 0.833 | 0.000 | 0.300 |
| qwen/qwen3-coder-next | 24 | 3 / 21 | 0.643 [0.214, 0.976] | 0.417 / 0.274 | 0.867 | 0.458 |
| z-ai/glm-5.3-flash | 15 | 4 / 11 | 0.511 [0.148, 0.864] | 0.625 / 0.614 | 0.364 | 0.500 |

## Direct same-subject pairs, one keep against one nope

- 40 valid pairs of 40 asked; the judge picked the keep 0.600 of the time; order flipped the answer in 0.450.
- rated by mixed: 6 pairs, keep picked 0.583
- rated by owner: 1 pairs, keep picked 0.500
- rated by reviewer: 33 pairs, keep picked 0.606
