# Judge calibration

Judge `moonshotai/kimi-k2.6`, 2 same-subject references per painting, both orders, never the painting itself. AUC is the chance a random keep outscores a random nope (0.5 is a coin flip). Agreement is win share above one half for a keep and below for a nope, ties left out; it is capped by the design, so read the AUC.

| set | scored | keeps / nopes | AUC [95%] | mean win share keep / nope | agreement | order flips |
|---|---|---|---|---|---|---|
| all | 108 | 15 / 93 | 0.589 [0.449, 0.739] | 0.533 / 0.422 | 0.598 | 0.256 |
| rated by owner | 108 | 15 / 93 | 0.589 [0.427, 0.736] | 0.533 / 0.422 | 0.598 | 0.256 |
| deepseek/deepseek-v4-flash | 29 | 3 / 26 | 0.609 [0.404, 0.801] | 0.583 / 0.471 | 0.571 | 0.345 |
| google/gemini-3.1-flash-lite | 35 | 3 / 32 | 0.500 [0.141, 0.906] | 0.417 / 0.375 | 0.690 | 0.214 |
| moonshotai/kimi-k2.6 | 5 | 2 / 3 | 0.083 [0.000, 0.333] | 0.125 / 0.500 | 0.250 | 0.300 |
| qwen/qwen3-coder-next | 24 | 3 / 21 | 0.746 [0.548, 0.913] | 0.500 / 0.250 | 0.750 | 0.191 |
| z-ai/glm-5.3-flash | 15 | 4 / 11 | 0.625 [0.239, 0.909] | 0.812 / 0.750 | 0.308 | 0.267 |

## Direct same-subject pairs, one keep against one nope

- 40 valid pairs of 40 asked; the judge picked the keep 0.762 of the time; order flipped the answer in 0.275.
- rated by mixed: 6 pairs, keep picked 0.750
- rated by owner: 1 pairs, keep picked 1.000
- rated by reviewer: 33 pairs, keep picked 0.758
