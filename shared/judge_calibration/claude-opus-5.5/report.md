# Judge calibration

Judge `anthropic/claude-opus-5.5`, 2 same-subject references per painting, both orders, never the painting itself. AUC is the chance a random keep outscores a random nope (0.5 is a coin flip). Agreement is win share above one half for a keep and below for a nope, ties left out; it is capped by the design, so read the AUC.

| set | scored | keeps / nopes | AUC [95%] | mean win share keep / nope | agreement | order flips |
|---|---|---|---|---|---|---|
| all | 108 | 15 / 93 | 0.657 [0.515, 0.794] | 0.600 / 0.368 | 0.667 | 0.069 |
| rated by owner | 108 | 15 / 93 | 0.657 [0.499, 0.792] | 0.600 / 0.368 | 0.667 | 0.069 |
| deepseek/deepseek-v4-flash | 29 | 3 / 26 | 0.442 [0.135, 0.885] | 0.333 / 0.375 | 0.625 | 0.121 |
| google/gemini-3.1-flash-lite | 35 | 3 / 32 | 0.672 [0.443, 0.885] | 0.583 / 0.375 | 0.667 | 0.043 |
| moonshotai/kimi-k2.6 | 5 | 2 / 3 | 0.167 [0.000, 0.500] | 0.250 / 0.667 | 0.000 | 0.000 |
| qwen/qwen3-coder-next | 24 | 3 / 21 | 0.873 [0.683, 1.000] | 0.583 / 0.131 | 0.895 | 0.083 |
| z-ai/glm-5.3-flash | 15 | 4 / 11 | 0.727 [0.591, 0.864] | 1.000 / 0.705 | 0.500 | 0.033 |

## Direct same-subject pairs, one keep against one nope

- 40 valid pairs of 40 asked; the judge picked the keep 0.787 of the time; order flipped the answer in 0.025.
- rated by mixed: 6 pairs, keep picked 0.833
- rated by owner: 1 pairs, keep picked 1.000
- rated by reviewer: 33 pairs, keep picked 0.773
