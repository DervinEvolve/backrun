# Judge calibration

Judge `x-ai/grok-4.7`, 2 same-subject references per painting, both orders, never the painting itself. AUC is the chance a random keep outscores a random nope (0.5 is a coin flip). Agreement is win share above one half for a keep and below for a nope, ties left out; it is capped by the design, so read the AUC.

| set | scored | keeps / nopes | AUC [95%] | mean win share keep / nope | agreement | order flips |
|---|---|---|---|---|---|---|
| all | 108 | 15 / 93 | 0.551 [0.396, 0.718] | 0.467 / 0.392 | 0.632 | 0.213 |
| rated by owner | 108 | 15 / 93 | 0.551 [0.371, 0.712] | 0.467 / 0.392 | 0.632 | 0.213 |
| deepseek/deepseek-v4-flash | 29 | 3 / 26 | 0.250 [0.038, 0.577] | 0.167 / 0.433 | 0.579 | 0.259 |
| google/gemini-3.1-flash-lite | 35 | 3 / 32 | 0.490 [0.156, 0.812] | 0.333 / 0.352 | 0.667 | 0.243 |
| moonshotai/kimi-k2.6 | 5 | 2 / 3 | 0.417 [0.000, 1.000] | 0.375 / 0.500 | 0.500 | 0.300 |
| qwen/qwen3-coder-next | 24 | 3 / 21 | 0.730 [0.333, 0.992] | 0.500 / 0.155 | 0.905 | 0.104 |
| z-ai/glm-5.3-flash | 15 | 4 / 11 | 0.477 [0.159, 0.818] | 0.812 / 0.841 | 0.231 | 0.200 |

## Direct same-subject pairs, one keep against one nope

- 40 valid pairs of 40 asked; the judge picked the keep 0.688 of the time; order flipped the answer in 0.225.
- rated by mixed: 6 pairs, keep picked 1.000
- rated by owner: 1 pairs, keep picked 1.000
- rated by reviewer: 33 pairs, keep picked 0.621
