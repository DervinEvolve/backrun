# Judge calibration

Judge `openai/gpt-5-mini`, 2 same-subject references per painting, both orders, never the painting itself. AUC is the chance a random keep outscores a random nope (0.5 is a coin flip). Agreement is win share above one half for a keep and below for a nope, ties left out; it is capped by the design, so read the AUC.

| set | scored | keeps / nopes | AUC [95%] | mean win share keep / nope | agreement | order flips |
|---|---|---|---|---|---|---|
| all | 108 | 15 / 93 | 0.585 [0.439, 0.731] | 0.483 / 0.384 | 0.622 | 0.324 |
| rated by owner | 108 | 15 / 93 | 0.585 [0.443, 0.731] | 0.483 / 0.384 | 0.622 | 0.324 |
| deepseek/deepseek-v4-flash | 29 | 3 / 26 | 0.442 [0.096, 0.847] | 0.333 / 0.385 | 0.667 | 0.414 |
| google/gemini-3.1-flash-lite | 35 | 3 / 32 | 0.641 [0.328, 0.969] | 0.583 / 0.406 | 0.593 | 0.300 |
| moonshotai/kimi-k2.6 | 5 | 2 / 3 | 0.500 [0.000, 1.000] | 0.500 / 0.500 | 0.500 | 0.400 |
| qwen/qwen3-coder-next | 24 | 3 / 21 | 0.635 [0.357, 0.897] | 0.250 / 0.155 | 0.842 | 0.167 |
| z-ai/glm-5.3-flash | 15 | 4 / 11 | 0.477 [0.170, 0.818] | 0.688 / 0.727 | 0.308 | 0.433 |

## Direct same-subject pairs, one keep against one nope

- 40 valid pairs of 40 asked; the judge picked the keep 0.700 of the time; order flipped the answer in 0.200.
- rated by mixed: 6 pairs, keep picked 1.000
- rated by owner: 1 pairs, keep picked 1.000
- rated by reviewer: 33 pairs, keep picked 0.636
