# Judge calibration

Judge `anthropic/claude-haiku-5.5`, 2 same-subject references per painting, both orders, never the painting itself. AUC is the chance a random keep outscores a random nope (0.5 is a coin flip). Agreement is win share above one half for a keep and below for a nope, ties left out; it is capped by the design, so read the AUC.

| set | scored | keeps / nopes | AUC [95%] | mean win share keep / nope | agreement | order flips |
|---|---|---|---|---|---|---|
| all | 108 | 15 / 93 | 0.563 [0.398, 0.744] | 0.500 / 0.404 | 0.629 | 0.259 |
| rated by owner | 108 | 15 / 93 | 0.563 [0.376, 0.730] | 0.500 / 0.404 | 0.629 | 0.259 |
| deepseek/deepseek-v4-flash | 29 | 3 / 26 | 0.301 [0.000, 0.788] | 0.250 / 0.455 | 0.591 | 0.375 |
| google/gemini-3.1-flash-lite | 35 | 3 / 32 | 0.526 [0.156, 0.953] | 0.417 / 0.367 | 0.633 | 0.257 |
| moonshotai/kimi-k2.6 | 5 | 2 / 3 | 0.167 [0.000, 0.500] | 0.250 / 0.667 | 0.000 | 0.200 |
| qwen/qwen3-coder-next | 24 | 3 / 21 | 0.706 [0.310, 0.976] | 0.500 / 0.190 | 0.857 | 0.174 |
| z-ai/glm-5.3-flash | 15 | 4 / 11 | 0.591 [0.307, 0.818] | 0.875 / 0.727 | 0.429 | 0.200 |

## Direct same-subject pairs, one keep against one nope

- 40 valid pairs of 40 asked; the judge picked the keep 0.713 of the time; order flipped the answer in 0.175.
- rated by mixed: 6 pairs, keep picked 0.833
- rated by owner: 1 pairs, keep picked 1.000
- rated by reviewer: 33 pairs, keep picked 0.682
