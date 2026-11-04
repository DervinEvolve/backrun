# Judge calibration

Judge `openai/gpt-5.2`, 2 same-subject references per painting, both orders, never the painting itself. AUC is the chance a random keep outscores a random nope (0.5 is a coin flip). Agreement is win share above one half for a keep and below for a nope, ties left out; it is capped by the design, so read the AUC.

| set | scored | keeps / nopes | AUC [95%] | mean win share keep / nope | agreement | order flips |
|---|---|---|---|---|---|---|
| all | 106 | 14 / 92 | 0.571 [0.414, 0.731] | 0.476 / 0.361 | 0.677 | 0.201 |
| rated by owner | 106 | 14 / 92 | 0.571 [0.409, 0.738] | 0.476 / 0.361 | 0.677 | 0.201 |
| deepseek/deepseek-v4-flash | 28 | 2 / 26 | 0.115 [0.038, 0.212] | 0.000 / 0.420 | 0.615 | 0.293 |
| google/gemini-3.1-flash-lite | 35 | 3 / 32 | 0.651 [0.281, 0.938] | 0.556 / 0.305 | 0.781 | 0.143 |
| moonshotai/kimi-k2.6 | 5 | 2 / 3 | 0.083 [0.000, 0.333] | 0.250 / 0.833 | 0.000 | 0.200 |
| qwen/qwen3-coder-next | 23 | 3 / 20 | 0.900 [0.742, 1.000] | 0.583 / 0.104 | 0.905 | 0.114 |
| z-ai/glm-5.3-flash | 15 | 4 / 11 | 0.534 [0.182, 0.875] | 0.688 / 0.720 | 0.357 | 0.318 |

## Direct same-subject pairs, one keep against one nope

- 25 valid pairs of 40 asked; the judge picked the keep 0.840 of the time; order flipped the answer in 0.240.
- rated by mixed: 4 pairs, keep picked 1.000
- rated by owner: 1 pairs, keep picked 0.500
- rated by reviewer: 20 pairs, keep picked 0.825
